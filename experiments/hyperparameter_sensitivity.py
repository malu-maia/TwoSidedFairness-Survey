"""Run the k=150 two-sided fairness sweep on matched random survey subsets."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
from dataclasses import dataclass
import fcntl
import io
import json
import math
import multiprocessing
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from experiments import grouped_experiments as survey
from experiments.experiments import _fit_mf, _matrix
from experiments.metrics import (
    envy_freeness,
    gini_coefficient,
    ndcg,
    utility_variance,
    worse_off_cumulative_utility,
)
from obm.ada2fair import ada2fair
from obm.ggf import get_w_quantile, ggf
from obm.leadfairrec import leadfairrec, profile_text_embeddings
from obm.tfld import tfld
from obm.tfld_groups import tfld_groups
from pam.fairrec import fairrec
from pam.greedy_cpfair import cpfair
from pam.seal import seal
from rr.fairsort import fairsort
from rr.tfrom import tfrom


DATASETS = ("amazon", "ambar", "movielens", "yelp")
SEEDS = (42, 43, 44)
K = 150
USER_CAP = 512
ITEM_CAP = 256
BPR = dict(factors=32, epochs=200, patience=10, batch_size=1024,
           learning_rate=0.01, regularization=1e-4)
ARCHIVE_ROOT = (
    survey.PROJECT_ROOT.parent / "old folders" / "TwoSided-Fair-Rec-Survey"
    / "experiments" / "runs"
)
RESULTS_ROOT = survey.PROJECT_ROOT / "results" / "hyperparameter_sensitivity" / "k_150"


@dataclass(frozen=True)
class Setting:
    method: str
    name: str
    parameters: dict
    reference: bool = False
    control: bool = False

    @property
    def key(self) -> str:
        return f"{self.method}__{self.name}"

    def manifest(self) -> dict:
        return dict(method=self.method, setting=self.name, parameters=self.parameters,
                    reference=self.reference, control=self.control)


def settings() -> list[Setting]:
    grid = [
        Setting("TFROM", "uniform", {"fairness": "offline_uniform"}, reference=True),
        Setting("FairRec", "full_mms", {"mms_fraction": 1.0}, reference=True),
    ]
    for gamma, delta in [(-256, 64), (-128, 64), (-64, 64), (-32, 64), (-256, 16), (-256, 128)]:
        grid.append(Setting(
            "LeadFairRec", f"gamma_{gamma}_delta_{delta}",
            dict(gamma=gamma, delta=delta, beta=1, similar_user_count=30,
                 semantic_weight=0.15, profile_embedding_dim=64, profile_source="local_fallback",
                 exclude_seen=True),
            reference=(gamma, delta) == (-256, 64),
        ))
    for shape, ratio, floor in [
        ("uniform", 0.10, 0.95), ("uniform", 0.20, 0.95), ("uniform", 1.0, 0.95),
        ("quality", 1.0, 0.95), ("uniform", 0.20, 0.90),
    ]:
        grid.append(Setting(
            "FairSort", f"{shape}_{ratio:g}_floor_{floor:g}",
            dict(fairness=shape, rerank_ratio=ratio, ndcg_floor=floor,
                 lambda_max=8.0, search_gap=2 ** -5, user_order="input"),
            reference=(shape, ratio, floor) == ("uniform", 0.10, 0.95),
        ))
    for l1, l2 in [(0.9, 0.1), (0.9, 0.03), (0.9, 0.05), (0.5, 0.1), (0.5, 0.5),
                   (0.05, 0.05), (0.0, 0.0)]:
        grid.append(Setting("CPFair", f"l1_{l1:g}_l2_{l2:g}", dict(lambda1=l1, lambda2=l2),
                            reference=(l1, l2) == (0.9, 0.1), control=(l1, l2) == (0.0, 0.0)))
    for alpha in (0.0, 0.30, 0.50, 0.75, 1.0):
        grid.append(Setting(
            "Ada2Fair", f"alpha_{alpha:g}",
            dict(alpha=alpha, epochs=20, weight_epochs=5, embedding_dim=64, generator_hidden_dim=32,
                 adapter_hidden_dim=16, batch_size=4096, learning_rate=1e-3, provider_eta=2.0,
                 user_eta=1.0, delta=1e-7, weight_topk=100, exclude_seen=True),
            reference=alpha == 0.50,
        ))
    for method in ("TFLD", "TFLD-Groups"):
        for weight, curvature in [(0.5, 0.0), (0.01, 0.0), (0.05, 0.0), (0.15, 0.0),
                                  (0.05, -0.5), (0.05, -2.0)]:
            grid.append(Setting(
                method, f"lamb_{weight:g}_alpha_{curvature:g}",
                dict(lamb=weight, alpha=[curvature, 0.0], epochs=200, eta=1e-9),
                reference=(weight, curvature) == (0.5, 0.0),
            ))
    for task, weight, omega in [(1, 0.50, None), (2, 0.10, 1.0), (2, 0.25, 1.0),
                                (2, 0.50, 1.0), (2, 0.25, 0.50)]:
        name = f"task{task}_lamb_{weight:g}" + (f"_omega_{omega:g}" if task == 2 else "")
        grid.append(Setting(
            "GGF", name, dict(task=task, lamb=weight, q=0.25 if task == 2 else None,
                              omega=omega, beta=None, epochs=200),
            reference=task == 1,
        ))
    for fraction, upper in [(1.0, "users"), (0.50, "users"), (0.75, "users"), (0.75, "twice_min")]:
        grid.append(Setting(
            "SEAL", f"min_{fraction:g}_max_{upper}", dict(fraction=fraction, upper=upper),
            reference=fraction == 1.0,
        ))
    return grid


@dataclass
class Sample:
    run: survey.PreparedRun
    validation: torch.Tensor
    user_groups: torch.Tensor
    item_groups: torch.Tensor
    manifest: dict


@dataclass(frozen=True)
class EvaluationData:
    user_groups: torch.Tensor
    item_groups: torch.Tensor
    provider_ids: torch.Tensor
    item_merit: torch.Tensor


def _sample_axes(train: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    universe = np.sort(train["user_id"].unique())
    users = rng.choice(universe, min(USER_CAP, len(universe)), replace=False)
    supported = np.sort(train.loc[train["user_id"].isin(users), "item_id"].unique())
    items = rng.choice(supported, min(ITEM_CAP, len(supported)), replace=False)
    while len(items) >= K:
        restricted = train[train["user_id"].isin(users) & train["item_id"].isin(items)]
        counts = restricted.groupby("user_id").size()
        supported = np.sort(restricted["item_id"].unique())
        retained = counts.index[counts <= len(supported) - K].to_numpy()
        if len(retained) == 0:
            break
        if len(retained) == len(users) and len(supported) == len(items):
            return np.sort(retained), supported
        users, items = retained, supported
    raise ValueError(f"Seed {seed} has no sample with {K} eligible items per user")


def _restrict(frame: pd.DataFrame, users: np.ndarray, items: np.ndarray) -> pd.DataFrame:
    subset = frame.loc[
        frame["user_id"].isin(users) & frame["item_id"].isin(items),
        ["user_id", "item_id", "rating"],
    ].copy()
    subset["user_id"] = subset["user_id"].map({old: new for new, old in enumerate(users)})
    subset["item_id"] = subset["item_id"].map({old: new for new, old in enumerate(items)})
    return subset.reset_index(drop=True)


def _head_tail(counts: torch.Tensor) -> torch.Tensor:
    head = torch.argsort(counts, descending=True, stable=True)[:math.ceil(0.20 * len(counts))]
    labels = torch.full((len(counts),), survey.PROTECTED_GROUP)
    labels[head] = survey.ADVANTAGED_GROUP
    return labels


def _gender_labels(original_ids: list, path: Path, id_column: str) -> torch.Tensor:
    genders = survey._raw_gender_lookup(path, id_column=id_column)
    return torch.tensor([
        survey.PROTECTED_GROUP if genders[str(identity)] in survey.FEMALE_LABELS
        else survey.ADVANTAGED_GROUP
        for identity in original_ids
    ])


def _load_sample(archive: Path, dataset: str, seed: int) -> Sample:
    run_number = seed - SEEDS[0]
    source = archive / dataset / f"run_{run_number:02d}" / f"seed_{seed}"
    begin = perf_counter()
    archived_train = pd.read_parquet(source / "train_interactions.parquet")
    users, items = _sample_axes(archived_train, seed)
    train = _restrict(archived_train, users, items)
    validation = _restrict(pd.read_parquet(source / "validation_interactions.parquet"), users, items)
    n_users, n_items = len(users), len(items)
    train_matrix = torch.from_numpy(_matrix(train, n_users, n_items))
    val_matrix = torch.from_numpy(_matrix(validation, n_users, n_items))
    eligible = train_matrix == 0
    if val_matrix.count_nonzero() == 0 or ((val_matrix > 0) & ~eligible).any():
        raise ValueError(f"{dataset}/{seed} requires nonempty validation disjoint from training")
    maps = json.loads((source / "mappings.json").read_text(encoding="utf-8"))
    original_users = [maps["users"][int(user)] for user in users]
    original_items = [maps["items"][int(item)] for item in items]
    providers = pd.read_csv(survey._provider_mapping_path(source)).set_index("item_id").loc[items]
    provider_ids = torch.as_tensor(pd.factorize(providers["provider_id"], sort=True)[0])
    if dataset == "ambar":
        user_groups = _gender_labels(
            original_users, survey.AMBAR_RAW_DIRECTORY / "users_info.csv", "user_id"
        )
        item_groups = _gender_labels(
            providers["original_artist_id"].tolist(),
            survey.AMBAR_RAW_DIRECTORY / "artists_info.csv", "artist_id",
        )
    else:
        user_groups = _head_tail(train_matrix.count_nonzero(dim=1))
        item_groups = _head_tail(train_matrix.count_nonzero(dim=0))

    fit_start = perf_counter()
    predictions, best_epoch, val_ndcg = _fit_mf(
        train, validation, n_users, n_items, seed=seed, **BPR
    )
    bpr_seconds = perf_counter() - fit_start
    scores = torch.from_numpy(predictions)
    merit = train_matrix.sum(dim=0) / train_matrix.count_nonzero(dim=0)
    base_topk = survey._topk_from_scores(scores, K, eligible)
    run = survey.PreparedRun(
        directory=source, dataset=dataset, run=run_number, seed=seed, k=K,
        scores=scores, train_matrix=train_matrix, heldout_relevance=val_matrix,
        eligible_items=eligible, base_topk=base_topk, item_merit=merit, provider_ids=provider_ids,
    )
    manifest = dict(
        dataset=dataset, seed=seed, source=str(source), k=K,
        users=n_users, items=n_items, providers=int(provider_ids.max()) + 1,
        min_eligible=int(eligible.sum(dim=1).min()), train_events=len(train),
        validation_users=validation["user_id"].nunique(),
        bpr_best_epoch=best_epoch, bpr_checkpoint_ndcg_at_10=val_ndcg, bpr_seconds=bpr_seconds,
        preparation_seconds=perf_counter() - begin,
        archive_user_ids=users.tolist(), archive_item_ids=items.tolist(),
        user_groups=user_groups.tolist(), item_groups=item_groups.tolist(),
    )
    return Sample(run, val_matrix, user_groups, item_groups, manifest)


def _profiles(run: survey.PreparedRun) -> tuple[torch.Tensor, torch.Tensor]:
    users = survey._local_leadfairrec_profiles(survey._leadfairrec_user_prompts(run))
    items = survey._local_leadfairrec_profiles(survey._leadfairrec_item_prompts(run))
    return profile_text_embeddings(users, 64), profile_text_embeddings(items, 64)


def _rank(sample: Sample, setting: Setting, parameters: dict, profiles: tuple) -> tuple[torch.Tensor, int]:
    run = sample.run
    scores, train, providers, eligible = run.scores, run.train_matrix, run.provider_ids, run.eligible_items
    n_users = scores.shape[0]
    survey._seed_everything(run.seed)
    masked_exposure = 0
    if setting.method == "TFROM":
        _, topk = tfrom(scores, k=K, item_provider_mapping=providers)
    elif setting.method == "FairRec":
        _, topk = fairrec(scores, k=K)
    elif setting.method == "LeadFairRec":
        arguments = {key: value for key, value in parameters.items() if not key.startswith("profile_")}
        _, topk = leadfairrec(
            train, providers, scores, k=K, eligible_items=eligible, device="cpu",
            user_profile_embeddings=profiles[0], item_profile_embeddings=profiles[1], **arguments,
        )
    elif setting.method == "FairSort":
        arguments = {key: value for key, value in parameters.items() if key != "user_order"}
        _, topk = fairsort(scores, providers, k=K, eligible_items=eligible, **arguments)
    elif setting.method == "CPFair":
        _, topk = cpfair(
            scores.masked_fill(~eligible, -torch.inf), dict(enumerate(sample.user_groups.tolist())),
            dict(enumerate(sample.item_groups.tolist())), run.base_topk, k=K,
            device="cpu", **parameters,
        )
        if setting.control and not torch.equal(topk, run.base_topk):
            raise ValueError("CPFair zero weights must reproduce the shared baseline")
    elif setting.method == "Ada2Fair":
        _, topk = ada2fair(train, providers, k=K, seed=run.seed, eligible_items=eligible,
                          device="cpu", **parameters)
    elif setting.method in {"TFLD", "GGF"}:
        if setting.method == "TFLD":
            _, exposure, _, _ = tfld(
                scores, k=K, device=torch.device("cpu"), exposure_only=True,
                **setting.parameters,
            )
        else:
            weights = (get_w_quantile(n_users, parameters["q"], parameters["omega"])
                       if parameters["task"] == 2 else None)
            _, exposure, _, _ = ggf(
                scores, k=K, lamb=parameters["lamb"], beta=parameters["beta"],
                epochs=parameters["epochs"], user_weights=weights, device=torch.device("cpu"),
                exposure_only=True,
            )
        if not torch.isfinite(exposure).all():
            raise ValueError(f"{setting.key} returned nonfinite expected exposure")
        masked_exposure = int((~eligible.gather(1, exposure.topk(K, dim=1).indices)).sum())
        topk = torch.topk(exposure.masked_fill(~eligible, -torch.inf), K, dim=1).indices
    elif setting.method == "TFLD-Groups":
        _, topk, _, _ = tfld_groups(
            scores, sample.user_groups, sample.item_groups, k=K, eligible_items=eligible,
            device=torch.device("cpu"), **parameters,
        )
    elif setting.method == "SEAL":
        _, topk = seal(scores, k=K, product_min=parameters["product_min"],
                       product_max=parameters["product_max"])
    else:
        raise ValueError(f"Unknown method {setting.method}")
    return topk.detach().cpu(), masked_exposure


def _resolved_parameters(setting: Setting, n_users: int, n_items: int) -> dict:
    parameters = setting.parameters.copy()
    if setting.method in {"TFLD", "GGF"}:
        parameters["representation"] = "float32_expected_exposure"
    if setting.method == "SEAL":
        lower = math.floor(parameters["fraction"] * n_users * K / n_items)
        upper = n_users if parameters["upper"] == "users" else min(n_users, 2 * lower)
        parameters.update(product_min=lower, product_max=upper)
    return parameters


def _valid_ranking(raw: torch.Tensor, run: survey.PreparedRun, repair: bool) -> tuple[torch.Tensor, int]:
    if raw.shape != run.base_topk.shape or (raw < 0).any() or (raw >= run.scores.shape[1]).any():
        raise ValueError("A method returned invalid ranking shape or item IDs")
    sorted_items = raw.sort(dim=1).values
    if (sorted_items[:, 1:] == sorted_items[:, :-1]).any():
        raise ValueError("A method returned duplicate items in a recommendation list")
    bad = ~run.eligible_items.gather(1, raw)
    replacements = int(bad.sum())
    if replacements == 0:
        return raw, 0
    if not repair:
        raise ValueError("A method returned ineligible items despite its ranking mask")

    rankings = raw.clone()
    baseline = torch.argsort(run.scores, dim=1, descending=True, stable=True)
    for user in bad.any(dim=1).nonzero().flatten().tolist():
        kept = raw[user, ~bad[user]]
        available = run.eligible_items[user].clone()
        available[kept] = False
        candidates = baseline[user, available[baseline[user]]]
        rankings[user] = torch.cat((kept, candidates[: int(bad[user].sum())]))
    if not run.eligible_items.gather(1, rankings).all():
        raise ValueError("Ranking completion produced ineligible items")
    return rankings, replacements


USER_VARIANCE = "variance_ndcg"
GROUP_NDCG = "variance_group_ndcg"
PROVIDER_VARIANCE = "variance_provider_exposure"
MERIT_VARIANCE = "provider_exposure_relevance_ratio_variance"
GROUP_EXPOSURE = "variance_item_group_exposure"
BOTTOM_NDCG = "worse_off_cumulative_utility"
BOTTOM_PROVIDER = "worse_off_cumulative_provider_exposure"
BOTTOM_ITEM = "worse_off_cumulative_item_exposure"
UNDERSERVED_NDCG = "underserved_consumer_total_ndcg"
UNDERSERVED_SUPPLY = "underserved_supplier_total_exposure"
PAIR_METRICS = {
    "TFROM": (USER_VARIANCE, PROVIDER_VARIANCE),
    "LeadFairRec": (USER_VARIANCE, MERIT_VARIANCE),
    "FairSort": (USER_VARIANCE, PROVIDER_VARIANCE),
    "CPFair": (GROUP_NDCG, GROUP_EXPOSURE),
    "FairRec": ("mean_average_envy", "full_mms_fraction"),
    "Ada2Fair": (BOTTOM_NDCG, BOTTOM_PROVIDER),
    "TFLD": (BOTTOM_NDCG, BOTTOM_ITEM),
    "TFLD-Groups": (UNDERSERVED_NDCG, UNDERSERVED_SUPPLY),
    "GGF": (BOTTOM_NDCG, BOTTOM_ITEM),
    "SEAL": ("user_ndcg_gini", "item_count_gini"),
}
LOWER_IS_BETTER = {
    USER_VARIANCE, GROUP_NDCG, PROVIDER_VARIANCE, MERIT_VARIANCE, GROUP_EXPOSURE,
    "mean_average_envy", "user_ndcg_gini", "item_count_gini",
}


def _measure(data: EvaluationData, topk: torch.Tensor, relevance: torch.Tensor, method: str) -> dict:
    active = relevance.gt(0).any(dim=1)
    values = ndcg(relevance[active], topk[active], k=K).values.to(torch.float64)
    item_exp = survey._exposure(topk, num_items=relevance.shape[1], k=K)
    _, provider_exp = survey._provider_values_from_item_values(item_exp, data.provider_ids)
    _, provider_merit = survey._provider_values_from_item_values(data.item_merit, data.provider_ids)
    item_counts = torch.bincount(topk.flatten(), minlength=len(item_exp))
    group_exp = survey._grouped_values(item_exp, data.item_groups, aggregation="sum")
    group_ndcg = survey._grouped_values(values, data.user_groups[active], aggregation="mean")
    if len(group_ndcg) < 2 or len(group_exp) < 2:
        raise ValueError("Both evaluation groups must have observed relevance")
    protected_user = data.user_groups[active] == survey.PROTECTED_GROUP
    protected_item = data.item_groups == survey.PROTECTED_GROUP
    result = {
        "mean_ndcg": float(values.mean()), "sum_ndcg": float(values.sum()),
        USER_VARIANCE: utility_variance(values), GROUP_NDCG: utility_variance(group_ndcg),
        PROVIDER_VARIANCE: utility_variance(provider_exp),
        MERIT_VARIANCE: survey._strictly_positive_ratio_variance(provider_exp, provider_merit),
        GROUP_EXPOSURE: utility_variance(group_exp),
        BOTTOM_NDCG: float(worse_off_cumulative_utility(values, 0.25)),
        BOTTOM_PROVIDER: float(worse_off_cumulative_utility(provider_exp, 0.25)),
        BOTTOM_ITEM: float(worse_off_cumulative_utility(item_exp, 0.25)),
        UNDERSERVED_NDCG: float(values[protected_user].sum()),
        UNDERSERVED_SUPPLY: float(item_exp[protected_item].sum()),
        "user_ndcg_gini": gini_coefficient(values), "item_count_gini": gini_coefficient(item_counts),
        "full_mms_fraction": float((item_counts >= len(topk) * K // len(item_counts)).double().mean()),
        "min_item_count": int(item_counts.min()), "max_item_count": int(item_counts.max()),
    }
    if method in {"FairRec", "BPR"}:
        result["mean_average_envy"] = envy_freeness(relevance[active], topk[active], k=K).mean_average_envy
    return result


def _gain(metric: str, before: float, after: float) -> float:
    if before == after == 0:
        return 0.0
    return (before - after if metric in LOWER_IS_BETTER else after - before) / (before + after)


def _row(sample_manifest: dict, entry: dict, values: dict, baseline: dict, split: str) -> dict:
    method = entry["method"]
    measures = PAIR_METRICS[method]
    result = {key: sample_manifest[key] for key in ("dataset", "seed", "k", "users", "items")}
    result.update(
        method=method, setting=entry["setting"], reference=entry["reference"], control=entry["control"],
        split=split, evaluation_users=sample_manifest[f"{split}_users"], status=entry["status"],
        parameters=json.dumps(entry["parameters"], sort_keys=True, separators=(",", ":")),
        seconds=entry["seconds"], replacements=entry.get("replacements"),
        masked_exposure_positions=entry.get("masked_exposure_positions"),
        consumer_metric=measures[0], supplier_metric=measures[1], error=entry.get("error", ""),
    )
    if entry["status"] != "ok":
        return result
    for name, before in baseline.items():
        if name in values:
            result[f"baseline_{name}"] = before
            result[name] = values[name]
    for side, metric in zip(("consumer", "supplier"), measures):
        result[f"{side}_gain"] = _gain(metric, baseline[metric], values[metric])
    result["ndcg_delta"] = values["mean_ndcg"] - baseline["mean_ndcg"]
    for label, metric in (("consumer", UNDERSERVED_NDCG), ("supplier", UNDERSERVED_SUPPLY)):
        result[f"underserved_{label}_gain"] = _gain(metric, baseline[metric], values[metric])
    if method == "SEAL":
        lower, upper = entry["parameters"]["product_min"], entry["parameters"]["product_max"]
        result["seal_quota_met"] = lower <= values["min_item_count"] <= values["max_item_count"] <= upper
    return result


def _atomic_json(path: Path, payload: dict) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def _run_sample(archive: Path, root: Path, dataset: str, seed: int, threads: int) -> tuple[str, int, int, float]:
    torch.set_num_threads(threads)
    directory = root / "samples" / dataset / f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    completed = directory / "validation.json"
    if completed.is_file() and (directory / "rankings.npz").is_file():
        stored = json.loads(completed.read_text(encoding="utf-8"))
        if stored.get("format_version") == 1:
            source = archive / dataset / f"run_{seed - SEEDS[0]:02d}" / f"seed_{seed}"
            if Path(stored["sample"]["source"]) != source:
                raise ValueError(f"Cached sample has a different input directory: {directory}")
            return dataset, seed, len(stored["entries"]), 0.0
    start = perf_counter()
    sample = _load_sample(archive, dataset, seed)
    profiles = _profiles(sample.run)
    data = EvaluationData(sample.user_groups, sample.item_groups,
                          sample.run.provider_ids, sample.run.item_merit)
    baseline = _measure(data, sample.run.base_topk, sample.validation, "BPR")
    rankings = dict(baseline=sample.run.base_topk.numpy().astype(np.uint16),
                    eligible=sample.run.eligible_items.numpy(),
                    validation=sample.validation.gt(0).numpy(),
                    provider_ids=data.provider_ids.numpy(), item_merit=data.item_merit.numpy())
    entries = []
    for setting in settings():
        begin = perf_counter()
        params = _resolved_parameters(setting, *sample.run.scores.shape)
        entry = setting.manifest() | {"status": "ok", "parameters": params}
        try:
            with redirect_stdout(io.StringIO()):
                raw, masked_exposure = _rank(sample, setting, params, profiles)
            topk, repairs = _valid_ranking(raw, sample.run, setting.method in {"TFROM", "FairRec", "SEAL"})
            values = _measure(data, topk, sample.validation, setting.method)
        except ValueError as error:
            if setting.method != "SEAL" or not str(error).startswith(("product_", "SEAL could")):
                raise
            entry.update(status="allocation_failed", error=str(error))
        else:
            entry.update(parameters=params, replacements=repairs, masked_exposure_positions=masked_exposure)
            rankings[setting.key] = topk.numpy().astype(np.uint16)
        entry["seconds"] = perf_counter() - begin
        row = _row(sample.manifest, entry, values if entry["status"] == "ok" else {}, baseline, "validation")
        entries.append(entry | {"key": setting.key, "validation": row})
    with (directory / "rankings.tmp").open("wb") as stream:
        np.savez_compressed(stream, **rankings)
    (directory / "rankings.tmp").replace(directory / "rankings.npz")
    _atomic_json(completed, dict(format_version=1, sample=sample.manifest, entries=entries))
    return dataset, seed, len(entries), perf_counter() - start


def _suite_pair(suite: str, method: str) -> tuple[str, str]:
    if suite == "RQ3":
        return UNDERSERVED_NDCG, UNDERSERVED_SUPPLY
    return PAIR_METRICS[method]


def _with_suite(row: dict, suite: str) -> dict:
    copy = row | {"suite": suite}
    if suite == "RQ3":
        copy.update(
            consumer_metric=UNDERSERVED_NDCG, supplier_metric=UNDERSERVED_SUPPLY,
            consumer_gain=row.get("underserved_consumer_gain"),
            supplier_gain=row.get("underserved_supplier_gain"),
        )
    return copy


def _suites(method: str) -> tuple[str, ...]:
    if method == "CPFair":
        return "RQ4", "RQ3"
    if method == "TFLD-Groups":
        return ("RQ3",)
    if method in {"LeadFairRec", "FairRec", "SEAL"}:
        return ("exploratory",)
    return ("RQ4",)


def _validation_summary(validation: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (suite, method, setting), rows in validation.groupby(["suite", "method", "setting"], sort=False):
        successful = rows.loc[rows["status"] == "ok"]
        metrics = _suite_pair(suite, method)
        record = dict(
            suite=suite, method=method, setting=setting,
            reference=bool(rows["reference"].iloc[0]), control=bool(rows["control"].iloc[0]),
            consumer_metric=metrics[0], supplier_metric=metrics[1],
            successful_samples=len(successful), expected_samples=len(DATASETS) * len(SEEDS),
        )
        if len(successful) == record["expected_samples"]:
            dataset_medians = successful.groupby("dataset")[["consumer_gain", "supplier_gain"]].median()
            consumer, supplier = dataset_medians.min()
            record.update(
                worst_consumer_gain=consumer, worst_supplier_gain=supplier,
                joint_gain=min(consumer, supplier), joint_improvement=bool(min(consumer, supplier) > 0),
                mean_ndcg_delta=successful["ndcg_delta"].mean(),
            )
        records.append(record)
    summary = pd.DataFrame.from_records(records)
    summary["pareto"] = False
    summary["selected"] = False
    for _, rows in summary.groupby(["suite", "method"], sort=False):
        available = rows.loc[rows["joint_gain"].notna() & ~rows["control"]]
        if available.empty:
            continue
        objectives = ["worst_consumer_gain", "worst_supplier_gain", "mean_ndcg_delta"]
        values = available[objectives].to_numpy()
        for index, point in zip(available.index, values):
            dominated = ((values >= point).all(axis=1) & (values > point).any(axis=1)).any()
            summary.loc[index, "pareto"] = not dominated
        frontier = available.loc[summary.loc[available.index, "pareto"]]
        best = frontier.sort_values(
            ["joint_gain", "mean_ndcg_delta", "setting"],
            ascending=[False, False, True],
        ).index[0]
        summary.loc[best, "selected"] = True
    return summary


def _test_sample(directory: Path, stored: dict, selection: pd.DataFrame) -> tuple[list[dict], dict]:
    manifest = stored["sample"].copy()
    frame = _restrict(
        pd.read_parquet(Path(manifest["source"]) / "test_interactions.parquet"),
        np.asarray(manifest["archive_user_ids"]), np.asarray(manifest["archive_item_ids"]),
    )
    relevance = torch.from_numpy(_matrix(frame, manifest["users"], manifest["items"]))
    manifest["test_users"] = int(relevance.gt(0).any(dim=1).sum())
    with np.load(directory / "rankings.npz") as ranks:
        eligible = torch.from_numpy(ranks["eligible"])
        if not manifest["test_users"] or (relevance.gt(0) & (
            ~eligible | torch.from_numpy(ranks["validation"])
        )).any():
            raise ValueError(f"{directory} requires a nonempty test split disjoint from training and validation")
        data = EvaluationData(
            torch.tensor(manifest["user_groups"]), torch.tensor(manifest["item_groups"]),
            torch.from_numpy(ranks["provider_ids"]), torch.from_numpy(ranks["item_merit"]),
        )
        baseline = _measure(data, torch.from_numpy(ranks["baseline"]).long(), relevance, "BPR")
        evaluations = []
        for entry in stored["entries"]:
            matched = selection[
                (selection["method"] == entry["method"]) & (selection["setting"] == entry["setting"])
            ]
            if matched.empty:
                continue
            values = {}
            if entry["status"] == "ok":
                topk = torch.from_numpy(ranks[entry["key"]]).long()
                if not eligible.gather(1, topk).all():
                    raise ValueError(f"{directory}/{entry['key']} contains ineligible items")
                values = _measure(data, topk, relevance, entry["method"])
            row = _row(manifest, entry, values, baseline, "test")
            for setting in matched.itertuples():
                role = "reference_and_selected" if setting.reference and setting.selected else (
                    "reference" if setting.reference else "selected"
                )
                evaluations.append(_with_suite(row, setting.suite) | {"role": role})
    return evaluations, manifest


def _test_dataset_summary(test: pd.DataFrame) -> pd.DataFrame:
    columns = ["consumer_gain", "supplier_gain", "ndcg_delta"]
    successful = test.loc[test["status"] == "ok"]
    grouping = ["suite", "method", "setting", "role", "dataset"]
    summary = successful.groupby(grouping)[columns].agg(["median", "min", "max"])
    summary.columns = ["_".join(column) for column in summary.columns]
    return summary.reset_index()


def _write_tsv(root: Path, name: str, frame: pd.DataFrame) -> None:
    destination = root / name
    temp = destination.with_suffix(".tmp")
    frame.to_csv(temp, sep="\t", index=False, float_format="%.12g")
    temp.replace(destination)


def _reviewer_summary(
    validation: pd.DataFrame, selection: pd.DataFrame, test: pd.DataFrame, samples: pd.DataFrame
) -> str:
    current = test.loc[test["role"].isin(["selected", "reference_and_selected"])]
    tested = selection.loc[selection["selected"]]
    lines = [
        "# Two-sided fairness sensitivity at k=150",
        "",
        "We evaluated the existing method parameters on three matched random samples per dataset. "
        "Each sample had a newly trained graded BPR baseline. Sample membership used training data, "
        "settings were selected on validation data, and test relevance was read after selection.",
        "",
        f"The {len(samples)} samples contained {samples['users'].min()}–{samples['users'].max()} users "
        f"and {samples['items'].min()}–{samples['items'].max()} items. Test consumer metrics cover "
        f"{samples['test_users'].min()}–{samples['test_users'].max()} users per sample with relevant "
        "items in the sampled catalog. Supplier metrics use lists from all sampled users. "
        "This analysis describes the reduced catalogs and three sampling seeds.",
        "",
        "## Selected settings",
        "",
        "The validation columns show the worst dataset median on each side. The test columns show "
        "medians across samples. Positive values indicate improved fairness or higher mean NDCG. "
        "Selection maximized the smaller validation gain and broke ties with mean NDCG. "
        "The Pareto frontier includes NDCG with no utility threshold.",
        "",
    ]
    if not tested.empty and not tested["joint_improvement"].any():
        lines.extend([
            "**No setting had positive worst-dataset validation gains on both sides.** "
            "The table gives the strongest measured joint trade-offs.",
            "",
        ])
    lines.extend([
        "| Analysis | Method | Setting | Validation consumer | Validation supplier | "
        "Test consumer | Test supplier | Test Δ NDCG | Test joint gains |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in tested.itertuples():
        matched = current[
            (current["suite"] == row.suite) & (current["method"] == row.method) &
            (current["status"] == "ok")
        ]
        test_gains = matched[["consumer_gain", "supplier_gain", "ndcg_delta"]].median()
        joint = int(((matched["consumer_gain"] > 0) & (matched["supplier_gain"] > 0)).sum())
        lines.append(
            f"| {row.suite} | {row.method} | `{row.setting}` | {row.worst_consumer_gain:+.3f} | "
            f"{row.worst_supplier_gain:+.3f} | {test_gains['consumer_gain']:+.3f} | "
            f"{test_gains['supplier_gain']:+.3f} | {test_gains['ndcg_delta']:+.3f} | {joint}/{len(matched)} |"
        )

    rqs = current.loc[current["suite"] == "RQ3"]
    rq4 = current.loc[(current["suite"] == "RQ4") & (current["status"] == "ok")]
    lines.extend([
        "", "## Interpretation for the revision", "",
        "- The historical RQ4 finding that TFROM alone consistently improved both measured sides "
        "applies to the full survey’s evaluated settings and datasets. The reduced-catalog runs "
        "measure sensitivity and show NDCG alongside fairness.",
        "- The evaluated Ada2Fair implementation gathers training ratings from an unseen-item ranking, "
        "which yields a constant consumer target. Describe its findings as implementation- and "
        "setting-specific observations on the Rawlsian metrics.",
        "- The RQ3 gains concern underserved-group utility and exposure. The per-run tables also carry "
        "group disparities, which answer a different empirical question.",
    ])
    for method, rows in rq4.groupby("method", sort=False):
        if method in {"TFROM", "FairSort", "Ada2Fair"}:
            joint = int(((rows["consumer_gain"] > 0) & (rows["supplier_gain"] > 0)).sum())
            lines.append(
                f"- {method} improved both RQ4 metrics in {joint}/{len(rows)} held-out samples "
                f"at its selected setting. Its median mean NDCG change was {rows['ndcg_delta'].median():+.3f}."
            )
    for method, rows in rqs.loc[rqs["status"] == "ok"].groupby("method", sort=False):
        consumer_up = int((rows["consumer_gain"] > 0).sum())
        supplier_up = int((rows["supplier_gain"] > 0).sum())
        lines.append(
            f"- For {method} at its selected RQ3 setting, underserved consumer utility rose in "
            f"{consumer_up}/{len(rows)} samples and underserved supplier exposure rose in "
            f"{supplier_up}/{len(rows)}."
        )
    failed = validation.drop_duplicates(["method", "setting", "dataset", "seed"])
    failed_count = int((failed["status"] != "ok").sum())
    repair_count = int(failed["replacements"].fillna(0).sum())
    seal_results = current.loc[(current["method"] == "SEAL") & (current["status"] == "ok")]
    seal_met = int(seal_results["seal_quota_met"].eq(True).sum())
    lines.extend([
        "", "## Implementation details", "",
        "- TFLD and GGF use float32 expected exposure directly. Its update is the same linear "
        "Frank–Wolfe update projected onto expected exposure. The older dense policy rounds "
        "probability entries to float16. Eligibility is applied when converting exposure to top-k.",
        "- TFROM, FairRec, and SEAL lists were completed after removing seen items. "
        f"The validation runs recorded {repair_count:,} replacements. SEAL quota satisfaction "
        f"was checked on delivered lists. Its selected setting met both bounds in "
        f"{seal_met}/{len(seal_results)} samples. The sweep recorded {failed_count} allocation failures.",
        "- For FairSort, ratios 0.10 and 0.20 both imply 150 candidates in these catalogs. "
        "The ratio-1 runs evaluate a larger prefix. The RQ4 supply metric is exposure variance, "
        "and the per-run files also include merit alignment.",
        "- The exploratory runs pair LeadFairRec with provider exposure-to-merit variance, "
        "FairRec with full maximin-share coverage, and SEAL with unweighted item-exposure Gini.",
        "", "The validation, held-out test, selected-settings, and Pareto-frontier tables accompany "
        "this summary. Sample-level intervals are descriptive min–max ranges.", "",
    ])
    return "\n".join(lines)


def analyze(archive: Path, root: Path) -> None:
    stored_runs = []
    validation_rows = []
    for dataset in DATASETS:
        for seed in SEEDS:
            directory = root / "samples" / dataset / f"seed_{seed}"
            stored = json.loads((directory / "validation.json").read_text(encoding="utf-8"))
            source = archive / dataset / f"run_{seed - SEEDS[0]:02d}" / f"seed_{seed}"
            if stored.get("format_version") != 1 or Path(stored["sample"]["source"]) != source:
                raise ValueError(f"Cached sample does not match this experiment: {directory}")
            stored_runs.append((directory, stored))
            for item in stored["entries"]:
                validation_rows.extend(_with_suite(item["validation"], suite) for suite in _suites(item["method"]))
    validation = pd.DataFrame.from_records(validation_rows)
    summary = _validation_summary(validation)
    selection = summary.loc[summary["selected"] | summary["reference"]]
    _write_tsv(root, "validation_metrics.tsv", validation)
    _write_tsv(root, "validation_summary.tsv", summary)
    _write_tsv(root, "pareto_frontier.tsv", summary.loc[summary["pareto"]])
    _write_tsv(root, "selected_settings.tsv", selection)

    test_rows, manifests = [], []
    for directory, stored in stored_runs:
        observations, manifest = _test_sample(directory, stored, selection)
        test_rows.extend(observations)
        manifests.append({key: value for key, value in manifest.items() if not isinstance(value, list)})
    test = pd.DataFrame.from_records(test_rows)
    samples = pd.DataFrame.from_records(manifests)
    _write_tsv(root, "test_metrics.tsv", test)
    _write_tsv(root, "test_by_dataset.tsv", _test_dataset_summary(test))
    _write_tsv(root, "samples.tsv", samples)
    (root / "reviewer_summary.md").write_text(
        _reviewer_summary(validation, selection, test, samples), encoding="utf-8",
    )


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def run_experiment(archive: Path, root: Path, workers: int, threads: int) -> None:
    manifest = dict(
        format_version=1, archive=str(archive), k=K, seeds=SEEDS,
        datasets=DATASETS, max_users=USER_CAP, max_items=ITEM_CAP,
        baseline=BPR | {"checkpoint_metric": "validation_ndcg_at_10"},
        method_settings=[setting.manifest() for setting in settings()],
        torch_version=str(torch.__version__), numpy_version=np.__version__,
        device="cpu", threads=threads,
    )
    manifest_path = root / "manifest.json"
    if manifest_path.is_file() and json.loads(manifest_path.read_text(encoding="utf-8")) != json.loads(
        json.dumps(manifest)
    ):
        raise ValueError(f"{root} contains a sweep with different settings")
    _atomic_json(manifest_path, manifest)
    torch.set_num_threads(threads)
    jobs = [(archive, root, dataset, seed, threads) for dataset in DATASETS for seed in SEEDS]
    if workers == 1:
        for job in jobs:
            print(f"Validation {_run_sample(*job)}", flush=True)
    else:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            futures = [pool.submit(_run_sample, *job) for job in jobs]
            for future in futures:
                print(f"Validation {future.result()}", flush=True)
    analyze(archive, root)
    print(f"Wrote analysis to {root}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=ARCHIVE_ROOT,
                        help="Root of the archived dataset/run/seed input directories.")
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT,
                        help="Destination for the new experiment and its resumable run files.")
    parser.add_argument("--workers", type=_positive_int, default=3)
    parser.add_argument("--threads", type=_positive_int, default=2)
    args = parser.parse_args()
    archive, root = args.archive_root.resolve(), args.results_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".run.lock").open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_experiment(archive, root, args.workers, args.threads)


if __name__ == "__main__":
    main()
