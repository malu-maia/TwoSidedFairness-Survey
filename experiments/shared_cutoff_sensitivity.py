"""Tune the manifest methods at each cutoff and select a shared list length."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from dataclasses import replace
import fcntl
from functools import lru_cache
import hashlib
import io
from itertools import groupby
import json
import math
import multiprocessing
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from . import grouped_experiments as survey
from . import sensitivity_inputs as inputs
from .experiments import _fit_mf_by_cutoff, _matrix
from .hyperparameter_sensitivity import (
    BPR, PAIR_METRICS, Setting, UNDERSERVED_NDCG, UNDERSERVED_SUPPLY,
    _atomic_json, _gain, _write_tsv, settings,
)
from .metrics import envy_freeness, gini_coefficient, ndcg, utility_variance, worse_off_cumulative_utility
from .prepare_explicit_method_inputs import DEFAULT_DATA_ROOT
from .sensitivity_methods import Ranking, local_profiles, rank_setting, resolve_parameters


CUTOFFS = (10, 25, 50, 100, 150)
NDCG_RETENTION = 0.95
FORMAT_VERSION = 3
BASELINE = {name: value for name, value in BPR.items() if name != "patience"}
GRID = {setting.key: setting for setting in [
    *settings(),
    *(Setting("SEAL", f"min_{fraction:g}_max_users", {"fraction": fraction, "upper": "users"})
      for fraction in (0.25, 0.10, 0.0)),
]}
REFERENCES = {setting.method: setting for setting in GRID.values() if setting.reference}
METHODS = tuple(REFERENCES)
METRIC_PAIRS = PAIR_METRICS | {"SEAL": ("sum_user_utilities", "item_count_gini")}
DEFAULT_RUNS = survey.PROJECT_ROOT / "experiments" / "runs" / "shared_cutoff"
DEFAULT_RESULTS = survey.PROJECT_ROOT / "results" / "hyperparameter_sensitivity" / "shared_cutoff_per_k"


def _frame_matrix(directory: Path, split: str, shape: tuple[int, int]) -> torch.Tensor:
    return torch.from_numpy(_matrix(pd.read_parquet(directory / f"{split}_interactions.parquet"), *shape))


def _sources() -> tuple[tuple[str, str], ...]:
    return tuple(sorted({
        (dataset, inputs.variant_for(dataset, method)) for dataset in inputs.DATASETS for method in METHODS
    }))


def _fit_checkpoints(directory: Path, destination: Path, seed: int, threads: int) -> dict:
    done = destination / "checkpoints.json"
    if done.is_file():
        cached = json.loads(done.read_text())
        if {row["k"] for row in cached["checkpoints"]} == set(CUTOFFS) and all(
            (destination / f"k_{k}.npy").is_file() for k in CUTOFFS
        ):
            return cached
    torch.set_num_threads(threads)
    start = perf_counter()
    metadata = survey._metadata(directory)
    print(f"Fitting BPR checkpoints for {metadata['dataset']}/{destination.parent.name} seed {seed}", flush=True)
    train = pd.read_parquet(directory / "train_interactions.parquet")
    validation = pd.read_parquet(directory / "validation_interactions.parquet")
    destination.mkdir(parents=True, exist_ok=True)
    checkpoints = []
    for k, scores, epoch, ndcg_value in _fit_mf_by_cutoff(
        train, validation, metadata["num_users"], metadata["num_items"], cutoffs=CUTOFFS, seed=seed, **BASELINE,
    ):
        temporary = destination / f"k_{k}.tmp"
        with temporary.open("wb") as stream:
            np.save(stream, scores, allow_pickle=False)
        temporary.replace(destination / f"k_{k}.npy")
        checkpoints.append({"k": k, "best_epoch": epoch, "validation_ndcg": ndcg_value})
    result = {"checkpoints": checkpoints, "seconds": perf_counter() - start}
    _atomic_json(done, result)
    return result


def _checkpoint_dir(root: Path, dataset: str, variant: str, seed: int) -> Path:
    return root / "checkpoints" / dataset / variant / f"seed_{seed}"


@lru_cache(maxsize=1)
def _parent(directory: Path, seed: int) -> survey.PreparedRun:
    metadata = survey._metadata(directory)
    train = survey._read_numeric_matrix(directory / "train_matrix.parquet")
    validation = _frame_matrix(directory, "validation", tuple(train.shape))
    eligible = train == 0
    if not validation.gt(0).any(dim=1).all() or (validation.gt(0) & ~eligible).any():
        raise ValueError(f"{directory} has invalid validation relevance")

    providers = survey._mapping_vector(
        survey._provider_mapping_path(directory), id_column="item_id", value_column="provider_id",
        size=train.shape[1],
    )
    merit = survey._read_item_merit(directory / "item_merit.csv", train.shape[1], train)
    return survey.PreparedRun(
        directory=directory, dataset=str(metadata["dataset"]), seed=seed, run=inputs.FIT_SEEDS.index(seed), k=0,
        scores=train, train_matrix=train, heldout_relevance=validation, eligible_items=eligible,
        base_topk=torch.empty((0, 0), dtype=torch.long), item_merit=merit, provider_ids=providers,
    )


def _group_vectors(run: survey.PreparedRun) -> tuple[torch.Tensor, torch.Tensor]:
    user_groups, item_groups = survey._cpfair_style_groups(run)
    survey._validate_cpfair_groups(user_groups, name="consumer groups")
    survey._validate_cpfair_groups(item_groups, name="provider-side item groups")
    return (
        torch.tensor([user_groups[user] for user in range(run.scores.shape[0])]),
        torch.tensor([item_groups[item] for item in range(run.scores.shape[1])]),
    )


def _cohort(
    parent: survey.PreparedRun, grouped: bool, split: str,
) -> tuple[survey.PreparedRun, tuple[torch.Tensor, torch.Tensor] | None, torch.Tensor]:
    eligible = parent.eligible_items
    users = torch.where(eligible.sum(dim=1) >= max(CUTOFFS))[0]
    if not len(users):
        raise ValueError(f"{parent.directory} has no users eligible for every cutoff")
    groups = _group_vectors(parent) if grouped else None
    if split == "test":
        relevance = _frame_matrix(parent.directory, split, tuple(parent.train_matrix.shape))
        if (relevance.gt(0) & (~eligible | parent.heldout_relevance.gt(0))).any():
            raise ValueError(f"{parent.directory} has overlapping holdout pairs")
    else:
        relevance = parent.heldout_relevance
    relevance = relevance[users]
    if not relevance.gt(0).any(dim=1).all():
        raise ValueError(f"{parent.directory} lacks {split} relevance for a selected user")

    if groups:
        groups = (groups[0][users], groups[1])
        if groups[0].unique().numel() != 2 or groups[1][eligible[users].any(dim=0)].unique().numel() != 2:
            raise ValueError(f"{parent.directory} lacks group support on both sides")
    return replace(
        parent, train_matrix=parent.train_matrix[users],
        heldout_relevance=relevance, eligible_items=eligible[users],
    ), groups, users


def _measure(run: survey.PreparedRun, topk: torch.Tensor, method: str, groups: tuple | None) -> dict:
    values = ndcg(run.heldout_relevance, topk, k=run.k).values.to(torch.float64)
    results = {"mean_ndcg": float(values.mean())}
    consumer, provider = METRIC_PAIRS[method]
    exposure = survey._exposure(topk, num_items=run.scores.shape[1], k=run.k)
    if method in {"TFROM", "FairSort", "LeadFairRec", "Ada2Fair"}:
        _, provider_exposure = survey._provider_values_from_item_values(exposure, run.provider_ids)
        _, provider_merit = survey._provider_values_from_item_values(run.item_merit, run.provider_ids)
        results["variance_provider_exposure"] = utility_variance(provider_exposure)
        results["provider_exposure_relevance_ratio_variance"] = survey._strictly_positive_ratio_variance(
            provider_exposure, provider_merit,
        )
    if method in {"TFROM", "FairSort", "LeadFairRec"}:
        results[consumer] = utility_variance(values)
    elif method == "FairRec":
        counts = torch.bincount(topk.flatten(), minlength=len(exposure))
        results[consumer] = envy_freeness(run.heldout_relevance, topk, k=run.k).mean_average_envy
        results[provider] = float((counts >= topk.numel() // len(exposure)).double().mean())
    elif method in {"Ada2Fair", "GGF", "TFLD"}:
        results[consumer] = float(worse_off_cumulative_utility(values, 0.25))
        supply = provider_exposure if method == "Ada2Fair" else exposure
        results[provider] = float(worse_off_cumulative_utility(supply, 0.25))
    elif method == "SEAL":
        results[consumer] = float(values.sum())
        results[provider] = gini_coefficient(torch.bincount(topk.flatten(), minlength=len(exposure)))
        results["user_ndcg_gini"] = gini_coefficient(values)
    else:
        user_groups, item_groups = groups
        results["variance_group_ndcg"] = utility_variance(survey._grouped_values(values, user_groups, aggregation="mean"))
        results["variance_item_group_exposure"] = utility_variance(
            survey._grouped_values(exposure, item_groups, aggregation="sum"),
        )
        results[UNDERSERVED_NDCG] = float(values[user_groups == survey.PROTECTED_GROUP].sum())
        results[UNDERSERVED_SUPPLY] = float(exposure[item_groups == survey.PROTECTED_GROUP].sum())
    return results


def _paired_row(run: survey.PreparedRun, setting: Setting, groups: tuple | None, baseline: dict, ranking) -> dict:
    consumer, provider = METRIC_PAIRS[setting.method]
    row = {
        "dataset": run.dataset, "variant": inputs.variant_for(run.dataset, setting.method), "seed": run.seed,
        "k": run.k, "method": setting.method, "setting": setting.name, "reference": setting.reference,
        "control": setting.control, "consumers": len(run.scores), "items": run.scores.shape[1],
        "provider_count": run.provider_ids.unique().numel(), "min_eligible": int(run.eligible_items.sum(dim=1).min()),
        "consumer_metric": consumer, "supplier_metric": provider,
        "parameters": json.dumps(resolve_parameters(setting.method, setting.parameters, run), sort_keys=True),
        "status": "allocation_failed",
    }
    if groups:
        row["protected_consumers"] = int((groups[0] == survey.PROTECTED_GROUP).sum())
        row["protected_items"] = int((groups[1] == survey.PROTECTED_GROUP).sum())
        row["protected_provider_count"] = run.provider_ids[groups[1] == survey.PROTECTED_GROUP].unique().numel()
    row.update({f"baseline_{key}": value for key, value in baseline.items()})
    if ranking is None:
        return row
    values = _measure(run, ranking.output.topk, setting.method, groups)
    if any(not math.isfinite(value) for value in values.values()):
        return row | {"status": "measurement_failed", "error": f"Nonfinite metrics for {setting.key}"}
    row.update(values)
    row.update(
        status="quota_violated" if ranking.quota_met is False else "ok",
        replacements=ranking.replacements, seal_quota_met=ranking.quota_met,
        masked_exposure_positions=ranking.masked_positions,
        consumer_gain=_gain(consumer, baseline[consumer], values[consumer]),
        supplier_gain=_gain(provider, baseline[provider], values[provider]),
        ndcg_delta=values["mean_ndcg"] - baseline["mean_ndcg"],
        ndcg_margin=values["mean_ndcg"] - NDCG_RETENTION * baseline["mean_ndcg"],
    )
    if groups:
        for metric in (UNDERSERVED_NDCG, UNDERSERVED_SUPPLY):
            row[f"{metric}_gain"] = _gain(metric, baseline[metric], values[metric])
    return row


def _evaluate_job(
    directory: Path, seed: int, results: Path, stage: str, tasks: tuple, threads: int,
) -> dict:
    torch.set_num_threads(threads)
    parent = _parent(directory, seed)
    split = "test" if stage.startswith("test_") else "validation"
    timing = perf_counter()
    method = GRID[tasks[0][0]].method
    variant = inputs.variant_for(parent.dataset, method)
    cohort, groups, users = _cohort(parent, method in inputs.GROUP_METHODS, split)
    checkpoints = _checkpoint_dir(results, parent.dataset, variant, seed)
    metadata = {row["k"]: row for row in json.loads(
        (checkpoints / "checkpoints.json").read_text()
    )["checkpoints"]}
    profiles = {}
    ada_scores = {}
    rows = []
    def cutoff(task):
        return task[1]

    for k, at_k in groupby(sorted(tasks, key=cutoff), key=cutoff):
        scores = torch.from_numpy(np.load(checkpoints / f"k_{k}.npy", allow_pickle=False))
        if scores.shape != parent.train_matrix.shape or not torch.isfinite(scores).all():
            raise ValueError(f"{checkpoints} has invalid scores at k={k}")
        scores = scores[users]
        base_topk = survey._topk_from_scores(scores, k, cohort.eligible_items)
        run = replace(cohort, scores=scores, k=k, base_topk=base_topk)
        baseline = _measure(run, base_topk, method, groups)
        for key, _, role in at_k:
            setting = GRID[key]
            start = perf_counter()
            rank_path = (results / "rankings" / run.dataset / variant
                         / f"seed_{seed}" / f"k_{k}" / f"{key}.npz")
            ranking, error = _ranking_for_setting(run, setting, groups, rank_path, profiles, ada_scores)
            row = _paired_row(run, setting, groups, baseline, ranking)
            row.update(
                split=split, stage=stage, role=role, error=row.get("error", error), seconds=perf_counter() - start,
                bpr_best_epoch=metadata[k]["best_epoch"], bpr_validation_ndcg=metadata[k]["validation_ndcg"],
            )
            rows.append(row)
    return {
        "rows": rows, "users": len(users), "dataset": parent.dataset, "seed": seed, "variant": variant,
        "source": str(directory), "seconds": perf_counter() - timing,
    }


def _ranking_for_setting(run, setting, groups, path, profiles, ada_scores) -> tuple[Ranking | None, str]:
    if path.is_file():
        with np.load(path, allow_pickle=False) as archive:
            topk = torch.from_numpy(archive["topk"].astype(np.int64))
            replacements, quota, masked = archive["diagnostics"].tolist()
        output = survey.MethodOutput(torch.empty(0), topk)
        return Ranking(output, replacements, None if quota == -1 else bool(quota), masked), ""
    failure_path = path.with_suffix(".json")
    if failure_path.is_file():
        return None, json.loads(failure_path.read_text())["error"]

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with redirect_stdout(io.StringIO()):
            dimension = setting.parameters.get("profile_embedding_dim")
            if dimension is not None and dimension not in profiles:
                profiles[dimension] = local_profiles(run, dimension)
            ranking = rank_setting(
                run, setting.method, setting.parameters, groups=groups, profiles=profiles.get(dimension),
                ada_scores=ada_scores.get(setting.key),
            )
    except torch.OutOfMemoryError:
        raise
    except (ValueError, RuntimeError) as failure:
        error = str(failure)
        _atomic_json(failure_path, {"error": error})
        return None, error

    if setting.method == "Ada2Fair":
        # The grid fixes weight_topk. The requested cutoff only forms the delivered list.
        ada_scores.setdefault(setting.key, ranking.output.scores)
    storage_type = np.uint16 if run.scores.shape[1] <= 2**16 else np.uint32
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        quota = -1 if ranking.quota_met is None else int(ranking.quota_met)
        np.savez_compressed(stream, topk=ranking.output.topk.numpy().astype(storage_type),
                            diagnostics=np.array([ranking.replacements, quota, ranking.masked_positions]))
    temporary.replace(path)
    return ranking, ""


def _stage(
    pool: ProcessPoolExecutor, runs: Path, root: Path, name: str, requests: dict, threads: int,
    *, seeds: tuple[int, ...] = inputs.FIT_SEEDS,
) -> pd.DataFrame:
    started = perf_counter()
    directory = root / "jobs" / name
    directory.mkdir(parents=True, exist_ok=True)
    pending = {}
    results = []
    jobs = []
    for dataset, tasks in requests.items():
        for method in METHODS:
            method_tasks = tuple(task for task in tasks if GRID[task[0]].method == method)
            if not method_tasks:
                continue
            variant = inputs.variant_for(dataset, method)
            source = inputs.input_directory(runs, dataset, variant)
            for seed in seeds:
                identity = hashlib.sha256(json.dumps(method_tasks).encode()).hexdigest()[:12]
                destination = directory / f"{dataset}_{variant}_{seed}_{method}_{identity}.json"
                jobs.append((dataset, method, source, seed, method_tasks, destination))
    expensive = {"TFLD", "TFLD-Groups", "GGF", "Ada2Fair", "SEAL"}
    jobs.sort(key=lambda job: (job[1] not in expensive, job[0] == "amazon"))
    for _, _, source, seed, method_tasks, destination in jobs:
        if destination.is_file():
            results.extend(json.loads(destination.read_text(encoding="utf-8"))["rows"])
        else:
            future = pool.submit(_evaluate_job, source, seed, root, name, method_tasks, threads)
            pending[future] = destination
    print(f"{name}: {len(pending)} jobs pending", flush=True)
    for index, future in enumerate(as_completed(pending), 1):
        try:
            outcome = future.result()
        except BaseException:
            for job in pending:
                job.cancel()
            print(f"{name}: input or measurement failed for {pending[future].stem}", flush=True)
            raise
        _atomic_json(pending[future], outcome)
        results.extend(outcome["rows"])
        if index == 1 or index % 10 == 0 or index == len(pending):
            print(f"{name}: {index}/{len(pending)} finished ({perf_counter() - started:.0f}s)", flush=True)
    return pd.DataFrame.from_records(results).sort_values(["dataset", "method", "setting", "k", "seed", "role"])


def _summarize(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for key, group in rows.groupby(["dataset", "method", "setting", "k"], sort=True):
        if group.seed.duplicated().any() or set(group.seed) != set(inputs.FIT_SEEDS):
            raise ValueError(f"Incomplete or duplicate fit-seed coverage for {key}")
        successful = group[group.status.eq("ok")]
        all_valid = len(successful) == len(inputs.FIT_SEEDS)
        record = dict(zip(("dataset", "method", "setting", "k"), key))
        record.update(
            variant=group["variant"].iloc[0], successful_seeds=len(successful),
            control=bool(group.control.iloc[0]), reference=bool(group.reference.iloc[0]),
            seed_statuses="; ".join(f"{row.seed}={row.status}" for row in group.itertuples()),
            failure_reasons="; ".join(dict.fromkeys(group["error"].dropna().loc[lambda values: values.ne("")])),
            product_min=json.loads(group["parameters"].iloc[0]).get("product_min"),
        )
        for metric in ("consumer_gain", "supplier_gain", "ndcg_delta", "ndcg_margin",
                       "mean_ndcg", "baseline_mean_ndcg"):
            median = group.get(metric, pd.Series(dtype=float)).median()
            record[metric] = median
        record["eligible"] = all_valid and record["ndcg_margin"] >= -1e-12
        record["weaker_gain"] = min(record["consumer_gain"], record["supplier_gain"])
        record["vote"] = record["eligible"] and record["weaker_gain"] > 1e-12
        records.append(record)
    return pd.DataFrame.from_records(records)


def _cutoff_votes(summary: pd.DataFrame) -> pd.DataFrame:
    denominator = len(inputs.DATASETS) * len(METHODS)
    records = []
    for k, pairs in summary.groupby("k", sort=True):
        if len(pairs) != denominator or pairs.duplicated(["dataset", "method"]).any():
            raise ValueError(f"Expected {denominator} unique method–dataset outcomes at k={k}")
        records.append({"k": int(k), "votes": int(pairs.vote.sum()), "method_dataset_pairs": denominator,
                        "ndcg_eligible_pairs": int(pairs.eligible.sum())})
    return pd.DataFrame.from_records(records).sort_values(["votes", "k"], ascending=[False, True])


def _request_all() -> dict:
    tasks = [(setting.key, k, "grid") for k in CUTOFFS for setting in GRID.values()]
    return {dataset: tasks for dataset in inputs.DATASETS}


def _select_setting(rows: pd.DataFrame, method: str) -> tuple[pd.Series, str]:
    reference = rows.loc[rows.reference].iloc[0]
    if len(rows) == 1:
        return reference, "fixed" if reference.successful_seeds == len(inputs.FIT_SEEDS) else "no_valid_setting"

    valid = rows.loc[~rows.control & rows.successful_seeds.eq(len(inputs.FIT_SEEDS))]
    if method == "SEAL":
        positive = valid.loc[valid.product_min.gt(0)]
        if positive.empty:
            disabled = valid.loc[valid.setting.eq("min_0_max_users")]
            if not disabled.empty:
                return disabled.iloc[0], "quota_disabled_fallback"
        valid = positive

    ordered = valid.sort_values(["weaker_gain", "ndcg_delta", "setting"], ascending=[False, False, True])
    qualifying = ordered.loc[ordered.eligible]
    if not qualifying.empty:
        return qualifying.iloc[0], "validation"
    if reference.successful_seeds == len(inputs.FIT_SEEDS):
        return reference, "reference_fallback"
    if not ordered.empty:
        return ordered.iloc[0], "valid_below_ndcg_fallback"
    return reference, "no_valid_setting"


def _freeze(summary: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    by_cutoff = {str(k): {} for k in CUTOFFS}
    records = []
    expected = {(dataset, method, k) for dataset in inputs.DATASETS for method in METHODS for k in CUTOFFS}
    for (dataset, method, k), rows in summary.groupby(["dataset", "method", "k"], sort=True):
        expected.remove((dataset, method, k))
        winner, reason = _select_setting(rows, method)
        setting = GRID[f"{method}__{winner.setting}"]
        by_cutoff[str(k)].setdefault(dataset, {})[method] = setting.manifest() | {"reason": reason}
        records.append(winner.to_dict() | {"reason": reason})
    if expected:
        raise ValueError(f"Incomplete validation grid: {sorted(expected)}")
    selected = pd.DataFrame.from_records(records)
    votes = _cutoff_votes(selected)
    document = {
        "format_version": FORMAT_VERSION, "k": int(votes.k.iloc[0]), "ndcg_retention": NDCG_RETENTION,
        "settings_by_cutoff": by_cutoff,
    }
    return document, selected, votes


def _test_requests(document: dict) -> dict:
    chosen = document["k"]
    by_cutoff = document["settings_by_cutoff"]
    if (document["format_version"] != FORMAT_VERSION or chosen not in CUTOFFS
            or set(by_cutoff) != {str(k) for k in CUTOFFS} or document["ndcg_retention"] != NDCG_RETENTION):
        raise ValueError("Frozen settings do not match the experiment manifest")
    requests = {dataset: [] for dataset in inputs.DATASETS}
    for k in CUTOFFS:
        if set(by_cutoff[str(k)]) != set(inputs.DATASETS):
            raise ValueError(f"Frozen settings are missing datasets at k={k}")
        for dataset, methods in by_cutoff[str(k)].items():
            if set(methods) != set(METHODS):
                raise ValueError(f"Frozen settings are incomplete for {dataset} at k={k}")
            for method, entry in methods.items():
                key = f"{method}__{entry['setting']}"
                if key not in GRID or entry["parameters"] != GRID[key].parameters:
                    raise ValueError(f"Frozen setting differs from the grid for {dataset}/{method} at k={k}")
                reference = REFERENCES[method].key
                if k == chosen and key != reference:
                    requests[dataset].append((reference, k, "reference"))
                requests[dataset].append((key, k, "selected"))
    return requests


def _write_report(root: Path, k: int, votes: pd.DataFrame, paired: pd.DataFrame) -> None:
    primary = paired.loc[paired.k.eq(k)]
    lines = [
        "# Shared recommendation cutoff and hyperparameter sensitivity", "",
        "This study reuses the previously inspected holdout. At every list length, BPR checkpoints and method "
        "settings were selected on validation. The three fitting seeds use the same interaction split.", "",
        f"The shared cutoff is **{k}**. A pair votes on validation when its ranking is valid on all seeds, "
        "both median fairness gains are positive, and the median NDCG margin meets the 95% retention rule.", "",
        "| Cutoff | Validation votes | Test pairs meeting the joint criterion | Test pairs valid on all seeds |",
        "|---:|---:|---:|---:|",
    ]
    test_votes = _cutoff_votes(paired).set_index("k")
    for row in votes.sort_values("k").itertuples():
        valid = paired.loc[paired.k.eq(row.k), "successful_seeds"].eq(len(inputs.FIT_SEEDS)).sum()
        lines.append(f"| {row.k} | {row.votes}/{row.method_dataset_pairs} | "
                     f"{test_votes.loc[row.k, 'votes']}/{row.method_dataset_pairs} | "
                     f"{valid}/{row.method_dataset_pairs} |")
    lines.extend([
        "", "## Primary held-out results", "",
        "| Method | Dataset | Setting | Selection reason | BPR NDCG | Method NDCG | Consumer gain | Supplier gain | Valid seeds |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ])

    def metric(value):
        return f"{value:.4f}" if pd.notna(value) else "unavailable"

    for row in primary.itertuples():
        lines.append(
            f"| {row.method} | {row.dataset} | {row.setting} | {row.reason} | "
            f"{metric(row.baseline_mean_ndcg)} | {metric(row.mean_ndcg)} | {metric(row.consumer_gain)} | "
            f"{metric(row.supplier_gain)} | {row.successful_seeds}/{len(inputs.FIT_SEEDS)} |"
        )
    lines.extend([
        "", "Metrics are medians across seeds with measurements. Metrics for rankings with quota violations "
        "are descriptive and those rankings do not count as valid. A zero-minimum SEAL setting disables the "
        "minimum item quota and is identified in the selection reason. Per-seed files retain all outcomes.", "",
    ])
    (root / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def _input_fingerprint(runs: Path, *, test: bool = False) -> str:
    digest = hashlib.sha256()
    for dataset, variant in _sources():
        directory = inputs.input_directory(runs, dataset, variant)
        stamp = directory / "sensitivity_input.json"
        if not stamp.is_file():
            raise FileNotFoundError(f"Missing coordinated input at {directory}. Run the prepare phase first.")
        if test:
            paths = [directory / "test_interactions.parquet"]
        else:
            paths = [stamp, *(directory / name for name in (
                "metadata.json", "mappings.json", "train_interactions.parquet", "train_matrix.parquet",
                "validation_interactions.parquet", "item_merit.csv", "provider_item_mapping.csv",
            )), *sorted((directory / "groups").glob("*.csv"))]
        for path in paths:
            digest.update(path.relative_to(runs).as_posix().encode())
            with path.open("rb") as stream:
                digest.update(hashlib.file_digest(stream, "sha256").digest())
    return digest.hexdigest()


def _check_manifest(path: Path, manifest: dict) -> None:
    if path.is_file() and json.loads(path.read_text()) != json.loads(json.dumps(manifest)):
        raise ValueError(f"{path.parent} contains results from different inputs or settings")
    _atomic_json(path, manifest)


def _checkpoint_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for dataset, variant in _sources():
        for seed in inputs.FIT_SEEDS:
            directory = _checkpoint_dir(root, dataset, variant, seed)
            paths = [directory / "checkpoints.json", *(directory / f"k_{k}.npy" for k in CUTOFFS)]
            for path in paths:
                digest.update(path.relative_to(root).as_posix().encode())
                with path.open("rb") as stream:
                    digest.update(hashlib.file_digest(stream, "sha256").digest())
    return digest.hexdigest()


def _checkpoint_stage(pool, runs, root, threads, sources, seeds) -> pd.DataFrame:
    futures = {}
    for dataset, variant in sources:
        source = inputs.input_directory(runs, dataset, variant)
        for seed in seeds:
            dest = _checkpoint_dir(root, dataset, variant, seed)
            future = pool.submit(_fit_checkpoints, source, dest, seed, threads)
            futures[future] = (dataset, variant, seed)
    records = []
    for future in as_completed(futures):
        dataset, variant, seed = futures[future]
        result = future.result()
        print(f"Checkpoints ready for {dataset}/{variant} seed {seed} ({result['seconds']:.1f}s fit)", flush=True)
        records.extend(row | {"dataset": dataset, "variant": variant, "seed": seed,
                              "fit_seconds": result["seconds"]} for row in result["checkpoints"])
    return pd.DataFrame.from_records(records).sort_values(["dataset", "variant", "seed", "k"])


def _benchmark(pool, runs, root, threads) -> None:
    dataset = "movielens"
    sources = tuple(source for source in _sources() if source[0] == dataset)
    seeds = inputs.FIT_SEEDS[:1]
    checkpoints = _checkpoint_stage(pool, runs, root, threads, sources, seeds)
    tasks = [(setting.key, k, "benchmark") for k in (CUTOFFS[0], CUTOFFS[-1])
             for setting in REFERENCES.values()]
    methods = _stage(pool, runs, root, "benchmark", {dataset: tasks}, threads, seeds=seeds)
    _write_tsv(root, "benchmark_checkpoints.tsv", checkpoints)
    _write_tsv(root, "benchmark_metrics.tsv", methods)
    durations = methods.groupby("method").seconds.mean()
    counts = pd.Series([setting.method for setting in GRID.values()]).value_counts()
    worker_hours = (durations * counts).sum() * len(inputs.DATASETS) * len(inputs.FIT_SEEDS) * len(CUTOFFS) / 3600
    print(f"Approximate validation grid time from the MovieLens benchmark: {worker_hours:.2f} worker-hours "
          "(method evaluation only, before parallelism; actual duration depends on dataset and setting).", flush=True)


def run(runs: Path, root: Path, workers: int, threads: int, phase: str) -> None:
    manifest = {
        "format_version": FORMAT_VERSION, "runs_root": str(runs), "input_fingerprint": _input_fingerprint(runs),
        "datasets": inputs.DATASETS, "fit_seeds": inputs.FIT_SEEDS, "split_seed": inputs.SPLIT_SEED,
        "cutoffs": CUTOFFS, "validation_cohort": "all_users_eligible_at_every_cutoff",
        "ndcg_retention": NDCG_RETENTION, "selection_metrics": METRIC_PAIRS,
        "baseline": BASELINE | {"checkpoint_metric": "validation_ndcg_at_each_cutoff", "epoch_tie": "earliest"},
        "method_settings": [setting.manifest() for setting in GRID.values()],
        "torch_version": str(torch.__version__), "numpy_version": np.__version__, "threads": threads,
    }
    _check_manifest(root / "manifest.json", manifest)
    best_path = root / "best_hyperparams_by_k.json"
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        if phase in {"all", "benchmark"}:
            _benchmark(pool, runs, root, threads)
        if phase == "benchmark":
            return
        if phase in {"all", "validation"}:
            checkpoints = _checkpoint_stage(pool, runs, root, threads, _sources(), inputs.FIT_SEEDS)
            _write_tsv(root, "checkpoints.tsv", checkpoints)
            validation = _stage(pool, runs, root, "validation", _request_all(), threads)
            _write_tsv(root, "validation_metrics.tsv", validation)
            summary = _summarize(validation)
            document, selected, votes = _freeze(summary)
            _write_tsv(root, "validation_summary.tsv", summary)
            _write_tsv(root, "selected_settings.tsv", selected)
            _write_tsv(root, "cutoff_votes.tsv", votes)
            document["checkpoint_fingerprint"] = _checkpoint_fingerprint(root)
            _atomic_json(best_path, document)
            print(f"Shared validation cutoff k={document['k']}", flush=True)
        if phase in {"all", "test"}:
            document = json.loads(best_path.read_text())
            requests = _test_requests(document)
            if document["checkpoint_fingerprint"] != _checkpoint_fingerprint(root):
                raise ValueError("BPR checkpoint artifacts differ from the frozen validation study")
            _check_manifest(root / "test_manifest.json", {
                "input_fingerprint": _input_fingerprint(runs, test=True),
                "selection_fingerprint": hashlib.sha256(best_path.read_bytes()).hexdigest(),
            })
            k = document["k"]
            test = _stage(pool, runs, root, "test_frozen", requests, threads)
            selected_test = test.loc[test.role.eq("selected")]
            report = _summarize(selected_test)
            selected = pd.DataFrame.from_records(
                {"dataset": dataset, "method": method, "k": int(cutoff),
                 "setting": entry["setting"], "reason": entry["reason"]}
                for cutoff, datasets in document["settings_by_cutoff"].items()
                for dataset, methods in datasets.items()
                for method, entry in methods.items()
            )
            report = report.merge(selected, on=["dataset", "method", "k", "setting"], validate="one_to_one")
            expected = len(METHODS) * len(inputs.DATASETS) * len(CUTOFFS)
            if len(report) != expected or len(selected_test) != expected * len(inputs.FIT_SEEDS):
                raise ValueError(f"Expected {expected} selected results with complete seed coverage")
            _write_tsv(root, "test_metrics.tsv", test)
            _write_tsv(root, "test_primary_metrics.tsv", test.loc[test.k.eq(k)])
            _write_tsv(root, "test_primary_by_dataset.tsv", report.loc[report.k.eq(k)])
            _write_tsv(root, "test_rq2_metrics.tsv", selected_test)
            _write_tsv(root, "test_by_dataset.tsv", report)
            votes = pd.read_csv(root / "cutoff_votes.tsv", sep="\t")
            _write_report(root, k, votes, report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--workers", type=survey._positive_int, default=6)
    parser.add_argument("--threads", type=survey._positive_int, default=1)
    parser.add_argument("--phase", choices=("all", "prepare", "benchmark", "validation", "test"), default="all")
    args = parser.parse_args()
    runs, root, data = (path.resolve() for path in (args.runs_root, args.results_root, args.data_root))
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".run.lock").open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.phase in {"prepare", "all"}:
            for dataset in inputs.DATASETS:
                prepared = all((inputs.input_directory(runs, dataset, variant) / "sensitivity_input.json").is_file()
                               for name, variant in _sources() if name == dataset)
                if args.phase == "all" and prepared:
                    continue
                inputs.prepare_dataset(runs, data, dataset, max(CUTOFFS), args.threads)
        if args.phase != "prepare":
            run(runs, root, args.workers, args.threads, args.phase)


if __name__ == "__main__":
    main()
