"""Run grouped two-sided fairness experiments over prepared run directories."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from .metrics import (
    envy_freeness,
    exposure_relevance_ratio_variance,
    ndcg,
    sum_user_utilities,
    utility_variance,
    worse_off_cumulative_utility,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

AMBAR_RAW_DIRECTORY = (
    PROJECT_ROOT / "data" / "raw_datasets" / "explicit_feedback" / "AMBAR"
)
# CPFair's encoding: +1 is boosted, -1 is penalized (pam/greedy_cpfair.py:8-9).
PROTECTED_GROUP = 1
ADVANTAGED_GROUP = -1
FEMALE_LABELS = frozenset({"f", "female"})

RESULT_FIELDS = [
    "method",
    "dataset",
    "seed",
    "run",
    "k",
    "ndcg_baseline",
    "ndcg_fair_method",
    "stakeholder",
    "granularity",
    "fairness_metric_name",
    "fairness_metric_baseline",
    "fairness_metric",
]


@dataclass(frozen=True)
class RunnerConfig:
    """Execution parameters for expensive methods."""

    ada2fair_epochs: int = 20
    ada2fair_weight_epochs: int = 5
    leadfairrec_semantic_weight: float = 0.15
    leadfairrec_profile_embedding_dim: int = 64
    leadfairrec_profile_model: str = "deepseek-chat"
    leadfairrec_profile_timeout: float = 30.0
    multifr_epochs: int = 20
    multifr_rounds: int = 1
    multifr_batch_size: int = 64
    ggf_epochs: int = 200
    tfld_epochs: int = 200
    tfld_alpha: tuple[float, float] = (0.0, 0.0)
    tfld_lambda: float = 0.5
    tsfd_maxiter: int = 500
    worse_off_fraction: float = 0.25
    multifr_positive_threshold: float = 0.8
    device: str = "cpu"


@dataclass(frozen=True)
class GroupSpec:
    """One comparison group requested by the survey experiments."""

    number: int
    fairness_definition: str
    stakeholder: str
    granularity: str
    methods: tuple[str, ...]
    metric_name: str

    @property
    def output_directory_name(self) -> str:
        return f"{self.granularity}_{self.stakeholder}"

    @property
    def output_file_name(self) -> str:
        return f"group_{self.number:02d}.tsv"


@dataclass(frozen=True)
class PreparedRun:
    """Input tensors and metadata loaded from one prepared run directory."""

    directory: Path
    dataset: str
    seed: int
    run: int
    k: int
    scores: torch.Tensor
    train_matrix: torch.Tensor
    heldout_relevance: torch.Tensor
    eligible_items: torch.Tensor | None
    base_topk: torch.Tensor
    item_merit: torch.Tensor
    provider_ids: torch.Tensor


@dataclass(frozen=True)
class MethodOutput:
    """Fair method output normalized to a top-k ranking."""

    scores: torch.Tensor
    topk: torch.Tensor


GROUPS: tuple[GroupSpec, ...] = (
    GroupSpec(1, "equal_satisfaction", "user", "individual", ( "TFROM", "LeadFairRec", "FairSort",), "variance_ndcg",),
    GroupSpec(2, "equal_satisfaction", "user", "group", ("CPFair",), "variance_group_ndcg",),
    GroupSpec(3, "envy_freeness", "user", "individual", ("FairRec",), "envy_freeness"),
    GroupSpec(4, "rawlsian", "user", "individual", ("Ada2Fair", "TFLD", "GGF",), "worse_off_cumulative_utility",),
    GroupSpec(5, "utilitarian", "user", "individual", ("SEAL",), "sum_user_utilities"),
    GroupSpec(6, "utilitarian", "user", "group", ("TSFD",), "sum_group_user_utilities"),
    GroupSpec(7, "ued", "provider", "individual", ("TFROM", "FairSort",), "variance_provider_exposure",),
    GroupSpec(8, "ued", "item", "group", ("CPFair",), "variance_item_group_exposure"),
    GroupSpec(9, "qwe", "provider", "individual", ("TFROM", "FairSort",), "provider_exposure_relevance_ratio_variance",),
    GroupSpec(10, "qwe", "item", "group", ("TSFD",), "item_group_exposure_relevance_ratio_variance",),
    GroupSpec(11, "rawlsian", "provider", "individual", ("Ada2Fair",), "worse_off_cumulative_provider_exposure",),
    GroupSpec(12, "rawlsian", "item", "individual", ("GGF", "TFLD",), "worse_off_cumulative_item_exposure",),
)

MethodAdapter = Callable[[PreparedRun, RunnerConfig], MethodOutput]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _resolve(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return PROJECT_ROOT / resolved


def _read_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"required experiment input not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".tab"}:
        return pd.read_csv(path, sep="\t")
    raise ValueError(f"unsupported table format for {path}")


def _read_numeric_matrix(
    path: Path, *, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    frame = _read_table(path)
    try:
        values = frame.to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path} must contain only numeric values") from error
    if values.ndim != 2 or values.size == 0:
        raise ValueError(f"{path} must contain a non-empty 2D matrix")
    if not np.isfinite(values).all():
        raise ValueError(f"{path} must contain only finite values")
    return torch.as_tensor(values, dtype=dtype)


def _read_topk(path: Path) -> torch.Tensor:
    rankings = _read_numeric_matrix(path, dtype=torch.float64)
    if not torch.equal(rankings, torch.round(rankings)):
        raise ValueError(f"{path} must contain integer item IDs")
    return rankings.to(torch.long)


def _read_optional_bool_matrix(
    path: Path, expected_shape: tuple[int, int]
) -> torch.Tensor | None:
    if not path.is_file():
        return None
    frame = _read_table(path)
    values = frame.to_numpy()
    if values.shape != expected_shape:
        raise ValueError(
            f"{path} shape {values.shape} does not match expected {expected_shape}"
        )
    return torch.as_tensor(values.astype(bool), dtype=torch.bool)


def _leadfairrec_profile_cache_path(run: PreparedRun) -> Path:
    return run.directory / "method_inputs" / "LeadFairRec" / "deepseek_profiles.json"


def _format_limited_ids(values: Iterable[int], *, limit: int = 12) -> str:
    ordered = list(values)
    shown = ", ".join(str(value) for value in ordered[:limit])
    if len(ordered) > limit:
        return f"{shown}, and {len(ordered) - limit} more"
    return shown or "none"


def _leadfairrec_user_prompts(run: PreparedRun) -> list[str]:
    train = run.train_matrix.detach().cpu()
    providers = run.provider_ids.detach().cpu().to(torch.long)
    prompts: list[str] = []
    for user_id in range(train.shape[0]):
        ratings = train[user_id]
        observed = torch.nonzero(ratings > 0, as_tuple=False).flatten()
        observed_items = observed.tolist()
        provider_ids = sorted({int(providers[item].item()) for item in observed_items})
        top_items = sorted(
            observed_items,
            key=lambda item: (-float(ratings[item].item()), item),
        )[:8]
        prompts.append(
            "Summarize this recommendation user's durable preferences. "
            f"Mapped user ID: {user_id}. "
            f"Observed train interactions: {len(observed_items)}. "
            f"Observed provider IDs: {_format_limited_ids(provider_ids)}. "
            f"Strongest observed item IDs: {_format_limited_ids(top_items, limit=8)}. "
            f"Mean observed rating: {float(ratings[observed].mean().item()) if len(observed_items) else 0.0:.4f}."
        )
    return prompts


def _leadfairrec_item_prompts(run: PreparedRun) -> list[str]:
    train = run.train_matrix.detach().cpu()
    providers = run.provider_ids.detach().cpu().to(torch.long)
    exposure = (train > 0).sum(dim=0)
    prompts: list[str] = []
    for item_id in range(train.shape[1]):
        ratings = train[:, item_id]
        observed = torch.nonzero(ratings > 0, as_tuple=False).flatten()
        observed_users = observed.tolist()
        mean_rating = float(ratings[observed].mean().item()) if observed_users else 0.0
        prompts.append(
            "Summarize this recommendation item's stable attributes. "
            f"Mapped item ID: {item_id}. "
            f"Provider ID: {int(providers[item_id].item())}. "
            f"Train exposure count: {int(exposure[item_id].item())}. "
            f"Mean observed train rating: {mean_rating:.4f}. "
            f"Observed user IDs: {_format_limited_ids(observed_users)}."
        )
    return prompts


def _validate_profile_list(
    value: object,
    *,
    expected_size: int,
    key: str,
    path: Path,
) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{path} must contain a string list at {key!r}")
    if len(value) != expected_size:
        raise ValueError(
            f"{path} contains {len(value)} {key} profiles, expected {expected_size}"
        )
    return value


def _read_leadfairrec_profile_cache(
    path: Path, run: PreparedRun
) -> tuple[list[str], list[str]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"{path} must contain a JSON object")
    users = _validate_profile_list(
        document.get("users"),
        expected_size=run.train_matrix.shape[0],
        key="users",
        path=path,
    )
    items = _validate_profile_list(
        document.get("items"),
        expected_size=run.train_matrix.shape[1],
        key="items",
        path=path,
    )
    return users, items


def _write_leadfairrec_profile_cache(
    path: Path,
    *,
    run: PreparedRun,
    config: RunnerConfig,
    users: list[str],
    items: list[str],
    source: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "metadata": {
            "model": config.leadfairrec_profile_model,
            "profile_embedding_dim": config.leadfairrec_profile_embedding_dim,
            "num_users": run.train_matrix.shape[0],
            "num_items": run.train_matrix.shape[1],
            "source": source,
        },
        "users": users,
        "items": items,
    }
    path.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")


def _local_leadfairrec_profiles(prompts: list[str]) -> list[str]:
    return [prompt.replace("Summarize", "Profile", 1) for prompt in prompts]


def _leadfairrec_profile_embeddings(
    run: PreparedRun,
    config: RunnerConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    from obm.leadfairrec import generate_deepseek_profiles, profile_text_embeddings

    cache_path = _leadfairrec_profile_cache_path(run)
    if cache_path.is_file():
        user_profiles, item_profiles = _read_leadfairrec_profile_cache(cache_path, run)
    else:
        user_prompts = _leadfairrec_user_prompts(run)
        item_prompts = _leadfairrec_item_prompts(run)
        source = "local_fallback"
        if os.environ.get("DEEPSEEK_API_KEY"):
            system_prompt = (
                "Create one concise recommendation profile sentence. "
                "Use stable preferences or item attributes from the supplied train-history summary. "
                "Do not copy raw IDs unless they are the only available signal."
            )
            user_profiles = generate_deepseek_profiles(
                user_prompts,
                system_prompt=system_prompt,
                model=config.leadfairrec_profile_model,
                timeout=config.leadfairrec_profile_timeout,
            )
            item_profiles = generate_deepseek_profiles(
                item_prompts,
                system_prompt=system_prompt,
                model=config.leadfairrec_profile_model,
                timeout=config.leadfairrec_profile_timeout,
            )
            source = "deepseek"
        else:
            user_profiles = _local_leadfairrec_profiles(user_prompts)
            item_profiles = _local_leadfairrec_profiles(item_prompts)
        _write_leadfairrec_profile_cache(
            cache_path,
            run=run,
            config=config,
            users=user_profiles,
            items=item_profiles,
            source=source,
        )
    return (
        profile_text_embeddings(
            user_profiles,
            embedding_dim=config.leadfairrec_profile_embedding_dim,
        ),
        profile_text_embeddings(
            item_profiles,
            embedding_dim=config.leadfairrec_profile_embedding_dim,
        ),
    )


def _split_file_pair(run_dir: Path) -> tuple[Path, Path]:
    candidates = (
        (run_dir / "train_interactions.parquet", run_dir / "test_interactions.parquet"),
        (run_dir / "train.parquet", run_dir / "test.parquet"),
    )
    for train_path, test_path in candidates:
        if train_path.is_file() and test_path.is_file():
            return train_path, test_path
    expected = " or ".join(
        f"{train_path.name}/{test_path.name}" for train_path, test_path in candidates
    )
    raise FileNotFoundError(
        f"{run_dir} must contain train/test split files: {expected}"
    )


def _validate_test_users_in_train(run_dir: Path) -> None:
    train_path, test_path = _split_file_pair(run_dir)
    train = _read_table(train_path)
    test = _read_table(test_path)
    for name, frame, path in (
        ("train", train, train_path),
        ("test", test, test_path),
    ):
        if "user_id" not in frame.columns:
            raise ValueError(f"{path} must contain a user_id column for {name} users")

    train_users = set(train["user_id"].dropna().tolist())
    test_users = set(test["user_id"].dropna().tolist())
    missing_users = sorted(
        test_users.difference(train_users), key=lambda value: str(value)
    )
    if missing_users:
        preview = ", ".join(map(str, missing_users[:5]))
        raise ValueError(f"{run_dir} has test users not present in train: {preview}")


def _metadata(run_dir: Path) -> dict[str, object]:
    path = run_dir / "metadata.json"
    if not path.is_file():
        return {}
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return document


def _metadata_int(
    metadata: Mapping[str, object],
    keys: Sequence[str],
    *,
    default: int,
) -> int:
    for key in keys:
        value = metadata.get(key)
        if value is not None:
            return int(value)
    return default


def _infer_dataset(run_dir: Path, metadata: Mapping[str, object]) -> str:
    value = metadata.get("dataset")
    if value is not None:
        return str(value)
    try:
        return run_dir.parents[1].name
    except IndexError:
        return run_dir.name


def _mapping_series(path: Path, id_column: str, value_column: str) -> pd.Series:
    frame = _read_table(path)
    missing = {id_column, value_column}.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    selected = frame[[id_column, value_column]].dropna()
    if selected.empty:
        raise ValueError(f"{path} must contain at least one mapping row")
    if selected[id_column].duplicated().any():
        raise ValueError(f"{path} contains duplicate {id_column} values")
    return selected.set_index(id_column)[value_column]


def _mapping_vector(
    path: Path,
    *,
    id_column: str,
    value_column: str,
    size: int,
    dtype: torch.dtype = torch.long,
) -> torch.Tensor:
    mapping = _mapping_series(path, id_column, value_column)
    missing = [index for index in range(size) if index not in mapping.index]
    if missing:
        preview = ", ".join(map(str, missing[:5]))
        raise ValueError(f"{path} does not cover IDs: {preview}")
    values = [mapping.loc[index] for index in range(size)]
    return torch.as_tensor(values, dtype=dtype)


def _mapping_dict_for_cpfair(
    path: Path, *, id_column: str, value_column: str
) -> dict[int, int]:
    mapping = _mapping_series(path, id_column, value_column)
    values = [int(value) for value in mapping.tolist()]
    unique_values = set(values)
    if unique_values == {0, 1}:
        values = [-1 if value == 0 else 1 for value in values]
    return {
        int(entity_id): int(value)
        for entity_id, value in zip(mapping.index.tolist(), values)
    }


def _read_item_merit(
    path: Path, num_items: int, relevance: torch.Tensor
) -> torch.Tensor:
    if not path.is_file():
        return torch.sum(relevance, dim=0).to(torch.float64)
    frame = _read_table(path)
    if {"item_id", "merit"}.issubset(frame.columns):
        mapping = frame.set_index("item_id")["merit"]
        values = [mapping.loc[item_id] for item_id in range(num_items)]
    elif frame.shape[1] == 1 and len(frame) == num_items:
        values = frame.iloc[:, 0].tolist()
    else:
        raise ValueError(
            f"{path} must contain item_id and merit columns or one aligned merit column"
        )
    merit = torch.as_tensor(values, dtype=torch.float64)
    if not torch.isfinite(merit).all() or (merit < 0).any():
        raise ValueError(f"{path} merit values must be finite and non-negative")
    return merit


def _topk_from_scores(
    scores: torch.Tensor, k: int, eligible_items: torch.Tensor | None = None
) -> torch.Tensor:
    if eligible_items is not None:
        scores = scores.masked_fill(~eligible_items.to(scores.device), -torch.inf)
    return torch.topk(scores, k=k, dim=1).indices


def _load_prepared_run(
    run_dir: Path,
    recommendation_list_size: int | None = None,
) -> PreparedRun:
    metadata = _metadata(run_dir)
    _validate_test_users_in_train(run_dir)
    scores = _read_numeric_matrix(run_dir / "scores.parquet")
    train_matrix = _read_numeric_matrix(run_dir / "train_matrix.parquet")
    relevance = _read_numeric_matrix(run_dir / "heldout_relevance.parquet")
    if scores.shape != train_matrix.shape or scores.shape != relevance.shape:
        raise ValueError(f"{run_dir} score, train, and relevance matrices must align")
    k = (
        recommendation_list_size
        if recommendation_list_size is not None
        else _metadata_int(
            metadata,
            ("recommendation_list_length", "effective_base_topk_length"),
            default=10,
        )
    )
    if not 1 <= k <= scores.shape[1]:
        raise ValueError(f"{run_dir} has invalid recommendation length: {k}")
    eligible_items = _read_optional_bool_matrix(
        run_dir / "eligible_items.parquet",
        tuple(scores.shape),
    )
    base_topk_path = run_dir / "base_topk.parquet"
    if recommendation_list_size is not None:
        base_topk = _topk_from_scores(scores, k, eligible_items)
    elif base_topk_path.is_file():
        base_topk = _read_topk(base_topk_path)
    else:
        base_topk = _topk_from_scores(scores, k, eligible_items)
    if base_topk.shape[0] != scores.shape[0] or base_topk.shape[1] < k:
        raise ValueError(f"{base_topk_path} must contain at least k rankings per user")
    provider_ids = _mapping_vector(
        _provider_mapping_path(run_dir),
        id_column="item_id",
        value_column="provider_id",
        size=scores.shape[1],
    )
    return PreparedRun(
        directory=run_dir,
        dataset=_infer_dataset(run_dir, metadata),
        seed=_metadata_int(
            metadata, ("effective_seed", "seed", "base_seed"), default=0
        ),
        run=_metadata_int(metadata, ("run",), default=0),
        k=k,
        scores=scores,
        train_matrix=train_matrix,
        heldout_relevance=relevance,
        eligible_items=eligible_items,
        base_topk=base_topk[:, :k],
        item_merit=_read_item_merit(
            run_dir / "item_merit.csv", scores.shape[1], relevance
        ),
        provider_ids=provider_ids,
    )


def _provider_mapping_path(run_dir: Path) -> Path:
    candidates = [
        run_dir / "provider_item_mapping.csv",
        run_dir / "method_inputs" / "provider_defaults" / "provider_item_mapping.csv",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


def _first_existing(paths: Sequence[Path]) -> Path:
    for path in paths:
        if path.is_file():
            return path
    return paths[0]


def _user_group_path(run: PreparedRun, method: str) -> Path:
    method_path = run.directory / "method_inputs" / method / "user_group_mapping.csv"
    fallback = run.directory / "groups" / "cpfair_user_activity.csv"
    generic = run.directory / "user_group_mapping.csv"
    return _first_existing([method_path, fallback, generic])


def _item_group_path(run: PreparedRun, method: str) -> Path:
    method_path = run.directory / "method_inputs" / method / "item_group_mapping.csv"
    method_fallbacks = {
        "CPFair": run.directory / "groups" / "cpfair_item_popularity.csv",
        "MultiFR": run.directory / "groups" / "multifr_item_popularity.csv",
        "TSFD": run.directory / "groups" / "tsfd_item_merit.csv",
    }
    generic = run.directory / "item_group_mapping.csv"
    return _first_existing(
        [method_path, method_fallbacks.get(method, generic), generic]
    )


def _validate_cpfair_groups(groups: Mapping[int, int], *, name: str) -> None:
    if set(groups.values()) != {ADVANTAGED_GROUP, PROTECTED_GROUP}:
        raise ValueError(
            f"{name} must contain both advantaged ({ADVANTAGED_GROUP}) and "
            f"protected ({PROTECTED_GROUP}) groups"
        )


def _mapped_original_ids(
    run: PreparedRun,
    *,
    stakeholder: str,
    expected_size: int,
) -> list[object]:
    path = run.directory / "mappings.json"
    if not path.is_file():
        raise FileNotFoundError(f"required experiment input not found: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    values = document.get(stakeholder)
    if not isinstance(values, list) or len(values) != expected_size:
        raise ValueError(f"{path} must contain {expected_size} aligned {stakeholder}")
    return values


def _raw_gender_lookup(path: Path, *, id_column: str) -> dict[str, str]:
    frame = _read_table(path)
    required = {id_column, "gender"}
    missing_columns = required.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"{path} is missing required columns: {sorted(missing_columns)}")
    # Match the preparation pipeline's existing first-record rule for duplicate
    # AMBAR user metadata.
    mapping = frame[[id_column, "gender"]].drop_duplicates(
        subset=[id_column],
        keep="first",
    )
    return {
        str(entity_id): str(gender).strip().lower()
        for entity_id, gender in mapping.itertuples(index=False, name=None)
        if not pd.isna(gender)
    }


def _ambar_cpfair_groups(
    run: PreparedRun,
) -> tuple[dict[int, int], dict[int, int], dict[int, int]]:
    """Return aligned AMBAR user, item, and provider gender groups.

    Female is the protected value and maps to ``PROTECTED_GROUP`` (+1); every
    other label, including ``undefined``, maps to ``ADVANTAGED_GROUP`` (-1).
    """

    original_users = _mapped_original_ids(
        run,
        stakeholder="users",
        expected_size=run.scores.shape[0],
    )
    user_gender = _raw_gender_lookup(
        AMBAR_RAW_DIRECTORY / "users_info.csv",
        id_column="user_id",
    )
    missing_users = [
        original_user
        for original_user in original_users
        if str(original_user) not in user_gender
    ]
    if missing_users:
        raise ValueError(
            f"AMBAR users_info.csv does not cover retained users: {missing_users[:5]}"
        )
    user_groups = {
        user_id: (
            PROTECTED_GROUP
            if user_gender[str(original_user)] in FEMALE_LABELS
            else ADVANTAGED_GROUP
        )
        for user_id, original_user in enumerate(original_users)
    }

    provider_path = _provider_mapping_path(run.directory)
    provider_frame = _read_table(provider_path)
    required = {"provider_id", "original_artist_id"}
    missing_columns = required.difference(provider_frame.columns)
    if missing_columns:
        raise ValueError(
            f"{provider_path} is missing required columns: {sorted(missing_columns)}"
        )
    provider_mapping = provider_frame[
        ["provider_id", "original_artist_id"]
    ].drop_duplicates()
    provider_ids = pd.to_numeric(provider_mapping["provider_id"], errors="coerce")
    if provider_ids.isna().any() or not provider_ids.eq(provider_ids.round()).all():
        raise ValueError(f"{provider_path} column 'provider_id' must contain integers")
    provider_mapping["provider_id"] = provider_ids.astype("int64")
    if provider_mapping["provider_id"].duplicated().any():
        raise ValueError(
            f"{provider_path} maps a provider_id to multiple original artists"
        )
    artist_by_provider = provider_mapping.set_index("provider_id")["original_artist_id"]
    recommended_provider_ids = set(run.provider_ids.tolist())
    missing_providers = sorted(
        recommended_provider_ids.difference(artist_by_provider.index)
    )
    if missing_providers:
        raise ValueError(
            f"{provider_path} does not cover provider IDs: {missing_providers[:5]}"
        )

    artist_gender = _raw_gender_lookup(
        AMBAR_RAW_DIRECTORY / "artists_info.csv",
        id_column="artist_id",
    )
    missing_artists = [
        artist_id
        for artist_id in artist_by_provider.loc[
            sorted(recommended_provider_ids)
        ].tolist()
        if str(artist_id) not in artist_gender
    ]
    if missing_artists:
        raise ValueError(
            "AMBAR artists_info.csv does not cover retained artists: "
            f"{missing_artists[:5]}"
        )
    provider_groups = {
        provider_id: (
            PROTECTED_GROUP
            if artist_gender[str(artist_by_provider.loc[provider_id])] in FEMALE_LABELS
            else ADVANTAGED_GROUP
        )
        for provider_id in sorted(recommended_provider_ids)
    }
    item_groups = {
        item_id: provider_groups[int(provider_id)]
        for item_id, provider_id in enumerate(run.provider_ids.tolist())
    }
    _validate_cpfair_groups(user_groups, name="AMBAR consumer gender groups")
    _validate_cpfair_groups(item_groups, name="AMBAR provider gender item groups")
    return user_groups, item_groups, provider_groups


# ponytail: unbounded dict cache, one entry per run directory per process. Without
# it the raw AMBAR CSVs are re-read ~10x per run (4 group specs x baseline+fair,
# plus the method calls).
_AMBAR_GROUP_CACHE: dict[Path, tuple[dict[int, int], dict[int, int], dict[int, int]]] = {}


def _ambar_gender_groups(
    run: PreparedRun,
) -> tuple[dict[int, int], dict[int, int], dict[int, int]] | None:
    """Return AMBAR user/item/provider gender groups, or ``None`` for other datasets."""

    if run.dataset.lower() != "ambar":
        return None
    if run.directory not in _AMBAR_GROUP_CACHE:
        _AMBAR_GROUP_CACHE[run.directory] = _ambar_cpfair_groups(run)
    return _AMBAR_GROUP_CACHE[run.directory]


def _group_vector(run: PreparedRun, method: str, stakeholder: str) -> torch.Tensor:
    gender_groups = _ambar_gender_groups(run)
    if gender_groups is not None and stakeholder in {"user", "item"}:
        groups, size = (
            (gender_groups[0], run.scores.shape[0])
            if stakeholder == "user"
            else (gender_groups[1], run.scores.shape[1])
        )
        return torch.as_tensor(
            [groups[entity_id] for entity_id in range(size)], dtype=torch.long
        )
    if stakeholder == "user":
        return _mapping_vector(
            _user_group_path(run, method),
            id_column="user_id",
            value_column="user_group",
            size=run.scores.shape[0],
        )
    if stakeholder == "item":
        return _mapping_vector(
            _item_group_path(run, method),
            id_column="item_id",
            value_column="item_group",
            size=run.scores.shape[1],
        )
    raise ValueError(f"unsupported group stakeholder: {stakeholder}")


def _provider_values_from_item_values(
    item_values: torch.Tensor,
    provider_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    frame = pd.DataFrame(
        {
            "provider_id": provider_ids.detach().cpu().numpy(),
            "value": item_values.detach().cpu().numpy(),
        }
    )
    grouped = frame.groupby("provider_id", sort=True)["value"].sum()
    return (
        torch.as_tensor(grouped.index.to_numpy(), dtype=torch.long),
        torch.as_tensor(grouped.to_numpy(), dtype=torch.float64),
    )


def _grouped_values(
    values: torch.Tensor,
    groups: torch.Tensor,
    *,
    aggregation: str,
) -> torch.Tensor:
    frame = pd.DataFrame(
        {
            "group": groups.detach().cpu().numpy(),
            "value": values.detach().cpu().numpy(),
        }
    )
    grouped = frame.groupby("group", sort=True)["value"]
    if aggregation == "mean":
        aggregated = grouped.mean()
    elif aggregation == "sum":
        aggregated = grouped.sum()
    else:
        raise ValueError(f"unsupported aggregation: {aggregation}")
    return torch.as_tensor(aggregated.to_numpy(), dtype=torch.float64)


def _grouped_values_and_merit(
    values: torch.Tensor,
    merit: torch.Tensor,
    groups: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    frame = pd.DataFrame(
        {
            "group": groups.detach().cpu().numpy(),
            "value": values.detach().cpu().numpy(),
            "merit": merit.detach().cpu().numpy(),
        }
    )
    grouped = frame.groupby("group", sort=True)[["value", "merit"]].sum()
    return (
        torch.as_tensor(grouped["value"].to_numpy(), dtype=torch.float64),
        torch.as_tensor(grouped["merit"].to_numpy(), dtype=torch.float64),
    )


def _exposure(rankings: torch.Tensor, *, num_items: int, k: int) -> torch.Tensor:
    topk = rankings[:, :k].to(torch.long)
    if (topk < 0).any() or (topk >= num_items).any():
        raise ValueError("rankings contain item IDs outside the item universe")
    discounts = torch.reciprocal(
        torch.log2(torch.arange(2, k + 2, dtype=torch.float64))
    )
    exposure = torch.zeros(num_items, dtype=torch.float64)
    for rank in range(k):
        exposure.scatter_add_(
            0,
            topk[:, rank].detach().cpu(),
            torch.full(
                (topk.shape[0],), float(discounts[rank].item()), dtype=torch.float64
            ),
        )
    return exposure


def _strictly_positive_ratio_variance(
    values: torch.Tensor, merit: torch.Tensor
) -> float:
    mask = merit > 0
    if not torch.any(mask):
        raise ValueError(
            "exposure/relevance ratio requires at least one positive merit"
        )
    return exposure_relevance_ratio_variance(values[mask], merit[mask])


def _mean_ndcg(relevance: torch.Tensor, rankings: torch.Tensor, k: int) -> float:
    return float(ndcg(relevance, rankings, k=k).mean)


def _policy_to_topk(policy: torch.Tensor, k: int) -> torch.Tensor:
    """Convert an ``(users, items, ranks)`` policy tensor to top-k item IDs."""

    tensor = torch.as_tensor(policy)
    if tensor.ndim != 3:
        raise ValueError("policy must have shape (users, items, ranks)")
    if not 1 <= k <= tensor.shape[1]:
        raise ValueError(f"k must be between 1 and items ({tensor.shape[1]}), got {k}")
    ranks = torch.arange(tensor.shape[2], device=tensor.device, dtype=tensor.dtype)
    weights = torch.reciprocal(torch.log2(ranks + 2))
    expected_exposure = torch.sum(tensor * weights.view(1, 1, -1), dim=2)
    return torch.topk(expected_exposure, k=k, dim=1).indices.to(torch.long)


def _as_output(scores: torch.Tensor, topk: torch.Tensor, k: int) -> MethodOutput:
    if topk.ndim != 2 or topk.shape[0] != scores.shape[0] or topk.shape[1] < k:
        raise ValueError("method top-k output must have shape (users, at least k)")
    return MethodOutput(
        scores=scores.detach().cpu(), topk=topk[:, :k].detach().cpu().to(torch.long)
    )


def _run_tfrom(run: PreparedRun, config: RunnerConfig) -> MethodOutput:
    from rr.tfrom import tfrom

    scores, topk = tfrom(
        run.scores.clone(), k=run.k, item_provider_mapping=run.provider_ids
    )
    return _as_output(scores, topk, run.k)


def _run_leadfairrec(run: PreparedRun, config: RunnerConfig) -> MethodOutput:
    from obm.leadfairrec import leadfairrec

    user_profile_embeddings, item_profile_embeddings = _leadfairrec_profile_embeddings(
        run,
        config,
    )
    scores, topk = leadfairrec(
        run.train_matrix.clone(),
        run.provider_ids,
        run.scores.clone(),
        k=run.k,
        user_profile_embeddings=user_profile_embeddings,
        item_profile_embeddings=item_profile_embeddings,
        semantic_weight=config.leadfairrec_semantic_weight,
        eligible_items=run.eligible_items,
        device=config.device,
    )
    return _as_output(scores, topk, run.k)


def _run_fairsort(run: PreparedRun, config: RunnerConfig) -> MethodOutput:
    from rr.fairsort import fairsort

    scores, topk = fairsort(
        run.scores.clone(),
        item_to_provider=run.provider_ids,
        k=run.k,
        eligible_items=run.eligible_items,
    )
    return _as_output(scores, topk, run.k)


def _cpfair_style_groups(run: PreparedRun) -> tuple[dict[int, int], dict[int, int]]:
    """Return the user and item groups CPFair scores against.

    AMBAR uses gender; every other dataset uses the top-20% activity split for
    users and the short-head/long-tail split for items. Shared with
    ``_run_tfld_groups`` so both group-fairness methods see the same groups.
    """

    gender_groups = _ambar_gender_groups(run)
    if gender_groups is not None:
        return gender_groups[0], gender_groups[1]

    user_activity = torch.count_nonzero(run.train_matrix, dim=1).tolist()
    active_count = max(1, math.ceil(0.2 * len(user_activity)))
    active_users = set(
        sorted(
            range(len(user_activity)),
            key=lambda user_id: (-user_activity[user_id], user_id),
        )[:active_count]
    )
    user_groups = {
        user_id: ADVANTAGED_GROUP if user_id in active_users else PROTECTED_GROUP
        for user_id in range(len(user_activity))
    }
    item_groups = _mapping_dict_for_cpfair(
        _item_group_path(run, "CPFair"),
        id_column="item_id",
        value_column="item_group",
    )
    return user_groups, item_groups


def _run_cpfair(run: PreparedRun, config: RunnerConfig) -> MethodOutput:
    from pam.greedy_cpfair import cpfair

    user_groups, item_groups = _cpfair_style_groups(run)
    scores, topk = cpfair(
        run.scores.clone(),
        user_groups=user_groups,
        item_groups=item_groups,
        base_topk=run.base_topk.detach().cpu().numpy(),
        k=run.k,
        device=config.device,
    )
    return _as_output(scores, topk, run.k)


def _run_multifr(run: PreparedRun, config: RunnerConfig) -> MethodOutput:
    from obm.multifr import multifr

    scores, topk = multifr(
        run.train_matrix.clone(),
        user_groups=_group_vector(run, "MultiFR", "user"),
        item_groups=_group_vector(run, "MultiFR", "item"),
        k=run.k,
        positive_threshold=config.multifr_positive_threshold,
        epochs=config.multifr_epochs,
        rounds=config.multifr_rounds,
        batch_size=config.multifr_batch_size,
        seed=run.seed,
        device=config.device,
    )
    return _as_output(scores, topk, run.k)


def _run_fairrec(run: PreparedRun, config: RunnerConfig) -> MethodOutput:
    from pam.fairrec import fairrec

    scores, topk = fairrec(run.scores.clone(), k=run.k)
    return _as_output(scores, topk, run.k)


def _run_ada2fair(run: PreparedRun, config: RunnerConfig) -> MethodOutput:
    from obm.ada2fair import ada2fair

    scores, topk = ada2fair(
        run.train_matrix.clone(),
        run.provider_ids,
        k=run.k,
        epochs=config.ada2fair_epochs,
        weight_epochs=config.ada2fair_weight_epochs,
        seed=run.seed,
        eligible_items=run.eligible_items,
        device=config.device,
    )
    return _as_output(scores, topk, run.k)


def _run_tfld(run: PreparedRun, config: RunnerConfig) -> MethodOutput:
    from obm.tfld import tfld

    scores, policy, _, _ = tfld(
        run.scores.clone(),
        epochs=config.tfld_epochs,
        k=run.k,
        alpha=config.tfld_alpha,
        lamb=config.tfld_lambda,
        device=torch.device(config.device),
    )
    return _as_output(scores, _policy_to_topk(policy.detach().cpu(), run.k), run.k)


def _run_tfld_groups(run: PreparedRun, config: RunnerConfig) -> MethodOutput:
    from obm.tfld_groups import tfld_groups

    user_groups, item_groups = _cpfair_style_groups(run)
    exposure, topk, _, _ = tfld_groups(
        run.scores.clone(),
        user_groups=torch.tensor(
            [user_groups[user_id] for user_id in range(run.scores.shape[0])],
            dtype=torch.long,
        ),
        item_groups=torch.tensor(
            [item_groups[item_id] for item_id in range(run.scores.shape[1])],
            dtype=torch.long,
        ),
        epochs=config.tfld_epochs,
        k=run.k,
        alpha=config.tfld_alpha,
        lamb=config.tfld_lambda,
        device=torch.device(config.device),
    )
    return _as_output(exposure, topk, run.k)


def _run_ggf(run: PreparedRun, config: RunnerConfig) -> MethodOutput:
    from obm.ggf import ggf

    scores, policy, _, _ = ggf(
        run.scores.clone(),
        epochs=config.ggf_epochs,
        k=run.k,
        device=torch.device(config.device),
    )
    return _as_output(scores, _policy_to_topk(policy.detach().cpu(), run.k), run.k)


def _run_ggf_user_quantile(run: PreparedRun, config: RunnerConfig) -> MethodOutput:
    """GGF with the paper's Task 2 user weights (Eq. 7, omega = 1).

    The default ``w1 = ones`` is Task 1: utilitarian on users, so it applies no
    Rawlsian pressure to them. Where GGF is scored on worse-off user utility it
    has to optimize that same criterion instead.
    """
    from obm.ggf import get_w_quantile, ggf

    scores, policy, _, _ = ggf(
        run.scores.clone(),
        epochs=config.ggf_epochs,
        k=run.k,
        user_weights=get_w_quantile(
            run.scores.shape[0], config.worse_off_fraction, omega=1.0
        ),
        device=torch.device(config.device),
    )
    return _as_output(scores, _policy_to_topk(policy.detach().cpu(), run.k), run.k)


def _run_seal(run: PreparedRun, config: RunnerConfig) -> MethodOutput:
    from pam.seal import seal

    scores, topk = seal(run.scores.clone(), k=run.k)
    return _as_output(scores, topk, run.k)


def _run_tsfd(run: PreparedRun, config: RunnerConfig) -> MethodOutput:
    from obm.tsfd import tsfd

    scores, topk = tsfd(
        run.scores.clone(),
        user_groups=_group_vector(run, "TSFD", "user"),
        item_groups=_group_vector(run, "TSFD", "item"),
        k=run.k,
        maxiter=config.tsfd_maxiter,
        seed=run.seed,
    )
    return _as_output(scores, topk, run.k)


_METHOD_ADAPTERS: dict[str, MethodAdapter] = {
    "TFROM": _run_tfrom,
    "LeadFairRec": _run_leadfairrec,
    "FairSort": _run_fairsort,
    "CPFair": _run_cpfair,
    #    "MultiFR": _run_multifr,
    "FairRec": _run_fairrec,
    "Ada2Fair": _run_ada2fair,
    "TFLD": _run_tfld,
    "TFLD-Groups": _run_tfld_groups,
    "GGF": _run_ggf,
    "SEAL": _run_seal,
    "TSFD": _run_tsfd,
}

# Methods whose instantiation depends on the fairness criterion being scored.
# Keyed by (method, group id); anything absent uses _METHOD_ADAPTERS.
_GROUP_ADAPTERS: dict[tuple[str, int], MethodAdapter] = {
    ("GGF", 4): _run_ggf_user_quantile,
}


def _fairness_metric(
    run: PreparedRun,
    method: str,
    output: MethodOutput,
    group: GroupSpec,
    config: RunnerConfig,
) -> float:
    user_ndcg = ndcg(run.heldout_relevance, output.topk, k=run.k).values.to(
        torch.float64
    )
    item_exposure = _exposure(output.topk, num_items=run.scores.shape[1], k=run.k)

    if group.number == 1:
        return utility_variance(user_ndcg)
    if group.number == 2:
        return utility_variance(
            _grouped_values(
                user_ndcg, _group_vector(run, method, "user"), aggregation="mean"
            )
        )
    if group.number == 3:
        return envy_freeness(
            run.heldout_relevance, output.topk, k=run.k
        ).mean_average_envy
    if group.number == 4:
        return float(
            worse_off_cumulative_utility(user_ndcg, config.worse_off_fraction).item()
        )
    if group.number == 5:
        return sum_user_utilities(user_ndcg)
    if group.number == 6:
        return sum_user_utilities(
            _grouped_values(
                user_ndcg, _group_vector(run, method, "user"), aggregation="mean"
            )
        )
    if group.number == 7:
        _, provider_exposure = _provider_values_from_item_values(
            item_exposure, run.provider_ids
        )
        return utility_variance(provider_exposure)
    if group.number == 8:
        return utility_variance(
            _grouped_values(
                item_exposure,
                _group_vector(run, method, "item"),
                aggregation="sum",
            )
        )
    if group.number == 9:
        _, provider_exposure = _provider_values_from_item_values(
            item_exposure, run.provider_ids
        )
        _, provider_merit = _provider_values_from_item_values(
            run.item_merit, run.provider_ids
        )
        return _strictly_positive_ratio_variance(provider_exposure, provider_merit)
    if group.number == 10:
        group_exposure, group_merit = _grouped_values_and_merit(
            item_exposure,
            run.item_merit,
            _group_vector(run, method, "item"),
        )
        return _strictly_positive_ratio_variance(group_exposure, group_merit)
    if group.number == 11:
        _, provider_exposure = _provider_values_from_item_values(
            item_exposure, run.provider_ids
        )
        return float(
            worse_off_cumulative_utility(
                provider_exposure, config.worse_off_fraction
            ).item()
        )
    if group.number == 12:
        return float(
            worse_off_cumulative_utility(
                item_exposure, config.worse_off_fraction
            ).item()
        )
    raise ValueError(f"unsupported group: {group.number}")


def _result_row(
    run: PreparedRun,
    method: str,
    output: MethodOutput,
    group: GroupSpec,
    config: RunnerConfig,
) -> dict[str, object]:
    baseline_output = MethodOutput(scores=run.scores, topk=run.base_topk)
    return {
        "method": method,
        "dataset": run.dataset,
        "seed": run.seed,
        "run": run.run,
        "k": run.k,
        "ndcg_baseline": _mean_ndcg(run.heldout_relevance, run.base_topk, run.k),
        "ndcg_fair_method": _mean_ndcg(run.heldout_relevance, output.topk, run.k),
        "stakeholder": group.stakeholder,
        "granularity": group.granularity,
        "fairness_metric_name": group.metric_name,
        "fairness_metric_baseline": _fairness_metric(
            run,
            method,
            baseline_output,
            group,
            config,
        ),
        "fairness_metric": _fairness_metric(run, method, output, group, config),
    }


def _write_tsv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=RESULT_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def discover_run_directories(runs_root: str | Path) -> list[Path]:
    """Return prepared ``seed_*`` run directories under ``runs_root``."""

    root = _resolve(runs_root)
    candidates = sorted(root.glob("*/run_*/seed_*"))
    return [path for path in candidates if path.is_dir()]


def run_grouped_experiments(
    *,
    runs_root: str | Path = PROJECT_ROOT / "experiments" / "runs",
    results_root: str | Path = PROJECT_ROOT / "results",
    run_dirs: Sequence[str | Path] | None = None,
    datasets: Sequence[str] | None = None,
    recommendation_list_size: int | None = None,
    config: RunnerConfig | None = None,
    groups: Sequence[GroupSpec] = GROUPS,
) -> list[Path]:
    """Execute configured groups and return written TSV paths."""

    resolved_config = config or RunnerConfig()
    directories = (
        [_resolve(path) for path in run_dirs]
        if run_dirs is not None
        else discover_run_directories(runs_root)
    )
    if datasets is not None:
        selected_datasets = set(datasets)
        directories = [
            path
            for path in directories
            if _infer_dataset(path, _metadata(path)) in selected_datasets
        ]
    if not directories:
        raise ValueError("no prepared run directories found")

    results_base = _resolve(results_root)
    written: list[Path] = []
    for run_dir in directories:
        prepared = _load_prepared_run(run_dir, recommendation_list_size)
        method_cache: dict[tuple[str, int], MethodOutput] = {}
        for group in groups:
            rows: list[dict[str, object]] = []
            for method in group.methods:
                # Group-specific adapters get their own cache entry; every other
                # method is computed once per run and shared across groups.
                group_adapter = _GROUP_ADAPTERS.get((method, group.number))
                cache_key = (method, group.number if group_adapter else -1)
                if cache_key not in method_cache:
                    if group_adapter is None:
                        try:
                            group_adapter = _METHOD_ADAPTERS[method]
                        except KeyError as error:
                            raise ValueError(
                                f"unsupported method: {method}"
                            ) from error
                    _seed_everything(prepared.seed)
                    method_cache[cache_key] = group_adapter(prepared, resolved_config)
                rows.append(
                    _result_row(
                        prepared,
                        method,
                        method_cache[cache_key],
                        group,
                        resolved_config,
                    )
                )
            output_path = (
                results_base
                / group.fairness_definition
                / group.output_directory_name
                / group.output_file_name
            )
            existing_rows: list[dict[str, object]] = []
            if output_path.is_file():
                with output_path.open(encoding="utf-8", newline="") as source:
                    existing_rows = list(csv.DictReader(source, delimiter="\t"))
            _write_tsv(output_path, [*existing_rows, *rows])
            written.append(output_path)
    return written


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run grouped two-sided fairness experiments.",
    )
    parser.add_argument(
        "--runs-root", default=str(PROJECT_ROOT / "experiments" / "runs")
    )
    parser.add_argument("--results-root", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--run-dir", action="append", default=None)
    parser.add_argument("--dataset", action="append", default=None)
    parser.add_argument(
        "--recommendation-list-size", type=_positive_int, default=None
    )
    parser.add_argument("--ada2fair-epochs", type=_positive_int, default=20)
    parser.add_argument("--ada2fair-weight-epochs", type=_positive_int, default=5)
    parser.add_argument("--leadfairrec-semantic-weight", type=float, default=0.15)
    parser.add_argument(
        "--leadfairrec-profile-embedding-dim", type=_positive_int, default=64
    )
    parser.add_argument("--leadfairrec-profile-model", default="deepseek-chat")
    parser.add_argument("--leadfairrec-profile-timeout", type=float, default=30.0)
    parser.add_argument("--multifr-epochs", type=_positive_int, default=20)
    parser.add_argument("--multifr-rounds", type=_positive_int, default=1)
    parser.add_argument("--multifr-batch-size", type=_positive_int, default=64)
    parser.add_argument("--ggf-epochs", type=_positive_int, default=200)
    parser.add_argument("--tfld-epochs", type=_positive_int, default=200)
    parser.add_argument("--tsfd-maxiter", type=_positive_int, default=500)
    parser.add_argument("--worse-off-fraction", type=float, default=0.25)
    parser.add_argument("--multifr-positive-threshold", type=float, default=0.8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    if (
        not math.isfinite(args.worse_off_fraction)
        or not 0 < args.worse_off_fraction <= 1
    ):
        raise ValueError("--worse-off-fraction must be in (0, 1]")
    if (
        not math.isfinite(args.leadfairrec_semantic_weight)
        or args.leadfairrec_semantic_weight < 0
    ):
        raise ValueError(
            "--leadfairrec-semantic-weight must be finite and non-negative"
        )
    if (
        not math.isfinite(args.leadfairrec_profile_timeout)
        or args.leadfairrec_profile_timeout <= 0
    ):
        raise ValueError("--leadfairrec-profile-timeout must be finite and positive")
    if (
        not math.isfinite(args.multifr_positive_threshold)
        or args.multifr_positive_threshold < 0
    ):
        raise ValueError("--multifr-positive-threshold must be non-negative")

    paths = run_grouped_experiments(
        runs_root=args.runs_root,
        results_root=args.results_root,
        run_dirs=args.run_dir,
        datasets=args.dataset,
        recommendation_list_size=args.recommendation_list_size,
        config=RunnerConfig(
            ada2fair_epochs=args.ada2fair_epochs,
            ada2fair_weight_epochs=args.ada2fair_weight_epochs,
            leadfairrec_semantic_weight=args.leadfairrec_semantic_weight,
            leadfairrec_profile_embedding_dim=args.leadfairrec_profile_embedding_dim,
            leadfairrec_profile_model=args.leadfairrec_profile_model,
            leadfairrec_profile_timeout=args.leadfairrec_profile_timeout,
            multifr_epochs=args.multifr_epochs,
            multifr_rounds=args.multifr_rounds,
            multifr_batch_size=args.multifr_batch_size,
            ggf_epochs=args.ggf_epochs,
            tfld_epochs=args.tfld_epochs,
            tsfd_maxiter=args.tsfd_maxiter,
            worse_off_fraction=args.worse_off_fraction,
            multifr_positive_threshold=args.multifr_positive_threshold,
            device=args.device,
        ),
    )
    for path in paths:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
