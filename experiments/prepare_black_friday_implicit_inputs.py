"""Prepare Black Friday implicit-feedback inputs for RR, PAM, and OBM methods."""

from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from .prepare_explicit_method_inputs import (
    _activity_groups,
    _age_label,
    _copy_method_inputs,
    _encode_labels,
    _quantile_groups,
    _run_seed_directory,
    _top_fraction_groups,
    _top_ids_by_count,
    _validate_optional_positive,
    _validate_same_user_partition,
    _write_base_topk,
    _write_csv,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw_datasets"
    / "implicit_feedback"
    / "Black Friday"
    / "black_friday.csv"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "runs"
DEFAULT_RUNS = 10


@dataclass(frozen=True)
class BlackFridayImplicitRun:
    """Paths produced for one Black Friday implicit preprocessing run."""

    run: int
    directory: Path
    train_interactions: Path
    validation_interactions: Path
    test_interactions: Path
    train_matrix: Path
    scores: Path
    heldout_relevance: Path
    eligible_items: Path
    base_topk: Path
    groups_directory: Path
    method_inputs_directory: Path
    mappings: Path
    metadata: Path


class _BPRMatrixFactorization(nn.Module):
    """Small BPR matrix-factorization model for implicit-feedback scoring."""

    def __init__(self, users: int, items: int, factors: int) -> None:
        super().__init__()
        self.user_factors = nn.Embedding(users, factors)
        self.item_factors = nn.Embedding(items, factors)
        nn.init.xavier_uniform_(self.user_factors.weight)
        nn.init.xavier_uniform_(self.item_factors.weight)

    def pair_scores(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return (self.user_factors(users) * self.item_factors(items)).sum(dim=1)

    def matrix(self) -> torch.Tensor:
        return self.user_factors.weight @ self.item_factors.weight.T


def _read_black_friday(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Black Friday data file not found: {path}")
    frame = pd.read_csv(path)
    required = {"User_ID", "Product_ID", "Gender", "Age", "Purchase"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    frame = frame[["User_ID", "Product_ID", "Gender", "Age", "Purchase"]].copy()
    frame["Purchase"] = pd.to_numeric(frame["Purchase"], errors="raise")
    purchases = frame["Purchase"].to_numpy(dtype=np.float64)
    if frame.empty or not np.isfinite(purchases).all() or (purchases <= 0).any():
        raise ValueError("Purchase must be non-empty, finite, and positive")
    return frame


def _mode_value(values: pd.Series) -> object:
    present = values.dropna()
    if present.empty:
        return np.nan
    modes = present.mode(dropna=True)
    if modes.empty:
        return present.iloc[0]
    return modes.iloc[0]


def _deduplicate_events(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["User_ID", "Product_ID"], as_index=False, sort=True)
        .agg(
            {
                "Gender": _mode_value,
                "Age": _mode_value,
                "Purchase": "mean",
            }
        )
        .reset_index(drop=True)
    )


def _split_by_user(
    interactions: pd.DataFrame,
    *,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    parts: list[list[int]] = [[], [], []]
    for _, group in interactions.groupby("User_ID", sort=True):
        indices = rng.permutation(group.index.to_numpy())
        validation_size = max(1, round(len(indices) * validation_ratio))
        test_size = max(1, round(len(indices) * test_ratio))
        if validation_size + test_size >= len(indices):
            validation_size = test_size = 1
        parts[1].extend(indices[:validation_size].tolist())
        parts[2].extend(indices[validation_size : validation_size + test_size].tolist())
        parts[0].extend(indices[validation_size + test_size :].tolist())
    return tuple(interactions.loc[index].copy() for index in parts)  # type: ignore[return-value]


def _warm_start_split(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    previous_users: set[object] | None = None
    while True:
        train_items = set(train["Product_ID"])
        validation = validation[validation["Product_ID"].isin(train_items)]
        test = test[test["Product_ID"].isin(train_items)]
        retained_users = set(validation["User_ID"]) & set(test["User_ID"])
        train = train[train["User_ID"].isin(retained_users)]
        validation = validation[validation["User_ID"].isin(retained_users)]
        test = test[test["User_ID"].isin(retained_users)]
        if retained_users == previous_users:
            break
        previous_users = retained_users
    if train.empty or validation.empty or test.empty:
        raise ValueError("split contains no users with warm-start validation and test data")
    return tuple(part.reset_index(drop=True) for part in (train, validation, test))


def _sorted_unique(values: pd.Series) -> list[object]:
    unique = values.drop_duplicates().tolist()
    return sorted(unique, key=lambda value: (type(value).__name__, str(value)))


def _map_events(
    frame: pd.DataFrame,
    users: dict[object, int],
    items: dict[object, int],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": frame["User_ID"].map(users).astype(int),
            "item_id": frame["Product_ID"].map(items).astype(int),
            "rating": 1.0,
            "purchase": frame["Purchase"].astype(float),
        }
    )


def _binary_matrix(frame: pd.DataFrame, users: int, items: int) -> np.ndarray:
    matrix = np.zeros((users, items), dtype=np.float32)
    matrix[
        frame["user_id"].to_numpy(dtype=int),
        frame["item_id"].to_numpy(dtype=int),
    ] = 1.0
    return matrix


def _fit_bpr(
    train: pd.DataFrame,
    users: int,
    items: int,
    *,
    factors: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    regularization: float,
    seed: int,
) -> np.ndarray:
    if min(factors, epochs, batch_size) <= 0:
        raise ValueError("factors, epochs, and batch_size must be positive")
    if learning_rate <= 0 or regularization < 0:
        raise ValueError("learning_rate must be positive and regularization non-negative")

    torch.manual_seed(seed)
    model = _BPRMatrixFactorization(users, items, factors)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    positive_users = torch.tensor(train["user_id"].to_numpy(), dtype=torch.long)
    positive_items = torch.tensor(train["item_id"].to_numpy(), dtype=torch.long)
    observed = _binary_matrix(train, users, items).astype(bool)
    negatives_by_user = [
        np.flatnonzero(~observed[user_id]).astype(np.int64)
        for user_id in range(users)
    ]
    if any(len(choices) == 0 for choices in negatives_by_user):
        raise ValueError("each user must have at least one unobserved item")

    rng = np.random.default_rng(seed)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.arange(len(train), dtype=torch.long)
    best_state = copy.deepcopy(model.state_dict())

    for _ in range(epochs):
        for batch_indices in indices[torch.randperm(len(indices), generator=generator)].split(
            batch_size
        ):
            users_batch = positive_users[batch_indices]
            positives_batch = positive_items[batch_indices]
            negative_items = [
                rng.choice(negatives_by_user[int(user_id)])
                for user_id in users_batch.tolist()
            ]
            negatives_batch = torch.tensor(negative_items, dtype=torch.long)

            positive_scores = model.pair_scores(users_batch, positives_batch)
            negative_scores = model.pair_scores(users_batch, negatives_batch)
            regularizer = (
                model.user_factors(users_batch).square().mean()
                + model.item_factors(positives_batch).square().mean()
                + model.item_factors(negatives_batch).square().mean()
            )
            loss = -F.logsigmoid(positive_scores - negative_scores).mean()
            loss = loss + regularization * regularizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    with torch.no_grad():
        return model.matrix().numpy().astype(np.float32)


def _user_attributes(
    frame: pd.DataFrame,
    original_users: Sequence[object],
) -> pd.DataFrame:
    raw = (
        frame[["User_ID", "Gender", "Age"]]
        .groupby("User_ID", as_index=False, sort=True)
        .agg({"Gender": _mode_value, "Age": _mode_value})
    )
    by_user = raw.set_index("User_ID")
    rows: list[dict[str, object]] = []
    for user_id, original in enumerate(original_users):
        row = by_user.loc[original]
        rows.append(
            {
                "user_id": user_id,
                "gender": str(row["Gender"]),
                "age_bin": _age_label(row["Age"]),
            }
        )
    return pd.DataFrame(rows)


def _demographic_groups(attributes: pd.DataFrame, *, composite: bool) -> pd.DataFrame:
    if composite:
        labels = [
            f"gender={row.gender}|age={row.age_bin}"
            for row in attributes.itertuples(index=False)
        ]
    else:
        labels = [f"gender={row.gender}" for row in attributes.itertuples(index=False)]
    return _encode_labels(
        attributes["user_id"].to_numpy(dtype=int).tolist(),
        labels,
        id_column="user_id",
        group_column="user_group",
    )


def _write_groups(
    *,
    run_dir: Path,
    train: pd.DataFrame,
    train_matrix: np.ndarray,
    attributes: pd.DataFrame,
    item_popularity_top_fraction: float,
    merit_quantiles: Sequence[float],
) -> tuple[Path, Path, dict[str, object]]:
    groups_dir = run_dir / "groups"
    method_dir = run_dir / "method_inputs"
    users, items = train_matrix.shape

    item_exposure = train_matrix.sum(axis=0).astype(np.float64)
    item_merit = item_exposure.copy()
    _write_csv(
        pd.DataFrame({"item_id": range(items), "merit": item_merit}),
        run_dir / "item_merit.csv",
    )

    cpfair_user = _write_csv(
        _activity_groups(train, users),
        groups_dir / "cpfair_user_activity.csv",
    )
    cpfair_item = _write_csv(
        _top_fraction_groups(
            item_exposure,
            id_column="item_id",
            group_column="item_group",
            value_column="train_exposure",
            top_fraction=item_popularity_top_fraction,
        ),
        groups_dir / "cpfair_item_popularity.csv",
    )
    multifr_user = _write_csv(
        _demographic_groups(attributes, composite=True),
        groups_dir / "multifr_user_gender_age.csv",
    )
    multifr_item = _write_csv(
        _top_fraction_groups(
            item_exposure,
            id_column="item_id",
            group_column="item_group",
            value_column="train_exposure",
            top_fraction=item_popularity_top_fraction,
        ),
        groups_dir / "multifr_item_popularity.csv",
    )
    tsfd_user = _write_csv(
        _demographic_groups(attributes, composite=False),
        groups_dir / "tsfd_user_gender.csv",
    )
    tsfd_item = _write_csv(
        _quantile_groups(
            item_merit,
            id_column="item_id",
            group_column="item_group",
            value_column="train_merit",
            quantiles=merit_quantiles,
        ),
        groups_dir / "tsfd_item_merit.csv",
    )
    provider_items = _write_csv(
        pd.DataFrame(
            {
                "item_id": range(items),
                "provider_id": range(items),
                "provider_source": ["item_as_provider_proxy"] * items,
            }
        ),
        run_dir / "provider_item_mapping.csv",
    )
    provider_groups = pd.read_csv(cpfair_item).rename(
        columns={"item_id": "provider_id", "item_group": "provider_group"}
    )
    provider_groups = provider_groups[
        ["provider_id", "provider_group", "group_label", "train_exposure"]
    ]
    provider_group_path = _write_csv(
        provider_groups,
        groups_dir / "provider_group_mapping.csv",
    )

    _copy_method_inputs(
        method_dir / "CPFair",
        user_groups=cpfair_user,
        item_groups=cpfair_item,
        provider_items=provider_items,
    )
    _copy_method_inputs(
        method_dir / "MultiFR",
        user_groups=multifr_user,
        item_groups=multifr_item,
        provider_items=provider_items,
    )
    _copy_method_inputs(
        method_dir / "TSFD",
        user_groups=tsfd_user,
        item_groups=tsfd_item,
        provider_items=provider_items,
    )
    _copy_method_inputs(method_dir / "provider_defaults", provider_items=provider_items)
    pd.read_csv(provider_group_path).to_csv(
        method_dir / "provider_defaults" / "provider_group_mapping.csv",
        index=False,
    )

    metadata = {
        "feedback_type": "implicit",
        "heldout_relevance": "binary_held_out_purchases",
        "item_provider_semantics": "item_as_provider_proxy",
        "cpfair_user_groups": "training_user_activity_median_split",
        "cpfair_item_groups": "training_item_exposure_short_head_long_tail",
        "multifr_user_groups": "real_gender_age_composite",
        "multifr_item_groups": "training_item_exposure_short_head_long_tail",
        "tsfd_user_groups": "real_gender",
        "tsfd_item_groups": "training_item_merit_quantile_bins",
        "item_popularity_top_fraction": item_popularity_top_fraction,
        "merit_quantiles": list(merit_quantiles),
    }
    return groups_dir, method_dir, metadata


def _write_shared_run_zero_artifacts(*, run: int, run_dir: Path, data_root: Path) -> None:
    if run != 0:
        return
    evaluation_dir = data_root / "evaluation"
    mapping_dir = data_root / "mappings"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    mapping_dir.mkdir(parents=True, exist_ok=True)

    relevance = pd.read_parquet(run_dir / "heldout_relevance.parquet")
    relevance.to_parquet(evaluation_dir / "black_friday_relevance.parquet", index=False)
    merit = relevance.mean(axis=0).reset_index()
    merit.columns = ["item_id", "merit"]
    merit.to_csv(evaluation_dir / "black_friday_merit.csv", index=False)

    pd.read_csv(run_dir / "groups" / "cpfair_user_activity.csv").to_csv(
        mapping_dir / "black_friday_user_group_mapping.csv",
        index=False,
    )
    pd.read_csv(run_dir / "groups" / "cpfair_item_popularity.csv").to_csv(
        mapping_dir / "black_friday_item_group_mapping.csv",
        index=False,
    )
    pd.read_csv(run_dir / "provider_item_mapping.csv").to_csv(
        mapping_dir / "black_friday_provider_item_mapping.csv",
        index=False,
    )
    pd.read_csv(run_dir / "groups" / "provider_group_mapping.csv").to_csv(
        mapping_dir / "black_friday_provider_group_mapping.csv",
        index=False,
    )


def prepare_black_friday_implicit_run(
    *,
    run: int,
    source_path: str | Path = DEFAULT_SOURCE_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    seed: int = 42,
    interactions: int | None = None,
    max_users: int | None = None,
    max_items: int | None = None,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    factors: int = 32,
    epochs: int = 20,
    batch_size: int = 4096,
    learning_rate: float = 0.01,
    regularization: float = 1e-4,
    recommendation_list_length: int = 10,
    item_popularity_top_fraction: float = 0.2,
    merit_quantiles: Sequence[float] = (0.5,),
    write_shared_run_zero: bool = True,
) -> BlackFridayImplicitRun:
    """Prepare one Black Friday implicit-feedback run."""

    if run < 0 or seed < 0:
        raise ValueError("seed and run must be non-negative")
    _validate_optional_positive(interactions, "interactions")
    _validate_optional_positive(max_users, "max_users")
    _validate_optional_positive(max_items, "max_items")
    if not 0 < validation_ratio < 1 or not 0 < test_ratio < 1:
        raise ValueError("validation_ratio and test_ratio must be in (0, 1)")
    if validation_ratio + test_ratio >= 1:
        raise ValueError("validation_ratio + test_ratio must be less than 1")
    if recommendation_list_length <= 0:
        raise ValueError("recommendation_list_length must be positive")

    effective_seed = seed + run
    source = _deduplicate_events(_read_black_friday(Path(source_path)))
    if interactions is not None:
        if interactions > len(source):
            raise ValueError(
                f"requested {interactions} interactions but data contains {len(source)}"
            )
        source = source.sample(interactions, random_state=effective_seed)

    item_ids = _top_ids_by_count(source["Product_ID"], max_items)
    if item_ids is not None:
        source = source[source["Product_ID"].isin(item_ids)].copy()

    counts = source["User_ID"].value_counts()
    source = source[source["User_ID"].isin(counts[counts >= 3].index)].copy()
    if source.empty:
        raise ValueError("data contains no users with at least three interactions")

    user_ids = _top_ids_by_count(source["User_ID"], max_users)
    if user_ids is not None:
        source = source[source["User_ID"].isin(user_ids)].copy()

    train_raw, validation_raw, test_raw = _warm_start_split(
        *_split_by_user(
            source,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
            seed=effective_seed,
        )
    )

    original_users = _sorted_unique(train_raw["User_ID"])
    original_items = _sorted_unique(train_raw["Product_ID"])
    users = {value: index for index, value in enumerate(original_users)}
    items = {value: index for index, value in enumerate(original_items)}

    train = _map_events(train_raw, users, items)
    validation = _map_events(validation_raw, users, items)
    test = _map_events(test_raw, users, items)
    user_attributes = _user_attributes(source, original_users)
    train_matrix = _binary_matrix(train, len(users), len(items))
    heldout_relevance = _binary_matrix(test, len(users), len(items))
    _validate_same_user_partition(
        train=train,
        validation=validation,
        test=test,
        train_matrix=train_matrix,
        heldout_relevance=heldout_relevance,
    )
    eligible_items = train_matrix == 0
    scores = _fit_bpr(
        train,
        len(users),
        len(items),
        factors=factors,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        regularization=regularization,
        seed=effective_seed,
    )

    run_dir = _run_seed_directory(
        output_root,
        "black_friday_implicit",
        run,
        effective_seed,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    train_path = run_dir / "train_interactions.parquet"
    validation_path = run_dir / "validation_interactions.parquet"
    test_path = run_dir / "test_interactions.parquet"
    train_matrix_path = run_dir / "train_matrix.parquet"
    scores_path = run_dir / "scores.parquet"
    heldout_path = run_dir / "heldout_relevance.parquet"
    eligible_path = run_dir / "eligible_items.parquet"
    base_topk_path = run_dir / "base_topk.parquet"
    mappings_path = run_dir / "mappings.json"
    metadata_path = run_dir / "metadata.json"

    train.to_parquet(train_path, index=False)
    validation.to_parquet(validation_path, index=False)
    test.to_parquet(test_path, index=False)
    pd.DataFrame(train_matrix).to_parquet(train_matrix_path, index=False)
    pd.DataFrame(scores).to_parquet(scores_path, index=False)
    pd.DataFrame(heldout_relevance).to_parquet(heldout_path, index=False)
    pd.DataFrame(eligible_items).to_parquet(eligible_path, index=False)
    effective_k = _write_base_topk(
        scores_path,
        eligible_path,
        base_topk_path,
        recommendation_list_length,
    )

    groups_dir, method_dir, group_metadata = _write_groups(
        run_dir=run_dir,
        train=train,
        train_matrix=train_matrix,
        attributes=user_attributes,
        item_popularity_top_fraction=item_popularity_top_fraction,
        merit_quantiles=merit_quantiles,
    )

    mappings = {
        "users": [value.item() if isinstance(value, np.generic) else value for value in original_users],
        "items": [value.item() if isinstance(value, np.generic) else value for value in original_items],
    }
    mappings_path.write_text(json.dumps(mappings, indent=2), encoding="utf-8")

    metadata = {
        "dataset": "black_friday_implicit",
        "source_path": str(source_path),
        "sampled_interactions": len(source),
        "preprocessing_max_users": max_users,
        "preprocessing_max_items": max_items,
        "output_root": str(Path(output_root)),
        "run_directory": str(run_dir),
        "train_interactions": len(train),
        "validation_interactions": len(validation),
        "test_interactions": len(test),
        "num_users": len(users),
        "num_items": len(items),
        "seed": seed,
        "base_seed": seed,
        "run": run,
        "effective_seed": effective_seed,
        "validation_ratio": validation_ratio,
        "test_ratio": test_ratio,
        "model": "bpr_matrix_factorization",
        "factors": factors,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "regularization": regularization,
        "recommendation_list_length": recommendation_list_length,
        "effective_base_topk_length": effective_k,
        "purchase_value_policy": "metadata_only_not_ndcg_relevance",
        "user_partition_policy": "same_users_in_train_validation_test",
        **group_metadata,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if write_shared_run_zero:
        _write_shared_run_zero_artifacts(
            run=run,
            run_dir=run_dir,
            data_root=PROJECT_ROOT / "data",
        )

    return BlackFridayImplicitRun(
        run=run,
        directory=run_dir,
        train_interactions=train_path,
        validation_interactions=validation_path,
        test_interactions=test_path,
        train_matrix=train_matrix_path,
        scores=scores_path,
        heldout_relevance=heldout_path,
        eligible_items=eligible_path,
        base_topk=base_topk_path,
        groups_directory=groups_dir,
        method_inputs_directory=method_dir,
        mappings=mappings_path,
        metadata=metadata_path,
    )


def prepare_black_friday_implicit_inputs(
    *,
    runs: int = DEFAULT_RUNS,
    source_path: str | Path = DEFAULT_SOURCE_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    seed: int = 42,
    interactions: int | None = None,
    max_users: int | None = None,
    max_items: int | None = None,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    factors: int = 32,
    epochs: int = 20,
    batch_size: int = 4096,
    learning_rate: float = 0.01,
    regularization: float = 1e-4,
    recommendation_list_length: int = 10,
    item_popularity_top_fraction: float = 0.2,
    merit_quantiles: Sequence[float] = (0.5,),
    write_shared_run_zero: bool = True,
) -> list[BlackFridayImplicitRun]:
    """Prepare all requested Black Friday implicit-feedback runs."""

    if runs <= 0:
        raise ValueError("runs must be positive")
    prepared: list[BlackFridayImplicitRun] = []
    for run in range(runs):
        prepared.append(
            prepare_black_friday_implicit_run(
                run=run,
                source_path=source_path,
                output_root=output_root,
                seed=seed,
                interactions=interactions,
                max_users=max_users,
                max_items=max_items,
                validation_ratio=validation_ratio,
                test_ratio=test_ratio,
                factors=factors,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                regularization=regularization,
                recommendation_list_length=recommendation_list_length,
                item_popularity_top_fraction=item_popularity_top_fraction,
                merit_quantiles=merit_quantiles,
                write_shared_run_zero=write_shared_run_zero,
            )
        )
    return prepared


def build_parser() -> argparse.ArgumentParser:
    """Create the Black Friday implicit preprocessing command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--source-path", type=Path, default=DEFAULT_SOURCE_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--interactions", type=int)
    parser.add_argument("--max-users", type=int)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--factors", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--regularization", type=float, default=1e-4)
    parser.add_argument("--recommendation-list-length", type=int, default=10)
    parser.add_argument("--item-popularity-top-fraction", type=float, default=0.2)
    parser.add_argument("--merit-quantiles", type=float, nargs="+", default=[0.5])
    parser.add_argument(
        "--no-shared-run-zero",
        action="store_true",
        help="Do not refresh data/evaluation and data/mappings from run 0.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run Black Friday implicit preprocessing from the command line."""

    args = build_parser().parse_args(argv)
    try:
        prepared = prepare_black_friday_implicit_inputs(
            runs=args.runs,
            source_path=args.source_path,
            output_root=args.output_root,
            seed=args.seed,
            interactions=args.interactions,
            max_users=args.max_users,
            max_items=args.max_items,
            validation_ratio=args.validation_ratio,
            test_ratio=args.test_ratio,
            factors=args.factors,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            regularization=args.regularization,
            recommendation_list_length=args.recommendation_list_length,
            item_popularity_top_fraction=args.item_popularity_top_fraction,
            merit_quantiles=args.merit_quantiles,
            write_shared_run_zero=not args.no_shared_run_zero,
        )
    except (FileNotFoundError, OSError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    for item in prepared:
        print(item.directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
