"""Prepare held-out explicit-rating data and shared matrix-factorization scores."""

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
import torch.nn.functional as F
from torch import nn

from .metrics import ndcg

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results"


@dataclass(frozen=True)
class ExperimentPaths:
    """Files produced by :func:`prepare_experiment`."""

    directory: Path
    train: Path
    validation: Path
    test: Path
    predictions: Path
    test_relevance: Path
    eligible_items: Path
    item_groups: Path
    provider_items: Path | None
    mappings: Path
    metadata: Path


class _BiasedMF(nn.Module):
    """Matrix factorization with an item bias, scored pairwise for ranking.

    User bias and global mean are omitted: they are constant within a user, so
    they cancel in pairwise score differences and cannot affect rankings.
    """

    def __init__(self, users: int, items: int, factors: int) -> None:
        super().__init__()
        self.user_factors = nn.Embedding(users, factors)
        self.item_factors = nn.Embedding(items, factors)
        self.item_bias = nn.Embedding(items, 1)
        nn.init.normal_(self.user_factors.weight, std=0.01)
        nn.init.normal_(self.item_factors.weight, std=0.01)
        nn.init.zeros_(self.item_bias.weight)

    def pair_scores(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        interaction = (self.user_factors(users) * self.item_factors(items)).sum(1)
        return self.item_bias(items).squeeze(1) + interaction

    def matrix_for_ranking(self) -> torch.Tensor:
        return self.item_bias.weight.T + self.user_factors.weight @ self.item_factors.weight.T


def _read_interactions(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"interaction file not found: {path}")
    frame = pd.read_csv(path, sep="\t", header=None)
    if frame.shape[1] not in {3, 4}:
        raise ValueError("interaction data must have 3 or 4 tab-separated columns")
    frame.columns = ["user_id", "item_id", "rating", "timestamp"][: frame.shape[1]]
    frame = frame[["user_id", "item_id", "rating"]].copy()
    frame["rating"] = pd.to_numeric(frame["rating"], errors="raise")
    ratings = frame["rating"].to_numpy(dtype=np.float64)
    if frame.empty or not np.isfinite(ratings).all() or (ratings <= 0).any():
        raise ValueError("ratings must be non-empty, finite, and positive")
    return frame


def _split(
    frame: pd.DataFrame,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    parts = [[], [], []]
    for _, group in frame.groupby("user_id", sort=True):
        indices = rng.permutation(group.index)
        validation_size = max(1, round(len(indices) * validation_ratio))
        test_size = max(1, round(len(indices) * test_ratio))
        if validation_size + test_size >= len(indices):
            validation_size = test_size = 1
        parts[1].extend(indices[:validation_size])
        parts[2].extend(indices[validation_size : validation_size + test_size])
        parts[0].extend(indices[validation_size + test_size :])

    train, validation, test = (frame.loc[indices].copy() for indices in parts)
    previous_users: set[object] | None = None
    while True:
        train_items = set(train["item_id"])
        validation = validation[validation["item_id"].isin(train_items)]
        test = test[test["item_id"].isin(train_items)]
        users = set(validation["user_id"]) & set(test["user_id"])
        train = train[train["user_id"].isin(users)]
        validation = validation[validation["user_id"].isin(users)]
        test = test[test["user_id"].isin(users)]
        if users == previous_users:
            break
        previous_users = users
    if train.empty or validation.empty or test.empty:
        raise ValueError("split contains no users with warm-start validation and test data")
    return tuple(part.reset_index(drop=True) for part in (train, validation, test))


def _mapped(
    frame: pd.DataFrame,
    users: dict[object, int],
    items: dict[object, int],
    rating_scale: float,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": frame["user_id"].map(users).astype(int),
            "item_id": frame["item_id"].map(items).astype(int),
            "rating": frame["rating"].astype(float) / rating_scale,
        }
    )


def _matrix(frame: pd.DataFrame, users: int, items: int) -> np.ndarray:
    result = np.zeros((users, items), dtype=np.float32)
    result[frame["user_id"], frame["item_id"]] = frame["rating"]
    return result


def _sample_graded_negatives(
    train_matrix: torch.Tensor,
    users: torch.Tensor,
    positive_ratings: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample one item per pair with a strictly lower rating (unrated counts as 0).

    ponytail: uniform draw + one resample round, leftovers masked out of the
    loss; sparsity makes invalid draws rare, so per-user rating buckets are
    not worth their complexity.
    """

    items = train_matrix.shape[1]
    negatives = torch.randint(items, (len(users),), generator=generator)
    invalid = train_matrix[users, negatives] >= positive_ratings
    if invalid.any():
        negatives[invalid] = torch.randint(items, (int(invalid.sum()),), generator=generator)
    valid = train_matrix[users, negatives] < positive_ratings
    return negatives, valid


def _validation_ndcg(
    model: _BiasedMF,
    train_matrix: torch.Tensor,
    validation_matrix: torch.Tensor,
    k: int,
) -> float:
    with torch.no_grad():
        scores = model.matrix_for_ranking().masked_fill(train_matrix > 0, -torch.inf)
        rankings = torch.topk(scores, k=k, dim=1).indices
        return float(ndcg(validation_matrix, rankings, k=k).mean)


def _per_user_min_max(matrix: torch.Tensor) -> torch.Tensor:
    low = matrix.min(dim=1, keepdim=True).values
    span = matrix.max(dim=1, keepdim=True).values - low
    normalized = (matrix - low) / span.clamp_min(torch.finfo(matrix.dtype).eps)
    return torch.where(span > 0, normalized, torch.full_like(matrix, 0.5))


def _fit_mf(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    users: int,
    items: int,
    *,
    factors: int,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    regularization: float,
    seed: int,
) -> tuple[np.ndarray, int, float]:
    torch.manual_seed(seed)
    model = _BiasedMF(users, items, factors)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    user_ids = torch.tensor(train["user_id"].to_numpy(), dtype=torch.long)
    item_ids = torch.tensor(train["item_id"].to_numpy(), dtype=torch.long)
    ratings = torch.tensor(train["rating"].to_numpy(), dtype=torch.float32)
    train_matrix = torch.zeros((users, items), dtype=torch.float32)
    train_matrix[user_ids, item_ids] = ratings
    validation_matrix = torch.tensor(
        _matrix(validation, users, items), dtype=torch.float32
    )
    evaluation_k = min(10, items)
    generator = torch.Generator().manual_seed(seed)
    best_state, best_ndcg, stale, best_epoch = copy.deepcopy(model.state_dict()), -math.inf, 0, 0

    for epoch in range(1, epochs + 1):
        for batch in torch.randperm(len(train), generator=generator).split(batch_size):
            batch_users = user_ids[batch]
            positives = item_ids[batch]
            negatives, valid = _sample_graded_negatives(
                train_matrix, batch_users, ratings[batch], generator
            )
            score_gap = model.pair_scores(batch_users, positives) - model.pair_scores(
                batch_users, negatives
            )
            ranking_loss = -(F.logsigmoid(score_gap) * valid).sum() / valid.sum().clamp_min(1)
            regularizer = (
                model.user_factors(batch_users).square().mean()
                + model.item_factors(positives).square().mean()
                + model.item_factors(negatives).square().mean()
                + model.item_bias(positives).square().mean()
                + model.item_bias(negatives).square().mean()
            )
            loss = ranking_loss + regularization * regularizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        validation_ndcg = _validation_ndcg(model, train_matrix, validation_matrix, evaluation_k)
        if validation_ndcg > best_ndcg + 1e-6:
            best_state, best_ndcg, stale, best_epoch = (
                copy.deepcopy(model.state_dict()),
                validation_ndcg,
                0,
                epoch,
            )
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(best_state)
    with torch.no_grad():
        predictions = _per_user_min_max(model.matrix_for_ranking()).numpy()
    return predictions, best_epoch, best_ndcg


def _json_value(value: object) -> object:
    return value.item() if isinstance(value, np.generic) else value


def _encode_mapping(
    path: Path,
    item_ids: list[object],
    value_column: str,
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"item_id", value_column}
    if missing := required.difference(frame.columns):
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if frame["item_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate item_id values")
    values = frame.set_index("item_id")[value_column]
    missing_items = [item for item in item_ids if item not in values.index]
    if missing_items:
        raise ValueError(f"{path} does not cover retained item IDs: {missing_items[:5]}")
    original = [values.loc[item] for item in item_ids]
    labels = sorted(set(original), key=lambda value: (type(value).__name__, str(value)))
    encoding = {label: index for index, label in enumerate(labels)}
    return pd.DataFrame(
        {
            "item_id": range(len(item_ids)),
            value_column: [encoding[value] for value in original],
            f"original_{value_column}": original,
        }
    )


def _popularity_groups(
    train_original: pd.DataFrame,
    item_ids: list[object],
    top_fraction: float,
) -> pd.DataFrame:
    counts = train_original["item_id"].value_counts()
    order = sorted(
        range(len(item_ids)),
        key=lambda index: (-int(counts.get(item_ids[index], 0)), index),
    )
    short_head = set(order[: max(1, math.ceil(len(order) * top_fraction))])
    return pd.DataFrame(
        {
            "item_id": range(len(item_ids)),
            "item_group": [-1 if index in short_head else 1 for index in range(len(item_ids))],
            "original_item_group": [
                "short_head" if index in short_head else "long_tail"
                for index in range(len(item_ids))
            ],
        }
    )


def prepare_experiment(
    data_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    interactions: int | None = None,
    seed: int = 42,
    run: int = 0,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    factors: int = 32,
    epochs: int = 200,
    patience: int = 10,
    batch_size: int = 1024,
    learning_rate: float = 0.01,
    regularization: float = 1e-4,
    provider_item_mapping: str | Path | None = None,
    item_group_mapping: str | Path | None = None,
    popularity_top_fraction: float = 0.2,
) -> ExperimentPaths:
    """Prepare aligned held-out data, MF predictions, groups, and providers."""

    if run < 0 or seed < 0:
        raise ValueError("seed and run must be non-negative")
    if not 0 < validation_ratio < 1 or not 0 < test_ratio < 1:
        raise ValueError("validation_ratio and test_ratio must be in (0, 1)")
    if validation_ratio + test_ratio >= 1:
        raise ValueError("validation_ratio + test_ratio must be less than 1")
    if min(factors, epochs, patience, batch_size) <= 0:
        raise ValueError("factors, epochs, patience, and batch_size must be positive")
    if learning_rate <= 0 or regularization < 0:
        raise ValueError("learning_rate must be positive and regularization non-negative")
    if not 0 < popularity_top_fraction <= 1:
        raise ValueError("popularity_top_fraction must be in (0, 1]")

    source_path = Path(data_path)
    effective_seed = seed + run
    source = _read_interactions(source_path)
    if interactions is not None:
        if interactions <= 0 or interactions > len(source):
            raise ValueError(f"interactions must be between 1 and {len(source)}")
        source = source.sample(interactions, random_state=effective_seed)
    aggregated = source.groupby(["user_id", "item_id"], as_index=False)["rating"].mean()
    counts = aggregated["user_id"].value_counts()
    aggregated = aggregated[aggregated["user_id"].isin(counts[counts >= 3].index)]
    if aggregated.empty:
        raise ValueError("data contains no users with at least three distinct interactions")

    train_original, validation_original, test_original = _split(
        aggregated, validation_ratio, test_ratio, effective_seed
    )
    user_ids = sorted(set(train_original["user_id"]), key=lambda value: (type(value).__name__, str(value)))
    item_ids = sorted(set(train_original["item_id"]), key=lambda value: (type(value).__name__, str(value)))
    users = {value: index for index, value in enumerate(user_ids)}
    items = {value: index for index, value in enumerate(item_ids)}
    rating_scale = float(aggregated["rating"].max())
    train = _mapped(train_original, users, items, rating_scale)
    validation = _mapped(validation_original, users, items, rating_scale)
    test = _mapped(test_original, users, items, rating_scale)
    groups = (
        _encode_mapping(Path(item_group_mapping), item_ids, "item_group")
        if item_group_mapping
        else _popularity_groups(train_original, item_ids, popularity_top_fraction)
    )
    providers = (
        _encode_mapping(Path(provider_item_mapping), item_ids, "provider_id")
        if provider_item_mapping
        else None
    )
    predictions, best_epoch, validation_ndcg = _fit_mf(
        train,
        validation,
        len(users),
        len(items),
        factors=factors,
        epochs=epochs,
        patience=patience,
        batch_size=batch_size,
        learning_rate=learning_rate,
        regularization=regularization,
        seed=effective_seed,
    )
    train_matrix = _matrix(train, len(users), len(items))
    test_matrix = _matrix(test, len(users), len(items))

    sample_name = "all" if interactions is None else str(interactions)
    destination = (
        DEFAULT_RESULTS_ROOT / f"{source_path.stem}_{sample_name}_run{run}"
        if output_dir is None
        else Path(output_dir)
    )
    destination.mkdir(parents=True, exist_ok=True)
    provider_path = destination / "provider_item_mapping.csv" if provider_item_mapping else None
    paths = ExperimentPaths(
        directory=destination,
        train=destination / "train.parquet",
        validation=destination / "validation.parquet",
        test=destination / "test.parquet",
        predictions=destination / "predictions.parquet",
        test_relevance=destination / "test_relevance.parquet",
        eligible_items=destination / "eligible_items.parquet",
        item_groups=destination / "item_group_mapping.csv",
        provider_items=provider_path,
        mappings=destination / "mappings.json",
        metadata=destination / "metadata.json",
    )
    train.to_parquet(paths.train, index=False)
    validation.to_parquet(paths.validation, index=False)
    test.to_parquet(paths.test, index=False)
    pd.DataFrame(predictions).to_parquet(paths.predictions, index=False)
    pd.DataFrame(test_matrix).to_parquet(paths.test_relevance, index=False)
    pd.DataFrame(train_matrix == 0).to_parquet(paths.eligible_items, index=False)

    groups.to_csv(paths.item_groups, index=False)
    if providers is not None and provider_path is not None:
        providers.to_csv(provider_path, index=False)

    paths.mappings.write_text(
        json.dumps(
            {
                "users": [_json_value(value) for value in user_ids],
                "items": [_json_value(value) for value in item_ids],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    metadata = {
        "source_path": str(source_path),
        "sampled_interactions": len(source),
        "aggregated_interactions": len(aggregated),
        "train_interactions": len(train),
        "validation_interactions": len(validation),
        "test_interactions": len(test),
        "num_users": len(users),
        "num_items": len(items),
        "seed": seed,
        "run": run,
        "effective_seed": effective_seed,
        "validation_ratio": validation_ratio,
        "test_ratio": test_ratio,
        "rating_scale": rating_scale,
        "model": "graded_bpr_matrix_factorization",
        "loss": "graded_pair_bpr",
        "score_normalization": "per_user_min_max",
        "early_stopping_metric": "validation_ndcg@10",
        "factors": factors,
        "maximum_epochs": epochs,
        "best_epoch": best_epoch,
        "validation_ndcg": validation_ndcg,
        "patience": patience,
        "item_group_source": str(item_group_mapping) if item_group_mapping else "training_popularity",
        "provider_mapping_source": str(provider_item_mapping) if provider_item_mapping else None,
        "generated_groups_are_providers": False,
        "popularity_top_fraction": popularity_top_fraction,
    }
    paths.metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--interactions", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run", type=int, default=0)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--factors", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--regularization", type=float, default=1e-4)
    parser.add_argument("--provider-item-mapping", type=Path)
    parser.add_argument("--item-group-mapping", type=Path)
    parser.add_argument("--popularity-top-fraction", type=float, default=0.2)
    return parser

if __name__ == "__main__":
    args = vars(build_parser().parse_args(argv))
    try:
        paths = prepare_experiment(**args)
    except (FileNotFoundError, OSError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error
