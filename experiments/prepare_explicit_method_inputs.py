"""Prepare explicit-feedback inputs for RR, PAM, and OBM methods."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .experiments import ExperimentPaths, prepare_experiment


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "runs"
DEFAULT_RUNS = 10
DEFAULT_AMBAR_MAX_USERS = 5_000
DEFAULT_AMBAR_MAX_ITEMS = 5_000
DEFAULT_MAX_DENSE_CELLS = 25_000_000
AMAZON_BRAND_METADATA = Path("raw_datasets/explicit_feedback/Amazon/meta_All_Beauty.json")
UNKNOWN_AMAZON_BRANDS = frozenset({
    "", "-", "generic", "n/a", "na", "no brand", "none", "not applicable", "unbranded", "unknown",
})

DATASETS: Mapping[str, str] = {
    "amazon": "raw_datasets/explicit_feedback/Amazon/All_Beauty.json",
    "ambar": "raw_datasets/explicit_feedback/AMBAR/ratings_info.csv",
    "black_friday": "raw_datasets/implicit_feedback/Black Friday/black_friday.csv",
    "movielens": "raw_datasets/explicit_feedback/Movie Lens/ratings.dat",
    "yelp": "raw_datasets/explicit_feedback/Yelp/yelp_academic_dataset_review.json",
}

USER_ALIASES = {"user", "userid", "useridx", "uid", "customerid", "reviewerid"}
ITEM_ALIASES = {
    "item",
    "itemid",
    "itemidx",
    "iid",
    "movieid",
    "productid",
    "businessid",
    "asin",
    "trackid",
}
RATING_ALIASES = {"rating", "score", "stars", "purchase", "purchaseamount", "overall"}
GENDER_ALIASES = {"gender", "sex"}
AGE_ALIASES = {"age", "agegroup"}


@dataclass(frozen=True)
class PreparedRun:
    """Paths produced for one dataset/run preprocessing job."""

    dataset: str
    run: int
    directory: Path
    core_paths: ExperimentPaths
    train_matrix: Path
    scores: Path
    heldout_relevance: Path
    base_topk: Path
    groups_directory: Path
    method_inputs_directory: Path
    metadata: Path


@dataclass(frozen=True)
class _RawDataset:
    interactions: pd.DataFrame
    user_attributes: pd.DataFrame


@dataclass(frozen=True)
class _PreprocessingResult:
    interactions: pd.DataFrame
    metadata: dict[str, object]


def _normalized_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())


def _run_seed_directory(
    output_root: str | Path,
    dataset: str,
    run: int,
    effective_seed: int,
) -> Path:
    """Return the canonical generated-artifact directory for one run/seed."""

    return Path(output_root) / dataset / f"run_{run:02d}" / f"seed_{effective_seed}"


def _validate_same_user_partition(
    *,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    train_matrix: np.ndarray,
    heldout_relevance: np.ndarray,
) -> None:
    """Ensure all evaluation matrices are aligned to the same warm-start users."""

    user_sets = {
        "train": set(train["user_id"].astype(int)),
        "validation": set(validation["user_id"].astype(int)),
        "test": set(test["user_id"].astype(int)),
    }
    if not user_sets["train"]:
        raise ValueError("split contains no train users")
    if user_sets["train"] != user_sets["validation"] or user_sets["train"] != user_sets["test"]:
        raise ValueError("train, validation, and test splits must contain the same users")
    if train_matrix.shape != heldout_relevance.shape:
        raise ValueError("train matrix and held-out relevance must have the same shape")
    users = train_matrix.shape[0]
    for name, frame in (("train", train), ("validation", validation), ("test", test)):
        counts = np.bincount(frame["user_id"].to_numpy(dtype=int), minlength=users)
        if counts.shape[0] != users or np.any(counts[:users] == 0):
            raise ValueError(f"each mapped user must have at least one {name} interaction")
    if np.any(np.sum(train_matrix > 0, axis=1) == 0):
        raise ValueError("each mapped user must have at least one train interaction")
    if np.any(np.sum(heldout_relevance > 0, axis=1) == 0):
        raise ValueError("each mapped user must have at least one held-out interaction")


def _column_by_alias(frame: pd.DataFrame, aliases: set[str]) -> str | None:
    for column in frame.columns:
        if _normalized_name(column) in aliases:
            return str(column)
    return None


def _read_headered_table(path: Path) -> pd.DataFrame | None:
    try:
        frame = pd.read_csv(path, sep=None, engine="python")
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        return None
    required = (
        _column_by_alias(frame, USER_ALIASES),
        _column_by_alias(frame, ITEM_ALIASES),
        _column_by_alias(frame, RATING_ALIASES),
    )
    if all(required):
        return frame
    return None


def _read_unheadered_table(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path, sep=None, engine="python", header=None)
    except pd.errors.ParserError:
        frame = pd.read_csv(path, sep="\t", header=None)
    if frame.shape[1] < 3:
        raise ValueError(f"{path} must contain at least user, item, and rating columns")
    names = ["user_id", "item_id", "rating", "timestamp"]
    frame = frame.iloc[:, : min(frame.shape[1], len(names))].copy()
    frame.columns = names[: frame.shape[1]]
    return frame


def _read_movielens_ratings(path: Path) -> pd.DataFrame | None:
    try:
        frame = pd.read_csv(path, sep="::", engine="python", header=None)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        return None
    if frame.shape[1] < 3:
        return None
    frame = frame.iloc[:, :4].copy()
    frame.columns = ["user_id", "item_id", "rating", "timestamp"][: frame.shape[1]]
    return frame


def _read_json_lines_table(path: Path) -> pd.DataFrame | None:
    try:
        frame = pd.read_json(path, lines=True)
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    required = (
        _column_by_alias(frame, USER_ALIASES),
        _column_by_alias(frame, ITEM_ALIASES),
        _column_by_alias(frame, RATING_ALIASES),
    )
    if all(required):
        return frame
    return None


def _mode_value(values: pd.Series) -> object:
    present = values.dropna()
    if present.empty:
        return np.nan
    modes = present.mode(dropna=True)
    if modes.empty:
        return present.iloc[0]
    return modes.iloc[0]


def _user_attributes_from_frame(frame: pd.DataFrame, user_column: str) -> pd.DataFrame:
    attribute_columns: list[str] = []
    gender_column = _column_by_alias(frame, GENDER_ALIASES)
    age_column = _column_by_alias(frame, AGE_ALIASES)
    if gender_column:
        attribute_columns.append(gender_column)
    if age_column:
        attribute_columns.append(age_column)
    if not attribute_columns:
        return pd.DataFrame(columns=["user_id"])

    attributes = (
        frame[[user_column, *attribute_columns]]
        .groupby(user_column, as_index=False, sort=True)
        .agg(_mode_value)
    )
    rename = {user_column: "user_id"}
    if gender_column:
        rename[gender_column] = "gender"
    if age_column:
        rename[age_column] = "age"
    return attributes.rename(columns=rename)


def _read_raw_dataset(path: Path) -> _RawDataset:
    if not path.is_file():
        raise FileNotFoundError(f"explicit dataset file not found: {path}")

    if path.name == Path(DATASETS["yelp"]).name:
        chunks = pd.read_json(path, lines=True, chunksize=100_000, convert_dates=False)
        interactions = pd.concat([
            chunk[["user_id", "business_id", "stars"]].astype({
                "user_id": "string[pyarrow]", "business_id": "string[pyarrow]",
            })
            for chunk in chunks
        ], ignore_index=True).rename(columns={"business_id": "item_id", "stars": "rating"})
        if interactions.empty or not np.isfinite(interactions["rating"]).all() or (interactions["rating"] <= 0).any():
            raise ValueError(f"{path} ratings must be non-empty, finite, and positive")
        return _RawDataset(interactions, pd.DataFrame(columns=["user_id"]))

    frame: pd.DataFrame | None = None
    if path.suffix.lower() == ".json":
        frame = _read_json_lines_table(path)
    elif path.name == "ratings.dat":
        frame = _read_movielens_ratings(path)
    if frame is None:
        frame = _read_headered_table(path)
    if frame is None:
        frame = _read_unheadered_table(path)

    user_column = _column_by_alias(frame, USER_ALIASES)
    item_column = _column_by_alias(frame, ITEM_ALIASES)
    rating_column = _column_by_alias(frame, RATING_ALIASES)
    if not user_column or not item_column or not rating_column:
        raise ValueError(
            f"{path} must expose user, item, and rating columns or be a 3/4-column file"
        )

    interactions = frame[[user_column, item_column, rating_column]].copy()
    interactions.columns = ["user_id", "item_id", "rating"]
    interactions["rating"] = pd.to_numeric(interactions["rating"], errors="raise")
    ratings = interactions["rating"].to_numpy(dtype=np.float64)
    if interactions.empty or not np.isfinite(ratings).all() or (ratings <= 0).any():
        raise ValueError(f"{path} ratings must be non-empty, finite, and positive")

    return _RawDataset(
        interactions=interactions,
        user_attributes=_user_attributes_from_frame(frame, user_column),
    )


def _read_user_sidecar(data_root: Path, dataset: str) -> pd.DataFrame:
    path = data_root / "attributes" / f"{dataset}_users.csv"
    if not path.is_file():
        if dataset == "ambar":
            ambar_users = (
                data_root / "raw_datasets" / "explicit_feedback" / "AMBAR" / "users_info.csv"
            )
            if ambar_users.is_file():
                frame = pd.read_csv(ambar_users)
                user_column = _column_by_alias(frame, USER_ALIASES)
                gender_column = _column_by_alias(frame, GENDER_ALIASES)
                if user_column is None or gender_column is None:
                    raise ValueError(f"{ambar_users} must contain user_id and gender columns")
                return frame[[user_column, gender_column]].rename(
                    columns={user_column: "user_id", gender_column: "gender"}
                )
        if dataset == "movielens":
            movielens_users = (
                data_root / "raw_datasets" / "explicit_feedback" / "Movie Lens" / "users.dat"
            )
            if movielens_users.is_file():
                frame = pd.read_csv(
                    movielens_users,
                    sep="::",
                    engine="python",
                    header=None,
                    names=["user_id", "gender", "age", "occupation", "zip_code"],
                )
                return frame[["user_id", "gender", "age"]]
        return pd.DataFrame(columns=["user_id"])
    frame = pd.read_csv(path)
    user_column = _column_by_alias(frame, USER_ALIASES)
    if user_column is None:
        raise ValueError(f"{path} must contain a user_id column")
    columns = [user_column]
    rename = {user_column: "user_id"}
    gender_column = _column_by_alias(frame, GENDER_ALIASES)
    age_column = _column_by_alias(frame, AGE_ALIASES)
    if gender_column:
        columns.append(gender_column)
        rename[gender_column] = "gender"
    if age_column:
        columns.append(age_column)
        rename[age_column] = "age"
    return frame[columns].rename(columns=rename)


def _merge_user_attributes(raw: pd.DataFrame, sidecar: pd.DataFrame) -> pd.DataFrame:
    if sidecar.empty:
        return raw.copy()
    if raw.empty:
        return sidecar.copy()
    return pd.concat([sidecar, raw], ignore_index=True).drop_duplicates(
        subset=["user_id"],
        keep="first",
    )


def _filter_interactions_to_attributed_users(
    interactions: pd.DataFrame,
    attributes: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    if attributes.empty:
        return interactions.copy(), 0

    exact_ids = set(attributes["user_id"])
    string_ids = {str(user_id) for user_id in exact_ids}
    mask = interactions["user_id"].isin(exact_ids) | interactions["user_id"].astype(
        str
    ).isin(string_ids)
    dropped = int((~mask).sum())
    filtered = interactions[mask].copy()
    if filtered.empty:
        raise ValueError("no interactions remain after filtering users with attributes")
    return filtered, dropped


def _validate_optional_positive(value: int | None, name: str) -> None:
    if value is not None and value <= 0:
        raise ValueError(f"{name} must be positive when provided")


def _top_ids_by_count(values: pd.Series, limit: int | None) -> set[object] | None:
    if limit is None or values.nunique(dropna=False) <= limit:
        return None
    counts = values.value_counts(sort=False)
    ordered = sorted(
        counts.index.tolist(),
        key=lambda value: (-int(counts.loc[value]), type(value).__name__, str(value)),
    )
    return set(ordered[:limit])


def _filter_to_warm_start_users(interactions: pd.DataFrame) -> pd.DataFrame:
    distinct_items = interactions.groupby("user_id")["item_id"].nunique()
    warm_users = set(distinct_items[distinct_items >= 3].index.tolist())
    filtered = interactions[interactions["user_id"].isin(warm_users)].copy()
    if filtered.empty:
        raise ValueError("preprocessing removed all users with at least three items")
    return filtered


def _limit_interactions_for_dense_pipeline(
    interactions: pd.DataFrame,
    *,
    dataset: str,
    max_users: int | None,
    max_items: int | None,
) -> _PreprocessingResult:
    original_interactions = len(interactions)
    original_users = int(interactions["user_id"].nunique())
    original_items = int(interactions["item_id"].nunique())
    filtered = interactions.copy()

    item_ids = _top_ids_by_count(filtered["item_id"], max_items)
    if item_ids is not None:
        filtered = filtered[filtered["item_id"].isin(item_ids)].copy()
        if filtered.empty:
            raise ValueError("item cap removed all interactions")

    filtered = _filter_to_warm_start_users(filtered)
    if max_users is not None and filtered["user_id"].nunique() > max_users:
        distinct_items = filtered.groupby("user_id")["item_id"].nunique()
        ordered_users = sorted(
            distinct_items.index.tolist(),
            key=lambda user_id: (
                -int(distinct_items.loc[user_id]),
                type(user_id).__name__,
                str(user_id),
            ),
        )
        filtered = filtered[filtered["user_id"].isin(set(ordered_users[:max_users]))].copy()
        filtered = _filter_to_warm_start_users(filtered)

    retained_users = int(filtered["user_id"].nunique())
    retained_items = int(filtered["item_id"].nunique())
    metadata = {
        "preprocessing_dataset": dataset,
        "preprocessing_max_users": max_users,
        "preprocessing_max_items": max_items,
        "preprocessing_original_interactions": original_interactions,
        "preprocessing_original_users": original_users,
        "preprocessing_original_items": original_items,
        "preprocessing_retained_interactions": len(filtered),
        "preprocessing_retained_users": retained_users,
        "preprocessing_retained_items": retained_items,
        "preprocessing_dropped_interactions": original_interactions - len(filtered),
        "preprocessing_dense_cells": retained_users * retained_items,
    }
    return _PreprocessingResult(interactions=filtered, metadata=metadata)


def _validate_dense_size(
    interactions: pd.DataFrame,
    *,
    max_dense_cells: int | None,
) -> int:
    users = int(interactions["user_id"].nunique())
    items = int(interactions["item_id"].nunique())
    dense_cells = users * items
    if max_dense_cells is not None and dense_cells > max_dense_cells:
        raise ValueError(
            "preprocessed data would require a dense matrix with "
            f"{dense_cells:,} cells ({users:,} users x {items:,} items), "
            f"which exceeds max_dense_cells={max_dense_cells:,}; lower "
            "max_users/max_items or raise max_dense_cells if the machine can handle it"
        )
    return dense_cells


def _encode_provider_mapping(original_provider_ids: Sequence[object]) -> pd.DataFrame:
    labels = sorted(
        set(original_provider_ids),
        key=lambda value: (type(value).__name__, str(value)),
    )
    encoding = {label: index for index, label in enumerate(labels)}
    return pd.DataFrame(
        {
            "provider_id": [encoding[value] for value in original_provider_ids],
            "original_artist_id": list(original_provider_ids),
        }
    )


def _ambar_provider_item_mapping(
    data_root: Path,
    original_items: Sequence[object],
) -> tuple[pd.DataFrame, str]:
    path = data_root / "raw_datasets" / "explicit_feedback" / "AMBAR" / "tracks_info.csv"
    if not path.is_file():
        raise FileNotFoundError(f"AMBAR track metadata file not found: {path}")

    frame = pd.read_csv(path)
    required = {"track_id", "artist_id"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if frame["track_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate track_id values")

    exact = frame.set_index("track_id")["artist_id"]
    string_index = frame.assign(_key=frame["track_id"].astype(str)).set_index("_key")[
        "artist_id"
    ]
    original_artist_ids: list[object] = []
    missing_items: list[object] = []
    for original_item in original_items:
        if original_item in exact.index:
            original_artist_ids.append(exact.loc[original_item])
        elif str(original_item) in string_index.index:
            original_artist_ids.append(string_index.loc[str(original_item)])
        else:
            missing_items.append(original_item)
    if missing_items:
        raise ValueError(
            f"{path} does not cover retained AMBAR track IDs: {missing_items[:5]}"
        )

    providers = _encode_provider_mapping(original_artist_ids)
    providers.insert(0, "item_id", range(len(original_items)))
    providers["provider_source"] = "ambar_tracks_info.artist_id"
    return providers, "artist_id_from_ambar_tracks_info"


def _provider_item_mapping(
    *,
    dataset: str,
    data_root: Path,
    original_items: Sequence[object],
    amazon_brands: Mapping[str, str] | None = None,
) -> tuple[pd.DataFrame, str]:
    if dataset == "ambar":
        return _ambar_provider_item_mapping(data_root, original_items)
    if amazon_brands is not None:
        asins = pd.Series(original_items, dtype="string")
        brands = asins.map(amazon_brands)
        if brands.isna().any():
            raise ValueError("Amazon brand partition requires a known brand for each retained ASIN")
        provider_ids, _ = pd.factorize(brands, sort=True)
        return pd.DataFrame({
            "item_id": range(len(asins)),
            "provider_id": provider_ids,
            "asin": asins,
            "brand": brands,
            "provider_source": "amazon_product_metadata.brand",
        }), "brand_from_amazon_product_metadata"
    return (
        pd.DataFrame(
            {
                "item_id": range(len(original_items)),
                "provider_id": range(len(original_items)),
                "provider_source": ["item_as_provider_proxy"] * len(original_items),
            }
        ),
        "item_as_provider_proxy",
    )


def _amazon_brands_by_asin(path: Path) -> dict[str, str]:
    """Keep ASINs with exactly one known, normalized brand in the metadata."""
    by_asin: dict[str, set[str]] = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            asin, raw_brand = row.get("asin"), row.get("brand")
            if not isinstance(asin, str) or not asin or not isinstance(raw_brand, str):
                continue
            brand = " ".join(raw_brand.casefold().split())
            if brand not in UNKNOWN_AMAZON_BRANDS:
                by_asin.setdefault(asin, set()).add(brand)
    return {asin: next(iter(brands)) for asin, brands in by_asin.items() if len(brands) == 1}


def _write_standard_interactions(interactions: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    interactions[["user_id", "item_id", "rating"]].to_csv(
        path,
        sep="\t",
        header=False,
        index=False,
    )


def _matrix_from_interactions(
    interactions: pd.DataFrame,
    *,
    users: int,
    items: int,
) -> np.ndarray:
    matrix = np.zeros((users, items), dtype=np.float32)
    matrix[
        interactions["user_id"].to_numpy(dtype=int),
        interactions["item_id"].to_numpy(dtype=int),
    ] = interactions["rating"].to_numpy(dtype=np.float32)
    return matrix


def _write_base_topk(scores_path: Path, eligible_path: Path, output_path: Path, k: int) -> int:
    scores = pd.read_parquet(scores_path).to_numpy(dtype=np.float64)
    eligible = pd.read_parquet(eligible_path).to_numpy(dtype=bool)
    if scores.shape != eligible.shape:
        raise ValueError("scores and eligible_items must have the same shape")
    if not eligible.any(axis=1).all():
        raise ValueError("each user must have at least one eligible item")
    eligible_counts = eligible.sum(axis=1)
    effective_k = min(k, scores.shape[1], int(eligible_counts.min()))
    if effective_k < 1:
        raise ValueError("each user must have at least one eligible item")
    masked = np.where(eligible, scores, -np.inf)
    ranking = np.argsort(-masked, axis=1)[:, :effective_k]
    pd.DataFrame(ranking).to_parquet(output_path, index=False)
    return effective_k


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _yelp_metro_groups(path: Path, original_items: Sequence[object]) -> pd.DataFrame:
    """Align the Yelp business-region mapping with this run's item IDs."""
    frame = pd.read_csv(path, dtype={"business_id": str})
    required = {"business_id", "region", "region_code", "match_status"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    if frame["business_id"].isna().any() or frame["business_id"].duplicated().any():
        raise ValueError(f"{path} must contain unique, nonmissing business IDs")
    items = pd.DataFrame({"item_id": range(len(original_items)), "business_id": list(original_items)})
    items["business_id"] = items["business_id"].astype(str)
    groups = items.merge(frame[list(required)], on="business_id", how="left", validate="one_to_one", sort=False)
    if groups[["region", "region_code", "match_status"]].isna().any(axis=None):
        missing_items = groups.loc[groups["region"].isna(), "business_id"].tolist()
        raise ValueError(f"{path} has missing region data for retained items: {missing_items[:5]}")
    codes = pd.to_numeric(groups["region_code"], errors="coerce")
    if codes.isna().any() or not codes.eq(np.floor(codes)).all() or not codes.between(-1, 10).all():
        raise ValueError(f"{path} must contain integer region codes from -1 to 10")
    groups["region_code"] = codes.astype(int)
    return groups.rename(columns={"region_code": "item_group", "region": "group_label"})[
        ["item_id", "item_group", "group_label", "business_id", "match_status"]
    ]


def _activity_groups(train: pd.DataFrame, users: int) -> pd.DataFrame:
    counts = np.bincount(train["user_id"].to_numpy(dtype=int), minlength=users)
    if np.all(counts == counts[0]):
        order = np.arange(users)
        active = set(order[: math.ceil(users / 2)])
    else:
        median = float(np.median(counts))
        active = {index for index, value in enumerate(counts) if value >= median}
    groups = [1 if user_id in active else 0 for user_id in range(users)]
    labels = ["active" if group == 1 else "inactive" for group in groups]
    return pd.DataFrame(
        {
            "user_id": range(users),
            "user_group": groups,
            "group_label": labels,
            "train_interactions": counts,
        }
    )


def _top_fraction_groups(
    values: np.ndarray,
    *,
    id_column: str,
    group_column: str,
    value_column: str,
    top_fraction: float,
) -> pd.DataFrame:
    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be in (0, 1]")
    count = values.shape[0]
    top_count = max(1, math.ceil(count * top_fraction))
    order = np.lexsort((np.arange(count), -values))
    short_head = set(order[:top_count].tolist())
    groups = [0 if index in short_head else 1 for index in range(count)]
    labels = ["short_head" if group == 0 else "long_tail" for group in groups]
    return pd.DataFrame(
        {
            id_column: range(count),
            group_column: groups,
            "group_label": labels,
            value_column: values,
        }
    )


def _quantile_groups(
    values: np.ndarray,
    *,
    id_column: str,
    group_column: str,
    value_column: str,
    quantiles: Sequence[float],
) -> pd.DataFrame:
    if not quantiles:
        raise ValueError("quantiles must not be empty")
    thresholds = np.asarray(sorted(quantiles), dtype=np.float64)
    if np.any(thresholds <= 0) or np.any(thresholds >= 1):
        raise ValueError("quantiles must be between 0 and 1")
    count = values.shape[0]
    if np.all(values == values[0]):
        order = np.arange(count)
        groups = np.zeros(count, dtype=int)
        bin_count = thresholds.size + 1
        for rank, index in enumerate(order):
            groups[index] = min(bin_count - 1, int(rank * bin_count / count))
    else:
        cutoffs = np.quantile(values, thresholds)
        groups = np.searchsorted(cutoffs, values, side="right")
    labels = [f"merit_bin_{group}" for group in groups]
    return pd.DataFrame(
        {
            id_column: range(count),
            group_column: groups.astype(int),
            "group_label": labels,
            value_column: values,
        }
    )


def _encode_labels(
    ids: Iterable[int],
    labels: Sequence[str],
    *,
    id_column: str,
    group_column: str,
) -> pd.DataFrame:
    distinct = sorted(set(labels))
    encoding = {label: index for index, label in enumerate(distinct)}
    return pd.DataFrame(
        {
            id_column: list(ids),
            group_column: [encoding[label] for label in labels],
            "group_label": list(labels),
        }
    )


def _age_label(value: object) -> str | None:
    if pd.isna(value):
        return None
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if not pd.isna(numeric):
        age = float(numeric)
        bins = [18, 25, 35, 45, 50, 56]
        labels = ["under_18", "18_24", "25_34", "35_44", "45_49", "50_55", "56_plus"]
        for threshold, label in zip(bins, labels, strict=False):
            if age < threshold:
                return label
        return labels[-1]
    return str(value)


def _mapped_user_attributes(
    attributes: pd.DataFrame,
    original_users: Sequence[object],
) -> pd.DataFrame:
    if attributes.empty:
        return pd.DataFrame({"user_id": range(len(original_users))})
    if attributes["user_id"].duplicated().any():
        attributes = attributes.drop_duplicates(subset=["user_id"], keep="first")
    exact = attributes.set_index("user_id", drop=False)
    string_index = attributes.assign(_key=attributes["user_id"].astype(str)).set_index(
        "_key",
        drop=False,
    )

    rows: list[dict[str, object]] = []
    for mapped_id, original_user in enumerate(original_users):
        if original_user in exact.index:
            row = exact.loc[original_user]
        elif str(original_user) in string_index.index:
            row = string_index.loc[str(original_user)]
        else:
            rows.append({"user_id": mapped_id})
            continue
        item = {"user_id": mapped_id}
        if "gender" in row.index and not pd.isna(row["gender"]):
            item["gender"] = str(row["gender"])
        if "age" in row.index and not pd.isna(row["age"]):
            item["age_bin"] = _age_label(row["age"])
        rows.append(item)
    return pd.DataFrame(rows)


def _demographic_groups(
    mapped_attributes: pd.DataFrame,
    *,
    mode: str,
) -> pd.DataFrame:
    if mode == "gender":
        if "gender" not in mapped_attributes.columns:
            raise ValueError("TSFD user gender groups require a real gender column")
        complete = mapped_attributes.dropna(subset=["gender"]).copy()
        if len(complete) != len(mapped_attributes):
            raise ValueError("TSFD user gender groups require gender for every retained user")
        labels = complete["gender"].astype(str).tolist()
    elif mode == "gender_or_age":
        has_gender = "gender" in mapped_attributes.columns
        has_age = "age_bin" in mapped_attributes.columns
        if not has_gender and not has_age:
            raise ValueError(
                "MultiFR demographic groups require real gender or age attributes"
            )
        labels = []
        for row in mapped_attributes.itertuples(index=False):
            parts: list[str] = []
            gender = getattr(row, "gender", None)
            age_bin = getattr(row, "age_bin", None)
            if has_gender and not pd.isna(gender):
                parts.append(f"gender={gender}")
            if has_age and not pd.isna(age_bin):
                parts.append(f"age={age_bin}")
            if not parts:
                raise ValueError(
                    "MultiFR demographic groups require gender or age for every retained user"
                )
            labels.append("|".join(parts))
    else:
        raise ValueError(f"unsupported demographic grouping mode: {mode}")
    return _encode_labels(
        mapped_attributes["user_id"].to_numpy(dtype=int).tolist(),
        labels,
        id_column="user_id",
        group_column="user_group",
    )


def _copy_method_inputs(
    method_root: Path,
    *,
    user_groups: Path | None = None,
    item_groups: Path | None = None,
    provider_items: Path | None = None,
) -> None:
    method_root.mkdir(parents=True, exist_ok=True)
    if user_groups is not None:
        shutil.copyfile(user_groups, method_root / "user_group_mapping.csv")
    if item_groups is not None:
        shutil.copyfile(item_groups, method_root / "item_group_mapping.csv")
    if provider_items is not None:
        shutil.copyfile(provider_items, method_root / "provider_item_mapping.csv")


def _write_run_groups(
    paths: ExperimentPaths,
    *,
    dataset: str,
    data_root: Path,
    run_dir: Path,
    attributes: pd.DataFrame,
    item_popularity_top_fraction: float,
    merit_quantiles: Sequence[float],
    allow_missing_demographics: bool,
    yelp_partition: str | None = None,
    movielens_director_gender: bool = False,
    movielens_director_groups: pd.DataFrame | None = None,
    amazon_brands: Mapping[str, str] | None = None,
) -> tuple[Path, Path, dict[str, object]]:
    train = pd.read_parquet(paths.train)
    train_matrix = _matrix_from_interactions(
        train,
        users=int(train["user_id"].max()) + 1,
        items=int(train["item_id"].max()) + 1,
    )
    users, items = train_matrix.shape
    mappings = json.loads(paths.mappings.read_text(encoding="utf-8"))
    mapped_attributes = _mapped_user_attributes(attributes, mappings["users"])
    item_exposure = (train_matrix > 0).sum(axis=0).astype(np.float64)
    item_merit = np.divide(
        train_matrix.sum(axis=0),
        np.maximum((train_matrix > 0).sum(axis=0), 1),
    )

    groups_dir = run_dir / "groups"
    method_dir = run_dir / "method_inputs"
    _write_csv(
        pd.DataFrame({"item_id": range(items), "merit": item_merit}),
        run_dir / "item_merit.csv",
    )
    if dataset == "yelp" and yelp_partition == "standard":
        provider_frame, semantics = _provider_item_mapping(
            dataset=dataset, data_root=data_root, original_items=mappings["items"],
        )
        _write_csv(provider_frame, run_dir / "provider_item_mapping.csv")
        return groups_dir, method_dir, {
            "item_provider_semantics": semantics, "yelp_partition": "standard",
        }

    metro_metadata: dict[str, object] = {}
    metro_source = data_root / "mappings" / "yelp_metro" / "yelp_business_metro.csv"
    if dataset == "yelp":
        if metro_source.is_file():
            metro_groups = _yelp_metro_groups(metro_source, mappings["items"])
            metro_path = _write_csv(metro_groups, groups_dir / "yelp_item_metro.csv")
            metro_metadata = {
                "yelp_item_metro_source": str(metro_source),
                "yelp_item_metro_mapping": str(metro_path),
                "yelp_item_metro_unmatched": int((metro_groups["item_group"] == -1).sum()),
            }
        else:
            if (groups_dir / "yelp_item_metro.csv").exists():
                raise FileNotFoundError(
                    f"{metro_source} is missing but an earlier geographic mapping exists "
                    f"in {groups_dir}; restore the source or use a new run directory"
                )
            metro_metadata = {"yelp_item_metro_source": None, "yelp_item_metro_mapping": None}
            print(f"Yelp metropolitan mapping not found at {metro_source}; skipping geographic groups")
    if dataset == "yelp":
        cpfair_user = _write_csv(_activity_groups(train, users), groups_dir / "cpfair_user_activity.csv")
        provider_frame, provider_semantics = _provider_item_mapping(
            dataset=dataset, data_root=data_root, original_items=mappings["items"]
        )
        provider_items = _write_csv(provider_frame, run_dir / "provider_item_mapping.csv")
        if yelp_partition == "cpfair-territorial":
            source = data_root / "mappings/yelp_census/yelp_business_poverty.csv"
            census = pd.read_csv(source, dtype={"business_id": str, "tract_id": str})
            frame = pd.DataFrame({"item_id": range(items), "business_id": mappings["items"]}).merge(
                census, on="business_id", how="left", validate="one_to_one", sort=False
            )
            if frame.item_group.isna().any() or set(frame.item_group) != {-1, 1}:
                raise ValueError("Yelp territorial run must contain both known poverty groups")
            frame["item_group"] = frame.item_group.astype(int)
            frame["group_label"] = frame.item_group.map({1: "high_poverty", -1: "lower_poverty"})
            item_path = _write_csv(frame, groups_dir / "yelp_item_poverty.csv")
            _copy_method_inputs(method_dir / "CPFair", user_groups=cpfair_user, item_groups=item_path, provider_items=provider_items)
            group_source = str(source)
        elif yelp_partition == "tfld-metro":
            if not metro_source.is_file():
                raise FileNotFoundError(f"required metropolitan mapping not found: {metro_source}")
            missing_groups = set(range(11)) - set(metro_groups.item_group)
            if metro_groups.item_group.isna().any() or missing_groups or set(metro_groups.item_group) - set(range(11)):
                raise ValueError(f"Yelp metro run must retain all 11 metropolitan groups; missing codes: {sorted(missing_groups)}")
            item_path = groups_dir / "yelp_item_metro.csv"
            _copy_method_inputs(method_dir / "TFLD-Groups", user_groups=cpfair_user, item_groups=item_path, provider_items=provider_items)
            group_source = str(metro_source)
        else:
            raise ValueError("Yelp requires a valid partition")
        return groups_dir, method_dir, {
            **metro_metadata, "item_provider_semantics": provider_semantics,
            "yelp_partition": yelp_partition, "yelp_group_source": group_source,
            "yelp_item_group_mapping": str(item_path),
            "yelp_item_group_counts": {str(k): int(v) for k, v in pd.read_csv(item_path).item_group.value_counts().items()},
            "cpfair_user_groups": "training_user_activity_top_20_percent",
        }

    user_groups = _activity_groups(train, users)
    cpfair_user_rule = "training_user_activity_median_split"
    if amazon_brands is not None:
        user_groups = _top_fraction_groups(
            np.count_nonzero(train_matrix, axis=1),
            id_column="user_id",
            group_column="user_group",
            value_column="train_interactions",
            top_fraction=0.2,
        )
        user_groups["group_label"] = user_groups.user_group.map({0: "active", 1: "inactive"})
        user_groups["user_group"] = user_groups.user_group.map({0: -1, 1: 1})
        cpfair_user_rule = "training_user_activity_top_20_percent"
    cpfair_user = _write_csv(user_groups, groups_dir / "cpfair_user_activity.csv")
    cpfair_item_rule = "training_item_exposure_short_head_long_tail"
    if amazon_brands is None:
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
    if dataset == "movielens" and movielens_director_gender:
        if movielens_director_groups is None:
            raise ValueError("MovieLens director groups are required for this partition")
        director_groups = movielens_director_groups.set_index("movie_id")["item_group"]
        original_items = [str(item) for item in mappings["items"]]
        aligned = director_groups.reindex(original_items)
        if aligned.isna().any() or set(aligned.astype(int)) != {-1, 1}:
            raise ValueError("MovieLens director groups must cover all retained items and contain both groups")
        cpfair_item = _write_csv(pd.DataFrame({
            "item_id": range(items), "item_group": aligned.to_numpy(dtype=int),
        }), groups_dir / "cpfair_item_director_gender.csv")
        cpfair_item_rule = "director_gender"
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
    provider_frame, provider_semantics = _provider_item_mapping(
        dataset=dataset,
        data_root=data_root,
        original_items=mappings["items"],
        amazon_brands=amazon_brands,
    )
    provider_items = _write_csv(
        provider_frame,
        run_dir / "provider_item_mapping.csv",
    )
    provider_ids = provider_frame["provider_id"].to_numpy(dtype=int)
    provider_exposure = np.zeros(int(provider_ids.max()) + 1, dtype=np.float64)
    np.add.at(provider_exposure, provider_ids, item_exposure)
    provider_groups = _top_fraction_groups(
        provider_exposure,
        id_column="provider_id",
        group_column="provider_group",
        value_column="train_exposure",
        top_fraction=item_popularity_top_fraction,
    )
    brand_metadata: dict[str, object] = {}
    if amazon_brands is not None:
        if set(provider_groups.provider_group) != {0, 1}:
            raise ValueError("Amazon brand partition must contain both short-head and long-tail brands")
        provider_groups = provider_groups.merge(
            provider_frame[["provider_id", "brand"]].drop_duplicates(),
            on="provider_id",
            validate="one_to_one",
        )
        item_groups = provider_frame[["item_id", "provider_id", "asin", "brand"]].merge(
            provider_groups[["provider_id", "provider_group", "group_label", "train_exposure"]],
            on="provider_id",
            validate="many_to_one",
        )
        item_groups["item_group"] = item_groups.pop("provider_group").map({0: -1, 1: 1})
        item_groups = item_groups.rename(columns={"train_exposure": "brand_train_interactions"})
        cpfair_item = _write_csv(item_groups, groups_dir / "cpfair_item_brand_popularity.csv")
        shutil.copyfile(cpfair_item, paths.item_groups)
        cpfair_item_rule = "training_brand_interactions_short_head_long_tail"
        brand_metadata = {
            "amazon_brand_group_counts": {
                str(label): int(count)
                for label, count in provider_groups.group_label.value_counts().items()
            },
            "amazon_brand_item_group_counts": {
                str(label): int(count)
                for label, count in item_groups.group_label.value_counts().items()
            },
        }
    provider_group_path = _write_csv(
        provider_groups,
        groups_dir / "provider_group_mapping.csv",
    )

    missing_demographics: list[str] = []
    multifr_user: Path | None = None
    tsfd_user: Path | None = None
    try:
        multifr_user = _write_csv(
            _demographic_groups(mapped_attributes, mode="gender_or_age"),
            groups_dir / "multifr_user_demographic.csv",
        )
    except ValueError as error:
        if not allow_missing_demographics:
            raise
        missing_demographics.append(str(error))
    try:
        tsfd_user = _write_csv(
            _demographic_groups(mapped_attributes, mode="gender"),
            groups_dir / "tsfd_user_gender.csv",
        )
    except ValueError as error:
        if not allow_missing_demographics:
            raise
        missing_demographics.append(str(error))

    _copy_method_inputs(
        method_dir / "CPFair",
        user_groups=cpfair_user,
        item_groups=cpfair_item,
        provider_items=provider_items,
    )
    if multifr_user is not None:
        _copy_method_inputs(
            method_dir / "MultiFR",
            user_groups=multifr_user,
            item_groups=multifr_item,
            provider_items=provider_items,
        )
    if tsfd_user is not None:
        _copy_method_inputs(
            method_dir / "TSFD",
            user_groups=tsfd_user,
            item_groups=tsfd_item,
            provider_items=provider_items,
        )
    _copy_method_inputs(
        method_dir / "provider_defaults",
        provider_items=provider_items,
    )
    shutil.copyfile(provider_group_path, method_dir / "provider_defaults" / "provider_group_mapping.csv")

    metadata = {
        **metro_metadata,
        **brand_metadata,
        "item_provider_semantics": provider_semantics,
        "cpfair_user_groups": cpfair_user_rule,
        "cpfair_item_groups": cpfair_item_rule,
        "multifr_user_groups": "real_gender_age_when_available",
        "multifr_item_groups": "training_item_exposure_short_head_long_tail",
        "tsfd_user_groups": "real_gender",
        "tsfd_item_groups": "training_item_merit_quantile_bins",
        "item_popularity_top_fraction": item_popularity_top_fraction,
        "merit_quantiles": list(merit_quantiles),
        "missing_demographic_groups": missing_demographics,
    }
    return groups_dir, method_dir, metadata


def _movielens_director_groups(data_root: Path) -> pd.DataFrame:
    """Classify original movie IDs without guessing unknown director genders."""
    path = data_root / "raw_datasets/explicit_feedback/Movie Lens/auxiliary_data/director_gender.csv"
    if not path.is_file():
        raise FileNotFoundError(f"MovieLens director metadata not found: {path}")
    frame = pd.read_csv(path, dtype={"movie_id": str, "director_id": str, "gender_id": str})
    required = {"movie_id", "director_id", "gender_id"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if frame.empty or frame.movie_id.isna().any():
        raise ValueError(f"{path} must contain nonempty movie IDs")

    def classify(rows: pd.DataFrame) -> int | None:
        values = set(rows.gender_id.dropna())
        if values.intersection({"Q6581072", "Q1052281"}):
            return 1
        if rows.director_id.notna().all() and rows.gender_id.notna().all() and values == {"Q6581097"}:
            return -1
        return None

    classified = pd.DataFrame(
        [(movie_id, classify(rows)) for movie_id, rows in frame.groupby("movie_id", sort=False)],
        columns=["movie_id", "item_group"],
    )
    movies_path = data_root / "raw_datasets/explicit_feedback/Movie Lens/movies.dat"
    if not movies_path.is_file():
        raise FileNotFoundError(f"MovieLens movie metadata not found: {movies_path}")
    movies = pd.read_csv(movies_path, sep="::", engine="python", encoding="latin-1",
                         header=None, names=["movie_id", "title", "genres"], dtype={"movie_id": str})
    if movies.empty or movies.movie_id.isna().any() or movies.movie_id.duplicated().any():
        raise ValueError(f"{movies_path} must contain unique, nonempty movie IDs")
    result = movies[["movie_id"]].merge(classified, on="movie_id", how="left", validate="one_to_one")
    result["item_group"] = result["item_group"].astype("Int64")
    return result


def _ensure_yelp_mapping(data_root: Path, partition: str) -> Path:
    """Build the selected dataset-wide Yelp mapping once, if it is missing."""
    if partition not in {"cpfair-territorial", "tfld-metro"}:
        raise ValueError(f"unsupported Yelp partition: {partition}")
    census = partition == "cpfair-territorial"
    target = data_root / (
        "mappings/yelp_census/yelp_business_poverty.csv" if census
        else "mappings/yelp_metro/yelp_business_metro.csv"
    )
    if target.is_file():
        return target
    yelp_root = data_root / "raw_datasets/explicit_feedback/Yelp"
    business = yelp_root / "yelp_academic_dataset_business.json"
    aux = yelp_root / "auxiliary_data"
    if census:
        tracts = aux / "census_acs_2019/cb_2019_us_tract_500k.zip"
        poverty = aux / "census_acs_2019/acsdt5y2019-b17001.dat"
        required = (business, tracts, poverty)
    else:
        us = aux / "boundaries/us_cbsa_2020/cb_2020_us_cbsa_500k.shp"
        canada = aux / "boundaries/edmonton_cma_2021.geojson"
        required = (business, us, canada)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Cannot generate Yelp {partition} mapping; missing input files: {missing}. "
            "See data/mappings/README.md for the required auxiliary data."
        )
    try:
        if census:
            from .yelp_census_mapping import build_mapping

            frame = build_mapping(business, tracts, poverty)
            metadata = {
                "businesses": str(business), "tracts": str(tracts), "poverty": str(poverty),
                "tract_source": "https://www2.census.gov/geo/tiger/GENZ2019/shp/cb_2019_us_tract_500k.zip",
                "poverty_source": "https://www2.census.gov/programs-surveys/acs/summary_file/2019/prototype/5YRData/acsdt5y2019-b17001.dat",
                "poverty_measure": "ACS 2015-2019 B17001_E002 / B17001_E001 * 100",
                "threshold_percent": 20,
            }
            metadata_path = target.with_suffix(".metadata.json")
        else:
            from .yelp_metro_mapping import REGION_CODES, load_boundaries, map_businesses

            frame = map_businesses(business, load_boundaries(us, canada))
            metadata = {"businesses": str(business), "us_boundaries": str(us),
                        "canada_boundaries": str(canada), "region_codes": REGION_CODES,
                        "unknown_code": -1}
            metadata_path = target.parent / "yelp_metro_metadata.json"
    except ModuleNotFoundError as error:
        if error.name == "geopandas":
            raise ValueError("Generating Yelp mappings requires geopandas; run uv sync --extra geographic") from error
        raise
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Generated Yelp {partition} mapping: {target}", flush=True)
    return target


def _write_movielens_gender_mapping(data_root: Path) -> Path:
    source = data_root / "raw_datasets" / "explicit_feedback" / "Movie Lens" / "users.dat"
    if not source.is_file():
        raise FileNotFoundError(f"MovieLens user metadata not found: {source}")
    users = pd.read_csv(
        source, sep="::", engine="python", header=None,
        names=["user_id", "gender", "age", "occupation", "zip_code"],
    )[["user_id", "gender"]]
    if users.empty or users["user_id"].isna().any() or users["user_id"].duplicated().any():
        raise ValueError(f"{source} must contain unique, nonempty user IDs")
    if not users["gender"].isin(["F", "M"]).all():
        raise ValueError(f"{source} must contain gender F or M for every user")
    users["user_group"] = users["gender"].map({"F": 1, "M": -1})
    destination = data_root / "mappings" / "movielens_user_gender_mapping.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    users.to_csv(destination, index=False)
    return destination


def _write_shared_run_zero_artifacts(
    *,
    dataset: str,
    run: int,
    run_dir: Path,
    data_root: Path,
) -> None:
    if run != 0:
        return
    evaluation_dir = data_root / "evaluation"
    mapping_dir = data_root / "mappings"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    mapping_dir.mkdir(parents=True, exist_ok=True)

    if dataset == "movielens":
        _write_movielens_gender_mapping(data_root)

    relevance = pd.read_parquet(run_dir / "heldout_relevance.parquet")
    relevance.to_parquet(evaluation_dir / f"{dataset}_relevance.parquet", index=False)
    merit = relevance.mean(axis=0).reset_index()
    merit.columns = ["item_id", "merit"]
    merit.to_csv(evaluation_dir / f"{dataset}_merit.csv", index=False)

    for name, source in (
        ("user_group_mapping", run_dir / "method_inputs" / "CPFair" / "user_group_mapping.csv"),
        ("item_group_mapping", run_dir / "method_inputs" / "CPFair" / "item_group_mapping.csv"),
        ("provider_item_mapping", run_dir / "provider_item_mapping.csv"),
        ("provider_group_mapping", run_dir / "groups" / "provider_group_mapping.csv"),
    ):
        shutil.copyfile(source, mapping_dir / f"{dataset}_{name}.csv")


def prepare_dataset_run(
    dataset: str,
    *,
    run: int,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    seed: int = 42,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    factors: int = 32,
    epochs: int = 200,
    patience: int = 10,
    batch_size: int = 1024,
    learning_rate: float = 0.01,
    regularization: float = 1e-4,
    recommendation_list_length: int = 10,
    item_popularity_top_fraction: float = 0.2,
    merit_quantiles: Sequence[float] = (0.5,),
    allow_missing_demographics: bool = False,
    write_shared_run_zero: bool = True,
    max_users: int | None = None,
    max_items: int | None = None,
    max_dense_cells: int | None = DEFAULT_MAX_DENSE_CELLS,
    yelp_partition: str | None = None,
    movielens_director_gender: bool = False,
    amazon_brand: bool = False,
    split_seed: int | None = None,
    split_assignments: pd.DataFrame | None = None,
    raw: _RawDataset | None = None,
) -> PreparedRun:
    """Prepare one run of one explicit dataset for method experiments."""

    if dataset not in DATASETS:
        raise ValueError(f"unsupported dataset {dataset!r}; choose from {sorted(DATASETS)}")
    if run < 0 or seed < 0:
        raise ValueError("seed and run must be non-negative")
    if dataset == "yelp" and yelp_partition not in {"standard", "cpfair-territorial", "tfld-metro"}:
        raise ValueError("Yelp requires --yelp-partition standard, cpfair-territorial, or tfld-metro")
    _validate_optional_positive(max_users, "max_users")
    _validate_optional_positive(max_items, "max_items")
    _validate_optional_positive(max_dense_cells, "max_dense_cells")

    data_root_path = Path(data_root)
    output_root_path = Path(output_root)
    amazon_brands = (
        _amazon_brands_by_asin(data_root_path / AMAZON_BRAND_METADATA)
        if dataset == "amazon" and amazon_brand
        else None
    )
    if dataset == "yelp" and yelp_partition != "standard":
        _ensure_yelp_mapping(data_root_path, yelp_partition)
    effective_seed = seed + run
    raw_path = data_root_path / DATASETS[dataset]
    print(f"Preparing {dataset} run {run:02d} seed {effective_seed}: reading {raw_path}")
    raw = _read_raw_dataset(raw_path) if raw is None else raw
    attributes = _merge_user_attributes(
        raw.user_attributes,
        _read_user_sidecar(data_root_path, dataset),
    )
    interactions = raw.interactions
    dropped_interactions_without_user_attributes = 0
    preprocessing_metadata: dict[str, object] = {
        "preprocessing_dataset": dataset,
        "preprocessing_original_interactions": len(interactions),
        "preprocessing_original_users": int(interactions["user_id"].nunique()),
        "preprocessing_original_items": int(interactions["item_id"].nunique()),
    }
    if amazon_brands is not None:
        before = interactions
        interactions = interactions[interactions.item_id.astype(str).isin(amazon_brands)].copy()
        if interactions.empty:
            raise ValueError("Amazon brand partition contains no reviews with an identified brand")
        preprocessing_metadata.update({
            "amazon_partition": "brand",
            "amazon_brand_metadata_source": str(data_root_path / AMAZON_BRAND_METADATA),
            "amazon_brand_dropped_interactions": len(before) - len(interactions),
            "amazon_brand_dropped_users": int(before.user_id.nunique() - interactions.user_id.nunique()),
            "amazon_brand_dropped_items": int(before.item_id.nunique() - interactions.item_id.nunique()),
        })
    director_groups: pd.DataFrame | None = None
    if dataset == "movielens" and movielens_director_gender:
        director_groups = _movielens_director_groups(data_root_path)
        mapping_path = data_root_path / "mappings" / "movielens_item_director_gender_mapping.csv"
        mapping_bytes = director_groups.to_csv(index=False, lineterminator="\n").encode("utf-8")
        if write_shared_run_zero:
            mapping_path.parent.mkdir(parents=True, exist_ok=True)
            mapping_path.write_bytes(mapping_bytes)
        auxiliary_path = data_root_path / "raw_datasets/explicit_feedback/Movie Lens/auxiliary_data/director_gender.csv"
        known = set(director_groups.loc[director_groups.item_group.notna(), "movie_id"])
        before = interactions
        interactions = interactions[interactions.item_id.astype(str).isin(known)].copy()
        preprocessing_metadata.update({
            "movielens_partition": "director-gender",
            "movielens_director_mapping_source": str(mapping_path) if write_shared_run_zero else "in_memory",
            "movielens_director_mapping_sha256": hashlib.sha256(mapping_bytes).hexdigest(),
            "movielens_director_auxiliary_source": str(auxiliary_path),
            "movielens_director_auxiliary_sha256": hashlib.sha256(auxiliary_path.read_bytes()).hexdigest(),
            "movielens_director_group_rule": "any_female_or_trans_woman_protected; all_male_reference; otherwise_unknown",
            "movielens_director_mapping_movies": len(director_groups),
            "movielens_director_mapping_unknown_movies": int(director_groups.item_group.isna().sum()),
            "movielens_director_dropped_interactions": len(before) - len(interactions),
            "movielens_director_dropped_users": int(before.user_id.nunique() - interactions.user_id.nunique()),
            "movielens_director_dropped_items": int(before.item_id.nunique() - interactions.item_id.nunique()),
        })
        if interactions.empty:
            raise ValueError("MovieLens director-gender partition contains no interactions")
    if dataset == "yelp" and yelp_partition != "standard":
        source = data_root_path / ("mappings/yelp_census/yelp_business_poverty.csv" if yelp_partition == "cpfair-territorial" else "mappings/yelp_metro/yelp_business_metro.csv")
        groups = pd.read_csv(source, dtype={"business_id": str})
        if groups.business_id.duplicated().any():
            raise ValueError(f"{source} contains duplicate business IDs")
        selected_groups = groups[groups.match_status.eq("matched")]
        if yelp_partition == "cpfair-territorial":
            selected_groups = selected_groups[selected_groups.item_group.isin([-1, 1])]
        else:
            selected_groups = selected_groups[selected_groups.region_code.isin(range(11))]
        before = interactions
        interactions = interactions[interactions.item_id.astype(str).isin(set(selected_groups.business_id))].copy()
        preprocessing_metadata.update({
            "yelp_partition": yelp_partition, "yelp_partition_source": str(source),
            "yelp_partition_dropped_interactions": len(before) - len(interactions),
            "yelp_partition_dropped_users": int(before.user_id.nunique() - interactions.user_id.nunique()),
            "yelp_partition_dropped_items": int(before.item_id.nunique() - interactions.item_id.nunique()),
        })
        if interactions.empty:
            raise ValueError("Yelp partition contains no interactions")
    if dataset == "ambar":
        interactions, dropped_interactions_without_user_attributes = (
            _filter_interactions_to_attributed_users(interactions, attributes)
        )
        effective_max_users = (
            DEFAULT_AMBAR_MAX_USERS if max_users is None else max_users
        )
        effective_max_items = (
            DEFAULT_AMBAR_MAX_ITEMS if max_items is None else max_items
        )
    else:
        effective_max_users = max_users
        effective_max_items = max_items

    if effective_max_users is not None or effective_max_items is not None:
        preprocessing = _limit_interactions_for_dense_pipeline(
            interactions,
            dataset=dataset,
            max_users=effective_max_users,
            max_items=effective_max_items,
        )
        interactions = preprocessing.interactions
        preprocessing_metadata.update(preprocessing.metadata)
        dense_cells = _validate_dense_size(
            interactions,
            max_dense_cells=max_dense_cells,
        )
    else:
        dense_cells = int(interactions["user_id"].nunique()) * int(
            interactions["item_id"].nunique()
        )
        preprocessing_metadata.update(
            {
                "preprocessing_max_users": None,
                "preprocessing_max_items": None,
                "preprocessing_retained_interactions": len(interactions),
                "preprocessing_retained_users": int(interactions["user_id"].nunique()),
                "preprocessing_retained_items": int(interactions["item_id"].nunique()),
                "preprocessing_dropped_interactions": 0,
                "preprocessing_dense_cells": dense_cells,
            }
        )
    preprocessing_metadata["preprocessing_dense_cells"] = dense_cells

    if amazon_brands is not None:
        run_dir = _run_seed_directory(output_root_path / "amazon", "brand", run, effective_seed)
    elif dataset == "movielens" and movielens_director_gender:
        run_dir = _run_seed_directory(output_root_path / "movielens", "director-gender", run, effective_seed)
    else:
        run_dir = _run_seed_directory(output_root_path / "yelp" if dataset == "yelp" else output_root_path, yelp_partition if dataset == "yelp" else dataset, run, effective_seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    standardized_path = run_dir / "standardized_interactions.tsv"
    _write_standard_interactions(interactions, standardized_path)
    print(
        f"Preparing {dataset} run {run:02d}: retained "
        f"{len(interactions):,} interactions, "
        f"{interactions['user_id'].nunique():,} users, "
        f"{interactions['item_id'].nunique():,} items; fitting MF"
    )

    core_paths = prepare_experiment(
        standardized_path,
        output_dir=run_dir,
        seed=seed,
        run=run,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        factors=factors,
        epochs=epochs,
        patience=patience,
        batch_size=batch_size,
        learning_rate=learning_rate,
        regularization=regularization,
        generate_item_groups=dataset != "yelp" and amazon_brands is None,
        split_seed=split_seed,
        split_assignments=split_assignments,
    )

    train = pd.read_parquet(core_paths.train)
    validation = pd.read_parquet(core_paths.validation)
    test = pd.read_parquet(core_paths.test)
    heldout_relevance = pd.read_parquet(core_paths.test_relevance)
    mappings = json.loads(core_paths.mappings.read_text(encoding="utf-8"))
    train_matrix = _matrix_from_interactions(
        train,
        users=len(mappings["users"]),
        items=len(mappings["items"]),
    )
    _validate_same_user_partition(
        train=train,
        validation=validation,
        test=test,
        train_matrix=train_matrix,
        heldout_relevance=heldout_relevance.to_numpy(dtype=np.float32),
    )
    train_matrix_path = run_dir / "train_matrix.parquet"
    scores_path = run_dir / "scores.parquet"
    heldout_relevance_path = run_dir / "heldout_relevance.parquet"
    base_topk_path = run_dir / "base_topk.parquet"
    pd.DataFrame(train_matrix).to_parquet(train_matrix_path, index=False)
    pd.read_parquet(core_paths.predictions).to_parquet(scores_path, index=False)
    heldout_relevance.to_parquet(heldout_relevance_path, index=False)
    train.to_parquet(run_dir / "train_interactions.parquet", index=False)
    validation.to_parquet(run_dir / "validation_interactions.parquet", index=False)
    test.to_parquet(run_dir / "test_interactions.parquet", index=False)
    effective_k = _write_base_topk(
        scores_path,
        core_paths.eligible_items,
        base_topk_path,
        recommendation_list_length,
    )

    groups_dir, method_dir, group_metadata = _write_run_groups(
        core_paths,
        dataset=dataset,
        data_root=data_root_path,
        run_dir=run_dir,
        attributes=attributes,
        item_popularity_top_fraction=item_popularity_top_fraction,
        merit_quantiles=merit_quantiles,
        allow_missing_demographics=allow_missing_demographics,
        yelp_partition=yelp_partition,
        movielens_director_gender=movielens_director_gender,
        movielens_director_groups=director_groups,
        amazon_brands=amazon_brands,
    )

    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(core_paths.metadata.read_text(encoding="utf-8"))
    metadata.update(
        {
            "dataset": dataset,
            "output_root": str(output_root_path),
            "run_directory": str(run_dir),
            "base_seed": seed,
            "raw_source_path": str(raw_path),
            "standardized_interactions": str(standardized_path),
            "recommendation_list_length": recommendation_list_length,
            "effective_base_topk_length": effective_k,
            "heldout_relevance": "test_ratings_only",
            "user_partition_policy": "same_users_in_train_validation_test",
            "demographic_sidecar": str(data_root_path / "attributes" / f"{dataset}_users.csv"),
            "dropped_interactions_without_user_attributes": (
                dropped_interactions_without_user_attributes
            ),
            **preprocessing_metadata,
            **group_metadata,
        }
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if write_shared_run_zero and dataset == "movielens" and movielens_director_gender and run == 0:
        _write_movielens_gender_mapping(data_root_path)
    elif write_shared_run_zero and dataset != "yelp" and not (dataset == "movielens" and movielens_director_gender):
        _write_shared_run_zero_artifacts(
            dataset="amazon_brand" if amazon_brands is not None else dataset,
            run=run,
            run_dir=run_dir,
            data_root=data_root_path,
        )

    print(f"Prepared {dataset} run {run:02d}: {run_dir}")
    return PreparedRun(
        dataset=dataset,
        run=run,
        directory=run_dir,
        core_paths=core_paths,
        train_matrix=train_matrix_path,
        scores=scores_path,
        heldout_relevance=heldout_relevance_path,
        base_topk=base_topk_path,
        groups_directory=groups_dir,
        method_inputs_directory=method_dir,
        metadata=metadata_path,
    )


def prepare_explicit_method_inputs(
    *,
    datasets: Sequence[str] | None = None,
    runs: int = DEFAULT_RUNS,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    seed: int = 42,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    factors: int = 32,
    epochs: int = 200,
    patience: int = 10,
    batch_size: int = 1024,
    learning_rate: float = 0.01,
    regularization: float = 1e-4,
    recommendation_list_length: int = 10,
    item_popularity_top_fraction: float = 0.2,
    merit_quantiles: Sequence[float] = (0.5,),
    allow_missing_demographics: bool = False,
    write_shared_run_zero: bool = True,
    max_users: int | None = None,
    max_items: int | None = None,
    max_dense_cells: int | None = DEFAULT_MAX_DENSE_CELLS,
    yelp_partition: str | None = None,
    movielens_director_gender: bool = False,
    amazon_brand: bool = False,
) -> list[PreparedRun]:
    """Prepare all requested explicit datasets and runs."""

    if runs <= 0:
        raise ValueError("runs must be positive")
    selected = list(DATASETS if datasets is None else datasets)
    if "yelp" in selected and yelp_partition not in {"standard", "cpfair-territorial", "tfld-metro"}:
        raise ValueError("Yelp requires --yelp-partition standard, cpfair-territorial, or tfld-metro")
    prepared: list[PreparedRun] = []
    for dataset in selected:
        for run in range(runs):
            prepared.append(
                prepare_dataset_run(
                    dataset,
                    run=run,
                    data_root=data_root,
                    output_root=output_root,
                    seed=seed,
                    validation_ratio=validation_ratio,
                    test_ratio=test_ratio,
                    factors=factors,
                    epochs=epochs,
                    patience=patience,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    regularization=regularization,
                    recommendation_list_length=recommendation_list_length,
                    item_popularity_top_fraction=item_popularity_top_fraction,
                    merit_quantiles=merit_quantiles,
                    allow_missing_demographics=allow_missing_demographics,
                    write_shared_run_zero=write_shared_run_zero,
                    max_users=max_users,
                    max_items=max_items,
                    max_dense_cells=max_dense_cells,
                    yelp_partition=yelp_partition,
                    movielens_director_gender=movielens_director_gender,
                    amazon_brand=amazon_brand,
                )
            )
    return prepared


def build_parser() -> argparse.ArgumentParser:
    """Create the explicit dataset preprocessing command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS))
    parser.add_argument("--amazon-brand", action="store_true", help="Prepare a separate Amazon partition grouped by brand demand in training.")
    parser.add_argument("--yelp-partition", choices=["standard", "cpfair-territorial", "tfld-metro"], help="Required when preparing Yelp")
    parser.add_argument("--movielens-director-gender", action="store_true", help="Prepare a separate MovieLens partition grouped by director gender.")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--factors", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--regularization", type=float, default=1e-4)
    parser.add_argument("--recommendation-list-length", type=int, default=10)
    parser.add_argument("--item-popularity-top-fraction", type=float, default=0.2)
    parser.add_argument(
        "--max-users",
        type=int,
        help=(
            "Maximum users retained before dense matrix preparation. "
            f"Defaults to {DEFAULT_AMBAR_MAX_USERS} for AMBAR and unlimited otherwise."
        ),
    )
    parser.add_argument(
        "--max-items",
        type=int,
        help=(
            "Maximum items retained before dense matrix preparation. "
            f"Defaults to {DEFAULT_AMBAR_MAX_ITEMS} for AMBAR and unlimited otherwise."
        ),
    )
    parser.add_argument(
        "--max-dense-cells",
        type=int,
        default=DEFAULT_MAX_DENSE_CELLS,
        help=(
            "Fail before fitting when retained users x items exceeds this limit. "
            "Use a larger value only if the machine can handle the dense matrices."
        ),
    )
    parser.add_argument(
        "--merit-quantiles",
        type=float,
        nargs="+",
        default=[0.5],
    )
    parser.add_argument(
        "--allow-missing-demographics",
        action="store_true",
        help="Skip demographic group files that cannot be built from real attributes.",
    )
    parser.add_argument(
        "--no-shared-run-zero",
        action="store_true",
        help="Do not refresh data/evaluation and data/mappings from run 0.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run explicit dataset preprocessing from the command line."""

    args = build_parser().parse_args(argv)
    try:
        prepared = prepare_explicit_method_inputs(
            datasets=args.datasets,
            runs=args.runs,
            data_root=args.data_root,
            output_root=args.output_root,
            seed=args.seed,
            validation_ratio=args.validation_ratio,
            test_ratio=args.test_ratio,
            factors=args.factors,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            regularization=args.regularization,
            recommendation_list_length=args.recommendation_list_length,
            item_popularity_top_fraction=args.item_popularity_top_fraction,
            merit_quantiles=args.merit_quantiles,
            allow_missing_demographics=args.allow_missing_demographics,
            write_shared_run_zero=not args.no_shared_run_zero,
            max_users=args.max_users,
            max_items=args.max_items,
            max_dense_cells=args.max_dense_cells,
            yelp_partition=args.yelp_partition,
            movielens_director_gender=args.movielens_director_gender,
            amazon_brand=args.amazon_brand,
        )
    except (FileNotFoundError, OSError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    for item in prepared:
        print(item.directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
