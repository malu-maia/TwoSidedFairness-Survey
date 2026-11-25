"""Prepare and verify the shared interaction split for the sensitivity study."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import prepare_explicit_method_inputs as preparation


DATASETS = ("amazon", "ambar", "movielens", "yelp")
SPLIT_SEED = 20261124
FIT_SEEDS = (42, 43, 44)
GROUP_VARIANTS = {"amazon": "brand", "movielens": "director-gender", "yelp": "cpfair-territorial"}
GROUP_METHODS = frozenset({"CPFair", "TFLD-Groups"})


def variant_for(dataset: str, method: str) -> str:
    return GROUP_VARIANTS.get(dataset, "standard") if method in GROUP_METHODS else "standard"


def input_directory(root: Path, dataset: str, variant: str) -> Path:
    directory = root / dataset
    if variant != "standard" or dataset == "yelp":
        directory /= variant
    return directory / "run_00" / f"seed_{FIT_SEEDS[0]}"


def pair_assignments(directory: Path) -> pd.DataFrame:
    mappings = json.loads((directory / "mappings.json").read_text(encoding="utf-8"))
    splits = []
    for label, name in enumerate(("train", "validation", "test")):
        pairs = pd.read_parquet(directory / f"{name}_interactions.parquet", columns=["user_id", "item_id"])
        for column, key in (("user_id", "users"), ("item_id", "items")):
            pairs[column] = pd.Series(mappings[key]).astype(str).iloc[pairs[column]].to_numpy()
        splits.append(pairs.assign(split=label))
    assignments = pd.concat(splits, ignore_index=True)
    if assignments.duplicated(["user_id", "item_id"]).any():
        raise ValueError(f"Repeated original user–item pair in {directory}")
    return assignments


def verify_shared_split(first: pd.DataFrame, second: pd.DataFrame, dataset: str) -> None:
    shared = first.merge(second, on=["user_id", "item_id"], validate="one_to_one")
    if shared.empty or not shared["split_x"].eq(shared["split_y"]).all():
        raise ValueError(f"{dataset} requires matching split membership on shared pairs")


def prepare_dataset(root: Path, data_root: Path, dataset: str, max_k: int, threads: int) -> None:
    import torch

    torch.set_num_threads(threads)
    variants = (GROUP_VARIANTS[dataset], "standard") if dataset in GROUP_VARIANTS else ("standard",)
    assignments = None
    raw = None
    for variant in variants:
        directory = input_directory(root, dataset, variant)
        stamp = directory / "sensitivity_input.json"
        expected = {"dataset": dataset, "variant": variant, "split_seed": SPLIT_SEED, "max_k": max_k}
        if stamp.is_file():
            if json.loads(stamp.read_text(encoding="utf-8")) != expected:
                raise ValueError(f"Prepared inputs in {directory} use another sensitivity protocol")
        else:
            if raw is None:
                raw = preparation._read_raw_dataset(data_root / preparation.DATASETS[dataset])
            caps = {} if dataset == "amazon" else {"max_users": 3500, "max_items": 3500}
            prepared = preparation.prepare_dataset_run(
                dataset, run=0, data_root=data_root, output_root=root, seed=FIT_SEEDS[0],
                recommendation_list_length=max_k, write_shared_run_zero=False,
                allow_missing_demographics=dataset == "amazon", amazon_brand=variant == "brand",
                movielens_director_gender=variant == "director-gender",
                yelp_partition=variant if dataset == "yelp" else None,
                split_seed=SPLIT_SEED, split_assignments=assignments, raw=raw, **caps,
            )
            if prepared.directory != directory:
                raise ValueError(f"Unexpected prepared input directory {prepared.directory}")
            stamp.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
        current = pair_assignments(directory)
        if assignments is not None:
            verify_shared_split(assignments, current, dataset)
        assignments = current
