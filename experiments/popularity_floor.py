"""Print the popularity-baseline NDCG floor for a prepared run directory.

Ranks eligible items by train interaction count (same order for every user)
and evaluates against the graded held-out relevance. The learned base model
must clear this floor to be worth using.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch

from .metrics import ndcg


def popularity_ndcg(run_dir: Path, k: int) -> float:
    train_matrix = torch.tensor(
        pd.read_parquet(run_dir / "train_matrix.parquet").to_numpy(), dtype=torch.float32
    )
    relevance = torch.tensor(
        pd.read_parquet(run_dir / "heldout_relevance.parquet").to_numpy(),
        dtype=torch.float32,
    )
    popularity = (train_matrix > 0).sum(dim=0).to(torch.float32)
    scores = popularity.unsqueeze(0).expand_as(train_matrix).clone()
    scores[train_matrix > 0] = -torch.inf
    rankings = torch.topk(scores, k=min(k, scores.shape[1]), dim=1).indices
    return float(ndcg(relevance, rankings, k=min(k, scores.shape[1])).mean)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="prepared run directory, e.g. experiments/runs/movielens/run_00/seed_42")
    parser.add_argument("--k", type=int, nargs="+", default=[10, 150])
    args = parser.parse_args(argv)
    for k in args.k:
        print(f"popularity NDCG@{k}: {popularity_ndcg(args.run_dir, k):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
