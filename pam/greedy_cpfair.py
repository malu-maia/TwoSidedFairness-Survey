from __future__ import annotations

import numpy as np
import torch


NAME = "CPFair"
PROTECTED_GROUP = 1
UNPROTECTED_GROUP = -1


def _group_vector(
    mapping: dict[int, int],
    *,
    size: int,
    name: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Validate a CPFair group mapping and return it as a tensor."""

    missing_ids = [
        entity_id for entity_id in range(size) if entity_id not in mapping
    ]
    if missing_ids:
        preview = ", ".join(map(str, missing_ids[:5]))
        raise ValueError(f"{name} does not cover IDs: {preview}")

    values = [mapping[entity_id] for entity_id in range(size)]
    if set(values) != {UNPROTECTED_GROUP, PROTECTED_GROUP}:
        raise ValueError(
            f"{name} must use -1 for the advantaged group and 1 for the "
            "protected group, with both groups represented"
        )
    return torch.as_tensor(values, device=device, dtype=dtype)


def _validate_base_topk(
    base_topk: np.ndarray | torch.Tensor,
    *,
    n_users: int,
    n_items: int,
    device: torch.device,
) -> None:
    """Validate the legacy baseline-list argument at the API boundary."""

    try:
        ranking = torch.as_tensor(base_topk, device=device, dtype=torch.long)
    except (TypeError, ValueError) as error:
        raise ValueError("base_topk must be a rectangular 2D array") from error
    if ranking.ndim != 2 or ranking.shape[0] != n_users:
        raise ValueError("base_topk must be a 2D array with one row per user")
    if ranking.shape[1] == 0:
        raise ValueError("base_topk must contain at least one item per user")
    if torch.any(ranking < 0) or torch.any(ranking >= n_items):
        raise ValueError("base_topk contains item IDs outside the score matrix")


def cpfair(
    scores: torch.Tensor,
    user_groups: dict[int, int],
    item_groups: dict[int, int],
    base_topk: np.ndarray | torch.Tensor,
    lambda1: float = 0.9,
    lambda2: float = 0.1,
    k: int = 150,
    *,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Re-rank the full item catalog with CPFair's greedy scoring rule.

    Group value ``1`` denotes the protected group and ``-1`` the advantaged
    group. Therefore, callers should encode inactive consumers and long-tail
    items as ``1``; short-head items (the popularity top 20%) are ``-1`` and
    the remaining 80% are long-tail.

    Following Algorithm 2, consumer utility uses the candidate relevance score
    and producer utility uses binary exposure. Every catalog item is rescored;
    ``k`` is used only to select the final recommendation list. ``base_topk``
    remains in the public interface for compatibility and boundary validation.
    """

    if scores.ndim != 2:
        raise ValueError(
            f"scores must be a 2D user-item matrix, got shape {tuple(scores.shape)}"
        )
    if not scores.is_floating_point():
        raise TypeError("scores must contain floating-point relevance values")

    n_users, n_items = scores.shape
    if not 1 <= k <= n_items:
        raise ValueError(
            f"k must be between 1 and the number of items ({n_items}), got {k}"
        )
    if not 0 <= lambda1 <= 1 or not 0 <= lambda2 <= 1:
        raise ValueError("lambda1 and lambda2 must be between 0 and 1")
    if torch.isnan(scores).any():
        raise ValueError("scores must not contain NaN values")

    resolved_device = scores.device if device is None else torch.device(device)
    resolved_scores = scores.to(resolved_device)
    _validate_base_topk(
        base_topk,
        n_users=n_users,
        n_items=n_items,
        device=resolved_device,
    )
    user_group_values = _group_vector(
        user_groups,
        size=n_users,
        name="user_groups",
        device=resolved_device,
        dtype=resolved_scores.dtype,
    )
    item_group_values = _group_vector(
        item_groups,
        size=n_items,
        name="item_groups",
        device=resolved_device,
        dtype=resolved_scores.dtype,
    )

    # Algorithm 2: CF_ui = UG_u * M_C(u, i), where M_C is relevance.
    # Masked candidates may use +/-inf, so they contribute no fairness value.
    candidate_relevance = torch.where(
        torch.isfinite(resolved_scores),
        resolved_scores,
        torch.zeros_like(resolved_scores),
    )
    consumer_fairness = user_group_values.unsqueeze(1) * candidate_relevance

    # Algorithm 2 uses binary item exposure, hence M_P(u, i) is one.
    producer_fairness = item_group_values.unsqueeze(0)
    fair_scores = (
        resolved_scores
        + lambda1 * consumer_fairness
        + lambda2 * producer_fairness
    )
    fair_topk = torch.topk(fair_scores, k=k, dim=1).indices
    return fair_scores, fair_topk
