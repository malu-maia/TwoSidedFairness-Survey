"""Offline FairSort reranking for provider-aware recommendation lists."""

from __future__ import annotations

import math

from typing import Literal, TypeAlias

import torch


NAME = "FairSort"
FairnessRule: TypeAlias = Literal["uniform", "quality"]

def _validate_inputs(
    scores: torch.Tensor,
    item_to_provider: torch.Tensor,
    k: int,
    fairness: FairnessRule,
    lambda_max: float,
    rerank_ratio: float,
    ndcg_floor: float,
    search_gap: float,
    user_order: torch.Tensor | None,
    eligible_items: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Validate public inputs and map provider IDs to contiguous indices."""

    if scores.ndim != 2:
        raise ValueError(
            f"scores must be a 2D user-item matrix, got shape {tuple(scores.shape)}"
        )
    if not scores.is_floating_point():
        raise ValueError("scores must be a floating point tensor")
    if not torch.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("scores must contain finite non-negative values")

    n_users, n_items = scores.shape
    if not 1 <= k <= n_items:
        raise ValueError(
            f"k must be between 1 and the number of items ({n_items}), got {k}"
        )
    if item_to_provider.ndim != 1 or item_to_provider.numel() != n_items:
        raise ValueError(
            "item_to_provider must be a 1D tensor with one provider ID per item"
        )
    if item_to_provider.is_complex():
        raise ValueError("item_to_provider must contain integer provider IDs")
    if item_to_provider.is_floating_point():
        if not torch.equal(item_to_provider, torch.round(item_to_provider)):
            raise ValueError("item_to_provider must contain integer provider IDs")
    if fairness not in {"uniform", "quality"}:
        raise ValueError("fairness must be either 'uniform' or 'quality'")
    if lambda_max < 0 or not math.isfinite(lambda_max):
        raise ValueError("lambda_max must be a finite non-negative value")
    if not 0 < rerank_ratio <= 1:
        raise ValueError("rerank_ratio must be in (0, 1]")
    if not 0 <= ndcg_floor <= 1:
        raise ValueError("ndcg_floor must be in [0, 1]")
    if search_gap <= 0 or not math.isfinite(search_gap):
        raise ValueError("search_gap must be a finite positive value")

    device = scores.device
    provider_ids = item_to_provider.to(device=device)
    _, provider_indices = torch.unique(provider_ids, sorted=True, return_inverse=True)
    provider_indices = provider_indices.to(device=device, dtype=torch.long)

    if user_order is None:
        order = torch.arange(n_users, device=device)
    else:
        if user_order.ndim != 1 or user_order.numel() != n_users:
            raise ValueError("user_order must be a 1D tensor with one entry per user")
        order = user_order.to(device=device, dtype=torch.long)
        if torch.any(order < 0) or torch.any(order >= n_users):
            raise ValueError("user_order contains user indices out of range")
        if torch.unique(order).numel() != n_users:
            raise ValueError("user_order must contain each user exactly once")

    if eligible_items is None:
        eligible = torch.ones_like(scores, dtype=torch.bool, device=device)
    else:
        if eligible_items.shape != scores.shape:
            raise ValueError("eligible_items must have the same shape as scores")
        eligible = eligible_items.to(device=device, dtype=torch.bool)
    if torch.any(eligible.sum(dim=1) < k):
        raise ValueError("each user must have at least k eligible items")

    return provider_indices, order, eligible, torch.unique(provider_indices)


def _ranking_weights(
    k: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return the logarithmic exposure/DCG weights used by FairSort."""

    ranks = torch.arange(2, k + 2, device=device, dtype=dtype)
    return 1 / torch.log2(ranks)


def _provider_statistics(
    scores: torch.Tensor,
    provider_indices: torch.Tensor,
    num_providers: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute provider catalog sizes and aggregate relevance quality."""

    sizes = torch.bincount(provider_indices, minlength=num_providers).to(scores.dtype)
    item_quality = scores.sum(dim=0)
    quality = torch.zeros(num_providers, device=scores.device, dtype=scores.dtype)
    quality.scatter_add_(0, provider_indices, item_quality)
    return sizes, quality


def _provider_exposure(
    rankings: torch.Tensor,
    provider_indices: torch.Tensor,
    weights: torch.Tensor,
    num_providers: int,
) -> torch.Tensor:
    """Aggregate position-weighted exposure for provider-owned ranked items."""

    ranked_providers = provider_indices[rankings[:, : weights.numel()]]
    flat_providers = ranked_providers.reshape(-1)
    weights = weights.repeat(rankings.size(0))
    exposure = torch.zeros(
        num_providers,
        device=rankings.device,
        dtype=weights.dtype,
    )
    exposure.scatter_add_(0, flat_providers, weights)
    return exposure


def _fair_lift_factors(
    current_exposure: torch.Tensor,
    target_exposure: torch.Tensor,
    denominator: torch.Tensor,
) -> torch.Tensor:
    """Compute normalized provider lift velocities from conversion-rate error."""

    safe_denominator = torch.clamp(denominator, min=torch.finfo(denominator.dtype).eps)
    lift = (target_exposure - current_exposure) / safe_denominator

    positive = lift > 0
    negative = lift < 0
    positive_sum = lift[positive].sum()
    negative_sum = lift[negative].abs().sum()
    if positive_sum > 0:
        lift = torch.where(positive, lift / positive_sum, lift)
    if negative_sum > 0:
        lift = torch.where(negative, lift / negative_sum, lift)
    return lift


def _rerank_candidates(
    user_scores: torch.Tensor,
    candidates: torch.Tensor,
    provider_indices: torch.Tensor,
    lift_factors: torch.Tensor,
    fairness_weight: float,
) -> torch.Tensor:
    """Sort candidate items by relevance plus provider fairness velocity."""

    adjusted_scores = (
        user_scores[candidates]
        + fairness_weight * lift_factors[provider_indices[candidates]]
    )
    order = torch.argsort(adjusted_scores, descending=True, stable=True)
    return candidates[order]


def _ndcg_for_ranking(
    user_scores: torch.Tensor,
    ranking: torch.Tensor,
    weights: torch.Tensor,
    ideal_dcg: torch.Tensor,
) -> torch.Tensor:
    """Compute one user's NDCG for a proposed ranking."""

    if ideal_dcg <= 0:
        return torch.ones((), device=user_scores.device, dtype=user_scores.dtype)
    dcg = (user_scores[ranking[: weights.numel()]] * weights).sum()
    return dcg / ideal_dcg


def _rerank_user(
    user_scores: torch.Tensor,
    baseline_ranking: torch.Tensor,
    provider_indices: torch.Tensor,
    lift_factors: torch.Tensor,
    k: int,
    candidate_count: int,
    weights: torch.Tensor,
    lambda_max: float,
    ndcg_floor: float,
    search_gap: float,
) -> torch.Tensor:
    """Rerank one user while preserving the FairSort minimum-utility guarantee."""

    candidates = baseline_ranking[:candidate_count]
    ideal_dcg = (user_scores[baseline_ranking[:k]] * weights).sum()

    left = 0.0
    right = lambda_max
    while right - left > search_gap:
        middle = (left + right) / 2
        reranked = _rerank_candidates(
            user_scores,
            candidates,
            provider_indices,
            lift_factors,
            middle,
        )
        ndcg = _ndcg_for_ranking(user_scores, reranked, weights, ideal_dcg)
        if float(ndcg.item()) < ndcg_floor:
            right = middle
        else:
            left = middle

    return _rerank_candidates(
        user_scores,
        candidates,
        provider_indices,
        lift_factors,
        left,
    )[:k]


def fairsort(
    scores: torch.Tensor,
    item_to_provider: torch.Tensor,
    k: int = 10,
    *,
    fairness: FairnessRule = "uniform",
    lambda_max: float = 8.0,
    rerank_ratio: float = 0.1,
    ndcg_floor: float = 0.95,
    search_gap: float = 0.03125,
    user_order: torch.Tensor | None = None,
    eligible_items: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the offline FairSort reranker to a prepared score matrix.

    FairSort starts from each user's relevance ranking, computes provider-side
    exposure deficits against a fair target, and reranks a candidate prefix
    using a binary search over the provider-fairness weight. The search chooses
    the largest weight that keeps the user's top-k NDCG above ``ndcg_floor``.
    """

    provider_indices, order, eligible, unique_providers = _validate_inputs(
        scores,
        item_to_provider,
        k,
        fairness,
        lambda_max,
        rerank_ratio,
        ndcg_floor,
        search_gap,
        user_order,
        eligible_items,
    )
    n_users, n_items = scores.shape
    num_providers = unique_providers.numel()
    weights = _ranking_weights(k, device=scores.device, dtype=scores.dtype)

    masked_scores = scores.masked_fill(~eligible, -torch.inf)
    baseline_ranking = torch.argsort(masked_scores, dim=1, descending=True, stable=True)
    recommendations = baseline_ranking[:, :k].clone()

    provider_sizes, provider_quality = _provider_statistics(
        scores,
        provider_indices,
        num_providers,
    )
    total_exposure = weights.sum() * n_users
    if fairness == "uniform":
        denominator = provider_sizes
    else:
        if provider_quality.sum() <= 0:
            raise ValueError("quality fairness requires positive provider quality")
        denominator = provider_quality
    target_exposure = total_exposure * denominator / denominator.sum()
    current_exposure = _provider_exposure(
        baseline_ranking[:, :k],
        provider_indices,
        weights,
        num_providers,
    )

    base_candidate_count = max(k, math.ceil(n_items * rerank_ratio))
    for user_index in order.tolist():
        user = int(user_index)
        lift_factors = _fair_lift_factors(
            current_exposure,
            target_exposure,
            denominator,
        )
        candidate_count = min(
            base_candidate_count,
            int(eligible[user].sum().item()),
        )
        reranked = _rerank_user(
            scores[user],
            baseline_ranking[user],
            provider_indices,
            lift_factors,
            k,
            candidate_count,
            weights,
            lambda_max,
            ndcg_floor,
            search_gap,
        )
        recommendations[user] = reranked
        baseline_topk = baseline_ranking[user, :k].unsqueeze(0)
        current_exposure -= _provider_exposure(
            baseline_topk,
            provider_indices,
            weights,
            num_providers,
        )
        current_exposure += _provider_exposure(
            reranked.unsqueeze(0),
            provider_indices,
            weights,
            num_providers,
        )

    return scores, recommendations