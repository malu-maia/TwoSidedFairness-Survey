"""Focused metric helpers for two-sided recommendation experiments.

The module contains scalar and distribution metrics used by the June 2026
experiments. Inputs are PyTorch tensors, and tensor-valued computations remain
on the input device and floating-point dtype where applicable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Literal, TypeAlias

import torch

DiscountFunction: TypeAlias = Callable[[torch.Tensor], torch.Tensor]
DiscountSpec: TypeAlias = Literal["log", "uniform"] | torch.Tensor | DiscountFunction


@dataclass(frozen=True)
class DistributionStats:
    """Per-agent values and population summary statistics.

    Attributes:
        values: One-dimensional tensor containing one observation per agent.
        mean: Population arithmetic mean.
        variance: Population variance computed with ``correction=0``.
        std: Population standard deviation.
    """

    values: torch.Tensor
    mean: float
    variance: float
    std: float


@dataclass(frozen=True)
class EnvyMetrics:
    """Consumer envy measured from paper-normalized recommendation utilities.

    Attributes:
        mean_average_envy: Mean average envy ``Y`` from Patro et al. (WWW
            2020), where lower values indicate fairer recommendations.
        envious_pair_rate: Fraction of ordered, distinct consumer pairs for
            which the first consumer prefers the second consumer's ranking.
        mean_positive_gap: Mean positive normalized utility gain among
            envious pairs.
        max_positive_gap: Largest positive normalized utility gain among
            envious pairs.
    """

    mean_average_envy: float
    envious_pair_rate: float
    mean_positive_gap: float
    max_positive_gap: float

    @property
    def envy_free_rate(self) -> float:
        """Compatibility alias for the configured envy-freeness score."""

        return self.mean_average_envy


@dataclass(frozen=True)
class _RankingContext:
    """Validated tensors shared by ranking-dependent metric computations."""

    scores: torch.Tensor
    rankings: torch.Tensor
    discounts: torch.Tensor
    allocation: torch.Tensor
    k: int


def _as_float_tensor(values: torch.Tensor, *, name: str) -> torch.Tensor:
    """Convert input to a finite floating-point tensor."""

    tensor = torch.as_tensor(values)
    if not tensor.is_floating_point():
        tensor = tensor.to(torch.get_default_dtype())
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain only finite values")
    return tensor


def _validate_nonnegative_vector(values: torch.Tensor, *, name: str) -> torch.Tensor:
    """Validate a non-empty, one-dimensional, non-negative tensor."""

    tensor = _as_float_tensor(values, name=name)
    if tensor.ndim != 1 or tensor.numel() == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional tensor")
    if (tensor < 0).any():
        raise ValueError(f"{name} values must be non-negative")
    return tensor


def _ranking_discounts(
    k: int,
    discount: DiscountSpec = "log",
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Build non-negative, non-increasing position weights for ranks ``0..k-1``."""

    if k <= 0:
        raise ValueError("k must be greater than zero")

    dtype = dtype or torch.get_default_dtype()
    positions = torch.arange(k, device=device, dtype=dtype)

    if isinstance(discount, str):
        if discount == "log":
            weights = torch.reciprocal(torch.log2(positions + 2))
        elif discount == "uniform":
            weights = torch.ones(k, device=device, dtype=dtype)
        else:
            raise ValueError("discount must be 'log', 'uniform', a tensor, or a callable")
    elif callable(discount):
        weights = torch.as_tensor(discount(positions), device=device, dtype=dtype)
    else:
        weights = torch.as_tensor(discount, device=device, dtype=dtype)

    if weights.ndim != 1 or weights.numel() != k:
        raise ValueError(f"discount weights must have shape ({k},)")
    if not torch.isfinite(weights).all():
        raise ValueError("discount weights must contain only finite values")
    if (weights < 0).any():
        raise ValueError("discount weights must be non-negative")
    if not (weights > 0).any():
        raise ValueError("at least one discount weight must be positive")
    if k > 1 and (weights[1:] > weights[:-1]).any():
        raise ValueError("discount weights must be non-increasing by rank")

    return weights


def _prepare_context(
    scores: torch.Tensor,
    rankings: torch.Tensor,
    k: int | None,
    discount: DiscountSpec,
) -> _RankingContext:
    """Validate relevance and ranking inputs for metric computation.

    ``scores`` is the evaluation relevance/utility matrix, typically extracted
    from the held-out test set. It is not the recommender's prediction matrix
    unless a caller intentionally evaluates against predictions.
    """

    score_tensor = _as_float_tensor(scores, name="scores")
    if score_tensor.ndim != 2:
        raise ValueError("scores must have shape (consumers, items)")
    if score_tensor.shape[0] == 0 or score_tensor.shape[1] == 0:
        raise ValueError("scores must contain at least one consumer and one item")
    if (score_tensor < 0).any():
        raise ValueError("scores must be non-negative")

    ranking_tensor = torch.as_tensor(rankings, device=score_tensor.device)
    if ranking_tensor.ndim != 2:
        raise ValueError("rankings must have shape (consumers, ranks)")
    if ranking_tensor.shape[0] != score_tensor.shape[0]:
        raise ValueError("scores and rankings must have the same number of consumers")
    if ranking_tensor.is_floating_point() or ranking_tensor.is_complex():
        raise ValueError("rankings must contain integer item IDs")
    ranking_tensor = ranking_tensor.to(torch.long)

    resolved_k = ranking_tensor.shape[1] if k is None else k
    if resolved_k <= 0:
        raise ValueError("k must be greater than zero")
    if resolved_k > ranking_tensor.shape[1]:
        raise ValueError("k cannot exceed the number of ranking columns")
    if resolved_k > score_tensor.shape[1]:
        raise ValueError("k cannot exceed the number of items")

    top_k = ranking_tensor[:, :resolved_k]
    if (top_k < 0).any() or (top_k >= score_tensor.shape[1]).any():
        raise ValueError("rankings contain an item ID outside the score matrix")
    if resolved_k > 1:
        sorted_rankings = torch.sort(top_k, dim=1).values
        if (sorted_rankings[:, 1:] == sorted_rankings[:, :-1]).any():
            raise ValueError("each consumer ranking must contain unique item IDs")

    discounts = _ranking_discounts(
        resolved_k,
        discount,
        device=score_tensor.device,
        dtype=score_tensor.dtype,
    )
    allocation = torch.zeros_like(score_tensor)
    allocation.scatter_(1, top_k, discounts.expand(score_tensor.shape[0], -1))

    return _RankingContext(
        scores=score_tensor,
        rankings=top_k,
        discounts=discounts,
        allocation=allocation,
        k=resolved_k,
    )


def _distribution_stats(values: torch.Tensor) -> DistributionStats:
    """Summarize a one-dimensional population distribution."""

    values = _as_float_tensor(values, name="values")
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("values must be a non-empty one-dimensional tensor")

    variance = torch.var(values, correction=0)
    return DistributionStats(
        values=values,
        mean=float(torch.mean(values).item()),
        variance=float(variance.item()),
        std=float(torch.sqrt(variance).item()),
    )


def _consumer_utilities(context: _RankingContext) -> torch.Tensor:
    """Compute discounted relevance utility from a validated ranking context."""

    ranked_relevance = torch.gather(context.scores, 1, context.rankings)
    return torch.sum(ranked_relevance * context.discounts, dim=1)


def _ndcg(context: _RankingContext) -> torch.Tensor:
    """Compute per-consumer NDCG from a validated ranking context."""

    actual_dcg = _consumer_utilities(context)
    ideal_relevance = torch.topk(
        context.scores,
        k=context.k,
        dim=1,
        largest=True,
        sorted=True,
    ).values
    ideal_dcg = torch.sum(ideal_relevance * context.discounts, dim=1)
    return torch.where(ideal_dcg > 0, actual_dcg / ideal_dcg, torch.zeros_like(actual_dcg))


def ndcg(
    scores: torch.Tensor,
    rankings: torch.Tensor,
    k: int | None = None,
    discount: DiscountSpec = "log",
) -> DistributionStats:
    """Compute per-consumer NDCG and population summary statistics.

    ``scores`` contains the relevance ground truth used for evaluation. In the
    experiment pipeline this should be the held-out test relevance matrix, while
    ``rankings`` contains item IDs output by the recommendation method.

    A consumer with zero ideal DCG is assigned NDCG zero.
    """

    return _distribution_stats(_ndcg(_prepare_context(scores, rankings, k, discount)))


def utility_variance(values: torch.Tensor) -> float:
    """Compute population variance of utility or satisfaction values."""

    tensor = _validate_nonnegative_vector(values, name="values")
    return float(torch.var(tensor, correction=0).item())


def envy_freeness(
    scores: torch.Tensor,
    rankings: torch.Tensor,
    k: int | None = None,
    discount: DiscountSpec = "log",
    *,
    tolerance: float = 1e-12,
) -> EnvyMetrics:
    """Compute paper-style mean average envy of a recommendation run.

    ``scores`` is the relevance/utility matrix used to value each recommended
    set, typically from held-out test data. ``rankings`` is the method-produced
    recommendation list, not the held-out ranking.

    The ``discount`` parameter is accepted for API compatibility; envy follows
    the FairRec paper's unweighted recommendation-set utility.
    """

    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")

    context = _prepare_context(scores, rankings, k, "uniform")
    set_allocation = torch.zeros_like(context.scores)
    set_allocation.scatter_(
        1,
        context.rankings,
        torch.ones_like(context.rankings, dtype=context.scores.dtype),
    )
    raw_cross_utilities = context.scores @ set_allocation.T
    ideal_utilities = torch.sum(
        torch.topk(context.scores, k=context.k, dim=1, largest=True).values,
        dim=1,
    )
    cross_utilities = torch.zeros_like(raw_cross_utilities)
    positive_ideal = ideal_utilities > 0
    cross_utilities[positive_ideal] = (
        raw_cross_utilities[positive_ideal] / ideal_utilities[positive_ideal, None]
    )
    own_utilities = torch.diagonal(cross_utilities)
    positive_gaps = torch.clamp(cross_utilities - own_utilities[:, None], min=0)
    positive_gaps.fill_diagonal_(0)
    envy_mask = positive_gaps > tolerance

    n_consumers = context.scores.shape[0]
    pair_count = n_consumers * (n_consumers - 1)
    mean_average_envy = (
        float(torch.sum(positive_gaps).item()) / pair_count if pair_count else 0.0
    )
    envious_pair_rate = (
        float(torch.sum(envy_mask).item()) / pair_count if pair_count else 0.0
    )
    observed_gaps = positive_gaps[envy_mask]
    mean_gap = float(torch.mean(observed_gaps).item()) if observed_gaps.numel() else 0.0
    max_gap = float(torch.max(observed_gaps).item()) if observed_gaps.numel() else 0.0

    return EnvyMetrics(
        mean_average_envy=mean_average_envy,
        envious_pair_rate=envious_pair_rate,
        mean_positive_gap=mean_gap,
        max_positive_gap=max_gap,
    )


def worse_off_cumulative_utility(values: torch.Tensor, fraction: float) -> torch.Tensor:
    """Sum utility for the bottom ``fraction`` of users.

    ``fraction`` is a population share in ``(0, 1]``; for example, ``0.1``
    selects the bottom 10% of users.
    """

    try:
        fraction_value = float(fraction)
    except (TypeError, ValueError) as error:
        raise ValueError("fraction must be a finite value in (0, 1]") from error
    if not math.isfinite(fraction_value) or not 0 < fraction_value <= 1:
        raise ValueError("fraction must be a finite value in (0, 1]")

    tensor = _validate_nonnegative_vector(values, name="values")
    selected_count = max(1, math.ceil(tensor.numel() * fraction_value))
    sorted_values = torch.sort(tensor).values
    return torch.sum(sorted_values[:selected_count])


def sum_user_utilities(values: torch.Tensor) -> float:
    """Compute the total utility across users."""

    tensor = _validate_nonnegative_vector(values, name="values")
    return float(torch.sum(tensor).item())


def gini_coefficient(values: torch.Tensor) -> float:
    """Compute the Gini coefficient of a non-negative distribution."""

    tensor = _validate_nonnegative_vector(values, name="values")
    total = torch.sum(tensor)
    if total == 0:
        return 0.0

    sorted_values = torch.sort(tensor).values
    n = sorted_values.numel()
    indices = torch.arange(1, n + 1, device=tensor.device, dtype=tensor.dtype)
    gini = (2 * torch.sum(indices * sorted_values)) / (n * total) - (n + 1) / n
    return float(torch.clamp(gini, min=0, max=1).item())


def exposure_relevance_ratio_variance(
    exposure_values: torch.Tensor,
    relevance_values: torch.Tensor,
) -> float:
    """Compute population variance of exposure divided by relevance."""

    exposure_tensor = _validate_nonnegative_vector(
        exposure_values,
        name="exposure_values",
    )
    relevance_tensor = _as_float_tensor(relevance_values, name="relevance_values").to(
        device=exposure_tensor.device,
        dtype=exposure_tensor.dtype,
    )
    if relevance_tensor.shape != exposure_tensor.shape:
        raise ValueError(
            "exposure_values and relevance_values must have the same shape"
        )
    if (relevance_tensor <= 0).any():
        raise ValueError("relevance values must be strictly positive")

    ratio = exposure_tensor / relevance_tensor
    return float(torch.var(ratio, correction=0).item())
