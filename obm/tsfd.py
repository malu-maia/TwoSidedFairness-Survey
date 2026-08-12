"""TSFD Rank for user fairness, item fairness, and diversity.

This module implements the two-step TSFD Rank procedure from Wang and
Joachims, "User Fairness, Item Fairness, and Diversity for Rankings in
Two-Sided Markets" (ICTIR 2021). The original paper assumes explicit item
relevance per intent. This repository does not store intent annotations, so
the public entry point adapts the paper by treating user groups as proxy
intents and estimating item-intent relevance from mean scores within each
user group.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, TypeAlias

import numpy as np
import torch
from scipy.optimize import LinearConstraint, Bounds, linear_sum_assignment, minimize


NAME = "TSFD"
ItemFairnessRule: TypeAlias = Literal["one_sided", "equality"]


def _group_indices(
    groups: torch.Tensor | Sequence[int] | Mapping[int, int],
    expected_size: int,
    name: str,
) -> torch.Tensor:
    """Validate group IDs and map them to contiguous integer indices."""

    if isinstance(groups, Mapping):
        missing = [index for index in range(expected_size) if index not in groups]
        if missing:
            raise ValueError(f"{name} must contain one group ID per entity")
        values = torch.tensor([groups[index] for index in range(expected_size)])
    else:
        values = torch.as_tensor(groups)

    if values.ndim != 1 or values.numel() != expected_size:
        raise ValueError(f"{name} must be a 1D tensor with {expected_size} entries")
    if values.is_complex():
        raise ValueError(f"{name} must contain integer group IDs")
    if values.is_floating_point():
        if not torch.isfinite(values).all():
            raise ValueError(f"{name} must contain only finite group IDs")
        if not torch.equal(values, values.round()):
            raise ValueError(f"{name} must contain integer group IDs")

    _, indices = torch.unique(values.to(torch.long), sorted=True, return_inverse=True)
    if torch.unique(indices).numel() < 2:
        raise ValueError(f"{name} must contain at least two groups")
    return indices.to(device="cpu", dtype=torch.long)


def _validate_scores(scores: torch.Tensor) -> torch.Tensor:
    if scores.ndim != 2:
        raise ValueError(
            f"scores must be a 2D user-item matrix, got shape {tuple(scores.shape)}"
        )
    if scores.numel() == 0:
        raise ValueError("scores must not be empty")
    if scores.is_complex():
        raise ValueError("scores must contain real values")

    values = scores.detach().to(dtype=torch.float64, device="cpu")
    if not torch.isfinite(values).all() or (values < 0).any():
        raise ValueError("scores must contain finite non-negative values")
    return values


def _exposure_weights(n_items: int, exposure_steepness: float) -> np.ndarray:
    if not np.isfinite(exposure_steepness) or exposure_steepness <= 0:
        raise ValueError("exposure_steepness must be finite and positive")
    ranks = np.arange(1, n_items + 1, dtype=np.float64)
    return 1.0 / np.power(ranks, exposure_steepness)


def _proxy_intent_relevance(
    scores: torch.Tensor,
    user_groups: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate paper intent inputs by treating user groups as proxy intents."""

    group_ids = torch.unique(user_groups, sorted=True)
    relevance_by_group = []
    proportions = []
    for group_id in group_ids:
        mask = user_groups == group_id
        relevance_by_group.append(scores[mask].mean(dim=0).numpy())
        proportions.append(float(mask.to(torch.float64).mean().item()))

    relevance = np.vstack(relevance_by_group).astype(np.float64)
    group_proportions = np.asarray(proportions, dtype=np.float64)
    expected_relevance = group_proportions @ relevance

    if np.any(relevance.sum(axis=1) <= 0):
        raise ValueError("each user group must have positive aggregate relevance")
    if np.all(expected_relevance <= 0):
        raise ValueError("scores must provide positive expected item relevance")
    return relevance, group_proportions, expected_relevance


def _item_group_statistics(
    expected_relevance: np.ndarray,
    item_groups: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    group_ids = torch.unique(item_groups, sorted=True)
    labels = item_groups.numpy()
    indicators = []
    sizes = []
    merits = []
    for group_id in group_ids.tolist():
        mask = labels == int(group_id)
        indicators.append(mask.astype(np.float64))
        size = float(mask.sum())
        sizes.append(size)
        merits.append(float(expected_relevance[mask].mean()))

    indicators_array = np.vstack(indicators)
    sizes_array = np.asarray(sizes, dtype=np.float64)
    merits_array = np.asarray(merits, dtype=np.float64)
    if np.any(merits_array <= 0):
        raise ValueError("each item group must have positive merit")
    return indicators_array, sizes_array, merits_array


def _doubly_stochastic_constraint(n_items: int) -> LinearConstraint:
    variables = n_items * n_items
    matrix = np.zeros((2 * n_items, variables), dtype=np.float64)
    for item in range(n_items):
        matrix[item, item * n_items : (item + 1) * n_items] = 1.0
    for position in range(n_items):
        matrix[n_items + position, position::n_items] = 1.0
    ones = np.ones(2 * n_items, dtype=np.float64)
    return LinearConstraint(matrix, ones, ones)


def _item_fairness_constraint(
    indicators: np.ndarray,
    sizes: np.ndarray,
    merits: np.ndarray,
    exposure: np.ndarray,
    rule: ItemFairnessRule,
) -> LinearConstraint | None:
    """Create linear disparate-treatment constraints from the paper."""

    if rule not in {"one_sided", "equality"}:
        raise ValueError("item_fairness must be either 'one_sided' or 'equality'")

    normalized = indicators / sizes[:, None] / merits[:, None]
    rows = []
    lower = []
    upper = []

    if rule == "equality":
        base = normalized[0]
        for group_index in range(1, normalized.shape[0]):
            rows.append(np.outer(normalized[group_index] - base, exposure).ravel())
            lower.append(0.0)
            upper.append(0.0)
    else:
        order = np.argsort(merits)
        for left, right in zip(order[:-1], order[1:]):
            if np.isclose(merits[left], merits[right]):
                continue
            # Paper experiment constraint: when merit(left) <= merit(right),
            # exposure(left) / merit(left) <= exposure(right) / merit(right).
            rows.append(np.outer(normalized[left] - normalized[right], exposure).ravel())
            lower.append(-np.inf)
            upper.append(0.0)

    if not rows:
        return None
    return LinearConstraint(
        np.vstack(rows),
        np.asarray(lower, dtype=np.float64),
        np.asarray(upper, dtype=np.float64),
    )


def _solve_marginal_rank_probabilities(
    relevance_by_group: np.ndarray,
    group_proportions: np.ndarray,
    item_indicators: np.ndarray,
    item_group_sizes: np.ndarray,
    item_group_merits: np.ndarray,
    exposure: np.ndarray,
    item_fairness: ItemFairnessRule,
    maxiter: int,
    tol: float,
    eps: float,
) -> np.ndarray:
    """Solve the TSFD convex optimization problem for Sigma."""

    n_items = relevance_by_group.shape[1]
    initial = np.full(n_items * n_items, 1.0 / n_items, dtype=np.float64)

    def objective(flat_sigma: np.ndarray) -> float:
        sigma = flat_sigma.reshape(n_items, n_items)
        utilities = relevance_by_group @ sigma @ exposure
        if np.any(utilities + eps <= 0):
            return np.inf
        return -float(np.dot(group_proportions, np.log(utilities + eps)))

    def gradient(flat_sigma: np.ndarray) -> np.ndarray:
        sigma = flat_sigma.reshape(n_items, n_items)
        utilities = relevance_by_group @ sigma @ exposure
        weights = group_proportions / (utilities + eps)
        weighted_relevance = weights @ relevance_by_group
        return -np.outer(weighted_relevance, exposure).ravel()

    constraints: list[LinearConstraint] = [_doubly_stochastic_constraint(n_items)]
    item_constraint = _item_fairness_constraint(
        item_indicators,
        item_group_sizes,
        item_group_merits,
        exposure,
        item_fairness,
    )
    if item_constraint is not None:
        constraints.append(item_constraint)

    result = minimize(
        objective,
        initial,
        jac=gradient,
        method="SLSQP",
        bounds=Bounds(0.0, 1.0),
        constraints=constraints,
        options={"maxiter": maxiter, "ftol": tol, "disp": False},
    )
    if not result.success:
        raise ValueError(f"TSFD convex optimization failed: {result.message}")

    sigma = result.x.reshape(n_items, n_items)
    sigma = np.clip(sigma, 0.0, 1.0)
    row_sums = sigma.sum(axis=1, keepdims=True)
    sigma = np.divide(sigma, row_sums, out=np.zeros_like(sigma), where=row_sums > 0)
    sigma = _sinkhorn_project(sigma, iterations=25)
    return sigma


def _sinkhorn_project(matrix: np.ndarray, iterations: int) -> np.ndarray:
    """Numerically clean a nearly doubly stochastic matrix."""

    result = matrix.copy()
    for _ in range(iterations):
        row_sums = result.sum(axis=1, keepdims=True)
        result = np.divide(result, row_sums, out=np.zeros_like(result), where=row_sums > 0)
        col_sums = result.sum(axis=0, keepdims=True)
        result = np.divide(result, col_sums, out=np.zeros_like(result), where=col_sums > 0)
    return result


def _ranking_diversity(
    ranking_item_to_position: np.ndarray,
    relevance_by_group: np.ndarray,
    group_proportions: np.ndarray,
    exposure: np.ndarray,
    eps: float = 1e-9,
) -> float:
    """Compute the paper's concave diversity objective for one ranking."""

    item_exposure = exposure[ranking_item_to_position]
    utilities = relevance_by_group @ item_exposure
    return float(np.dot(group_proportions, np.log(utilities + eps)))


def _local_search_match(
    sigma: np.ndarray,
    relevance_by_group: np.ndarray,
    group_proportions: np.ndarray,
    expected_relevance: np.ndarray,
    exposure: np.ndarray,
    tol: float,
) -> np.ndarray:
    """Algorithm 2: utility matching followed by diversity-improving swaps."""

    graph = sigma > tol
    n_items = sigma.shape[0]
    utility = np.outer(expected_relevance, exposure)
    cost = np.where(graph, -utility, 1e12)
    rows, cols = linear_sum_assignment(cost)
    if len(rows) != n_items or np.any(cost[rows, cols] >= 1e11):
        raise ValueError("TSFD BvN decomposition could not find a perfect matching")

    ranking = np.empty(n_items, dtype=np.int64)
    ranking[rows] = cols
    current = _ranking_diversity(
        ranking,
        relevance_by_group,
        group_proportions,
        exposure,
    )

    improved = True
    while improved:
        improved = False
        for left in range(n_items):
            for right in range(left + 1, n_items):
                left_position = ranking[left]
                right_position = ranking[right]
                if not graph[left, right_position] or not graph[right, left_position]:
                    continue
                candidate = ranking.copy()
                candidate[left], candidate[right] = right_position, left_position
                candidate_diversity = _ranking_diversity(
                    candidate,
                    relevance_by_group,
                    group_proportions,
                    exposure,
                )
                if candidate_diversity > current + tol:
                    ranking = candidate
                    current = candidate_diversity
                    improved = True
        # Keep scanning until a full pass finds no improving pair.
    return ranking


def _bvn_decomposition(
    sigma: np.ndarray,
    relevance_by_group: np.ndarray,
    group_proportions: np.ndarray,
    expected_relevance: np.ndarray,
    exposure: np.ndarray,
    tol: float = 1e-8,
) -> list[tuple[float, np.ndarray]]:
    """Algorithm 1: greedy Birkhoff-von Neumann decomposition."""

    residual = sigma.copy()
    residual[residual < tol] = 0.0
    policy: list[tuple[float, np.ndarray]] = []
    max_rounds = (sigma.shape[0] - 1) ** 2 + 1

    for _ in range(max_rounds + sigma.shape[0]):
        if residual.max() <= tol:
            break
        ranking = _local_search_match(
            residual,
            relevance_by_group,
            group_proportions,
            expected_relevance,
            exposure,
            tol,
        )
        selected = residual[np.arange(residual.shape[0]), ranking]
        weight = float(selected.min())
        if weight <= tol:
            break
        policy.append((weight, ranking.copy()))
        residual[np.arange(residual.shape[0]), ranking] -= weight
        residual[np.abs(residual) < tol] = 0.0
        residual[residual < 0] = 0.0

    if residual.max() > 1e-5:
        raise ValueError("TSFD BvN decomposition did not converge")

    total_weight = sum(weight for weight, _ in policy)
    if total_weight <= 0:
        raise ValueError("TSFD BvN decomposition produced an empty policy")
    return [(weight / total_weight, ranking) for weight, ranking in policy]


def _policy_to_recommendations(
    policy: list[tuple[float, np.ndarray]],
    n_users: int,
    k: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    weights = np.asarray([weight for weight, _ in policy], dtype=np.float64)
    weights = weights / weights.sum()
    rng = np.random.default_rng(seed)
    choices = rng.choice(len(policy), size=n_users, replace=True, p=weights)

    rows = []
    for choice in choices:
        ranking = policy[int(choice)][1]
        rows.append(np.argsort(ranking)[:k])
    return torch.as_tensor(np.vstack(rows), dtype=torch.long, device=device)


def tsfd(
    scores: torch.Tensor,
    user_groups: torch.Tensor | Sequence[int] | Mapping[int, int],
    item_groups: torch.Tensor | Sequence[int] | Mapping[int, int],
    k: int = 10,
    *,
    exposure_steepness: float = 1.0,
    item_fairness: ItemFairnessRule = "one_sided",
    maxiter: int = 1000,
    tol: float = 1e-8,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run TSFD Rank using user groups as proxy intents.

    Args:
        scores: Non-negative user-item score matrix with shape
            ``(n_users, n_items)``.
        user_groups: One group ID per user. Groups are used as proxy intents.
        item_groups: One group ID per item for item-fairness constraints.
        k: Number of recommended items to return per user.
        exposure_steepness: Paper exposure parameter for ``1 / rank**eta``.
        item_fairness: ``"one_sided"`` uses the paper's experiment constraint;
            ``"equality"`` enforces equal exposure/merit ratios.
        maxiter: Maximum SLSQP iterations for the convex TSFD step.
        tol: Numerical tolerance for optimization and decomposition.
        seed: Seed used when sampling rankings from the decomposed policy.

    Returns:
        The unchanged score matrix and top-k item indices sampled from the TSFD
        ranking policy.
    """

    validated_scores = _validate_scores(scores)
    n_users, n_items = validated_scores.shape
    if not 1 <= k <= n_items:
        raise ValueError(
            f"k must be between 1 and the number of items ({n_items}), got {k}"
        )
    if maxiter < 1:
        raise ValueError(f"maxiter must be at least 1, got {maxiter}")
    if tol <= 0 or not np.isfinite(tol):
        raise ValueError("tol must be finite and positive")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    resolved_user_groups = _group_indices(user_groups, n_users, "user_groups")
    resolved_item_groups = _group_indices(item_groups, n_items, "item_groups")
    exposure = _exposure_weights(n_items, exposure_steepness)
    relevance_by_group, group_proportions, expected_relevance = _proxy_intent_relevance(
        validated_scores,
        resolved_user_groups,
    )
    item_indicators, item_group_sizes, item_group_merits = _item_group_statistics(
        expected_relevance,
        resolved_item_groups,
    )

    sigma = _solve_marginal_rank_probabilities(
        relevance_by_group,
        group_proportions,
        item_indicators,
        item_group_sizes,
        item_group_merits,
        exposure,
        item_fairness,
        maxiter,
        tol,
        eps=1e-9,
    )
    policy = _bvn_decomposition(
        sigma,
        relevance_by_group,
        group_proportions,
        expected_relevance,
        exposure,
        tol=tol,
    )
    recommendations = _policy_to_recommendations(
        policy,
        n_users,
        k,
        seed,
        scores.device,
    )
    return scores, recommendations


def train(
    scores: torch.Tensor,
    user_groups: torch.Tensor | Sequence[int] | Mapping[int, int],
    item_groups: torch.Tensor | Sequence[int] | Mapping[int, int],
    k: int = 10,
    *,
    exposure_steepness: float = 1.0,
    item_fairness: ItemFairnessRule = "one_sided",
    maxiter: int = 1000,
    tol: float = 1e-8,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compatibility wrapper for :func:`tsfd`."""

    return tsfd(
        scores=scores,
        user_groups=user_groups,
        item_groups=item_groups,
        k=k,
        exposure_steepness=exposure_steepness,
        item_fairness=item_fairness,
        maxiter=maxiter,
        tol=tol,
        seed=seed,
    )
