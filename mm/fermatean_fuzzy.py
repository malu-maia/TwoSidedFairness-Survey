"""Fermatean-fuzzy-inspired two-sided fair matching.

The public entry point keeps the matching-market contract of the source model:
agents on side A are matched one-to-one with agents on side B. It returns the
objective coefficients used for the final fair matching and the selected pairs;
it does not produce top-k recommendation lists.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


_EPSILON = 1e-12


def _as_float_matrix(values: torch.Tensor, *, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(values)
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix, got shape {tuple(tensor.shape)}")
    if tensor.numel() == 0:
        raise ValueError(f"{name} must be non-empty")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain only finite values")
    if tensor.is_complex():
        raise ValueError(f"{name} must contain real-valued preferences")
    return tensor.to(dtype=torch.float64)


def _validate_eligible_pairs(
    eligible_pairs: torch.Tensor | None,
    shape: tuple[int, int],
    *,
    device: torch.device,
) -> torch.Tensor:
    if eligible_pairs is None:
        return torch.ones(shape, dtype=torch.bool, device=device)
    eligible = torch.as_tensor(eligible_pairs, device=device)
    if eligible.shape != shape:
        raise ValueError(
            "eligible_pairs must have shape "
            f"{shape}, got {tuple(eligible.shape)}"
        )
    return eligible.to(dtype=torch.bool)


def _normalize_valid(values: torch.Tensor, eligible: torch.Tensor) -> torch.Tensor:
    normalized = torch.zeros_like(values, dtype=torch.float64)
    valid_values = values[eligible]
    minimum = torch.min(valid_values)
    maximum = torch.max(valid_values)
    scale = maximum - minimum
    if torch.abs(scale) <= _EPSILON:
        normalized[eligible] = 1.0
        return normalized
    normalized[eligible] = (valid_values - minimum) / scale
    return normalized.clamp(min=0.0, max=1.0)


def _normalized_weights(first: float, second: float) -> tuple[float, float]:
    if first < 0 or second < 0:
        raise ValueError("weights must be non-negative")
    total = first + second
    if total <= 0:
        raise ValueError("at least one weight must be positive")
    return first / total, second / total


def _matching_objective_value(
    coefficients: torch.Tensor,
    row_indices: np.ndarray,
    column_indices: np.ndarray,
) -> float:
    selected = coefficients[
        torch.as_tensor(row_indices, device=coefficients.device),
        torch.as_tensor(column_indices, device=coefficients.device),
    ]
    return float(selected.sum().detach().cpu())


def _solve_assignment(
    coefficients: torch.Tensor,
    eligible: torch.Tensor,
    *,
    maximize: bool,
    objective_name: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    cost = coefficients.detach().cpu().numpy().copy()
    eligible_np = eligible.detach().cpu().numpy()
    finite_values = cost[eligible_np]
    if finite_values.size == 0:
        raise ValueError("eligible_pairs must contain at least one feasible pair")

    penalty = float(np.max(np.abs(finite_values)) + 1.0)
    penalty *= max(coefficients.shape) + 1
    if maximize:
        cost = -cost
    cost[~eligible_np] = penalty

    try:
        row_indices, column_indices = linear_sum_assignment(cost)
    except ValueError as error:
        raise ValueError(f"{objective_name} matching problem is infeasible") from error

    if row_indices.size != coefficients.shape[0]:
        raise ValueError(f"{objective_name} matching problem is infeasible")
    if not eligible_np[row_indices, column_indices].all():
        raise ValueError(f"{objective_name} matching problem is infeasible")

    value = _matching_objective_value(coefficients, row_indices, column_indices)
    return row_indices, column_indices, value


def _objective_bounds(
    coefficients: torch.Tensor,
    eligible: torch.Tensor,
    *,
    objective_name: str,
) -> tuple[float, float]:
    _, _, maximum = _solve_assignment(
        coefficients,
        eligible,
        maximize=True,
        objective_name=f"{objective_name} maximum",
    )
    _, _, minimum = _solve_assignment(
        coefficients,
        eligible,
        maximize=False,
        objective_name=f"{objective_name} minimum",
    )
    return maximum, minimum


def fermatean_fuzzy(
    side_a_preferences: torch.Tensor,
    side_b_preferences: torch.Tensor,
    eligible_pairs: torch.Tensor | None = None,
    side_a_weight: float = 0.5,
    side_b_weight: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute a fair one-to-one matching between two sides of a market.

    Args:
        side_a_preferences: Matrix of side-A preferences toward side-B agents
            with shape ``(n_a, n_b)``.
        side_b_preferences: Matrix of side-B preferences toward side-A agents
            with shape ``(n_b, n_a)``.
        eligible_pairs: Optional boolean matrix with shape ``(n_a, n_b)``.
            Ineligible pairs cannot be selected.
        side_a_weight: Non-negative weight for side A in the normalized
            multi-objective matching problem.
        side_b_weight: Non-negative weight for side B in the normalized
            multi-objective matching problem.

    Returns:
        A tuple ``(fair_scores, matched_pairs)``. ``fair_scores`` is the final
        objective coefficient matrix with shape ``(n_a, n_b)``. ``matched_pairs``
        is an integer tensor with shape ``(n_a, 2)`` where each row contains
        ``(side_a_index, side_b_index)``.
    """

    side_a = _as_float_matrix(side_a_preferences, name="side_a_preferences")
    device = side_a.device
    side_b = _as_float_matrix(side_b_preferences, name="side_b_preferences").to(
        device=device
    )

    n_side_a, n_side_b = side_a.shape
    if side_b.shape != (n_side_b, n_side_a):
        raise ValueError(
            "side_b_preferences must have shape "
            f"({n_side_b}, {n_side_a}), got {tuple(side_b.shape)}"
        )
    if n_side_a > n_side_b:
        raise ValueError(
            "one-to-one matching requires at least as many side-B agents as "
            f"side-A agents, got {n_side_a} and {n_side_b}"
        )

    eligible = _validate_eligible_pairs(
        eligible_pairs,
        side_a.shape,
        device=device,
    )
    if torch.any(torch.sum(eligible, dim=1) == 0):
        raise ValueError("each side-A agent must have at least one eligible pair")

    side_a_weight, side_b_weight = _normalized_weights(
        side_a_weight,
        side_b_weight,
    )

    side_b_aligned = side_b.T.contiguous()
    side_a_satisfaction = _normalize_valid(side_a, eligible)
    side_b_satisfaction = _normalize_valid(side_b_aligned, eligible)

    joint_satisfaction = 0.5 * (side_a_satisfaction + side_b_satisfaction)
    satisfaction_equity = 1.0 - torch.abs(side_a_satisfaction - side_b_satisfaction)
    willingness = 0.5 * (joint_satisfaction + satisfaction_equity)
    willingness = torch.where(eligible, willingness, torch.zeros_like(willingness))

    side_a_coefficients = side_a_satisfaction * willingness
    side_b_coefficients = side_b_satisfaction * willingness

    max_side_a, min_side_a = _objective_bounds(
        side_a_coefficients,
        eligible,
        objective_name="side-A objective",
    )
    max_side_b, min_side_b = _objective_bounds(
        side_b_coefficients,
        eligible,
        objective_name="side-B objective",
    )

    fair_scores = torch.zeros_like(side_a_coefficients)
    side_a_range = max_side_a - min_side_a
    side_b_range = max_side_b - min_side_b
    if side_a_range > _EPSILON:
        fair_scores = fair_scores + side_a_weight * side_a_coefficients / side_a_range
    if side_b_range > _EPSILON:
        fair_scores = fair_scores + side_b_weight * side_b_coefficients / side_b_range
    fair_scores = torch.where(
        eligible,
        fair_scores,
        torch.full_like(fair_scores, -torch.inf),
    )

    row_indices, column_indices, _ = _solve_assignment(
        fair_scores,
        eligible,
        maximize=True,
        objective_name="fair matching",
    )
    matched_pairs = torch.stack(
        (
            torch.as_tensor(row_indices, dtype=torch.long, device=device),
            torch.as_tensor(column_indices, dtype=torch.long, device=device),
        ),
        dim=1,
    )
    output_dtype = torch.float64
    if (
        isinstance(side_a_preferences, torch.Tensor)
        and side_a_preferences.is_floating_point()
    ):
        output_dtype = side_a_preferences.dtype
    return fair_scores.to(dtype=output_dtype), matched_pairs
