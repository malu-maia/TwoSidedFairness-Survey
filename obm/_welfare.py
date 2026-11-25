"""Shared expected-exposure / utility helpers for the welfare-based
Frank-Wolfe methods (GGF and TFLD)."""

import numpy as np
import torch
import torch.nn.functional as F


def sorted_prefix(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Return the sort prefix, preserving the original tie order."""
    if k == scores.shape[1]:
        return scores.sort(dim=1, descending=True).indices
    values, indices = scores.topk(k + 1, dim=1)
    tied = (values[:, :-1] == values[:, 1:]).any(dim=1) | values.isnan().any(dim=1)
    prefix = indices[:, :k].contiguous()
    if tied.any():
        prefix[tied] = scores[tied].sort(dim=1, descending=True).indices[:, :k]
    return prefix


def ranking_distribution(
    rankings: torch.Tensor,
    n_items: int,
    discounts: torch.Tensor,
    *,
    exposure_only: bool,
) -> torch.Tensor:
    """Represent a deterministic ranking as a policy or its expected exposure."""
    if exposure_only:
        exposure = discounts.new_zeros((rankings.shape[0], n_items))
        return exposure.scatter_(1, rankings, discounts.expand_as(rankings))
    return F.one_hot(rankings, num_classes=n_items).permute(0, 2, 1).to(torch.float16)


def update_exposure(
    exposure: torch.Tensor, rankings: torch.Tensor, discounts: torch.Tensor, step: float,
) -> torch.Tensor:
    """Apply the Frank–Wolfe update directly to the expected exposure matrix."""
    exposure.mul_(1 - step)
    return exposure.scatter_add_(1, rankings, (step * discounts).expand_as(rankings))


def policy_utilities(
    scores: torch.Tensor, distribution: torch.Tensor, discounts: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute user and item utilities from either policy representation."""
    exposure = distribution
    if distribution.ndim == 3:
        exposure = (distribution * discounts.view(1, 1, -1)).sum(dim=2)
    return (scores * exposure).sum(dim=1), exposure.sum(dim=0)


def get_u(mu: torch.Tensor, P_stochastic: torch.Tensor, v_weights: torch.Tensor) -> torch.Tensor:
    """User utilities: u_i(P) = sum_{j,k} mu_ij * P_ijk * v_k. Shape (N,)."""
    expected_exposure = (P_stochastic * v_weights.view(1, 1, -1)).sum(dim=2)
    return (mu * expected_exposure).sum(dim=1)


def get_v(P_stochastic: torch.Tensor, v_weights: torch.Tensor) -> torch.Tensor:
    """Item exposures: u_j(P) = sum_{i,k} P_ijk * v_k. Shape (M,)."""
    expected_exposure = (P_stochastic * v_weights.view(1, 1, -1)).sum(dim=2)
    return expected_exposure.sum(dim=0)


def get_b(k: int) -> torch.Tensor:
    """Log-discounted exposure weight for each of the top-k ranks."""
    return torch.tensor([1.0 / np.log2(i + 2) for i in range(k)], dtype=torch.float32)
