"""Shared expected-exposure / utility helpers for the welfare-based
Frank-Wolfe methods (GGF and TFLD)."""

import numpy as np
import torch


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
