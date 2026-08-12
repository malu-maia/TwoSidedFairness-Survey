"""TFLD for sensitive groups (Do et al., NeurIPS 2021, Appendix B).

Individual TFLD (``obm/tfld.py``) maximizes one welfare term per user and one
per item::

    W_theta(u) = (1 - lamb) * sum_i psi(u_i, alpha_1)
               + lamb * sum_j psi(u_j, alpha_2)

Appendix B replaces each individual utility with its group total::

    u_s   = sum_{i in s} u_i          (user group s)
    u_c   = sum_{j in c} u_j          (item category c)
    W_gr  = (1 - lamb) * sum_s psi(u_s, alpha_1)
          + lamb * sum_c psi(u_c, alpha_2)

Because ``dW_gr/du_i = (1 - lamb) * psi'(u_{s(i)}, alpha_1)``, the Frank-Wolfe
linear subproblem is unchanged (Theorem 1: one sort per user). Only the
per-entity gradient weight changes, from the entity's own ``psi'`` to its
group's ``psi'`` broadcast back to members. Proposition 5 guarantees any
maximizer is ``(S, C)``-Lorenz efficient.
"""

import torch

from obm._welfare import get_b
from obm.tfld import psi, psi_prime

NAME = "Lorenz-Groups"


def _dense_group_index(groups: torch.Tensor, *, size: int, name: str) -> torch.Tensor:
    """Map arbitrary group labels onto contiguous ``0..G-1`` indices."""

    index = torch.as_tensor(groups).flatten()
    if index.numel() != size:
        raise ValueError(f"{name} must contain one label per entity, expected {size}, got {index.numel()}")
    return torch.unique(index, return_inverse=True)[1]


def _exposure_from_topk(
    top_k_indices: torch.Tensor,
    num_items: int,
    rank_weights: torch.Tensor,
) -> torch.Tensor:
    """Expected exposure of a deterministic top-k ranking, shape (N, M)."""

    exposure = torch.zeros(
        top_k_indices.shape[0],
        num_items,
        dtype=rank_weights.dtype,
        device=top_k_indices.device,
    )
    return exposure.scatter_(
        1, top_k_indices, rank_weights.expand(top_k_indices.shape[0], -1)
    )


def group_welfare(
    user_group_utils: torch.Tensor,
    item_group_utils: torch.Tensor,
    lamb: float,
    alpha_1: float,
    alpha_2: float,
    eta: float,
) -> torch.Tensor:
    """Calculate W_gr over group-aggregated utilities (Appendix B)."""

    user_welfare = (1 - lamb) * psi(user_group_utils, alpha_1, eta).sum()
    item_welfare = lamb * psi(item_group_utils, alpha_2, eta).sum()
    return user_welfare + item_welfare


def tfld_groups(
    scores: torch.Tensor,
    user_groups: torch.Tensor,
    item_groups: torch.Tensor,
    epochs: int = 1000,
    k: int = 10,
    alpha: list[float] | tuple[float, float] = (0.0, 0.0),
    lamb: float = 0.5,
    eta: float = 1e-9,
    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the two-sided Lorenz-dominance optimization over sensitive groups.

    Args:
        scores: Estimated preferences, shape ``(n_users, n_items)``.
        user_groups: One group label per user; any labels, any encoding.
        item_groups: One category label per item; any labels, any encoding.

    Returns:
        Expected exposure ``(n_users, n_items)``, the fair top-k
        ``(n_users, k)``, user-group utilities, and item-category exposures.
        No data files are read by this function.
    """
    if scores.ndim != 2:
        raise ValueError(
            f"scores must be a 2D user-item matrix, got shape {tuple(scores.shape)}"
        )
    if scores.numel() == 0:
        raise ValueError("scores must not be empty")
    if epochs < 1:
        raise ValueError(f"epochs must be at least 1, got {epochs}")
    if len(alpha) != 2:
        raise ValueError(f"alpha must contain exactly two values, got {len(alpha)}")
    if not 0 <= lamb <= 1:
        raise ValueError(f"lamb must be between 0 and 1, got {lamb}")

    alpha_1, alpha_2 = alpha
    # psi is concave, and the Lorenz guarantee of Proposition 5 holds, only on
    # (-inf, 1). tfld() omits this check; a group run is worth failing loudly.
    if alpha_1 >= 1 or alpha_2 >= 1:
        raise ValueError(f"alpha values must be below 1, got {(alpha_1, alpha_2)}")

    N, M = scores.size()
    if not 1 <= k <= M:
        raise ValueError(f"k must be between 1 and the number of items ({M}), got {k}")

    S = scores.to(device)
    user_index = _dense_group_index(user_groups, size=N, name="user_groups").to(device)
    item_index = _dense_group_index(item_groups, size=M, name="item_groups").to(device)
    n_user_groups = int(user_index.max().item()) + 1
    n_item_groups = int(item_index.max().item()) + 1

    # Exposure weights for ranks
    v_exposure_weights = get_b(k).to(device)

    print(f'Running with: {N} users, {M} items, Top-{k} recommendations')
    print(f'Groups: {n_user_groups} user groups, {n_item_groups} item categories')
    print(f'Parameters: lambda={lamb}, alpha=[{alpha_1}, {alpha_2}], epochs={epochs}')
    print("-" * 30)

    # ponytail: track expected exposure E_ij = sum_k P_ijk * v_k, shape (N, M),
    # instead of the (N, M, k) policy tfld() carries. Every quantity below
    # depends on P only through E, and the Frank-Wolfe update is linear, so the
    # recurrence is identical -- at k=150 this is ~150x less memory, and float32
    # avoids the fp16 step-size freeze logged against tfld.py.
    E = _exposure_from_topk(
        torch.topk(S, k, dim=1).indices, M, v_exposure_weights
    )

    # Frank-Wolfe
    for t in range(epochs):
        # Current individual utilities, then Appendix B group aggregation.
        user_utilities = (S * E).sum(dim=1)
        item_utilities = E.sum(dim=0)
        user_group_utils = torch.zeros(
            n_user_groups, dtype=E.dtype, device=device
        ).index_add_(0, user_index, user_utilities)
        item_group_utils = torch.zeros(
            n_item_groups, dtype=E.dtype, device=device
        ).index_add_(0, item_index, item_utilities)

        # Score matrix for the linear subproblem, with each entity carrying its
        # group's marginal welfare weight:
        # A_ij = (1-lambda)*psi'(u_s(i))*mu_ij + lambda*psi'(u_c(j))
        term_user = (1 - lamb) * psi_prime(user_group_utils, alpha_1, eta)[
            user_index
        ].unsqueeze(1) * S
        term_item = lamb * psi_prime(item_group_utils, alpha_2, eta)[
            item_index
        ].unsqueeze(0)
        A = term_user + term_item

        # Frank-Wolfe direction: the top-k of A per user (Theorem 1).
        E_tilde = _exposure_from_topk(
            torch.topk(A, k, dim=1).indices, M, v_exposure_weights
        )

        gamma = 2 / (t + 2)  # Step size
        E = (1 - gamma) * E + gamma * E_tilde

        if (t + 1) % 100 == 0:
            total_welfare = group_welfare(
                user_group_utils,
                item_group_utils,
                lamb=lamb,
                alpha_1=alpha_1,
                alpha_2=alpha_2,
                eta=eta,
            )
            print(f'Epoch {t+1}/{epochs}: Welfare = {total_welfare.item():.4f}')

    # Recomputed after the final update so the returned utilities describe the
    # returned E rather than the previous iterate.
    user_group_utils = torch.zeros(
        n_user_groups, dtype=E.dtype, device=device
    ).index_add_(0, user_index, (S * E).sum(dim=1))
    item_group_utils = torch.zeros(
        n_item_groups, dtype=E.dtype, device=device
    ).index_add_(0, item_index, E.sum(dim=0))

    print("-" * 30)
    print("Optimization finished.")

    return E, torch.topk(E, k, dim=1).indices, user_group_utils, item_group_utils
