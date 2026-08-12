import math

import numpy as np
import torch
import torch.nn.functional as F

from scipy.optimize import isotonic_regression

from obm._welfare import get_b, get_u, get_v

NAME='Gini'


def get_w(n: int):
    """
    Returns the Gini weights w_i = (n - i + 1) / n, sorted in decreasing order.
    These weights give more importance to lower-ranked (worse-off) individuals.
    """
    # Equivalent to (n, n-1, ..., 1) / n
    return torch.arange(n, 0, -1, dtype=torch.float32) / n


def get_w_quantile(n: int, q: float, omega: float = 1.0) -> torch.Tensor:
    """GGF weights for the paper's Task 2 (Do & Usunier, SIGIR'22, Eq. 7).

    Eq. (7) gives the *Lorenz* weights w'_{ceil(qn)} = omega, w'_n = 1 - omega,
    all others zero. Eq. (5) recovers the GGF weights as w_i = sum_{j>=i} w'_j,
    i.e. 1 for the worst-off ceil(qn) individuals and 1 - omega for the rest.
    With omega = 1 the welfare term is exactly the cumulative utility of the
    worst-off q fraction, which is the metric these runs are scored on.
    """
    if not 0 < q <= 1:
        raise ValueError(f"q must be between 0 and 1, got {q}")
    if not 0 <= omega <= 1:
        raise ValueError(f"omega must be between 0 and 1, got {omega}")
    w = torch.full((n,), 1.0 - omega, dtype=torch.float32)
    w[: math.ceil(n * q)] = 1.0
    return w


def pav(z: torch.Tensor, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')) -> torch.Tensor:
    # z is sorted descending, so the isotonic fit must be non-increasing.
    # scipy returns a reversed view for increasing=False; copy it for torch.
    isotonic_func = isotonic_regression(z.cpu().numpy(), increasing=False)
    isotonic_z = torch.from_numpy(isotonic_func.x.copy()).to(z.dtype).to(device)
    return isotonic_z

def projection(z: torch.Tensor, w: torch.Tensor, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')) -> torch.Tensor:
    """
    Supergradient of the beta-smoothed GGF: the projection of ``-z`` onto the
    permutahedron of ``w``, i.e. a convex combination of permutations of ``w``
    that pairs the largest weight with the smallest entry of ``z``.
    """
    flipped_w = w.view(1, -1).flip([0,1]).squeeze(0).to(device)
    w_tilde = -flipped_w

    z_sorted, idx_sorted = torch.sort(z, descending=True)
    x = pav(z_sorted - w_tilde, device)
    inverse_sort = torch.argsort(idx_sorted)
    y = x[inverse_sort] - z

    return y

def loss_function(u: torch.Tensor, v: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, lamb: torch.float, beta: float=.1):
    u_sorted_ascending, _ = torch.sort(u, descending=False)
    g_u = (w1 * u_sorted_ascending).sum()

    # Calculate GGF for items, g_w2(v)
    v_sorted_ascending, _ = torch.sort(v, descending=False)
    g_v = (w2 * v_sorted_ascending).sum()

    # The full objective function F(P) to be MAXIMIZED
    F_P = (1 - lamb) * g_u + lamb * g_v

    # Optimizers perform minimization, so the loss is the negative of the objective.
    return -F_P

def prop5_beta(
    n: int, k: int, w1: torch.Tensor, w2: torch.Tensor, lamb: float
) -> float:
    """Smoothing constant beta_0 = 2 * D_P * b_1 / ||w|| (Do & Usunier, Prop. 5).

    ``D_P = sqrt(2nk)`` bounds the diameter of the policy space: two top-k
    selection matrices differ in at most two entries per rank, over k ranks and
    n users. ``b_1 = 1/log2(2) = 1``. Since ``h_w`` is ``||w||``-Lipschitz
    (Sec. 3.2.1), the two-sided objective is bounded by the mixed norm below.

    Assumes mu_ij in [0,1], which the prepared scores satisfy (they are
    per-user min-max normalized). beta_t = beta_0 / sqrt(t) is applied by the
    caller, so this is beta_0 only.
    """
    w_norm = float(
        (1 - lamb) * torch.linalg.vector_norm(w1)
        + lamb * torch.linalg.vector_norm(w2)
    )
    return 2 * math.sqrt(2 * n * k) / w_norm


def ggf(
    scores: torch.Tensor,
    beta: float | None = None,
    epochs: int = 5000,
    lamb: float = .5,
    k: int | None = None,
    user_weights: torch.Tensor | None = None,
    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the Generalized Gini fairness optimization on a score matrix.

    ``user_weights`` are the GGF weights w^1 over users, non-increasing with
    w^1_1 = 1. Defaults to ``ones(n)``, the paper's Task 1 instantiation
    (Eq. 6): utilitarian on users, Gini on item exposure. Pass
    :func:`get_w_quantile` for Task 2, which also puts fairness on users.

    ``beta`` defaults to :func:`prop5_beta`, computed from the problem size.
    A fixed value oversmooths or undersmooths depending on n, k and the
    weights; with a constant w^1 the user-side projection is beta-invariant,
    but nothing else is.

    Returns:
        Fair scores, stochastic ranking policy, user utilities, and item
        utilities. No data files are read by this function.
    """
    if scores.ndim != 2:
        raise ValueError(
            f"scores must be a 2D user-item matrix, got shape {tuple(scores.shape)}"
        )
    if scores.numel() == 0:
        raise ValueError("scores must not be empty")
    if epochs < 1:
        raise ValueError(f"epochs must be at least 1, got {epochs}")
    if not 0 <= lamb <= 1:
        raise ValueError(f"lamb must be between 0 and 1, got {lamb}")

    scores = scores.to(device)
    n, m = scores.size()
    if user_weights is not None and user_weights.shape != (n,):
        raise ValueError(
            f"user_weights must have shape ({n},), got {tuple(user_weights.shape)}"
        )
    resolved_k = m if k is None else k
    if not 1 <= resolved_k <= m:
        raise ValueError(
            f"k must be between 1 and the number of items ({m}), got {resolved_k}"
        )

    if beta is not None and beta <= 0:
        raise ValueError(f"beta must be positive, got {beta}")

    _, sorted_indices = torch.sort(scores, descending=True, dim=1)
    top_k_indices = sorted_indices[:, :resolved_k]
    P_stochastic = (
        F.one_hot(top_k_indices, num_classes=m)
        .permute(0, 2, 1)
        .to(torch.float16)
        .to(device)
    )
    
    w1 = (torch.ones(n) if user_weights is None else user_weights).to(device)
    w2 = get_w(m).to(device)

    beta_0 = prop5_beta(n, resolved_k, w1, w2, lamb) if beta is None else float(beta)
    beta_0 = torch.tensor(beta_0).to(device)

    b = get_b(resolved_k).to(device)

    losses = []

    for t in range(1, epochs+1):
        beta_t = beta_0 / np.sqrt(t)
        u = get_u(scores, P_stochastic, b).to(device)
        v = get_v(P_stochastic, b).to(device)
        y1 = projection(u/beta_t, w1, device).to(device)
        y2 = projection(v/beta_t, w2, device).to(device)

        #scores_tilde = (1 - lamb) * torch.einsum('i,ij->ij', y1, scores) + lamb * y2
        scores_tilde = (1 - lamb) * y1.view(-1, 1) * scores + lamb * y2.view(1, -1)

        _, sorted_indices = torch.sort(scores_tilde, descending=True, dim=1)
        top_k_indices = sorted_indices[:, :resolved_k]
    
        Q_t = (
            F.one_hot(top_k_indices, num_classes=m)
            .permute(0, 2, 1)
            .to(torch.float16)
            .to(device)
        )
        P_stochastic = (1 - (2/(t+2))) * P_stochastic + (2/(t+2)) * Q_t

        loss = loss_function(u, v, w1, w2, lamb)
        losses.append(-loss)

        if (t+1)%100==0:
            print(f'Epoch {t+1}/{epochs}: Loss {-loss}')

    return scores_tilde, P_stochastic, u, v


def train(
    scores: torch.Tensor,
    beta: float | None = None,
    epochs: int = 5000,
    lamb: float = .5,
    k: int | None = None,
    user_weights: torch.Tensor | None = None,
    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compatibility wrapper for :func:`ggf`."""
    return ggf(
        scores=scores,
        beta=beta,
        epochs=epochs,
        lamb=lamb,
        k=k,
        user_weights=user_weights,
        device=device,
    )
