import torch
import torch.nn.functional as F
import numpy as np

from scipy.optimize import linear_sum_assignment
from typing import Literal

import math

NAME = 'FR'

# --- Helper Functions based on the Paper ---
def compute_exposure_vector(k: int, type: Literal['log', 'inv'] = 'log') -> torch.Tensor:
    """
    Computes the examination probability vector v(k).
    - 'log': v(k) = 1 / log2(k+1)
    - 'inv': v(k) = 1 / k
    """
    if k == 0:
        return torch.tensor([])
    if type == 'log':
        return torch.tensor([1/math.log2(i+1) for i in range(1, k+1)])
    elif type == 'inv':
        return torch.tensor([1/i for i in range(1, k+1)])
    else:
        raise ValueError("Choose inv or log as the exposure type.")

# base of A computation for test
def get_A(scores: torch.Tensor, n_users: torch.int, m_items: torch.int, k: torch.int, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
   return scores[:n_users, n_users:].unsqueeze(2).expand(n_users, m_items, k).to(device)

# base of B computation for test
def get_B(scores: torch.Tensor, n_users: torch.int, m_items: torch.int, k: torch.int, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
   return scores[n_users:, :n_users].unsqueeze(2).expand(m_items, n_users, k).to(device)

def a_chooses_b(scores: torch.tensor, A: torch.tensor, v: torch.tensor, n_users: torch.int) -> torch.Tensor:
    """
    scores: square matrix with preference estimatives of users in A about items in B and items in B about users in A
    v: exposure vector 
    n_users: we make operarions
    """
    inner_sum = (v.view(1,1,-1) * A).sum(dim=2)
    return scores[:n_users, n_users:] * inner_sum

def b_chooses_a(scores: torch.tensor, B: torch.tensor, v: torch.tensor, n_users: torch.int) -> torch.Tensor:
    """
    scores: square matrix with preference estimatives of users in A about items in B and items in B about users in A
    v: exposure vector 
    n_users: we make operarions
    """
    inner_sum = (v.view(1,1,-1) * B).sum(dim=2)
    return scores[n_users:, :n_users] * inner_sum

def prob_a_b_match(scores: torch.Tensor, A: torch.Tensor, B: torch.Tensor, n_users: torch.int, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
   m_items = scores.shape[0] - n_users

   P_a_chooses_b = a_chooses_b(scores, A, compute_exposure_vector(k=m_items, type='log').to(device), n_users)
   P_b_chooses_a = b_chooses_a(scores, B, compute_exposure_vector(k=n_users, type='log').to(device), n_users)

   return P_a_chooses_b * P_b_chooses_a.T


def get_utility_of_A(prob_matches: torch.Tensor) -> torch.Tensor:
   """
   prob_matches: matrix of size (n_users, m_items) that contain the probability of two users, one at each side, to match. 
   U_i = sum of prob_matches at dimension 1.
   
   Returns a tensor of size (1, n_users)
   """
   U = prob_matches.sum(dim=1)
   return U

def get_utility_of_B(prob_matches: torch.Tensor) -> torch.Tensor:
   V = prob_matches.sum(dim=0)
   return V


def compute_welfare(U: torch.Tensor) -> torch.Tensor:
    """
    Calculates the total welfare which is the sum of the utilities of all users, 
    that is equivalent to the sum of utility of all items.

    Returns a number which is the sum of the matches probabilities for all users or all items
    """
    SW = U.sum(dim=0)
    return SW


def compute_sw_gradient_A(p_ij: torch.Tensor, B_jil: torch.Tensor,v_k: torch.Tensor,v_l: torch.Tensor) -> torch.Tensor:
    """
    Computes the gradient of the Social Welfare (SW) function with respect to
    the recommendation matrices A.
    """
    grad_A = torch.einsum('ij,k,l,jil->ijk', p_ij, v_k, v_l, B_jil)
    return grad_A

def compute_sw_gradient_B(p_ij: torch.Tensor, A_ijk: torch.Tensor,v_k: torch.Tensor,v_l: torch.Tensor) -> torch.Tensor:
    grad_B = torch.einsum('ij,k,l,ijk->jil', p_ij, v_k, v_l, A_ijk)
    return grad_B

def compute_NSW(U: torch.Tensor) -> torch.Tensor:
    """
    Computes the Nash Social Welfare (NSW).
    Formula: NSW_1 = product_i U_i 
    """
    return torch.prod(U)

# --- Gradient Functions ---

def compute_sw_gradient_A(p_ij: torch.Tensor, B: torch.Tensor, v_k: torch.Tensor, v_l: torch.Tensor) -> torch.Tensor:
    """
    Computes the gradient of the Social Welfare (SW) function w.r.t. A.
    Formula: ∂SW/∂A_i(j,k) = p_ij * v_k * sum_l(v_l * B_j(i,l))
    """
    grad_A = torch.einsum('ij,k,l,jil->ijk', p_ij, v_k, v_l, B)
    return grad_A

def compute_sw_gradient_B(p_ij: torch.Tensor, A: torch.Tensor, v_k: torch.Tensor, v_l: torch.Tensor) -> torch.Tensor:
    """
    Computes the gradient of the Social Welfare (SW) function w.r.t. B.
    Formula p_ij * v_l * sum_k(v_k * A_i(j,k))
    """
    grad_B = torch.einsum('ij,k,l,ijk->jil', p_ij, v_k, v_l, A)
    return grad_B

def compute_nsw_gradient_A(p_ij: torch.Tensor, B: torch.Tensor, v_k: torch.Tensor, v_l: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Computes the gradient of the log Nash Social Welfare (log NSW_2) function w.r.t. A.
    This is used to find an NSW-best recommendation for the right-side agents. 
    Formula: (1/V_j) * p_ij * v_k * sum_l(v_l * B_j(i,l))
    """
    # We add a small epsilon to V for numerical stability.
    #print(f'P size, B size: {p_ij.size()}, {B.size()}')
    grad_A = torch.einsum('ij,k,l,j,jil->ijk', p_ij, v_k, v_l, 1 / (V + 1e-9), B)
    return grad_A

def compute_nsw_gradient_B(p_ij: torch.Tensor, A: torch.Tensor, v_k: torch.Tensor, v_l: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    """
    Computes the gradient of the log Nash Social Welfare (log NSW_1) function w.r.t. B.
    This is used to find an NSW-best recommendation for the left-side agents. 
    Formula: (1/U_i) * p_ij * v_l * sum_k(v_k * A_i(j,k))
    """
    grad_B = torch.einsum('ij,k,l,i,ijk->jil', p_ij, v_k, v_l, 1 / (U + 1e-9), A)
    return grad_B

# %% Assignment exact algorithm
def hungarian_algorithm(grad):
    """
    cost: grad
    """
    cost = grad.cpu().numpy()
    assignment = torch.zeros_like(grad)
    for i in range(grad.size(0)):
        row_ind, col_ind = linear_sum_assignment(cost[i, :, :], maximize=True)
        assignment[i, row_ind, col_ind] = 1

    return assignment.to(grad.device)

# Envy metric
def compute_envy(S, A, B, v_a, v_b):
    """
    Computes the number of envious pairs on both sides of the market.
    """
    n, m = v_b.size(0), v_a.size(0)
    prob_matches = prob_a_b_match(S, A, B, n)
    U_actual = get_utility_of_A(prob_matches)
    V_actual = get_utility_of_B(prob_matches)
    prob_a_to_b = a_chooses_b(S, A, v_a, n)
    prob_b_to_a = b_chooses_a(S, B, v_b, n)
    expected_exposure_A = torch.einsum('ijk,k->ij', A, v_a)
    expected_exposure_B = torch.einsum('ijk,k->ij', B, v_b)

    p1_hat = S[:n, n:]
    p2_hat = S[n:, :n]

    left_side_envies = 0
    for i in range(n):
        for i_prime in range(n):
            if i == i_prime: 
                continue
            hypo_prob_b_applies = p2_hat[:, i] * expected_exposure_B[:, i_prime]
            hypo_match_prob_for_i = prob_a_to_b[i, :] * hypo_prob_b_applies
            if U_actual[i] < hypo_match_prob_for_i.sum():
                left_side_envies += 1
    right_side_envies = 0
    for j in range(m):
        for j_prime in range(m):
            if j == j_prime: 
                continue
            hypo_prob_a_applies = p1_hat[:, j] * expected_exposure_A[:, j_prime]
            prob_b_j_applies = prob_b_to_a[j, :]
            hypo_match_prob_for_j = prob_b_j_applies * hypo_prob_a_applies
            if V_actual[j] < hypo_match_prob_for_j.sum():
                right_side_envies += 1
    return left_side_envies, right_side_envies
    

# %%
# --- Frank-Wolfe Algorithm ---
def fr(
    scores: torch.Tensor,
    n_users: int,
    epochs: int = 200,
    welfare_function: Literal['SW', 'NSW'] = 'SW',
    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ),
    k: int = 10,
) -> (
    tuple[torch.Tensor, list[float], torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
):
    """Run Fair Reciprocal optimization on a two-sided square score matrix.

    ``scores`` contains both directions of preference and must have shape
    ``(n_users + n_items, n_users + n_items)``. For ``SW`` this returns
    ``(fair_scores, welfare_history, user_utilities)``; for ``NSW`` it returns
    ``(fair_scores, stochastic_policy, user_utilities, item_utilities)``.
    """
    if scores.ndim != 2 or scores.size(0) != scores.size(1):
        raise ValueError(
            "scores must be a square 2D matrix containing both preference directions"
        )
    if not 0 < n_users < scores.size(0):
        raise ValueError(
            f"n_users must be between 1 and {scores.size(0) - 1}, got {n_users}"
        )
    if epochs < 1:
        raise ValueError(f"epochs must be at least 1, got {epochs}")
    if welfare_function not in {"SW", "NSW"}:
        raise ValueError("welfare_function must be 'SW' or 'NSW'")

    # Initialization
    S = scores
    N, I = n_users, S.size(0)-n_users


    A = get_A(S, N, I, I).to(torch.float16).to(device)
    B = get_B(S, N, I, N).to(torch.float16).to(device)

    A_stochastic = hungarian_algorithm(A).to(torch.float16).to(device)
    B_stochastic = hungarian_algorithm(B).to(torch.float16).to(device)

    prob_matches = prob_a_b_match(S, A_stochastic, B_stochastic, n_users).to(torch.float16)

    #if welfare_function == 'SW':
        # U is equivalent to V which is equivalent to prob_matches.sum()
    U = get_utility_of_A(prob_matches).to(torch.float16).to(device)

    print(f'Welfare: {compute_welfare(U)}')

    # exposure vector for users searching for items
    v_a = compute_exposure_vector(k=I, type='log').to(torch.float16).to(device)
    # exposure vector for items searching for users
    v_b = compute_exposure_vector(k=N, type='log').to(torch.float16).to(device)

    A_envies, B_envies = compute_envy(S, A_stochastic, B_stochastic, v_a, v_b)
    print(f'Envies on A: {A_envies} ({100*A_envies/(N**2)}%)\
          \nEnvies on B: {B_envies} ({100*B_envies/(I**2)}%)')

    lr = 0.1

    welfare_history = []

    for t in range(epochs):
        #lr = 2 / (t + 2) # Step size
        if welfare_function == 'SW':
            A_grad = compute_sw_gradient_A(prob_matches, B_stochastic, v_a, v_b).to(torch.float16)
        else: # NSW
            V = get_utility_of_B(prob_matches) #prob_matches_pre_A)
            A_grad = compute_nsw_gradient_A(prob_matches, B_stochastic, v_a, v_b, V).to(torch.float16).to(device)

        A_tilde = hungarian_algorithm(A_grad).to(torch.float16).to(device)
        A_stochastic = (1 - lr) * A_stochastic + lr * A_tilde

        # --- Update B (Recommendation for right-side users) ---
        if welfare_function == 'SW':
            B_grad = compute_sw_gradient_B(prob_matches, A_stochastic, v_a, v_b).to(torch.float16)
        else: # NSW
            # updating prob of maches considering the A_stochastic that were just updated
            prob_matches = prob_a_b_match(S, A_stochastic, B_stochastic, N).to(torch.float16)
            U = get_utility_of_A(prob_matches).to(torch.float16).to(device)
            B_grad = compute_nsw_gradient_B(prob_matches, A_stochastic, v_a, v_b, U).to(torch.float16).to(device)

        B_tilde = hungarian_algorithm(B_grad).to(torch.float16).to(device)
        B_stochastic = (1 - lr) * B_stochastic + lr * B_tilde

        # --- Logging ---
        # updating prob of maches considering the A_stochastic and B_stochastic that were just updated
        prob_matches = prob_a_b_match(S, A_stochastic, B_stochastic, N).to(torch.float16)
        U = get_utility_of_A(prob_matches).to(torch.float16)
        if welfare_function == 'SW':
            welfare = compute_welfare(U)
        elif welfare_function == 'NSW':
            welfare = compute_NSW(U)

        welfare_history.append(welfare.item())

        if t%10 == 0:
            print(f'Época: {t}      Welfare: {welfare}')

    A_envies, B_envies = compute_envy(S, A_stochastic, B_stochastic, v_a, v_b)
    print(f'Envies on A: {A_envies} ({100*A_envies/(N**2)}%)\
          \nEnvies on B: {B_envies} ({100*B_envies/(I**2)}%)')

    if welfare_function == 'SW':
        return prob_matches, welfare_history, U
    print(f'size probability matches: {prob_matches.size()}')
    return prob_matches, A_stochastic, U, V


def train(
    scores: torch.Tensor,
    n_users: int,
    epochs: int = 200,
    welfare_function: Literal['SW', 'NSW'] = 'SW',
    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ),
    k: int = 10,
) -> (
    tuple[torch.Tensor, list[float], torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
):
    """Compatibility wrapper for :func:`fr`."""
    return fr(
        scores=scores,
        n_users=n_users,
        epochs=epochs,
        welfare_function=welfare_function,
        device=device,
        k=k,
    )
