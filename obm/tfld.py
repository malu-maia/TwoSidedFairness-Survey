import torch
import torch.nn.functional as F

from obm._welfare import get_b, get_u, get_v

NAME = 'Lorenz'

def psi(x: torch.Tensor, alpha: float, eta: float = 1e-9) -> torch.Tensor:
    """
    Implements the psi function from the paper's welfare definition.
    A small eta is added to prevent log(0) or division by zero.
    """
    x_safe = x + eta
    if alpha == 0:
        return torch.log(x_safe)
    elif alpha > 0:
        return torch.pow(x_safe, alpha)
    else: # alpha < 0
        return -torch.pow(x_safe, alpha)
    

def psi_prime(x: torch.Tensor, alpha: float, eta: float = 1e-9) -> torch.Tensor:
    """
    Implements the derivative of the psi function.
    """
    x_safe = x + eta
    if alpha == 0:
        return 1 / x_safe
    elif alpha > 0:
        return alpha * torch.pow(x_safe, alpha - 1)
    else: # alpha < 0
        return -alpha * torch.pow(x_safe, alpha - 1)
    

def phi(x, alpha, eta):
    return (1/2)*psi(x, alpha, eta)

def phi_prime(x, alpha, eta):
    return 1/2 * (psi_prime(x, alpha, eta))


def compute_welfare(user_utils: torch.Tensor, item_utils: torch.Tensor, lamb: float, alpha_1: float, alpha_2: float, eta: float) -> torch.Tensor:
    """
    Calculates the total welfare W_theta(u).
    """
    user_welfare = (1 - lamb) * psi(user_utils, alpha_1, eta).sum()
    item_welfare = lamb * psi(item_utils, alpha_2, eta).sum()
    return user_welfare + item_welfare


def tfld(
    scores: torch.Tensor,
    epochs: int = 1000,
    k: int = 10,
    alpha: list[float] | tuple[float, float] = (0.0, 0.0),
    lamb: float = 0.5,
    eta: float = 1e-9,
    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Run the two-sided Lorenz-dominance optimization.

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
  if len(alpha) != 2:
      raise ValueError(f"alpha must contain exactly two values, got {len(alpha)}")
  if not 0 <= lamb <= 1:
      raise ValueError(f"lamb must be between 0 and 1, got {lamb}")

  alpha_1, alpha_2 = alpha
  N, M = scores.size()
  S = scores.to(device)
  if not 1 <= k <= M:
      raise ValueError(f"k must be between 1 and the number of items ({M}), got {k}")
  # Exposure weights for ranks
  v_exposure_weights = get_b(k).to(device)

  print(f'Running with: {N} users, {M} items, Top-{k} recommendations')
  print(f'Parameters: lambda={lamb}, alpha=[{alpha_1}, {alpha_2}], epochs={epochs}')
  print("-" * 30)

  # Start with a deterministic ranking where each user gets the items they have the highest score for.
  _, sorted_indices = torch.sort(S, descending=True, dim=1)
  top_k_indices = sorted_indices[:, :k]
  # P_stochastic is the probability tensor. Initially, it's deterministic (0 or 1).
  P_stochastic = (
      F.one_hot(top_k_indices, num_classes=M)
      .permute(0, 2, 1)
      .to(torch.float16)
      .to(device)
  )

  welfare_store_vector = []

  # Frank-Wolfe
  for t in range(epochs):
      # Current user and item utilities
      user_utilities = get_u(S, P_stochastic, v_exposure_weights).to(device)
      item_utilities = get_v(P_stochastic, v_exposure_weights).to(device)

      # Gradient components
      phi_prime_u = phi_prime(user_utilities, alpha_1, eta) # Shape: (N,)
      phi_prime_i = phi_prime(item_utilities, alpha_2, eta) # Shape: (M,)

      # Computing the score matrix A for the linear subproblem.
      # A_ij = (1-lambda)*psi'(u_i)*mu_ij + lambda*psi'(u_j)
      term_user = (1 - lamb) * phi_prime_u.unsqueeze(1) * S # Broadcasts to (N, M)
      term_item = lamb * phi_prime_i.unsqueeze(0)           # Broadcasts to (1, M)
 
      A = term_user + term_item # Broadcasts to (N, M) + (N, M)

      # Finding the next ranking direction P_tilde by sorting based on scores A
      _, sorted_indices_tilde = torch.sort(A, descending=True, dim=1)
      top_k_indices_tilde = sorted_indices_tilde[:, :k]
      P_tilde = (
          F.one_hot(top_k_indices_tilde, num_classes=M)
          .permute(0, 2, 1)
          .to(torch.float16)
          .to(device)
      )

      # Updating the stochastic ranking tensor P
      gamma = 2 / (t + 2) # Step size
      P_stochastic = (1 - gamma) * P_stochastic + gamma * P_tilde

      if (t + 1) % 100 == 0:
        total_welfare = compute_welfare(user_utilities, item_utilities, lamb=lamb, alpha_1=alpha_1, alpha_2=alpha_2, eta=eta)
        welfare_store_vector.append(total_welfare)
        print(f'Epoch {t+1}/{epochs}: Welfare = {total_welfare.item():.4f}')

  print("-" * 30)
  print("Optimization finished.")

  return A, P_stochastic, user_utilities, item_utilities


def train(
    scores: torch.Tensor,
    epochs: int = 1000,
    k: int = 10,
    alpha: list[float] | tuple[float, float] = (0.0, 0.0),
    lamb: float = 0.5,
    eta: float = 1e-9,
    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compatibility wrapper for :func:`tfld`."""
    return tfld(
        scores=scores,
        epochs=epochs,
        k=k,
        alpha=alpha,
        lamb=lamb,
        eta=eta,
        device=device,
    )
