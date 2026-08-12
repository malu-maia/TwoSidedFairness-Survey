import torch

from typing import List, Dict, Tuple

NAME = 'Fairrec'

def greedy_round_robin(
    scores: torch.Tensor,
    available_copies: Dict[int, int],
    total_items_to_allocate: int,
    feasible_products: Dict[int, List[int]],
    user_ordering: torch.Tensor
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]], Dict[int, int], int]:
    m, n = scores.size()
    allocation = {u: [] for u in range(m)}
    last_allocated_idx = m - 1  # Initialize to last user
    
    round_num = 0

    while total_items_to_allocate > 0:
        round_num += 1
        for i in range(m):
            u = int(user_ordering[i])
            
            # Get feasible products with available copies
            feasible = [p for p in feasible_products[u] if available_copies[p] > 0]
            if not feasible:
                if i > 0:
                    last_allocated_idx = i - 1
                else:
                    last_allocated_idx = m - 1
                return allocation, feasible_products, available_copies, last_allocated_idx
            
            # Allocate best available product
            p = feasible[torch.argmax(scores[u, feasible]).item()]
            
            allocation[u].append(p)
            feasible_products[u].remove(p)
            available_copies[p] -= 1
            total_items_to_allocate -= 1
            
            if total_items_to_allocate == 0:
                last_allocated_idx = i
                return allocation, feasible_products, available_copies, last_allocated_idx
    
    return allocation, feasible_products, available_copies, last_allocated_idx

def first_phase(
    user_ordering: torch.Tensor,
    scores: torch.Tensor,
    k: int
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]], Dict[int, int], int]:
    """Execute the first phase of FairRec algorithm."""
    m, n = scores.size()

    if k>n:#if k >= n:
        raise ValueError(f"k (recommendation size) must be less than n (number of products). Got k={k}, n={n}")
    if n > m * k:
        raise ValueError(f"n (number of products) must be <= m*k. Got n={n}, m*k={m*k}")
    
    feasible_products = {u: list(range(n)) for u in range(m)}
    
    # Calculate maximin share (MMS) for producers
    l = (m * k)// n
    available_copies = {p: l for p in range(n)}
    total_items = l * n
    
    allocation, feasible_products, available_copies, last_allocated = greedy_round_robin(
        scores, available_copies, total_items, feasible_products, user_ordering
    )
    
    return allocation, feasible_products, available_copies, last_allocated

def second_phase(
    phase1_allocation: Dict[int, List[int]],
    user_ordering: torch.Tensor,
    feasible_products: Dict[int, List[int]],
    scores: torch.Tensor,
    last_allocated_idx: int,
    k: int
) -> Dict[int, List[int]]:
    """Execute the second phase of FairRec algorithm to complete recommendations."""
    m, n = scores.size()
    allocation = {u: phase1_allocation[u].copy() for u in range(m)}
    
    # Check if we need to enter phase 2
    next_user = int(user_ordering[(last_allocated_idx + 1) % m])
    if len(allocation[next_user]) >= k:
        return allocation  # All users already have k recommendations
    
    # Prepare for phase 2
    new_ordering = torch.roll(user_ordering, -(last_allocated_idx + 1))
    available_copies = {p: m for p in range(n)}  # Unlimited copies in phase 2
    remaining_items = sum(max(0, k - len(allocation[u])) for u in range(m))

    phase2_allocation, _, _, _ = greedy_round_robin(
        scores, available_copies, remaining_items, feasible_products, new_ordering
    )

    # Merge allocations
    for u in range(m):
        needed = k - len(allocation[u])
        if needed > 0:
            allocation[u].extend(phase2_allocation[u][:needed])
    
    return allocation

def fairrec(scores: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run FairRec on a prepared user-item score matrix.

    Args:
        scores: Two-dimensional tensor with shape ``(n_users, n_items)``.
        k: Number of items to recommend to each user.

    Returns:
        A tuple containing the unchanged score matrix and a tensor of item
        indices with shape ``(n_users, k)``.
    """
    if scores.ndim != 2:
        raise ValueError(
            f"scores must be a 2D user-item matrix, got shape {tuple(scores.shape)}"
        )

    m, n = scores.size()
    if not 1 <= k <= n:
        raise ValueError(f"k must be between 1 and the number of items ({n}), got {k}")
    
    # Generate random user ordering
    user_ordering = torch.randperm(m)
    
    # Execute phase 1
    phase1_allocation, feasible_products, _, last_allocated = first_phase(
        user_ordering, scores, k
    )
    
    # Execute phase 2 if needed
    final_allocation = second_phase(
        phase1_allocation, user_ordering, feasible_products, scores, last_allocated, k
    )
    
    final_allocation = torch.tensor(list(final_allocation.values()), device=scores.device)
    return scores, final_allocation


def train(scores: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compatibility wrapper for :func:`fairrec`."""
    return fairrec(scores=scores, k=k)
