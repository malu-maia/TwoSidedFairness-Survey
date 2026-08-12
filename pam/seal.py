from __future__ import annotations

import heapq

import torch

NAME = "SEAL"


def _validate_bounds(
    scores: torch.Tensor,
    k: int,
    product_min: int | None,
    product_max: int | None,
) -> tuple[int, int]:
    if scores.ndim != 2:
        raise ValueError(
            f"scores must be a 2D user-item matrix, got shape {tuple(scores.shape)}"
        )

    n_users, n_items = scores.shape
    if n_users == 0 or n_items == 0:
        raise ValueError("scores must contain at least one user and one item")
    if not 1 <= k <= n_items:
        raise ValueError(
            f"k must be between 1 and the number of items ({n_items}), got {k}"
        )

    total_allocations = n_users * k
    lower = total_allocations // n_items if product_min is None else product_min
    upper = n_users if product_max is None else product_max

    if lower < 0:
        raise ValueError(f"product_min must be non-negative, got {lower}")
    if lower > n_users:
        raise ValueError(
            f"product_min must be at most the number of users ({n_users}), got {lower}"
        )
    if upper < lower:
        raise ValueError(
            f"product_max must be at least product_min, got {upper} < {lower}"
        )
    if upper > n_users:
        raise ValueError(
            f"product_max must be at most the number of users ({n_users}), got {upper}"
        )
    if lower * n_items > total_allocations:
        raise ValueError("product_min is infeasible for n_users, n_items, and k")
    if upper * n_items < total_allocations:
        raise ValueError("product_max is infeasible for n_users, n_items, and k")

    return lower, upper


def _ranked_items(scores: torch.Tensor) -> list[list[int]]:
    cpu_scores = scores.detach().cpu()
    return [
        sorted(
            range(cpu_scores.shape[1]),
            key=lambda item: (-float(cpu_scores[user, item]), item),
        )
        for user in range(cpu_scores.shape[0])
    ]


def _best_available_item(
    ranking: list[int],
    allocation: set[int],
    exposure: list[int],
    exposure_limit: int,
) -> int | None:
    for item in ranking:
        if item not in allocation and exposure[item] < exposure_limit:
            return item
    return None


def _repair_underexposed_products(
    scores: torch.Tensor,
    allocations: list[list[int]],
    allocation_sets: list[set[int]],
    exposure: list[int],
    product_min: int,
) -> None:
    if product_min == 0:
        return

    cpu_scores = scores.detach().cpu()
    n_items = cpu_scores.shape[1]

    for target_item in range(n_items):
        while exposure[target_item] < product_min:
            best_candidate: tuple[float, int, int, int] | None = None
            for user, user_allocation in enumerate(allocations):
                if target_item in allocation_sets[user]:
                    continue

                target_score = float(cpu_scores[user, target_item])
                for position, old_item in enumerate(user_allocation):
                    if exposure[old_item] <= product_min:
                        continue
                    utility_loss = float(cpu_scores[user, old_item]) - target_score
                    candidate = (utility_loss, user, position, old_item)
                    if best_candidate is None or candidate < best_candidate:
                        best_candidate = candidate

            if best_candidate is None:
                raise ValueError("SEAL could not satisfy product_min with greedy repair")

            _, user, position, old_item = best_candidate
            allocations[user][position] = target_item
            allocation_sets[user].remove(old_item)
            allocation_sets[user].add(target_item)
            exposure[old_item] -= 1
            exposure[target_item] += 1


def _sort_allocations_by_score(
    scores: torch.Tensor,
    allocations: list[list[int]],
) -> torch.Tensor:
    cpu_scores = scores.detach().cpu()
    sorted_allocations = [
        sorted(
            user_allocation,
            key=lambda item: (-float(cpu_scores[user, item]), item),
        )
        for user, user_allocation in enumerate(allocations)
    ]
    return torch.tensor(sorted_allocations, dtype=torch.long, device=scores.device)


def seal(
    scores: torch.Tensor,
    k: int,
    product_min: int | None = None,
    product_max: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run SEAL as a fixed-length top-k allocation method.

    SEAL is the sequential egalitarian heuristic from Gupta et al.,
    adapted to recommendation top-k by setting the user-side lower and upper
    cardinality bounds to ``L1 = L2 = k``.

    Args:
        scores: Two-dimensional user-item utility matrix.
        k: Number of unique items to allocate to each user.
        product_min: Minimum number of users each item should be allocated to.
            Defaults to ``floor(n_users * k / n_items)``.
        product_max: Maximum number of users each item may be allocated to.
            Defaults to ``n_users``.

    Returns:
        The unchanged score matrix and a top-k allocation tensor.
    """
    product_min, product_max = _validate_bounds(scores, k, product_min, product_max)

    n_users, n_items = scores.shape
    cpu_scores = scores.detach().cpu()
    rankings = _ranked_items(scores)
    allocations: list[list[int]] = [[] for _ in range(n_users)]
    allocation_sets: list[set[int]] = [set() for _ in range(n_users)]
    exposure = [0 for _ in range(n_items)]
    user_utilities = [0.0 for _ in range(n_users)]

    for _ in range(k):
        queue = [(user_utilities[user], user) for user in range(n_users)]
        heapq.heapify(queue)

        while queue:
            _, user = heapq.heappop(queue)
            if len(allocations[user]) >= k:
                continue

            item = _best_available_item(
                rankings[user],
                allocation_sets[user],
                exposure,
                product_min,
            )
            if item is None:
                item = _best_available_item(
                    rankings[user],
                    allocation_sets[user],
                    exposure,
                    product_max,
                )
            if item is None:
                raise ValueError("SEAL could not find enough feasible items")

            allocations[user].append(item)
            allocation_sets[user].add(item)
            exposure[item] += 1
            user_utilities[user] += float(cpu_scores[user, item])

    _repair_underexposed_products(
        scores,
        allocations,
        allocation_sets,
        exposure,
        product_min,
    )

    fair_topk = _sort_allocations_by_score(scores, allocations)
    return scores, fair_topk


def train(
    scores: torch.Tensor,
    k: int,
    product_min: int | None = None,
    product_max: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compatibility wrapper for :func:`seal`."""
    return seal(
        scores=scores,
        k=k,
        product_min=product_min,
        product_max=product_max,
    )
