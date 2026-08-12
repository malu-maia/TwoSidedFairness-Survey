from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


NAME = "Ada2Fair"


class _MatrixFactorization(nn.Module):
    """Explicit-feedback matrix-factorization recommender used by Ada2Fair."""

    def __init__(self, n_users: int, n_items: int, embedding_dim: int) -> None:
        super().__init__()
        self.user_embeddings = nn.Embedding(n_users, embedding_dim)
        self.item_embeddings = nn.Embedding(n_items, embedding_dim)
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)

    def pair_scores(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
    ) -> torch.Tensor:
        return (self.user_embeddings(users) * self.item_embeddings(items)).sum(dim=1)

    def full_scores(self) -> torch.Tensor:
        return self.user_embeddings.weight @ self.item_embeddings.weight.T


class _AdaptiveWeighting(nn.Module):
    """Adaptive generator and two fairness-aware adapters from Eqs. 7 and 8."""

    def __init__(
        self,
        n_items: int,
        generator_hidden_dim: int,
        adapter_hidden_dim: int,
    ) -> None:
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(n_items, generator_hidden_dim),
            nn.Tanh(),
            nn.Linear(generator_hidden_dim, n_items),
            nn.Sigmoid(),
        )
        self.provider_adapter = nn.Sequential(
            nn.Linear(n_items, adapter_hidden_dim),
            nn.Tanh(),
        )
        self.customer_adapter = nn.Sequential(
            nn.Linear(n_items, adapter_hidden_dim),
            nn.Tanh(),
        )

    def forward(self, rating_inputs: torch.Tensor) -> torch.Tensor:
        return self.generator(rating_inputs)

    def fairness_loss(
        self,
        rating_inputs: torch.Tensor,
        observed: torch.Tensor,
        provider_targets: torch.Tensor,
        customer_targets: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        generated = self.generator(rating_inputs) * observed
        provider_hidden = self.provider_adapter(generated)
        customer_hidden = self.customer_adapter(generated)
        provider_target_hidden = self.provider_adapter(provider_targets)
        customer_target_hidden = self.customer_adapter(customer_targets)
        provider_loss = F.mse_loss(provider_hidden, provider_target_hidden)
        customer_loss = F.mse_loss(customer_hidden, customer_target_hidden)
        return (1 - alpha) * provider_loss + alpha * customer_loss


def _validate_inputs(
    interactions: torch.Tensor,
    item_provider_mapping: torch.Tensor,
    k: int,
    epochs: int,
    weight_epochs: int,
    embedding_dim: int,
    generator_hidden_dim: int,
    adapter_hidden_dim: int,
    batch_size: int,
    learning_rate: float,
    alpha: float,
    provider_eta: float,
    user_eta: float,
    delta: float,
    weight_topk: int,
    eligible_items: torch.Tensor | None,
    exclude_seen: bool,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if interactions.ndim != 2:
        raise ValueError(
            "interactions must be a 2D user-item matrix, "
            f"got shape {tuple(interactions.shape)}"
        )
    if interactions.numel() == 0:
        raise ValueError("interactions must not be empty")
    if interactions.is_complex():
        raise ValueError("interactions must contain real values")

    values = interactions.to(device=device, dtype=torch.float32)
    if not torch.isfinite(values).all():
        raise ValueError("interactions must contain only finite values")
    if torch.any(values < 0):
        raise ValueError("interactions must be non-negative")
    observed = values > 0
    n_users, n_items = values.shape
    if torch.any(observed.sum(dim=1) == 0):
        raise ValueError("each user must have at least one observed rating")
    if not 1 <= k <= n_items:
        raise ValueError(
            f"k must be between 1 and the number of items ({n_items}), got {k}"
        )

    if item_provider_mapping.ndim != 1 or item_provider_mapping.numel() != n_items:
        raise ValueError(
            "item_provider_mapping must be a 1D tensor with one provider ID per item"
        )
    if item_provider_mapping.is_complex():
        raise ValueError("item_provider_mapping must contain integer provider IDs")
    providers = item_provider_mapping.to(device=device)
    if providers.is_floating_point() and not torch.equal(providers, providers.round()):
        raise ValueError("item_provider_mapping must contain integer provider IDs")
    _, provider_indices = torch.unique(
        providers.to(torch.long),
        sorted=True,
        return_inverse=True,
    )

    integer_parameters = {
        "epochs": epochs,
        "weight_epochs": weight_epochs,
        "embedding_dim": embedding_dim,
        "generator_hidden_dim": generator_hidden_dim,
        "adapter_hidden_dim": adapter_hidden_dim,
        "batch_size": batch_size,
        "weight_topk": weight_topk,
    }
    for name, value in integer_parameters.items():
        if value < 1:
            raise ValueError(f"{name} must be at least 1, got {value}")

    real_parameters = {
        "learning_rate": learning_rate,
        "alpha": alpha,
        "provider_eta": provider_eta,
        "user_eta": user_eta,
        "delta": delta,
    }
    for name, value in real_parameters.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    if provider_eta < 0:
        raise ValueError("provider_eta must be non-negative")
    if user_eta < 0:
        raise ValueError("user_eta must be non-negative")
    if delta <= 0:
        raise ValueError("delta must be positive")

    if eligible_items is None:
        eligible = torch.ones_like(observed)
    else:
        if eligible_items.shape != interactions.shape:
            raise ValueError("eligible_items must have the same shape as interactions")
        eligible = eligible_items.to(device=device, dtype=torch.bool)
    if torch.any(eligible.sum(dim=1) == 0):
        raise ValueError("each user must have at least one eligible item")

    recommendation_candidates = eligible & ~observed if exclude_seen else eligible
    if torch.any(recommendation_candidates.sum(dim=1) < k):
        raise ValueError("each user must have at least k recommendation candidates")

    resolved_weight_topk = min(weight_topk, int(eligible.sum(dim=1).min().item()))
    return values, observed, provider_indices.to(torch.long), resolved_weight_topk


def _discounts(k: int, *, device: torch.device) -> torch.Tensor:
    ranks = torch.arange(2, k + 2, device=device, dtype=torch.float32)
    return 1 / torch.log2(ranks)


def _normalize_over_interactions(
    weights: torch.Tensor,
    observed: torch.Tensor,
) -> torch.Tensor:
    observed_weights = weights[observed]
    mean = observed_weights.mean()
    if not torch.isfinite(mean) or mean <= 0:
        return torch.ones_like(weights)
    return weights / mean


def _fairness_targets(
    scores: torch.Tensor,
    normalized_ratings: torch.Tensor,
    observed: torch.Tensor,
    provider_indices: torch.Tensor,
    eligible: torch.Tensor,
    weight_topk: int,
    provider_eta: float,
    user_eta: float,
    delta: float,
    exclude_seen: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate provider and customer weights from the current rankings."""

    # Use the same candidate set that is actually served (matches the recommended
    # list L_u in the paper): drop already-observed items when exclude_seen.
    candidates = eligible & ~observed if exclude_seen else eligible
    usable_topk = min(weight_topk, int(candidates.sum(dim=1).min().item()))
    masked_scores = scores.masked_fill(~candidates, -torch.inf)
    rankings = torch.topk(masked_scores, k=usable_topk, dim=1).indices
    discounts = _discounts(usable_topk, device=scores.device)

    item_exposure = torch.zeros(
        scores.size(1),
        device=scores.device,
        dtype=scores.dtype,
    )
    item_exposure.scatter_add_(
        0,
        rankings.reshape(-1),
        discounts.repeat(scores.size(0)),
    )
    n_providers = int(provider_indices.max().item()) + 1
    provider_exposure = torch.zeros(
        n_providers,
        device=scores.device,
        dtype=scores.dtype,
    )
    provider_exposure.scatter_add_(0, provider_indices, item_exposure)
    provider_sizes = torch.bincount(
        provider_indices,
        minlength=n_providers,
    ).to(device=scores.device, dtype=scores.dtype)
    provider_exposure = provider_exposure / provider_sizes.clamp_min(1)
    provider_item_weights = 1 / (
        provider_exposure.pow(provider_eta) + delta
    )[provider_indices]
    provider_matrix = provider_item_weights.unsqueeze(0).expand_as(scores)
    provider_matrix = _normalize_over_interactions(provider_matrix, observed)

    ranked_ratings = torch.gather(normalized_ratings, 1, rankings)
    user_utility = (ranked_ratings * discounts).sum(dim=1)
    # Paper Eq. 6 normalizes by the recommended-list size |L_u| (constant), not the
    # per-user observed-rating count.
    user_utility = user_utility / rankings.size(1)
    customer_weights = 1 / (user_utility.pow(user_eta) + delta)
    customer_matrix = customer_weights.unsqueeze(1).expand_as(scores)
    customer_matrix = _normalize_over_interactions(customer_matrix, observed)
    return provider_matrix * observed, customer_matrix * observed


def _sample_graded_negatives(
    values: torch.Tensor,
    users: torch.Tensor,
    positives: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample one item per pair with a strictly lower rating (unrated counts as 0).

    ponytail: uniform draw + one resample round, leftovers masked out of the
    loss; sparsity makes invalid draws rare.
    """

    n_items = values.shape[1]
    positive_ratings = values[users, positives]
    negatives = torch.randint(
        n_items, (users.size(0),), generator=generator
    ).to(values.device)
    invalid = values[users, negatives] >= positive_ratings
    if invalid.any():
        negatives[invalid] = torch.randint(
            n_items, (int(invalid.sum()),), generator=generator
        ).to(values.device)
    valid = values[users, negatives] < positive_ratings
    return negatives, valid


def _weighted_graded_bpr_loss(
    score_gaps: torch.Tensor,
    weights: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    masked_weights = weights * valid
    return -(masked_weights * F.logsigmoid(score_gaps)).sum() / masked_weights.sum().clamp_min(
        torch.finfo(score_gaps.dtype).eps
    )


def ada2fair(
    interactions: torch.Tensor,
    item_provider_mapping: torch.Tensor,
    k: int = 10,
    *,
    epochs: int = 100,
    weight_epochs: int = 10,
    embedding_dim: int = 64,
    generator_hidden_dim: int = 32,
    adapter_hidden_dim: int = 16,
    batch_size: int = 4096,
    learning_rate: float = 1e-3,
    alpha: float = 0.5,
    provider_eta: float = 2.0,
    user_eta: float = 1.0,
    delta: float = 1e-7,
    weight_topk: int = 100,
    seed: int = 42,
    eligible_items: torch.Tensor | None = None,
    exclude_seen: bool = True,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Train Ada2Fair with adaptive weights and a graded-pair BPR recommender.

    Positive entries in ``interactions`` are treated as observed ratings, while
    zeros represent missing values. The recommender minimizes a weighted
    graded-pair BPR loss: for each observed (user, item) pair a lower-rated
    item is sampled (unrated counts as rating 0) and the pair is weighted by
    the adaptive generator's weight for the positive interaction.
    Provider and customer fairness targets are recomputed from the current
    recommender rankings during every epoch. The adaptive generator learns
    from those targets, then its per-interaction weights scale pair losses.
    """

    if seed < 0:
        raise ValueError("seed must be non-negative")
    resolved_device = (
        interactions.device if device is None else torch.device(device)
    )
    values, observed, provider_indices, resolved_weight_topk = _validate_inputs(
        interactions,
        item_provider_mapping,
        k,
        epochs,
        weight_epochs,
        embedding_dim,
        generator_hidden_dim,
        adapter_hidden_dim,
        batch_size,
        learning_rate,
        alpha,
        provider_eta,
        user_eta,
        delta,
        weight_topk,
        eligible_items,
        exclude_seen,
        resolved_device,
    )
    eligible = (
        torch.ones_like(observed)
        if eligible_items is None
        else eligible_items.to(device=resolved_device, dtype=torch.bool)
    )
    n_users, n_items = values.shape
    rating_scale = values[observed].max()
    normalized_ratings = values / rating_scale

    rng_devices = (
        [
            resolved_device.index
            if resolved_device.index is not None
            else torch.cuda.current_device()
        ]
        if resolved_device.type == "cuda"
        else []
    )
    with torch.random.fork_rng(devices=rng_devices):
        torch.manual_seed(seed)
        if resolved_device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        recommender = _MatrixFactorization(
            n_users,
            n_items,
            embedding_dim,
        ).to(resolved_device)
        weighting = _AdaptiveWeighting(
            n_items,
            generator_hidden_dim,
            adapter_hidden_dim,
        ).to(resolved_device)

    recommender_optimizer = torch.optim.Adam(
        recommender.parameters(),
        lr=learning_rate,
    )
    weighting_optimizer = torch.optim.Adam(
        weighting.parameters(),
        lr=learning_rate,
    )
    cpu_generator = torch.Generator(device="cpu")
    cpu_generator.manual_seed(seed)
    observed_pairs = torch.nonzero(observed, as_tuple=False)

    for _ in range(epochs):
        recommender.eval()
        with torch.no_grad():
            current_scores = recommender.full_scores()
            provider_targets, customer_targets = _fairness_targets(
                current_scores,
                normalized_ratings,
                observed,
                provider_indices,
                eligible,
                resolved_weight_topk,
                provider_eta,
                user_eta,
                delta,
                exclude_seen,
            )

        weighting.train()
        for _ in range(weight_epochs):
            order = torch.randperm(n_users, generator=cpu_generator)
            for start in range(0, n_users, batch_size):
                users = order[start : start + batch_size].to(resolved_device)
                batch_ratings = normalized_ratings[users]
                batch_observed = observed[users].to(values.dtype)
                weighting_optimizer.zero_grad()
                fairness_loss = weighting.fairness_loss(
                    batch_ratings,
                    batch_observed,
                    provider_targets[users],
                    customer_targets[users],
                    alpha,
                )
                fairness_loss.backward()
                weighting_optimizer.step()

        weighting.eval()
        with torch.no_grad():
            generated_weights = weighting(normalized_ratings)
            generated_weights = _normalize_over_interactions(
                generated_weights,
                observed,
            )

        recommender.train()
        order = torch.randperm(observed_pairs.size(0), generator=cpu_generator)
        for start in range(0, observed_pairs.size(0), batch_size):
            pairs = observed_pairs[order[start : start + batch_size].to(
                observed_pairs.device
            )]
            users = pairs[:, 0]
            items = pairs[:, 1]
            negatives, valid = _sample_graded_negatives(
                values, users, items, cpu_generator
            )
            recommender_optimizer.zero_grad()
            score_gaps = recommender.pair_scores(users, items) - recommender.pair_scores(
                users, negatives
            )
            loss = _weighted_graded_bpr_loss(
                score_gaps,
                generated_weights[users, items].detach(),
                valid,
            )
            loss.backward()
            recommender_optimizer.step()

    recommender.eval()
    with torch.no_grad():
        learned_scores = recommender.full_scores()
        recommendation_candidates = (
            eligible & ~observed if exclude_seen else eligible
        )
        ranking_scores = learned_scores.masked_fill(
            ~recommendation_candidates,
            -torch.inf,
        )
        topk = torch.topk(ranking_scores, k=k, dim=1).indices
    return learned_scores, topk


def train(
    interactions: torch.Tensor,
    item_provider_mapping: torch.Tensor,
    k: int = 10,
    **kwargs: object,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compatibility wrapper for :func:`ada2fair`."""

    return ada2fair(
        interactions=interactions,
        item_provider_mapping=item_provider_mapping,
        k=k,
        **kwargs,
    )

