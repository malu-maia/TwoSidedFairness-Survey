from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


NAME = "MultiFR"


class _MatrixFactorization(nn.Module):
    """BPR matrix factorization backbone used by MultiFR."""

    def __init__(self, n_users: int, n_items: int, embedding_dim: int) -> None:
        super().__init__()
        self.user_embeddings = nn.Embedding(n_users, embedding_dim)
        self.item_embeddings = nn.Embedding(n_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_embeddings.weight)
        nn.init.xavier_uniform_(self.item_embeddings.weight)

    def pair_scores(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
    ) -> torch.Tensor:
        return (self.user_embeddings(users) * self.item_embeddings(items)).sum(dim=1)

    def scores_for_users(self, users: torch.Tensor) -> torch.Tensor:
        return self.user_embeddings(users) @ self.item_embeddings.weight.T

    def full_scores(self) -> torch.Tensor:
        return self.user_embeddings.weight @ self.item_embeddings.weight.T


def _group_indices(
    groups: torch.Tensor,
    expected_size: int,
    name: str,
    device: torch.device,
) -> torch.Tensor:
    if groups.ndim != 1 or groups.numel() != expected_size:
        raise ValueError(
            f"{name} must be a 1D tensor with {expected_size} entries"
        )
    if groups.is_complex():
        raise ValueError(f"{name} must contain integer group IDs")
    values = groups.to(device=device)
    if values.is_floating_point():
        if not torch.isfinite(values).all():
            raise ValueError(f"{name} must contain only finite group IDs")
        if not torch.equal(values, values.round()):
            raise ValueError(f"{name} must contain integer group IDs")
    _, indices = torch.unique(
        values.to(torch.long),
        sorted=True,
        return_inverse=True,
    )
    if torch.unique(indices).numel() < 2:
        raise ValueError(f"{name} must contain at least two groups")
    return indices


def _validate_inputs(
    interactions: torch.Tensor,
    user_groups: torch.Tensor,
    item_groups: torch.Tensor,
    k: int,
    epochs: int,
    embedding_dim: int,
    batch_size: int,
    learning_rate: float,
    regularization: float,
    fairness_k: int,
    gamma: float,
    temperature: float,
    gumbel_samples: int,
    rounds: int,
    seed: int,
    exclude_seen: bool,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if interactions.ndim != 2:
        raise ValueError(
            "interactions must be a 2D user-item matrix, "
            f"got shape {tuple(interactions.shape)}"
        )
    if interactions.numel() == 0:
        raise ValueError("interactions must not be empty")
    if interactions.is_complex():
        raise ValueError("interactions must contain real ratings")

    ratings = interactions.to(device=device, dtype=torch.float32)
    if not torch.isfinite(ratings).all():
        raise ValueError("interactions must contain only finite ratings")
    if torch.any(ratings < 0):
        raise ValueError("interactions must be non-negative")

    n_users, n_items = ratings.shape
    if not 1 <= k <= n_items:
        raise ValueError(
            f"k must be between 1 and the number of items ({n_items}), got {k}"
        )

    integer_parameters = {
        "epochs": epochs,
        "embedding_dim": embedding_dim,
        "batch_size": batch_size,
        "fairness_k": fairness_k,
        "gumbel_samples": gumbel_samples,
        "rounds": rounds,
    }
    for name, value in integer_parameters.items():
        if value < 1:
            raise ValueError(f"{name} must be at least 1, got {value}")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    real_parameters = {
        "learning_rate": learning_rate,
        "regularization": regularization,
        "gamma": gamma,
        "temperature": temperature,
    }
    for name, value in real_parameters.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if regularization < 0:
        raise ValueError("regularization must be non-negative")
    if not 0 < gamma < 1:
        raise ValueError("gamma must be between 0 and 1")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    observed_per_user = (ratings > 0).sum(dim=1)
    if torch.any(observed_per_user == 0):
        raise ValueError("each user must have at least one observed rating")
    if torch.any(observed_per_user == n_items):
        raise ValueError("each user must have at least one unobserved item")
    if exclude_seen and torch.any(n_items - observed_per_user < k):
        raise ValueError("each user must have at least k unobserved items")

    resolved_user_groups = _group_indices(
        user_groups,
        n_users,
        "user_groups",
        device,
    )
    resolved_item_groups = _group_indices(
        item_groups,
        n_items,
        "item_groups",
        device,
    )
    return ratings, resolved_user_groups, resolved_item_groups


def _smooth_rank_chunk(
    scores: torch.Tensor,
    selected_scores: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    differences = scores.unsqueeze(1) - selected_scores.unsqueeze(2)
    return 0.5 + torch.sigmoid(differences / temperature).sum(dim=2)


def _smooth_rank_values(
    scores: torch.Tensor,
    selected_scores: torch.Tensor,
    temperature: float,
    *,
    chunk_size: int = 256,
) -> torch.Tensor:
    """Compute smooth ranks while bounding tensors retained for backpropagation."""

    ranks = []
    requires_grad = torch.is_grad_enabled() and (
        scores.requires_grad or selected_scores.requires_grad
    )
    for start in range(0, selected_scores.size(1), chunk_size):
        selected = selected_scores[:, start : start + chunk_size]
        if requires_grad:
            ranks.append(
                checkpoint(
                    _smooth_rank_chunk,
                    scores,
                    selected,
                    temperature,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            )
        else:
            ranks.append(_smooth_rank_chunk(scores, selected, temperature))
    return torch.cat(ranks, dim=1)


def _smooth_ranks(
    scores: torch.Tensor,
    temperature: float,
    *,
    chunk_size: int = 256,
) -> torch.Tensor:
    """Compute the paper's sigmoid approximation of all item ranks."""

    return _smooth_rank_values(
        scores,
        scores,
        temperature,
        chunk_size=chunk_size,
    )


def _consumer_fairness_loss(
    scores: torch.Tensor,
    relevance: torch.Tensor,
    user_groups: torch.Tensor,
    fairness_k: int,
    temperature: float,
) -> torch.Tensor:
    """Mean pairwise disparity of group SmoothNDCG prefix vectors.

    Relevance is a graded matrix of shape (n_users, n_items) holding the
    normalized held-in ratings; zeros are unobserved. Gains are linear in the
    graded relevance, matching :func:`experiments.metrics.ndcg`."""
    resolved_k = min(fairness_k, scores.size(1))
    order = torch.argsort(scores, dim=1, descending=True)[:, :resolved_k]
    ranked_scores = torch.gather(scores, 1, order)
    ranked_relevance = torch.gather(relevance.to(scores.dtype), 1, order)
    ranked_ranks = _smooth_rank_values(scores, ranked_scores, temperature)
    dcg = torch.cumsum(
        ranked_relevance / torch.log2(ranked_ranks + 1),
        dim=1,
    )

    positions = torch.arange(
        1,
        resolved_k + 1,
        device=scores.device,
        dtype=scores.dtype,
    )
    ideal_relevance = torch.topk(
        relevance.to(scores.dtype), k=resolved_k, dim=1
    ).values
    ideal_gains = ideal_relevance / torch.log2(positions + 1)
    ideal_dcg = torch.cumsum(ideal_gains, dim=1).clamp_min(
        torch.finfo(scores.dtype).eps
    )
    smooth_ndcg = dcg / ideal_dcg

    group_satisfaction = [
        smooth_ndcg[user_groups == group].mean(dim=0)
        for group in torch.unique(user_groups)
        if torch.any(user_groups == group)
    ]
    if len(group_satisfaction) < 2:
        return scores.sum() * 0

    disparities = [
        (group_satisfaction[left] - group_satisfaction[right]).square().mean()
        for left in range(len(group_satisfaction))
        for right in range(left + 1, len(group_satisfaction))
    ]
    return torch.stack(disparities).mean()


def _target_exposure(
    relevance: torch.Tensor,
    gamma: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    # ponytail: target exposure keeps the paper's binary observed/unobserved
    # split (observed items share the top-|R_u| discount mass equally); a
    # rating-proportional target is the upgrade path if graded merit matters.
    observed = relevance > 0
    n_items = relevance.size(1)
    discounts = torch.pow(
        torch.as_tensor(gamma, device=relevance.device, dtype=dtype),
        torch.arange(1, n_items + 1, device=relevance.device, dtype=dtype),
    )
    prefix = torch.cumsum(discounts, dim=0)
    positive_counts = observed.sum(dim=1)
    positive_exposure = prefix[positive_counts - 1] / positive_counts
    negative_exposure = (
        prefix[-1] - prefix[positive_counts - 1]
    ) / (n_items - positive_counts)
    return torch.where(
        observed,
        positive_exposure.unsqueeze(1),
        negative_exposure.unsqueeze(1),
    )


def _sample_gumbel(
    shape: torch.Size,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> torch.Tensor:
    uniform = torch.rand(
        shape,
        device=device,
        dtype=dtype,
        generator=generator,
    ).clamp_(1e-7, 1 - 1e-7)
    return -torch.log(-torch.log(uniform))


def _producer_fairness_loss(
    scores: torch.Tensor,
    relevance: torch.Tensor,
    item_groups: torch.Tensor,
    gamma: float,
    temperature: float,
    gumbel_samples: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Squared error between system and relevance-based group exposure."""

    system_exposure = torch.zeros_like(scores)
    for _ in range(gumbel_samples):
        noisy_probabilities = torch.softmax(
            scores
            + _sample_gumbel(
                scores.shape,
                device=scores.device,
                dtype=scores.dtype,
                generator=generator,
            ),
            dim=1,
        )
        ranks = _smooth_ranks(noisy_probabilities, temperature)
        system_exposure += torch.pow(
            torch.as_tensor(gamma, device=scores.device, dtype=scores.dtype),
            ranks,
        )
    system_exposure /= gumbel_samples
    target_exposure = _target_exposure(relevance, gamma, scores.dtype)

    system_groups = []
    target_groups = []
    for group in torch.unique(item_groups):
        mask = item_groups == group
        system_groups.append(system_exposure[:, mask].mean())
        target_groups.append(target_exposure[:, mask].mean())
    return (
        torch.stack(system_groups) - torch.stack(target_groups)
    ).square().sum()


def _bpr_loss(
    model: _MatrixFactorization,
    users: torch.Tensor,
    positive_items: torch.Tensor,
    negative_items: torch.Tensor,
    valid: torch.Tensor,
    regularization: float,
) -> torch.Tensor:
    positive_scores = model.pair_scores(users, positive_items)
    negative_scores = model.pair_scores(users, negative_items)
    pair_losses = -F.logsigmoid(positive_scores - negative_scores)
    ranking_loss = (pair_losses * valid).sum() / valid.sum().clamp_min(1)
    if regularization == 0:
        return ranking_loss
    embeddings = (
        model.user_embeddings(users).square().mean()
        + model.item_embeddings(positive_items).square().mean()
        + model.item_embeddings(negative_items).square().mean()
    )
    return ranking_loss + regularization * embeddings


def _frank_wolfe_weights(
    gram: torch.Tensor,
    *,
    max_iterations: int = 50,
    tolerance: float = 1e-6,
) -> torch.Tensor:
    """Solve the MGDA minimum-norm convex combination.
    - gram matrix refer to the M matrix, weights refer to alpha,
    and  step refer to w from the paper."""
    n_objectives = gram.size(0)
    weights = torch.full(
        (n_objectives,),
        1 / n_objectives,
        device=gram.device,
        dtype=gram.dtype,
    )
    epsilon = torch.finfo(gram.dtype).eps

    for _ in range(max_iterations):
        gram_weights = gram @ weights
        vertex = int(torch.argmin(gram_weights).item())
        current_norm = weights @ gram_weights
        denominator = (
            gram[vertex, vertex]
            - 2 * gram_weights[vertex]
            + current_norm
        )
        if denominator <= epsilon:
            break
        step = ((current_norm - gram_weights[vertex]) / denominator).clamp(0, 1)
        updated = (1 - step) * weights
        updated[vertex] += step
        if torch.max(torch.abs(updated - weights)) <= tolerance:
            weights = updated
            break
        weights = updated
    return weights


def _objective_gradients(
    losses: tuple[torch.Tensor, ...],
    parameters: tuple[nn.Parameter, ...],
) -> tuple[list[tuple[torch.Tensor, ...]], torch.Tensor]:
    gradients = [
        torch.autograd.grad(loss, parameters, retain_graph=True)
        for loss in losses
    ]
    gram = torch.empty(
        (len(losses), len(losses)),
        device=losses[0].device,
        dtype=losses[0].dtype,
    )
    for left in range(len(losses)):
        for right in range(left, len(losses)):
            dot_product = sum(
                (left_gradient * right_gradient).sum()
                for left_gradient, right_gradient in zip(
                    gradients[left],
                    gradients[right],
                )
            )
            gram[left, right] = dot_product
            gram[right, left] = dot_product
    return gradients, gram


def _sample_graded_negatives(
    ratings: torch.Tensor,
    users: torch.Tensor,
    positives: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample one item per pair with a strictly lower rating (unrated counts as 0).

    ponytail: uniform draw + one resample round, leftovers masked out of the
    loss; sparsity makes invalid draws rare.
    """

    n_items = ratings.shape[1]
    positive_ratings = ratings[users, positives]
    negatives = torch.randint(
        n_items, (users.size(0),), generator=generator
    ).to(ratings.device)
    invalid = ratings[users, negatives] >= positive_ratings
    if invalid.any():
        negatives[invalid] = torch.randint(
            n_items, (int(invalid.sum()),), generator=generator
        ).to(ratings.device)
    valid = ratings[users, negatives] < positive_ratings
    return negatives, valid


def _train_round(
    relevance: torch.Tensor,
    user_groups: torch.Tensor,
    item_groups: torch.Tensor,
    *,
    epochs: int,
    embedding_dim: int,
    batch_size: int,
    learning_rate: float,
    regularization: float,
    fairness_k: int,
    gamma: float,
    temperature: float,
    gumbel_samples: int,
    seed: int,
    device: torch.device,
) -> tuple[_MatrixFactorization, tuple[float, float, float]]:
    n_users, n_items = relevance.shape
    rng_devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    with torch.random.fork_rng(devices=rng_devices):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        model = _MatrixFactorization(n_users, n_items, embedding_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    parameters = tuple(model.parameters())
    positive_pairs = torch.nonzero(relevance, as_tuple=False).cpu()
    cpu_generator = torch.Generator(device="cpu")
    cpu_generator.manual_seed(seed)
    fairness_generator = torch.Generator(device=device)
    fairness_generator.manual_seed(seed)
    group_representatives = torch.stack(
        [
            torch.nonzero(user_groups == group, as_tuple=False)[0, 0]
            for group in torch.unique(user_groups)
        ]
    )
    final_losses = (math.inf, math.inf, math.inf)

    for _ in range(epochs):
        order = torch.randperm(
            positive_pairs.size(0),
            generator=cpu_generator,
        )
        epoch_losses = torch.zeros(3, dtype=torch.float64)
        batches = 0

        for start in range(0, positive_pairs.size(0), batch_size):
            pairs = positive_pairs[order[start : start + batch_size]]
            users = pairs[:, 0].to(device)
            positive_items = pairs[:, 1].to(device)
            negative_items, valid_pairs = _sample_graded_negatives(
                relevance,
                users,
                positive_items,
                cpu_generator,
            )

            fairness_users = torch.unique(
                torch.cat((users, group_representatives))
            )
            fairness_scores = model.scores_for_users(fairness_users)
            losses = (
                _bpr_loss(
                    model,
                    users,
                    positive_items,
                    negative_items,
                    valid_pairs,
                    regularization,
                ),
                _consumer_fairness_loss(
                    fairness_scores,
                    relevance[fairness_users],
                    user_groups[fairness_users],
                    fairness_k,
                    temperature,
                ),
                _producer_fairness_loss(
                    fairness_scores,
                    relevance[fairness_users],
                    item_groups,
                    gamma,
                    temperature,
                    gumbel_samples,
                    fairness_generator,
                ),
            )
            gradients, gram = _objective_gradients(losses, parameters)
            weights = _frank_wolfe_weights(gram.detach())

            optimizer.zero_grad()
            for parameter_index, parameter in enumerate(parameters):
                parameter.grad = sum(
                    weights[objective] * gradients[objective][parameter_index]
                    for objective in range(len(losses))
                )
            optimizer.step()

            epoch_losses += torch.tensor(
                [loss.detach().item() for loss in losses],
                dtype=torch.float64,
            )
            batches += 1

        final_losses = tuple((epoch_losses / batches).tolist())
    return model, final_losses


def multifr(
    interactions: torch.Tensor,
    user_groups: torch.Tensor,
    item_groups: torch.Tensor,
    k: int = 10,
    *,
    positive_threshold: float = 4.0,
    epochs: int = 100,
    embedding_dim: int = 50,
    batch_size: int = 1024,
    learning_rate: float = 1e-3,
    regularization: float = 1e-4,
    fairness_k: int = 20,
    gamma: float = 0.8,
    temperature: float = 0.1,
    gumbel_samples: int = 1,
    rounds: int = 5,
    seed: int = 42,
    exclude_seen: bool = True,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Train MultiFR with a graded-pair BPR matrix-factorization backbone.

    Every positive entry in ``interactions`` is an observed graded rating;
    zeros are unobserved. The accuracy objective samples pairs (i, j) with
    rating(i) > rating(j), where unrated items count as rating 0. Each
    training batch jointly optimizes graded-pair BPR accuracy, user-group
    SmoothNDCG parity with graded gains, and item-group exposure. MGDA with a
    Frank-Wolfe solver determines the three objective weights. The final
    model is the least-misery solution across ``rounds`` independent runs.

    ``positive_threshold`` is deprecated and ignored: the graded objective
    consumes rating magnitudes directly instead of thresholding them.
    """

    del positive_threshold
    resolved_device = (
        interactions.device if device is None else torch.device(device)
    )
    relevance, resolved_user_groups, resolved_item_groups = _validate_inputs(
        interactions,
        user_groups,
        item_groups,
        k,
        epochs,
        embedding_dim,
        batch_size,
        learning_rate,
        regularization,
        fairness_k,
        gamma,
        temperature,
        gumbel_samples,
        rounds,
        seed,
        exclude_seen,
        resolved_device,
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_misery = math.inf
    for round_index in range(rounds):
        model, losses = _train_round(
            relevance,
            resolved_user_groups,
            resolved_item_groups,
            epochs=epochs,
            embedding_dim=embedding_dim,
            batch_size=batch_size,
            learning_rate=learning_rate,
            regularization=regularization,
            fairness_k=fairness_k,
            gamma=gamma,
            temperature=temperature,
            gumbel_samples=gumbel_samples,
            seed=seed + round_index,
            device=resolved_device,
        )
        misery = max(losses)
        if misery < best_misery:
            best_misery = misery
            best_state = {
                name: value.detach().clone()
                for name, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("MultiFR training did not produce a valid solution")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        learned_scores = model.full_scores()
        ranking_scores = (
            learned_scores.masked_fill(relevance > 0, -torch.inf)
            if exclude_seen
            else learned_scores
        )
        topk = torch.topk(ranking_scores, k=k, dim=1).indices
    return learned_scores, topk
