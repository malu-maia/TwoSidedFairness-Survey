from __future__ import annotations

import hashlib
import json
import math
import os
import urllib.error
import urllib.request
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


NAME = "LeadFairRec"


class _MatrixFactorization(nn.Module):
    """BPR matrix-factorization backbone used by LeadFairRec."""

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


class _BiasEstimators(nn.Module):
    """Small estimators for PBE, UA, and IP from the LeadFairRec graph."""

    def __init__(self, n_users: int, n_items: int, embedding_dim: int) -> None:
        super().__init__()
        self.user_activity = nn.Embedding(n_users, 1)
        self.item_popularity = nn.Embedding(n_items, 1)
        self.pbe = nn.Sequential(
            nn.Linear(2 * embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
        )

    def ua(self, users: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.user_activity(users).squeeze(-1))

    def ip(self, items: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.item_popularity(items).squeeze(-1))

    def pbe_score(
        self,
        user_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sigmoid(
            self.pbe(torch.cat((user_embeddings, item_embeddings), dim=1)).squeeze(-1)
        )


class _SemanticProjector(nn.Module):
    """Project profile embeddings into the recommender embedding space."""

    def __init__(self, input_dim: int, embedding_dim: int) -> None:
        super().__init__()
        hidden_dim = max(embedding_dim, min(input_dim, 128))
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.layers(embeddings)


def _validate_inputs(
    interactions: torch.Tensor,
    item_provider_mapping: torch.Tensor,
    k: int,
    similar_user_count: int,
    eligible_items: torch.Tensor | None,
    exclude_seen: bool,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    if torch.any(observed.sum(dim=1) == 0):
        raise ValueError("each user must have at least one observed interaction")

    n_users, n_items = values.shape
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
    if providers.is_floating_point():
        if not torch.isfinite(providers).all():
            raise ValueError("item_provider_mapping must contain finite provider IDs")
        if not torch.equal(providers, providers.round()):
            raise ValueError("item_provider_mapping must contain integer provider IDs")
    _, provider_indices = torch.unique(
        providers.to(torch.long),
        sorted=True,
        return_inverse=True,
    )

    if similar_user_count < 1:
        raise ValueError(
            f"similar_user_count must be at least 1, got {similar_user_count}"
        )

    if eligible_items is None:
        eligible = torch.ones_like(observed)
    else:
        if eligible_items.shape != interactions.shape:
            raise ValueError("eligible_items must have the same shape as interactions")
        eligible = eligible_items.to(device=device, dtype=torch.bool)
    if torch.any(eligible.sum(dim=1) == 0):
        raise ValueError("each user must have at least one eligible item")

    candidates = eligible & ~observed if exclude_seen else eligible
    if torch.any(candidates.sum(dim=1) < k):
        raise ValueError("each user must have at least k recommendation candidates")
    return values, observed, provider_indices.to(torch.long), eligible


def _validate_scores(
    scores: torch.Tensor,
    expected_shape: torch.Size,
    device: torch.device,
) -> torch.Tensor:
    if scores.ndim != 2 or scores.shape != expected_shape:
        raise ValueError(
            "scores must be a 2D tensor with the same shape as interactions"
        )
    if scores.is_complex():
        raise ValueError("scores must contain real values")
    values = scores.to(device=device, dtype=torch.float32)
    if not torch.isfinite(values).all():
        raise ValueError("scores must contain only finite values")
    return values


def _provider_history(
    observed: torch.Tensor,
    provider_indices: torch.Tensor,
) -> torch.Tensor:
    n_providers = int(provider_indices.max().item()) + 1
    history_counts = torch.zeros(
        observed.size(0),
        n_providers,
        device=observed.device,
        dtype=torch.float32,
    )
    expanded_providers = provider_indices.unsqueeze(0).expand(observed.size(0), -1)
    history_counts.scatter_add_(1, expanded_providers, observed.to(torch.float32))
    return history_counts > 0


def _jaccard_similarity(provider_history: torch.Tensor) -> torch.Tensor:
    history = provider_history.to(torch.float32)
    intersection = history @ history.T
    counts = history.sum(dim=1)
    union = counts.unsqueeze(1) + counts.unsqueeze(0) - intersection
    similarity = torch.divide(
        intersection,
        union.clamp_min(torch.finfo(history.dtype).eps),
    )
    similarity.fill_diagonal_(0)
    return similarity


def _pbe_targets(
    observed: torch.Tensor,
    provider_indices: torch.Tensor,
    similar_user_count: int,
) -> torch.Tensor:
    """Compute paper PBE: provider penetration in similar users' histories."""

    provider_history = _provider_history(observed, provider_indices)
    similarity = _jaccard_similarity(provider_history)
    n_users = observed.size(0)
    resolved_count = min(similar_user_count, max(n_users - 1, 1))
    similar_users = torch.topk(similarity, k=resolved_count, dim=1).indices
    similar_provider_history = provider_history[similar_users]
    provider_penetration = similar_provider_history.to(torch.float32).mean(dim=1)
    return provider_penetration[:, provider_indices]


def _bias_targets(
    observed: torch.Tensor,
    provider_indices: torch.Tensor,
    similar_user_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_users = observed.size(0)
    # Def. 2: user activity is the raw interaction count a_u = |I_u|.
    user_activity = observed.sum(dim=1).to(torch.float32)
    # Def. 1: item popularity is the user fraction g_i = |U_i| / |U|.
    item_popularity = observed.sum(dim=0).to(torch.float32) / max(n_users, 1)
    pbe = _pbe_targets(observed, provider_indices, similar_user_count)
    return user_activity, item_popularity, pbe


def _sample_negative_items(
    users: torch.Tensor,
    negative_candidates: list[torch.Tensor],
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    sampled = [
        candidates[
            torch.randint(
                candidates.numel(),
                (1,),
                generator=generator,
            ).item()
        ]
        for candidates in (negative_candidates[int(user)] for user in users)
    ]
    return torch.stack(sampled).to(device=device)


def _bpr_loss(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
) -> torch.Tensor:
    return -F.logsigmoid(positive_scores - negative_scores).mean()


def _contrastive_alignment_loss(
    id_embeddings: torch.Tensor,
    semantic_embeddings: torch.Tensor,
    projector: _SemanticProjector,
    temperature: float,
) -> torch.Tensor:
    projected = F.normalize(projector(semantic_embeddings), dim=1)
    ids = F.normalize(id_embeddings, dim=1)
    logits = projected @ ids.T / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, labels)


def _resolve_semantic_inputs(
    user_profile_embeddings: torch.Tensor | None,
    item_profile_embeddings: torch.Tensor | None,
    n_users: int,
    n_items: int,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if user_profile_embeddings is None and item_profile_embeddings is None:
        return None, None
    if user_profile_embeddings is None or item_profile_embeddings is None:
        raise ValueError(
            "user_profile_embeddings and item_profile_embeddings must be supplied together"
        )
    if user_profile_embeddings.ndim != 2 or user_profile_embeddings.size(0) != n_users:
        raise ValueError(
            "user_profile_embeddings must be a 2D tensor with one row per user"
        )
    if item_profile_embeddings.ndim != 2 or item_profile_embeddings.size(0) != n_items:
        raise ValueError(
            "item_profile_embeddings must be a 2D tensor with one row per item"
        )
    if user_profile_embeddings.size(1) != item_profile_embeddings.size(1):
        raise ValueError("profile embeddings must have the same embedding dimension")
    if user_profile_embeddings.is_complex() or item_profile_embeddings.is_complex():
        raise ValueError("profile embeddings must contain real values")

    users = user_profile_embeddings.to(device=device, dtype=torch.float32)
    items = item_profile_embeddings.to(device=device, dtype=torch.float32)
    if not torch.isfinite(users).all() or not torch.isfinite(items).all():
        raise ValueError("profile embeddings must contain only finite values")
    return users, items


def _semantic_affinity(
    user_profile_embeddings: torch.Tensor,
    item_profile_embeddings: torch.Tensor,
) -> torch.Tensor:
    users = F.normalize(user_profile_embeddings, dim=1)
    items = F.normalize(item_profile_embeddings, dim=1)
    return users @ items.T


def _counterfactual_penalty(
    pbe: torch.Tensor,
    user_activity: torch.Tensor,
    item_popularity: torch.Tensor,
    gamma: float,
    delta: float,
    beta: float,
) -> torch.Tensor:
    return (
        gamma * pbe
        + delta * item_popularity.unsqueeze(0)
        + beta * user_activity.unsqueeze(1)
    )


def _text_embedding(text: str, embedding_dim: int) -> torch.Tensor:
    vector = torch.zeros(embedding_dim, dtype=torch.float32)
    tokens = text.lower().split()
    if not tokens:
        return vector
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "big") % embedding_dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    return F.normalize(vector, dim=0)


def profile_text_embeddings(
    profiles: Sequence[str],
    embedding_dim: int = 64,
) -> torch.Tensor:
    """Convert generated profile text into deterministic lightweight embeddings."""

    if embedding_dim < 1:
        raise ValueError("embedding_dim must be at least 1")
    return torch.stack([_text_embedding(profile, embedding_dim) for profile in profiles])


# Use before calling the main function to generate enhanced profiles
def _deepseek_profile(
    system_prompt: str,
    user_prompt: str,
    *,
    api_key: str,
    model: str,
    base_url: str,
    timeout: float,
) -> str:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(
            {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.2,
            }
        ).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as error:
        raise RuntimeError(f"DeepSeek profile generation failed: {error}") from error
    return str(payload["choices"][0]["message"]["content"]).strip()

# Use before calling the main function to generate enhanced profiles
def generate_deepseek_profiles(
    prompts: Sequence[str],
    *,
    system_prompt: str,
    api_key: str | None = None,
    model: str = "deepseek-chat",
    base_url: str = "https://api.deepseek.com",
    timeout: float = 30.0,
) -> list[str]:
    """Generate compact profiles with DeepSeek's OpenAI-compatible API.

    This helper is intentionally separate from ``leadfairrec`` so normal model
    training remains deterministic and does not require network access.
    """

    resolved_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not resolved_key:
        raise ValueError("api_key or DEEPSEEK_API_KEY is required")
    if timeout <= 0 or not math.isfinite(timeout):
        raise ValueError("timeout must be finite and positive")
    return [
        _deepseek_profile(
            system_prompt,
            prompt,
            api_key=resolved_key,
            model=model,
            base_url=base_url,
            timeout=timeout,
        )
        for prompt in prompts
    ]


def leadfairrec(
    interactions: torch.Tensor,
    item_provider_mapping: torch.Tensor,
    scores: torch.Tensor,
    k: int = 10,
    *,
    user_profile_embeddings: torch.Tensor | None = None,
    item_profile_embeddings: torch.Tensor | None = None,
    semantic_weight: float = 0.0,
    gamma: float = -256.0,
    delta: float = 64.0,
    beta: float = 1.0,
    similar_user_count: int = 30,
    eligible_items: torch.Tensor | None = None,
    exclude_seen: bool = True,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply LeadFairRec counterfactual debiasing to shared base scores.

    Positive entries in ``interactions`` are observed implicit feedback used to
    estimate PBE, user activity, and item popularity. ``scores`` is the
    standardized base recommender matrix shared across approaches. Optional
    profile embeddings, typically generated from DeepSeek profiles, add the
    paper's semantic signal before the counterfactual penalty.
    """

    for name, value in {
        "semantic_weight": semantic_weight,
        "gamma": gamma,
        "delta": delta,
        "beta": beta,
    }.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if semantic_weight < 0:
        raise ValueError("semantic_weight must be non-negative")

    resolved_device = scores.device if device is None else torch.device(device)
    values, observed, provider_indices, eligible = _validate_inputs(
        interactions,
        item_provider_mapping,
        k,
        similar_user_count,
        eligible_items,
        exclude_seen,
        resolved_device,
    )
    base_scores = _validate_scores(scores, values.shape, resolved_device)
    user_profiles, item_profiles = _resolve_semantic_inputs(
        user_profile_embeddings,
        item_profile_embeddings,
        values.size(0),
        values.size(1),
        resolved_device,
    )
    enhanced_scores = base_scores
    if user_profiles is not None and item_profiles is not None:
        enhanced_scores = enhanced_scores + semantic_weight * _semantic_affinity(
            user_profiles,
            item_profiles,
        )
    user_activity, item_popularity, pbe = _bias_targets(
        observed,
        provider_indices,
        similar_user_count,
    )
    fair_scores = enhanced_scores - _counterfactual_penalty(
        pbe,
        user_activity,
        item_popularity,
        gamma,
        delta,
        beta,
    )

    candidates = eligible & ~observed if exclude_seen else eligible
    ranking_scores = fair_scores.masked_fill(~candidates, -torch.inf)
    topk = torch.topk(ranking_scores, k=k, dim=1).indices
    return fair_scores, topk


def train(
    interactions: torch.Tensor,
    item_provider_mapping: torch.Tensor,
    scores: torch.Tensor,
    k: int = 10,
    **kwargs: object,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compatibility wrapper for :func:`leadfairrec`."""

    return leadfairrec(
        interactions=interactions,
        item_provider_mapping=item_provider_mapping,
        scores=scores,
        k=k,
        **kwargs,
    )
