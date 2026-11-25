"""Run the manifest settings with the study's candidate and ranking rules."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from . import grouped_experiments as survey


@dataclass(frozen=True)
class Ranking:
    output: survey.MethodOutput
    replacements: int = 0
    quota_met: bool | None = None
    masked_positions: int = 0


def local_profiles(run: survey.PreparedRun, dimension: int) -> tuple[torch.Tensor, torch.Tensor]:
    from obm.leadfairrec import profile_text_embeddings

    return tuple(
        profile_text_embeddings(survey._local_leadfairrec_profiles(prompts(run)), dimension)
        for prompts in (survey._leadfairrec_user_prompts, survey._leadfairrec_item_prompts)
    )


def resolve_parameters(method: str, parameters: dict, run: survey.PreparedRun) -> dict:
    if method != "SEAL":
        return parameters.copy()
    users, items = run.scores.shape
    lower = math.floor(parameters["fraction"] * users * run.k / items)
    upper = users if parameters["upper"] == "users" else min(users, 2 * lower)
    return parameters | {"product_min": lower, "product_max": upper}


def complete_ranking(raw: torch.Tensor, run: survey.PreparedRun, *, repair: bool) -> tuple[torch.Tensor, int]:
    raw = raw.detach().cpu().long()
    if raw.shape != run.base_topk.shape or (raw < 0).any() or (raw >= run.scores.shape[1]).any():
        raise ValueError("Invalid ranking shape or item IDs")
    ordered = raw.sort(dim=1).values
    if (ordered[:, 1:] == ordered[:, :-1]).any():
        raise ValueError("Duplicate items in a ranking")
    eligible = run.eligible_items if run.eligible_items is not None else run.train_matrix == 0
    bad = ~eligible.gather(1, raw)
    replacements = int(bad.sum())
    if not replacements:
        return raw, 0
    if not repair:
        raise ValueError("Ineligible items in a masked ranking")

    rankings = raw.clone()
    for user in bad.any(dim=1).nonzero().flatten().tolist():
        kept = raw[user, ~bad[user]]
        available = eligible[user].clone()
        available[kept] = False
        missing = run.k - len(kept)
        candidates = survey._topk_from_scores(run.scores[user:user + 1], missing, available[None])[0]
        rankings[user] = torch.cat((kept, candidates))
    if not eligible.gather(1, rankings).all():
        raise ValueError("Cannot complete ranking with eligible items")
    return rankings, replacements


def rank_setting(
    run: survey.PreparedRun,
    method: str,
    parameters: dict,
    *,
    device: str = "cpu",
    groups: tuple[torch.Tensor, torch.Tensor] | None = None,
    profiles: tuple[torch.Tensor, torch.Tensor] | None = None,
    ada_scores: torch.Tensor | None = None,
) -> Ranking:
    survey._seed_everything(run.seed)
    params = resolve_parameters(method, parameters, run)
    scores, train, providers, k = run.scores, run.train_matrix, run.provider_ids, run.k
    eligible = run.eligible_items if run.eligible_items is not None else train == 0
    masked = 0
    if method == "TFROM":
        from rr.tfrom import tfrom

        fair, raw = tfrom(scores, k=k, item_provider_mapping=providers)
    elif method == "FairRec":
        from pam.fairrec import fairrec

        fair, raw = fairrec(scores, k=k)
    elif method == "LeadFairRec":
        from obm.leadfairrec import leadfairrec

        dimension = params.pop("profile_embedding_dim")
        if params.pop("profile_source") != "local_fallback":
            raise ValueError("The study requires a fixed LeadFairRec profile source")
        user_profiles, item_profiles = profiles if profiles is not None else local_profiles(run, dimension)
        fair, raw = leadfairrec(
            train, providers, scores, k=k, eligible_items=eligible, device=device,
            user_profile_embeddings=user_profiles, item_profile_embeddings=item_profiles, **params,
        )
    elif method == "FairSort":
        from rr.fairsort import fairsort

        params.pop("user_order", None)
        fair, raw = fairsort(scores, providers, k=k, eligible_items=eligible, **params)
    elif method in {"CPFair", "TFLD-Groups"}:
        if groups is None:
            group_maps = survey._cpfair_style_groups(run)
            groups = tuple(torch.tensor([mapping[index] for index in range(size)])
                           for mapping, size in zip(group_maps, scores.shape))
        users, items = groups
        if method == "CPFair":
            from pam.greedy_cpfair import cpfair

            fair, raw = cpfair(
                scores.masked_fill(~eligible, -torch.inf), dict(enumerate(users.tolist())),
                dict(enumerate(items.tolist())), run.base_topk, k=k, device=device, **params,
            )
            if params["lambda1"] == params["lambda2"] == 0 and not torch.equal(raw.cpu(), run.base_topk):
                raise ValueError("CPFair zero weights must reproduce BPR")
        else:
            from obm.tfld_groups import tfld_groups

            fair, raw, _, _ = tfld_groups(
                scores, users, items, k=k, eligible_items=eligible, device=torch.device(device), **params,
            )
    elif method == "Ada2Fair":
        from obm.ada2fair import ada2fair

        if ada_scores is None:
            fair, raw = ada2fair(train, providers, k=k, seed=run.seed, eligible_items=eligible,
                                 device=device, **params)
        else:
            fair = ada_scores
            raw = survey._topk_from_scores(fair, k, eligible)
    elif method in {"TFLD", "GGF"}:
        if method == "TFLD":
            from obm.tfld import tfld

            fair, exposure, _, _ = tfld(scores, k=k, device=torch.device(device), exposure_only=True, **params)
        else:
            from obm.ggf import get_w_quantile, ggf

            task, q, omega = (params.pop(name) for name in ("task", "q", "omega"))
            weights = get_w_quantile(len(scores), q, omega) if task == 2 else None
            fair, exposure, _, _ = ggf(
                scores, k=k, user_weights=weights, device=torch.device(device), exposure_only=True, **params,
            )
        exposure = exposure.detach().cpu()
        if not torch.isfinite(exposure).all():
            raise ValueError("Nonfinite expected exposure")
        masked = int((~eligible.gather(1, exposure.topk(k, dim=1).indices)).sum())
        raw = survey._topk_from_scores(exposure, k, eligible)
    elif method == "SEAL":
        from pam.seal import seal

        fair, raw = seal(scores, k=k, product_min=params["product_min"], product_max=params["product_max"])
    else:
        raise ValueError(f"Unknown manifest method {method}")

    topk, replacements = complete_ranking(raw, run, repair=method in {"TFROM", "FairRec", "SEAL"})
    quota = None
    if method == "SEAL":
        counts = torch.bincount(topk.flatten(), minlength=scores.shape[1])
        quota = bool(params["product_min"] <= counts.min() <= counts.max() <= params["product_max"])
    return Ranking(survey._as_output(fair, topk, k), replacements, quota, masked)
