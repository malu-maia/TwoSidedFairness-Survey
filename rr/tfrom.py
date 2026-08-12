import torch

NAME = 'TFROM'

def tfrom_uniform_pytorch(scores: torch.Tensor, k: int, item_to_provider: torch.Tensor):
    device = scores.device
    m_users, n_items = scores.shape
    unique_providers = torch.unique(item_to_provider)
    n_providers = len(unique_providers)
    provider_map = {p.item(): i for i, p in enumerate(unique_providers)}
    mapped_item_to_provider = torch.tensor([provider_map[p.item()] for p in item_to_provider], device=device)
    
    discounts = 1 / torch.log2(torch.arange(2, k + 2, device=device).float())
    
    total_exposure = m_users * torch.sum(discounts)
    provider_item_counts = torch.bincount(mapped_item_to_provider, minlength=n_providers).float()
    fair_exposure = total_exposure * (provider_item_counts / n_items)

    recommendations = torch.full((m_users, k), -1, device=device) # -1 indica nenhuma recomendação
    provider_exposure = torch.zeros_like(fair_exposure, device=device)
    customer_satisfaction = torch.zeros(m_users, device=device)

    # IDCG por usuário para satisfação normalizada por NDCG (paper Alg. 1, linha 17).
    # ponytail: usa o próprio score como ganho, consistente com o acúmulo de DCG abaixo.
    ideal_scores = torch.topk(scores, k, dim=1).values
    idcg = (ideal_scores * discounts).sum(dim=1).clamp_min(torch.finfo(scores.dtype).eps)

    # Máscara booleana para itens disponíveis
    available_items_mask = torch.ones((m_users, n_items), dtype=torch.bool, device=device)

    for rank in range(k):
        if rank == 0:
            # Ordem aleatória para o primeiro rank
            user_order = torch.randperm(m_users, device=device)
        else:
            # Ordena usuários pela satisfação acumulada
            user_order = torch.argsort(customer_satisfaction, descending=True)
        
        current_rank_discount = discounts[rank]

        # Loop sobre os usuários na ordem definida
        for user_idx in user_order:
            user_scores = scores[user_idx].clone()
            user_scores[~available_items_mask[user_idx]] = -torch.inf # Invalida itens já usados

            # Restrição de 'fair_exposure' para tds os itens de uma vez
            current_providers_exposure = provider_exposure[mapped_item_to_provider]
            
            # Cria uma máscara de itens que satisfazem a restrição de exposição
            exposure_constraint_mask = (current_providers_exposure + current_rank_discount <= fair_exposure[mapped_item_to_provider])
            
            # Invalida os scores dos itens que não cumprem a restrição
            user_scores[~exposure_constraint_mask] = -torch.inf
            
            best_score, best_item_idx = torch.max(user_scores, dim=0)

            if best_score > -torch.inf:
                provider_idx = mapped_item_to_provider[best_item_idx]
                
                # Atualiza as estruturas de dados
                recommendations[user_idx, rank] = best_item_idx
                provider_exposure[provider_idx] += current_rank_discount
                customer_satisfaction[user_idx] += (
                    scores[user_idx, best_item_idx] * current_rank_discount / idcg[user_idx]
                )
                
                # Marca o item como indisponível para este usuário
                available_items_mask[user_idx, best_item_idx] = False

    # fill any remaining empty slots
    for user_idx in range(m_users):
        for rank in range(k):
            if recommendations[user_idx, rank] == -1:
                # get all items that were not yet recommended to this user
                available_indices = available_items_mask[user_idx].nonzero(as_tuple=True)[0]
                
                if len(available_indices) == 0:
                    continue 

                # Find the item from the provider with the minimum current exposure 
                available_providers = mapped_item_to_provider[available_indices]
                exposures_of_available = provider_exposure[available_providers]
                
                # In case of ties in exposure, argmin picks the first one.
                # A more advanced tie-breaker could consider item scores.
                min_exposure_item_in_available = torch.argmin(exposures_of_available)
                item_to_add = available_indices[min_exposure_item_in_available]
                provider_idx = mapped_item_to_provider[item_to_add]

                # Assign item and update metrics
                recommendations[user_idx, rank] = item_to_add
                provider_exposure[provider_idx] += discounts[rank]
                available_items_mask[user_idx, item_to_add] = False
                
    return recommendations


def define_random_providers(
    n_items: int,
    min_size: int = 1,
    max_size: int = 50,
) -> torch.Tensor:
    """
    Simulates providers of different sizes as described in the paper.
    Each item is assigned to a provider.
    """
    item_to_provider_map = torch.zeros(n_items, dtype=torch.long)
    current_item_idx = 0
    provider_id = 0
    while current_item_idx < n_items:
        # Determine size of the current provider
        size = torch.randint(min_size, max_size + 1, (1,)).item()
        end_idx = min(current_item_idx + size, n_items)
        
        # Assign items to this provider
        item_to_provider_map[current_item_idx:end_idx] = provider_id
        
        provider_id += 1
        current_item_idx = end_idx
        
    return item_to_provider_map


def tfrom(
    scores: torch.Tensor,
    k: int = 10,
    item_provider_mapping: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run TFROM on a prepared score matrix.

    Args:
        scores: Two-dimensional tensor with shape ``(n_users, n_items)``.
        k: Number of items to recommend to each user.
        item_provider_mapping: Optional provider ID for every item. When it is
            omitted, the original synthetic random-provider strategy is used.

    Returns:
        The unchanged score matrix and reranked top-k item indices.
    """
    if scores.ndim != 2:
        raise ValueError(
            f"scores must be a 2D user-item matrix, got shape {tuple(scores.shape)}"
        )
    n_items = scores.size(1)
    if not 1 <= k <= n_items:
        raise ValueError(
            f"k must be between 1 and the number of items ({n_items}), got {k}"
        )
    if item_provider_mapping is None:
        item_provider_mapping = define_random_providers(
            n_items=n_items,
            max_size=max(1, int(n_items * 0.3)),
        )
    elif item_provider_mapping.ndim != 1 or len(item_provider_mapping) != n_items:
        raise ValueError(
            "item_provider_mapping must be a 1D tensor with one provider ID per item"
        )

    item_provider_mapping = item_provider_mapping.to(scores.device)
    fair_recs = tfrom_uniform_pytorch(scores=scores, k=k, item_to_provider=item_provider_mapping)

    return scores, fair_recs
