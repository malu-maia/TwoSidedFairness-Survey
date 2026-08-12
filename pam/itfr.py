import torch
import json
import math
import os

from torch import nn
from torch import optim
from collections import defaultdict

import torch.nn.functional as F
import pandas as pd
import numpy as np



#%load_ext cudf.pandas
NAME = 'ITFR'

class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64):
        super().__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)

        # Initialize weights as per standard practice
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)

    def forward(self, user_ids, pos_item_ids, neg_item_ids=None):
        user_embeds = self.user_embeddings(user_ids)
        pos_item_embeds = self.item_embeddings(pos_item_ids)
        
        pos_scores = torch.sum(user_embeds * pos_item_embeds, dim=1)

        if neg_item_ids is None:
            return pos_scores

        neg_item_embeds = self.item_embeddings(neg_item_ids)
        neg_scores = torch.sum(user_embeds * neg_item_embeds, dim=1)

        return pos_scores, neg_scores

def prepare_intersectional(
    data_groups_path: str='data/temp_file_black_friday_groups.csv', user_col: str='User_ID',
    item_col: str='Product_ID', user_group_col: str='Gender',
    item_group_col: str='Product_Category_1'):
    """
    Prepares data by creating mappings for users, items, and their intersectional groups.
    """
    df_groups = pd.read_csv(data_groups_path)
    df_groups = df_groups[[user_col, item_col, user_group_col, item_group_col]].dropna()

    # Create consistent integer mappings for users, items, and their groups
    df_groups['user_id'] = pd.Categorical(df_groups[user_col]).codes
    df_groups['item_id'] = pd.Categorical(df_groups[item_col]).codes
    df_groups['user_group'] = pd.Categorical(df_groups[user_group_col]).codes
    df_groups['item_group'] = pd.Categorical(df_groups[item_group_col]).codes
    
    user_map = df_groups[['user_id', 'user_group']].drop_duplicates().reset_index(drop=True)
    item_map = df_groups[['item_id', 'item_group']].drop_duplicates().reset_index(drop=True)

    # Group interactions by intersectional group
    interactions = {}
    for (user_grp, item_grp), group_df in df_groups.groupby(['user_group', 'item_group']):
        interactions[(user_grp, item_grp)] = list(zip(group_df['user_id'], group_df['item_id']))

    return user_map, item_map, interactions


def predicted_score_normalization(model):
    """
    Normalizes the user and item embeddings to have unit L2 norm.
    """
    with torch.no_grad():
        user_norm = torch.norm(model.user_embeddings.weight, p=2, dim=1, keepdim=True)
        item_norm = torch.norm(model.item_embeddings.weight, p=2, dim=1, keepdim=True)
        
        # avoiding division by zero for zero-norm embeddings
        model.user_embeddings.weight.div_(user_norm.clamp(min=1e-8))
        model.item_embeddings.weight.div_(item_norm.clamp(min=1e-8))
    return model


def itfr(
    user_map: pd.DataFrame, item_map: pd.DataFrame,
    interactions: dict, k: int, epochs: int=10, lr: float=.01, emb_dim: int=64,
    rho: float = 0.5, gamma: float = 1.0, eta: float = 1.0,
    device: torch.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run ITFR using prepared user/item mappings and grouped interactions.

    ``user_map`` must contain ``user_id`` and ``user_group`` columns, while
    ``item_map`` must contain ``item_id`` and ``item_group``. ``interactions``
    maps ``(user_group, item_group)`` pairs to ``(user_id, item_id)`` pairs.

    Returns:
        Normalized fair scores with shape ``(n_users, n_items)`` and top-k item
        indices with shape ``(n_users, k)``.
    """
    required_user_columns = {'user_id', 'user_group'}
    required_item_columns = {'item_id', 'item_group'}
    missing_user_columns = required_user_columns.difference(user_map.columns)
    missing_item_columns = required_item_columns.difference(item_map.columns)
    if missing_user_columns:
        raise ValueError(
            f"user_map is missing required columns: {sorted(missing_user_columns)}"
        )
    if missing_item_columns:
        raise ValueError(
            f"item_map is missing required columns: {sorted(missing_item_columns)}"
        )

    num_users = len(user_map)
    num_items = len(item_map)
    if num_users == 0 or num_items == 0:
        raise ValueError("user_map and item_map must not be empty")
    if not interactions:
        raise ValueError("interactions must contain at least one intersectional group")
    if not 1 <= k <= num_items:
        raise ValueError(
            f"k must be between 1 and the number of items ({num_items}), got {k}"
        )
    if epochs < 1:
        raise ValueError(f"epochs must be at least 1, got {epochs}")

    print(f'k: {k}, num items: {num_items}')
    
    model = MatrixFactorization(num_users, num_items, embedding_dim=emb_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    group_keys = list(interactions.keys())
    num_groups = len(group_keys)
    
    group_weights = (torch.ones(num_groups, device=device) / num_groups)
    last_epoch_grads_user = torch.zeros_like(model.user_embeddings.weight, device=device)
    last_epoch_grads_item = torch.zeros_like(model.item_embeddings.weight, device=device)

    print(f"Starting training for {epochs} epochs on {device}...")
    for epoch in range(epochs):
        model.train()
        
        epoch_triples = {key: [] for key in group_keys}
        for group_key, group_data in interactions.items():
            if not group_data: continue
            for user, pos_item in group_data:
                neg_not_found = True
                while neg_not_found:
                    neg_item = np.random.randint(0, num_items)
                    if neg_item not in np.array(group_data).T[1]:
                        neg_not_found = False
                        
                epoch_triples[group_key].append((user, pos_item, neg_item))
        
        sa_losses = {}
        sa_grads_user = {}
        sa_grads_item = {}

        # Sharpness-aware Loss values and sharpness-aware Gradients for each group 
        for i, group_key in enumerate(group_keys):
            triples = epoch_triples[group_key]
            if not triples: continue

            user_ids, pos_items, neg_items = zip(*triples)
            user_ids = torch.tensor(user_ids, device=device, dtype=torch.long)
            pos_items = torch.tensor(pos_items, device=device, dtype=torch.long)
            neg_items = torch.tensor(neg_items, device=device, dtype=torch.long)

            # forward/backward for original loss to get gradients 
            optimizer.zero_grad()
            pos_scores, neg_scores = model(user_ids, pos_items, neg_items)
            # numerical stability
            original_loss = F.softplus(-(pos_scores - neg_scores)).mean()
            original_loss.backward()

            #store original weights and perturb them 
            with torch.no_grad():
                user_weights_orig = model.user_embeddings.weight.data.clone()
                item_weights_orig = model.item_embeddings.weight.data.clone()
                
                user_grad = model.user_embeddings.weight.grad
                item_grad = model.item_embeddings.weight.grad
                
                # check for None grads in case a group's data doesn't touch all params
                if user_grad is None or item_grad is None: continue

                epsilon_user = rho * user_grad / (torch.norm(user_grad) + 1e-8)
                epsilon_item = rho * item_grad / (torch.norm(item_grad) + 1e-8)
                
                model.user_embeddings.weight.data += epsilon_user
                model.item_embeddings.weight.data += epsilon_item

            # forward/backward at perturbed weights to get SA gradients
            optimizer.zero_grad()
            pos_scores_p, neg_scores_p = model(user_ids, pos_items, neg_items)
            sa_loss = F.softplus(-(pos_scores_p - neg_scores_p)).mean()
            sa_loss.backward()

            # store the SA loss value (detached) and the computed SA gradients
            sa_losses[group_key] = sa_loss.detach()
            if model.user_embeddings.weight.grad is not None:
                sa_grads_user[group_key] = model.user_embeddings.weight.grad.clone()
            if model.item_embeddings.weight.grad is not None:
                sa_grads_item[group_key] = model.item_embeddings.weight.grad.clone()

            # Restore original weights for the next group's calculation
            with torch.no_grad():
                model.user_embeddings.weight.data.copy_(user_weights_orig)
                model.item_embeddings.weight.data.copy_(item_weights_orig)
        
        # Collaborative Loss Balance (Update Group Weights)
        with torch.no_grad():
            contributions = torch.zeros(num_groups, device=device)
            group_sa_losses = torch.tensor([sa_losses.get(key, 0.0) for key in group_keys], device=device)
            # larger gammas give more attention to disadvantaged groups
            loss_powered = group_sa_losses.pow(gamma)
            beta = loss_powered / (loss_powered.sum() + 1e-8)

            for i in range(num_groups):
                group_i_key = group_keys[i]
                if group_i_key not in sa_grads_user: continue
                
                grad_i_user = sa_grads_user[group_i_key]
                grad_i_item = sa_grads_item[group_i_key]
                loss_i = sa_losses[group_i_key]
                
                approx_grad_i_user = torch.sqrt(loss_i) * grad_i_user / (torch.norm(grad_i_user) + 1e-8)
                approx_grad_i_item = torch.sqrt(loss_i) * grad_i_item / (torch.norm(grad_i_item) + 1e-8)

                for j in range(num_groups):
                    if i == j: continue
                    user_contrib = torch.sum(approx_grad_i_user * last_epoch_grads_user)
                    item_contrib = torch.sum(approx_grad_i_item * last_epoch_grads_item)
                    contributions[i] += beta[j] * (user_contrib + item_contrib)
            
            group_weights = group_weights * torch.exp(eta * contributions)
            group_weights /= (group_weights.sum() + 1e-8)

        # compute Final Weighted Gradient and Update Model
        optimizer.zero_grad() 
        with torch.no_grad():
            final_grad_user = torch.zeros_like(model.user_embeddings.weight)
            final_grad_item = torch.zeros_like(model.item_embeddings.weight)

            for i, group_key in enumerate(group_keys):
                if group_key not in sa_grads_user: 
                    continue
                weight = group_weights[i]
                final_grad_user += weight * sa_grads_user[group_key]
                final_grad_item += weight * sa_grads_item[group_key]

            model.user_embeddings.weight.grad = final_grad_user
            model.item_embeddings.weight.grad = final_grad_item

            last_epoch_grads_user.copy_(final_grad_user)
            last_epoch_grads_item.copy_(final_grad_item)

        optimizer.step()
        model = predicted_score_normalization(model)

        if (epoch+1)%100==0:        
            final_loss_value = torch.sum(group_weights * group_sa_losses).item()
            print(f"Epoch {epoch+1}/{epochs}, Collaborative Loss: {final_loss_value:.4f}")


    print("\nGenerating Recommendations")
    fair_topk, fair_scores = get_recs(model, num_users, num_items, k) 

    fair_scores = (fair_scores - torch.min(fair_scores))/(torch.max(fair_scores) - torch.min(fair_scores))
    print(fair_topk)

    return fair_scores, fair_topk


def train(
    user_map: pd.DataFrame, item_map: pd.DataFrame,
    interactions: dict, k: int, epochs: int=10, lr: float=.01, emb_dim: int=64,
    rho: float = 0.5, gamma: float = 1.0, eta: float = 1.0,
    device: torch.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compatibility wrapper for :func:`itfr`."""
    return itfr(
        user_map=user_map,
        item_map=item_map,
        interactions=interactions,
        k=k,
        epochs=epochs,
        lr=lr,
        emb_dim=emb_dim,
        rho=rho,
        gamma=gamma,
        eta=eta,
        device=device,
    )


def get_recs(model, num_users, num_items, k):
    """
    Generates top-k recommendations for all users.
    """
    model.eval()
    recommendations = {}
    with torch.no_grad():
        user_embeds = model.user_embeddings.weight
        item_embeds = model.item_embeddings.weight
        
        full_scores = torch.matmul(user_embeds, item_embeds.T)

        top_k_scores, recs = torch.topk(full_scores, k)
    print(f'scores size: {full_scores.size()}')
    return recs, full_scores

def get_interactions_dict(interactions_data):
    interactions_data = interactions_data.values
    interactions_dict = defaultdict(set)
    for user_id, item_id in interactions_data:
        interactions_dict[user_id].add(item_id)

    return interactions_dict
        

def itg_utility(recs: torch.Tensor, interactions_dict: dict, user_map: pd.DataFrame, item_map: pd.DataFrame, device: torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    user_to_group = dict(zip(user_map['user_id'], user_map['user_group']))
    item_to_group = dict(zip(item_map['item_id'], item_map['item_group']))

    group_recalls = defaultdict(list)

    recs_sets = [set(user_recs.cpu().numpy()) for user_recs in recs]
    
    num_users = len(user_map)
    for user_id in range(num_users):
        if user_id not in interactions_dict or not interactions_dict[user_id]:
            continue

        user_group = user_to_group.get(user_id)
        if user_group is None: continue

        recommendations_for_user = recs_sets[user_id]
        true_positives_for_user = interactions_dict[user_id]
        
        hits_per_item_group = defaultdict(int)
        total_relevant_per_item_group = defaultdict(int)

        for item_id in true_positives_for_user:
            item_group = item_to_group.get(item_id)
            if item_group is None: continue
            
            total_relevant_per_item_group[item_group] += 1
            if item_id in recommendations_for_user:
                hits_per_item_group[item_group] += 1

        for item_group, total_relevant in total_relevant_per_item_group.items():
            if total_relevant > 0:
                recall = hits_per_item_group[item_group] / total_relevant
                group_recalls[(user_group, item_group)].append(recall)

    itg_utilities = {}
    for group, recalls_list in group_recalls.items():
        if recalls_list:
            itg_utilities[group] = np.mean(recalls_list)

    return itg_utilities
