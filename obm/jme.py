import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from torch.utils.data import DataLoader, TensorDataset
from typing import Mapping

from utils import mapping_indexes, pivot_gen_score_matrix

NAME = 'JME'

class MatrixFactorization(nn.Module):
    """
    A simple Matrix Factorization model.
    """
    def __init__(self, num_users, num_items, embedding_dim=64):
        super().__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)

        # Initialize embeddings with Xavier initialization as mentioned in the paper
        nn.init.xavier_uniform_(self.user_embeddings.weight)
        nn.init.xavier_uniform_(self.item_embeddings.weight)

    def forward(self, user_indices, item_indices):
        """
        Computes relevance scores for given user-item pairs.
        """
        user_embs = self.user_embeddings(user_indices)
        item_embs = self.item_embeddings(item_indices)

        # In a training loop, user_indices might be (batch_size) and
        # item_indices might be (num_all_items).
        # We need to compute the score for each user with all items.
        return torch.matmul(user_embs, item_embs.t())


def jme(data_path: str, user_groups: Mapping[int, int], run_i: int, item_groups: Mapping[int, int],
        emb_size: int=32, batch_size: int=100, epochs: int=200, lr: float=1e-2,
        alpha: float=.0, gamma: float=.8, tau: float=0.1, k: int=100, n_samples: int=100, 
        device: torch.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ) -> tuple[torch.Tensor, torch.Tensor]:
    """Run JME from an interaction file and sampled-row index file.

    The interaction file is read as tab-separated ``user_id``, ``item_id``,
    ``rating`` and optional ``timestamp`` columns. Sampled row indices are read
    from ``data/temp_data/temp_indexes_<dataset>_run<run_i>.npz``.

    Returns:
        Normalized fair scores and top-k item indices.
    """
    if not user_groups:
        raise ValueError("user_groups must not be empty")
    if not item_groups:
        raise ValueError("item_groups must not be empty")
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    if epochs < 1:
        raise ValueError(f"epochs must be at least 1, got {epochs}")

    #### loading data
    try:
        df = pd.read_csv(data_path, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    except:
        df = pd.read_csv(data_path, sep='\t', names=['user_id', 'item_id', 'rating'])

    #### preparing data
    df = df[['user_id', 'item_id', 'rating']]

    user_groups = torch.tensor(list(user_groups.values())).to(device)
    item_groups = torch.tensor(list(item_groups.values())).to(device)
    
    indexes_sample = np.load(f'data/temp_data/temp_indexes_{data_path.split('/')[-1].split('.')[0]}_run{run_i}.npz')
    data = df.loc[indexes_sample]

    rows_idx, cols_idx = mapping_indexes(np.zeros(len(data)), data.to_numpy()[:, 0], data.to_numpy()[:, 1])
    data.iloc[:,0], data.iloc[:,1] = rows_idx, cols_idx

    true_interactions = torch.tensor(pivot_gen_score_matrix(data['rating'], data['user_id'], data['item_id']))
    if k > true_interactions.size(1):
        raise ValueError(
            f"k must not exceed the number of sampled items "
            f"({true_interactions.size(1)}), got {k}"
        )
    if len(user_groups) != true_interactions.size(0):
        raise ValueError(
            "user_groups size must match the sampled score matrix user dimension"
        )
    if len(item_groups) != true_interactions.size(1):
        raise ValueError(
            "item_groups size must match the sampled score matrix item dimension"
        )
    
    # normalizing between 0-1
    true_interactions = true_interactions/torch.max(true_interactions)
    true_interactions = true_interactions.to(device)
    #print(true_interactions.device, user_groups.device, item_groups.device)
    _, pred_scores = train(
        true_interactions,
        user_groups,
        item_groups,
        emb_size=emb_size,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        alpha=alpha,
        gamma=gamma,
        tau=tau,
        k=k,
        n_samples=n_samples,
        device=device,
    )
    print(pred_scores.size(), k)
    _, fair_topk = pred_scores.topk(k=k)

    fair_topk = fair_topk.to(device)

    # normalizing pred_scores between 0-1
    pred_scores = (pred_scores - torch.min(pred_scores))/(torch.max(pred_scores) - torch.min(pred_scores))

    return pred_scores, fair_topk



# train while updating the matrix factorization embeddings
def train(true_interactions: torch.Tensor, user_groups: torch.Tensor, item_groups: torch.Tensor,
             emb_size: int=32, batch_size: int=100, epochs: int=200, lr: float=1e-2,
             alpha: float=.0, gamma: float=.8, tau: float=0.1, k: int=100, n_samples: int=100, 
             device: str=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    n_users, n_items = true_interactions.size()
    n_user_groups = user_groups.size() 
    n_item_groups = item_groups.size() 

    # assert n_users == len(user_groups), f"User count mismatch: {n_users} from interactions vs {len(user_groups)} from groups."
    # assert n_items == len(item_groups), f"Item count mismatch: {n_items} from interactions vs {len(item_groups)} from groups."

    print(f"Using device: {device}")

    # DataLoader for user batches
    user_dataset = TensorDataset(torch.arange(n_users, device=device))
    user_loader = DataLoader(user_dataset, batch_size=batch_size)

    model = MatrixFactorization(n_users, n_items, emb_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        #progress bar
        pbar = tqdm.tqdm(user_loader, desc=f"Epoch {epoch+1}/{epochs}")

        losses = []

        for (batch_user_ids,) in pbar:
            optimizer.zero_grad()

            # Get scores for all items for the users in the batch
            pred_scores = model(batch_user_ids, torch.arange(n_items, device=device))
        
            # Get corresponding ground truth and group info for the batch
            true_scores_batch = true_interactions[batch_user_ids]
            user_groups_batch = user_groups[batch_user_ids]

            #print(pred_scores.device, true_scores_batch.device)

            # Compute loss
            # loss = loss_fn(pred_scores, true_scores_batch, user_groups_batch, item_groups)
            loss = F.mse_loss(pred_scores.to(float), true_scores_batch.to(float))

            #print(loss.dtype, pred_scores.dtype, true_scores_batch.dtype)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        if (epoch+1)%100 == 0:
            avg_loss = total_loss / len(user_loader)
            print(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.4f}")
            losses.append(avg_loss)

    # print("Training finished.")
    # print(f'pred_scores: {pred_scores.size()}')

    pred_scores = model(torch.arange(n_users, device=device), torch.arange(n_items, device=device))
    print(f'pred_scores: {pred_scores.size()}, true interactions: {true_interactions.size()}')
    return losses, pred_scores
