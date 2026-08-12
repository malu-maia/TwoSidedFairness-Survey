import numpy as np
import pandas as pd
import torch

from scipy.optimize import linprog
from collections import defaultdict
from typing import Mapping

from utils import mapping_indexes, pivot_gen_score_matrix, pref_estimation

NAME = 'PCT'

def create_item_group_mapping(item_ids: list, num_groups: int = 3) -> dict:
    mapping = {item_id: np.random.randint(0, num_groups) for item_id in item_ids}
    return mapping

def df_with_group(df_interactions: pd.DataFrame, item_col: str, num_groups: int=3, item_mapping=None) -> pd.DataFrame:
    placeholder = np.zeros(len(df_interactions))
    data = df_interactions.to_numpy()
    rows_idx, cols_idx = mapping_indexes(placeholder, data[:, 0], data[:, 1])
    df_interactions.iloc[:,0], df_interactions.iloc[:,1] = rows_idx, cols_idx
    df_interactions_w_group = df_interactions.copy()

    items_id = df_interactions_w_group[item_col].unique()

    # Assigning random groups
    if item_mapping == None:
        mapping = create_item_group_mapping(items_id, num_groups)
    else:
        mapping = item_mapping

    df_interactions_w_group['item_group'] = df_interactions_w_group[item_col].map(mapping)
    return df_interactions_w_group, mapping

def compute_historical_interest_distribution(df_interactions_w_group: pd.DataFrame, user_col: str):#, num_groups: int) -> dict:
    num_groups = len(df_interactions_w_group['item_group'].unique())
    
    user_interest = {}
    # group by users
    user_group_counts = df_interactions_w_group.groupby([user_col, 'item_group']).size().unstack(fill_value=0)

    total_interactions = user_group_counts.sum(axis=1)
    p_u_df = user_group_counts.div(total_interactions, axis=0)
    for user_id in p_u_df.index:
        user_interest[user_id] = p_u_df.loc[user_id].values
        # dist = np.zeros(num_groups)
        # dist[p_u_df.columns] = p_u_df.loc[user_id].values
        # user_interest[user_id] = dist
        
    return user_interest

def ranking_weights(k: int) -> np.ndarray:
    # ranks vai de 1 a K. Adicionamos 1 para evitar log(1)=0.
    ranks = np.arange(1, k + 1)
    return 1 / np.log2(ranks + 1)


# not used since the ideal q_u is computed by pct_solver 
def compute_user_group_exposure(reco_list: list, item_group_map: dict, num_groups: int, r_k: np.ndarray) -> np.ndarray:
    k = len(reco_list)
    q_u = np.zeros(num_groups)

    for i, item_id in enumerate(reco_list):
        group = item_group_map.get(item_id)
        if group is not None:
            q_u[group] += r_k[i]
            
    total_weight = np.sum(r_k)
    if total_weight > 0:
        q_u /= total_weight
        
    return q_u



def pct_solver(p_users: dict, q_target: np.ndarray) -> dict:
    """
    Find the hat_q_u for each user.
    """
    user_ids = list(p_users.keys())
    #print(list(p_users.values()))
    p_matrix = np.array([p_users[uid] for uid in user_ids])
    num_users, num_groups = p_matrix.shape

    # p_bar: average interest distribution
    p_bar = np.mean(p_matrix, axis=0)
    
    # computing gradient g direction
    print(p_bar, q_target)
    grad = p_bar - q_target
    # get the gradient of the distance wrt p
    g = grad / np.linalg.norm(grad)

    # Setting the linear problem such as equation 6.
    c = np.ones(num_users)

    # Constraints: sum(gamma_u * g) = sum(p_u - q_target)
    A_eq = g.reshape(1, -1).repeat(num_users, axis=0).T.reshape(num_groups, num_users)
    b_eq = np.sum(p_matrix - q_target, axis=0)
    
    # Bounds of gamma constraints (0 <= gamma_u <= l_u)
    bounds = np.zeros((num_users, 2))
    for i in range(num_users):
        p_u = p_matrix[i]
        # Computing the max step size l_u for each user to guarantee that: 0 <= hat_q_u <= 1
        # Avoiding division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            upper_limits = np.where(g > 1e-9, p_u / g, np.inf)
            lower_limits = np.where(g < -1e-9, (p_u - 1) / g, np.inf)
        
        l_u = min(np.min(upper_limits), np.min(lower_limits))
        bounds[i] = (0, l_u)

    # Solving the linear problem
    #print(f'A: {A_eq}\nb: {b_eq}')
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    #print(f'result: {result}')
    if not result.success:
        print("Failed to optimize, verify the constraints.")
        return p_users
        
    gammas = result.x

    # Computing hat_q_u = p_u - gamma_u * g
    hat_q_users = {}
    for i, user_id in enumerate(user_ids):
        hat_q_u = p_matrix[i] - gammas[i] * g
        hat_q_users[user_id] = np.clip(hat_q_u, 0, 1)
        hat_q_users[user_id] /= np.sum(hat_q_users[user_id]) 

    return hat_q_users


def pct_reranker(
    original_recos: list,
    hat_q_u: np.ndarray,
    item_group_map: dict,
    k: int,
    lambda_tradeoff: float = 0.5
) -> list:
    num_groups = len(hat_q_u)
    final_recos = [-1] * k
    selected_items = set()
    
    r_k = ranking_weights(k)
    total_resource = np.sum(r_k)
    target_resource = hat_q_u * total_resource 
    current_exposure = np.zeros(num_groups)

    for i in range(k): # Itera sobre as posições na lista final
        for item_id in original_recos:
            group = item_group_map.get(item_id)
            if item_id not in selected_items and group is not None:
                # Verifica se adicionar este item excede o recurso de exposição do grupo
                if current_exposure[group] + r_k[i] <= target_resource[group] + 1e-9: # Adiciona tolerância
                    final_recos[i] = item_id
                    selected_items.add(item_id)
                    current_exposure[group] += r_k[i]
                    break # Passa para a próxima posição

    # preencher vagas com MMR modificado
    vacancies = [i for i, item in enumerate(final_recos) if item == -1]
    
    # prepara listas de candidatos por grupo
    candidates_by_group = defaultdict(list)
    for rank, item_id in enumerate(original_recos):
        if item_id not in selected_items:
            group = item_group_map.get(item_id)
            if group is not None:
                candidates_by_group[group].append((item_id, rank))

    for i in vacancies:
        best_item = -1
        max_score = -np.inf
        
        top_candidates = {}
        for group_id, candidates in candidates_by_group.items():
            if candidates:
                top_candidates[group_id] = candidates[0] # Item de maior classificação não selecionado

        if not top_candidates: 
            break

        # Calcula a pontuação de relevância marginal para cada top candidato
        for group_id, (item_id, rank) in top_candidates.items():
            relevance_score = 1 / (rank + 1)

            # Disparidade: usa distância L2 
            assumed_exposure = current_exposure.copy()
            assumed_exposure[group_id] += r_k[i]
            disparity = np.sum((assumed_exposure - target_resource)**2)

            # Pontuação marginal 
            score = lambda_tradeoff * relevance_score - (1 - lambda_tradeoff) * disparity

            if score > max_score:
                max_score = score
                best_item = item_id
                best_group = group_id
        
        if best_item != -1:
            final_recos[i] = best_item
            selected_items.add(best_item)
            current_exposure[best_group] += r_k[i]
            # Remove o item selecionado da lista de candidatos
            candidates_by_group[best_group].pop(0)

    # Preenche as vagas restantes caso algum erro ocorra
    unselected_items = [item for item in original_recos if item not in selected_items]
    for i in range(k):
        if final_recos[i] == -1 and unselected_items:
            final_recos[i] = unselected_items.pop(0)

    return list(map(int, final_recos))

def pct(
    data_path: str,
    run_i: int,
    item_mapping: Mapping[int, int] | None = None,
    num_groups: int = 2,
    k: int = 10,
    lambda_tradeoff: float = .5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run PCT from an interaction file and sampled-row index file.

    The interaction file is read as tab-separated ``user_id``, ``item_id``,
    ``rating`` and optional ``timestamp`` columns. Sampled row indices are read
    from ``data/temp_data/temp_indexes_<dataset>_run<run_i>.npz``.

    Returns:
        The estimated score matrix and reranked top-k item indices.
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    if num_groups < 1:
        raise ValueError(f"num_groups must be at least 1, got {num_groups}")
    if not 0 <= lambda_tradeoff <= 1:
        raise ValueError(
            f"lambda_tradeoff must be between 0 and 1, got {lambda_tradeoff}"
        )
    if item_mapping is not None and not item_mapping:
        raise ValueError("item_mapping must not be empty when provided")

    if item_mapping is not None:
        num_groups = len(set(list(item_mapping.values())))
    
    try:
        df = pd.read_csv(data_path, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    except:
        df = pd.read_csv(data_path, sep='\t', names=['user_id', 'item_id', 'rating'])

    df = df[['user_id', 'item_id', 'rating']]
    
    indexes_sample = np.load(f'data/temp_data/temp_indexes_{data_path.split('/')[-1].split('.')[0]}_run{run_i}.npz')
    data = df.loc[indexes_sample]

    df_w_groups, mapping = df_with_group(data, 'item_id', item_mapping=item_mapping)

    scores = pivot_gen_score_matrix(df_w_groups['rating'], df_w_groups['user_id'], df_w_groups['item_id'])
    scores = torch.tensor(pref_estimation(scores))
    if k > scores.size(1):
        raise ValueError(
            f"k must not exceed the number of sampled items ({scores.size(1)}), got {k}"
        )

    _, initial_recs_idx = torch.topk(scores, k=k)

    initial_recs = {}
    for i in range(len(initial_recs_idx)):
        initial_recs[i] = initial_recs_idx[i]
 
    p_u = compute_historical_interest_distribution(df_w_groups, 'user_id')
    #print(f'p_u: {p_u}')
    # Garante que apenas usuários com p_u e recomendações sejam usados
    common_user_ids = set(p_u.keys()).intersection(set(initial_recs.keys()))
    p_u = {uid: p_u[uid] for uid in common_user_ids}

    initial_recs = {uid: initial_recs[uid] for uid in common_user_ids}
    
    #q_u = compute_user_group_exposure(initial_recs, mapping, num_groups=3, r_k=r_k)
    q_target = np.ones(num_groups)*(1/num_groups)
    hat_q_users = pct_solver(p_u, q_target)

    fair_recs = {}
    for u, u_rec in initial_recs.items():
        hat_q_u = hat_q_users.get(u)
        if hat_q_u is not None:
            fair_recs[u] = pct_reranker(u_rec, hat_q_u, mapping, k=k, lambda_tradeoff=lambda_tradeoff)

    return scores, torch.tensor(list(fair_recs.values()))
