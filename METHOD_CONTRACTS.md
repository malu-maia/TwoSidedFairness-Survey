# Recommendation Method Contracts

Each method module exposes one public entry point whose name matches the
module filename. Experiment code should call these functions instead of
calling internal training helpers directly.

## Common Pattern

- Prepared inputs are validated at the public function boundary.
- Score matrices use shape `(n_users, n_items)` unless stated otherwise.
- Top-k outputs contain item indices with shape `(n_users, k)`.
- Functions do not write result files. The experiment runner owns persistence.
- Existing `train` functions remain available as compatibility wrappers where
  they were previously the public entry point.
- Returned tensors stay on the algorithm's existing device. Callers must move
  and detach tensors before converting them to NumPy or writing result files.

## Re-Ranking Methods (`rr`)

| Method | Main function and inputs | Data files | Outputs |
| --- | --- | --- | --- |
| FairSort | `fairsort(scores, item_to_provider, k=10, fairness="uniform", lambda_max=8.0, rerank_ratio=0.1, ndcg_floor=0.95, search_gap=0.03125, user_order=None, eligible_items=None)` | None. `item_to_provider` must contain one provider ID per item. Optional `eligible_items` masks unavailable user-item pairs. | `(scores, fair_topk)` on the score tensor's device. Fairness targets are provider-size proportional for `uniform` and aggregate-score proportional for `quality`. |
| PCT | `pct(data_path, run_i, item_mapping=None, num_groups=2, k=10, lambda_tradeoff=0.5)` | Reads the tab-separated interaction file and `data/temp_data/temp_indexes_<dataset>_run<run_i>.npz`. Expected interaction columns are `user_id`, `item_id`, `rating`, and optional `timestamp`. | `(scores, fair_topk)`, both CPU tensors. `scores` is `(n_users, n_items)` and `fair_topk` is `(n_users, k)`. |
| TFROM | `tfrom(scores, k=10, item_provider_mapping=None)` | None. If no provider mapping is supplied, providers are generated randomly in memory. | `(scores, fair_topk)` on the score tensor's device. |

## Position-Agnostic Methods (`pam`)

| Method | Main function and inputs | Data files | Outputs |
| --- | --- | --- | --- |
| FairRec | `fairrec(scores, k)` | None. | `(scores, fair_topk)`. The score tensor is unchanged; `fair_topk` is created on CPU. Legacy `train` delegates to `fairrec`. |
| Fermatean Fuzzy | `fermatean_fuzzy(side_a_preferences, side_b_preferences, eligible_pairs=None, side_a_weight=0.5, side_b_weight=0.5)` | None. | `(fair_scores, matched_pairs)` on the side-A tensor's device. `matched_pairs` contains `(side_a_index, side_b_index)` rows from a one-to-one fair matching. This method does not return top-k recommendations. |
| ITFR | `itfr(user_map, item_map, interactions, k, epochs=10, ...)` | None. The caller prepares maps and grouped interactions. Required map columns are `user_id`, `user_group`, `item_id`, and `item_group`. | `(fair_scores, fair_topk)` on the selected device. Legacy `train` delegates to `itfr`. |
| CPFair | `cpfair(scores, user_groups, item_groups, base_topk, lambda1=0.5, lambda2=0.5, k=10, *, device=None)` | None. | `(fair_scores, fair_topk)` on the selected device, or the score tensor's device when `device` is omitted. |
| SEAL | `seal(scores, k, product_min=None, product_max=None)` | None. | `(scores, fair_topk)` on the score tensor's device. The user-side cardinality constraints are fixed to `L1 = L2 = k`. Defaults are `product_min=floor(n_users*k/n_items)` and `product_max=n_users`. Legacy `train` delegates to `seal`. |

## Optimization-Based Methods (`obm`)

| Method | Main function and inputs | Data files | Outputs |
| --- | --- | --- | --- |
| Ada2Fair | `ada2fair(interactions, item_provider_mapping, k=10, epochs=100, weight_epochs=10, embedding_dim=64, generator_hidden_dim=32, adapter_hidden_dim=16, batch_size=4096, learning_rate=1e-3, alpha=0.5, provider_eta=2.0, user_eta=1.0, delta=1e-7, weight_topk=100, seed=42, eligible_items=None, exclude_seen=True, device=None)` | None. Positive entries are explicit observed ratings and zero denotes missing data. `item_provider_mapping` must contain the actual provider ID for every item. | Trains the adaptive generator, fairness adapters, and matrix-factorization recommender jointly. The recommender minimizes a weighted graded-pair BPR loss: each observed (user, item) pair is ranked against a sampled lower-rated item (unrated counts as rating 0) and weighted by the generator's weight for the positive interaction. Customer utility uses normalized graded ratings. Returns `(learned_scores, fair_topk)` on the selected device. Legacy `train` delegates to `ada2fair`. |
| LeadFairRec | `leadfairrec(interactions, item_provider_mapping, scores, k=10, user_profile_embeddings=None, item_profile_embeddings=None, semantic_weight=0.0, gamma=-256.0, delta=64.0, beta=1.0, similar_user_count=30, eligible_items=None, exclude_seen=True, device=None)` | None. Positive entries in `interactions` are observed implicit feedback used to estimate UA, IP, and PBE. `scores` must be the standardized base recommender matrix with the same shape as `interactions`. `item_provider_mapping` must contain one provider ID per item. Experiment code for LeadFairRec's LLM-enhanced setting should call `generate_deepseek_profiles`, convert the returned text with `profile_text_embeddings`, then pass the resulting user/item profile embeddings here. | Adds optional semantic profile affinity to the supplied base scores, subtracts the LeadFairRec counterfactual bias term, then ranks eligible items. Returns `(fair_scores, fair_topk)` on the selected device. Legacy `train` delegates to `leadfairrec`. |
| MultiFR | `multifr(interactions, user_groups, item_groups, k=10, positive_threshold=4.0, epochs=100, embedding_dim=50, batch_size=1024, learning_rate=1e-3, regularization=1e-4, fairness_k=20, gamma=0.8, temperature=0.1, gumbel_samples=1, rounds=5, seed=42, exclude_seen=True, device=None)` | None. Every positive entry is an observed graded rating; zeros are unobserved. `positive_threshold` is deprecated and ignored. Group tensors contain one group ID per user or item. | Trains graded-pair BPR matrix factorization (pairs satisfy rating(i) > rating(j), unrated counts as 0) against accuracy, user-group SmoothNDCG parity with graded gains, and item-group exposure over the observed set. MGDA/Frank-Wolfe learns objective weights and Least Misery selects among independent rounds. Returns `(learned_scores, fair_topk)` on the selected device. |
| Fair Reciprocal | `fr(scores, n_users, epochs=200, welfare_function="SW", ...)` | None. `scores` is a square two-sided matrix with shape `(n_users + n_items, n_users + n_items)`. | For `SW`: `(fair_scores, welfare_history, user_utilities)`. For `NSW`: `(fair_scores, stochastic_policy, user_utilities, item_utilities)`. Legacy `train` delegates to `fr`. |
| GGF | `ggf(scores, beta=1000, epochs=5000, lamb=0.5, ...)` | None. | `(fair_scores, stochastic_policy, user_utilities, item_utilities)`. Legacy `train` delegates to `ggf`. |
| JME | `jme(data_path, user_groups, run_i, item_groups, ..., k=100)` | Reads the tab-separated interaction file and `data/temp_data/temp_indexes_<dataset>_run<run_i>.npz`. Expected interaction columns are `user_id`, `item_id`, `rating`, and optional `timestamp`. | `(fair_scores, fair_topk)` on the selected device. The module's `train` remains an internal matrix-factorization helper. |
| TFLD | `tfld(scores, epochs=1000, k=10, alpha=(0.0, 0.0), lamb=0.5, ...)` | None. | `(fair_scores, stochastic_policy, user_utilities, item_utilities)`. Legacy `train` delegates to `tfld`. |
| TFLD-Groups | `tfld_groups(scores, user_groups, item_groups, epochs=1000, k=10, alpha=(0.0, 0.0), lamb=0.5, eta=1e-9, device=None)` | None. `user_groups` and `item_groups` hold one label per user and per item; any labels and any encoding (`{0,1}` or CPFair's `{-1,+1}`) are accepted and normalized internally. | The Appendix B group extension of TFLD: each individual utility is replaced by its group total before the welfare function is applied, so the Frank-Wolfe direction weights every entity by its group's marginal welfare. Both `alpha` values must be below 1, where concavity and the Proposition 5 Lorenz guarantee hold. Returns `(expected_exposure, fair_topk, user_group_utilities, item_group_utilities)` on the selected device. **Deviates from `tfld`/`ggf`:** it returns the `(n_users, n_items)` expected-exposure matrix and a ready `(n_users, k)` top-k rather than an `(n_users, n_items, k)` stochastic policy. Every quantity the algorithm needs depends on the policy only through its expected exposure, and the Frank-Wolfe update is linear, so the two are equivalent; tracking exposure alone costs ~`k` times less memory. |
| TSFD | `tsfd(scores, user_groups, item_groups, k=10, exposure_steepness=1.0, item_fairness="one_sided", maxiter=1000, tol=1e-8, seed=42)` | None. The original paper is intent-aware; this implementation uses user groups as proxy intents and estimates item-intent relevance from mean scores within each user group. | `(scores, fair_topk)` on the score tensor's device. It solves the TSFD marginal-rank convex program for user and item fairness, then applies the greedy Birkhoff-von Neumann decomposition with diversity-improving local search. Legacy `train` delegates to `tsfd`. |

```text
results/<method>/<dataset>_<result_type>_<n_interactions>_interactions_<run>.parquet
```

The standard recommendation result types are `fair_scores` and `top_k`.
Methods that expose stakeholder utilities additionally produce
`users_utility` and `items_utility`. Fermatean Fuzzy is a matching method and
returns `matched_pairs` instead of `top_k`.
