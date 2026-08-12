# Dataset Mapping Files

Group-level experiment configurations expect dataset-qualified CSV files in
this directory. Stakeholder memberships must be derived from the corresponding
dataset; they are not generated automatically.

For each dataset slug (`amazon`, `black_friday`, `movielens`, and `yelp`), the
following files are expected:

| Filename pattern | Required columns |
| --- | --- |
| `<dataset>_user_group_mapping.csv` | `user_id,user_group` |
| `<dataset>_item_group_mapping.csv` | `item_id,item_group` |
| `<dataset>_provider_group_mapping.csv` | `provider_id,provider_group` |
| `<dataset>_provider_item_mapping.csv` | `item_id,provider_id` |

Identifiers must use the same ID space as the relevance, interaction,
exposure, and merit files referenced by the experiment configuration.
