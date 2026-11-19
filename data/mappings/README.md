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

## Amazon brands

Use `experiments.prepare_explicit_method_inputs --datasets amazon --amazon-brand`
to generate an Amazon All Beauty partition under
`experiments/runs/amazon/brand/`. The preprocessor joins product metadata
by ASIN, normalizes brand labels for case and whitespace, and excludes ASINs
with missing, generic, or conflicting brand labels before splitting.

| Run artifact | Mapping |
| --- | --- |
| `provider_item_mapping.csv` | Internal item ID and original ASIN to internal provider ID and normalized brand |
| `groups/provider_group_mapping.csv` | Brand to `short_head` (0) or `long_tail` (1), with its total training interactions |
| `groups/cpfair_item_brand_popularity.csv` | Each product inherits its brand's group using CPFair labels −1 and +1 |

The first run also exports aligned copies under the `amazon_brand_*` prefix in
`data/mappings/` and `data/evaluation/`. Read the provider and item mappings from
the same run as the rankings.

## MovieLens users

The MovieLens preparation also writes `movielens_user_gender_mapping.csv` from
`Movie Lens/users.dat`. Its `user_id` values are original MovieLens IDs, rather
than run-specific IDs; `gender` is `F` or `M`, and `user_group` is +1 for F
and -1 for M. Experiments align it with each run using `mappings.json`. The
existing `movielens_user_group_mapping.csv` retains its activity-based groups.

## MovieLens directors

`experiments.movielens_director_mapping` joins the original MovieLens 1M movie
IDs to IMDb IDs from [MovieLens 20M](https://grouplens.org/datasets/movielens/20m/),
then queries Wikidata for films, directors, and their recorded gender. First
download the 20M archive and extract only its links file:

```bash
curl --fail --location https://files.grouplens.org/datasets/movielens/ml-20m.zip --output /tmp/ml-20m.zip
mkdir -p 'data/raw_datasets/explicit_feedback/Movie Lens/auxiliary_data'
unzip -p /tmp/ml-20m.zip ml-20m/links.csv > 'data/raw_datasets/explicit_feedback/Movie Lens/auxiliary_data/links-20m.csv'
python -m experiments.movielens_director_mapping
```

The resulting `Movie Lens/auxiliary_data/director_gender.csv` retains every 1M movie, including
unmatched records. IDs are original MovieLens IDs; use each run's
`mappings.json` to align them with internal item IDs. A film can have multiple
directors, and a director can have multiple recorded gender values, so there
can be multiple rows per film. `match_status` describes missing IMDb IDs,
Wikidata films, directors, or gender. `matched` means a gender statement was
found; it does not assert that the value is binary or self-reported. Results
reflect the Wikidata snapshot at query time. The preparation step classifies a
film as protected if at least one director has a `female` or `trans woman`
gender statement; a film is in the reference group if all director statements
are `male`. All other films are excluded from the separate director-gender run.
`movielens_item_director_gender_mapping.csv` covers the full MovieLens 1M
movie catalogue using original `movie_id` values; `item_group` is +1, -1, or
empty for unknown. This replaces the old shared run-zero format. The per-run
`groups/cpfair_item_director_gender.csv` contains only retained films and uses
the run's internal `item_id` values. Preparation records the mapping and source
hashes, classification rule, and filtering counts in the run metadata.
`--no-shared-run-zero` leaves the shared CSV untouched and uses the same
classification in memory.
Wikidata may rate-limit requests; completed batches are saved in a checkpoint
next to the CSV so the command can be resumed after an interruption. Remove
that checkpoint before requesting a completely fresh snapshot.

## Yelp metropolitan areas

`prepare_explicit_method_inputs` automatically creates the geographic mapping
needed by the selected Yelp partition when it does not already exist. It uses
the functions in `experiments.yelp_metro_mapping` for TFLD-Groups and
`experiments.yelp_census_mapping` for CPFair. Install their
optional dependency with `uv sync --extra geographic` (or install
`geopandas>=1.0,<2` into the project's Python environment).

Supply these official boundary files locally; the script does not download them:

- US: [2020 Census cartographic CBSA boundaries, 1:500,000](https://www.census.gov/geographies/mapping-files/time-series/geo/cartographic-boundary.2020.html),
  national Core Based Statistical Areas shapefile. It must contain `GEOID`.
- Canada: [Statistics Canada 2021 CMA/CA boundary file](https://www150.statcan.gc.ca/n1/en/catalogue/98260002),
  including Edmonton (`CMAUID` 835). Use the national file with `CMAUID`.

Download and extract the files according to their providers' terms. Record the
download date and any choice between digital and cartographic Canadian boundaries
with the experiment; boundaries are a snapshot, not the historical locations of
all reviews. The 2020 US file is a simplified cartographic representation.

Alternatively, create or refresh the mapping explicitly using the boundary files
placed in Yelp's auxiliary-data directory:

```bash
python -m experiments.yelp_metro_mapping \
  --businesses data/raw_datasets/explicit_feedback/Yelp/yelp_academic_dataset_business.json \
  --us-boundaries data/raw_datasets/explicit_feedback/Yelp/auxiliary_data/boundaries/us_cbsa_2020/cb_2020_us_cbsa_500k.shp \
  --canada-boundaries data/raw_datasets/explicit_feedback/Yelp/auxiliary_data/boundaries/edmonton_cma_2021.geojson \
  --output-dir data/mappings/yelp_metro
```

Add `--run experiments/runs/yelp/tfld-metro/run_00/seed_42` for a prepared run, or repeat
`--run` for more runs. The Edmonton GeoJSON contains only the Edmonton CMA,
retrieved from Statistics Canada's 2021 cartographic boundary service. The
auxiliary boundary directory includes a README with its exact source.
The output directory contains `yelp_business_metro.csv`, a metadata JSON with
the input paths and region-code dictionary, and one `yelp_items_<run>_<seed>.csv`
per requested run. `item_id` follows that run's `mappings.json`; a missing
business or duplicate ID causes an error before output is written.

`region_code` is a stable integer from 0 to 10; -1 and `unmatched`
represent an unknown region. `match_status` distinguishes invalid coordinates,
points outside the selected boundaries, and ambiguous intersections. Points on
the boundary of a single polygon are assigned to it; points intersecting more
than one region are ambiguous. The script prints region and status counts.
These are geographic labels, not evidence of protected status or disadvantage.
Check coverage after the experiment's own filters and review unknowns before
using any mapping in a method.

## Yelp census poverty mapping

The CPFair territorial partition requires [2019 national tract boundaries](https://www2.census.gov/geo/tiger/GENZ2019/shp/cb_2019_us_tract_500k.zip) and
[ACS 2015–2019 detailed table B17001](https://www2.census.gov/programs-surveys/acs/summary_file/2019/prototype/5YRData/acsdt5y2019-b17001.dat).
Download both into `data/raw_datasets/explicit_feedback/Yelp/auxiliary_data/census_acs_2019/`.
Preparation creates the mapping automatically if missing; to create or refresh it
explicitly, run:

```bash
python -m experiments.yelp_census_mapping \
  --businesses data/raw_datasets/explicit_feedback/Yelp/yelp_academic_dataset_business.json \
  --tracts data/raw_datasets/explicit_feedback/Yelp/auxiliary_data/census_acs_2019/cb_2019_us_tract_500k.zip \
  --poverty data/raw_datasets/explicit_feedback/Yelp/auxiliary_data/census_acs_2019/acsdt5y2019-b17001.dat \
  --output data/mappings/yelp_census/yelp_business_poverty.csv
```

The rate is `100 * B17001_E002 / B17001_E001`, with invalid or zero denominators
left unknown. Values >=20% map to +1 and lower values to -1. Coordinates outside
tracts, ambiguous intersections, and missing estimates remain unknown. The data
describe the surrounding tract, not the business owner's income. The 2015–2019
estimates were released in December 2020, so document their temporal relationship
to the interactions; this is not a strictly time-causal analysis by default.
