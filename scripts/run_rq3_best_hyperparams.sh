#!/usr/bin/env bash
set -euo pipefail

BEST_HYPERPARAMS=${BEST_HYPERPARAMS:-results/hyperparameter_sensitivity/shared_cutoff_per_k/best_hyperparams_by_k.json} \
RESULTS_ROOT=${RESULTS_ROOT:-results/best_hyperparams} \
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/run_group_fairness.sh" rq3 "$@"
