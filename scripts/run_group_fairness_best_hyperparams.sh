#!/usr/bin/env bash
set -euo pipefail

BEST_HYPERPARAMS=${BEST_HYPERPARAMS:-experiments/best_hyperparams.json} \
RESULTS_ROOT=${RESULTS_ROOT:-results/best_hyperparams} \
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/run_group_fairness.sh" grouped_experiments "$@"
