#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

case ${1:-} in
  grouped_experiments|rq3)
    runner=$1
    shift
    ;;
  *) runner=grouped_experiments ;;
esac

mode=${1:-all}
if [[ $# -gt 1 || ! $mode =~ ^(all|prepare|run)$ ]]; then
  printf 'Usage: bash scripts/run_%s.sh [all|prepare|run]\n' "$runner" >&2
  exit 2
fi

python=${PYTHON:-.venv/bin/python}
run_count=${RUNS:-10}
base_seed=${SEED:-42}
k_values=(10 25 50 100 150)
device=${DEVICE:-cpu}
tfld_epochs=${TFLD_EPOCHS:-200}
runs_root=${RUNS_ROOT:-experiments/runs/group_fairness}
results_root=${RESULTS_ROOT:-results_11}
best_hyperparams=${BEST_HYPERPARAMS:-}

if [[ ! $run_count =~ ^[1-9][0-9]*$ || ! $base_seed =~ ^(0|[1-9][0-9]*)$ ]]; then
  printf 'RUNS must be a positive integer and SEED a nonnegative integer.\n' >&2
  exit 2
fi

if [[ $runner == rq3 ]]; then
  k_values=(150)
elif [[ -n ${K:-} ]]; then
  if [[ ! $K =~ ^[1-9][0-9]*$ ]]; then
    printf 'K must be a positive integer.\n' >&2
    exit 2
  fi
  k_values=("$K")
fi

if ! command -v "$python" >/dev/null; then
  printf 'Python executable not found: %s\n' "$python" >&2
  exit 2
fi

if [[ $mode != prepare && -n $best_hyperparams && ! -f $best_hyperparams ]]; then
  printf 'Best hyperparameter file not found: %s\n' "$best_hyperparams" >&2
  exit 2
fi

if [[ $mode != run ]]; then
  common=(--output-root "$runs_root" --runs "$run_count" --seed "$base_seed"
          --recommendation-list-length "${k_values[-1]}")
  caps=(--max-users 3500 --max-items 3500)
  "$python" -m experiments.prepare_explicit_method_inputs "${common[@]}" \
    --datasets ambar --no-shared-run-zero "${caps[@]}"
  "$python" -m experiments.prepare_explicit_method_inputs "${common[@]}" \
    --datasets movielens --movielens-director-gender "${caps[@]}"
  "$python" -m experiments.prepare_explicit_method_inputs "${common[@]}" \
    --datasets amazon --amazon-brand --allow-missing-demographics --no-shared-run-zero
  "$python" -m experiments.prepare_explicit_method_inputs "${common[@]}" \
    --datasets yelp --yelp-partition cpfair-territorial "${caps[@]}"
fi

if [[ $mode == prepare ]]; then
  exit 0
fi

run_dirs=()
for partition in ambar movielens/director-gender amazon/brand yelp/cpfair-territorial; do
  for ((run = 0; run < run_count; run++)); do
    printf -v run_name 'run_%02d' "$run"
    run_dir="$runs_root/$partition/$run_name/seed_$((base_seed + run))"
    if [[ ! -f $run_dir/metadata.json ]]; then
      printf 'Prepared run missing: %s. Use the prepare or all mode first.\n' "$run_dir" >&2
      exit 1
    fi
    run_dirs+=(--run-dir "$run_dir")
  done
done

mkdir -p -- "$results_root"
result_dir=$(mktemp -d -- "$results_root/$runner.XXXXXX")
printf 'Writing results to %s\n' "$result_dir"

common=(--device "$device" --tfld-epochs "$tfld_epochs"
        --yelp-tfld-partition cpfair-territorial "${run_dirs[@]}")
if [[ -n $best_hyperparams ]]; then
  cp -- "$best_hyperparams" "$result_dir/best_hyperparams.json"
  common+=(--best-hyperparams "$result_dir/best_hyperparams.json")
fi
for k in "${k_values[@]}"; do
  printf 'Running %s with k=%s\n' "$runner" "$k"
  k_results="$result_dir/k_$k"
  if [[ $runner == rq3 ]]; then
    "$python" -m experiments.RQ3_groups "${common[@]}" --results-root "$k_results" --k "$k" \
      --dataset ambar --dataset movielens --dataset amazon --dataset yelp \
      --method cpfair --method tfld-groups
  else
    "$python" -m experiments.grouped_experiments "${common[@]}" --results-root "$k_results" \
      --recommendation-list-size "$k" --include-tfld-groups
  fi
done
