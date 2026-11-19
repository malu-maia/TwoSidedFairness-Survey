#!/usr/bin/env bash
set -euo pipefail

exec bash "$(dirname -- "${BASH_SOURCE[0]}")/run_group_fairness.sh" rq3 "$@"
