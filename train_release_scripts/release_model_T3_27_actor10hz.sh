#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PGMT_TRAIN_RUN_NAME="${PGMT_TRAIN_RUN_NAME:-$(basename -- "${BASH_SOURCE[0]}" .sh)}"
export PGMT_ACTOR_HEIGHT_SCAN_HZ=10
exec bash "$script_dir/release_model_T3_27.sh"
