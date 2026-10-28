#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../../.." && pwd)"
cd "$repo_dir"
reference_dir="${PGMT_REFERENCE_DIR:-${repo_dir}-reference}"
export DISPLAY="${DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-/run/user/1000/gdm/Xauthority}"
export PYTHONPATH="$repo_dir/source/instinctlab:$reference_dir/instinct_rl:$reference_dir/play_deps${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec "${PGMT_MUJOCO_PYTHON:-$repo_dir/.venv-play/bin/python}" -u scripts/pgmt/play_mujoco.py \
  --checkpoint "${PGMT_CHECKPOINT:-$repo_dir/artifacts/pgmt_lafan_terrain_play/model_9600/model_9600.pt}" "$@"
