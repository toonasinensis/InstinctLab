#!/usr/bin/env bash
# Source agents/computation/activate-gpu8.sh on the compute host before launch.
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_dir"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
run_name="${RUN_NAME:-$(TZ=Asia/Shanghai date +%m%d_%H%M)__PGMT_LAFAN_Newton}"
exec torchrun --standalone --nproc_per_node="${NUM_GPUS:-4}" scripts/instinct_rl/train.py \
  --task Instinct-PGMT-Shadowing-G1-Newton-v0 \
  --num_envs "${NUM_ENVS:-4096}" --max_iterations "${MAX_ITERATIONS:-30000}" \
  --viz none --run_name "$run_name" \
  --logroot "${LOGROOT:-/home/xiechunyang/Data/instinctlab_logs/g1_pgmt_lafan}" "$@"
