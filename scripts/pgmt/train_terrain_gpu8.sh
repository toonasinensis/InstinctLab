#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
exec torchrun --standalone --nproc_per_node="${NUM_GPUS:-4}" scripts/instinct_rl/train.py \
  --task Instinct-PGMT-Terrain-G1-Newton-v0 --num_envs "${NUM_ENVS:-4096}" \
  --max_iterations "${MAX_ITERATIONS:-30000}" --viz none \
  --run_name "${RUN_NAME:-PGMT_LAFAN_Terrain}" \
  --logroot "${LOGROOT:-/home/xiechunyang/Data/instinctlab_logs/g1_pgmt_lafan_terrain}" \
  agent.save_interval=100 "$@"
