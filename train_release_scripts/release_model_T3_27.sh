#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

export PGMT_T3_27_MOTION_H5="${PGMT_T3_27_MOTION_H5:-${project_dir%/*}/supermimic/data/motions/T3_27/soma_0.6/moving_upright_middlespeed.h5}"
if [[ ! -f "$PGMT_T3_27_MOTION_H5" || ! -r "$PGMT_T3_27_MOTION_H5" ]]; then
    printf 'Cannot read T3_27 motion data: %s\n' "$PGMT_T3_27_MOTION_H5" >&2
    exit 1
fi

activate_training="${project_dir%/*}/InstinctLab-local/envs/pgmt_training/activate-training.sh"
if [[ -z "${INSTINCTLAB_PYTHON:-}" && -r "$activate_training" ]]; then
    source "$activate_training"
fi

visible_devices="${CUDA_VISIBLE_DEVICES-0}"
export CUDA_VISIBLE_DEVICES="${visible_devices%%,*}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export PYTHONPATH="$project_dir/source/instinctlab${PYTHONPATH:+:$PYTHONPATH}"
unset LOCAL_RANK

run_name="$(basename -- "${BASH_SOURCE[0]}" .sh)"
log_dir_name="$(TZ=CST-8 date +%Y%m%d_%H%M%S)_${run_name}_$$"
log_root="${LOGROOT:-${project_dir%/*}/InstinctLab-local/logs/t3_27_pgmt_supermimic}"
mkdir -p -- "$log_root/$log_dir_name"

"${INSTINCTLAB_PYTHON:-python}" -u scripts/instinct_rl/train.py \
    --task Instinct-PGMT-SuperMimic-T3_27-Newton-v0 \
    --num_envs 4096 --device cuda:0 --viz none \
    --max_iterations "${MAX_ITERATIONS:-30000}" --seed "${SEED:-42}" \
    --run_name "$run_name" --logroot "$log_root" --log_dir_name "$log_dir_name" \
    agent.progress_reward.enabled=true \
    env.commands.motion.foothold_adaptation=false \
    env.commands.motion.curriculum_motion_end_min_s=null \
    'env.scene.terrain.terrain_generator.sub_terrains.stairs_up.step_height_range=[0.02,0.25]' \
    'env.scene.terrain.terrain_generator.sub_terrains.stairs_down.step_height_range=[0.02,0.25]' \
    env.rewards.lower.position.weight=1.0 \
    env.rewards.lower.position.params.std=0.45 \
    env.rewards.lower.orientation.weight=1.0 \
    env.rewards.lower.orientation.params.std=0.6 \
    env.rewards.lower.linear_velocity.weight=0.5 \
    env.rewards.lower.linear_velocity.params.std=1.5 \
    env.rewards.lower.angular_velocity.weight=0.5 \
    env.rewards.lower.angular_velocity.params.std=4.71 \
    env.rewards.lower.joint_position.weight=1.0 \
    env.rewards.lower.joint_position.params.std=0.75 \
    env.rewards.lower.joint_velocity.weight=0.5 \
    env.rewards.lower.joint_velocity.params.std=7.5 \
    2>&1 | tee "$log_root/$log_dir_name/console.log"
