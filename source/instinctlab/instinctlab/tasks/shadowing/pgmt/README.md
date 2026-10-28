# PGMT flat LAFAN stage one

`Instinct-PGMT-Shadowing-G1-Newton-v0` trains a G1 CrossAttention actor and independent scalar critic
on flat ground. The implementation follows the `shadowing/perceptive` configuration layout and uses
its public motion-reference, reward, event, termination and monitor components.

## Layout and reuse

- `pgmt_env_cfg.py`: scene/commands/observations/rewards/events/curriculum/termination configuration,
  composed from `perceptive/perceptive_env_cfg.py`. The height scanner and depth camera are disabled.
- `config/g1/pgmt_shadowing_cfg.py`: G1 asset, ordered joint selectors, LAFAN paths and physics presets.
- `config/g1/agents/instinct_rl_ppo_cfg.py`: PPO, synchronized normalizers, actor and single-head critic.
- `models/cross_attention.py`: the validated original two-block CrossAttention actor.
- `mdp/motion.py`: named-NPZ input through the native `AmassMotion` / `MotionReferenceManager` pipeline.
  It preserves supplied link poses and velocities, maps names and converts WXYZ to XYZW on load.
- `mdp/commands.py`: the original continuous-episode clip-tail resampling and tracking metrics.
- `mdp/curriculums.py`: the original .2/.6/.2 uniform/middle/hard failure-bin mixture on the native
  adaptive curriculum. EMA updates use the existing event function.
- `mdp/observations.py`: exact 1645-dimensional policy / 296-dimensional critic layout and the extra
  pelvis-height reward. Nine of ten rewards use existing public functions.
- `runner.py`: checkpoint native motion weights, per-rank failure statistics and Torch/CUDA RNG with
  the policy, optimizer, learning rate and normalizers. Physical episodes restart on resume.

The robot is `instinctlab/assets/pgmt_g1.py`, with the original pelvis-root URDF, PD, armature, limits
and meshes reused from `assets/resources/unitree_g1`; the distinct head mesh remains under `assets/resources/pgmt_g1`. Joint ordering is explicit; robot-specific
selectors use `set_cfg_joint_order` as in the existing shadowing tasks.

Two common fixes were needed to use the shared pipeline:

1. The root package now imports `tasks` explicitly. Wildcard import previously replaced
   `instinctlab.utils` with `instinctlab.tasks.utils`, breaking lazy imports of motion commands.
2. Shared monitor maximum buffers update in place, preserving normal tensors across inference-mode
   rollouts and subsequent ordinary resets. Train/reload tests cover the failure that prompted this fix.

## Data and training contract

Use **40 original LAFAN clips only**. The three extra ground clips from the first migration are excluded.
Motion selection comes from the dataset manifest; source files are unchanged.

Default compute-host data: `/home/xiechunyang/Datasets/pgmt_lafan_g1/{manifest.json,named/}`.
Set `PGMT_LAFAN_DATA` before launch to relocate it. Each rank loads the complete selection (`mp_split_method=None`).

The actor uses 10 x 93 proprioceptive values and 11 x 65 reference values, flattened term by term.
It has two 256-wide, four-head CrossAttention blocks with 1024-wide feedforward layers. The reference
fusion uses RoPE. The action head is 349 -> 1024 -> 512 -> 256 -> 29. The scalar critic is
296 -> 4096 -> 2048 -> 1024 -> 512 -> 256 -> 1. No height-map/depth tokens, recovery, AMP or distillation.

Physics/control: 200/50 Hz, ten-second episodes. Reference windows reserve ten future frames.
Height and end-effector-height failures use .4 m; gravity-projection-z difference uses 1.0.
The native reference lifecycle and link-frame reward functions are retained, so the acceptance target
is similar learning curves rather than bit-identical physical rollouts with the legacy command implementation.

## Launch on gpu8-xiechunyang

```bash
source /home/xiechunyang/workspace/instinctlab_ws/instinctlab/agents/computation/activate-gpu8.sh
CUDA_VISIBLE_DEVICES=4,5,6,7 bash scripts/pgmt/train_gpu8.sh
```

Defaults: four GPUs, **4096 environments per GPU**, 30000 updates, checkpoint every 100 updates.
Override `NUM_GPUS`, `NUM_ENVS`, `MAX_ITERATIONS`, `RUN_NAME` and `LOGROOT` as needed.
Resume with the same rank/environment counts:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 bash scripts/pgmt/train_gpu8.sh \
  --resume --load_run /absolute/path/to/run --checkpoint model_100.pt
```

## Acceptance, 2026-10-27

Matched A/B: two GPUs per variant, 4096 environments per GPU, seed 42 + rank, identical 40-clip LAFAN
manifest and PPO settings, 300 updates. We compare matched update indices, common rollout reward and
tracking metrics. Native multi-reward episode logs have a different normalization, so raw episode-reward
tag names are not substituted for rollout reward.

| Metric | Legacy tail-50 mean | Native tail-50 mean | Relative difference |
|---|---:|---:|---:|
| Reward per control step | .037699 | .037178 | 1.38% |
| Episode length / steps | 324.945 | 326.509 | .48% |
| Aligned body error / m | .174135 | .167947 | 3.55% |
| Joint error / rad | .298478 | .297283 | .40% |

All four 25-update-smoothed traces and tail means pass the declared 25% engineering tolerance.
Their smoothed mean absolute relative differences are 2.79%, 2.91%, 2.35% and .82%, respectively.
This is a single-seed, 300-update refactor check, not a final convergence or held-out evaluation.

Single- and two-GPU integration: all 40 LAFAN clips checked against Newton FK, maximum position
error 2.21e-6 m; history reset passed; three updates + exact model/optimizer/normalizer/learning-rate
and failure-statistics reload + two updates passed; resumed parameters match across ranks.

Reports are mirrored in `agents/pgmt_native_validation_20261027/`. Remote A/B events and checkpoints:
`/home/xiechunyang/Data/instinctlab_logs/pgmt_lafan_curve_ab/`. Current launch and preserved old checkpoint
are recorded under `/home/xiechunyang/Data/instinctlab_logs/pgmt_native_launch_20261027/`.
TensorBoard at `http://127.0.0.1:6016` shows `lafan_comparison/*` and `pgmt_lafan/*` alongside the old run.

## Stage two: unpaired terrain adaptation

`Instinct-PGMT-Terrain-G1-Newton-v0` extends this implementation with terrain PPO.
`Instinct-PGMT-TerrainFlat-G1-Newton-v0` uses the same stage-two network/reward contract on a plane
for migration checks. The corresponding `-Play-v0` variants use 32 environments by default.

The actor keeps the stage-one history and reference attention blocks. Its input is 930 proprioception
values, 748 reference values (11 x 68, adding corrected planar/yaw-frame velocity), and a raw 21 x 21
height map (441 values, clipped to [-5,5] metres relative to the pelvis). Four learned 5 x 5 terrain
glimpses become four additional intent-attention tokens. Terrain values bypass empirical normalization.
A four-output critic consumes 296 privileged values and the actor's shared 256-dimensional intent;
shared parameters belong to the optimizer once and receive both policy and value gradients.

`agent.warm_start` explicitly transfers the native stage-one actor, action noise and old observation
normalizer channels. New velocity embedding columns start at zero; the new critic and optimizer start
fresh, using the source checkpoint's learning rate. A terrain-attention prior ramps from zero to one
across the first `agent.terrain_gate_ramp` updates (default 200). At gate zero the old action mapping
is preserved exactly for identical old inputs. This ramp is an engineering transition, not a claimed
paper hyperparameter.

`mdp/terrain_commands.py` composes the existing native LAFAN motion buffer with episode translation
and yaw placement, filtered terrain support height, velocity correction, and terrain-family x level x
motion failure sampling (20% uniform). The unchanged 40-clip manifest is retained. Five terrain
families occupy ten fixed columns: flat, up/down slopes, up/down stairs, boxes and roughness. Ten rows
represent L0--L9. Only resets change assigned difficulty; leaving the centre platform is required for
non-flat promotion. Lower-body position/orientation/joint tracking tolerances increase with difficulty.
Recovery is disabled.

Rewards are native `MultiRewardCfg` groups: upper (6 terms), lower (6), auxiliary (9), terrain (6).
The kernels, weights, contact hysteresis and collision-plane label definitions follow the local
`whole_body_tracking` terrain implementation. These include local engineering choices; they do not
establish reproduction of the complete PGMT paper. Contact labels are loaded by motion filename. Unknown contacts retain invalid masks.

`terrain_env.py` updates terrain reference/contact state before termination and rewards, records
pre-reset terminal observations without double-appending history, and exports tracking metrics for
the transformed reference. Time limits, motion end and tile exit truncate; reference-relative height,
tilt and body mismatch fail. `terrain_ppo.py` bootstraps only non-failing truncations from their actual
terminal values, computes four GAE returns, then sums group advantages before global normalization.
The four value losses are averaged. Normalizer synchronization and persistent contact buffers support
in-process reload after inference-mode rollouts.

`terrain_runner.py` atomically saves the policy, optimizer, normalizers, per-rank failure/visit counts,
terrain assignments, learning rate, Torch/CUDA RNG, source checkpoint.
Ordinary resume requires matching ranks and environment counts, and starts fresh physical
episodes. Stage-one warm-start and stage-two resume are separate operations.

Generate contact labels once on the compute host:

```bash
python scripts/pgmt/prepare_contacts.py \
  --data_dir /home/xiechunyang/Datasets/pgmt_lafan_g1/named \
  --manifest /home/xiechunyang/Datasets/pgmt_lafan_g1/manifest.json \
  --output /home/xiechunyang/Datasets/pgmt_lafan_g1/contacts \
  --urdf source/instinctlab/instinctlab/assets/resources/pgmt_g1/urdf/g1/main.urdf
```

Launch with a native stage-one checkpoint (four GPUs, 4096 environments per GPU):

```bash
source agents/computation/activate-gpu8.sh
CUDA_VISIBLE_DEVICES=4,5,6,7 bash scripts/pgmt/train_terrain_gpu8.sh \
  agent.warm_start=/absolute/path/to/stage_one/model_6100.pt
```

Use `--resume --load_run /absolute/path/to/stage_two/run --checkpoint model_N.pt` for stage-two resume,
without `agent.warm_start`. `scripts/pgmt/validate_terrain.py` checks channel migration, critic gradients,
grouped returns, terminal bootstrap, exact checkpoint reload and multi-rank parameter/normalizer sync.

## Scheme one: differentiable terrain imagination

`Instinct-PGMT-WorldModel-G1-Newton-v0` keeps the terrain actor, four-value critic and all 27 real
reward terms. It adds SuperMimic-style frame-local delta dynamics and short imagined actor updates
after each native PPO update. This is a hybrid implementation, not a reproduction of SuperMimic's
complete optimizer or a compatible loader for its flat-ground world-model weights.

The complete path is action → predicted root pose/joints → bilinearly sampled local height map →
next policy observation → next action. `heightfield.py` bakes the known static mesh once and shares
one world raster across the batch. Gradients flow through sampling coordinates (XY, Z and yaw),
not through the mesh or Newton. Keep `global_world_only=True` on the Newton height scanner: otherwise
the current backend also hits robot geometry. The raster uses 5 cm cells; the policy retains its
existing 21 x 21 scan at 10 cm spacing and exact native flattening order. Bilinear smoothing around
sharp edges is an approximation, and finite-map validity masks stop invalid imagined objectives.

The 71-D state uses pelvis XYZ, XYZW orientation, world linear/angular velocity, and 29 PGMT-ordered
joint positions/velocities. The WM predicts a normalized 70-D local delta and 14 contact probabilities.
Previous action, contact belief, proprioceptive history and the support-height filter are carried
through imagination. Reference windows remain exogenous; orientation and tracking-error corrections
are rebuilt from the predicted state. Contact probabilities help model dynamics. Surrogate version 2
also penalizes the squared horizontal finite-difference velocity of predicted stance feet. Contact
weights are detached in this penalty so the actor cannot reduce it by merely predicting lost contact.
This is a differentiable stance-motion proxy, not an exact contact-force or terrain-tangent reward.
All original contact rewards still use real PPO. The imagined reward combines tracking, limits,
action rate and stance motion; it is not the exact sum of real contact rewards.

Default training uses three-step supervised rollouts, four-step actor rollouts, 50 warm-up iterations,
16 WM minibatches per PPO iteration and a frozen initial actor anchor. Every eighth environment is
held out of WM fitting and normalizer estimation. A held-out four-step normalized loss below 1 gates
imagined policy updates. This gate is a practical error screen, not a guarantee against model bias.
`world-model.jsonl` records losses, gradient norms and the actual count of imagination updates.

For adaptation from a weaker flat checkpoint, `agent.world_model.teacher_ema` can replace the
permanently frozen anchor with a teacher that slowly follows the actor after each real PPO update.
Its terrain gate and other buffers follow the current actor exactly. Zero retains the frozen-teacher
behavior. The teacher loss excludes post-terminal reference windows. This teacher is derived from
the same training run; it does not load the stage-two evaluation policy.

`agent.world_model.max_imagination_kl` optionally bounds each auxiliary step's mean Gaussian KL on
fixed, reconstructed observations at real replay states. Since imagination does not update action
noise, the KL formula is exact on that batch. Oversized steps are backtracked identically across
ranks; rejected steps restore both actor and optimizer. A nonfinite proposal raises after restoration.
This is a local step-size constraint, not a guarantee of real-environment improvement. Zero disables
the constraint for legacy experiments. Logs include the accepted KL and parameter-step scale.

The current stage-one experiment uses 200 warm-up iterations, EMA coefficient 0.05, anchor weight 10
and maximum imagination KL 0.002. These are experimental settings, not a claim of stage-one
convergence to the existing terrain checkpoint. Warm-up only fits the WM and runs real PPO.

Start from a native stage-one checkpoint:

```bash
export PGMT_LAFAN_DATA=/absolute/path/to/lafan1_umr_heft_pgmt40
python scripts/instinct_rl/train.py --task Instinct-PGMT-WorldModel-G1-Newton-v0 \
  --device cuda:0 --viz none --num_envs 512 --max_iterations 400 --run_name heightmap-from-stage1 \
  --logroot /absolute/path/to/InstinctLab-local/scheme1 \
  agent.warm_start=/absolute/path/to/model_2700.pt
```

Alternatively, continue a terrain policy with `agent.stage2_initialization=/absolute/path/to/model_9600.pt`
instead of `agent.warm_start`. This preserves actor/critic/normalizers, source learning rate and terrain
gate, while starting new optimizers and WM. It is different from proving equivalent convergence from
stage one. The actor used for inference is unchanged and does not need the WM, global raster or
privileged contact state; it still needs the same local height map and reference/proprioceptive inputs.

Scheme-one checkpoints additionally contain WM, its normalization statistics/optimizer, teacher actor,
imagination optimizer, update counts and configuration. Resume uses the usual `--resume --load_run ...
--checkpoint model_N.pt`, matching environment count, WM settings and gate ramp; omit initialization
overrides on resume. Physical episodes and replay restart together. Multi-GPU runs share WM parameters,
normalization moments, supervised gradients and imagined actor gradients. The model-error gate is
pooled across ranks so all ranks take the same update branch. Each rank retains its own terrain and
replay; checkpoints preserve the native per-rank curriculum/RNG state. Only rank zero publishes the
complete checkpoint, after adding WM/optimizer state, through an atomic rename.
Checkpoint surrogate versions are checked on resume. Use `agent.allow_surrogate_change=true` only
when deliberately starting a new experiment with a different surrogate reward; this opt-in is
recorded in the saved run configuration.
The same opt-in permits changing the teacher-anchor weight, EMA coefficient and imagination KL
limit for an explicit experiment; it does not permit changing model shape, raster resolution or
rollout settings. Activating EMA on a previously frozen-teacher checkpoint initializes the teacher
from that checkpoint's actor. An ordinary EMA-to-EMA resume restores the saved teacher exactly.
Old checkpoints without the EMA/KL settings retain their zero defaults.

The benchmark fixes motion, start frame, yaw, terrain family and level for every case, records only
the first episode and captures failure metrics before auto-reset. Failure and tile exit are reported
separately. Run multiple checkpoints in one invocation to use identical environment randomization:

```bash
python scripts/instinct_rl/benchmark_pgmt.py --checkpoint /absolute/path/to/model_9600.pt \
  /absolute/path/to/scheme1/model_400.pt --output /absolute/path/to/InstinctLab-local/comparison \
  --num-envs 120 --steps 500 --levels 0 3 6 --seed 17
```

Repeat with another seed for new start frames/yaws and terrain realization. These remain the same
40 motion clips, not unseen-motion generalization. `--terrain-gate 0` disables the terrain attention
tokens for a perception ablation; use a separate output directory. Inspect tracking error, failure,
tile exit and slip together, rather than selecting by reward alone. Result JSON includes checkpoint
SHA-256 and per-case measurements. Run geometry, finite-difference, history and dynamics contracts with
`python -m unittest discover -s source/instinctlab/test -p test_pgmt_world_model.py`.

`scripts/pgmt/validate_world_model.py` provides a bounded real-simulator test: four updates, exact
module/optimizer reload, two further updates, then exact cross-rank equality. Use the training task,
an explicit stage-one `agent.warm_start`, `--report-dir`, and (for a short smoke only)
`agent.world_model.warmup=1 agent.world_model.updates=2 agent.world_model.validation_threshold=100.0`.
The permissive smoke gate exercises gradients; it must not be used as evidence of model accuracy or
copied into a performance training recipe. Launch multi-GPU training/validation through
`python -m torch.distributed.run --standalone --nproc_per_node=2 ...`; `--num_envs` is per rank.

### Validated local candidate, 2026-10-29

The selected local checkpoint uses surrogate v2 and `agent.world_model.anchor=10.0`. It was initialized
from the existing PGMT terrain PPO `model_9600`, then selected after 4,032 WM optimizer steps and 202
imagination updates. This establishes a working second-stage refinement of that terrain policy;
the separate short warm-start run from flat `model_2700` did not reach the terrain baseline.

Native Newton evaluation used the local UMR HEFT retargeted 40 clips, five terrain families, L0/L3/L6,
ten-second episodes and seeds 17/23: 1,200 matched cases per policy. Seeds were used during development,
not reserved as a final untouched test set. Aggregate results:

| Metric | Existing terrain PPO | Selected scheme one |
| --- | ---: | ---: |
| Horizon completed | 77.17% | 77.75% |
| Failure | 5.92% | 5.17% |
| Tile exit | 17.00% | 17.08% |
| Joint RMSE / rad | 0.15261 | 0.15323 |
| Aligned body error / m | 0.06183 | 0.06367 |
| Contact-weighted slip indicator | 0.04624 | 0.04780 |

Both seeds individually passed the declared engineering limits: at most 3 percentage points less
completion, 2 points more failure, 5% more joint/body error, and 10% more slip. This supports comparable
performance on this protocol, not uniform improvement, a statistical equivalence proof, improved sample
efficiency, or robot deployment. The selected actor also completed one ten-second standalone MuJoCo
stairs-down replay without reset; that demonstration has different physics and is not pooled into
the native benchmark.

Artifacts live outside Git under `../InstinctLab-local/checkpoints/scheme1_heightmap/`: full
`model_best.pt`, saved run parameters, SHA/provenance, replay video, and `resume.sh` / `benchmark.sh`.
The complete per-case results and failed experiments remain under `../InstinctLab-local/scheme1/`.
To resume this selected checkpoint, retain 512 environments and `agent.world_model.anchor=10.0`.
