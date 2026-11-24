# SuperMimic runtime with PGMT task extensions

This directory copies the desktop SuperMimic runtime and keeps its original module layout.
PGMT subclasses are colocated with their bases and use a `pgmt_` prefix. There is one
`learners/` directory. The previous Collector/Components/core framework is removed.

## Source provenance

Copy source: `/home/thl/Desktop/实验/supermimic/src/supermimic`, frozen on 2026-11-12 at
`/home/thl/wt_wbc/work/InstinctLab-reference/supermimic-desktop-source-20261112`.
The desktop Git base is `5efbf114e11f725c31d83e217be987c13c0e4ce2`, but the copied working
tree includes local learner/replay edits and its local FK utility. The frozen source includes
Git status and a patch of tracked local changes. The commit alone does not identify this copy.
Apache-2.0 license and attribution are in `LICENSE-SuperMimic` and `NOTICE`.

The runtime tree, task definitions and original CLI modules are copied. Local untracked
experiment scripts (`diagnostics.py`, optimizer comparison/evaluation scripts and
`tasks/public_tracking.py`) are excluded. Desktop package metadata is not installed over
InstinctLab: PGMT uses the existing IsaacLab/Newton environment. The original mjlab CLI
modules retain their optional dependencies and are not PGMT entry points.

## Base changes versus PGMT extensions

| Copied base | Generic changes in this copy | PGMT extension |
|---|---|---|
| `runners/supermimic_runner.py` | Component factories; post-step terminal fields; shared collect/iteration; phase/log/save hooks extracted from run; optional W&B imports deferred | `runners/pgmt_runner.py` |
| `envs/world_model_env.py` | Reset/step lifecycle hooks for native history and deferred reward extras | `envs/pgmt_world_model_env.py` |
| `learners/wm_learner.py` | Loss/batch/optimizer/capture/summary hooks | `learners/pgmt_wm_learner.py` |
| `learners/policy_learner.py` | Optimizer/capture/summary hooks; reusable critic regression template | `learners/pgmt_policy_learner.py` |
| `learners/learner_utils.py` | Optimizer-group acceptance and capture state support | `learners/pgmt_learner_utils.py` |
| `replay_buffer.py` | FieldSpec supports shaped fields in addition to scalar widths | `pgmt_replay_buffer.py` |
| `models.py`, `representations.py` | Original bases retained; package-local imports/formatting | `pgmt_models.py`, `pgmt_representation.py` |
| `sim_context.py` | Package-local imports/formatting | `pgmt_sim_context.py` |
| `utils/logger.py` | Overridable write sink; lazy optional W&B import | `utils/pgmt_logger.py` |

Across copied files, imports are package-prefixed and formatting follows this repository.
`utils/math.py` explicitly bridges the original WXYZ math API to IsaacLab's XYZW API.
Original tasks no longer auto-register on import. Original artifact downloading delegates
to the provider without a custom digest/cache system. These are real source adaptations,
not a claim of an unmodified vendor snapshot.

PGMT model constructors preserve the current network and initialization instead of allocating
unused upstream models. `PGMTDynamics` and `PGMTGenCoordRepresentation` implement the upstream
representation contract and retain the PGMT contact head, state integration and terrain conditioning.

## Entry and responsibility

Use `scripts/instinct_rl/train.py` with `Instinct-PGMT-SuperMimic-G1-Newton-v0`.
The task recipe is `../config/g1/pgmt_supermimic_cfg.py`; typed algorithm options live in
`config.py`, state slices in `layouts.py`. The default is the existing CNN187 AllCombined recipe.

`BaseRunner → SuperMimicRunner → PGMTSuperMimicRunner` owns assembly and training scheduling.
`WorldModelEnv → MotionTrackingEnv → PGMTWorldModelEnv` owns predicted state, references,
terrain/history and rewards, using inherited reset/step(actions). PolicyLearner owns actor
imagination, objective and critic returns/fitting; WorldModelLearner owns supervised loss.
Batched reward evaluation is explicitly deferred until the end of the imagined trajectory.

See the repository-root `REVIEW.md`, `SM_ARCHITECTURE.md` and `SM_VALIDATION.md` for review order,
contracts and measured validation. Only the Newton PGMT path is validated by this migration;
original mjlab scripts and convergence equivalence require their own checks.

## G1 head-depth training

The depth extension inherits the same collection and training loop. Its task recipes are:

- `Instinct-PGMT-SuperMimicHeadDepth-G1-Newton-v0`: D1S2, G1 head-mounted camera.
- `Instinct-PGMT-SuperMimicHeadDepthLafanBox-G1-Newton-v0`: full LAFAN and Boxes20, same camera.
- `Instinct-PGMT-SuperMimicHeightLafanBox-G1-Newton-v0`: paired full-LAFAN/Boxes20 heightscan recipe.

`pgmt_head_camera_cfg.py` defines the depth observation, supervision scanner and SM recipe, and
constructs the camera directly at the official G1 head-relative mount on the local torso/head asset. Optical pose and sensor parameters
are saved in the checkpoint contract. The reconstruction target remains on the pelvis/yaw grid.

`PGMTDepthRunner` specializes assembly, replay fields and perception diagnostics. The actor takes
one 48x64 depth frame through a 64-dimensional latent; its existing hidden layers and the critic
remain unchanged. The encoder is trained by masked metric-height reconstruction and world-model
latent prediction. Actor loss detaches the latent at the action-network boundary. Physical inputs
use frozen/online empirical normalization as appropriate; image preprocessing belongs to the sensor.
There is no visual EMA target; policy EMA is a separate checkpoint/deployment choice.

Copied visual primitives and their upstream attribution are documented in
[`perception/PROVENANCE.md`](perception/PROVENANCE.md). Masking, gradient ownership and terminal
image handling are checked by `source/instinctlab/test/pgmt/test_depth_perception.py`; the sibling
`validate_depth.py` checks the Newton training integration and checkpoint reload in an external
validation directory.

### Saved-config depth evaluation

The evaluator uses the registered **training** entry and restores `params/env.yaml`.
It never selects a play-specific environment class. No dataset, terrain mixture, difficulty,
phase or episode-limit override is implicit: omitted settings inherit the saved training value.
Runtime curriculum/sampler history is not reconstructed from the checkpoint.

```text
config/g1/pgmt_terrain_recipe_cfg.py   shared terrain/spawn/curriculum/query-grid recipe
  ├── pgmt_scan187_cfg.py             heightscan policy task
  └── pgmt_head_camera_cfg.py         head-depth task and paired LAFAN/Boxes20 recipe

supermimic/evaluation/
  config.py     training-config restoration and typed, explicit overrides
  policy.py     frozen depth actor/normalizer loading using training model classes
  runner.py     native environment stepping and first-episode lifecycle
  observer.py   collector interface, composition and optional state/RNG purity audit
  metrics.py    raw aligned/world tracking errors, native termination/return accumulation
  contact.py    optional read-only ground-contact measurements
  report.py     per-episode, per-terrain and aggregate output
  cli.py        argument parsing and simulation launch
  presets/      explicit common-cohort evaluation settings
```

Depth and heightscan inherit the shared terrain recipe independently. The shared query grid
also serves the world model and reconstruction supervision; preserving that grid and the legacy
observation layout does not require inheriting the heightscan task. The optical camera is created once at the G1 head mount; its reconstruction labels remain
on the original pelvis/yaw grid. The legacy pelvis-depth task and configuration are removed.

The actor class and constructor arguments come from the checkpoint contract; inference calls
training `PGMTDepthPolicy.collect()`. Collectors attach to the existing
`PGMTTerrainEnv.add_transition_observer()` before reset. They do not change rewards, terminations,
lower-body tolerance, action handling, observations/noise, camera configuration or physics.
The optional contact sensor supplements the original sensor and samples the native control
boundary without changing backend decimation. Only each environment's first episode is scored,
including its terminal step. Primary body errors use the training XY/yaw-aligned target without
subtracting tolerance. World-frame body/root errors are separate; failure flags are native.

Configuration order is task defaults → saved training YAML → settings YAML → explicit CLI flags.
`None` means inherit. Outputs include `training_env.yaml`, `effective_env.yaml`, `config_diff.json`
and `overrides.json`; `env.yaml` retains the effective-config filename used by existing tooling.
The loader currently supports depth policies, not arbitrary PPO/TF checkpoints.

Inherit the training environment (set paths to a checkpoint and an external validation output):

```bash
python scripts/pgmt/evaluate.py \
  --task Instinct-PGMT-SuperMimicHeadDepth-G1-Newton-v0 \
  --checkpoint "$CHECKPOINT" --output "$VALIDATION_OUTPUT" --weights ema --contacts
```

Reproduce the previous A/B cohort explicitly (D1S2, six equal shared terrains, L9, 4000 environments,
9999 seconds, post-flat-warmup phase; geometry stays training-native):

```bash
python scripts/pgmt/evaluate.py \
  --task Instinct-PGMT-SuperMimicHeadDepth-G1-Newton-v0 \
  --checkpoint "$CHECKPOINT" --output "$VALIDATION_OUTPUT" --weights ema --contacts \
  --settings source/instinctlab/instinctlab/tasks/shadowing/pgmt/supermimic/evaluation/presets/d1s2_l9.yaml
```

`python -m instinctlab.tasks.shadowing.pgmt.supermimic.evaluation` remains supported.
`--smoke --num-envs 32 --steps 20` enables within-step state/RNG purity checks. Independent
GPU physics runs are not assumed bitwise deterministic. Retain the same settings preset when
comparing results across evaluator versions; the new default no longer selects that cohort.

## T3_27 heightscan

The Newton task is `Instinct-PGMT-SuperMimic-T3_27-Newton-v0`. It uses the 27-joint
T3 URDF, a 17 × 11 height grid, the existing flat warmup and the terrain curriculum
with flat, ramps, stairs and rough terrain. The URDF and its meshes ship in
`$PROJECT_DIR/source/instinctlab/instinctlab/assets/resources/t3_27/`.
From the center platform to the rim, `stairs_up` rises and `stairs_down` descends.
Saved configurations and checkpoints with the original stair names are translated
on load. TensorBoard logs from those runs retain the labels used when recorded.

TensorBoard's `Terrain/mean_level` and `Terrain/<terrain>_level` curves report means
across the matching environments. `Terrain/<terrain>_max_level` reports their highest
current level at each training update, including separate curves for the two ramp
and stair directions. A terrain with no matching environment has no sample for
that update. Level 9 on this curve means at least one environment is on level 9.

Use the default T3_27 training script for the configuration from DLC job 34984.

```bash
bash "$PROJECT_DIR/train_release_scripts/release_model_T3_27.sh"
```

The script starts fresh on one GPU with 4096 environments, 30000 iterations,
and seed 42. It uses stairs up to 25 cm in both directions, equal upper- and
lower-body tracking weights, and lower-body tracking sigmas widened by 50%.
Stair progress reward gating is enabled, and normal completion of the reference
motion can advance the curriculum without a minimum episode duration. Online
foothold adaptation is disabled.

The script uses `moving_upright_middlespeed.h5` from the sibling `supermimic`
checkout and writes logs to the sibling `InstinctLab-local` directory by
default. Set `PGMT_T3_27_MOTION_H5` to use a different grouped 50 Hz T3 dataset.
`INSTINCTLAB_PYTHON`, `MAX_ITERATIONS`, `SEED`, and `LOGROOT` are optional overrides.

When launching `scripts/instinct_rl/train.py` directly with the task ID, the
motion path can also be set with
`env.scene.motion_reference.motion_buffers.SOMA.path`. The optional
`env.scene.motion_reference.motion_buffers.SOMA.motion_names` selects exact HDF5
group names for a smaller run. The optional `cache_dir` in the same motion buffer
sets the cache location. By default the cache is written to the sibling
`InstinctLab-local/motion_cache/` directory.

The first load validates the named joint columns and computes velocities and
world-frame body targets for each clip. Position and angle differences use a
centered interval within the clip and one-sided intervals at its ends. The cache
stores the named NPZ motion fields, clip boundaries and
`contact`, `valid`, and `bottom`. On disk, quaternions are WXYZ. Loading builds
the same XYZW `MotionSequence` used by the G1 named-NPZ path. Contact labels and
reset clearance use the T3 URDF collision meshes. The differentiable foot reward
uses 32 representative vertices per foot and retains the axis extrema of each
mesh. Disk lookup is keyed by source metadata and the preprocessing contract.
The checkpoint data key also identifies byte-identical source files on different
hosts. Allow about 3.3 GiB for the full motion tensors at 14 tracked bodies, plus
the contact tensors and the memory used by simulation and training.

Set `env.commands.motion.foothold_adaptation=true` to enable online T3 stair
reference adaptation. The existing contact labels determine swing and stance
windows. The planner searches near the source landing point and checks the
center and corners of the URDF foot footprint against the shared height field.
Swing lift anticipates a rising landing surface, and lowering starts later in
the swing when the landing surface is below the takeoff surface. Bounded
Jacobian corrections feed the existing ankle position and joint reference
targets. The real environment and imagined rollout build those targets from
the same motion frames, terrain field and episode placement. The source H5,
reward weights, and course progression rules are kept as configured.

To compare both settings from one trained run, set
`agent.initialize_checkpoint` to its SuperMimic checkpoint. The initializer
loads the actor, EMA actor, critic, world model and normalizers. The optimizers,
replay and curriculum start from the new run's settings. Evaluate on paired
motion phases across fixed levels and report ascent and descent separately.

Set `agent.progress_reward.enabled=true` to gate positive rewards on stairs
while the reference is moving. The default 0.32 s window uses the net XY
displacement of the reference root to define the walking direction. The gate
is active at reference speeds of at least 0.20 m/s. It scales positive rewards
from zero at 0.05 m/s of projected robot progress to one at half the reference
speed. Penalties keep their original values. A signed progress term, weighted
at 0.5 and capped at the reference speed, preserves a gradient at zero progress.
The native reward and imagined return use the same function. Replay carries
the causal position history at each rollout start, with history padded using
the current spawn pose after a reset. The base task configuration leaves the
gate disabled. The default T3_27 training script enables it explicitly.
