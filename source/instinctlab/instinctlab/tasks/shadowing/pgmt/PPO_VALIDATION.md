# PPO terrain and motion validation

This branch starts at native PGMT `ec5237e`. The PPO presets described here
do not train a world model or use imagined policy updates or WM checkpoints.
The separate supervised WM tasks are described in `WM_DIAGNOSTICS.md`.
The baseline PPO uses four reward groups and the native terrain height scanner.

## Presets and reversible configuration

The original `Instinct-PGMT-Terrain-G1-Newton-v0` task retains all 40 manifest
motions and the original terrain proportions. The new
`Instinct-PGMT-TerrainPPOValidation-G1-Newton-v0` task selects only
`dance1_subject2` and changes proportions, leaving geometry and L0--L9 intact:

| Terrain | New share | Original share | Original difficulty parameters retained |
| --- | --- | --- | --- |
| Flat | 20% | 20% | Plane |
| Uphill slope | 10% | 10% | Rise/run 0.03--0.30 |
| Downhill slope | 10% | 10% | Rise/run 0.03--0.30, inverted |
| Upstairs | 20% | 10% | Step height 0.02--0.20 m; step width 0.30 m |
| Downstairs | 20% | 10% | Step height 0.02--0.20 m; step width 0.30 m |
| Boxes | 0% | 20% | Retained configuration: width 0.45 m, height 0.02--0.25 m |
| Rough | 20% | 20% | Amplitude +/-0.005--0.050 m, noise increment 0.005 m |

There are 10 rows and 10 columns of 8 x 8 m tiles, a 10 m outer border,
0.10 m horizontal and 0.005 m vertical terrain resolution. The column order is
`flat, flat, slope_up, slope_down, stairs_up, stairs_up, stairs_down,
stairs_down, rough, rough`. Slopes and stairs retain their original
2 m center platform; rough terrain has no separately
flattened center platform. Slope/rough border width is 0.25 m and stairs border
width is 1 m. The height scanner is 21 x 21 points over 2 x 2 m, at 0.10 m spacing,
yaw-aligned and updated at 50 Hz. Newton scans exclude robot geometry; this is
the isolated fix previously recorded as `9652e32`, not a WM migration.

Level `l` uses the original `l/9` difficulty, so upstairs/downstairs rise is
`0.02 + 0.18*l/9` m, slope is `0.03 + 0.27*l/9`, and rough amplitude is
`0.005 + 0.045*l/9` m. Removing boxes does not lower these ranges.

Overrides are applied after preset construction and consumed at runtime:

```bash
# Select multiple motions, preserving their manifest order.
'env.scene.motion_reference.motion_buffers.LAFAN.motion_names=[dance1_subject2,walk2_subject1]'
# Restore the full manifest, without changing the diagnostic terrain preset.
env.scene.motion_reference.motion_buffers.LAFAN.motion_names=null
# Restore the original terrain mixture on the diagnostic task.
env.scene.terrain.terrain_generator.sub_terrains.stairs_up.proportion=0.1
env.scene.terrain.terrain_generator.sub_terrains.stairs_down.proportion=0.1
env.scene.terrain.terrain_generator.sub_terrains.boxes.proportion=0.2
```

Changing the task back to `Instinct-PGMT-Terrain-G1-Newton-v0` restores both
original selections. Explicit lists accept canonical names or `.named.npz`
filenames. The loader filters the manifest before loading tensors/contact
labels, so the diagnostic run actually loads one motion. Empty, duplicate and
unknown selections are rejected. The motion axis of failure statistics uses
the actual number of loaded files, not a hard-coded 40.

Terrain proportions are normalized by Isaac Lab and realized as fixed columns,
not fresh per-reset random draws. With 10 columns the current shares are exact.
For custom mixtures, choose enough columns to represent every positive terrain
weight. Runtime family IDs derive from the exact pinned generator column rule;
removing boxes does not turn rough terrain into the boxes family. Family IDs
remain flat=0, slope=1, stairs=2, boxes=3, rough=4, preserving reward tolerances.
The mapping requires the original ten-row curriculum-generated grid.

## Terrain curriculum

The default training configuration starts every environment at L0
(`env.scene.terrain.max_init_terrain_level=0`). L1--L9 remain generated and are
reached through the original curriculum; this is not a difficulty-range cut.
An environment retains its terrain column/type and changes only its level.
At reset:

- Actual failure termination decreases the level by one.
- Reaching the 10 s time limit without failure or leaving the assigned tile
  increases the level by one. On nonflat terrain it additionally requires
  **more than** 50 accumulated control steps outside the center region.
- The center-region condition is max absolute XY offset from the tile origin
  greater than 1.1 m. Those steps are accumulated, not necessarily consecutive.
- Timeouts without sufficient platform departure keep the level unchanged.
  Motion-end and tile-exit timeouts do not themselves promote the level.
- L8 success advances to L9. A successful episode at L9 attempts to advance
  beyond the ten-row grid and uniformly resamples a level from L0--L9,
  including L9. Merely entering L9 or resetting there without success does
  not trigger randomization. Failure at L9 still moves to L8; failure at L0
  stays at L0. No reset accounting occurs at first placement.

The reset delegates level and origin updates to the installed Isaac Lab
`TerrainImporter.update_env_origins`, matching the last-level behavior in
[Isaac Lab](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/terrains/terrain_importer.py)
and [legged_gym](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot.py).
Its `max_terrain_level` is the exclusive row count (10), not the highest valid
index (9). Each environment retains its column/type after randomization.
Previous-episode failures/visits are recorded at the old level before the
transition; the next motion is sampled using the newly assigned level.
Consequently, mean training terrain level can decrease after mastering L9:
the population revisits easier rows instead of accumulating permanently at
the highest row. Mean level is not a monotonic measure of tracking ability.

Checkpoints record `terrain_curriculum_version=max_level_resample_v1` in their
sampling contract. The existing resume check rejects stage-two checkpoints
with the old clamped curriculum contract; start this experiment from the
original stage-one warm start. Existing jobs keep their immutable old source.
The native updater draws random candidates for every settling environment,
so seeded runs are reproducible within this version but do not preserve the
old version's frame/yaw random sequence.

The focused regression entry point is `scripts/pgmt/validate_curriculum.py`.
It uses real environment resets to exercise the L8/L9 boundaries, failure,
neutral/insufficient-platform/tile-exit resets, first placement, frozen
curriculum, origin/type consistency, and old-level sampler accounting.

Configuration controls:

```bash
env.commands.motion.enable_terrain_curriculum=true
env.commands.motion.curriculum_off_platform_steps=50
env.scene.terrain.max_init_terrain_level=0
```

The generator's `curriculum=true` creates ordered difficulty rows and column
types. `commands.motion.enable_terrain_curriculum` separately controls whether
episodes promote/demote the assigned level. Disabling the latter leaves the
grid intact. Fixed-level/balanced L0--L9 evaluation is separate from training;
a low mean training level does not establish high-difficulty performance.

## Motion sampling

Motion and terrain are unpaired. For each reset, motion probabilities are
conditioned on the assigned terrain family `f` and level `l`:

`p(m|f,l) = (1-u)*(1+F[f,l,m])/sum_j(1+F[f,l,j]) + u/M`.

`M` is the selected manifest size. `u=0.2` preserves a 20% uniform mixture;
with one selected motion its probability is exactly one. The original score
update is retained: `F <- d*F + (1-d)*failed_counts`, with `d=0.99`, once per
reset callback containing previous episodes. It is a decayed failure **count**,
not a failure rate divided by visits. Scores and visits are per rank and saved
for each rank; no new global sampling synchronization is introduced.

```bash
env.commands.motion.motion_uniform_fraction=0.2
env.commands.motion.motion_failure_decay=0.99
```

Within the selected motion, start frames are uniformly sampled using the
original rule `floor(U*(N-11))`, hence 0 through N-12. Eleven reference frames
are reserved. Playback is 50 Hz, with no time warping or retargeting changes.
Initial yaw is uniform in [-pi,pi], XY is aligned to the tile origin, and the
existing reference support-height filter (0.10 s time constant) is retained.
Frame sampling is not failure-adaptive in this terrain task. The separate
flat stage-one bin curriculum is disabled by the existing terrain preset.

## Stage-one initialization and launch

Use the original native `model_2700.pt`, not candidate600 or a stage-two policy.
`agent.warm_start` copies actor weights, policy noise and observation statistics;
the new terrain channels use the existing expansion and gate ramp. The terrain
critic and optimizer are initialized afresh. Learning rate is inherited from
the stage-one checkpoint, then follows the original PPO adaptive-KL schedule.
The terrain gate ramps from 0 to 1 over 200 PPO iterations, as before.

```bash
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  scripts/instinct_rl/train.py \
  --task Instinct-PGMT-TerrainPPOValidation-G1-Newton-v0 \
  --num_envs 4096 --max_iterations 2000 --viz none \
  --run_name dance1_subject2_no_boxes_ppo \
  --logroot /mnt/data/JIN/PGMT/InstinctLab-local/ppo-validation-20261031/train \
  agent.warm_start=/mnt/data/JIN/PGMT/InstinctLab-local/checkpoints/stage1/pgmt_native_2700/model_2700.pt \
  agent.save_interval=100
```

The deployment wrapper supplies the verified interpreter, source path, assets,
data paths and persistent caches. Four ranks each have 4096 environments:
16384 total, 24 steps/environment/iteration, 393216 samples per PPO iteration.
PPO retains five learning epochs and four minibatches, four reward groups,
gamma=0.99, lambda=0.95 and its original noise, randomization and terminations.

## Implementation decisions and verification boundary

- Capacity tests exposed MuJoCo Warp `nefc overflow` with the original 384
  constraint rows per environment. This drops constraints beyond the buffer,
  even when PPO losses and checkpoints stay finite. The diagnostic preset now
  uses 1024 constraint rows on Newton, keeping the original 192 contact-point
  budget, physics solver, terrains and PPO settings. Capacity must be measured
  again with this larger buffer. The native task still uses its original value.
  `env.sim.physics.solver_cfg.njmax` overrides the resolved Newton setting.
- The diagnostic behavior is a new task preset; the original task defaults
  remain available. No global environment variables select a hidden profile.
- The initial formal run is bounded to 2000 iterations, rather than the native
  30000 default, after a short same-scale startup smoke. Duration is controlled
  by `--max_iterations`; this is an initial training run, not a quality claim.
- Initial level remains the native L0 for reproducible stage-one adaptation.
  High difficulty is retained in geometry and curriculum, not assumed learned.
- Changing selection starts a new warm-started run. New checkpoints record
  motion filenames, realized terrain columns and sampling settings, and reject
  resume with changed recorded settings. Legacy checkpoints without this field
  retain their original loading behavior; geometry changes still require a new
  run, since this field is not a full terrain-mesh identity hash.
- Logs, runtime manifests, checkpoint hashes, smoke/train job IDs and measured
  startup evidence live outside Git under `InstinctLab-local/ppo-validation-20261031/`.
- Finite losses and saved checkpoints demonstrate operational training only.
  Tracking, terrain traversal, and WM feasibility require later evaluations.

## Pure PPO reward ablations (2026-11-01)

These two tasks change the **PPO reward objective**, unlike the preceding WM
head ablations, which all retained the same 27 PPO terms:

| Task | Removed terms | Active terms / critic values |
| --- | --- | --- |
| `Instinct-PGMT-TerrainPPONoForce-G1-Newton-v0` | `auxiliary/impact`, `terrain/stumble`, `terrain/force` | 24 / 4 |
| `Instinct-PGMT-TerrainPPONoContactForce-G1-Newton-v0` | Above, plus `auxiliary/undesired_contact`, `terrain/touchdown`, `terrain/match`, `terrain/slip`, `terrain/switch` | 19 / 3 |

The force boundary follows the actual reward dependencies: impact, stumble and
force use force magnitude/direction, while the other five terms depend on
contact state or contact timing. Binary contact sensors may themselves use a
force threshold; the first task retains that contact-state signal. Neither
task removes contact sensors or contact labels: they remain necessary for
reference support height, terrain adaptation and unchanged curriculum behavior.
Thus the second task removes explicit contact/force **rewards**, not all contact
information from the environment.

Terms are set to `None`, rather than weight zero. The second task has no terrain
reward group at all; its critic and rollout storage therefore use three values
(upper/lower/auxiliary). Keeping an empty fourth head would add a noisy learned
baseline to the actor advantage. The critic still has one shared trunk, and
group advantages are summed and normalized by the existing PPO implementation.
The number of critic outputs is the only network difference between the two
reward objectives; the actor, observation dimensions and height scanner match.

Both runs independently warm-start the original stage-one `model_2700.pt` and
start with fresh terrain critics/optimizers. They use seed 42, four GPUs with
4096 environments each, 2000 iterations, 24 rollout steps, five epochs and four
minibatches. Motion selection, terrain proportions/difficulty, randomization,
terminations, last-level resampling curriculum and 200-iteration terrain gate
are identical to `TerrainPPOValidation`. There is no WM fit or BPTT update.
Launch using the command above with the desired task and a distinct run name
and log root. No additional reward override is required. The full-reward task
remains available to restore the original objective.

Checkpoints have distinct task schemas and record the active reward names,
weights and parameters. Resume rejects changed recorded objectives. Original
stage-one initialization uses `agent.warm_start`, not stage-two resume; the
three-value critic cannot load a four-value stage-two critic. Legacy full-task
checkpoints without the reward-contract field retain their original behavior.

`scripts/pgmt/validate_terrain.py` checks actual term counts/value widths,
stage-one action parity, shared critic gradients, summed grouped returns,
timeout bootstrap, rejection of changed reward weights and exact restoration
of model, optimizer, normalizers and sampler. It supports one or 40 motions.
Resource snapshots, immutable source manifests, smoke reports, distributed job
IDs, runtime/GPU measurements and checkpoints are stored outside Git under
`InstinctLab-local/ppo-reward-ablation-20261101/`.

Compare tracking errors, failures/completions, curriculum exposure and videos
at matched evaluation settings; reward sums across these objectives have
different meanings. A successful launch or finite checkpoint does not establish
tracking quality. The proposed SuperMimic height-map experiment without contact
rewards is a subsequent experiment after inspecting these PPO results.
