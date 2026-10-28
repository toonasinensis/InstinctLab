# PGMT MuJoCo playback

Run from this checkout:

```bash
bash source/instinctlab/instinctlab/tasks/shadowing/pgmt/msic/play.sh
```

The default checkpoint is the downloaded native terrain `model_9600.pt`. Override it with
`--checkpoint /path/model_N.pt` (the run must also contain `params/env.yaml`), or set
`PGMT_CHECKPOINT`. Set `PGMT_MUJOCO_PYTHON` to select a prepared Python environment.
Local playback dependencies live outside the checkout in the sibling `InstinctLab-reference/`
directory. Set `PGMT_REFERENCE_DIR` to override that location.
The web control panel is http://127.0.0.1:6020 ; change it with `--port`.

This existing MuJoCo player supports one physical robot. The reference ghost, height scan,
and four terrain attention regions are enabled. The camera remains free; scene changes
preserve the camera instead of following the robot.

In the panel, select a motion, terrain and L0--L9; click the terrain map for XY, set heading
and source time, then choose placement-and-pause or placement-and-play. The scene preview
and MuJoCo collision geometry use the same layout. Flat, upward/downward stairs, random
boxes and the fixed 10/20/30/40 cm box scene are supported. `--start-xy`, `--heading-deg`,
`--motion`, `--start-frame`, `--terrain` and `--level` also set the initial scene.
Space pauses; R resets; G toggles the reference; H toggles the scan; V switches source vs
terrain-adjusted reference display. Time-based reset defaults off (`--reset-seconds 0`);
failure, motion-end and tile-exit resets stay active.

The player loads the canonical native `TerrainActor` weights, keeps checkpoint observation
statistics frozen, and bypasses normalization for the 441 terrain values. Joint mapping,
PD, action scales, armature, limits and reset duration come from the saved environment.
NPZ remains WXYZ at the boundary. Scan point order matches the native ray pattern exactly;
the panel converts it to physical XY for display. There is no training, automatic evaluation,
rollout recording or background counterfactual evaluation. Input comparison modes run only
when explicitly selected in the panel.

The G1 MJCF/common helpers are copied from `wbc_parkour/whole_body_tracking/scripts/deploy`;
the existing PGMT placement panel and terrain/runtime are copied from the corresponding
`Desktop/hajimi/whole_body_tracking` player. They are package-owned here, with no runtime
imports from either source repository. Meshes reuse `instinctlab/assets/resources/unitree_g1`, with only the distinct head mesh under `pgmt_g1`.
MuJoCo's explicit clipped torque PD and local primitive terrain differ from Newton training.
This is visual sim-to-sim playback, not a quantitative training evaluation.

`python scripts/pgmt/validate_mujoco.py` checks actor-output parity, native scan ordering,
all 40 catalog entries and placements/headings on all supported terrain layouts. Evidence:
`agents/pgmt_mujoco_validation_20261028/`.
