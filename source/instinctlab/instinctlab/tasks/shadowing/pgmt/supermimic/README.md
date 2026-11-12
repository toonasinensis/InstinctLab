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
