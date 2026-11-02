# BPTT training with real or imagined critic supervision

This is the pure model-gradient actor training path. The PPO and supervised-WM
training paths remain available. The actor keeps its cross-attention and terrain
network. The independent scalar critic uses 512/256/128 SiLU hidden layers,
with no actor intent or shared parameters.

## Current baseline

The task is `Instinct-PGMT-TerrainBPTT-G1-Newton-v0`. Actor imagination defaults
to **H32**. The supported comparison is critic supervision from **real** or
**imagined** transitions, selected by `agent.bptt.critic_source=real|imagined`.
The default remains `imagined`; both modes use the same actor/WM training path.
`agent.bptt.horizon` remains one recorded length parameter, rather than a set of
H8/H16/H32 experiment presets.

Use two GPUs with 4096 real environments and actor batch 4096 per GPU. Collection
alternates 2/3 real control steps per learner round (UTD 0.4). The default run has
5000 actual actor updates, seed 42 and motion `dance1_subject2`. Terrain
proportions, L0--L9 geometry and highest-level resampling follow the PPO validation
preset. Nineteen native noncontact reward terms, including their weights and
control-dt scaling, form the complete imagined objective. There are no contact
or force WM heads or dynamics inputs. Reference contact labels still determine
the reference support-height filter used by the real task.

The decision to use H32 and its limits are recorded in the
[debugging experience index](../../../../../../agents/pgmt/DEBUGGING_NOTES.md).
The completed six-condition horizon study is historical; its exact source,
configuration and results remain in the immutable `c0fa1cc` snapshot and the
external `InstinctLab-local/bptt-vibed-aligned-20261101/` artifacts. This document
describes the current training protocol, not the historical submission queue.

## Initialization, sampling and optimization

Choose an explicit `agent.ppo_checkpoint`. The current baseline uses the completed
19-reward PPO25836 `model_2000.pt`: actor, policy normalization and terrain gate
are transferred. Its grouped PPO critic is frozen and unused, retained for native
checkpoint/play compatibility. The independent value MLP and its optimizer are
fresh. This is a transfer from a terrain PPO actor, not stage1-only learning.

The WM uses 2048/2048/2048 Linear/LayerNorm/SiLU layers. By default,
`agent.wm_checkpoint=None` initializes it from scratch. An optional WM checkpoint
must have matching width and no contact/force heads; incompatible widths are not
partially loaded. Simulator episodes, failure sampling and curriculum start
fresh at L0. Critic architecture and normalization are identical in both modes.

The replay window covers the maximum of the actor rollout, WM training and WM
validation lengths: **32 steps with the current defaults**. The configured initial
collection must cover that window. Prefill 32 real steps without optimizer
updates, then run 32 actor-frozen collection/WM/value-fitting rounds, followed by
5000 actual actor updates. `warmup.pt` preserves the boundary. Candidate reference
windows must fit the complete replay window plus the reference observation
context. Eligibility is filtered before gathering full replay tensors.

WM supervision remains **H8**, batch **2048 per GPU**, with **10 AdamW updates per
round**, lr 3e-4 and weight decay 1e-5. Loss is the valid-transition mean. Actor
Adam uses lr 1e-5, betas (0.9, 0.95), global gradient norm 1 and elementwise clip
0.1. Critic Adam uses lr 3e-4, five epochs and four minibatches: 20 updates per
round. Gamma/lambda are 0.99/0.95. The independent value head starts at zero.
**FP32 remains in use and WM AMP is disabled.**

Actor and WM forwards use non-reentrant activation recomputation during actor
backward. Complete recomputation avoids propagating the checkpoint early-stop
exception through scripted quaternion operations. Observation noise is drawn
outside those deterministic forwards and is not resampled during backward.
Direct/recomputed loss and gradient parity is part of acceptance validation.

Replay stores **1000 control steps per environment** (20 seconds). One already
noisy proprioceptive frame, an episode-step counter and a nine-frame prefix allow
exact reconstruction of 10-frame histories through resets, overwrite and refill.
Physical state/reward/context storage remains FP32. WM prediction validation
retains **H1/4/8/16/32**; these diagnostic horizons are independent of the retired
actor-horizon sweep. Local `env_id % 8 == 0` is excluded from fitting, WM moments
and actor/value training.

Without interruption, expected per-rank totals are 5032 learner rows, 5000 actor,
100640 critic, 50320 new WM updates, 12612 collected vectorized control steps
and zero PPO updates when the WM starts fresh. The collection total includes the
initial 32 steps. The alternating 2/3 cadence and separate RNGs are checkpointed.
A resume starts fresh simulator episodes and refills replay, legitimately adding
collection steps. It is not bitwise physics continuation. Checkpoint option and
schema validation still reject mismatched training settings.

## Critic supervision

Both critics condition on normalized policy observations/history, current
kinematic features, fixed simulator randomization, support clearance, terrain
level/family and motion phase. Neither reads an actor latent. Policy normalization
is frozen; critic features use the existing WM input statistics.

- **Real:** states and native rewards come from recent real replay. Lambda
  returns use pre-reset successors. Failures zero bootstrap; timeouts bootstrap
  terminal value but stop recursion into the reset episode. A post-reset mask
  excludes later transitions in that sampled window. This is bootstrapped
  behavior-data supervision, not an exact infinite-horizon return.
- **Imagined:** states, analytic rewards and soft continuation come from the
  actor's WM rollout. Lambda targets are computed once, detached and held fixed
  throughout critic fitting. This includes model-state/target error and soft
  versus hard failure semantics, not only a state-distribution difference.

Both use B x H candidate targets and the same optimizer-update count. The real
mode can have fewer valid targets; `critic_valid_fraction` records this. Critic
training-window length still follows actor H. Decoupling them is a separate
algorithm change, not part of the default-H32 cleanup.

## Noise, termination and gradients

Only real collection adds Gaussian action exploration (std 0.1). Imagined actions
are exactly the actor output. Real and imagined policy inputs use native
additive-uniform noise: gravity +/-0.05, angular velocity +/-0.2, joint position
+/-0.01, joint velocity +/-0.5, reference joint position +/-0.05, reference joint
velocity +/-0.5 and reference orientation/height +/-0.05. Last action, corrected
velocity and heightmap have no configured noise. Noise is applied before native
clipping/scaling. Historical frames are not noised again; physical WM targets,
analytic reward states and soft-failure states remain clean. Actor and real-critic
noise use independent saved generators.

Soft failure uses the actual root-height, reference-relative tilt and maximum
body-error thresholds, with the same lower-body tolerance. Sigmoid continuation
widths are 0.03 m for height, 0.04 in cosine tilt margin and 0.04 m for body error.
The terminating transition receives its reward before later survival decays.
Normal horizon truncation bootstraps terminal value. Real termination and
curriculum rules are unchanged.

WM and critic parameters are frozen during actor backward; their input
derivatives remain active. State/history/heightmap tensors are not detached.
WM and critic have independent optimizers. The PPO optimizer is never stepped.
Terrain descriptors remain fixed at the sampled anchor during imagination;
tile-boundary descriptor changes are not modeled. Local heightmaps are sampled
at each predicted pose. The raster covers the physical mesh border, keeping one
cell inside its outer ray edge. Leaving its finite domain rejects the update
collectively and saves the model plus each rank's actual sampled window and
imagined trajectory. Invalid samples are not silently dropped or filled.

## Verification and evidence

`scripts/pgmt/validate_bptt.py` checks native observation parity/noise, timeout
returns, independent critic parameters, future heightmap gradients, frozen WM
parameters, policy directional finite differences, exact optimizer/RNG reload,
invalid-window capture and distributed consistency. Computational gradient
agreement does not establish the accuracy of real dynamics Jacobians.

Runs record warmup separately, update counts, critic validity, real tracking and
failure/completion metrics, imagined survival, WM fixed/current holdouts,
phase timings and CUDA peaks. Keep manifests, reports, logs and checkpoints in
`InstinctLab-local/` outside Git. Use matched evaluation conditions to compare
policies: finite checkpoints and an improving imagined return alone do not
establish tracking quality. See the linked experience index for observed trends,
mechanism hypotheses and future diagnostic directions.
