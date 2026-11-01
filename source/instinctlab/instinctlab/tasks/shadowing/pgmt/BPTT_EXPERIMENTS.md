# BPTT critic-source and horizon experiment

This is a new pure model-gradient actor experiment. The existing PPO and
supervised-WM tasks remain available. The actor uses its existing cross-attention
and terrain-glimpse network. The new critic is an independent scalar MLP with
hidden widths 512/256/128 and SiLU, with no actor intent or shared parameters.

## Six cases

| Critic supervision | Policy imagination horizon |
| --- | --- |
| Real simulator transitions | 8, 16, 32 |
| Imagined transitions | 8, 16, 32 |

All cases use `Instinct-PGMT-TerrainBPTT-G1-Newton-v0` with
`agent.bptt.critic_source=real|imagined` and `agent.bptt.horizon=8|16|32`.
Each case has two GPUs and 4096 real environments per GPU, alternating 2/3 real control
steps per collection (UTD 0.4), 5000 actor-training iterations and seed 42. Motion is
`dance1_subject2`; terrain proportions, L0--L9 geometry and the highest-level
resampling curriculum are inherited from the PPO validation preset.

The 19 remaining PPO reward formulas and their weights/control-dt scaling are
also the complete imagined reward objective. There are no contact/force reward
terms or contact/force WM heads or contact inputs to dynamics. Reference contact
labels still determine the reference support-height filter, as in the real task.
The actual reward contract is saved in the checkpoint and per-rank JSON.

## Initialization and data

The aligned matrix initializes `agent.ppo_checkpoint` from the completed
19-reward PPO run 25836, `model_2000.pt`. Actor, policy normalizer and terrain
gate are loaded. Its grouped PPO critic is frozen and unused, retained for
native checkpoint/play compatibility. The independent value MLP is fresh.

The WM is now 2048/2048/2048 with LayerNorm and SiLU, instead of 512/512/512.
`agent.wm_checkpoint=None` explicitly creates a fresh WM: the old WM25779 cannot
be strictly loaded at this width. No partial loading, width transformation or
old optimizer/moment transfer occurs. The optional WM checkpoint argument only
accepts a matching width. Each case receives the same initialization seeds.
Simulator episodes, failure sampling and curriculum start fresh at L0; this is
still a transfer experiment from a terrain PPO actor, not stage1-only learning.

Prefill 32 real control steps without optimizer updates, enough for the shared
H32 anchors and validation. Retain the existing 32 actor-frozen collection/WM/
value-fitting rounds, followed by **5000 actual actor updates**. Do not introduce
the proposed 500 WM-only rounds: the user approved only the listed alignment
changes. `warmup.pt` preserves the boundary. Fresh large-WM prediction metrics
must be reported; a finite gradient does not establish a reliable model.

WM training remains H8, now with **10 AdamW steps per round**, batch **2048 per
GPU**, lr 3e-4 and weight decay 1e-5. Loss remains the valid-transition mean.
Actor batch is **4096 per GPU / 8192 globally**. Actor Adam remains lr 1e-5,
betas (0.9, 0.95), global gradient norm 1 and elementwise clip 0.1. Critic Adam
remains lr 3e-4, 5 epochs x 4 minibatches, hence 20 updates per round. Critic
512/256/128 SiLU, zero final head and its existing input/normalization are
unchanged. Gamma/lambda remain 0.99/0.95. There is no new LR scaling or schedule.
**FP32 remains in use and WM AMP stays disabled**; BF16 is deferred.

Replay capacity is exactly **1000 control steps per environment** (20 seconds).
It stores one already-noisy proprioceptive frame, an episode-step counter and a
nine-frame prefix before the oldest retained row. Sampling reconstructs exact
10-frame histories, including native first-frame repetition after reset, ring
overwrite and refill. All physical state/reward/context storage remains FP32.
This avoids storing the same ten-frame history 1000 times without changing the
sample distribution. Validation remains H1/4/8/16/32; local env_id % 8 == 0 is
excluded from WM fitting/moments and actor/value training.

With no interruption the expected per-rank totals are 5032 learner rows,
5000 actor, 100640 critic and 50320 new WM updates, 12612 collected vectorized
control steps and zero PPO updates. The 32-step prefill is additional; alternating
2/3 cadence is saved in checkpoints. Resuming uses fresh simulator episodes and
refills 32 steps, so collection totals then legitimately include that refill.
New checkpoints use `pgmt-bptt-v2` and preserve separate observation-noise RNGs.

## What real versus imagined means

The two critics have identical architecture and initialization. Both condition
on normalized policy observations/history, current kinematic features, fixed
simulator randomization, support clearance, terrain level/family and motion
phase. Neither reads an actor latent. Policy normalization is frozen. Critic
features use existing WM input statistics rather than fitting source-dependent
normalization models.

All cases sample a 32-step real window before selecting the requested H-prefix.
This keeps anchor-age/reference eligibility rules independent of H. The proposed
separate actor/real-critic sampling redesign is not included in this selective
alignment; actor anchors still require a completed H32 real window. Reference
windows come from the known motion bank and episode placement, not future
simulated robot states. Proprioceptive history and previous actions are carried
through the differentiable imagined rollout.
Terrain level/family descriptors stay fixed at the sampled anchor during an
imagined window, while the raster is sampled at each predicted pose. Descriptor
changes when crossing a tile boundary are not modeled in this initial matrix.

- **Real critic:** kinematic states and native 19-term rewards come from the
  recent real window. Lambda returns use each transition's pre-reset successor.
  Physical failures zero bootstrap; timeouts bootstrap terminal value but stop
  lambda recursion from entering the reset episode. A post-reset mask excludes
  subsequent transitions in the same sampled window. This is bootstrapped
  supervision from recent behavior data, not an exact infinite-horizon oracle.
- **Imagined critic:** states, analytic rewards and analytic soft continuation
  come from the actor's WM rollout. Lambda targets are computed once, detached
  and held fixed during critic fitting. The comparison therefore includes the
  effects of model-state/target error and soft versus hard failure semantics;
  it is not solely a test of state-distribution coverage.

Both use B x H candidate targets and the same optimizer-update count. The real
group may have fewer valid targets because of actual episode boundaries; log
`critic_valid_fraction` rather than silently filling missing targets with zeros.
Only real collection adds Gaussian action exploration (std 0.1). Imagined
actions are exactly the actor output, without recorded or freshly sampled action
noise. Real and imagined policy observations both use the native additive-uniform
noise amplitudes: gravity +/-0.05, angular velocity +/-0.2, joint position
+/-0.01, joint velocity +/-0.5, reference joint position +/-0.05, reference joint
velocity +/-0.5, reference orientation/height +/-0.05. Last action, corrected
velocity and heightmap have no configured noise. Noise is added before native
clipping/scaling. Historical frames are not noised again. Noise does not modify
WM ground-truth supervision, analytic reward states or soft-failure states.
Actor-imagination noise and real-critic noise use independent saved generators;
finite-difference validation restores the same noise draw on each perturbation.

## Analytic soft termination and actor gradients

Failure errors use the actual task thresholds: root height, reference-relative
tilt, and maximum body tracking error with the same lower-body tolerance.
Their sigmoid continuations are multiplied. Height width is 0.03 m, tilt uses
cosine margin width 0.04 and body-error width is 0.04 m. Real simulator
termination and curriculum rules are unchanged.

The actor maximizes discounted analytic rewards plus terminal value. The
terminating transition receives its reward before subsequent survival decays.
Normal finite-horizon truncation bootstraps terminal value. Imagined reference
windows must fit inside the source clip. Leaving the global terrain raster
rejects the actor update collectively and fails explicitly; it is not rewarded
by dropping invalid samples. Motion/tile reset discontinuities do not enter the
differentiated simulator trajectory.

During actor backward, WM and value parameters are frozen but their input
derivatives remain active. Subsequent state/history/heightmap tensors are not
detached. WM and value have separate optimizers; the PPO optimizer is never
stepped. The WM continues using FP32; all six cases now enable configured
observation corruption in real and imagined policy inputs. Real action exploration
and physical randomization remain. This alone does not establish robustness.

## Verification and reporting

`scripts/pgmt/validate_bptt.py` checks nonterrain observation parity, timeout
returns, independent value parameters, gradient flow from future heightmap to
earlier actions, frozen WM parameters, a policy-parameter directional finite
difference, actual actor updates, frozen policy statistics, exact optimizer
reload and distributed model consistency. Computational derivative agreement
does not prove WM Jacobians match real contact dynamics. A real-simulator
counterfactual gradient test remains a separate validation concern.

Every case records warmup separately, actor/critic/WM update counts, critic data
validity, real tracking errors/failure/completion counts, imagined survival,
validation errors, phase timings and CUDA peaks. Checkpoints include both source
paths, independent value/actor optimizers and per-rank RNGs. Resume refills real
episodes/replay; it is not bitwise physics continuation. Runtime manifests,
resource evidence, immutable snapshots and results live outside Git in
`InstinctLab-local/bptt-vibed-aligned-20261101/`. The prior matrix is retained
under `bptt-critic-horizon-20261101/` with its original immutable snapshots.

The aligned run counters are specified above. The historical matrix used
40256 new WM updates plus a 16000-update source, 100620 critic updates and
120768 real steps per rank; these old expectations must not validate new runs.

Compare real tracking/error/failure metrics at matched checkpoints and
evaluation conditions. Imagined return can improve while real behavior worsens;
finite checkpoints or successful launches alone do not establish policy quality.

## Raster coverage correction

The first imagined-H16 attempt (25880) completed 140 actor updates, then failed
the collective validity guard. Its original offending window was not saved.
A fixed-model100 diagnostic made 56 scans with no optimization and did not
reproduce that window. It did independently confirm that all four points just
outside the old raster `[-42,42]^2` still hit the physical terrain mesh: the
mesh has a 10 m border, whereas the old bake included only 2 m. This establishes
a coverage bug, not the original failure's exact cause or policy convergence.

The corrected bake uses the configured physical border, keeping one raster
cell inside its outer ray edge. It adds neither terrain nor an invented floor.
Samples beyond that finite domain still fail collectively before optimization.
Such failures now save the current model and each rank's sampled window and
imagined states/actions, allowing the actual failed batch to be examined.
The validator checks native-mesh border heights and deliberate invalid-window
capture without a parameter update. Existing healthy attempts keep their
immutable snapshots; a repaired attempt records its distinct version and budget.
The same old guard subsequently ended real-H32 at 218 actor updates,
imagined-H32 at 210 and real-H16 at 272. Their replacements use the same
coverage correction; all original failure records are retained. The two H8
attempts continue on the original snapshot while valid. Version differences
and actual task IDs are recorded in `active-experiments.json` outside Git.
