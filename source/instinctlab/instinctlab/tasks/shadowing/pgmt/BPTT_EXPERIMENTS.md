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
Each case has four GPUs and 4096 real environments per GPU, 24 real control
steps per collection, 2000 actor-training iterations and seed 42. Motion is
`dance1_subject2`; terrain proportions, L0--L9 geometry and the highest-level
resampling curriculum are inherited from the PPO validation preset.

The 19 remaining PPO reward formulas and their weights/control-dt scaling are
also the complete imagined reward objective. There are no contact/force reward
terms or contact/force WM heads or contact inputs to dynamics. Reference contact
labels still determine the reference support-height filter, as in the real task.
The actual reward contract is saved in the checkpoint and per-rank JSON.

## Initialization and data

Both initialization sources are explicit CLI arguments:

- `agent.ppo_checkpoint`: the completed 19-reward PPO run 25836, `model_2000.pt`.
  Actor, policy normalizer and terrain gate are loaded. Its grouped PPO critic
  is retained only for native checkpoint/play compatibility and is frozen/unused.
- `agent.wm_checkpoint`: the prior no-contact/no-force supervised WM run 25779,
  `model_2000.pt`. Only WM weights and moments are initialized, not its PPO policy
  or optimizers. It was fitted under a 27-reward PPO collector, so recalibration
  on the new policy distribution is necessary. The source has 16000 WM updates;
  the new optimizer and new update count start afresh.

These are transfer initializations, not exact continuation of either source run.
This first matrix tests BPTT integration from a stable terrain policy, not whether
stage1-only BPTT can learn stage2 from scratch.

There are 32 preliminary collection/WM/value-fitting rounds with the actor
frozen, followed by 2000 actor updates. Warmup is additional to the requested
2000 actor-training iterations. The first round has only 24 replay steps, so
value fitting starts in round two. `warmup.pt` preserves the state at that boundary.
WM train length stays 8, with eight AdamW updates per collection, local batch
512 and learning rate 3e-4. Real replay spans 48 steps. Validation retains
H1/4/8/16/32 and environment holdout; neither WM nor actor/value fitting samples
local `env_id % 8 == 0`.

Actor batch is separately 256 per GPU (1024 globally); it does not equal the
4096 real simulator environments. Actor Adam uses learning rate 1e-5 and
betas (0.9, 0.95); gradients are averaged across ranks, then clipped by global
norm 1 and elementwise magnitude 0.1. Critic Adam uses learning rate 3e-4,
five epochs and four minibatches, hence 20 updates per fitting round in both
groups. Gamma/lambda are 0.99/0.95. These are controlled diagnostic choices,
not a reproduction claim for the original paper's training budget.

## What real versus imagined means

The two critics have identical architecture and initialization. Both condition
on normalized policy observations/history, current kinematic features, fixed
simulator randomization, support clearance, terrain level/family and motion
phase. Neither reads an actor latent. Policy normalization is frozen. Critic
features use existing WM input statistics rather than fitting source-dependent
normalization models.

All cases sample a 32-step real window before selecting the requested H-prefix.
This keeps anchor-age/reference eligibility rules independent of H. Reference
windows come from the known motion bank and episode placement, not future
simulated robot states. Proprioceptive history and previous actions are carried
through the differentiable imagined rollout.

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
Actions are the actor output plus fixed recorded Gaussian noise (std 0.1).
Using the same noise prefix in real and imagined windows avoids changing the
exploration distribution when comparing the two value targets. The policy
remains differentiable through its mean actions.

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
stepped. The WM uses FP32 for this initial gradient experiment to make numerical
derivative checks interpretable. All six cases disable observation corruption
in both real and imagined inputs; this isolates the mechanism and does not
establish sensor-noise robustness. Action noise and physical randomization remain.

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
`InstinctLab-local/bptt-critic-horizon-20261101/`.

Compare real tracking/error/failure metrics at matched checkpoints and
evaluation conditions. Imagined return can improve while real behavior worsens;
finite checkpoints or successful launches alone do not establish policy quality.
