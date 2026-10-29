# Pure world-model terrain tracker

`Instinct-PGMT-PureWM-G1-Newton-v0` uses the original PGMT stage1 actor and a
standalone SuperMimic learner. The historical `WorldModel` task remains the
PPO-plus-imagination experiment. The new runner does not construct PPO, a teacher,
or the PGMT shared-intent critic. Stage2 is an evaluation reference only.

Fresh initialization requires the original `model_2700.pt` SHA256
`a4620507acd757baf31c239f8503eb4ccd72b950e21dfd8ece94d2f0ebf54305`.
The deployment network is the unchanged `TerrainActor`. Its first 1645 serialized
observation channels and normalization statistics retain the stage1 layout;
serialization is term-major, while attention tokens are frame-major.

## Initial experiment

- WM: three 2048-wide Linear/LayerNorm/SiLU layers, local 70-value state delta,
  14 contact logits, and 42 force components. Only the MLP uses bf16.
- The fixed mesh is baked at 5 cm. Bilinear queries at predicted XY/yaw rebuild
  the original 21 by 21 actor crop at 10 cm spacing. Mesh-raster resolution and
  actor observation spacing are distinct.
- Training-only simulator context contains the applied 32 friction values,
  29 joint offsets and 3 torso COM offsets. The deployment actor does not receive
  these values. Randomized restitution did not affect the pinned MuJoCo Warp
  solver's material arrays, so it is excluded.
  The supported material, joint-default and COM events run only at startup,
  before runner construction. Their per-environment snapshots therefore remain
  valid across resets. Construction rejects reset/interval modes, and
  `xi_schema` records the event modes and binding lifetime. Episode-varying
  randomization requires transition-local context snapshots before it can be
  supported. Older checkpoints without this metadata fail strict resume; any
  metadata migration must first verify their recorded event configurations and
  preserve the original checkpoint and all tensor contents.
- Chronological replay holds 1000 real steps per environment. Initial/refill
  collection is 64 steps; the learner collects 2/3 alternating steps per update
  (`utd=0.4`). WM fitting uses recursive 8-step windows and online normalization
  updated once per collected transition.
- WM defaults: 10 updates per iteration, global batch 2048. Actor global batch
  is 4096, accumulated in microbatches. All three optimizers use base LR 0.0002
  divided by the square root of rank count; actor Adam betas are 0.9/0.95.
- The initial actor horizon is **8 steps (0.16 s)**. H32 prediction errors are
  measured from the start. This shorter first experiment limits accumulated
  model error while terrain adaptation is established; a subsequent H32 run
  needs an explicit configuration change and renewed physical checks.
- Actor updates begin after 500 WM-only iterations and physical error checks.
  Initial H8 limits are root coordinate RMSE 0.08 m, rotation error 0.3 rad and
  joint RMSE 0.2 rad, with at least 50% valid windows. Both a frozen held-out set
  and current held-out data must pass. Stationary-state errors and per-family
  diagnostics provide scale/context; these gates do not prove gradient fidelity.
- The independent scalar critic uses reconstructed native critic features,
  contact state, simulator context and terrain. It fits detached imagined
  gamma=0.99/lambda=0.95 returns. It never consumes mutable actor intent.
- Policy moments are frozen at the transferred stage1 values; added velocity
  channels initially use identity statistics, and heights remain raw. The first
  experiment disables observation corruption to match reconstructed histories.
  Collection and imagination both add action noise of standard deviation 0.1
  and clamp normalized targets to [-20,20]. Evaluation uses clamped mean actions.

## Reward and boundaries

Native weights/stds are read from the live reward manager. Tracking, root motion,
action rate, joint limits and acceleration costs use the native analytic kernels
and actual COM velocity convention. Reference motion advances from the sampled
episode's frozen placement and local frame, without reading future replay rows.

Terrain tangent slip and touchdown quality use differentiable heightfield normals
and relief. Contact probabilities are detached when weighting rewards, so lowering
a predicted contact probability cannot directly erase a slip cost. Contact match,
switch and undesired-contact terms remain detached return components. The first
actor objective omits force, stumble and impact costs; their signed native reward
contributions are logged, and a force head is supervised for later experiments.
The checkpoint records this explicit approximation contract and its hash.

Imagined trajectories receive the current transition reward before survival is
updated. Physical survival uses native height/tilt/body thresholds, including
lower-body tolerance. Leaving the assigned tile absorbs the imagined trajectory.
Replay transitions include the terminal state captured before reset, and all
successor rows after a reset are masked. Actor starts require a complete motion
window; a real replay termination never truncates a different imagined action
trajectory.
If reference coverage is exhausted on any rank, every rank receives the same
collective failure before starting that microbatch's imagined rollout or entering
gradient averaging. No rank continues the actor optimizer after that failure.

## Running and interpreting results

Use `scripts/instinct_rl/train.py` with the new task and
`agent.warm_start=/absolute/path/to/model_2700.pt`. `--num_envs` is **per rank**;
WM/actor batch settings are global. The standard distributed launcher remains
applicable. Source data/checkpoints and logs belong outside the repository.

Pure checkpoints use `pgmt-purewm-v1`, preserve all three optimizers and counters,
and reject hybrid/stage2 resumes. Resume starts new episodes and refills replay;
it is optimizer continuation, not exact simulator continuation. Algorithm or
reward configuration changes require a new explicitly designed experiment.
This includes a horizon-only change: strict resume deliberately rejects it.

Execution settings live separately in `agent.pure_wm_execution`: `compile`,
`wm_microbatch`, `actor_microbatch`, and `profile`. Zero microbatch means one full
local batch. The algorithm's global batches, unroll lengths, rewards, and optimizer
step counts are unchanged. Old checkpoint options remain strictly checked; both
the previous execution settings and the current settings are written to the run
contract. Splitting random samples differently can change stochastic trajectories
and floating-point reduction order; this is not bitwise training replay.

The optimized learner compiles tensor-only WM sequences, causal actor steps,
batched imagined objectives, kinematics, and critic forwards. Compilation uses
`fullgraph=True` and fails visibly on graph breaks. Sampling, reference validation,
collectives, and moment updates stay outside compiled graphs. Reference windows
are read for all H+1 frames together. Future local maps, support filtering, and
policy actions still advance sequentially and preserve BPTT. Each state's
kinematics is reused by support, reward, and critic features; rewards and critic
features are then computed over the combined time/sample batch. Survival still
pays a terminating transition's reward before absorbing later transitions.

Use `agent.pure_wm_execution.profile=True` for synchronized CUDA phase timings
in `timing/*` log fields. With profiling off these fields measure host elapsed
time, not a CUDA kernel trace. Separate the initial compile/fill time from steady
iteration throughput and report peak allocated/reserved GPU memory. Compare the
same rank count, per-rank environment count, global learner batches, H, and update
counts; a shorter rollout or fewer optimizer steps is not an execution speedup.

`benchmark.load_actor` accepts pure checkpoints alongside stage1/stage2. A smoke
run with small networks and permissive gates verifies execution only. It must
never be reported as a trained terrain result. First-result evaluation must show
reference-versus-actual trajectories and videos, global position/orientation and
motion progress, and applicable terrain passage. Survival alone is insufficient.
The MuJoCo replay `TerrainPolicy` also accepts pure checkpoints and retains their
action clamp; these videos are a separate deployment replay from the native
Newton benchmark and must be identified as such.
Matched PPO controls, systematic seeds and detailed derivative studies follow
the first useful result; final stage2-equivalence remains an empirical objective.
