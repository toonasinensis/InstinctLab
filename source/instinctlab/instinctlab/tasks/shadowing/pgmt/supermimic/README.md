# Composed SuperMimic backend for PGMT

This backend follows the desktop SuperMimic separation of representation,
world-model environment, and learners. It preserves the PGMT observation,
terrain, reference-motion, reset and reward semantics. `SuperMimicRunner` is a
standalone coordinator; it does not inherit any of the old PGMT runners or
construct PPO rollout storage/an optimizer for PPO.

## Review order

1. `interfaces.py`: six small abstract boundaries: `StateRepresentation`,
   `TerrainQuery`, `DynamicsModel`, `Policy`, `WorldModelEnv`, `Learner`.
2. `config.py` and `layouts.py`: frozen algorithm configuration and named tensor
   layouts. State dimensions, prediction widths, history, reference windows and
   observation slices are derived from the robot/task rather than repeated offsets.
3. `setup.py`: one composition root returning an explicit `Components` dataclass.
4. `runner.py`: collection, supervised WM fitting, actor/critic update, logging.
5. `world_model_env.py` and `learners/`: numerical optimization and gradient ownership.
6. `execution.py` and `checkpoint.py`: execution policy and persistence.

```text
Newton environment -> Collector -> ReplayBuffer
                                  |          |
                                  |          +-> sampled starts + external noise
                                  v                              |
                         WorldModelLearner                 WorldModelEnv
                         H8 supervised fit                reset / step / rollout
                                                                |
                                                          PolicyLearner
                                                          BPTT actor
                                                          imagined critic
                                                          passive actor EMA
```

The model environment receives the policy, dynamics, value model and task MDP;
it does not create optimizers. The learners do not step Newton. Compilation and
CUDA Graph capture are independent configuration choices and do not own model
parameters. Captures fail explicitly if state storage moves; checkpoint loading
invalidates captures before rebuilding optimizer state.

## Reuse and deliberate PGMT adaptations

The reference is HybridRobotics/SuperMimic commit
`5efbf114e11f725c31d83e217be987c13c0e4ce2`, the local desktop source used for the
comparison. `execution.py` reuses its `CapturedUpdate`/`graph_state` implementation.
The learner flow, optimizer partition, and representation/environment boundary
follow its learner and generalized-coordinate modules. Attribution and the
Apache-2.0 license are included in `NOTICE` and `LICENSE-SuperMimic`.

The native learners cannot be imported unchanged: PGMT's actor uses history and
reference windows, terminal transitions carry actual pre-reset states, and
support height/contact memory feed later imagined observations. These adaptations
are explicit in `mdp.py`, `collector.py`, `replay_buffer.py` and
`world_model_env.py`. No old Runner is instantiated or monkey-patched.

The existing PGMT CNN/TF networks, FK/reward implementations, contact rewards,
heightfield queries, quaternion math, running statistics and episode accumulator
are reused. The legacy network constructor also builds a frozen PPO critic; it
is retained only in the portable inference export, and never used in SM learning.
The trainable critic is the independent scalar imagined critic in `models.py`.

## Configuration axes

All algorithm fields below are under `agent.supermimic`.

| Feature | Configuration | Semantics |
|---|---|---|
| WM state | `representation.kind` | `delta`: all local deltas; `velocity`: velocity deltas with analytic pose integration; `structured`: drift plus learned residual |
| WM scales | `representation.loss_scale` | `running` or fixed `bryson`; errors are measured on integrated states |
| Contact prediction | `dynamics.predict_contact`, `contact_bce_weight` | Two-foot probability head; no force head |
| Actor horizon/batch | `policy.horizon`, `batch_size` | Independent of WM supervised horizon/batch |
| Imagined critic | `policy.gamma`, `lam`, `critic_epochs`, `critic_minibatches` | Detached imagined features and lambda-return targets |
| Value bound | `policy.value_upper_bound` | Optional cap on bootstrap/target values; default `None` |
| Value-gradient fitting | `policy.value_gradient_weight` | Start-state imagined-return derivative; default 0; requires `execution.cuda_graph=false` |
| EMA | `policy.ema_decay` | Passive actor average; real collection and training continue to use raw actor |
| Optimizers | `{actor,wm,critic}_optimizer.kind` | Adam, AdamW or Muon plus Adam; hidden Linear matrices follow the desktop grouping rule |
| Schedule | `schedule.duration`, `min_factor` | Cosine schedule, indexed by learner rounds including WM warmup; device LR stays mutable inside captured Muon updates |
| Real collection | `collection.steps_per_update` | Fractional accumulator, e.g. 2.5 alternates 2 and 3; no imagined transitions counted as real data |
| Execution | `execution.compile` | `none`, `selective`, `chain` |
| CUDA Graph | `execution.cuda_graph` | Captures WM, actor and critic optimizer updates separately after eager warmup |
| Activation recomputation | `execution.actor_recompute`, `wm_recompute` | Observation noise is sampled outside recomputation/capture |

`selective` compiles actor/dynamics network forwards, WM loss, rollout objective,
and critic loss forward. `chain` additionally compiles the imagined step, feature
construction and batched FK; the horizon loop stays explicit Python control flow.
Neither mode silently falls back to eager on compilation failure.

The task config retains `foot_contact`, `analytic_foot`, `flat_warmup_updates` and
`termination_softness`. Rewards reuse the existing formulas and scales. Disabling
the BCE head also requires disabling contact-dependent rewards. Analytic support,
penetration, foot-edge and acceleration calculations still use FK/terrain; the
contact-gated terms use predicted probabilities during imagination.

Real scans come from the task's configured native observation term (including
raycast when configured there). Imagined scans use differentiable bilinear
queries. Root ground height is queried under the root and never taken from the
center index of an offset scan. Physics timestep/decimation determine the
semi-implicit integration coefficient rather than a fixed `0.625` literal.

## Recipes and entry points

The isolated recipes live in `../config/g1/pgmt_supermimic_cfg.py`:

- `Instinct-PGMT-SuperMimicBaseline-G1-Newton-v0`: previous A semantics;
  delta state, running loss scale, eager, WM Muon/AdamW and actor/critic Adam.
- `Instinct-PGMT-SuperMimic-G1-Newton-v0`: previous F semantics;
  velocity state, Bryson scale, Muon/Adam, selective compilation and CUDA Graphs.
- `Instinct-PGMT-SuperMimicTF-G1-Newton-v0`: original four-glimpse TF and 21x21
  scan using the same composed backend. Selector diagnostics are intentionally
  omitted per the user request.

Both CNN recipes reuse the scan187 Newton environment. The default recipe uses
4096 environments, actor H32/batch4096, WM H8/batch2048/10 updates, prefill32,
warmup32, collect2.5, seed42 and the previously tested learning rates. The launch
script's existing CLI can override these explicitly; no training is started by
importing this package.

```bash
python scripts/instinct_rl/train.py \
  --task Instinct-PGMT-SuperMimic-G1-Newton-v0 \
  --num_envs 4096 --max_iterations 50000 --run_name REVIEWED_NAME \
  --logroot /absolute/path/to/formal/training --viz none
```

## Checkpoints and outputs

The new schema saves models, normalizers, both learners' optimizers/counters,
EMA, RNG states, LR clock and curriculum atomically in one write. Resume validates
the effective algorithm/task contract. Simulator episodes and replay are reset
and refilled; resume is not an exact simulator-state continuation.

`initialize_checkpoint` imports actor/normalization from native stage-one TF
weights, or model weights from a compatible legacy stage-two checkpoint. It is
weight initialization, not optimizer continuation. A loaded normalizer stays
frozen during warmup. Changed architectures fail on incompatible tensor shapes.
Existing actor/normalizer keys and raw/EMA weights remain available to inference
loaders; this does not make the legacy WM-viewer Runner a new-schema trainer.

Formal events are written only to `<run>/tensorboard/`. Completed episodes retain
`Episode_Reward = episode sum / configured maximum seconds`; `RealReward` and
`ImaginedReward` retain dt-scaled per-transition means. Detailed WM diagnostics
are JSONL; TensorBoard keeps a small WM overview. Collection, WM, actor/critic and
full iteration timings are logged separately. Full iteration time includes the
preceding iteration's logging/save overhead; startup/prefill and compile warmups
are identifiable separately. Save interval is 1000 and all saved nodes persist.

Validation belongs outside formal training directories. Tests are under
`source/instinctlab/test/pgmt/test_supermimic_components.py` and
`validate_supermimic.py`. Run them in the pinned Newton training environment.
The implementation currently supports single-GPU learners; distributed launches
raise explicitly. Multi-GPU gradient synchronization is not claimed by this refactor.
