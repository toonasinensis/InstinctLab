"""CrossAttention actor and scalar critic for G1 LAFAN tracking."""

from isaaclab.utils import configclass

from instinctlab.utils.wrappers.instinct_rl import (
    InstinctRlActorCriticCfg,
    InstinctRlNormalizerCfg,
    InstinctRlOnPolicyRunnerCfg,
    InstinctRlPpoAlgorithmCfg,
)


@configclass
class PolicyCfg(InstinctRlActorCriticCfg):
    class_name = "instinctlab.tasks.shadowing.pgmt.models.cross_attention:CrossAttentionActorCritic"
    init_noise_std = 1.0
    actor_hidden_dims = [1024, 512, 256]
    critic_hidden_dims = [4096, 2048, 1024, 512, 256]
    activation = "silu"


@configclass
class AlgorithmCfg(InstinctRlPpoAlgorithmCfg):
    value_loss_coef = 1.0
    use_clipped_value_loss = True
    clip_param = 0.2
    entropy_coef = 0.005
    num_learning_epochs = 5
    num_mini_batches = 4
    learning_rate = 1e-3
    optimizer_class_name = "Adam"
    schedule = "adaptive"
    gamma = 0.99
    lam = 0.95
    desired_kl = 0.01
    max_grad_norm = 1.0


@configclass
class NormalizersCfg:
    policy = InstinctRlNormalizerCfg()
    critic = InstinctRlNormalizerCfg()


@configclass
class G1PGMTShadowingPPORunnerCfg(InstinctRlOnPolicyRunnerCfg):
    runner_class_name: str = "instinctlab.tasks.shadowing.pgmt.runner:PGMTRunner"
    policy: PolicyCfg = PolicyCfg()
    algorithm: AlgorithmCfg = AlgorithmCfg()
    normalizers: NormalizersCfg = NormalizersCfg()
    seed = 42
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 100
    log_interval = 1
    experiment_name = "g1_pgmt_lafan"
    run_name = "CrossAttentionFlat"
    init_at_random_ep_len = False
