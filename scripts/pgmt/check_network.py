"""Compare the migrated actor with the original module, then check finite PPO gradients."""

import argparse
import importlib.util
import json
import sys
import torch
from pathlib import Path
from tensordict import TensorDict

from instinctlab.tasks.shadowing.pgmt.models.cross_attention import CrossAttentionActor, CrossAttentionActorCritic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-network", required=True)
    parser.add_argument("--source-rsl", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    sys.path.insert(0, args.source_rsl)
    spec = importlib.util.spec_from_file_location("original_pgmt_network", args.source_network)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(42)
    obs = TensorDict({"prop": torch.randn(16, 930), "rbt_cmd_mf": torch.randn(16, 715)}, batch_size=[16])
    original = module.PGMTCrossAttentionModel(
        obs, {"actor": ["prop", "rbt_cmd_mf"]}, "actor", 29, obs_normalization=False
    ).eval()
    migrated = CrossAttentionActor().eval()
    migrated.load_state_dict(original.state_dict(), strict=True)
    flat = torch.cat((obs["prop"], obs["rbt_cmd_mf"]), -1)
    expected = original(obs)["actions"]
    actual = migrated(flat)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    # Exercise the actual learner interface and both policy and value gradients.
    policy = CrossAttentionActorCritic(
        obs_format={"policy": {"state": (1645,)}, "critic": {"state": (296,)}},
        num_actions=29,
        critic_hidden_dims=[4096, 2048, 1024, 512, 256],
        activation="silu",
    )
    actions = policy.act(flat)
    values = policy.evaluate(torch.randn(16, 296))
    loss = -policy.get_actions_log_prob(actions.detach()).mean() + values.square().mean()
    loss.backward()
    gradients = [p.grad for p in policy.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in gradients)
    assert sum(g.abs().sum() for g in gradients) > 0
    assert not any("terrain" in n for n, _ in migrated.named_parameters())
    report = {
        "source_actor_max_abs_error": (actual - expected).abs().max().item(),
        "actor_parameters": sum(p.numel() for p in migrated.parameters()),
        "policy_input": 1645,
        "critic_input": 296,
        "action_dim": 29,
        "finite_actor_and_critic_gradients": True,
        "height_scan": False,
    }
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
