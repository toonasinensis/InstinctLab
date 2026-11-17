"""Load frozen depth actors through the same model and policy classes used in training."""

import copy

from instinctlab.tasks.shadowing.pgmt.supermimic.config import ExecutionCfg
from instinctlab.tasks.shadowing.pgmt.supermimic.perception.config import DepthCfg
from instinctlab.tasks.shadowing.pgmt.supermimic.perception.pgmt_models import DepthPolicyNormalizer, PGMTDepthPolicy


def load_policy(saved, num_actions, device, weights):
    from isaaclab.utils import string_to_callable

    policy_cfg = copy.deepcopy(saved["supermimic"]["contract"]["policy"])
    actor_type = string_to_callable(policy_cfg.pop("class_name"))
    spec = DepthCfg(**policy_cfg["perception"])
    actor = actor_type(num_actions, **policy_cfg).to(device).eval().requires_grad_(False)
    actor.load_state_dict(
        (
            saved["scan187"]["actor_ema"]
            if weights == "ema"
            else {k.removeprefix("actor."): v for k, v in saved["model_state_dict"].items() if k.startswith("actor.")}
        ),
        strict=True,
    )
    norm = DepthPolicyNormalizer(spec.physical_width + spec.image_width, spec.height_points, spec)
    norm.load_state_dict(saved["policy_normalizer_state_dict"], strict=True)
    norm = norm.to(device).eval().requires_grad_(False)
    execution = ExecutionCfg(**saved["supermimic"]["contract"]["options"]["execution"])
    policy = PGMTDepthPolicy(actor, norm, execution).eval()
    return policy, norm
