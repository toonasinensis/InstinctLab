"""The optical camera is constructed directly at the G1 head; labels stay pelvis-aligned."""

import gymnasium as gym
from unittest.mock import patch

import pytest

from instinctlab.tasks.shadowing.pgmt.config.g1 import pgmt_head_camera_cfg as recipe
from instinctlab.tasks.shadowing.pgmt.config.g1.pgmt_terrain_recipe_cfg import G1PGMTTerrainRecipeEnvCfg
from instinctlab.utils.urdf import urdf_importer_link_prim_path


@pytest.mark.parametrize("env_type", [recipe.G1PGMTHeadDepthEnvCfg, recipe.G1PGMTHeadDepthLafanBoxEnvCfg])
def test_only_head_camera_is_constructed_and_supervision_stays_on_pelvis(env_type):
    with patch.object(recipe, "NoisyGroupedRayCasterCameraCfg", wraps=recipe.NoisyGroupedRayCasterCameraCfg) as camera:
        cfg = env_type()
    assert camera.call_count == 1
    mount = cfg.head_camera
    path = cfg.scene.robot.spawn.asset_path
    assert mount["parent"] == "torso_link"
    assert camera.call_args.kwargs["prim_path"] == urdf_importer_link_prim_path(path, "torso_link")
    assert cfg.scene.camera.offset.pos == mount["position"]
    assert cfg.scene.camera.offset.rot == mount["rotation"]
    assert cfg.scene.camera.offset.convention == mount["convention"] == "opengl"
    assert cfg.scene.reconstruction_scanner.prim_path == urdf_importer_link_prim_path(path, "pelvis")
    assert recipe.G1PGMTHeadDepthEnvCfg.__bases__ == (G1PGMTTerrainRecipeEnvCfg,)


def test_pose_is_required_and_legacy_pelvis_task_is_not_registered():
    with pytest.raises(TypeError):
        recipe.HeadDepthSensorCfg()
    assert not any(task.startswith("Instinct-PGMT-SuperMimicDepth-G1") for task in gym.registry)
