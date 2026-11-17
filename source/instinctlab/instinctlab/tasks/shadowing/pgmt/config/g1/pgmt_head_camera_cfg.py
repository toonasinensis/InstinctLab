"""G1 head-mounted depth and paired full-LAFAN/box recipes.

Reference: unitreerobotics/unitree_ros, robots/g1_description/g1_29dof.urdf,
head_joint and d435_joint. The head-relative mount is transferred to the local
PGMT asset, whose head_joint is 1 cm higher than the reference asset.
"""

import math
import numpy as np
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, replace
from scipy.spatial.transform import Rotation

from isaaclab.utils import configclass

from instinctlab.tasks.shadowing.pgmt.supermimic.config import ContactRewardCfg
from instinctlab.utils.urdf import urdf_importer_link_prim_path

from .pgmt_depth_cfg import DepthSensorCfg, G1PGMTDepthEnvCfg, G1PGMTDepthSMCfg
from .pgmt_scan187_cfg import G1PGMTScan187EnvCfg
from .pgmt_supermimic_cfg import G1PGMTSuperMimicCfg


@dataclass(frozen=True)
class G1CameraReference:
    """Meters and radians from the official fixed joints, not Roban offsets."""

    torso_head_xyz: tuple = (0.0039635, 0.0, -0.054)
    torso_camera_xyz: tuple = (0.0576235, 0.01753, 0.41987)
    torso_camera_pitch: float = 0.8307767239493009
    optical_convention: str = "opengl"


def head_camera_mount(asset_path):
    """Preserve the official head-relative camera pose on the local G1 asset."""
    reference = G1CameraReference()
    joint = ET.parse(asset_path).getroot().find("joint[@name='head_joint']")
    if joint is None or joint.find("parent").get("link") != "torso_link":
        raise ValueError("Head-camera recipe requires the fixed G1 torso/head joint")
    if joint.get("type") != "fixed":
        raise ValueError("A movable head needs an explicit camera body instead")
    origin = joint.find("origin")
    local_head = np.eye(4)
    local_head[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
    local_head[:3, :3] = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")).as_matrix()
    reference_head = np.eye(4)
    reference_head[:3, 3] = reference.torso_head_xyz
    reference_camera = np.eye(4)
    reference_camera[:3, 3] = reference.torso_camera_xyz
    reference_camera[:3, :3] = Rotation.from_euler("y", reference.torso_camera_pitch).as_matrix()
    mount = local_head @ np.linalg.inv(reference_head) @ reference_camera
    # The sensor accepts XYZW in OpenGL axes; URDF camera link is X-forward/Z-up.
    optical = mount[:3, :3] @ Rotation.from_euler("XYZ", (math.pi / 2, -math.pi / 2, 0)).as_matrix()
    return dict(
        parent="torso_link",
        position=tuple(float(x) for x in mount[:3, 3]),
        rotation=tuple(float(x) for x in Rotation.from_matrix(optical).as_quat()),
        convention=reference.optical_convention,
        reference=asdict(reference),
    )


@configclass
class G1PGMTHeadDepthEnvCfg(G1PGMTDepthEnvCfg):
    head_camera: dict = {}

    def __post_init__(self):
        super().__post_init__()
        self.head_camera = mount = head_camera_mount(self.scene.robot.spawn.asset_path)
        camera = self.scene.camera
        camera.prim_path = urdf_importer_link_prim_path(self.scene.robot.spawn.asset_path, mount["parent"])
        camera.offset.pos, camera.offset.rot, camera.offset.convention = (
            mount["position"],
            mount["rotation"],
            mount["convention"],
        )
        self.depth_sensor = asdict(replace(DepthSensorCfg(), position=mount["position"], rotation=mount["rotation"]))
        camera.debug_vis = False
        # Reconstruction targets stay on the original pelvis/yaw grid. Moving
        # the optical camera must not silently move the 187 ground-truth labels.
        self.scene.reconstruction_scanner.debug_vis = False


def use_lafan_boxes(env):
    """The existing 40-clip corpus and Boxes20 recipe, shared by both sensors."""
    env.scene.motion_reference.motion_buffers["LAFAN"].motion_names = None
    proportions = dict(flat=0.2, slope_up=0.1, slope_down=0.1, stairs_up=0.1, stairs_down=0.1, boxes=0.2, rough=0.2)
    for name, terrain in env.scene.terrain.terrain_generator.sub_terrains.items():
        terrain.proportion = proportions[name]


@configclass
class G1PGMTHeadDepthLafanBoxEnvCfg(G1PGMTHeadDepthEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        use_lafan_boxes(self)


@configclass
class G1PGMTHeightLafanBoxEnvCfg(G1PGMTScan187EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        use_lafan_boxes(self)


@configclass
class G1PGMTHeadDepthSMCfg(G1PGMTDepthSMCfg):
    run_name: str = "G1HeadDepth"


@configclass
class G1PGMTHeightContact5SMCfg(G1PGMTSuperMimicCfg):
    foot_contact: dict = asdict(replace(ContactRewardCfg(), penetration=0.0))
    run_name: str = "HeightLafanBoxContact5"
