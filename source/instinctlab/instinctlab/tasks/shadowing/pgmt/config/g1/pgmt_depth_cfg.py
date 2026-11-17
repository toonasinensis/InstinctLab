"""v010 vision on the PGMT SM recipe, Contact5 without penetration."""

from dataclasses import asdict, dataclass, replace

from isaaclab.managers import ObservationTermCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass

from instinctlab.assets.unitree_g1 import G1_29DOF_LINKS
from instinctlab.sensors import get_link_prim_targets
from instinctlab.tasks.shadowing.perceptive.perceptive_env_cfg import PerceptiveShadowingSceneCfg
from instinctlab.tasks.shadowing.pgmt.mdp.depth_observations import depth_image
from instinctlab.tasks.shadowing.pgmt.supermimic.config import ContactRewardCfg
from instinctlab.tasks.shadowing.pgmt.supermimic.perception.config import DepthCfg
from instinctlab.utils.noise import DepthNormalizationCfg, GaussianBlurNoiseCfg
from instinctlab.utils.urdf import urdf_importer_link_prim_path

from .pgmt_scan187_cfg import G1PGMTScan187EnvCfg
from .pgmt_supermimic_cfg import G1PGMTSuperMimicCfg


@dataclass(frozen=True)
class DepthSensorCfg:
    # Roban base-mounted v010 camera mapped onto the G1 pelvis (XYZW, OpenGL).
    position: tuple = (0.11108819629771827, -0.0012500096857547757, 0.3453298227590574)
    rotation: tuple = (0.17969101517256078, -0.17969101517256078, -0.6838940993065041, 0.6838940993065041)
    focal_length: float = 2.12
    horizontal_aperture: float = 4.24
    vertical_aperture: float = 2.4479651413640133
    min_distance: float = 0.1
    max_distance: float = 1000000.0
    blur_sigma: float = 1.0
    scanner_height: float = 20.0


@configclass
class DepthActorCfg:
    class_name: str = "instinctlab.tasks.shadowing.pgmt.supermimic.perception.pgmt_models:PGMTDepthActor"
    perception: dict = asdict(DepthCfg())


@configclass
class G1PGMTDepthEnvCfg(G1PGMTScan187EnvCfg):
    depth_sensor: dict = asdict(DepthSensorCfg())

    def __post_init__(self):
        super().__post_init__()
        visual, spec = DepthCfg(), DepthSensorCfg(**self.depth_sensor)
        self.scene.camera = camera = PerceptiveShadowingSceneCfg().camera.copy()
        camera.prim_path = urdf_importer_link_prim_path(self.scene.robot.spawn.asset_path, "pelvis")
        camera.offset.pos, camera.offset.rot, camera.offset.convention = spec.position, spec.rotation, "opengl"
        camera.mesh_prim_paths.extend(get_link_prim_targets(G1_29DOF_LINKS, self.scene.robot.spawn.asset_path))
        camera.update_period = self.sim.dt * self.decimation
        camera.pattern_cfg = patterns.PinholeCameraPatternCfg(
            focal_length=spec.focal_length,
            horizontal_aperture=spec.horizontal_aperture,
            vertical_aperture=spec.vertical_aperture,
            height=visual.image_shape[1],
            width=visual.image_shape[2],
        )
        camera.min_distance, camera.max_distance = spec.min_distance, spec.max_distance
        camera.depth_clipping_behavior = "max"
        camera.data_histories = {}
        camera.noise_pipeline = dict(
            gaussian_blur=GaussianBlurNoiseCfg(sigma=spec.blur_sigma),
            depth_normalization=DepthNormalizationCfg(depth_range=(0.0, visual.depth_far_m), output_range=(-0.5, 0.5)),
        )
        self.observations.policy.depth_image = ObservationTermCfg(func=depth_image, history_length=0)
        scan = self.height_scan
        self.scene.reconstruction_scanner = RayCasterCfg(
            prim_path=camera.prim_path,
            offset=RayCasterCfg.OffsetCfg(pos=(*scan["offset"], spec.scanner_height)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=scan["spacing"], size=scan["size"], ordering="xy"),
            mesh_prim_paths=["/World/ground"],
            update_period=self.sim.dt * self.decimation,
            debug_vis=False,
        )


@configclass
class G1PGMTDepthSMCfg(G1PGMTSuperMimicCfg):
    runner_class_name: str = "instinctlab.tasks.shadowing.pgmt.supermimic.runners.pgmt_depth_runner:PGMTDepthRunner"
    policy: DepthActorCfg = DepthActorCfg()
    foot_contact: dict = asdict(replace(ContactRewardCfg(), penetration=0.0))
    experiment_name: str = "pgmt_supermimic_depth"
    run_name: str = "V010_SingleDepth_ZDetach_Contact5"
