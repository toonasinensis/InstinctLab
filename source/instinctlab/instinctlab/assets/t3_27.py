from pathlib import Path

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from instinctlab.sim import UrdfFileCfg

T3_27_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
)

T3_27_TRACKED_BODIES = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_roll_link",
)

T3_27_ACTION_SCALE = {
    name: 0.5 if name == "waist_yaw_joint" or "_hip_" in name or "_knee_" in name or "_ankle_" in name else 1.0
    for name in T3_27_JOINT_NAMES
}


def _actuator(joint_names, stiffness, damping, effort_limit, velocity_limit):
    return ImplicitActuatorCfg(
        joint_names_expr=joint_names,
        stiffness=stiffness,
        damping=damping,
        armature=0.01,
        effort_limit_sim=effort_limit,
        velocity_limit_sim=velocity_limit,
    )


T3_27_CFG = ArticulationCfg(
    spawn=UrdfFileCfg(
        asset_path=str(Path(__file__).resolve().parent / "resources" / "t3_27" / "t_0_3_2_27dof.urdf"),
        fix_base=False,
        merge_fixed_joints=False,
        activate_contact_sensors=True,
        self_collision=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.9),
        joint_pos={
            ".*_hip_pitch_joint": -0.15,
            ".*_knee_joint": 0.3,
            ".*_ankle_pitch_joint": -0.15,
        },
        joint_vel={".*": 0.0},
    ),
    joint_ordering=T3_27_JOINT_NAMES,
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "hip_pitch_roll": _actuator(
            [".*_hip_pitch_joint", ".*_hip_roll_joint"],
            stiffness=60.0,
            damping=10.0,
            effort_limit=180.0,
            velocity_limit=16.232,
        ),
        "hip_yaw": _actuator(
            [".*_hip_yaw_joint"], stiffness=100.0, damping=4.0, effort_limit=120.0, velocity_limit=16.092
        ),
        "knees": _actuator([".*_knee_joint"], stiffness=100.0, damping=6.0, effort_limit=120.0, velocity_limit=16.092),
        "ankle_pitch": _actuator(
            [".*_ankle_pitch_joint"], stiffness=55.2, damping=6.0, effort_limit=45.0, velocity_limit=9.555
        ),
        "ankle_roll": _actuator(
            [".*_ankle_roll_joint"], stiffness=55.2, damping=8.0, effort_limit=45.0, velocity_limit=9.555
        ),
        "waist": _actuator(["waist_yaw_joint"], stiffness=48.0, damping=2.0, effort_limit=120.0, velocity_limit=16.092),
        "arms": _actuator(
            [".*_shoulder_.*_joint", ".*_elbow_joint"],
            stiffness=48.0,
            damping=2.0,
            effort_limit=45.0,
            velocity_limit=9.555,
        ),
        "wrists": _actuator(
            [".*_wrist_.*_joint"], stiffness=40.0, damping=2.0, effort_limit=10.0, velocity_limit=7.854
        ),
    },
)
