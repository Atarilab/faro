"""G1 asset loading and robot-specific planning defaults."""
import mujoco
import numpy as np
import pinocchio as pin

from assets.g1_29dof import G1_XML
from faro.configs.robot import ActuatorLimits, RobotConfig
from faro.collision.robot_collision_model import CapsuleSpec, RobotCollisionModel

_DEFAULT_G1_QJ = [
    -0.15, 0, 0, 0.45, -0.3, 0,
    -0.15, 0, 0, 0.45, -0.3, 0,
    0, 0, 0,
    0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0,
]

# Joint torque (Nm) and no-load velocity (rad/s) used by the torque-speed envelope.
_ACTUATORS = {
    "left_hip_pitch_joint": (88, 32),
    "left_hip_roll_joint": (139, 20),
    "left_hip_yaw_joint": (88, 32),
    "left_knee_joint": (139, 20),
    "left_ankle_pitch_joint": (50, 30),
    "left_ankle_roll_joint": (50, 30),
    "right_hip_pitch_joint": (88, 32),
    "right_hip_roll_joint": (139, 20),
    "right_hip_yaw_joint": (88, 32),
    "right_knee_joint": (139, 20),
    "right_ankle_pitch_joint": (50, 30),
    "right_ankle_roll_joint": (50, 30),
    "waist_yaw_joint": (88, 32),
    "waist_roll_joint": (50, 30),
    "waist_pitch_joint": (50, 30),
    "left_shoulder_pitch_joint": (25, 37),
    "left_shoulder_roll_joint": (25, 37),
    "left_shoulder_yaw_joint": (25, 37),
    "left_elbow_joint": (25, 37),
    "left_wrist_roll_joint": (25, 37),
    "left_wrist_pitch_joint": (5, 22),
    "left_wrist_yaw_joint": (5, 22),
    "right_shoulder_pitch_joint": (25, 37),
    "right_shoulder_roll_joint": (25, 37),
    "right_shoulder_yaw_joint": (25, 37),
    "right_elbow_joint": (25, 37),
    "right_wrist_roll_joint": (25, 37),
    "right_wrist_pitch_joint": (5, 22),
    "right_wrist_yaw_joint": (5, 22),
}


def create_g1_models(mj_xml_model_path: str):
    """Load G1 models and register its contact and collision sites as frames."""
    mj_model = mujoco.MjModel.from_xml_path(mj_xml_model_path)
    pin_model = pin.buildModelFromMJCF(mj_xml_model_path)
    pin_model.lowerPositionLimit[:3] = [-1e10] * 3
    pin_model.upperPositionLimit[:3] = [1e10] * 3

    sites = [
        ("right_sole", "right_ankle_roll_joint"),
        ("left_sole", "left_ankle_roll_joint"),
        ("right_ee", "right_wrist_yaw_joint"),
        ("left_ee", "left_wrist_yaw_joint"),
    ]
    sites.extend((cap.frame, cap.parent_frame)
                 for cap in make_g1_collision_model().capsules.values())
    for site_name, parent_name in sites:
        site = mj_model.site(site_name)
        quat_xyzw = site.quat[[1, 2, 3, 0]]
        placement = pin.XYZQUATToSE3(np.r_[site.pos, quat_xyzw])
        pin_model.addFrame(pin.Frame(
            name=site_name,
            parent_joint=pin_model.getJointId(parent_name),
            placement=placement,
            type=pin.FrameType.OP_FRAME,
        ))
    return pin_model, mj_model


def make_g1_collision_model() -> "RobotCollisionModel":
    capsules = {
        "waist" :       CapsuleSpec(
            frame="floating_base_cap",
            parent_frame="floating_base_joint",
            axis=np.array([0.0, 0.0, 1.0]),
            length=0.075,
            radius=0.075,
        ),
        "torso" :       CapsuleSpec(
            frame="waist_pitch_cap",
            parent_frame="waist_pitch_joint",
            axis=np.array([0.0, 0.0, 1.0]),
            length=0.3,
            radius=0.1,
        ),
        "l_upper_arm" : CapsuleSpec(
            frame="left_shoulder_roll_cap",
            parent_frame="left_shoulder_roll_joint",
            axis=np.array([0.0, 0.0, -1.0]),
            length=0.185,
            radius=0.035,
        ),
        "l_forearm" :   CapsuleSpec(
            frame="left_elbow_cap",
            parent_frame="left_elbow_joint",
            axis=np.array([1.0, 0.0, 0.0]),
            length=0.15,
            radius=0.035,
        ),
        "l_wrist" :   CapsuleSpec(
            frame="left_wrist_cap",
            parent_frame="left_wrist_yaw_joint",
            axis=np.array([1.0, 0.0, 0.0]),
            length=0.010,
            radius=0.035,
        ),
        "r_upper_arm" : CapsuleSpec(
            frame="right_shoulder_roll_cap",
            parent_frame="right_shoulder_roll_joint",
            axis=np.array([0.0, 0.0, -1.0]),
            length=0.185,
            radius=0.035,
        ),
        "r_forearm" :   CapsuleSpec(
            frame="right_elbow_cap",
            parent_frame="right_elbow_joint",
            axis=np.array([1.0, 0.0, 0.0]),
            length=0.15,
            radius=0.035,
        ),
        "r_wrist" :   CapsuleSpec(
            frame="right_wrist_cap",
            parent_frame="right_wrist_yaw_joint",
            axis=np.array([1.0, 0.0, 0.0]),
            length=0.010,
            radius=0.035,
        ),
        "l_thigh" :     CapsuleSpec(
            frame="left_hip_roll_cap",
            parent_frame="left_hip_roll_joint",
            axis=np.array([0.0, 0.0, -1.0]),
            length=0.25,
            radius=0.05,
        ),
        "l_shin" :      CapsuleSpec(
            frame="left_knee_cap",
            parent_frame="left_knee_joint",
            axis=np.array([0.0, 0.0, -1.0]),
            length=0.25,
            radius=0.05,
        ),
        "l_foot" :      CapsuleSpec(
            frame="left_foot_cap",
            parent_frame="left_ankle_roll_joint",
            axis=np.array([1.0, 0.0, 0.0]),
            length=0.15,
            radius=0.03,
        ),
        "r_thigh" :     CapsuleSpec(
            frame="right_hip_roll_cap",
            parent_frame="right_hip_roll_joint",
            axis=np.array([0.0, 0.0, -1.0]),
            length=0.25,
            radius=0.05,
        ),
        "r_shin" :      CapsuleSpec(
            frame="right_knee_cap",
            parent_frame="right_knee_joint",
            axis=np.array([0.0, 0.0, -1.0]),
            length=0.25,
            radius=0.05,
        ),
        "r_foot" :      CapsuleSpec(
            frame="right_foot_cap",
            parent_frame="right_ankle_roll_joint",
            axis=np.array([1.0, 0.0, 0.0]),
            length=0.15,
            radius=0.03,
        ),
    }

    self_collision_map = {
        "torso": ["l_upper_arm", "r_upper_arm", "l_forearm", "r_forearm",
                  "l_thigh", "r_thigh",],
        "l_forearm": ["l_thigh", "r_thigh", "l_shin"],
        "r_forearm": ["r_thigh", "l_thigh", "r_shin"],
        "l_wrist": ["r_thigh", "l_thigh", "r_shin"],
        "r_wrist": ["r_thigh", "l_thigh", "r_shin", "r_foot"],
        "l_shin": ["r_thigh", "r_shin", "r_foot"],
        "r_shin": ["l_thigh", "l_shin", "l_foot"],
        "l_foot": ["r_thigh", "r_shin", "r_foot"],
        "r_foot": ["l_thigh", "l_shin", "l_foot"],
    }
    self_pairs = RobotCollisionModel.build_unique_pairs(self_collision_map)

    return RobotCollisionModel(
        capsules=capsules,
        self_collision_pairs=self_pairs,
    )


def make_g1_robot() -> RobotConfig:
    g1_pin_model, _ = create_g1_models(str(G1_XML))
    g1_mj_spec = mujoco.MjSpec.from_file(str(G1_XML))

    return RobotConfig(
        name="g1",
        pin_model=g1_pin_model,
        mj_spec=g1_mj_spec,
        actuator_limits={name: ActuatorLimits(*limits) for name, limits in _ACTUATORS.items()},
        collision_model=make_g1_collision_model(),
        default_qj=list(_DEFAULT_G1_QJ),
        standing_contacts=("left_sole", "right_sole"),
        default_standing_height=0.77468499337981356,
        switch_contacts=("left_ee", "right_ee"),
        plane_contact_exemptions={"l_foot": "left_sole", "r_foot": "right_sole"},
        contact_capsules={"left_sole": ("l_foot",), "right_sole": ("r_foot",),
                          "left_ee": ("l_wrist",), "right_ee": ("r_wrist",)},
        patch_ees={
            "left_sole": np.array([0.05, 0.026, 0.13, 0.026]),
            "right_sole": np.array([0.05, 0.026, 0.13, 0.026]),
            "left_ee": np.array([0.0175, 0.0175, 0.0175, 0.0175]),
            "right_ee": np.array([0.0175, 0.0175, 0.0175, 0.0175]),
        },
    )
