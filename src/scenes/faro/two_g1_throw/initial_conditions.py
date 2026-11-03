import numpy as np
from faro.configs.scene import StateSpec, Standing, Pose, OnFace


def default(scene):
    return StateSpec(
        robots={
            "g1a": Standing(on=("floor", "surface"), xy_yaw=(-0.35, 0.0, 0.0)),
            "g1b": Standing(on=("floor", "surface"), xy_yaw=(1.8, 0.0, np.pi)),
        },
        objects={
            "floor": Pose(),
            "white_box": OnFace(on=("floor", "surface"), xy_yaw=(0.0, 0.0, 0.0)),
        },
    )

def lift(scene):
    return StateSpec(
        robots={
            "g1a": Standing(on=("floor", "surface"), xy_yaw=(-0.35, 0.0, 0.0)),
            "g1b": Standing(on=("floor", "surface"), xy_yaw=(.5, 0.0, np.pi)),
        },
        objects={
            "floor": Pose(),
            "white_box": OnFace(on=("floor", "surface"), xy_yaw=(0.0, 3.0, 0.0)),
        },
    )


INITIAL_CONDITIONS = {"default": default,
                      "lift": lift}
