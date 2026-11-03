import numpy as np
from faro.configs.scene import StateSpec, Standing, Pose, OnFace


def default(scene):
    return StateSpec(
        robots={
            "g1a": Standing(on=("floor", "surface"), xy_yaw=(-0.35, 0.0, 0.0)),
            "g1b": Standing(on=("floor", "surface"), xy_yaw=(1.8, 0.0, np.pi)),
            "g1c": Standing(on=("floor", "surface"), xy_yaw=(0.725, 1.86, -np.pi / 2)),
        },
        objects={
            "floor": Pose(),
            "white_box": OnFace(on=("floor", "surface"), xy_yaw=(0.0, 0.0, 0.0)),
        },
    )


INITIAL_CONDITIONS = {"default": default}
