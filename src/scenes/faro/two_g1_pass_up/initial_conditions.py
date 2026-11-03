import numpy as np
from faro.configs.scene import StateSpec, Standing, Pose, OnFace


def default(scene):
    return StateSpec(
        robots={
            "g1a": Standing(on=("floor", "surface"), xy_yaw=(-0.35, 0.0, 0.0)),
            "g1b": Standing(on=("platform", "top"), xy_yaw=(0.0, 0.8, 3*np.pi/2)),
        },
        objects={
            "floor": Pose(),
            "platform": OnFace(on=("floor", "surface"), xy_yaw=(0.0, 2.0, 0.0)),
            "table": OnFace(on=("platform", "top"), xy_yaw=(-1.0, 0.5, 0.0)),
            "white_box": OnFace(on=("floor", "surface"), xy_yaw=(0.0, 0.0, 0.0)),
        },
    )


INITIAL_CONDITIONS = {"default": default,}
