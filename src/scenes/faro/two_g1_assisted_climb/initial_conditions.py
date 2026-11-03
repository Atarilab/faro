import numpy as np
from faro.configs.scene import StateSpec, Standing, Pose, OnFace


def default(scene):
    return StateSpec(
        robots={
            "g1a": Standing(on=("floor", "surface"), xy_yaw=(-0.3, 0.0, 0.0)),
            "g1b": Standing(on=("floor", "surface"), xy_yaw=(.3, 0.0, np.pi)),
        },
        objects={
            "floor": Pose(),
            "table_tall": OnFace(on=("floor", "surface"), xy_yaw=(0.0, 0.75, 0.0)),
        },
    )


INITIAL_CONDITIONS = {"default": default}
