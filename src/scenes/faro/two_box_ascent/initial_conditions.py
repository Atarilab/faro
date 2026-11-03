import numpy as np
from faro.configs.scene import OnFace, Pose, Standing, StateSpec


def default(scene):
    return StateSpec(
        robots={"robot": Standing(on=('floor', 'surface'))},
        objects={
            'floor': Pose(),
            'table_short': OnFace(on=('floor', 'surface'), xy_yaw=[1.1, 0.0, 0.0]),
            'table_tall': OnFace(on=('floor', 'surface'), xy_yaw=[1.8, 0.0, 0.0]),
            'white_box': OnFace(on=('floor', 'surface'), xy_yaw=[0.3, 0.05, 0.0]),
            'black_box': OnFace(on=('floor', 'surface'), xy_yaw=[0.3, -0.4, 0.0]),
        },
    )


INITIAL_CONDITIONS = {'default': default}
