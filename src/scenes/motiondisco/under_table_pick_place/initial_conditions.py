from dataclasses import replace
import numpy as np
from faro.configs.scene import OnFace, Pose, Standing, StateSpec


rob_x_disp = 0.85 + 0.0458282865
rob_y_disp = -0.0706162381

def default(scene):
    return StateSpec(
        robots={"robot": Standing(on=('floor', 'surface'))},
        objects={
            'floor': Pose(),
            'table_top_short': Pose(q=[1.0, 0.0, 0.6965, 0.0, 0.0, 0.0, 1.0]),
            'table_side_left': OnFace(on=('floor', 'surface'), xy_yaw=[1.0, 0.665, 0.0]),
            'table_side_right': OnFace(on=('floor', 'surface'), xy_yaw=[1.0, -0.665, 0.0]),
            'table_back': OnFace(on=('floor', 'surface'), xy_yaw=[1.315, 0.0, 0.0]),
            'white_box': OnFace(on=('floor', 'surface'), xy_yaw=[0.9, 0.0, 0.0]),
        },
    )


def goal(scene):
    initial = default(scene)
    return replace(
        initial,
        objects={
            **initial.objects,
            'white_box': OnFace(on=('table_top_short', 'top')),
        },
    )


INITIAL_CONDITIONS = {'default': default, 'goal': goal}
