from dataclasses import replace
from faro.configs.scene import OnFace, Pose, Standing, StateSpec


def default(scene):
    return StateSpec(
        robots={"robot": Standing(on=('floor', 'surface'))},
        objects={
            'floor': Pose(),
            'table_short1': OnFace(on=('floor', 'surface'), xy_yaw=[1.0, 0.8, 3.14 / 2]),
            'table_short2': OnFace(on=('floor', 'surface'), xy_yaw=[1.0, -0.8, -3.14 / 2]),
            'white_box': OnFace(on=('floor', 'surface'), xy_yaw=[0.9, 0.2, 0.0]),
            'yellow_box': OnFace(on=('floor', 'surface'), xy_yaw=[1.0, -0.175, 0.0]),
        },
    )


def goal(scene):
    initial = default(scene)
    return replace(
        initial,
        robots={"robot": Standing(xy_yaw=[2.0, 0.0, 0], on=('floor', 'surface'))},
    )


INITIAL_CONDITIONS = {'default': default, 'goal': goal}
