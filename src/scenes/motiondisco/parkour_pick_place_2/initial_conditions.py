from dataclasses import replace
from faro.configs.scene import OnFace, Pose, Standing, StateSpec


def default(scene):
    return StateSpec(
        robots={"robot": Standing(xy_yaw=[-1.0, 0.0, 0], on=('floor', 'surface'))},
        objects={
            'floor': Pose(),
            'table_short': OnFace(on=('floor', 'surface')),
            'table_tall': OnFace(on=('floor', 'surface'), xy_yaw=[0.8, 0.0, 0.0]),
            'white_box': OnFace(on=('floor', 'surface'), xy_yaw=[-0.7, 0.0, 0.0]),
        },
    )


def goal(scene):
    initial = default(scene)
    return replace(
        initial,
        robots={"robot": Standing(xy_yaw=[0.0 - 1.0, 0.0, 0], on=('floor', 'surface'))},
        objects={
            **initial.objects,
            'white_box': OnFace(on=('table_tall', 'top'), xy_yaw=[-0.1, 0.0, 0.0]),
        },
    )


INITIAL_CONDITIONS = {'default': default, 'goal': goal}
