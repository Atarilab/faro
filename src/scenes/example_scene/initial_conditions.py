from dataclasses import replace
from faro.configs.scene import OnFace, Pose, Standing, StateSpec


def default(scene):
    return StateSpec(
        robots={"robot": Standing(on=('floor', 'surface'))},
        objects={
            'floor': Pose(),
            'table1': OnFace(on=('floor', 'surface'), xy_yaw=[1.0, 0.0, 0.0]),
            'box1': OnFace(on=('floor', 'surface'), xy_yaw=[0.3, 0.0, 0.0]),
        },
    )


def goal(scene):
    initial = default(scene)
    return replace(
        initial,
        objects={
            **initial.objects,
            'box1': OnFace(on=('table1', 'top')),
        },
    )


INITIAL_CONDITIONS = {'default': default, 'goal': goal}
