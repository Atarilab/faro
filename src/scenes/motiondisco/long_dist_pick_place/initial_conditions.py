from dataclasses import replace
from faro.configs.scene import OnFace, Pose, Standing, StateSpec


def default(scene):
    return StateSpec(
        robots={"robot": Standing(on=('floor', 'surface'))},
        objects={
            'floor': Pose(),
            'table_short': OnFace(on=('floor', 'surface'), xy_yaw=[2.5, 0.0, 0.0]),
            'yellow_box': OnFace(on=('floor', 'surface'), xy_yaw=[0.5, 0.0, 0.0]),
        },
    )


def goal(scene):
    initial = default(scene)
    return replace(
        initial,
        objects={
            **initial.objects,
            'yellow_box': OnFace(on=('table_short', 'top')),
        },
    )


INITIAL_CONDITIONS = {'default': default, 'goal': goal}
