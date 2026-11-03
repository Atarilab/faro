from dataclasses import replace
from faro.configs.scene import OnFace, Pose, Standing, StateSpec


def default(scene):
    return StateSpec(
        robots={"robot": Standing(on=('floor', 'surface'))},
        objects={
            'floor': Pose(),
            'table_tall': OnFace(on=('floor', 'surface'), xy_yaw=[2.0, 0.0, 0.0]),
            'white_box': OnFace(on=('floor', 'surface'), xy_yaw=[0.35, 0.0, 0.0]),
        },
    )


def goal(scene):
    initial = default(scene)
    return replace(
        initial,
        objects={
            **initial.objects,
            'table_tall': OnFace(on=('floor', 'surface'), xy_yaw=[1.0, 0.0, 0.0]),
            'white_box': OnFace(on=('table_tall', 'top')),
        },
    )


INITIAL_CONDITIONS = {'default': default, 'pick_place': default, 'goal': goal}
