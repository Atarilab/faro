from dataclasses import replace
from faro.configs.scene import OnFace, Pose, Standing, StateSpec


def default(scene):
    return StateSpec(
        robots={"robot": Standing(on=('floor', 'surface'))},
        objects={
            'floor': Pose(),
            'white_box': OnFace(on=('floor', 'surface'), xy_yaw=[0.6, 0.0, 0.0]),
        },
    )


def goal(scene):
    initial = default(scene)
    return replace(
        initial,
        objects={
            **initial.objects,
            'white_box': OnFace(on=('floor', 'surface'), xy_yaw=[1.6, 0.0, 0.0]),
        },
    )


INITIAL_CONDITIONS = {'default': default, 'goal': goal}
