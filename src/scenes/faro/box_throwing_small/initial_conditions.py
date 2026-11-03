from faro.configs.scene import OnFace, Pose, Standing, StateSpec


def default(scene):
    return StateSpec(
        robots={"robot": Standing(on=('floor', 'surface'))},
        objects={
            'floor': Pose(),
            'table_short': OnFace(on=('floor', 'surface'), xy_yaw=[1.25, 0.0, 0.0]),
            'white_box': OnFace(on=('floor', 'surface'), xy_yaw=[0.35, 0.0, 0.0]),
        },
    )


def pick_place(scene):
    return StateSpec(
        robots={"robot": Standing(on=('floor', 'surface'))},
        objects={
            'floor': Pose(),
            'table_short': OnFace(on=('floor', 'surface'), xy_yaw=[2.0, 0.0, 0.0]),
            'white_box': OnFace(on=('floor', 'surface'), xy_yaw=[0.35, 0.0, 0.0]),
        },
    )


def goal(scene):
    return StateSpec(
        robots={"robot": Standing(on=('floor', 'surface'))},
        objects={
            'floor': Pose(),
            'table_short': OnFace(on=('floor', 'surface'), xy_yaw=[1.0, 0.0, 0.0]),
            'white_box': OnFace(on=('table_short', 'top')),
        },
    )


INITIAL_CONDITIONS = {'default': default, 'pick_place': pick_place, 'goal': goal}
