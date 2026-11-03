from faro.configs.scene import OnFace, Pose, Standing, StateSpec


def default(scene):
    return StateSpec(
        robots={"robot": Standing(on=('floor', 'surface'))},
        objects={
            'floor': Pose(),
            'table_short': OnFace(on=('floor', 'surface'), xy_yaw=[1.2, 0.0, 0.0]),
            'table_tall': OnFace(on=('floor', 'surface'), xy_yaw=[0.0, 1.0, 0.0]),
            'white_box': OnFace(on=('floor', 'surface'), xy_yaw=[0.35, 0.25, 0.0]),
            'yellow_box': OnFace(on=('floor', 'surface'), xy_yaw=[0.35, -0.25, 0.0]),
        },
    )


def goal(scene):
    return StateSpec(
        robots={"robot": Standing(xy_yaw=[0.0 - 1.0, 0.0, 0], on=('floor', 'surface'))},
        objects={
            **default(scene).objects,
            'white_box': OnFace(on=('table_tall', 'top'), xy_yaw=[-0.1, 0.0, 0.0]),
        },
    )


INITIAL_CONDITIONS = {'default': default, 'goal': goal}
