from faro.configs.scene import OnFace, Pose, Standing, StateSpec


def default(scene):
    return StateSpec(
        robots={"robot": Standing(on=('floor', 'surface'))},
        objects={
            'floor': Pose(),
            'table_short': OnFace(on=('floor', 'surface'), xy_yaw=[0.7, 0.0, 0.0]),
            'green_box': OnFace(on=('table_short', 'top'), xy_yaw=[-0.1, 0.35, 0.0]),
            'white_box': OnFace(on=('table_short', 'top'), xy_yaw=[-0.1, 0.035, 0.0]),
            'yellow_box': OnFace(on=('table_short', 'top'), xy_yaw=[-0.1, -0.35, 0.0]),
        },
    )


INITIAL_CONDITIONS = {'default': default}
