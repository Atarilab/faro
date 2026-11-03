from faro.configs.scene import OnFace, Pose, Standing, StateSpec


def default(scene):
    return StateSpec(
        robots={"robot": Standing(on=('floor', 'surface'))},
        objects={
            'floor': Pose(),
            'white_box': OnFace(on=('floor', 'surface'), xy_yaw=[0.35, 0.0, 0.0]),
        },
    )


INITIAL_CONDITIONS = {'default': default, 'pick_place': default}
