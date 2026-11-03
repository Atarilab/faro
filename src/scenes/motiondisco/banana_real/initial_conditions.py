from dataclasses import replace
import numpy as np
from faro.configs.scene import OnFace, Pose, RobotConfiguration, Standing, StateSpec


def default(scene):
    return StateSpec(
        robots={"robot": Standing(on=('floor', 'surface'))},
        objects={
            'floor': Pose(),
            'table_short': OnFace(on=('floor', 'surface'), xy_yaw=[1.0, 0.0, 0.0]),
            'banana': Pose(q=[1.0, 0.0, 2.5, 0.0, 0.0, 0.0, 1.0]),
            'crate': OnFace(on=('floor', 'surface'), xy_yaw=[0.4, 0.0, 0.0]),
        },
    )


def goal(scene):
    initial = default(scene)
    return replace(
        initial,
        robots={"robot": RobotConfiguration(
            q=np.array(
                  [
                      0.988414145, -1.87900172e-06, 1.82100482, 3.34835795e-06,
                      0.00145798744, -3.06944713e-06, 0.999998937, -0.0561523016,
                      -0.00342339255, -0.000991617568, 0.129954787, -0.0767236278,
                      0.00311705163, -0.0561504756, 0.0034030613, 0.000997947249,
                      0.129951106, -0.0767217405, -0.00310955522, 4.87437692e-06,
                      2.70988435e-06, -0.34773539, -2.06827589, 0.348605386,
                      -0.0841331186, 0.609155351, -0.205786693, 0.209792651,
                      -1.25718394, -2.0682905, -0.348587905, 0.0841326235, 0.609170796,
                      0.205781331, 0.209790688, 1.25717731
                  ],
              ),
        )},
        objects={
            **initial.objects,
            'crate': Pose(
                q=np.array(
                      [
                          1.02179561, -7.71760739e-05, 0.880341643, -7.85299875e-08,
                          -4.83093434e-06, 1.41018461e-05, 1.0
                      ],
                  ),
            ),
        },
        contacts={
            ('ee', 'robot', 'left_ee'): (scene.face_id('banana', 'left'), 1),
            ('ee', 'robot', 'right_ee'): (scene.face_id('banana', 'right'), 1),
            ('ee', 'robot', 'left_sole'): (scene.face_id('crate', 'top'), 1),
            ('ee', 'robot', 'right_sole'): (scene.face_id('crate', 'top'), 1),
            ('obj', 'crate'): (6, scene.face_id('table_short', 'top'), 1),
        },
    )


INITIAL_CONDITIONS = {'default': default, 'goal': goal}
