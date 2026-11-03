from dataclasses import replace
import numpy as np
from faro.configs.scene import OnFace, Pose, RobotConfiguration, Standing, StateSpec


rob_x_disp = 0.85 + 0.0458282865
rob_y_disp = -0.0706162381

def default(scene):
    return StateSpec(
        robots={"robot": Standing(on=('floor', 'surface'))},
        objects={
            'floor': Pose(),
            'table_short': OnFace(on=('floor', 'surface'), xy_yaw=[1.0, 0.0, 0.0]),
            'white_box': OnFace(on=('floor', 'surface'), xy_yaw=[0.3, 0.1, 0.0]),
        },
    )


def goal(scene):
    initial = default(scene)
    return replace(
        initial,
        robots={"robot": RobotConfiguration(
            q=np.array(
                  [
                      0.85, 0.0, 1.504679, 9.56344919e-07, 0.000144175707,
                      -0.00687136433, 0.999976382, -0.150097495, 6.67267369e-06,
                      -3.08468997e-06, 0.449915432, -0.300106179, -7.5316763e-06,
                      -0.150096053, 1.49453144e-06, -1.93104916e-06, 0.449920925,
                      -0.300113116, -1.03790895e-06, -1.68112672e-05, 1.60643144e-05,
                      0.000164579921, -0.0776969428, 0.55788514, -0.287421849,
                      -0.301462528, -0.346188216, -0.304769552, -1.13062679,
                      -0.0776491679, -0.557925512, 0.287470671, -0.301473953,
                      0.34610661, -0.304858059, 1.13062097
                  ],
              ),
        )},
        objects={
            **initial.objects,
            'white_box': Pose(
                q=np.array(
                      [
                          0.214249139 + rob_x_disp, 0.0670483549 + rob_y_disp, 1.606902,
                          -3.64026332e-05, -0.000330036577, -0.00687214872, 0.999976331
                      ],
                  ),
            ),
        },
        contacts={
            ('ee', 'robot', 'left_ee'): (scene.face_id('white_box', 'left'), 1),
            ('ee', 'robot', 'right_ee'): (scene.face_id('white_box', 'right'), 1),
            ('ee', 'robot', 'left_sole'): (scene.face_id('table_short', 'top'), 1),
            ('ee', 'robot', 'right_sole'): (scene.face_id('table_short', 'top'), 1),
            ('obj', 'white_box'): (),
        },
    )


INITIAL_CONDITIONS = {'default': default, 'goal': goal}
