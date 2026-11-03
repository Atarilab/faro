from faro.configs.objects import Plane, FixedBox, MovableBox
from faro.configs.robot import make_g1_robot
from faro.configs.scene import Scene

SCENE_NAME = 'faro.box_throwing'


def make_scene():
    return Scene(
        name=SCENE_NAME,
        robots={"robot": make_g1_robot()},
        objects={
            'floor': Plane(),
            'white_box': MovableBox(
                half_extents=[0.1325, 0.1275, 0.135],
                mass=1.18,
                color=[1.0, 1.0, 1.0, 1.0],
            ),
        },
    )


if __name__ == "__main__":
    from scenes.common.viz import view_initial_condition
    view_initial_condition(SCENE_NAME, "default")
