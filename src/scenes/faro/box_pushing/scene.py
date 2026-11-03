from faro.configs.objects import Plane, FixedBox, MovableBox
from faro.configs.robot import make_g1_robot
from faro.configs.scene import Scene

SCENE_NAME = 'faro.box_pushing'


def make_scene():
    return Scene(
        name=SCENE_NAME,
        robots={"robot": make_g1_robot()},
        objects={
            'floor': Plane(),
            'white_box': MovableBox(
                half_extents=[0.3, 0.3, 0.5],
                mass=10,
                color=[1.0, 1.0, 1.0, 1.0],
            ),
        },
    )


if __name__ == "__main__":
    from scenes.common.viz import view_initial_condition
    view_initial_condition(
        SCENE_NAME, "default",
        compare_to="goal",
        terminal_goal_name="box_at_goal",
    )
