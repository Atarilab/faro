"""Two G1 robots and a shared box on the floor."""
from faro.configs.objects import Plane, MovableBox
from faro.configs.robot import make_g1_robot
from faro.configs.scene import Scene

SCENE_NAME = "faro.two_g1_throw"


def make_scene():
    return Scene(
        name=SCENE_NAME,
        robots={
            "g1a": make_g1_robot(),
            "g1b": make_g1_robot(),
        },
        objects={
            "floor": Plane(),
            "white_box": MovableBox(
                half_extents=[.1325, .1275, .135],
                mass=1.18,
                color=[1., 1., 1., 1.],
            ),
        },
        include_robot_ees=True,
    )


if __name__ == "__main__":
    from scenes.common.viz import view_initial_condition
    view_initial_condition(SCENE_NAME, "default")
