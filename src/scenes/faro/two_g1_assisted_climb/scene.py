"""Two G1 robots and a shared box on the floor."""
from faro.configs.objects import Plane, FixedBox
from faro.configs.robot import make_g1_robot
from faro.configs.scene import Scene

SCENE_NAME = "faro.two_g1_assisted_climb"


def make_scene():
    return Scene(
        name=SCENE_NAME,
        robots={
            "g1a": make_g1_robot(),
            "g1b": make_g1_robot(),
        },
        objects={
            "floor": Plane(),
            'table_tall': FixedBox(half_extents=[0.75, 0.395, 0.75], color=[0.5, 0.5, 0.5, 1.0]),
        },
        include_robot_ees=True,
    )


if __name__ == "__main__":
    from scenes.common.viz import view_initial_condition
    view_initial_condition(SCENE_NAME, "default")
