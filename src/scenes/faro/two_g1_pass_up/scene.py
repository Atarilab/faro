"""Two G1 robots and a shared box on the floor."""
from faro.collision.collision_rules import CollisionRules
from faro.configs.objects import Plane, FixedBox, MovableBox
from faro.configs.robot import make_g1_robot
from faro.configs.scene import Scene

SCENE_NAME = "faro.two_g1_pass_up"


def make_scene():
    return Scene(
        name=SCENE_NAME,
        robots={
            "g1a": make_g1_robot(),
            "g1b": make_g1_robot(),
        },
        objects={
            "floor": Plane(),
            'platform': FixedBox(
                half_extents=[1.5, 1.0, 0.6],
                color=[0.5, 0.5, 0.5, 1.0]
            ),
            'table': FixedBox(
                half_extents=[0.5, 0.5, 0.4],
                color=[0.5, 0.5, 0.5, 1.0]
            ),
            "white_box": MovableBox(
                half_extents=[.1325, .1275, .135],
                mass=1.18,
                color=[1., 1., 1., 1.],
            ),
        },
        include_robot_ees=False,
        collision_rules=CollisionRules(enable_robot_robot=False),
    )


if __name__ == "__main__":
    from scenes.common.viz import view_initial_condition
    view_initial_condition(SCENE_NAME, "default")
