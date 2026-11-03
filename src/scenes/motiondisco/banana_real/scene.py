from faro.configs.objects import Plane, FixedBox, MovableBox
from faro.configs.robot import make_g1_robot
from faro.configs.scene import Scene

SCENE_NAME = 'motiondisco.banana_real'


def make_scene():
    return Scene(
        name=SCENE_NAME,
        robots={"robot": make_g1_robot()},
        objects={
            'floor': Plane(half_extents=[5, 3.5]),
            'table_short': FixedBox(half_extents=[0.35, 0.7, 0.365], color=[0.5, 0.5, 0.5, 1.0]),
            'banana': FixedBox(half_extents=[0.03, 0.03, 0.075], color=[1.0, 1.0, 0.0, 1.0]),
            'crate': MovableBox(half_extents=[0.15, 0.2, 0.15], mass=2.078, color=[0.58, 0.29, 0.0, 1.0]),
        },
    )


if __name__ == "__main__":
    from scenes.common.viz import view_initial_condition
    view_initial_condition(
        SCENE_NAME, "default",
        compare_to="goal",
        terminal_goal_name="robot_base_while_grasping",
        show_all_comparison=True,
    )
