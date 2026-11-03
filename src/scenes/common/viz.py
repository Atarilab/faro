"""View one resolved scene state or overlay two states."""
import time

import mujoco
import mujoco.viewer

from faro.configs.mujoco import build_mujoco, build_mujoco_comparison
from scenes.common.registry import (
    get_initial_condition, get_terminal_goal, make_scene_with_state,
)


def view(scene, state, comparison_state=None, *, terminal_goal=None,
         show_all_comparison: bool = False,
         comparison_alpha: float = 0.25, sleep: float = 0.1, hide_sites: bool = True):
    """Show explicit poses; an optional goal selects comparison bodies only.

    Without a goal, every comparison robot and movable object is shown. An
    empty goal selects none. Set show_all_comparison=True to override goal
    selection. The first state retains its normal appearance; comparison_alpha
    scales the second state's opacity. Goal coordinates and solver tuning are
    ignored.
    """
    if comparison_state is None:
        if terminal_goal is not None:
            raise ValueError("terminal_goal requires a comparison_state.")
        model, qpos = build_mujoco(scene, state)
    else:
        select_all = show_all_comparison or terminal_goal is None
        model, qpos = build_mujoco_comparison(
            scene, state, comparison_state,
            robots=None if select_all else terminal_goal.robots,
            objects=None if select_all else terminal_goal.objects,
            comparison_alpha=comparison_alpha,
        )
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        if hide_sites:
            viewer.opt.sitegroup[0] = 0
        while viewer.is_running():
            viewer.sync()
            time.sleep(sleep)


def view_initial_condition(scene_name: str, ic_name: str = "default", *,
                           compare_to: str | None = None,
                           terminal_goal_name: str | None = None,
                           show_all_comparison: bool = False,
                           comparison_alpha: float = 0.25, sleep: float = 0.1):
    """Overlay named presets; show_all_comparison overrides goal selection."""
    if terminal_goal_name is not None and compare_to is None:
        raise ValueError("terminal_goal_name requires compare_to.")
    scene, state = make_scene_with_state(scene_name, ic_name)
    comparison = None if compare_to is None else scene.resolve(
        get_initial_condition(scene_name, compare_to, scene=scene)
    )
    goal = None if terminal_goal_name is None else get_terminal_goal(
        scene_name, terminal_goal_name, scene=scene, resolved_state=state,
    )
    view(scene, state, comparison, terminal_goal=goal,
         show_all_comparison=show_all_comparison,
         comparison_alpha=comparison_alpha, sleep=sleep)
