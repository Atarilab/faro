"""Two robots transfer a box from the floor to a raised table."""
from faro.tasks.to_task import ToTask
from faro.solvers.moto import MotoBuildConfig
from faro.cost.state_ctrl_cost import make_ocp_weights
from scenes.common.registry import make_scene_with_state


def make_contact_sequence(scene):
    floor = (scene.face_id("floor", "surface"), 1)
    platform = (scene.face_id("platform", "top"), 1)
    a_left = (scene.face_id("white_box", "left"), 1)
    a_right = (scene.face_id("white_box", "right"), 1)
    # G1b initially faces -Y, so its left/right hands use the +X/-X faces.
    b_left = (scene.face_id("white_box", "front"), 1)
    b_right = (scene.face_id("white_box", "back"), 1)
    box_floor = (6, floor[0], 1)  # Box bottom on the floor.
    table_top = (6, scene.face_id("table", "top"), 1)

    return {
        ("ee", "g1a", "left_sole"):  [floor,     floor,     floor,    (),       floor,    floor,    floor,    floor,    floor],
        ("ee", "g1a", "right_sole"): [floor,     floor,     floor,    floor,    (),       floor,    floor,    floor,    floor],
        ("ee", "g1a", "left_ee"):    [(),        a_left,    a_left,   a_left,   a_left,   (),       (),       (),       ()],
        ("ee", "g1a", "right_ee"):   [(),        a_right,   a_right,  a_right,  a_right,  (),       (),       (),       ()],
        ("ee", "g1b", "left_sole"):  [platform,  platform,  (),       platform, (),       platform, (),       platform, platform],
        ("ee", "g1b", "right_sole"): [platform,  (),        platform, (),       platform, (),       platform, (),       platform],
        ("ee", "g1b", "left_ee"):    [(),        (),        (),       (),       (),       b_left,   b_left,   b_left,   b_left],
        ("ee", "g1b", "right_ee"):   [(),        (),        (),       (),       (),       b_right,  b_right,  b_right,  b_right],
        ("obj", "white_box"):        [box_floor, (),        (),       (),       (),       (),       (),       (),       table_top],
    }


if __name__ == "__main__":
    scene, resolved = make_scene_with_state("faro.two_g1_pass_up", "default")
    task = ToTask(scene=scene, resolved_state=resolved)
    interface_dict = make_contact_sequence(scene)

    num_modes = len(next(iter(interface_dict.values())))
    nodes_per_mode = 20
    time_per_mode = 1.0
    num_nodes = num_modes * nodes_per_mode

    task.build_solver(
        cfg=MotoBuildConfig(
            name="two_robots",
            num_modes=num_modes,
            nodes_per_mode=nodes_per_mode,
            time_per_mode=time_per_mode,
            feasibility_tol=5e-2,
            tol=1e-2,
        ),
    )

    task.initialize_solver(interface_dict)
    task.update_mode_oh_via_tup(interface_dict)
    task.set_fixed_params(n_nodes=num_nodes, weights=make_ocp_weights(scene.robots))
    task.solve(max_nlp_iters=200, verbose=True)

    results = task.get_results(include_torque=True, coll=True)
    task.visualize_trajectory(
        results["robot_xs"], results["obj_xs"],
        results["rob_Ws"], results["obj_Ws"], results["t_scale"],
        transparent_robot=False,
        show_forces=False,
        robot_alpha=0.5,
    )
