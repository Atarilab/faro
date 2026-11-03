"""Manually edit the contact sequence and solver settings below."""
from faro.tasks.to_task import ToTask
from faro.solvers.moto import MotoBuildConfig
from faro.cost.state_ctrl_cost import make_ocp_weights
from scenes.common.registry import make_scene_with_state


if __name__ == "__main__":
    scene, resolved = make_scene_with_state("faro.two_g1_throw", "default")
    task = ToTask(scene=scene, resolved_state=resolved)

    interface_dict = {
        ("ee", "g1a", "left_sole"):  [(1, 1),    (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
        ("ee", "g1a", "right_sole"): [(1, 1),    (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
        ("ee", "g1a", "left_ee"):    [(),        (3, 1), (),     (),     (),     (3, 1)],
        ("ee", "g1a", "right_ee"):   [(),        (5, 1), (),     (),     (),     (5, 1)],
        ("ee", "g1b", "left_sole"):  [(1, 1),    (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
        ("ee", "g1b", "right_sole"): [(1, 1),    (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
        ("ee", "g1b", "left_ee"):    [(),        (),     (),     (5, 1), (),     ()],
        ("ee", "g1b", "right_ee"):   [(),        (),     (),     (3, 1), (),     ()],
        ("obj", "white_box"):        [(6, 1, 1), (),     (),     (),     (),     ()],
    }

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
    task.solve(max_nlp_iters=100, verbose=True)

    results = task.get_results(include_torque=True, coll=True)
    task.visualize_trajectory(
        results["robot_xs"], results["obj_xs"],
        results["rob_Ws"], results["obj_Ws"], results["t_scale"],
        transparent_robot=False,
        robot_alpha=0.5,
    )