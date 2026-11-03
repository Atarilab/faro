from faro.cost.state_ctrl_cost import make_ocp_weights
from faro.solvers.moto import MotoBuildConfig
from faro.tasks.to_task import ToTask
from scenes.common.registry import get_terminal_goal, make_scene_with_state


def main():
    scene, resolved = make_scene_with_state("faro.box_pushing", "default")

    goal = get_terminal_goal(
        scene.name, "box_at_goal", scene=scene, resolved_state=resolved,
    )
    task = ToTask(scene=scene, 
                  resolved_state=resolved,
                  terminal_goal=goal
                  )

    interface_dict = {
        ("ee", "robot", "left_sole"):  [(1, 1),    (),        (1, 1),    (),        (1, 1)],
        ("ee", "robot", "right_sole"): [(1, 1),    (1, 1),    (),        (1, 1),    (1, 1)],
        ("ee", "robot", "left_ee"):    [(),        (6, 1),    (6, 1),    (6, 1),    (6, 1)],
        ("ee", "robot", "right_ee"):   [(),        (6, 1),    (6, 1),    (6, 1),    (6, 1)],
        ("obj", "white_box"):          [(6, 1, 1), (6, 1, 1), (6, 1, 2), (6, 1, 2), (6, 1, 1)],
    }

    num_modes = len(next(iter(interface_dict.values())))
    nodes_per_mode = 20
    time_per_mode = 1.0
    num_nodes = num_modes * nodes_per_mode

    task.build_solver(
        cfg=MotoBuildConfig(
            num_modes=num_modes,
            nodes_per_mode=nodes_per_mode,
            time_per_mode=time_per_mode,
            tol=1e-2,
            collision_update_batch_size=5,
        ),
    )
    task.print_scene_info()
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


if __name__ == "__main__":
    main()
