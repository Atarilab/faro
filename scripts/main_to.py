from faro.tasks.to_task import ToTask
from faro.solvers.moto import MotoBuildConfig
from faro.cost.state_ctrl_cost import make_ocp_weights
from scenes.common.registry import make_scene_with_state

if __name__ == "__main__":
    scene, resolved = make_scene_with_state("faro.box_throwing_tall", "default")

    task = ToTask(scene=scene, resolved_state=resolved)

    num_modes = 5
    nodes_per_mode = 20
    time_per_mode = 1.0
    num_nodes = num_modes*nodes_per_mode

    task.build_solver(
            cfg=MotoBuildConfig(
                num_modes=num_modes,
                nodes_per_mode=nodes_per_mode,
                time_per_mode=time_per_mode,
                tol=1e-2,
            ),
        )

    task.print_scene_info()

    # Named faces are authoring helpers; tasks receive the resulting global IDs.
    print(scene.faces.by_name)
    print(scene.faces.by_id)

    # Interface schema (per mode):
    #   EEs:        () | (env_id: int, contact_type: int)
    #   Objects:    () | (self_id: int, env_id: int, contact_type: int)
    # Notes:
    #   - () means free (no contact)
    #   - contact_type: 1=unilateral, 2=sliding, 3=bilateral
    #     (unilateral and sliding are supported; bilateral is reserved)

    interface_dict = {
        ('ee', 'robot', 'left_sole'):  [(1, 1), (), (1, 1), (), (1, 1),],
        ('ee', 'robot', 'right_sole'): [(1, 1), (1, 1), (), (1, 1), (),],
        ('ee', 'robot', 'left_ee'):    [(), (9,1), (), (), (),],
        ('ee', 'robot', 'right_ee'):   [(), (11, 1), (), (11,1), (),],
        ('obj', 'white_box'):  [(6, 1, 1), (), (), (), (6, 2, 1),],
    }

    task.initialize_solver(interface_dict)

    task.update_mode_oh_via_tup(interface_dict)

    task.set_fixed_params(n_nodes= num_nodes,
                          weights=make_ocp_weights(scene.robots))

    task.solve(max_nlp_iters=100, verbose=True)

    results = task.get_results(include_torque=True, coll=True)

    task.visualize_trajectory(results["robot_xs"], results["obj_xs"],
                              results["rob_Ws"], results["obj_Ws"],
                              results["t_scale"],
                              transparent_robot=False,
                              robot_alpha=0.5)

    task.print_top_n_constr_vio(n=20)
