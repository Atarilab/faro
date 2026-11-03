from faro.tasks.kso_task import KsoTask
from faro.solvers.acados import AcadosBuildConfig
from scenes.common.registry import make_scene_with_state
from faro.cost.kso_state_ctrl_cost import make_ocp_weights

if __name__ == "__main__":

    scene, resolved = make_scene_with_state("faro.box_throwing_small", "default")

    task = KsoTask(
        scene=scene, 
        resolved_state=resolved,
    )

    num_modes = 4

    code_gen_path      = "solvers/c_generated_code"
    solver_config_path = "solvers"

    task.build_solver(
            num_modes=num_modes,
            build_cfg=AcadosBuildConfig(
                        name=f"{scene.name.replace('.', '_')}_{num_modes}",
                        solver_config_path=solver_config_path,
                        code_gen_path=code_gen_path,
                        force_recompile=False,
                        qp_max_iter=100,
                        max_nlp_iters=500,
                        tol=1e-2,
                        free_init_state=False
                    )
        )

    contact_mode_seq = {
        ('ee', 'robot', 'left_sole'): [(1, 1), (1, 1), (1, 1), (1, 1)],
        ('ee', 'robot', 'right_sole'): [(1, 1), (1, 1), (1, 1), (1, 1)],
        ('ee', 'robot', 'left_ee'): [(), (9,1), (), (),],
        ('ee', 'robot', 'right_ee'): [(), (11, 1), (), ()],
        ('obj', 'white_box'): [(6, 1, 1), (), (), (5, 2, 1)],
    }

    task.update_mode_oh_via_tup(contact_mode_seq)
    
    task.set_fixed_params(weights=make_ocp_weights(scene.robots))

    task.initialize_solver()

    task.solve(nlp_iters=100, print_stats=True,
               sqp_iters_per_collision=5, res_ineq_tol=1e-2)
    
    results = task.get_results()
    
    task.visualize_trajectory(results["robot_xs"], results["obj_xs"],
                              transparent_robot=False,
                              robot_alpha=0.5)
