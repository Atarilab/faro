from faro.tasks.ik_task import IkTask
from faro.solvers.acados import AcadosStaticBuildConfig
from scenes.common.registry import make_scene_with_state
from faro.cost.ik_state_ctrl_cost import make_ocp_weights

if __name__ == "__main__":

    scene, resolved = make_scene_with_state("faro.single_box_ascent", "default")

    task = IkTask(scene=scene, resolved_state=resolved)

    code_gen_path      = "solvers/c_generated_code"
    solver_config_path = "solvers"

    task.build_solver(
            build_cfg=AcadosStaticBuildConfig(
                        name=scene.name.replace('.', '_'),
                        solver_config_path=solver_config_path,
                        code_gen_path=code_gen_path,
                        force_recompile=False,
                        qp_max_iter=100,
                        max_nlp_iters=500,
                        tol=1e-2,
                    ),
        )

    mode_interface_dict = {
      ('ee', 'robot', 'left_ee'):    (9,1),
      ('ee', 'robot', 'left_sole'):  (2, 1),
      ('ee', 'robot', 'right_ee'):   (11,1),
      ('ee', 'robot', 'right_sole'): (2, 1),
      ('obj', 'white_box'):  (),
    }

    task.update_contact_mode(mode_interface_dict)
    
    task.set_fixed_params(weights=make_ocp_weights(scene.robots))
    
    task.initialize_solver()

    task.solve(nlp_iters=100, print_stats=True, sqp_iters_per_collision=5)

    results = task.get_results()

    task.visualize_trajectory(results["robot_xs"], results["obj_xs"],
                              transparent_robot=False,
                              robot_alpha=0.5)
