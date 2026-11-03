from faro.tasks.to_task import ToTask
from faro.solvers.moto import MotoBuildConfig
from faro.cost.state_ctrl_cost import make_ocp_weights
from scenes.common.registry import make_scene_with_state

if __name__ == "__main__":

    scene, resolved = make_scene_with_state("faro.box_throwing", "default")

    task = ToTask(scene=scene, resolved_state=resolved)

    num_modes = 10
    nodes_per_mode = 20
    time_per_mode = 0.8
    num_nodes = num_modes*nodes_per_mode

    task.build_solver(
            cfg=MotoBuildConfig(
                num_modes=num_modes,
                nodes_per_mode=nodes_per_mode,
                time_per_mode=time_per_mode,
                tol=1e-2,
            )
        )

    cms = {
        ('ee', 'robot', 'left_sole'):  [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1),],
        ('ee', 'robot', 'right_sole'): [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1),],
        ('ee', 'robot', 'left_ee'):    [(), (3,1),   (3,1), (), (),     (), (3,1), (), (),    (3,1),],
        ('ee', 'robot', 'right_ee'):   [(), (5, 1), (),    (), (5,1), (), (),    (), (5,1), (5,1),],
        ('obj', 'white_box'):  [(6, 1, 1), (), (), (), (), (), (), (), (), (),],
    }

    task.initialize_solver(cms)

    task.update_mode_oh_via_tup(cms)

    task.set_fixed_params(n_nodes= num_modes * nodes_per_mode,
                          weights=make_ocp_weights(scene.robots))

    task.solve(max_nlp_iters=100, verbose=True)

    results = task.get_results(include_torque=True, coll=True)

    task.visualize_trajectory(results["robot_xs"], results["obj_xs"],
                              results["rob_Ws"], results["obj_Ws"],
                              results["t_scale"],
                              transparent_robot=False,
                              robot_alpha=0.5)
