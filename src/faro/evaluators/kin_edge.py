from faro.tasks.ik_task import IkTask
from faro.solvers.acados import AcadosStaticBuildConfig
from faro.cost.ik_state_ctrl_cost import make_ocp_weights
from faro.evaluators._results import OracleEvaluationDetails
from faro.utils.contact_mode import contact_mode_union

from faro.graph.modespace import ModeSpace
from faro.configs.scene import Scene, ResolvedSceneState

def create_edge_oracle(
        ms: ModeSpace,
        scene: Scene,
        resolved_state: ResolvedSceneState,
        *,
        return_details: bool = False,
    ):
    """Build an IK transition-union oracle, optionally returning solve timing."""

    task = IkTask(scene=scene, resolved_state=resolved_state)

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

    def oracle(a: tuple[int, ...], b: tuple[int, ...]) -> bool | OracleEvaluationDetails:
        try:
            contact_mode = contact_mode_union(ms.decode_contact_mode(a), ms.decode_contact_mode(b))
            task.update_contact_mode(contact_mode)
        except ValueError:
            return OracleEvaluationDetails(False, 0.0) if return_details else False

        task.set_fixed_params(weights=make_ocp_weights(scene.robots))

        task.initialize_solver()

        status = task.solve(nlp_iters=100, print_stats=False)
        metrics = task.get_performance_metrics()

        feasible = bool(status == 0 or metrics["max_violation"] < 1e-2)
        return OracleEvaluationDetails(feasible, task.get_solve_time()) if return_details else feasible
    
    return oracle
