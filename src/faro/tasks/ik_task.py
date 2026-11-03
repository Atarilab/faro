from faro.configs.identifiers import robot_name

import numpy as np

from faro.configs.mujoco import build_mujoco
from faro.configs.scene import Scene, ResolvedSceneState
from faro.ocp.formulation.ik import IkFormulation
from faro.solvers.acados import Acados, AcadosStaticBuildConfig
from faro.tasks import _common, _contacts, _diagnostics
from faro.tasks._timing import TimedTask, timed_solve
from faro.utils.contact_mode import ContactMode


class IkTask(TimedTask):
    """One free configuration satisfying a contact mode, with no transition."""

    def __init__(self, scene: Scene, resolved_state: ResolvedSceneState):

        _common.initialize_scene(self, scene, resolved_state)

        self.formulation = IkFormulation(
            scene=scene,
            collision_index=self.collision_index,
            robot_collision_models=self.robot_collision_models,
            verbose=False
        )

        self.solver = None

    def build_solver(self, build_cfg: AcadosStaticBuildConfig | None = None):
        build_cfg = build_cfg or AcadosStaticBuildConfig()
        if not isinstance(build_cfg, AcadosStaticBuildConfig):
            raise TypeError("IK requires AcadosStaticBuildConfig.")

        self.collision_mgr.allocate(0)
        self.collision_mgr.set_fixed_env_qs(self.resolved)

        ocp = self.formulation.build_ocp_description()

        self.solver = Acados(ocp, build_cfg=build_cfg, dyn="implicit")

    def update_contact_mode(self, contact_mode: ContactMode):
        """Set one already-assembled contact mode."""
        interface_one_hots = _contacts.ik_interface_params(self.scene, contact_mode)
        self._validate_interface_containment(interface_one_hots)
        self.interface_ohs = interface_one_hots
        self.solver.update_params(trajectory_params=interface_one_hots)

    def set_fixed_params(self, weights=None):
        """Bind reference configurations, object sizes and regularization weights."""
        const_params = _common.ik_constant_params(self.scene, self.resolved)

        const_params.update(_common.flatten_cost_weights(
            self.scene, weights, self.formulation.params_syms))

        self.solver.update_params(constant_params=const_params)

    def apply_params(self):
        self.solver.apply_params()

    def resolve_collisions(self, robot_states, obj_states):
        self.collision_mgr.set_robot_trajectories({
            rid: robot_states[rid][:, :cfg.nq]
            for rid, cfg in self.robot_cfgs.items()
        })

        self.collision_mgr.set_movable_poses(
            {name: obj_states[name][:, :7] for name in self.env_objs}
        )

        self.collision_mgr.resolve_all()
        self.solver.update_params(trajectory_params=\
                            self.collision_mgr.buffers.coll_pair_vals)

    def get_top_n_collisions(self, robot_states, obj_states, n=10, do_print=False):
        self.resolve_collisions(robot_states, obj_states)
        return _diagnostics.top_collisions(
            self.collision_mgr.buffers.coll_pair_dists, n, do_print)

    def set_initial_coll_params(self):
        self.resolve_collisions(
            {rid: state.q[None, :] for rid, state in self.resolved.robots.items()},
            {name: self.resolved.env_objects[name].q[None, :] for name in self.env_objs},
        )

    def reset_initial_state(self):
        """Seed the sole state without imposing fixed-state bounds."""
        x0_dict = {
            **{robot_name(rid, "dq_rob"): np.hstack((np.zeros(6), state.q[7:]))
               for rid, state in self.resolved.robots.items()},
            **{name: np.zeros(self.formulation.obj_n_v) for name in self.env_objs}
        }

        self.solver.solver.set_state_constant(x0_dict)

    def initialize_solver(self):
        self.reset_initial_state()
        self.set_initial_coll_params()
        self.apply_params()

    @timed_solve
    def solve(self, nlp_iters=50, print_stats=False,
              sqp_iters_per_collision: int = 5):
        """Run outer collision updates with the requested SQP count per solve."""
        _, full_stats_dict, _ = self.solver.solve(
            nlp_iters=sqp_iters_per_collision, first_iteration=True)

        if print_stats:
            self.print_iterate_stats(full_stats_dict, first_iteration=True)

        for _ in range(nlp_iters):
            results = self.get_results()

            self.resolve_collisions(results["robot_xs"], results["obj_xs"])
            self.apply_params()
            status, stats_dict, _ = self.solver.solve(
                nlp_iters=sqp_iters_per_collision, first_iteration=False)

            self.merge_stats_into_full(full_stats_dict, stats_dict)

            if print_stats:
                self.print_iterate_stats(full_stats_dict)

            if status == 0:
                if print_stats:
                    print("Solver status = 0: SUCCESS — optimal solution found.")
                return 0

        return 1

    def merge_stats_into_full(self, full_stats_dict, stats_dict):
        _diagnostics.merge_stats_into_full(full_stats_dict, stats_dict)

    def print_iterate_stats(self, stats_dict, first_iteration: bool = False, debug_qp: bool = False):
        self._iter_stats_rows_printed = _diagnostics.print_iterate_stats(
            stats_dict, getattr(self, "_iter_stats_rows_printed", 0),
            first_iteration, debug_qp)

    def get_results(self):
        """Return one configuration row per robot/object and no control samples."""
        return self.solver.get_results(self.robot_cfgs, self.env_objs,
                                       self.resolved, rob_state_name="dq_rob")

    def _validate_interface_containment(self, interface_one_hots):
        _contacts.validate_interface_containment(
            self.scene, self.formulation.object_features,
            interface_one_hots, interface_one_hots, range(1), use_object_sizes=True)

    def visualize_trajectory(self, rob_qs, obj_qs,
                             transparent_robot = False,
                             robot_alpha = 0.1):
        _common.visualize_ik_trajectory(
            self.scene, build_mujoco(self.scene, self.resolved)[0], rob_qs, obj_qs,
            next(iter(rob_qs.values())).shape[0], transparent_robot, robot_alpha)

    def get_performance_metrics(self):
        return _diagnostics.performance_metrics(self.solver, include_details=False)

    def get_collision_metrics(self, rob_qs, obj_qs, penetration_threshold: float = 1e-3):
        num_steps = next(iter(rob_qs.values())).shape[0]
        distances = {}
        if num_steps:
            self.get_top_n_collisions(rob_qs, obj_qs, n=1, do_print=False)
            distances = self.collision_mgr.buffers.coll_pair_dists
        return _diagnostics.collision_metrics(distances, num_steps, penetration_threshold)

    def print_max_k_constr_violations(self, perf_metrics, k=20):
        _diagnostics.print_max_k_constr_violations(perf_metrics, k, include_details=False)
