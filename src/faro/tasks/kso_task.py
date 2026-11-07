from typing import Mapping, List, Union
from dataclasses import replace
from faro.configs.identifiers import robot_name, interface_name, InterfaceKey

import numpy as np

from faro.configs.mujoco import build_mujoco
from faro.configs.scene import Scene, ResolvedSceneState
from faro.ocp.formulation.kso import KsoFormulation
from faro.search.goal import TerminalGoal
from faro.solvers.acados import Acados, AcadosBuildConfig
from faro.tasks import _common, _contacts, _diagnostics
from faro.tasks._timing import TimedTask, timed_solve
from faro.utils.contact_type import TupleEE, TupleObj


class KsoTask(TimedTask):
    def __init__(self, scene: Scene, resolved_state: ResolvedSceneState,
                       terminal_goal: TerminalGoal | None = None):

        _common.initialize_scene(self, scene, resolved_state)

        self.terminal_goal = terminal_goal

        self.formulation = KsoFormulation(
            scene=scene,
            collision_index=self.collision_index,
            robot_collision_models=self.robot_collision_models,
            terminal_goal=terminal_goal,
            verbose=False
        )

        self.active_interfaces = scene.active_patches

        self.solver = None
        self.presolve_solver = None
        self.presolve_max_iters = 0

        self.models = self.formulation.robot_models
        self.datas = self.formulation.robot_datas
        self.has_terminal_goal = bool(self.formulation.goal_targets)

    def build_solver(self, num_modes, build_cfg=None):
        if isinstance(num_modes, bool) or not isinstance(num_modes, (int, np.integer)) or num_modes < 1:
            raise ValueError("num_modes must be a positive integer.")
        build_cfg = build_cfg or AcadosBuildConfig()
        if (isinstance(build_cfg.presolve_max_iters, bool)
                or not isinstance(build_cfg.presolve_max_iters, (int, np.integer))
                or build_cfg.presolve_max_iters < 0):
            raise ValueError("presolve_max_iters must be a nonnegative integer.")
        self.presolve_max_iters = int(build_cfg.presolve_max_iters)
        self.num_modes = int(num_modes)
        self.hard_initial = not build_cfg.free_init_state
        self.num_states = self.num_modes - 1 + int(self.hard_initial) + int(self.has_terminal_goal)
        if self.num_states == 0:
            raise ValueError("KSO requires at least one transition or an endpoint.")
        self.N = self.num_states  # N counts states; horizon counts intervals.
        self.horizon = self.num_states - 1
        build_cfg = replace(build_cfg, N_horizon=self.horizon)
        if not np.isfinite(build_cfg.t_total) or build_cfg.t_total <= 0:
            raise ValueError("t_total must be finite and positive.")
        self.T_total = build_cfg.t_total

        self.collision_mgr.allocate(self.horizon)
        self.collision_mgr.set_fixed_env_qs(self.resolved)

        ocp = self.formulation.build_ocp_description(self.N, hard_initial=self.hard_initial)

        self.solver = Acados(ocp, build_cfg=build_cfg, dyn="implicit")
        self.solver.solver.set_single_threaded()
        self.solver.solver.set_final_residual_evaluation(True)
        self.presolve_solver = None
        if self.presolve_max_iters:
            presolve_ocp = self.formulation.build_ocp_description(
                self.N, hard_initial=self.hard_initial, include_collisions=False)
            presolve_cfg = replace(build_cfg, name=build_cfg.name + '_presolve',
                                   max_nlp_iters=max(build_cfg.max_nlp_iters, self.presolve_max_iters))
            self.presolve_solver = Acados(presolve_ocp, build_cfg=presolve_cfg, dyn="implicit")
            self.presolve_solver.solver.set_single_threaded()
            self.presolve_solver.solver.set_final_residual_evaluation(True)

    def update_mode_oh_via_tup(self, tuple_dict:
                               Mapping[InterfaceKey, List[Union[TupleEE, TupleObj]]]):
        interface_one_hots = _contacts.kso_interface_params(
            self.scene, tuple_dict, self.num_modes,
            hard_initial=self.hard_initial, terminal_endpoint=self.has_terminal_goal)

        self._validate_interface_containment(interface_one_hots)
        self.interface_ohs = interface_one_hots
        self.solver.update_params(trajectory_params=interface_one_hots)

    def set_fixed_params(self, weights):
        const_params = _common.ik_constant_params(self.scene, self.resolved, self.horizon)
        const_params['total_time'] = self.T_total

        cost_params = _common.flatten_cost_weights(
            self.scene, weights, self.formulation.params_syms)

        const_params.update(cost_params)
        const_params.update(self.formulation.goal_parameters())
        self.solver.update_params(constant_params=const_params)

    def apply_params(self):
        self.solver.apply_params()

    def resolve_collisions(self, robot_states, obj_states):
        self.collision_mgr.set_robot_trajectories({
            rid: robot_states[rid][:, :model.nq] for rid, model in self.models.items()
        })

        self.collision_mgr.set_movable_poses(
            {name: obj_states[name][:,:7] for name in self.env_objs}
        )

        self.collision_mgr.resolve_all()
        self.solver.update_params(trajectory_params=\
                            self.collision_mgr.buffers.coll_pair_vals)

    def get_top_n_collisions(self, robot_states, obj_states, n=10, do_print=False):
        self.resolve_collisions(robot_states, obj_states)
        return _diagnostics.top_collisions(
            self.collision_mgr.buffers.coll_pair_dists, n, do_print)

    def set_initial_coll_params(self):
        self.collision_mgr.set_robot_trajectories({
            rid: np.tile(state.q, (self.N, 1)) for rid, state in self.resolved.robots.items()
        })

        self.collision_mgr.set_movable_poses(
            {name: np.tile(self.resolved.env_objects[name].q, (self.N, 1))
                for name in self.env_objs}
        )

        self.collision_mgr.resolve_all()

        self.solver.update_params(trajectory_params=\
                                  self.collision_mgr.buffers.coll_pair_vals)

    def reset_initial_state(self):
        """Resets x0 constraint (Robot + Objects)."""
        x0_dict = {
            **{robot_name(rid, "q_rob"): state.q.copy()
               for rid, state in self.resolved.robots.items()},
            **{name: self.resolved.env_objects[name].q.copy() for name in self.env_objs}
        }

        u0_dict = {
            **{robot_name(rid, "q_next_rob"): state.q.copy()
               for rid, state in self.resolved.robots.items()},
            **{name: self.resolved.env_objects[name].q.copy() for name in self.env_objs}
        }

        if self.hard_initial:
            self.solver.solver.set_initial_state(x0_dict)
        self.solver.solver.set_state_constant(x0_dict)
        self.solver.solver.set_input_constant(u0_dict)

    def initialize_solver(self):
        self.reset_initial_state()
        self.set_initial_coll_params()
        self.apply_params()

    def _prepare_presolve(self):
        """Use the latest full-solver guess, parameters, and fixed initial state."""
        source, target = self.solver, self.presolve_solver
        target.copy_primal_iterate_from(source)
        parameters = source.solver.get_parameter_buffer()
        target.update_params(trajectory_params={
            name: parameters[:, slice(*source.spec.block_slice('p', name))].T
            for name in source.ocp.p_dict
        })
        if self.hard_initial:
            target.solver.set_initial_state({
                name: source.spec._x0_val[slice(*source.spec.block_slice('x', name))].copy()
                for name in source.ocp.x_dict
            })
        target.apply_params()

    @timed_solve
    def solve(self, nlp_iters = 50, print_stats = False,
                    sqp_iters_per_collision = 5, res_ineq_tol = None):
        """Run a contact-only presolve, then outer collision updates and SQP batches.

        ``nlp_iters`` counts outer updates independently of
        ``sqp_iters_per_collision``. ``res_ineq_tol`` permits early feasibility
        acceptance only when both dynamics and constraint residuals meet it.
        The presolve has a separate budget, configured by ``presolve_max_iters``.
        """
        def feasibility_satisfied(stats_dict):
            if res_ineq_tol is None:
                return False
            # An acceptable contact residual alone does not ensure x_next == u.
            # Never fall back to an earlier finite value after a failed iterate.
            for field in ('res_eq', 'res_ineq'):
                values = stats_dict.get(field, [])
                try:
                    value = float(values[-1])
                except (IndexError, TypeError, ValueError):
                    return False
                if not np.isfinite(value) or value >= res_ineq_tol:
                    return False
            return True

        if self.presolve_solver is not None:
            self._prepare_presolve()
            status, stats, cost = self.presolve_solver.solve(
                nlp_iters=self.presolve_max_iters, first_iteration=True)
            if print_stats:
                print(f"KSO presolve (no collisions, at most {self.presolve_max_iters} SQP iterations).")
                self.print_iterate_stats(stats, first_iteration=True)
            # A finite MAXITER iterate is the expected outcome of a short warm-up.
            if status not in (0, 2) or not np.isfinite(cost):
                return 1
            if not all(np.isfinite(value).all()
                       for value in self.presolve_solver.solver.get_traj().values()):
                return 1
            self.solver.copy_primal_iterate_from(self.presolve_solver)
            results = self.get_results()
            self.resolve_collisions(results['robot_xs'], results['obj_xs'])
            self.apply_params()

        status, full_stats_dict, cost = self.solver.solve(
            nlp_iters=sqp_iters_per_collision, first_iteration=True)

        if print_stats:
            self.print_iterate_stats(full_stats_dict, first_iteration=True)

        if feasibility_satisfied(full_stats_dict):
            if print_stats:
                print(f"res_eq and res_ineq < {res_ineq_tol}: accepted constraint satisfaction.")
            return 0

        for i in range(nlp_iters):
            results = self.get_results()
            self.resolve_collisions(results['robot_xs'], results['obj_xs'])
            self.apply_params()
            status, stats_dict, cost = self.solver.solve(
                nlp_iters=sqp_iters_per_collision, first_iteration=False)
            self.merge_stats_into_full(full_stats_dict, stats_dict)
            if print_stats:
                self.print_iterate_stats(full_stats_dict)

            if status == 0:
                if print_stats:
                    print("Solver status = 0: SUCCESS — optimal solution found.")
                return 0

            if feasibility_satisfied(full_stats_dict):
                if print_stats:
                    print(f"res_eq and res_ineq < {res_ineq_tol}: accepted constraint satisfaction.")
                return 0

        return 1

    def merge_stats_into_full(self, full_stats_dict, stats_dict):
        _diagnostics.merge_stats_into_full(full_stats_dict, stats_dict)

    def print_iterate_stats(self, stats_dict, first_iteration: bool = False, debug_qp: bool = False):
        self._iter_stats_rows_printed = _diagnostics.print_iterate_stats(
            stats_dict, getattr(self, "_iter_stats_rows_printed", 0),
            first_iteration, debug_qp)

    def get_results(self,):
        results = self.solver.get_results(self.robot_cfgs, self.env_objs,
                                          self.resolved, rob_state_name="q_rob")

        return results

    def _validate_interface_containment(self, interface_one_hots):
        for side in ('', '_incoming'):
            contacts = {
                key: interface_one_hots[key.removesuffix('_oh') + side + '_oh']
                for key in _contacts.interface_param_keys(self.scene)
            }
            _contacts.validate_interface_containment(
                self.scene, self.formulation.object_features,
                contacts, contacts, range(self.num_states), use_object_sizes=True)

    def visualize_trajectory(self, rob_qs, obj_qs,
                             transparent_robot = False,
                             robot_alpha = 0.1):
        _common.visualize_ik_trajectory(
            self.scene, build_mujoco(self.scene, self.resolved)[0], rob_qs, obj_qs,
            next(iter(rob_qs.values())).shape[0], transparent_robot, robot_alpha)

    def get_performance_metrics(self):
        return _diagnostics.performance_metrics(self.solver, include_details=True)

    def get_collision_metrics(self, rob_qs, obj_qs, penetration_threshold: float = 1e-3):
        num_steps = next(iter(rob_qs.values())).shape[0]
        distances = {}
        if num_steps:
            self.get_top_n_collisions(rob_qs, obj_qs, n=1, do_print=False)
            distances = self.collision_mgr.buffers.coll_pair_dists
        return _diagnostics.collision_metrics(distances, num_steps, penetration_threshold)

    def print_max_k_constr_violations(self, perf_metrics, k=20):
        _diagnostics.print_max_k_constr_violations(perf_metrics, k, include_details=True)
