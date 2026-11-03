from faro.ocp.contact_wrenches import robot_wrench_trajectory
from typing import Mapping, List, Union

import moto
import numpy as np
import pinocchio as pin

from faro.configs.mujoco import build_mujoco, pack_trajectory
from faro.configs.identifiers import InterfaceKey, robot_name, solver_name
from faro.configs.scene import Scene, ResolvedSceneState
from faro.ocp.formulation.to import ToFormulation
from faro.search.goal import TerminalGoal
from faro.solvers.moto import Moto
from faro.tasks import _common, _contacts, _diagnostics
from faro.tasks._timing import TimedTask, timed_solve
from faro.utils.contact_type import (
    encode_contact_schedule_from_tuples, interval_time_factors,
    TupleEE,
    TupleObj,
)
from kinevis.vis_tools import KinematicRobotVisualizer
from kinevis.vis_utils import plot_joint_dynamics, plot_timescale_traj


class ToTask(TimedTask):
    def __init__(self, scene: Scene, resolved_state: ResolvedSceneState,
                 terminal_goal: TerminalGoal | None = None):

        _common.initialize_scene(self, scene, resolved_state)
        self.terminal_goal = terminal_goal

        self.formulation = ToFormulation(
            scene=scene,
            collision_index=self.collision_index,
            robot_collision_models=self.robot_collision_models,
            terminal_goal=terminal_goal,
            verbose=False
        )

        self.active_interfaces = scene.active_patches

        self.solver = None

        self.models = {rid: robot.model for rid, robot in self.formulation.robots.items()}
        self.datas = {rid: robot.data for rid, robot in self.formulation.robots.items()}

    def print_scene_info(self):
        """Helper to pretty-print the scene definition."""
        self.scene.print_scene_info()

    def build_solver(self, cfg):
        self.num_modes = cfg.num_modes
        self.nodes_per_mode = cfg.nodes_per_mode
        self.N = int(cfg.num_modes * cfg.nodes_per_mode)
        self.dt = cfg.time_per_mode / cfg.nodes_per_mode
        self.dt_traj = np.full(self.N, self.dt)
        self.T_total = self.N * self.dt

        self.collision_mgr.allocate(self.N)
        self.collision_mgr.set_fixed_env_qs(self.resolved)

        self.moto_wrap = Moto(self.formulation, build_cfg=cfg)

        self.moto_wrap.build_ocp()

    def update_mode_oh_via_tup(self, tuple_dict:
                               Mapping[InterfaceKey, List[Union[TupleEE, TupleObj]]]):
        if hasattr(self, "_compiled_contact_types"):
            contact_types = {key: tuple(value[-1] if value else 0 for value in values)
                             for key, values in tuple_dict.items()}
            if contact_types != self._compiled_contact_types:
                raise ValueError("Contact types or FREE phases changed; call initialize_solver with the new schedule first.")
        traj_params = encode_contact_schedule_from_tuples(
            self.scene,
            tuple_dict,
            num_modes=self.num_modes,
            nodes_per_mode=self.nodes_per_mode,
            extend_contact_into_release=False,
        )

        interface_one_hots = _contacts.select_interface_params(
            self.scene, traj_params, include_switches=True)

        self.dt_traj[:] = self.dt * interval_time_factors(self.scene, interface_one_hots)[:self.N]

        self.interface_ohs = interface_one_hots
        self._validate_interface_containment(interface_one_hots)
        if self.sqp is not None:
            self.moto_wrap.update_params_traj(self.sqp, interface_one_hots)
        self.moto_wrap.update_params_traj(self.sqp_coll, interface_one_hots)

    def set_fixed_params(self, n_nodes=None, weights=None):
        """Sets constant parameters (physics, weights, bounds)."""
        const_params = {
            "total_time": self.T_total,
            "n_nodes": self.N if n_nodes is None else n_nodes,
        }

        for rid, robot in self.robot_cfgs.items():
            const_params[robot_name(rid, "x0")] = self.resolved.robots[rid].x
            const_params[robot_name(rid, "mass")] = robot.mass

        for name, config in self.fixed_objs.items():
            const_params[name + "_q0"] = self.resolved.fixed_env_objects[name].q
            const_params[name + "_size"] = np.array(config.half_extents)

        for name, config in self.env_objs.items():
            const_params[name + "_x0"] = self.resolved.env_objects[name].x
            const_params[name + "_mass"] = config.mass
            const_params[name + "_inertia"] = np.array(config.inertia)
            const_params[name + "_size"] = np.array(config.half_extents)

        const_params.update(_common.flatten_cost_weights(
            self.scene, weights, self.formulation.params))

        goal = self.terminal_goal
        if goal is not None and goal.to != "off":
            for kind, targets in (("robot", goal.robots), ("object", goal.objects)):
                for owner, target in targets.items():
                    const_params[solver_name(kind, owner, "goal_reference")] = target.value
                    if goal.to == "soft":
                        const_params[solver_name(kind, owner, "goal_weight")] = target.weights("to")

        if self.sqp is not None:
            self.moto_wrap.update_params_const(self.sqp, const_params)
        self.moto_wrap.update_params_const(self.sqp_coll, const_params)

    def _update_collision_params(self):
        # Interval k evaluates collision constraints at its endpoint x[k+1].
        # Contact schedules retain their separate interval-based indexing.
        endpoint_params = {
            name: values[:, 1:]
            for name, values in self.collision_mgr.buffers.coll_pair_vals.items()
        }
        self.moto_wrap.update_params_traj(self.sqp_coll, endpoint_params)

    def resolve_collisions(self, robot_states, obj_states):
        self.collision_mgr.set_robot_trajectories({
            rid: robot_states[rid][:, :model.nq] for rid, model in self.models.items()
        })

        self.collision_mgr.set_movable_poses(
            {name: obj_states[name][:,:7] for name in self.env_objs}
        )

        self.collision_mgr.resolve_all()

        self._update_collision_params()

    def set_initial_coll_params(self):
        self.collision_mgr.set_robot_trajectories({
            rid: np.tile(state.q, (self.N + 1, 1))
            for rid, state in self.resolved.robots.items()
        })

        self.collision_mgr.set_movable_poses(
            {name: np.tile(self.resolved.env_objects[name].q, (self.N+1, 1))
                for name in self.env_objs}
        )

        self.collision_mgr.resolve_all()

        self._update_collision_params()

    def reset_initial_state(self):
        """Initialize physical robot/object states, including their velocities."""
        x0_dict = {
            "robots": {rid: np.tile(state.x, (self.N + 1, 1))
                       for rid, state in self.resolved.robots.items()},
            "objects": {name: np.tile(state.x, (self.N + 1, 1))
                        for name, state in self.resolved.env_objects.items()},
        }
        u0_dict = {
            "robots": {rid: {
                "acceleration": np.zeros((self.N, robot.nv)),
                "wrenches": {patch: np.zeros((self.N, 6)) for patch in robot.cfg.patch_ees},
            } for rid, robot in self.formulation.robots.items()},
            "objects": {name: np.zeros((self.N, 12)) for name in self.env_objs},
            "t_scale": np.ones(self.N),
        }

        if self.sqp is not None:
            self.moto_wrap.set_warm_start(self.sqp, x0_dict, u0_dict)
        self.moto_wrap.set_warm_start(self.sqp_coll, x0_dict, u0_dict)

    def initialize_solver(self, interface_tups):
        self.sqp = (
            self.moto_wrap.build_sqp(interface_tups)
            if self.moto_wrap.build_cfg.presolve_max_iters > 0 else None
        )
        self.sqp_coll = self.moto_wrap.build_sqp(interface_tups, coll=True)
        self._compiled_contact_types = {
            key: tuple(value[-1] if value else 0 for value in values)
            for key, values in interface_tups.items()}
        self.reset_initial_state()
        self.set_initial_coll_params()

    def _has_nonfinite_result_component(self, value):
        if isinstance(value, Mapping):
            return any(self._has_nonfinite_result_component(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return any(self._has_nonfinite_result_component(v) for v in value)
        if isinstance(value, np.ndarray):
            if value.dtype == object:
                return any(self._has_nonfinite_result_component(v) for v in value.flat)
            return np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all()
        if np.isscalar(value):
            try:
                return not np.isfinite(value)
            except TypeError:
                return False
        return False

    def _has_nonfinite_results(self, results):
        return self._has_nonfinite_result_component(results)

    def _sqp_update_failed(self, res):
        if res is None:
            return True

        kkt_values = {
            "objective": getattr(
                getattr(res, "barrier_objective", None), "cost", 0.0
            ),
            "inf_prim_res": getattr(res, "inf_prim_res", 0.0),
            "inf_dual_res": getattr(res, "inf_dual_res", 0.0),
            "inf_comp_res": getattr(res, "inf_comp_res", 0.0),
            "inf_prim_step": getattr(
                getattr(res, "step", None), "inf_prim_step", 0.0
            ),
            "inf_dual_step": getattr(
                getattr(res, "step", None), "inf_dual_step", 0.0
            ),
        }
        if self._has_nonfinite_result_component(kkt_values):
            return True

        return getattr(res, "result", None) in (
            moto.sqp.iter_result_numerical_failure,
            moto.sqp.iter_result_infeasible_stationary,
            moto.sqp.iter_result_restoration_failed,
        )

    @timed_solve
    def solve(self, max_nlp_iters = 50, verbose=False):
        """Run up to ``max_nlp_iters`` outer collision-linearization steps.

        Each outer step runs at most ``collision_update_batch_size`` inner
        Moto SQP iterations before collision parameters are refreshed.
        The collision-free presolve uses ``presolve_max_iters``; zero skips it
        and preserves the collision solver's current initial guess. Success means
        the primal residual is below ``feasibility_tol``, not full KKT convergence.
        A finite restoration-budget result can continue in the next outer step.
        """
        max_nlp_iters = int(max_nlp_iters)
        if max_nlp_iters < 1:
            raise ValueError("max_nlp_iters must be at least 1")

        inner_batch_size = max(
            1, int(self.moto_wrap.build_cfg.collision_update_batch_size)
        )

        # The collision-free pass only initializes the trajectory.  It has no
        # collision linearization to refresh, so it is a single inner solve and
        # does not consume an outer NLP iteration.
        presolve_max_iters = self.moto_wrap.build_cfg.presolve_max_iters
        if presolve_max_iters > 0:
            res = self._moto_update(self._get_sqp(False), int(presolve_max_iters), verbose=verbose)
            if self._sqp_update_failed(res):
                return 2

            results = self.moto_wrap.get_results(self.sqp)
            if self._has_nonfinite_results(results):
                return 2

            self.moto_wrap.set_warm_start(
                self.sqp_coll, results["x0_dict"], results["u0_dict"]
            )
        else:
            results = self.moto_wrap.get_results(self.sqp_coll)
            if self._has_nonfinite_results(results):
                return 2

        # Match the collision linearization to the starting trajectory before
        # taking the first collision SQP step, including when presolve is skipped.
        self.resolve_collisions(results["robot_xs"], results["obj_xs"])

        inner_sqp_iters = 0
        for outer_iter in range(1, max_nlp_iters + 1):
            res = self._moto_update(self.sqp_coll, inner_batch_size, verbose=verbose)
            if self._sqp_update_failed(res):
                if verbose:
                    print("sqp failed")
                return 2

            inner_sqp_iters += int(getattr(res, "num_iter", 0))

            results = self.moto_wrap.get_results(self.sqp_coll)
            if self._has_nonfinite_results(results):
                if verbose:
                    print("nonfinite results")
                return 2

            if (
                res.result in (
                    moto.sqp.iter_result_success,
                    moto.sqp.iter_result_exceed_max_iter,
                )
                and res.inf_prim_res < self.moto_wrap.build_cfg.feasibility_tol
            ):
                print(
                    f"Feasibility reached after {outer_iter} outer NLP iterations "
                    f"({inner_sqp_iters} inner SQP iterations)"
                )
                return 0

            # Refresh the linearization for the next positive SQP update.
            # update(0) cannot be used as a read-only check in Moto 2.x: it
            # reinitializes the solver.
            self.resolve_collisions(results["robot_xs"], results["obj_xs"])

            self.sqp_coll.settings.ipm.warm_start = True

        return 1

    def _get_sqp(self, coll):
        if coll is None:
            coll = self.sqp is None
        if not coll and self.sqp is None:
            raise RuntimeError(
                "The collision-free solver is disabled; set presolve_max_iters > 0 "
                "and initialize_solver again to use it."
            )
        return self.sqp_coll if coll else self.sqp

    def get_results(self, include_torque=False, coll=None):
        """Read the presolve result when available, otherwise the collision result.

        Pass ``coll=True`` for the final collision solve, or ``coll=False`` to
        require a collision-free solver.
        """
        results = self.moto_wrap.get_results(self._get_sqp(coll))

        if include_torque:
            total_wrenches = robot_wrench_trajectory(
                self.scene, self.formulation.robots, results["robot_xs"], results["obj_xs"],
                results["rob_Ws"], results["obj_Ws"], self.interface_ohs)
            torques, limits = {}, {}
            for rid, robot in self.formulation.robots.items():
                xs = results["robot_xs"][rid]
                wrenches = total_wrenches[rid]
                torques[rid] = np.zeros((self.N, robot.nv - 6))
                for k in range(self.N):
                    torques[rid][k] = np.asarray(self.formulation.get_torques_funcs[rid](
                        xs[k + 1, :robot.nq], xs[k + 1, robot.nq:robot.nq + robot.nv],
                        results["vdot_rob"][rid][k],
                        *(wrenches[patch][k] for patch in robot.cfg.patch_ees))[1]).reshape(-1)
                limits[rid] = robot.cfg.tau_max.copy()
            results["torques"], results["torque_limit"] = torques, limits

        return results

    def print_top_n_constr_vio(self, n=20):
        self.moto_wrap.print_top_constraint_violations(
            sqp=self.sqp_coll,
            top_n=n,
            tol=1e-8
        )

    def sum_constraint_violations(self):
        return self.moto_wrap.sum_constraint_violations(self.sqp_coll)

    def max_constraint_violation(self):
        return self.moto_wrap.max_constraint_violation(self.sqp_coll)

    def get_param_traj(self, param_name):
        return self.moto_wrap.get_param_traj(self._get_sqp(None), param_name)

    def get_top_n_collisions(self, robot_states, obj_states, n=10, do_print=False):
        self.resolve_collisions(robot_states, obj_states)
        return _diagnostics.top_collisions(
            self.collision_mgr.buffers.coll_pair_dists, n, do_print)

    def _validate_interface_containment(self, interface_one_hots):
        _contacts.validate_interface_containment(
            self.scene, self.formulation.object_features,
            interface_one_hots, self.interface_ohs, range(0, self.num_modes * self.nodes_per_mode, self.nodes_per_mode))

    def get_finite_diff_jerk_cost(self, rob_vds):
        return sum(float(np.sum(np.linalg.norm(np.diff(values, axis=0) / self.dt, axis=1)))
                   for values in rob_vds.values())

    def visualize_trajectory(self, rob_xs, obj_xs, rob_Ws, obj_Ws,
                             t_scale, transparent_robot = False,
                             robot_alpha = 0.1, show_forces = True):
        """Play N+1 knots using N outgoing, boundary-scaled intervals.

        show_forces sets initial force-arrow visibility; G toggles it in playback.
        """
        step_ts = self.get_step_ts({"t_scale": t_scale})
        mj_model = build_mujoco(self.scene, self.resolved)[0]
        viz = KinematicRobotVisualizer(
            mj_model,
            transparent_robot=transparent_robot,
            robot_alpha=robot_alpha,
            show_forces=show_forces,
        )

        N_nodes = self.N + 1
        total_wrenches = robot_wrench_trajectory(
            self.scene, self.formulation.robots, rob_xs, obj_xs, rob_Ws, obj_Ws, self.interface_ohs)

        for rid, cfg in self.robot_cfgs.items():
            model, data = self.models[rid], self.datas[rid]
            for ee in cfg.patch_ees:
                R_traj = np.zeros((N_nodes, 3, 3))
                p_traj = np.zeros((N_nodes, 3))
                bnds_traj = np.tile(cfg.patch_ees[ee], (N_nodes, 1))
                # Interval k controls are displayed at state k+1; x_0 has no
                # preceding interval wrench.
                wrenches = np.vstack((np.zeros((1, 6)), total_wrenches[rid][ee]))
                for i in range(N_nodes):
                    pin.framesForwardKinematics(model, data, rob_xs[rid][i, :model.nq])
                    transform = data.oMf[model.getFrameId(ee)]
                    rotation = pin.SE3(transform.rotation, np.zeros(3))
                    force = pin.Force(linear=wrenches[i, :3], angular=wrenches[i, 3:])
                    wrenches[i] = (rotation.inverse() * force).vector
                    R_traj[i], p_traj[i] = transform.rotation, transform.translation
                viz.add_patch_wrench(R_traj=R_traj, p_traj=p_traj,
                    wrench_local_traj=wrenches, bounds_local_traj=bnds_traj,
                    rgba=(0.9, 0.3, 0.3, 0.8), length_scale=0.005, flip=True)

        for obj_name, cfg in self.env_objs.items():
            R_traj     = np.zeros((N_nodes, 3, 3))   # (T,3,3)
            p_traj     = np.zeros((N_nodes, 3))      # (T,3)
            bnds_traj  = np.zeros((N_nodes, 4))      # (T,4)
            wrench_traj = np.zeros((N_nodes, 6))      # (T,6)

            body_wrenches = np.vstack((np.zeros((1, 6)), obj_Ws[obj_name]))

            self_oh_traj = self.interface_ohs[f"{obj_name}_self_feat_oh"]

            for i in range(N_nodes):
                x_i = obj_xs[obj_name][i, :]

                self_oh_i = self_oh_traj[:, max(i - 1, 0)]

                self_feature = self.formulation.object_features.local_feature(
                    obj_name, cfg.half_extents, self_oh_i
                )
                Rvec_i, p_i, bnds_i = \
                    self.formulation.object_features.face_pose(
                        x_i[:7], self_feature, flip=True
                    )
                R_i = np.array(Rvec_i.full()).reshape(3, 3, order="F")
                p_i = np.array(p_i.full()).reshape(3)
                bnds_i = np.array(bnds_i.full()).reshape(4)

                SE3 = pin.XYZQUATToSE3(x_i[:7])
                SE3_feat = pin.SE3(R_i, p_i)
                SE3_feat_body = SE3_feat.inverse() * SE3

                pin_body_wrench = pin.Force(linear=body_wrenches[i, 3:],
                                            angular=body_wrenches[i, :3])
                pin_local = SE3_feat_body * pin_body_wrench

                R_traj[i, :, :] = SE3_feat.rotation
                p_traj[i, :]    = SE3_feat.translation
                bnds_traj[i, :] = bnds_i
                wrench_traj[i, :3] = pin_local.linear
                wrench_traj[i, 3:] = pin_local.angular

            viz.add_patch_wrench(
                R_traj=R_traj,
                p_traj=p_traj,
                wrench_local_traj=wrench_traj,
                bounds_local_traj=bnds_traj,
                rgba=(0.9, 0.3, 0.3, 0.8),
                length_scale=0.005,
                flip=True,
            )

        mujoco_q_traj, _ = pack_trajectory(self.scene, mj_model, rob_xs, obj_xs)

        viz.visualize_trajectory(mujoco_q_traj, step_ts)

    def get_ts(self, results):
        """Return knot times t[0..N]; interval k ends at t[k+1]."""
        return np.concatenate([[0.0], np.cumsum(self.get_step_ts(results))])

    def get_step_ts(self, results):
        """Return durations for x[k] -> x[k+1], including switch factors.

        dt_traj already contains the contact-boundary multiplier. Its index
        matches t_scale, even when a legacy array includes a terminal entry.
        """
        t_scale = np.asarray(results["t_scale"], dtype=float).reshape(-1)
        dt_traj = np.asarray(self.dt_traj, dtype=float).reshape(-1)

        if dt_traj.size == t_scale.size + 1:
            dt_traj = dt_traj[:-1]
        elif dt_traj.size != t_scale.size:
            raise ValueError(
                "dt_traj must have either len(t_scale) or len(t_scale) + 1 entries."
            )

        return dt_traj * t_scale

    def plot_joint_trajectories(self, rob_xs, rob_torques, t_scale=None):
        """Plot on the physical time grid; omitted scales default to one.

        Pass results["t_scale"] to include the optimized interval durations.
        Contact-boundary multipliers apply in either case.
        """
        if t_scale is None:
            t_scale = np.ones(self.N)
        step_ts = self.get_step_ts({"t_scale": t_scale})
        for rid, robot in self.formulation.robots.items():
            xs = rob_xs[rid]
            layout = self.formulation.robot_layouts[rid]
            plot_joint_dynamics(robot.model, xs[:, layout.q], xs[:, layout.v],
                rob_torques[rid], xs[:, layout.r], step_ts,
                v_lims=robot.cfg.vel_max, tau_lims=robot.cfg.tau_max,
                endpoint_controls=True)

    def plot_timescale_traj(self, t_scale):
        """Plot each interval's effective scale, including boundary factors."""
        step_ts = self.get_step_ts({"t_scale": t_scale})
        plot_timescale_traj(step_ts / self.dt)

    def get_collision_metrics(self, rob_xs, obj_xs, penetration_threshold: float = 1e-3):
        num_steps = next(iter(rob_xs.values())).shape[0]
        distances = {}
        if num_steps:
            self.get_top_n_collisions(rob_xs, obj_xs, n=1, do_print=False)
            distances = self.collision_mgr.buffers.coll_pair_dists
        return _diagnostics.collision_metrics(distances, num_steps, penetration_threshold)
