"""Coupled manifold trajectory optimization for an ordered scene of robots."""
from dataclasses import replace
from functools import cached_property

import casadi
import numpy as np
import moto

from faro.configs.scene import Scene
from faro.configs.identifiers import robot_name, interface_name, capsule_name, solver_name
from faro.ocp.robot_blocks import setup_robot_blocks, BlockLayout
from faro.ocp.moto_states import StateLayout, configuration_states, difference
from faro.ocp.utils import pose_error
from faro.search.goal import PositionTarget, ConfigurationTarget
from faro.ocp.collision import collision_distances, object_poses, contact_collision_multiplier
from faro.ocp.object_features import ObjectFeatures
from faro.ocp.contact_wrenches import robot_contact_wrenches, robot_ee_poses
from faro.ocp.kinematics import rob_com
from faro.utils.contact_type import NUM_CONTACT_TYPES
from faro.ocp.dynamics import (
    kinematic_centroidal_momentum, centroidal_dynamics, get_joint_torques,
    ext_Wobj_body, single_rigid_body_dynamics,
)
from faro.ocp.moto_contact.rob_constr_factory import (
    RobConstraintVars, EnvConstraintVars, RobotPatchEnvPatchConstraint,
)
from faro.ocp.moto_contact.obj_constr_factory import (
    ObjConstraintVars, ObjectPatchEnvPatchConstraint,
)


class ToFormulation:
    def __init__(self, scene: Scene, collision_index, robot_collision_models=None,
                 verbose=False, terminal_goal=None):
        self.scene = scene
        self.verbose = verbose
        self.terminal_goal = terminal_goal
        if terminal_goal is not None:
            terminal_goal.validate(scene)
        self.robots = setup_robot_blocks(scene, robot_collision_models)
        self.robot_layouts = {
            rid: StateLayout(robot.nq, robot.nv, centroidal=True)
            for rid, robot in self.robots.items()
        }
        self.ee_constr_models = {
            key: RobotPatchEnvPatchConstraint() for key in scene.robot_interfaces
        }
        self.fixed_env_objects = scene.fixed_env_objects
        self.env_objects = scene.env_objects
        self.total_env_features = scene.total_env_features
        self.object_features = ObjectFeatures(scene)
        self.coll_pair_names = list(collision_index.pair_names)
        self.collision_idx = collision_index
        self.obj_constr_models = {
            name: ObjectPatchEnvPatchConstraint() for name in self.env_objects
        }
        self.obj_n_q, self.obj_n_v = 7, 6
        self.object_layout = StateLayout(7, 6)
        self.obj_n_x = self.object_layout.dim
        self.obj_n_dx = self.object_layout.tdim
        self.obj_n_u = 12
        self.parameters = {}
        self.contact_type_constraints = {}
        self.use_gauss_newton_costs = True
        self.rigid_body_dynamics = single_rigid_body_dynamics()
        self.momentum_funcs = {
            rid: kinematic_centroidal_momentum(robot.cmodel, robot.cdata)
            for rid, robot in self.robots.items()
        }
        self.centroidal_funcs = {
            rid: centroidal_dynamics(robot.cmodel, robot.cdata, list(robot.cfg.patch_ees))
            for rid, robot in self.robots.items()
        }
        self.get_torques_funcs = {
            rid: get_joint_torques(robot.cmodel, robot.cdata, list(robot.cfg.patch_ees))
            for rid, robot in self.robots.items()
        }
        self.rob_com_funcs = {
            rid: rob_com(robot.cmodel, robot.cdata)
            for rid, robot in self.robots.items()
        }
        self._setup_param_symbols()
        self._build_stage_symbols()

    def _sumsqr_cost(self, name, residual, weight, *, nonlinear=False):
        """Build a weighted sum-of-squares with configurable curvature.

        Moto uses a one-half objective convention. Scaling the residual by
        sqrt(2) gives the objective r.T W r. Gauss-Newton is the default;
        for an affine residual it is already the exact Hessian. In
        exact mode, nonlinear vector residuals are decomposed into scalar
        costs because Moto's public vector factory is Gauss-Newton-only.
        """
        scaled_residual = np.sqrt(2.0) * residual
        if (
            self.use_gauss_newton_costs
            or not nonlinear
            or residual.numel() == 1
        ):
            if residual.numel() == 1:
                term = moto.cost.from_scalar(
                    name, scaled_residual, weight=weight
                )
            else:
                term = moto.cost.from_vector(
                    name, scaled_residual, weight=weight
                )
            return {name: term}

        return {
            f"{name}_{idx}": self._exact_scalar_sumsqr_cost(
                f"{name}_{idx}", residual[idx], weight[idx]
            )
            for idx in range(residual.numel())
        }

    @staticmethod
    def _exact_scalar_sumsqr_cost(name, residual, weight):
        """Build one nonlinear squared residual with exact second derivatives."""
        return moto.cost.from_scalar(
            name, casadi.sqrt(2.0 * weight) * residual
        )

    def _setup_param_symbols(self):
        params = {}
        def add(name, size):
            params[name] = moto.sym.params(name, size)

        for rid, robot in self.robots.items():
            layout = self.robot_layouts[rid]
            for field, size in (("x0", layout.dim), ("mass", 1),
                                ("state_cost_W", layout.tdim),
                                ("state_cost_We", layout.tdim),
                                ("acc_cost_W", robot.nv), ("wrench", 6)):
                add(robot_name(rid, field), size)
        for name, size in (("total_time", 1), ("n_nodes", 1),
                           ("is_mode_switch", 1),
                           ("time_scale_cost_W", 1), ("object_state_cost_W", 12),
                           ("object_state_cost_We", 12), ("object_ctrl_cost_W", 12)):
            add(name, size)
        for key in self.ee_constr_models:
            name = interface_name(key)
            add(name + "_env_feat_oh", self.total_env_features)
            add(name + "_type_oh", NUM_CONTACT_TYPES)
            add(name + "_is_int_switch", 1)
        for name, cfg in self.fixed_env_objects.items():
            add(name + "_q0", 7)
            add(name + "_size", len(cfg.half_extents))
        for name, cfg in self.env_objects.items():
            for suffix, size in (("x0", 13), ("mass", 1), ("inertia", 3),
                                 ("size", 3),
                                 ("self_feat_oh", cfg.num_features),
                                 ("env_feat_oh", self.total_env_features),
                                 ("type_oh", NUM_CONTACT_TYPES),
                                 ("is_int_switch", 1)):
                add(name + "_" + suffix, size)
        for pair_name in self.coll_pair_names:
            add(pair_name, 9)
        for kind, owner, target in self._goal_targets():
            add(solver_name(kind, owner, "goal_reference"), target.value.size)
            if self.terminal_goal.to == "soft":
                add(solver_name(kind, owner, "goal_weight"), target.residual_size)
        self.params = params

    def _goal_targets(self):
        goal = self.terminal_goal
        if goal is not None and goal.to != "off":
            for rid, target in goal.robots.items():
                yield "robot", rid, target
            for name, target in goal.objects.items():
                yield "object", name, target

    def goal_residuals(self):
        """Terminal errors in physical coordinates, independent of regularization."""
        residuals = {}
        for kind, owner, target in self._goal_targets():
            q = (self.sym["q_rob"][owner] if kind == "robot"
                 else self.sym["obj_qs"][owner]).sx
            reference = self.params[solver_name(kind, owner, "goal_reference")]
            if isinstance(target, PositionTarget):
                error = q[:3] - reference
            else:
                error = pose_error(q[:7], reference[:7])
                if isinstance(target, ConfigurationTarget):
                    error = casadi.vertcat(error, q[7:] - reference[7:])
            residuals[solver_name(kind, owner, "goal")] = error
        return residuals

    def _build_stage_symbols(self):
        sym = {key: {} for key in (
            "q_rob", "qn_rob", "v_rob", "vn_rob", "r_rob", "rn_rob", "h_rob", "hn_rob",
            "rob_x", "rob_xn", "vdot_rob", "rob_Ws",
            "obj_qs", "obj_qns", "obj_Vs", "obj_Vns", "obj_xs", "obj_xns",
            "obj_Vdots", "obj_Ws", "obj_us",
        )}
        self.x_args, self.y_args = [], []
        for rid, robot in self.robots.items():
            sym["q_rob"][rid], sym["qn_rob"][rid] = configuration_states(
                robot_name(rid, "q"), robot.model)
            for key, next_key, size in (("v_rob", "vn_rob", robot.nv),
                                        ("r_rob", "rn_rob", 3), ("h_rob", "hn_rob", 6)):
                sym[key][rid], sym[next_key][rid] = moto.sym.states(robot_name(rid, key), size)
            xs = [sym[key][rid] for key in ("q_rob", "v_rob", "r_rob", "h_rob")]
            ys = [sym[key][rid] for key in ("qn_rob", "vn_rob", "rn_rob", "hn_rob")]
            sym["rob_x"][rid], sym["rob_xn"][rid] = casadi.vertcat(*xs), casadi.vertcat(*ys)
            self.x_args.extend(xs)
            self.y_args.extend(ys)
            sym["vdot_rob"][rid] = moto.sym.inputs(robot_name(rid, "acceleration"), robot.nv)
            sym["rob_Ws"][rid] = {
                patch: moto.sym.inputs(interface_name(("ee", rid, patch)) + "_W", 6)
                for patch in robot.cfg.patch_ees
            }
        for name in self.env_objects:
            sym["obj_qs"][name], sym["obj_qns"][name] = configuration_states(solver_name("object", name, "q"))
            sym["obj_Vs"][name], sym["obj_Vns"][name] = moto.sym.states(solver_name("object", name, "V"), 6)
            sym["obj_xs"][name] = casadi.vertcat(sym["obj_qs"][name], sym["obj_Vs"][name])
            sym["obj_xns"][name] = casadi.vertcat(sym["obj_qns"][name], sym["obj_Vns"][name])
            self.x_args.extend((sym["obj_qs"][name], sym["obj_Vs"][name]))
            self.y_args.extend((sym["obj_qns"][name], sym["obj_Vns"][name]))
            sym["obj_Vdots"][name] = moto.sym.inputs(solver_name("object", name, "Vdot"), 6)
            sym["obj_Ws"][name] = moto.sym.inputs(solver_name("object", name, "W"), 6)
            sym["obj_us"][name] = casadi.vertcat(sym["obj_Vdots"][name], sym["obj_Ws"][name])
        sym["t_scale"] = moto.sym.inputs("t_scale", 1)
        sym["x"] = casadi.vertcat(*sym["rob_x"].values(), *sym["obj_xs"].values())
        sym["y"] = casadi.vertcat(*sym["rob_xn"].values(), *sym["obj_xns"].values())
        robot_controls = []
        for rid in self.robots:
            robot_controls.extend((sym["vdot_rob"][rid], *sym["rob_Ws"][rid].values()))
        sym["u"] = casadi.vertcat(*robot_controls, *sym["obj_us"].values(), sym["t_scale"])
        sym["p"] = casadi.vertcat(*[casadi.vec(value) for value in self.params.values()])
        self.state_layout = BlockLayout({
            **{("robot", rid): layout.dim for rid, layout in self.robot_layouts.items()},
            **{("object", name): self.object_layout.dim for name in self.env_objects},
        })
        self.control_layout = BlockLayout({
            **{("robot", rid): robot.nv + 6 * len(robot.cfg.patch_ees)
               for rid, robot in self.robots.items()},
            **{("object", name): self.obj_n_u for name in self.env_objects},
            ("time",): 1,
        })
        self.sym = sym

    def interval_duration(self):
        switch_sum = sum(
            self.params[interface_name(("ee", rid, patch)) + "_is_int_switch"]
            for rid, robot in self.robots.items() for patch in robot.cfg.switch_contacts
        )
        return (self.sym["t_scale"] * self.params["total_time"] / self.params["n_nodes"]
                * casadi.if_else(switch_sum >= 1, 0.1, 1.0))

    @cached_property
    def control_endpoint(self):
        """Endpoint states and loads shared by Moto's x/u control constraints.

        Substitute the configuration/velocity integration equations to avoid
        next-state/control constraints. These expressions equal the explicit
        next states on the dynamics manifold; u_k uses endpoint wrench frames.
        """
        dt = self.interval_duration()
        endpoint = {key: {} for key in (
            "q_rob", "v_rob", "obj_qs", "obj_Vs", "obj_xs", "ee_states")}
        for rid, robot in self.robots.items():
            q = self.sym["q_rob"][rid]
            v = self.sym["v_rob"][rid] + dt * self.sym["vdot_rob"][rid]
            qn = q.symbolic_integrate(q.sx, dt * v)
            endpoint["q_rob"][rid], endpoint["v_rob"][rid] = qn, v
            endpoint["ee_states"].update({("ee", rid, patch): state
                for patch, state in robot.kinematics.states(qn, v).items()})
        for name in self.env_objects:
            q = self.sym["obj_qs"][name]
            v = self.sym["obj_Vs"][name] + dt * self.sym["obj_Vdots"][name]
            qn = q.symbolic_integrate(q.sx, dt * v)
            endpoint["obj_qs"][name], endpoint["obj_Vs"][name] = qn, v
            endpoint["obj_xs"][name] = casadi.vertcat(qn, v)
        endpoint["ee_poses"] = {key: state[:2] for key, state in endpoint["ee_states"].items()}
        endpoint["wrenches"] = robot_contact_wrenches(
            self.scene, endpoint["ee_poses"], endpoint["obj_qs"],
            self.sym["rob_Ws"], self.sym["obj_Ws"], self.params)
        return endpoint

    def total_robot_wrenches(self, *, next_state=False):
        """Endpoint loads: explicit next states for dynamics, substituted for x/u."""
        if not next_state:
            return self.control_endpoint["wrenches"]
        return robot_contact_wrenches(
            self.scene, robot_ee_poses(self.robots, self.sym["qn_rob"]),
            {name: q.sx for name, q in self.sym["obj_qns"].items()},
            self.sym["rob_Ws"], self.sym["obj_Ws"], self.params)

    def dynamics_residual(self):
        """Fully implicit manifold step with controls expressed at the endpoint."""
        dt = self.interval_duration()
        residuals = []
        total_wrenches = self.total_robot_wrenches(next_state=True)
        for rid, robot in self.robots.items():
            q, qn = self.sym["q_rob"][rid], self.sym["qn_rob"][rid]
            v, vn = self.sym["v_rob"][rid], self.sym["vn_rob"][rid]
            r, rn = self.sym["r_rob"][rid], self.sym["rn_rob"][rid]
            h, hn = self.sym["h_rob"][rid], self.sym["hn_rob"][rid]
            mass = self.params[robot_name(rid, "mass")]
            poses = robot.kinematics.poses(qn)
            hdot = self.centroidal_funcs[rid](
                qn, rn, mass, *(total_wrenches[rid][ee] for ee in robot.cfg.patch_ees),
                *(poses[("ee", ee)][1] for ee in robot.cfg.patch_ees))
            residuals.extend((
                difference(q, q.symbolic_integrate(q.sx, dt * vn), qn.sx),
                vn - v - dt * self.sym["vdot_rob"][rid],
                rn - r - dt * hn[:3] / mass,
                hn - h - dt * hdot,
            ))
        for name in self.env_objects:
            q, qn = self.sym["obj_qs"][name], self.sym["obj_qns"][name]
            v, vn = self.sym["obj_Vs"][name], self.sym["obj_Vns"][name]
            residuals.extend((difference(q, q.symbolic_integrate(q.sx, dt * vn), qn.sx),
                              vn - v - dt * self.sym["obj_Vdots"][name]))
        residual = casadi.vertcat(*residuals)
        assert residual.numel() == sum(arg.tdim for arg in self.y_args)
        return residual

    def get_dynamics(self):
        return moto.dense_dynamics.create("dynamics", self.dynamics_residual())

    def state_residuals(self):
        robots, objects = {}, {}
        for rid, robot in self.robots.items():
            ref = self.params[robot_name(rid, "x0")]
            q = self.sym["q_rob"][rid]
            robots[rid] = casadi.vertcat(difference(q, ref[:robot.nq], q.sx),
                self.sym["rob_x"][rid][robot.nq:] - ref[robot.nq:])
        for name in self.env_objects:
            ref, q = self.params[name + "_x0"], self.sym["obj_qs"][name]
            objects[name] = casadi.vertcat(difference(q, ref[:7], q.sx),
                                         self.sym["obj_Vs"][name] - ref[7:])
        return robots, objects

    def get_object_fma(self):
        """Endpoint force balance, including reactions from every robot/object."""
        endpoint = self.control_endpoint
        poses = {interface_name(key): pose for key, pose in endpoint["ee_poses"].items()}
        equalities = {}
        for name in self.env_objects:
            robot_forces = {
                interface_name(("ee", rid, ee)): wrench * self.object_features.object_weight(
                    name, self.params[interface_name(("ee", rid, ee)) + "_env_feat_oh"])
                for rid, wrenches in self.sym["rob_Ws"].items()
                for ee, wrench in wrenches.items()
            }
            object_forces = {
                other: self.sym["obj_Ws"][other] * self.object_features.object_weight(
                    name, self.params[other + "_env_feat_oh"])
                for other in self.env_objects if other != name
            }
            W_ext = ext_Wobj_body(robot_forces, object_forces)(
                endpoint["obj_qs"][name], self.params[name + "_mass"],
                *robot_forces.values(), *(poses[key][0] for key in robot_forces),
                *(poses[key][1] for key in robot_forces), *object_forces.values(),
                *(endpoint["obj_qs"][other] for other in object_forces))
            W_inertial = self.rigid_body_dynamics(
                endpoint["obj_Vs"][name], self.sym["obj_Vdots"][name],
                self.params[name + "_mass"], self.params[name + "_inertia"])
            key = name + "_obj_f_eq_ma"
            equalities[key] = moto.constr.create(key, -W_inertial + W_ext + self.sym["obj_Ws"][name])
        return equalities

    def get_cost(self):
        robots, objects = self.state_residuals()
        cost = self._sumsqr_cost("time_scale", self.sym["t_scale"] - 1., self.params["time_scale_cost_W"])
        for rid in self.robots:
            cost.update(self._sumsqr_cost(robot_name(rid, "state"), robots[rid],
                        self.params[robot_name(rid, "state_cost_W")], nonlinear=True))
            cost.update(self._sumsqr_cost(robot_name(rid, "acc"), self.sym["vdot_rob"][rid],
                        self.params[robot_name(rid, "acc_cost_W")]))
            for ee, wrench in self.sym["rob_Ws"][rid].items():
                cost.update(self._sumsqr_cost(interface_name(("ee", rid, ee)) + "_wrench_cost",
                            wrench, self.params[robot_name(rid, "wrench")]))
        for name, error in objects.items():
            cost.update(self._sumsqr_cost(name + "_state", error,
                        self.params["object_state_cost_W"], nonlinear=True))
            cost.update(self._sumsqr_cost(name + "_ctrl", self.sym["obj_us"][name],
                        self.params["object_ctrl_cost_W"]))
        return cost

    def get_cost_term(self):
        robots, objects = self.state_residuals()
        cost = {}
        for rid in self.robots:
            cost.update(self._sumsqr_cost(robot_name(rid, "state_term"), robots[rid],
                        self.params[robot_name(rid, "state_cost_We")], nonlinear=True))
        for name, error in objects.items():
            cost.update(self._sumsqr_cost(name + "_state_term", error,
                        self.params["object_state_cost_We"], nonlinear=True))
        if self.terminal_goal is not None and self.terminal_goal.to == "soft":
            residuals = self.goal_residuals()
            for kind, owner, _ in self._goal_targets():
                name = solver_name(kind, owner, "goal")
                # _sumsqr_cost emits r.T W r; goals use 1/2 r.T W r.
                cost.update(self._sumsqr_cost(name, residuals[name],
                    .5 * self.params[solver_name(kind, owner, "goal_weight")],
                    nonlinear=True))
        return cost

    def get_terminal_constraints(self):
        """Send hard terminal goals to the IPM, pairing exact residuals at zero."""
        equalities, inequalities = {}, {}
        if self.terminal_goal is None or self.terminal_goal.to != "hard":
            return equalities, inequalities
        residuals = self.goal_residuals()
        for kind, owner, target in self._goal_targets():
            name = solver_name(kind, owner, "goal")
            error, tolerance = residuals[name], target.tolerance("to")
            exact = np.flatnonzero(tolerance == 0).tolist()
            bounded = np.flatnonzero(tolerance > 0).tolist()
            if exact:
                e = error[exact]
                inequalities[name] = moto.ineq.create(name, casadi.vertcat(e, -e))
            if bounded:
                name += "_bounds"
                e, t = error[bounded], tolerance[bounded]
                inequalities[name] = moto.ineq.create(name, casadi.vertcat(-t - e, e - t))
        return equalities, inequalities

    def get_global_path_constraints(self):
        equalities, inequalities = {}, {}
        endpoint = self.control_endpoint
        total_wrenches = self.total_robot_wrenches()
        for rid, robot in self.robots.items():
            q, v = self.sym["q_rob"][rid], self.sym["v_rob"][rid]
            name = robot_name(rid, "Momentum_constr")
            equalities[name] = moto.constr.create(name,
                self.sym["h_rob"][rid] - self.momentum_funcs[rid](q, v))
            qn, vn = endpoint["q_rob"][rid], endpoint["v_rob"][rid]
            _, tau = self.get_torques_funcs[rid](qn, vn, self.sym["vdot_rob"][rid],
                           *(total_wrenches[rid][ee] for ee in robot.cfg.patch_ees))
            slope = robot.cfg.tau_max / robot.cfg.vel_max
            for suffix, expr in (("pos", tau + slope * vn[6:]), ("neg", tau - slope * vn[6:])):
                name = robot_name(rid, "Power_limit_" + suffix)
                inequalities[name] = moto.ineq.create(name,
                    casadi.vertcat(-robot.cfg.tau_max - expr, expr - robot.cfg.tau_max))
            name = robot_name(rid, "joint_pos_bound")
            inequalities[name] = moto.ineq.create(name,
                casadi.vertcat(robot.cfg.q_min - q[7:], q[7:] - robot.cfg.q_max))
            name = robot_name(rid, "joint_vel_bound")
            inequalities[name] = moto.ineq.create(name,
                casadi.vertcat(-robot.cfg.vel_max - v[6:], v[6:] - robot.cfg.vel_max))
        inequalities["time_scale_bound"] = moto.ineq.create("time_scale_bound",
            casadi.vertcat(.5 - self.sym["t_scale"], self.sym["t_scale"] - 2.))
        return equalities, inequalities

    def get_stage_constraints(self):
        obj_xs, obj_xns = self.sym["obj_xs"], self.sym["obj_xns"]
        obj_us, obj_Ws = self.sym["obj_us"], self.sym["obj_Ws"]
        endpoint = self.control_endpoint
        equalities, inequalities = {}, {}
        # Phase selection removes unused equality rows, instead of leaving
        # zero rows in the KKT system for either sticking or sliding phases.
        self.contact_type_constraints = {}
        sizes = {name: self.params[name + "_size"] for name in self.object_features.object_names}
        obj_features = self.object_features.local_features(sizes)
        fixed_obj_states = {
            name: casadi.vertcat(self.params[name + "_q0"], casadi.SX.zeros(6))
            for name in self.fixed_env_objects
        }
        ee_world_states = {("ee", rid, patch): state
                           for rid, robot in self.robots.items()
                           for patch, state in robot.kinematics.states(
                               self.sym["q_rob"][rid], self.sym["v_rob"][rid]).items()}
        next_ee_poses = robot_ee_poses(self.robots, self.sym["qn_rob"])
        world_features = self.object_features.world_features(
            sizes, {**fixed_obj_states, **obj_xs}, ee_world_states, velocity=True)
        next_features = self.object_features.world_features(
            sizes, {**fixed_obj_states, **obj_xns}, next_ee_poses)
        endpoint_features = self.object_features.world_features(
            sizes, {**fixed_obj_states, **endpoint["obj_xs"]}, endpoint["ee_states"], velocity=True)
        for rid, robot in self.robots.items():
            for ee in robot.cfg.patch_ees:
                key = ("ee", rid, ee)
                prefix = interface_name(key)
                constr = self.ee_constr_models[key]
                feat_oh = self.params[prefix + "_env_feat_oh"]
                Rf, pf, bounds, Vf = self.object_features.select(world_features, feat_oh)
                Rfn, pfn, _ = self.object_features.select(next_features, feat_oh)
                R, p, V = ee_world_states[key]
                Rn, pn = next_ee_poses[key]
                wrench = self.sym["rob_Ws"][rid][ee]
                switch = self.params[prefix + "_is_int_switch"] if ee in robot.cfg.switch_contacts else 0
                rob_vars = RobConstraintVars(p_i_ee=p, R_i_ee=R, V_i_ee=V,
                    p_n_ee=pn, R_n_ee=Rn, p_bnds=robot.cfg.patch_ees[ee],
                    W_rob=wrench, active=1, is_int_switch=switch)
                env_vars = EnvConstraintVars(fp_p=pf, fp_R=Rf, fp_V=Vf,
                    fp_pn=pfn, fp_Rn=Rfn, fp_bnds=bounds, mu=casadi.SX(.5),
                    containment_margin=.05 * casadi.sum1(self.object_features.object_weights(feat_oh)))
                end_R, end_p, end_V = endpoint["ee_states"][key]
                end_Rf, end_pf, _, end_Vf = self.object_features.select(endpoint_features, feat_oh)
                force_vars = replace(rob_vars, p_i_ee=end_p, R_i_ee=end_R, V_i_ee=end_V)
                force_env_vars = replace(env_vars, fp_p=end_pf, fp_R=end_Rf, fp_V=end_Vf)
                wrench_ineqs = constr.get_wrench_ineqs(force_vars, force_env_vars)
                def add(target, suffix, expression, *, kinematic=False):
                    name = prefix + suffix
                    factory = moto.constr if target is equalities else moto.ineq
                    kwargs = {"field": moto.field.field___eq_x} if kinematic else {}
                    target[name] = factory.create(name, expression, **kwargs)
                    target[name].enable_if_all([wrench])
                if constr.has_ctrl_eqs and ee in robot.cfg.switch_contacts:
                    add(equalities, "_ctrl_eqs", constr.get_wrench_eqs(rob_vars, env_vars))
                if constr.has_ctrl_ineqs:
                    add(inequalities, "_ctrl_ineqs", -wrench_ineqs)
                kin_eqs, kin_ineqs = constr.get_kin_constraints(rob_vars, env_vars)
                if constr.has_kin_eqs:
                    add(equalities, "_kin_eqs", kin_eqs, kinematic=True)
                if constr.has_kin_ineqs:
                    add(inequalities, "_kin_ineqs", -kin_ineqs)
                # Both contact modes use the same endpoint wrench cone.
                slide_eqs, slide_ineqs = constr.get_sliding_constraints(
                    force_vars, force_env_vars)
                # Sliding geometry depends only on the interval endpoint.
                # Author it on x so Moto can place it through stage.ed.
                sliding_geometry = casadi.substitute(
                    constr.get_sliding_kin_eqs(rob_vars, env_vars),
                    self.sym["y"], self.sym["x"])
                add(equalities, "_sliding_kin_eqs", sliding_geometry, kinematic=True)
                add(equalities, "_sliding_ctrl_eqs", slide_eqs)
                add(inequalities, "_sliding_ctrl_ineqs", -slide_ineqs)
                self.contact_type_constraints[key] = {
                    "sticking": [equalities[prefix + "_kin_eqs"]],
                    "sliding": [equalities[prefix + "_sliding_kin_eqs"],
                                equalities[prefix + "_sliding_ctrl_eqs"],
                                inequalities[prefix + "_sliding_ctrl_ineqs"]],
                }

        for obj_name, obj_constr in self.obj_constr_models.items():           
            # Object Self-Face Constraints
            obj_face_one_hot = self.params[obj_name + "_self_feat_oh"]
            env_obj_feat_one_hot = self.params[obj_name + "_env_feat_oh"]
            active = 1

            self_feature = (
                obj_features[:, self.object_features.feature_slices[obj_name]]
                @ obj_face_one_hot
            )
            Rvec_face_obj, p_face_obj, box_face_bnds, v_face_obj = \
                self.object_features.face_pose_velocity(
                    obj_xs[obj_name], self_feature
                )
            Rnvec_face_obj, pn_face_obj, _ = \
                self.object_features.face_pose(
                    obj_xns[obj_name], self_feature
                )

            impact = self.params[f'{obj_name}_is_int_switch'] * (1 - active)

            obj_vars = ObjConstraintVars(
                        q_i_obj = obj_xs[obj_name][:self.obj_n_q],
                        Wb_obj  = obj_us[obj_name][6:],
                        p_i_obj = p_face_obj,
                        R_i_obj = Rvec_face_obj,
                        V_i_obj = v_face_obj,
                        p_n_obj = pn_face_obj,
                        R_n_obj = Rnvec_face_obj,
                        p_bnds  = box_face_bnds,
                        active  = active,
                        impact  = impact,
                        )

            R_face_env, p_face_env, env_face_bnds, v_face_env = self.object_features.select(
                world_features, env_obj_feat_one_hot)
            Rn_face_env, pn_face_env, _ = self.object_features.select(
                next_features, env_obj_feat_one_hot)

            obj_env_vars = EnvConstraintVars(
                        fp_p = p_face_env,
                        fp_R = R_face_env,
                        fp_V = v_face_env,
                        fp_pn = pn_face_env,
                        fp_Rn = Rn_face_env,
                        fp_bnds = env_face_bnds,
                        mu=casadi.SX(0.5), # Friction coefficient
                containment_margin=.05 * casadi.sum1(
                    self.object_features.object_weights(env_obj_feat_one_hot)),
            )
            end_R, end_p, _, end_V = self.object_features.face_pose_velocity(
                endpoint["obj_xs"][obj_name], self_feature)
            end_Rf, end_pf, _, end_Vf = self.object_features.select(endpoint_features, env_obj_feat_one_hot)
            force_vars = replace(obj_vars, q_i_obj=endpoint["obj_qs"][obj_name],
                                 p_i_obj=end_p, R_i_obj=end_R, V_i_obj=end_V)
            force_env_vars = replace(obj_env_vars, fp_p=end_pf, fp_R=end_Rf, fp_V=end_Vf)
            wrench_ineqs = obj_constr.get_wrench_ineqs(force_vars, force_env_vars)

            if obj_constr.has_ctrl_ineqs:
                inequalities[f"obj_{obj_name}_ctrl_ineqs"] = moto.ineq.create(
                    f"obj_{obj_name}_ctrl_ineqs",
                    -wrench_ineqs,
                )
                inequalities[f"obj_{obj_name}_ctrl_ineqs"].enable_if_all(
                    [obj_Ws[obj_name]]
                )
            kin_eqs, kin_ineqs = obj_constr.get_kin_constraints(obj_vars, obj_env_vars)

            if obj_constr.has_kin_eqs:
                equalities[f"obj_{obj_name}_kin_eqs"] = moto.constr.create(
                    f"obj_{obj_name}_kin_eqs",
                    kin_eqs,
                    field=moto.field.field___eq_x
                )
                equalities[f"obj_{obj_name}_kin_eqs"].enable_if_all(
                    [obj_Ws[obj_name]]
                )
            if obj_constr.has_kin_ineqs:
                inequalities[f"obj_{obj_name}_kin_ineqs"] = moto.ineq.create(
                    f"obj_{obj_name}_kin_ineqs",
                    -kin_ineqs,
                )
                inequalities[f"obj_{obj_name}_kin_ineqs"].enable_if_all(
                    [obj_Ws[obj_name]]
                )
            prefix = f"obj_{obj_name}"
            slide_eqs, slide_ineqs = obj_constr.get_sliding_constraints(
                force_vars, force_env_vars)
            equalities[prefix + "_sliding_kin_eqs"] = moto.constr.create(
                prefix + "_sliding_kin_eqs", casadi.substitute(
                    obj_constr.get_sliding_kin_eqs(obj_vars, obj_env_vars),
                    self.sym["y"], self.sym["x"]),
                field=moto.field.field___eq_x)
            equalities[prefix + "_sliding_ctrl_eqs"] = moto.constr.create(
                prefix + "_sliding_ctrl_eqs", slide_eqs)
            inequalities[prefix + "_sliding_ctrl_ineqs"] = moto.ineq.create(
                prefix + "_sliding_ctrl_ineqs", -slide_ineqs)
            sliding_constraints = [equalities[prefix + "_sliding_kin_eqs"],
                                   equalities[prefix + "_sliding_ctrl_eqs"],
                                   inequalities[prefix + "_sliding_ctrl_ineqs"]]
            for constraint in sliding_constraints:
                constraint.enable_if_all([obj_Ws[obj_name]])
            self.contact_type_constraints['obj', obj_name] = {
                "sticking": [equalities[prefix + "_kin_eqs"]],
                "sliding": sliding_constraints,
            }

        return equalities, inequalities

    def get_collision_constraints(self):
        states = {name: self.params[name + "_q0"].sx for name in self.fixed_env_objects}
        states.update(self.sym["obj_xs"])
        robot_poses = {
            capsule_name(rid, name): pose
            for rid, robot in self.robots.items()
            for (kind, name), pose in robot.kinematics.poses(self.sym["q_rob"][rid]).items()
            if kind == "collision"
        }
        distances = collision_distances(self.collision_idx.groups, robot_poses,
                                       object_poses(states), self.params)
        inequalities = {}
        for group, pairs in self.collision_idx.groups.items():
            for pair, values in pairs.items():
                dist = distances[group][pair]
                dist *= contact_collision_multiplier(
                    self.scene, self.object_features, group, values, self.params, trajectory=True)
                name = pair + "_distance"
                inequalities[name] = moto.ineq.create(name, -dist)
        return inequalities
