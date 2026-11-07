"""Kinematic sequence optimization at contact-mode transitions.

Contact parameters without a side suffix describe the outgoing mode;
``*_incoming_oh`` parameters describe the incoming mode at the same state.
"""

import casadi as cs
import numpy as np
from faro.configs.identifiers import robot_name, interface_name, capsule_name, solver_name
from faro.ocp.robot_blocks import setup_robot_blocks, BlockLayout
from faro.ocp.configuration import configuration_difference
from faro.ocp.object_features import ObjectFeatures
from faro.ocp.collision import collision_distances, object_poses, contact_collision_multiplier
from faro.ocp.ocp_description import OCPDescription, ConstraintBuilder, LinearBound
from faro.ocp.kso_contact import rob_constr_factory, obj_constr_factory
from faro.ocp.utils import pose_error
from faro.search.goal import PositionTarget, ConfigurationTarget
from faro.utils.contact_type import NUM_CONTACT_TYPES, CONTACT_TYPE_SLIDING, CONTACT_TYPE_TO_COL


def _stack(values):
    return cs.vertcat(*values) if values else cs.SX.zeros(0, 1)


class KsoFormulation:
    """XYZ/XYZW/joint states, with a full next configuration as each control."""

    obj_n_q, obj_n_v, obj_n_u = (7, 6, 7)

    def __init__(
        self,
        scene,
        collision_index,
        robot_collision_models=None,
        verbose=False,
        terminal_goal=None,
    ):
        self.scene, self.verbose = (scene, verbose)
        self.terminal_goal = terminal_goal
        if terminal_goal is not None:
            terminal_goal.validate(scene)
        self.goal_targets = tuple(
            [("robot", name, target) for name, target in terminal_goal.robots.items()]
            + [("object", name, target) for name, target in terminal_goal.objects.items()]
        ) if terminal_goal is not None and terminal_goal.kso != "off" else ()
        self.robots = setup_robot_blocks(scene, robot_collision_models)
        self.robot_models = {rid: b.model for rid, b in self.robots.items()}
        self.robot_datas = {rid: b.data for rid, b in self.robots.items()}
        self.rob_n_q = {rid: b.nq for rid, b in self.robots.items()}
        self.rob_n_v = {rid: b.nv for rid, b in self.robots.items()}
        self.all_ees = {rid: tuple(b.cfg.patch_ees) for rid, b in self.robots.items()}
        self.fixed_env_objects, self.env_objects = (scene.fixed_env_objects, scene.env_objects)
        self.total_env_features = scene.total_env_features
        self.object_features = ObjectFeatures(scene)
        self.collision_idx = collision_index
        self.coll_pair_names = list(collision_index.pair_names)
        self.ee_constr_models = {
            rid: {
                patch: rob_constr_factory.RobotPatchEnvPatchConstraint()
                for patch in block.cfg.patch_ees
            }
            for rid, block in self.robots.items()
        }
        self.obj_constr_models = {
            name: obj_constr_factory.ObjectPatchEnvPatchConstraint()
            for name in self.env_objects
        }
        self.params_syms = self._setup_param_symbols()

    def _setup_param_symbols(self):
        params = {}

        def add(name, size):
            params[name] = cs.SX.sym(name, size)

        for rid, block in self.robots.items():
            add(robot_name(rid, 'q0'), block.nq)
            for field in ('rob_state_cost_W', 'rob_ctrl_cost_W', 'rob_state_cost_We'):
                add(robot_name(rid, field), block.nv)
            for patch in block.cfg.patch_ees:
                prefix = interface_name(('ee', rid, patch))
                add(prefix + '_env_feat_oh', self.total_env_features)
                add(prefix + '_env_feat_incoming_oh', self.total_env_features)
                add(prefix + '_type_oh', NUM_CONTACT_TYPES)
                add(prefix + '_type_incoming_oh', NUM_CONTACT_TYPES)
        add('n_nodes', 1)
        add('total_time', 1)
        for name, cfg in {**self.fixed_env_objects, **self.env_objects}.items():
            add(name + '_q0', 7)
            add(name + '_size', len(cfg.half_extents))
        for name, cfg in self.env_objects.items():
            add(name + '_self_feat_oh', cfg.num_features)
            add(name + '_env_feat_oh', self.total_env_features)
            add(name + '_self_feat_incoming_oh', cfg.num_features)
            add(name + '_env_feat_incoming_oh', self.total_env_features)
            add(name + '_type_oh', NUM_CONTACT_TYPES)
            add(name + '_type_incoming_oh', NUM_CONTACT_TYPES)
        for name in self.coll_pair_names:
            add(name, 9)
        for field in ('object_state_cost_W', 'object_ctrl_cost_W', 'object_state_cost_We'):
            add(field, 6)
        for kind, name, target in self.goal_targets:
            add(solver_name(kind, name, "goal_reference"), target.value.size)
            if self.terminal_goal.kso == "soft":
                add(solver_name(kind, name, "goal_weight"), target.residual_size)
        return params

    def goal_parameters(self):
        """Bind only the targets compiled into this formulation."""
        params = {}
        for kind, name, target in self.goal_targets:
            params[solver_name(kind, name, "goal_reference")] = target.value
            if self.terminal_goal.kso == "soft":
                params[solver_name(kind, name, "goal_weight")] = target.weights("kso")
        return params

    def _build_stage_symbols(self):
        robot_qs = {
            rid: cs.SX.sym(robot_name(rid, 'q_rob'), b.nq)
            for rid, b in self.robots.items()
        }
        obj_qs = {name: cs.SX.sym(name + '_q', 7) for name in self.env_objects}
        x_dict = {robot_name(rid, 'q_rob'): q for rid, q in robot_qs.items()}
        x_dict.update(obj_qs)
        symbols = {'q_rob': robot_qs, 'obj_qs': obj_qs}
        robot_next = {
            rid: cs.SX.sym(robot_name(rid, 'q_next_rob'), b.nq)
            for rid, b in self.robots.items()
        }
        obj_next = {name: cs.SX.sym(name + '_q_next', 7) for name in self.env_objects}
        u_dict = {robot_name(rid, 'q_next_rob'): q for rid, q in robot_next.items()}
        u_dict.update(obj_next)
        symbols.update(q_next_rob=robot_next, obj_qs_next=obj_next)
        self.x_layout, self.u_layout = (
            BlockLayout.from_symbols(x_dict),
            BlockLayout.from_symbols(u_dict)
        )
        symbols.update(
            x=_stack(list(x_dict.values())),
            u=_stack(list(u_dict.values())),
            p=_stack(list(self.params_syms.values())),
            x_dict=x_dict,
            u_dict=u_dict
        )
        return symbols

    def _build_dynamics(self, sym):
        """Link each next-configuration control to the next shooting state.

        Acados uses one Gauss-Legendre stage and one step (implicit midpoint).
        With h = total_time / n_nodes, xdot = 2/h * (u - x_mid) gives
        x_next = u exactly. These are ambient-coordinate chords, not angular
        velocities: replacing them with an nv-dimensional log error would break
        both the IRK dimensions and the exact endpoint link. Keep this map linear:
        a quaternion sign branch at the unknown midpoint admits multiple IRK
        roots and makes its result depend on the cached Newton initialization.
        Signs are aligned in warm starts; geometric errors handle q and -q.
        Unit norms are imposed at shooting nodes. No geometry is evaluated at
        the midpoint, so even an antipodal pair's zero midpoint is safe.
        """
        params = self.params_syms
        x, u, p = (sym['x'], sym['u'], sym['p'])
        derivative = 2 * params['n_nodes'] / params['total_time'] * (u - x)
        xdot = cs.SX.sym('xdot', x.size1())
        f_impl = cs.Function('f_impl', [x, xdot, u, p], [xdot - derivative])
        f_expl = cs.Function('f_expl', [x, u, p], [derivative])
        f_discr = cs.Function('f_discr', [x, u, p], [u])
        return (f_impl, f_expl, f_discr)

    def _build_costs(self, sym):
        """Build this formulation's path and terminal residuals."""
        params = self.params_syms
        x, u, p = (sym['x'], sym['u'], sym['p'])
        cost, terminal = self.get_regularization_cost(
            sym['q_rob'],
            sym['obj_qs'],
            sym['q_next_rob'],
            sym['obj_qs_next'],
            params
        )
        terminal.update(self.get_task_cost(sym['q_rob'], sym['obj_qs'], params))
        residual = cs.Function('r_path', [x, u, p], [_stack(list(cost.values()))])
        residual_terminal = cs.Function('r_terminal', [x, u, p], [_stack(list(terminal.values()))])
        return (residual, residual_terminal)

    def get_regularization_cost(self, q_rob, obj_qs, q_next_rob, obj_qs_next, params):
        cost, terminal = ({}, {})
        for rid in self.robots:
            reference = params[robot_name(rid, 'q0')]
            error = configuration_difference(reference, q_rob[rid])
            next_error = configuration_difference(reference, q_next_rob[rid])
            cost[robot_name(rid, 'state')] = cs.sqrt(params[robot_name(
                rid, 'rob_state_cost_W')]) * error
            terminal[robot_name(rid, 'state_e')] = cs.sqrt(params[robot_name(
                rid, 'rob_state_cost_We')]) * error
            cost[robot_name(rid, 'ctrl')] = cs.sqrt(params[robot_name(
                rid, 'rob_ctrl_cost_W')]) * next_error
        for name, q in obj_qs.items():
            reference = params[name + '_q0']
            error = configuration_difference(reference, q)
            next_error = configuration_difference(reference, obj_qs_next[name])
            cost[name + '_state'] = cs.sqrt(params['object_state_cost_W']) * error
            terminal[name + '_state_terminal'] = cs.sqrt(params['object_state_cost_We']) * error
            cost[name + '_ctrl'] = cs.sqrt(params['object_ctrl_cost_W']) * next_error
        return (cost, terminal)

    def _goal_residuals(self, q_rob, obj_qs, params):
        for kind, name, target in self.goal_targets:
            q = q_rob[name] if kind == "robot" else obj_qs[name]
            reference = params[solver_name(kind, name, "goal_reference")]
            if isinstance(target, PositionTarget):
                error = q[:3] - reference
            else:
                error = pose_error(q[:7], reference[:7])
                if isinstance(target, ConfigurationTarget):
                    error = cs.vertcat(error, q[7:] - reference[7:])
            yield kind, name, target, error

    def get_task_cost(self, q_rob, obj_qs, params):
        if not self.goal_targets or self.terminal_goal.kso != "soft":
            return {}
        return {
            solver_name(kind, name, "goal_cost"):
                cs.sqrt(params[solver_name(kind, name, "goal_weight")]) * error
            for kind, name, target, error in self._goal_residuals(q_rob, obj_qs, params)
        }

    def get_terminal_constraints(self, q_rob, obj_qs, params):
        equalities, inequalities = {}, {}
        if not self.goal_targets or self.terminal_goal.kso != "hard":
            return equalities, inequalities
        for kind, name, target, error in self._goal_residuals(q_rob, obj_qs, params):
            tolerance = target.tolerance("kso")
            exact = np.flatnonzero(tolerance == 0).tolist()
            bounded = np.flatnonzero(tolerance > 0).tolist()
            if exact:
                equalities[solver_name(kind, name, "goal_exact")] = error[exact]
            if bounded:
                constraint_name = solver_name(kind, name, "goal_tolerance")
                inequalities[constraint_name] = dict(
                    name=constraint_name, expr=error[bounded],
                    lb=-tolerance[bounded], ub=tolerance[bounded],
                )
        return equalities, inequalities

    def _union_geometry_params(self, params):
        """Select one geometry: outgoing when active, otherwise incoming."""
        union = dict(params)
        for source in self.scene.active_patches:
            prefix = interface_name(source)
            use_incoming = 1 - cs.sum1(params[prefix + '_env_feat_oh'])
            fields = ('env_feat', 'self_feat') if source[0] == 'obj' else ('env_feat',)
            for field in fields:
                key = prefix + '_' + field
                union[key + '_oh'] = (params[key + '_oh']
                                     + use_incoming * params[key + '_incoming_oh'])
        return union

    def get_stage_constraints(self, q_rob, obj_qs, q_next_rob, obj_qs_next, params):
        """Constrain the union at x and outgoing-mode consistency from x to u."""
        equalities, inequalities, terminal_equalities = ({}, {}, {})
        robot_qs, object_qs = q_rob, obj_qs
        features = self.object_features.local_features({
            name: params[name + '_size']
            for name in self.object_features.object_names
        })
        fixed_qs = {name: params[name + '_q0'] for name in self.fixed_env_objects}
        sizes = {name: params[name + '_size'] for name in self.scene.objects}
        ee_poses = {("ee", rid, patch): pose
                    for rid, block in self.robots.items()
                    for (kind, patch), pose in block.kinematics.poses(robot_qs[rid]).items()
                    if kind == 'ee'}
        world_features = self.object_features.world_features(
            sizes, {**fixed_qs, **object_qs}, ee_poses)
        next_qs, next_objects = q_next_rob, obj_qs_next
        next_ee_poses = {("ee", rid, patch): pose
                         for rid, block in self.robots.items()
                         for (kind, patch), pose in block.kinematics.poses(next_qs[rid]).items()
                         if kind == 'ee'}
        next_features = self.object_features.world_features(
            sizes, {**fixed_qs, **next_objects}, next_ee_poses)

        union = self._union_geometry_params(params)

        def add(name, constr, local_vars, env_vars):
            result = constr.get_kin_constraints(local_vars, env_vars)
            if constr.has_kin_eqs:
                equalities[name + '_kin_eqs'] = result[0]
                terminal_equalities[name + '_kin_eqs'] = result[2]
            if constr.has_kin_ineqs:
                inequalities[name + '_kin_ineqs'] = dict(
                    name=name + '_kin_ineqs',
                    expr=result[1],
                    lb=0.0,
                    ub=100000000.0
                )
        for rid, block in self.robots.items():
            poses = block.kinematics.poses(robot_qs[rid])
            next_poses = block.kinematics.poses(next_qs[rid])
            for patch, constr in self.ee_constr_models[rid].items():
                prefix = interface_name(('ee', rid, patch))
                R, p = poses['ee', patch]
                Rnext, pnext = next_poses['ee', patch]
                oh = union[prefix + '_env_feat_oh']
                rotation, position, bounds = self.object_features.select(world_features, oh)
                face_Rnext, face_pnext, _ = self.object_features.select(next_features, oh)
                rv = rob_constr_factory.RobConstraintVars(
                    p_i_ee=p, R_i_ee=R, p_bnds=block.cfg.patch_ees[patch],
                    p_i_next_ee=pnext, R_i_next_ee=Rnext, active=cs.sum1(oh),
                    repeat=cs.sum1(params[prefix + '_env_feat_oh']),
                    sliding=params[prefix + '_type_oh'][CONTACT_TYPE_TO_COL[CONTACT_TYPE_SLIDING]],
                )
                ev = rob_constr_factory.EnvConstraintVars(
                    fp_p=position, fp_R=rotation, fp_bnds=bounds,
                    fp_p_next=face_pnext, fp_R_next=face_Rnext,
                )
                add('Stage_' + prefix, constr, rv, ev)
        for name, constr in self.obj_constr_models.items():
            self_oh = union[name + '_self_feat_oh']
            env_oh = union[name + '_env_feat_oh']
            feature = features[:, self.object_features.feature_slices[name]] @ self_oh
            R, p, bounds = self.object_features.face_pose(object_qs[name], feature)
            env_R, env_p, env_bounds = self.object_features.select(world_features, env_oh)
            Rnext, pnext, _ = self.object_features.face_pose(next_objects[name], feature)
            env_Rnext, env_pnext, _ = self.object_features.select(next_features, env_oh)
            ov = obj_constr_factory.ObjConstraintVars(
                p_i_obj=p, R_i_obj=R, p_bnds=bounds, active=cs.sum1(env_oh),
                p_i_next_obj=pnext, R_i_next_obj=Rnext,
                repeat=cs.sum1(params[name + '_env_feat_oh']),
                sliding=params[name + '_type_oh'][CONTACT_TYPE_TO_COL[CONTACT_TYPE_SLIDING]],
            )
            ev = obj_constr_factory.EnvConstraintVars(
                fp_p=env_p, fp_R=env_R, fp_bnds=env_bounds,
                fp_p_next=env_pnext, fp_R_next=env_Rnext,
            )
            add('Stage_' + name, constr, ov, ev)
        return (equalities, inequalities, terminal_equalities)

    def get_collision_constraints(self, robot_qs, obj_qs, params):
        states = {name: params[name + '_q0'] for name in self.fixed_env_objects}
        states.update(obj_qs)
        robot_poses = {
            capsule_name(rid, name): pose
            for rid, block in self.robots.items()
            for (kind, name), pose in block.kinematics.poses(robot_qs[rid]).items()
            if kind == 'collision'
        }
        distances = collision_distances(
            self.collision_idx.groups,
            robot_poses,
            object_poses(states),
            params
        )
        union = self._union_geometry_params(params)
        for group, pairs in self.collision_idx.groups.items():
            for pair, values in pairs.items():
                distances[group][pair] *= contact_collision_multiplier(
                    self.scene, self.object_features, group, values, union)
        return {
            pair + '_distance': dict(name=pair + '_distance', expr=distance, lb=0.0, ub=100000000.0)
            for group in distances.values()
            for pair, distance in group.items()
        }

    def _build_constraints(self, sym, *, hard_initial=False, N_nodes=None, include_collisions=True):
        """Apply initial, path, and terminal constraints and optional terminal goals."""
        params = self.params_syms
        x, u, p = (sym['x'], sym['u'], sym['p'])
        builder = ConstraintBuilder(x, u, p)
        current_qs = sym['q_rob']
        current_objects = sym['obj_qs']
        eqs, ineqs, teqs = self.get_stage_constraints(
            current_qs,
            current_objects,
            sym['q_next_rob'],
            sym['obj_qs_next'],
            params
        )
        # Only constrain x norms. Dynamics transfers the next node's norm to u,
        # so also constraining u would duplicate equalities across intervals.
        norms = {
            name + '_quaternion_norm': cs.dot(q[3:7], q[3:7]) - 1
            for name, q in sym['x_dict'].items()
        }
        for role in ('initial', 'path', 'terminal'):
            if hard_initial and (role == 'initial' or (role == 'terminal' and N_nodes == 1)):
                # Full x0 bounds already enforce this; avoid dependent equalities.
                continue
            builder.equalities(norms, role)
        roles = ('initial', 'path')
        for role in roles:
            builder.equalities(eqs, role)
            builder.inequalities(ineqs, role)
        collisions = (self.get_collision_constraints(current_qs, current_objects, params)
                      if include_collisions else {})
        for role in roles:
            builder.inequalities(collisions, role)
        builder.equalities(teqs, 'terminal')
        builder.inequalities(ineqs, 'terminal')
        builder.inequalities(collisions, 'terminal')
        goal_eqs, goal_ineqs = self.get_terminal_constraints(current_qs, current_objects, params)
        builder.equalities(goal_eqs, 'terminal')
        builder.inequalities(goal_ineqs, 'terminal')
        return builder.specs

    def _build_box_constraints(self):
        bounds = []
        for rid, block in self.robots.items():
            offset = self.x_layout.slices[robot_name(rid, 'q_rob')].start
            bounds.append(LinearBound(
                robot_name(rid, 'joint_pos'),
                'x',
                list(range(offset + 7, offset + block.nq)),
                block.cfg.q_min,
                block.cfg.q_max
            ))
        for name in self.env_objects:
            offset = self.x_layout.slices[name].start
            bounds.append(LinearBound(
                name + '_pos',
                'x',
                list(range(offset, offset + 3)),
                np.array([-20.0, -20.0, -10.0]),
                np.array([20.0, 20.0, 10.0])
            ))
        return bounds

    def build_ocp_description(self, N_nodes=None, *, hard_initial=False, include_collisions=True):
        sym = self._build_stage_symbols()
        f_impl, f_expl, f_discr = self._build_dynamics(sym)
        r_path, r_terminal = self._build_costs(sym)
        constraints = self._build_constraints(sym, hard_initial=hard_initial, N_nodes=N_nodes,
                                              include_collisions=include_collisions)
        return OCPDescription(
            x=sym['x'],
            u=sym['u'],
            p=sym['p'],
            x_dict=sym['x_dict'],
            u_dict=sym['u_dict'],
            p_dict=self.params_syms,
            f_expl=f_expl,
            f_impl=f_impl,
            f_discr=f_discr,
            r_path=r_path,
            r_terminal=r_terminal,
            constraints=constraints,
            linear_bounds=self._build_box_constraints(),
            state_coordinates="configuration",
        )
