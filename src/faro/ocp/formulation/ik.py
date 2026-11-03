"""Inverse kinematics for all robots and shared objects in a scene."""

import casadi as cs
from faro.configs.identifiers import robot_name, interface_name, capsule_name
from faro.ocp.robot_blocks import setup_robot_blocks, BlockLayout
from faro.ocp.kinematics import q_integrate
from faro.ocp.object_features import ObjectFeatures
from faro.ocp.collision import collision_distances, object_poses, contact_collision_multiplier
from faro.ocp.ocp_description import OCPDescription, ConstraintBuilder, LinearBound
from faro.ocp.ik_contact import rob_constr_factory, obj_constr_factory


def _stack(values):
    return cs.vertcat(*values) if values else cs.SX.zeros(0, 1)


class IkFormulation:
    """Contact geometry and regularization on a single Acados terminal state.

    This formulation has no controls, dynamics or path costs.
    """

    obj_n_q, obj_n_v = (7, 6)

    def __init__(self, scene, collision_index, robot_collision_models=None, verbose=False):
        self.scene, self.verbose = (scene, verbose)
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
        self.q_integrate = q_integrate()
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
            add(robot_name(rid, 'rob_state_cost_We'), block.nv)
            for patch in block.cfg.patch_ees:
                prefix = interface_name(('ee', rid, patch))
                add(prefix + '_env_feat_oh', self.total_env_features)
        for name, cfg in {**self.fixed_env_objects, **self.env_objects}.items():
            add(name + '_q0', 7)
            add(name + '_size', len(cfg.half_extents))
        for name, cfg in self.env_objects.items():
            add(name + '_self_feat_oh', cfg.num_features)
            add(name + '_env_feat_oh', self.total_env_features)
        for name in self.coll_pair_names:
            add(name, 9)
        add('object_state_cost_We', 6)
        return params

    def _build_stage_symbols(self):
        robot_dqs = {
            rid: cs.SX.sym(robot_name(rid, 'dq_rob'), b.nv)
            for rid, b in self.robots.items()
        }
        obj_dqs = {name: cs.SX.sym(name + '_dq', 6) for name in self.env_objects}
        x_dict = {robot_name(rid, 'dq_rob'): dq for rid, dq in robot_dqs.items()}
        x_dict.update(obj_dqs)
        symbols = {'dq_rob': robot_dqs, 'obj_dqs': obj_dqs}
        self.x_layout = BlockLayout.from_symbols(x_dict)
        symbols.update(
            x=_stack(list(x_dict.values())),
            u=cs.SX.zeros(0, 1),
            p=_stack(list(self.params_syms.values())),
            x_dict=x_dict,
            u_dict={}
        )
        return symbols

    def _robot_qs(self, dqs, params):
        return {
            rid: block.integrate(params[robot_name(rid, 'q0')], dqs[rid])
            for rid, block in self.robots.items()
        }

    def _object_qs(self, dqs, params):
        return {
            name: self.q_integrate(params[name + '_q0'], dqs[name])
            for name in self.env_objects
        }

    def _build_cost(self, sym):
        params = self.params_syms
        residuals = []
        for rid in self.robots:
            reference = cs.vertcat(cs.SX.zeros(6), params[robot_name(rid, 'q0')][7:])
            residuals.append(cs.sqrt(params[robot_name(rid, 'rob_state_cost_We')])
                             * (sym['dq_rob'][rid] - reference))
        for dq in sym['obj_dqs'].values():
            residuals.append(cs.sqrt(params['object_state_cost_We']) * dq)
        return cs.Function('r_terminal', [sym['x'], sym['u'], sym['p']], [_stack(residuals)])

    def get_stage_constraints(self, dq_rob, obj_dqs, params):
        """Assemble robot contacts independently and shared object contacts once."""
        equalities, inequalities = ({}, {})
        robot_qs = self._robot_qs(dq_rob, params)
        object_qs = self._object_qs(obj_dqs, params)
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

        def add(name, constr, local_vars, env_vars):
            result = constr.get_kin_constraints(local_vars, env_vars)
            if constr.has_kin_eqs:
                equalities[name + '_kin_eqs'] = result[0]
            if constr.has_kin_ineqs:
                inequalities[name + '_kin_ineqs'] = dict(
                    name=name + '_kin_ineqs',
                    expr=result[1],
                    lb=0.0,
                    ub=100000000.0
                )
        for rid, block in self.robots.items():
            poses = block.kinematics.poses(robot_qs[rid])
            for patch, constr in self.ee_constr_models[rid].items():
                prefix = interface_name(('ee', rid, patch))
                oh = params[prefix + '_env_feat_oh']
                rotation, position, bounds = self.object_features.select(world_features, oh)
                R, p = poses['ee', patch]
                rv = dict(p_i_ee=p, R_i_ee=R, p_bnds=block.cfg.patch_ees[patch], active=cs.sum1(oh))
                ev = dict(fp_p=position, fp_R=rotation, fp_bnds=bounds)
                add(
                    'Stage_' + prefix,
                    constr,
                    rob_constr_factory.RobConstraintVars(**rv),
                    rob_constr_factory.EnvConstraintVars(**ev)
                )
        for name, constr in self.obj_constr_models.items():
            self_oh, env_oh = (params[name + '_self_feat_oh'], params[name + '_env_feat_oh'])
            feature = features[:, self.object_features.feature_slices[name]] @ self_oh
            R, p, bounds = self.object_features.face_pose(object_qs[name], feature)
            env_R, env_p, env_bounds = self.object_features.select(world_features, env_oh)
            ov = dict(p_i_obj=p, R_i_obj=R, p_bnds=bounds, active=cs.sum1(env_oh))
            ev = dict(fp_p=env_p, fp_R=env_R, fp_bnds=env_bounds)
            add(
                'Stage_' + name,
                constr,
                obj_constr_factory.ObjConstraintVars(**ov),
                rob_constr_factory.EnvConstraintVars(**ev)
            )
        return (equalities, inequalities)

    def get_collision_constraints(self, robot_dqs, obj_dqs, params):
        robot_qs = self._robot_qs(robot_dqs, params)
        states = {name: params[name + '_q0'] for name in self.fixed_env_objects}
        states.update(self._object_qs(obj_dqs, params))
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
        for group, pairs in self.collision_idx.groups.items():
            for pair, values in pairs.items():
                distances[group][pair] *= contact_collision_multiplier(
                    self.scene, self.object_features, group, values, params)
        return {
            pair + '_distance': dict(name=pair + '_distance', expr=distance, lb=0.0, ub=100000000.0)
            for group in distances.values()
            for pair, distance in group.items()
        }

    def _build_constraints(self, sym):
        """Constrain the sole configuration at Acados's zero-horizon terminal stage."""
        params = self.params_syms
        x, u, p = (sym['x'], sym['u'], sym['p'])
        builder = ConstraintBuilder(x, u, p)
        current_dqs = sym['dq_rob']
        current_objects = sym['obj_dqs']
        eqs, ineqs = self.get_stage_constraints(current_dqs, current_objects, params)
        builder.equalities(eqs, 'terminal')
        builder.inequalities(ineqs, 'terminal')
        collisions = self.get_collision_constraints(current_dqs, current_objects, params)
        builder.inequalities(collisions, 'terminal')
        return builder.specs

    def _build_box_constraints(self):
        bounds = []
        for rid, block in self.robots.items():
            offset = self.x_layout.slices[robot_name(rid, 'dq_rob')].start
            bounds.append(LinearBound(
                robot_name(rid, 'joint_pos'),
                'x',
                list(range(offset + 6, offset + block.nv)),
                block.cfg.q_min,
                block.cfg.q_max
            ))
        return bounds

    def build_ocp_description(self):
        sym = self._build_stage_symbols()
        r_terminal = self._build_cost(sym)
        constraints = self._build_constraints(sym)
        return OCPDescription(
            x=sym['x'],
            u=sym['u'],
            p=sym['p'],
            x_dict=sym['x_dict'],
            u_dict=sym['u_dict'],
            p_dict=self.params_syms,
            f_expl=None,
            f_impl=None,
            f_discr=None,
            r_path=None,
            r_terminal=r_terminal,
            constraints=constraints,
            linear_bounds=self._build_box_constraints()
        )
