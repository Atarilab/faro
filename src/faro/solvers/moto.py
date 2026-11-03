from __future__ import annotations

import moto
import numpy as np
from typing import Dict, Any
from dataclasses import dataclass
from numbers import Integral
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

from faro.ocp.formulation.to import ToFormulation
from faro.ocp.moto_states import normalize_pose_trajectory
from faro.utils.contact_type import CONTACT_TYPE_SLIDING, require_supported_contact_type

@dataclass
class MotoBuildConfig:
    # Both native solvers share generated artifacts under code_gen_path/name.
    name: str = "default"
    code_gen_path: str = "moto_gen"
    num_modes: int = 5
    nodes_per_mode: int = 5
    time_per_mode: float = 1.0
    tol: float = 1e-4
    free_init_state: bool = False
    use_gauss_newton_costs: bool = True
    collision_update_batch_size: int = 1
    restoration_enabled: bool = True
    presolve_max_iters: int = 10
    feasibility_tol: float = 0.05
    n_job: int = 6
    # Check recovered Newton directions against the full KKT equations.
    validate_direction: bool = True

    def __post_init__(self):
        if (
            not isinstance(self.name, str)
            or self.name in ("", ".", "..")
            or "/" in self.name
            or "\\" in self.name
            or "\0" in self.name
        ):
            raise ValueError("name must be a nonempty single folder component")
        if (
            isinstance(self.n_job, bool)
            or not isinstance(self.n_job, Integral)
            or self.n_job < 1
        ):
            raise ValueError("n_job must be a positive integer")
        if (
            isinstance(self.presolve_max_iters, bool)
            or not isinstance(self.presolve_max_iters, Integral)
            or self.presolve_max_iters < 0
        ):
            raise ValueError("presolve_max_iters must be a nonnegative integer")


class Moto:
    def __init__(self, formulation: ToFormulation, 
                       build_cfg: MotoBuildConfig):
        self.form = formulation
        self.form.use_gauss_newton_costs = build_cfg.use_gauss_newton_costs

        self.M = build_cfg.num_modes
        self.NpM = build_cfg.nodes_per_mode
        self.N = int(self.M * self.NpM)
        self.build_cfg = build_cfg
        self.codegen = moto.codegen_context(Path(build_cfg.code_gen_path) / build_cfg.name)

        self.named_constraints = {}

    @staticmethod
    def _is_node_term(expr):
        """Return whether an expression depends only on node state/parameters."""
        interval_fields = {
            moto.field.field___u,
            moto.field.field___y,
            moto.field.field___l,
        }
        try:
            return all(
                arg.__sym__.field not in interval_fields
                for arg in expr.in_args
            )
        except (AttributeError, TypeError):
            return False

    @classmethod
    def _add_stage_term(cls, stage, expr):
        if cls._is_node_term(expr):
            # Evaluate node-only path terms at x_1, ..., x_N using each
            # preceding interval's parameters; x_0 has no path term.
            stage.ed.add(expr)
        else:
            stage.add(expr)

    def build_ocp(self):
        ocp = moto.stage(codegen=self.codegen)

        dynamics = self.form.get_dynamics()
        ocp.add(dynamics)
        self.named_constraints["dynamics"] = ("dyn", dynamics)

        for name, eq in self.form.get_object_fma().items():
            self._add_stage_term(ocp, eq)
            self.named_constraints[name] = ("eq", eq)

        running_costs = list(self.form.get_cost().values())
        for cost in running_costs:
            self._add_stage_term(ocp, cost)

        eqs, ineqs = self.form.get_global_path_constraints()
        for name, eq in eqs.items():
            self._add_stage_term(ocp, eq)
            self.named_constraints[name] = ("eq", eq)
        for name, ineq in ineqs.items():
            self._add_stage_term(ocp, ineq)
            self.named_constraints[name] = ("ineq", ineq)

        eqs, ineqs = self.form.get_stage_constraints()
        for name, eq in eqs.items():
            self._add_stage_term(ocp, eq)
            self.named_constraints[name] = ("eq", eq)
        for name, ineq in ineqs.items():
            self._add_stage_term(ocp, ineq)
            self.named_constraints[name] = ("ineq", ineq)

        ocp_coll = ocp.copy()
        ineqs = self.form.get_collision_constraints()
        for name, ineq in ineqs.items():
            self._add_stage_term(ocp_coll, ineq)
            self.named_constraints[name] = ("ineq", ineq)

        self.ocp = ocp
        self.ocp_coll = ocp_coll
        self.running_costs = running_costs
        self.terminal_exprs = list(self.form.get_cost_term().values())

        eqs, ineqs = self.form.get_terminal_constraints()
        for kind, constraints in (("eq", eqs), ("ineq", ineqs)):
            for name, constraint in constraints.items():
                self.terminal_exprs.append(constraint)
                self.named_constraints[name] = (kind, constraint)

    def build_sqp(self, interface_tups, coll=False):
        sqp = moto.sqp(n_job=int(self.build_cfg.n_job), codegen=self.codegen)

        if coll:
            ocp = self.ocp_coll
        else:
            ocp = self.ocp

        expected = set(self.form.scene.active_patches)
        if set(interface_tups) != expected:
            raise ValueError(f"Contact schedule keys must match the scene interfaces: {expected}")
        if any(len(values) != self.M for values in interface_tups.values()):
            raise ValueError(f"Every contact schedule must have {self.M} modes")

        def create_phase_problem(mode_idx: int):
            disabled = []
            for key, contacts in interface_tups.items():
                value = contacts[mode_idx]
                if value:
                    require_supported_contact_type(value[-1], context=f"{key}[mode={mode_idx}]")
                alternatives = getattr(self.form, "contact_type_constraints", {}).get(key)
                if alternatives is not None:
                    unused = "sticking" if value and value[-1] == CONTACT_TYPE_SLIDING else "sliding"
                    disabled.extend(alternatives[unused])
                if contacts[mode_idx] == ():
                    if key[0] == "ee":
                        disabled.append(self.form.sym["rob_Ws"][key[1]][key[2]])
                    else:
                        disabled.append(self.form.sym["obj_Ws"][key[1]])
            return ocp.copy(disable=disabled)

        for mode_idx in range(self.M):
            phase = create_phase_problem(mode_idx)
            sqp.stages.extend([phase.copy() for _ in range(self.NpM)])

        sqp.ed.add(self.terminal_exprs)

        # Surface Moto failures to the caller instead of converting arbitrary
        # exceptions into an ambiguous solver result.
        sqp.settings.no_except = False
        sqp.settings.initial_state = (
            moto.sqp.initial_state_mode.optimized
            if self.build_cfg.free_init_state
            else moto.sqp.initial_state_mode.fixed
        )
        sqp.settings.ipm.mu0 = 1.0
        sqp.settings.ipm.mu_method = (
            moto.sqp.adaptive_mu_t.mehrotra_predictor_corrector
        )
        sqp.settings.ipm_conditional_corrector = True
        sqp.settings.prim_tol = self.build_cfg.tol
        sqp.settings.dual_tol = 1e0
        sqp.settings.comp_tol = self.build_cfg.tol
        sqp.settings.rf.max_iters = 2
        sqp.settings.regularization.validate_direction = self.build_cfg.validate_direction
        sqp.settings.ls.update_alpha_dual = False
        sqp.settings.ls.primal_gamma = 1e-4
        sqp.settings.ls.method = moto.sqp.search_method_filter
        sqp.settings.restoration.enabled = self.build_cfg.restoration_enabled
        sqp.settings.restoration.max_iter = 100
        # Penalize hard contact and force-balance violations during restoration.
        sqp.settings.restoration.rho_eq = 1e2
        sqp.settings.restoration.rho_ineq = 1e2

        sqp.settings.ls.on_failure = (
            moto.sqp.linesearch_setting.on_failure_action_accept_fallback
        )

        return sqp

    def _validated_warm_start(self, x0_dict, u0_dict):
        """Validate and copy every owned block before touching solver storage."""
        state_shapes = {
            "robots": {rid: (self.N + 1, layout.dim)
                       for rid, layout in self.form.robot_layouts.items()},
            "objects": {name: (self.N + 1, self.form.object_layout.dim)
                        for name in self.form.env_objects},
        }
        control_shapes = {
            "robots": {rid: {
                "acceleration": (self.N, robot.nv),
                "wrenches": {patch: (self.N, 6) for patch in robot.cfg.patch_ees},
            } for rid, robot in self.form.robots.items()},
            "objects": {name: (self.N, self.form.obj_n_u) for name in self.form.env_objects},
            "t_scale": (self.N,),
        }

        def validate(values, shapes, label):
            if not isinstance(values, Mapping) or set(values) != set(shapes):
                actual = set(values) if isinstance(values, Mapping) else set()
                raise ValueError(f"{label} keys: missing {set(shapes) - actual}, unexpected {actual - set(shapes)}")
            result = {}
            for name, shape in shapes.items():
                path = f"{label}.{name}"
                if isinstance(shape, dict):
                    result[name] = validate(values[name], shape, path)
                    continue
                array = np.asarray(values[name], dtype=float)
                if array.shape != shape:
                    raise ValueError(f"{path}: expected shape {shape}, got {array.shape}")
                if not np.isfinite(array).all():
                    raise ValueError(f"{path}: trajectory must contain only finite values")
                result[name] = array.copy()
            return result

        states = validate(x0_dict, state_shapes, "Physical state")
        controls = validate(u0_dict, control_shapes, "Control")
        for section in states:
            states[section] = {
                name: normalize_pose_trajectory(values, name=f"{section}.{name}")
                for name, values in states[section].items()
            }
        return states, controls

    def _state_blocks(self, next_state=False):
        """Yield ownership section, owner, symbol and physical storage slice."""
        keys = ("qn_rob", "vn_rob", "rn_rob", "hn_rob") if next_state else (
            "q_rob", "v_rob", "r_rob", "h_rob")
        for rid, layout in self.form.robot_layouts.items():
            for key, indices in zip(keys, (layout.q, layout.v, layout.r, layout.h)):
                yield "robots", rid, self.form.sym[key][rid], indices
        layout = self.form.object_layout
        qkey, vkey = ("obj_qns", "obj_Vns") if next_state else ("obj_qs", "obj_Vs")
        for name in self.form.env_objects:
            yield "objects", name, self.form.sym[qkey][name], layout.q
            yield "objects", name, self.form.sym[vkey][name], layout.v

    def set_warm_start(self, sqp, x0_dict, u0_dict):
        """Set N+1 physical states and N controls, including both stage ends."""
        states, controls = self._validated_warm_start(x0_dict, u0_dict)
        nodes = sqp.nodes
        if len(nodes) != self.N:
            raise ValueError(f"Expected {self.N} solver intervals, got {len(nodes)}")
        for k, node in enumerate(nodes):
            for next_state in (False, True):
                for section, name, symbol, indices in self._state_blocks(next_state):
                    node.value[symbol] = states[section][name][k + int(next_state), indices]
            node.value[self.form.sym["t_scale"]] = controls["t_scale"][k]
            for rid, robot in self.form.robots.items():
                control = controls["robots"][rid]
                node.value[self.form.sym["vdot_rob"][rid]] = control["acceleration"][k]
                for patch in robot.cfg.patch_ees:
                    wrench = self.form.sym["rob_Ws"][rid][patch]
                    if node.prob.is_active(wrench.sym):
                        node.value[wrench] = control["wrenches"][patch][k]
            for name in self.form.env_objects:
                control = controls["objects"][name]
                node.value[self.form.sym["obj_Vdots"][name]] = control[k, :6]
                wrench = self.form.sym["obj_Ws"][name]
                if node.prob.is_active(wrench.sym):
                    node.value[wrench] = control[k, 6:]

    def update_params_const(self, sqp, constant_params):
        def set_all(data: moto.sqp.data_type):
            for name, value in constant_params.items():
                param = self.form.params[name]
                if data.prob.is_active(param.sym):
                    data.value[param] = value

        for data in sqp.nodes:
            set_all(data)

    def update_params_traj(self, sqp, trajectory_params):
        def make_traj_param_cb(param, trajectory):
            traj_idx = 0
            def callback(data):
                nonlocal traj_idx
                if data.prob.is_active(param.sym):
                    if trajectory.ndim == 1:
                        data.value[param] = float(trajectory[traj_idx])
                    else:
                        value = np.ascontiguousarray(
                            trajectory[:, traj_idx],
                            dtype=np.float64
                        )
                        data.value[param] = value
                traj_idx += 1

            return callback

        for name, param_val_traj in trajectory_params.items():
            cb = make_traj_param_cb(self.form.params[name], param_val_traj)
            for data in sqp.nodes:
                cb(data)

    def get_results(self, sqp) -> Dict[str, Any]:
        """Extract independent arrays for each robot and each shared object."""
        states = {
            "robots": {rid: np.empty((self.N + 1, layout.dim))
                       for rid, layout in self.form.robot_layouts.items()},
            "objects": {name: np.empty((self.N + 1, self.form.object_layout.dim))
                        for name in self.form.env_objects},
        }
        accelerations = {rid: np.empty((self.N, robot.nv)) for rid, robot in self.form.robots.items()}
        robot_wrenches = {rid: {patch: np.zeros((self.N, 6)) for patch in robot.cfg.patch_ees}
                          for rid, robot in self.form.robots.items()}
        object_accelerations = {name: np.empty((self.N, 6)) for name in self.form.env_objects}
        object_wrenches = {name: np.zeros((self.N, 6)) for name in self.form.env_objects}
        t_scale = np.empty(self.N)
        nodes = sqp.nodes
        if len(nodes) != self.N:
            raise ValueError(f"Expected {self.N} solver intervals, got {len(nodes)}")
        for k, node in enumerate(nodes):
            for section, name, symbol, indices in self._state_blocks():
                states[section][name][k, indices] = np.asarray(node.value[symbol]).reshape(-1)
            if k == self.N - 1:
                for section, name, symbol, indices in self._state_blocks(next_state=True):
                    states[section][name][self.N, indices] = np.asarray(node.value[symbol]).reshape(-1)
            t_scale[k] = np.asarray(node.value[self.form.sym["t_scale"]]).item()
            for rid in self.form.robots:
                accelerations[rid][k] = np.asarray(node.value[self.form.sym["vdot_rob"][rid]]).reshape(-1)
                for patch, values in robot_wrenches[rid].items():
                    wrench = self.form.sym["rob_Ws"][rid][patch]
                    if node.prob.is_active(wrench.sym):
                        values[k] = np.asarray(node.value[wrench]).reshape(-1)
            for name in self.form.env_objects:
                object_accelerations[name][k] = np.asarray(node.value[self.form.sym["obj_Vdots"][name]]).reshape(-1)
                wrench = self.form.sym["obj_Ws"][name]
                if node.prob.is_active(wrench.sym):
                    object_wrenches[name][k] = np.asarray(node.value[wrench]).reshape(-1)
        controls = {
            "robots": {rid: {"acceleration": accelerations[rid], "wrenches": robot_wrenches[rid]}
                       for rid in self.form.robots},
            "objects": {name: np.hstack((object_accelerations[name], object_wrenches[name]))
                        for name in self.form.env_objects},
            "t_scale": t_scale,
        }
        return {
            "x0_dict": deepcopy(states), "u0_dict": deepcopy(controls),
            "robot_xs": states["robots"], "obj_xs": states["objects"],
            "vdot_rob": accelerations, "rob_Ws": robot_wrenches,
            "obj_Vdots": object_accelerations, "obj_Ws": object_wrenches,
            "t_scale": t_scale,
        }

    def get_param_traj(self, sqp, param_name: str):
        param_traj = []
        def make_const_param_cb(param):
            def callback(data: moto.sqp.data_type):
                param_traj.append(data.value[param])
            return callback

        cb = make_const_param_cb(self.form.params[param_name])
        for data in sqp.nodes:
            cb(data)
        
        return np.array(param_traj)

    @staticmethod
    def _scalar_violation(kind: str, raw_scalar: float) -> float:
        if kind in ("eq", "dyn"):
            return abs(float(raw_scalar))
        elif kind == "ineq":
            return max(float(raw_scalar), 0.0)
        else:
            raise ValueError(f"Unknown constraint kind: {kind}")

    def collect_constraint_violations(self, sqp, early_stop=None):
        """
        Returns one record per scalar residual entry.

        Inequality values include Moto's IPM slack. These are solver residual
        diagnostics, not physical inequality violations or collision depths.
        """
        rows = []
        node_idx = 0

        def inspect(node: moto.sqp.data_type):
            nonlocal node_idx

            for name, (kind, f) in self.named_constraints.items():
                # Endpoint lowering changes function identity. Moto's data
                # lookup resolves the original handle to its runtime copy.
                try:
                    fad = node.data(f)
                except IndexError as exc:
                    if str(exc) != "function is not active in this node":
                        raise
                    continue
                raw_vec = np.array(fad.v, dtype=float).reshape(-1)

                for i, raw in enumerate(raw_vec):
                    viol = self._scalar_violation(kind, raw)
                    rows.append({
                        "node": node_idx,
                        "name": name,
                        "idx": i,
                        "kind": kind,
                        "raw": float(raw),
                        "violation": float(viol),
                    })

            node_idx += 1

        nodes = sqp.nodes
        if early_stop is not None:
            nodes = nodes[:early_stop]
        for node in nodes:
            inspect(node)

        return rows

    def filtered_constraint_violations(
        self,
        sqp,
        tol=1e-8,
        early_stop=None,
        kind_filter=None,
        name_filter=None,
    ):
        rows = self.collect_constraint_violations(sqp, early_stop=early_stop)

        if kind_filter is not None:
            if isinstance(kind_filter, str):
                allowed = {kind_filter}
            else:
                allowed = set(kind_filter)
            rows = [r for r in rows if r["kind"] in allowed]

        if name_filter is not None:
            rows = [r for r in rows if name_filter in r["name"]]

        rows = [r for r in rows if r["violation"] > tol]
        return rows

    def sum_constraint_violations(
        self,
        sqp,
        tol=1e-8,
        early_stop=None,
        kind_filter=None,
        name_filter=None,
    ):
        rows = self.filtered_constraint_violations(
            sqp,
            tol=tol,
            early_stop=early_stop,
            kind_filter=kind_filter,
            name_filter=name_filter,
        )
        return sum(r["violation"] for r in rows)


    def max_constraint_violation(
        self,
        sqp,
        tol=1e-8,
        early_stop=None,
        kind_filter=None,
        name_filter=None,
    ):
        rows = self.filtered_constraint_violations(
            sqp,
            tol=tol,
            early_stop=early_stop,
            kind_filter=kind_filter,
            name_filter=name_filter,
        )
        return max((r["violation"] for r in rows), default=0.0)


    def print_top_constraint_violations(
        self,
        sqp,
        top_n=20,
        tol=1e-8,
        early_stop=None,
        kind_filter=None,
        name_filter=None,
    ):
        rows = self.filtered_constraint_violations(
            sqp,
            tol=tol,
            early_stop=early_stop,
            kind_filter=kind_filter,
            name_filter=name_filter,
        )

        rows.sort(key=lambda r: r["violation"], reverse=True)

        print(f"Top {min(top_n, len(rows))} scalar constraint violations")
        print(
            f"{'node':>4s} | "
            f"{'kind':<4s} | "
            f"{'name':<30} | "
            f"{'idx':>3s} | "
            f"{'raw':>12s} | "
            f"{'violation':>12s}"
        )
        for r in rows[:top_n]:
            print(
                f"{r['node']:4d} | "
                f"{r['kind']:<4s} | "
                f"{r['name']:<30} | "
                f"{r['idx']:3d} | "
                f"{r['raw']:+.6e} | "
                f"{r['violation']:.6e}"
            )

        return rows[:top_n]
