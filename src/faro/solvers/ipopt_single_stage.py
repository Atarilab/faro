from __future__ import annotations

from faro.configs.identifiers import robot_name
from faro.solvers._owned_results import owned_warm_start, validate_owned_warm_start

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import casadi as cs
import numpy as np
import pinocchio as pin

from faro.ocp.ocp_description import OCPDescription


def _as_1d_array(value: Any) -> np.ndarray:
    if isinstance(value, (float, int)):
        return np.asarray([float(value)], dtype=float)
    try:
        if isinstance(value, (cs.DM, cs.SX, cs.MX)):
            value = value.full()
    except Exception:
        pass
    return np.asarray(value, dtype=float).reshape(-1)


@dataclass
class IpoptSingleStageConfig:
    name: str = "ipopt_single_stage"
    max_iter: int = 500
    tol: float = 1e-4
    acceptable_tol: float = 1e-3
    print_level: int = 5
    print_time: bool = True
    expand: bool = True
    fix_state: bool = False
    error_on_fail: bool = False
    ipopt_options: Dict[str, Any] = field(default_factory=dict)
    casadi_options: Dict[str, Any] = field(default_factory=dict)


class _SingleStageOptiCore:
    def __init__(self, ocp: OCPDescription, cfg: IpoptSingleStageConfig):
        self.ocp = ocp
        self.cfg = cfg

        self.opti = cs.Opti()
        self.x = self.opti.variable(int(ocp.x.size1()))
        self.u = self.opti.variable(int(ocp.u.size1()))
        self.p = self.opti.parameter(int(ocp.p.size1()))
        self.x_ref = self.opti.parameter(int(ocp.x.size1()))

        self._nx = int(ocp.x.size1())
        self._nu = int(ocp.u.size1())
        self._np = int(ocp.p.size1())

        self._p_buf = np.zeros(self._np, dtype=float)
        self._x_ref = np.zeros(self._nx, dtype=float)
        self._x_val = np.zeros(self._nx, dtype=float)
        self._u_val = np.zeros(self._nu, dtype=float)

        self._objective = self._build_objective()
        self._objective_fun = cs.Function(
            f"{cfg.name}_objective",
            [self.x, self.u, self.p],
            [self._objective],
        )
        self.opti.minimize(self._objective)

        if cfg.fix_state and self._nx > 0:
            self.opti.subject_to(self.x == self.x_ref)

        self._build_constraints()
        self._configure_solver()
        self.apply_parameters()
        self.opti.set_initial(self.x, self._x_val)
        self.opti.set_initial(self.u, self._u_val)

        self.last_status = "not_solved"
        self.last_exception: Optional[str] = None
        self._solution = None
        self._last_raw_stats: Dict[str, Any] = {}

    def _configure_solver(self) -> None:
        p_opts = {"expand": self.cfg.expand, **self.cfg.casadi_options}
        s_opts = {
            "max_iter": int(self.cfg.max_iter),
            "tol": float(self.cfg.tol),
            "acceptable_tol": float(self.cfg.acceptable_tol),
            "print_level": int(self.cfg.print_level),
        }
        s_opts.update(self.cfg.ipopt_options)
        if not self.cfg.print_time:
            p_opts["print_time"] = False
        self.opti.solver("ipopt", p_opts, s_opts)

    def _build_objective(self) -> cs.MX:
        terms = []
        if self.ocp.r_path is not None:
            terms.append(self.ocp.r_path(self.x, self.u, self.p))
        if self.ocp.r_terminal is not None:
            terms.append(self.ocp.r_terminal(self.x, self.u, self.p))
        if not terms:
            return cs.MX(0)
        return cs.sumsqr(cs.vertcat(*terms))

    def _build_constraints(self) -> None:
        for constr in self.ocp.constraints:
            expr = constr.fun(self.x, self.u, self.p)
            lb, ub = self._broadcast_bounds(constr.lb, constr.ub, int(expr.shape[0]))
            self.opti.subject_to(self.opti.bounded(lb, expr, ub))

        for bound in self.ocp.linear_bounds:
            var = self.x if bound.var_type == "x" else self.u
            expr = var[bound.indices]
            lb, ub = self._broadcast_bounds(bound.lb, bound.ub, len(bound.indices))
            self.opti.subject_to(self.opti.bounded(lb, expr, ub))

    @staticmethod
    def _broadcast_bounds(lb: Any, ub: Any, dim: int) -> Tuple[cs.DM, cs.DM]:
        lb_arr = _as_1d_array(lb)
        ub_arr = _as_1d_array(ub)
        if lb_arr.size == 1 and dim > 1:
            lb_arr = np.full(dim, lb_arr[0], dtype=float)
        if ub_arr.size == 1 and dim > 1:
            ub_arr = np.full(dim, ub_arr[0], dtype=float)
        if lb_arr.size != dim or ub_arr.size != dim:
            raise ValueError(
                f"Constraint bound dimension mismatch: expr dim={dim}, "
                f"lb dim={lb_arr.size}, ub dim={ub_arr.size}"
            )
        return cs.DM(lb_arr), cs.DM(ub_arr)

    def _block_slices(self, field: str) -> List[Tuple[str, int, int]]:
        blocks = self.ocp.x_dict if field == "x" else self.ocp.u_dict if field == "u" else self.ocp.p_dict
        offset = 0
        slices = []
        for name, sym in blocks.items():
            dim = int(sym.shape[0])
            slices.append((name, offset, offset + dim))
            offset += dim
        return slices

    def _block_slice(self, field: str, name: str) -> Tuple[int, int]:
        for block_name, i0, i1 in self._block_slices(field):
            if block_name == name:
                return i0, i1
        known = [block_name for block_name, _, _ in self._block_slices(field)]
        raise KeyError(f"Unknown {field} block '{name}'. Known: {known}")

    def _stack_block(self, field: str, data: Dict[str, Iterable[float]]) -> np.ndarray:
        from faro.ocp.robot_blocks import BlockLayout
        sizes = {name: i1 - i0 for name, i0, i1 in self._block_slices(field)}
        if set(data) != set(sizes):
            raise KeyError(f"{field} blocks differ: missing {set(sizes) - set(data)}, unexpected {set(data) - set(sizes)}")
        return BlockLayout(sizes).pack(data)

    def _coerce_single_parameter_block(self, name: str, value: Any) -> np.ndarray:
        i0, i1 = self._block_slice("p", name)
        dim = i1 - i0

        try:
            if isinstance(value, (cs.DM, cs.SX, cs.MX)):
                value = value.full()
        except Exception:
            pass

        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        if arr.ndim == 1:
            flat = arr.reshape(-1)
            if flat.size != dim:
                raise ValueError(f"Parameter '{name}' size {flat.size} != expected {dim}")
            return flat
        if arr.ndim == 2:
            if arr.shape[0] == dim:
                return arr[:, -1].reshape(-1)
            if arr.shape[1] == dim:
                return arr[-1, :].reshape(-1)
        squeezed = np.squeeze(arr)
        if squeezed.ndim == 1 and squeezed.size == dim:
            return squeezed.reshape(-1)
        raise ValueError(f"Parameter '{name}' shape {arr.shape} cannot be coerced to dim {dim}")

    def _apply_param_block_updates(self, updates: Dict[str, Any]) -> None:
        for name, value in updates.items():
            i0, i1 = self._block_slice("p", name)
            self._p_buf[i0:i1] = self._coerce_single_parameter_block(name, value)

    def set_parameters_constant(self, data: Dict[str, Any]) -> None:
        if self._np == 0:
            return
        self._apply_param_block_updates(data)

    def set_parameters_trajectory(self, data: Dict[str, Any]) -> None:
        if self._np == 0:
            return
        self._apply_param_block_updates(data)

    def apply_parameters(self) -> None:
        if self._np > 0:
            self.opti.set_value(self.p, self._p_buf)
        if self._nx > 0:
            self.opti.set_value(self.x_ref, self._x_ref)

    def set_initial_state(self, data: Dict[str, Iterable[float]]) -> None:
        self._x_ref = self._stack_block("x", data)
        self._x_val = self._x_ref.copy()
        self.opti.set_initial(self.x, self._x_val)
        self.apply_parameters()

    def set_state_constant(self, data: Dict[str, Iterable[float]]) -> None:
        self._x_val = self._stack_block("x", data)
        self.opti.set_initial(self.x, self._x_val)

    def set_input_constant(self, data: Dict[str, Iterable[float]]) -> None:
        self._u_val = self._stack_block("u", data)
        self.opti.set_initial(self.u, self._u_val)

    def set_max_iter(self, n_iter: int) -> None:
        self.cfg.max_iter = int(n_iter)
        self._configure_solver()

    def set_nlp_tol(self, tol: float) -> None:
        self.cfg.tol = float(tol)
        self.cfg.acceptable_tol = max(float(tol), self.cfg.acceptable_tol)
        self._configure_solver()

    def solve(self, first_iteration: bool = True) -> Tuple[int, Dict[str, List[float]]]:
        start = time.perf_counter()
        status = 0
        self.last_exception = None

        try:
            self._solution = self.opti.solve()
            self._last_raw_stats = dict(self._solution.stats())
            self.last_status = str(self._last_raw_stats.get("return_status", "Solve_Succeeded"))
        except RuntimeError as exc:
            status = 1
            self._solution = None
            self._last_raw_stats = self._read_stats_after_failure()
            self.last_exception = str(exc)
            self.last_status = str(
                self._last_raw_stats.get("return_status", "RuntimeError")
            )
            if self.cfg.error_on_fail:
                raise

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self._x_val = self._value(self.x, fallback=self._x_val)
        self._u_val = self._value(self.u, fallback=self._u_val)
        self.opti.set_initial(self.x, self._x_val)
        self.opti.set_initial(self.u, self._u_val)

        stats = self._stats_dict(elapsed_ms)
        return status, stats

    def _read_stats_after_failure(self) -> Dict[str, Any]:
        try:
            return dict(self.opti.stats())
        except RuntimeError:
            return {}

    def _stats_dict(self, elapsed_ms: float) -> Dict[str, List[float]]:
        raw = self._last_raw_stats
        iter_count = int(raw.get("iter_count", 0))
        iterations = raw.get("iterations", {})

        def last_iter_value(name: str, default: float = 0.0) -> float:
            value = iterations.get(name, default) if isinstance(iterations, dict) else default
            arr = np.asarray(value, dtype=float).reshape(-1)
            return float(arr[-1]) if arr.size else float(default)

        inf_pr = last_iter_value("inf_pr", float("inf"))
        inf_du = last_iter_value("inf_du")
        alpha_pr = last_iter_value("alpha_pr", 1.0)
        obj = last_iter_value("obj")

        return {
            "#it": [iter_count],
            "objective": [obj],
            "constraint_violation": [inf_pr],
            "res_stat": [inf_du],
            "res_eq": [inf_pr],
            "res_ineq": [inf_pr],
            "res_comp": [0.0],
            "qp1_stat": [0],
            "qp1_it": [0],
            "qp2_stat": [0],
            "qp2_it": [0],
            "qp3_stat": [0],
            "qp3_it": [0],
            "alpha": [alpha_pr],
            "||pi||": [0.0],
            "||lam||": [0.0],
            "time_tot_ms_cpu": [elapsed_ms],
        }

    def _value(self, expr: cs.MX, fallback: Optional[np.ndarray] = None) -> np.ndarray:
        if self._solution is not None:
            return np.asarray(self._solution.value(expr), dtype=float).reshape(-1)
        try:
            return np.asarray(self.opti.debug.value(expr), dtype=float).reshape(-1)
        except RuntimeError:
            pass
        try:
            return np.asarray(self.opti.value(expr), dtype=float).reshape(-1)
        except RuntimeError:
            pass
        if fallback is not None:
            return np.asarray(fallback, dtype=float).reshape(-1)
        raise RuntimeError(
            f"Could not read Opti value after solver status '{self.last_status}'."
        )

    def get_cost(self) -> float:
        try:
            return float(self._value(self._objective)[0])
        except RuntimeError:
            return float(self._objective_fun(self._x_val, self._u_val, self._p_buf))

    def get_states_by_block(self) -> Dict[str, np.ndarray]:
        out = {}
        for name, i0, i1 in self._block_slices("x"):
            out[name] = self._x_val[i0:i1].reshape(-1, 1)
        return out

    def get_inputs_by_block(self) -> Dict[str, np.ndarray]:
        out = {}
        for name, i0, i1 in self._block_slices("u"):
            out[name] = self._u_val[i0:i1].reshape(-1, 1)
        return out

    def get_parameter_vector(self) -> np.ndarray:
        return self._p_buf.copy()


class IpoptSingleStage:
    def __init__(
        self,
        ocp: OCPDescription,
        build_cfg: Optional[IpoptSingleStageConfig] = None,
    ):
        self.ocp = ocp
        self.build_cfg = build_cfg or IpoptSingleStageConfig()
        self.solver = _SingleStageOptiCore(ocp, self.build_cfg)
        self._violation_funs = {}
        self.last_violation_details: List[Dict[str, Any]] = []

    def update_params(
        self,
        constant_params: Optional[Dict[str, Any]] = None,
        trajectory_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        if constant_params:
            self.solver.set_parameters_constant(constant_params)
        if trajectory_params:
            self.solver.set_parameters_trajectory(trajectory_params)

    def apply_params(self) -> None:
        self.solver.apply_parameters()

    def solve(self, first_iteration: bool = True) -> Tuple[int, Dict[str, List[float]], float]:
        status, stats_dict = self.solver.solve(first_iteration=first_iteration)
        return status, stats_dict, self.solver.get_cost()

    def compute_detailed_metrics(self) -> Tuple[float, float, float, np.ndarray, List[str]]:
        x_val = self.solver._x_val
        u_val = self.solver._u_val
        p_val = self.solver.get_parameter_vector()

        all_vios: List[float] = []
        all_names: List[str] = []
        all_details: List[Dict[str, Any]] = []
        sum_vio = 0.0

        for constr in self.ocp.constraints:
            val = np.asarray(constr.fun(x_val, u_val, p_val), dtype=float).reshape(-1)
            lb, ub = _SingleStageOptiCore._broadcast_bounds(constr.lb, constr.ub, val.size)
            lb_arr = np.asarray(lb, dtype=float).reshape(-1)
            ub_arr = np.asarray(ub, dtype=float).reshape(-1)
            vio_lb = np.maximum(lb_arr - val, 0.0)
            vio_ub = np.maximum(val - ub_arr, 0.0)
            vio = np.maximum(vio_lb, vio_ub)
            sum_vio += float(np.sum(vio_lb) + np.sum(vio_ub))

            for idx, amount in enumerate(vio):
                amount = float(amount)
                if amount <= 0.0:
                    continue
                if vio_lb[idx] >= vio_ub[idx]:
                    side = "lb"
                    bound = float(lb_arr[idx])
                else:
                    side = "ub"
                    bound = float(ub_arr[idx])
                name = f"{constr.role}:{constr.name}[{idx}]"
                all_vios.append(amount)
                all_names.append(name)
                all_details.append({
                    "name": name,
                    "side": side,
                    "value": float(val[idx]),
                    "lb": float(lb_arr[idx]),
                    "ub": float(ub_arr[idx]),
                    "bound": bound,
                    "violation": amount,
                })

        if all_vios:
            order = np.argsort(-np.asarray(all_vios))
            sorted_vios = np.asarray(all_vios, dtype=float)[order]
            sorted_names = [all_names[i] for i in order]
            self.last_violation_details = [all_details[i] for i in order]
            max_vio = float(sorted_vios[0])
        else:
            sorted_vios = np.asarray([], dtype=float)
            sorted_names = []
            self.last_violation_details = []
            max_vio = 0.0

        return self.solver.get_cost(), max_vio, float(sum_vio), sorted_vios, sorted_names

    def set_initial_guess(self, x0_dict, u0_dict):
        """Restore owned offset blocks without changing the fixed-state reference."""
        states, controls = validate_owned_warm_start(
            x0_dict, u0_dict, self.ocp.x_dict, self.ocp.u_dict, 1, 1)
        self.solver.set_state_constant({name: values[0] for name, values in states.items()})
        self.solver.set_input_constant({name: values[0] for name, values in controls.items()})

    def get_results(self, robot_cfgs, env_cfgs, resolved, rob_state_name: str = "rob_dx") -> Dict[str, Any]:
        states_single = self.solver.get_states_by_block()
        inputs_single = self.solver.get_inputs_by_block()

        robot_xs = {
            rid: self._retract_manifold(states_single[robot_name(rid, rob_state_name)], resolved.robots[rid].x).T
            for rid in robot_cfgs
        }

        obj_xs = {}
        for name in env_cfgs:
            if name in states_single:
                obj_xs[name] = self._retract_manifold(
                    states_single[name],
                    resolved.env_objects[name].x,
                ).T

        return {
            **owned_warm_start(states_single, inputs_single, robot_cfgs, env_cfgs, rob_state_name),
            "robot_xs": robot_xs,
            "obj_xs": obj_xs,
            "states": states_single,
            "inputs": inputs_single,
        }

    @staticmethod
    def _retract_manifold(states: np.ndarray, x0: np.ndarray) -> np.ndarray:
        integrate = pin.liegroups.SE3().integrate
        base_ref = np.asarray(x0[:7], dtype=float)
        full_states = []
        for k in range(states.shape[1]):
            q_base = np.asarray(integrate(base_ref, states[:6, k]), dtype=float).reshape(7)
            rest = states[6:, k]
            full_states.append(np.concatenate([q_base, rest]))
        return np.stack(full_states, axis=1)
