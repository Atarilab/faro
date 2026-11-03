from __future__ import annotations

import time
from enum import Enum
from typing import Dict, List, Tuple, Iterable, Optional, Union

import numpy as np
import casadi as cs

from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

def _as_1d_floats(x: Union[float, Iterable[float], np.ndarray]) -> List[float]:
    if isinstance(x, (float, int)):
        return [float(x)]
    if isinstance(x, np.ndarray):
        return list(x.reshape(-1).astype(float))
    return [float(v) for v in x]


class HPIPM_MODE(Enum):
    speed = "SPEED"
    balance = "BALANCE"
    robust = "ROBUST"


class ProblemSpec:
    def __init__(self) -> None:
        self.blocks: Dict[str, Dict[str, cs.SX]] = {k: {} for k in ['x', 'u', 'p']}
        self.dim: Dict[str, int] = {k: 0 for k in ['x', 'u', 'p']}

        self._xn: List[cs.SX] = []
        self._f_expl: List[cs.SX] = []
        self._f_impl: List[cs.SX] = []
        self._xdot: List[cs.SX] = []
        self._z: List[cs.SX] = []

        # name -> (expr, use_in_path, use_in_terminal)
        self.costs: Dict[str, Tuple[cs.SX, bool, bool]] = {}
        self._W_defaults: Dict[str, List[float]] = {}

        self._var_bounds: Dict[str, Tuple[List[float], List[float], List[int]]] = {
            'x': ([], [], []),
            'u': ([], [], []),
        }

        self._constr: Dict[str, List[cs.SX]] = {k: [] for k in ['h_0', 'h', 'h_e']}
        self._constr_lb: Dict[str, List[float]] = {k: [] for k in ['h_0', 'h', 'h_e']}
        self._constr_ub: Dict[str, List[float]] = {k: [] for k in ['h_0', 'h', 'h_e']}

        self._constr_blocks = {k: [] for k in ['h_0', 'h', 'h_e']}
        self._constr_order  = {k: [] for k in ['h_0', 'h', 'h_e']}

        self._slack_idx: Dict[str, List[int]] = {k: [] for k in ['h_0', 'h', 'h_e']}
        self._slack_Zl: Dict[str, float] = {k: 1e4 for k in ['h_0', 'h', 'h_e']}
        self._slack_Zu: Dict[str, float] = {k: 1e4 for k in ['h_0', 'h', 'h_e']}
        self._slack_zl: Dict[str, float] = {k: 0.0 for k in ['h_0', 'h', 'h_e']}
        self._slack_zu: Dict[str, float] = {k: 0.0 for k in ['h_0', 'h', 'h_e']}

        # Full initial-state equality (no partial); handled via idxbx_0/lbx_0/ubx_0
        self._x0_val: Optional[np.ndarray] = None

    def add_states(self, name: str, x: cs.SX) -> None:
        self.blocks['x'][name] = x
        self.dim['x'] += int(x.shape[0])

    def add_inputs(self, name: str, u: cs.SX) -> None:
        self.blocks['u'][name] = u
        self.dim['u'] += int(u.shape[0])

    def add_parameters(self, name: str, p: cs.SX) -> None:
        self.blocks['p'][name] = p
        self.dim['p'] += int(p.shape[0])

    def set_discrete_dynamics(self, xnext: cs.SX) -> None:
        self._xn.append(xnext)

    def set_explicit_dynamics(self, f_expl: cs.SX) -> None:
        self._f_expl.append(f_expl)

    def set_implicit_dynamics(self, f_impl: cs.SX, xdot: cs.SX, z: Optional[cs.SX] = None) -> None:
        self._f_impl.append(f_impl)
        self._xdot.append(xdot)
        if z is not None and int(z.shape[0]) > 0:
            self._z.append(z)

    def add_cost(self, name: str, expr: cs.SX, weight: Union[float, Iterable[float]] = 1.0,
                 at: Tuple[str, ...] = ('path',)) -> None:
        if name in self.costs:
            raise ValueError(f"Duplicate cost name '{name}'.")
        # force column vector
        nr, nc = int(expr.shape[0]), int(expr.shape[1])
        if nc != 1:
            expr = cs.reshape(expr, nr * nc, 1)
        path = 'path' in at
        terminal = 'terminal' in at
        self.costs[name] = (expr, path, terminal)
        self._W_defaults[name] = _as_1d_floats(weight)

    def add_bounds(self, var_field: str, lb: Iterable[float], ub: Iterable[float],
                idx: Optional[Union[Iterable[int], slice, range, tuple]] = None) -> None:
        
        def _sanitize_bounds_array(arr: Iterable[float], *, kind: str, var_field: str) -> np.ndarray:
            a = np.asarray(list(arr), dtype=float).reshape(-1)
            if np.any(np.isnan(a)):
                bad = np.where(np.isnan(a))[0]
                raise ValueError(f"{kind}{var_field} contains NaN at indices {bad.tolist()}")
            # Replace +/-inf with large finite sentinels (acados requires finite)
            pos_inf = np.isposinf(a)
            neg_inf = np.isneginf(a)
            if np.any(pos_inf) or np.any(neg_inf):
                a = a.copy()
                a[pos_inf] = 1e8
                a[neg_inf] = -1e8
            return a
        
        if var_field not in ('x', 'u'):
            raise ValueError("var_field must be 'x' or 'u'")
        # sanitize before anything else
        lb = _sanitize_bounds_array(lb, kind='lb', var_field=var_field)
        ub = _sanitize_bounds_array(ub, kind='ub', var_field=var_field)

        if lb.size != ub.size:
            raise ValueError('lb and ub must be same length')
        total = self.dim[var_field]
        if idx is None:
            idx_list = list(range(int(lb.size)))
        else:
            idx_list = self._normalize_index_spec(idx, total)
        if len(idx_list) != int(lb.size):
            raise ValueError(f"Length of idx ({len(idx_list)}) must match lb/ub length ({lb.size}).")
        L, U, I = self._var_bounds[var_field]
        L.extend(lb.tolist()); U.extend(ub.tolist()); I.extend(idx_list)

    def add_constraint(self, name: str, expr: cs.SX,
                       lb: Union[float, Iterable[float]],
                       ub: Union[float, Iterable[float]],
                       at: Tuple[str, ...] = ('path',),
                       slack: Union[None, str, Iterable[int]] = None,
                       Z: Optional[float] = None,
                       z: Optional[float] = None) -> None:
        lbv = _as_1d_floats(lb)
        ubv = _as_1d_floats(ub)
        n = int(expr.shape[0])
        if len(lbv) == 1 and n > 1:
            lbv = lbv * n
        if len(ubv) == 1 and n > 1:
            ubv = ubv * n
        if len(lbv) != n or len(ubv) != n:
            raise ValueError('Constraint bound length mismatch.')

        def _stage_add(tag: str):
            i0 = sum(int(e.shape[0]) for e in self._constr[tag])
            self._constr[tag].append(expr)
            self._constr_lb[tag].extend(lbv)
            self._constr_ub[tag].extend(ubv)
            n = int(expr.shape[0])
            i1 = i0 + n
            self._constr_blocks[tag].append((name, i0, i1))
            self._constr_order[tag].append(name)
            if slack is not None:
                idxs = list(range(n)) if slack == 'all' else list(slack)
                offset = i0
                self._slack_idx[tag].extend([offset + i for i in idxs])
                if Z is not None: self._slack_Zl[tag] = float(Z); self._slack_Zu[tag] = float(Z)
                if z is not None: self._slack_zl[tag] = float(z); self._slack_zu[tag] = float(z)

        if 'initial' in at:
            _stage_add('h_0')
        if 'path' in at:
            _stage_add('h')
        if 'terminal' in at:
            _stage_add('h_e')

    def set_x0(self, x0: Iterable[float]) -> None:
        self._x0_val = np.asarray(list(x0), dtype=float).reshape(-1)

    def _normalize_index_spec(self, idx_spec: Union[Iterable[int], slice, range, tuple], total_len: int) -> List[int]:
        if isinstance(idx_spec, slice):
            return list(range(*idx_spec.indices(total_len)))
        if isinstance(idx_spec, range):
            return list(idx_spec)
        if isinstance(idx_spec, tuple) and len(idx_spec) == 2:
            start, end = idx_spec
            return list(range(int(start), int(end)))
        out = [int(i) for i in idx_spec]  # type: ignore[arg-type]
        if any(i < 0 or i >= total_len for i in out):
            raise ValueError('Index out of range')
        return out

    def block_slice(self, field: str, name: str) -> Tuple[int, int]:
        if name not in self.blocks[field]:
            raise KeyError(f"Unknown block '{name}' in field '{field}'.")
        start = 0
        for k, sym in self.blocks[field].items():
            n = int(sym.shape[0])
            if k == name:
                return start, start + n
            start += n
        raise KeyError(f"Block '{name}' not found in field '{field}'.")

    def get_constraint_blocks(self, stage: str):
        if stage not in self._constr_blocks:
            raise KeyError(f"Unknown stage '{stage}'")
        return list(self._constr_blocks[stage])

    def _vcat(self, field: str) -> cs.SX:
        if field == 'x':
            return cs.vertcat(*[self.blocks['x'][k] for k in self.blocks['x']]) if self.blocks['x'] else cs.SX.zeros(0, 1)
        if field == 'u':
            return cs.vertcat(*[self.blocks['u'][k] for k in self.blocks['u']]) if self.blocks['u'] else cs.SX.zeros(0, 1)
        if field == 'p':
            return cs.vertcat(*[self.blocks['p'][k] for k in self.blocks['p']]) if self.blocks['p'] else cs.SX.zeros(0, 1)
        raise KeyError(field)

    def has_discrete(self) -> bool:  return len(self._xn) > 0
    def has_explicit(self) -> bool:  return len(self._f_expl) > 0
    def has_implicit(self) -> bool:  return len(self._f_impl) > 0


class AcadosBuilder:
    def __init__(self, name: str, spec: ProblemSpec, N: int, tf,
                 parameter_defaults: Optional[Dict[str, Iterable[float]]] = None) -> None:
        self.name = name
        self.spec = spec
        self.N = int(N)
        self.tf = tf
        self.parameter_defaults = parameter_defaults or {}
        self.model: Optional[AcadosModel] = None
        self.ocp: Optional[AcadosOcp] = None

    # -------- MODEL --------
    def _build_model(self) -> AcadosModel:
        m = AcadosModel()
        m.name = f"{self.name}_model"
        m.x = self.spec._vcat('x')
        m.u = self.spec._vcat('u')
        m.p = self.spec._vcat('p')
        if self.spec.has_discrete():
            m.disc_dyn_expr = cs.vertcat(*self.spec._xn)
        elif self.spec.has_explicit():
            m.f_expl_expr = cs.vertcat(*self.spec._f_expl)
        elif self.spec.has_implicit():
            m.xdot = cs.vertcat(*self.spec._xdot)
            if self.spec._z:
                m.z = cs.vertcat(*self.spec._z)
            m.f_impl_expr = cs.vertcat(*self.spec._f_impl)
        else:
            # trivial identity discrete map (avoid undefined dynamics)
            m.disc_dyn_expr = m.x
        return m

    # -------- COSTS --------
    def _build_costs(self, ocp: AcadosOcp) -> None:
        path_terms = [expr for (expr, do_path, _) in self.spec.costs.values() if do_path]
        if path_terms:
            ocp.cost.cost_type = 'NONLINEAR_LS'
            ocp.model.cost_y_expr = cs.vertcat(*path_terms)
            ny = int(ocp.model.cost_y_expr.shape[0])
            ocp.cost.W = np.eye(ny)
            ocp.cost.yref = np.zeros(ny)
        # only set type if non-empty
        term_terms = [expr for (expr, _, do_term) in self.spec.costs.values() if do_term]
        if term_terms:
            ocp.cost.cost_type_e = 'NONLINEAR_LS'
            ocp.model.cost_y_expr_e = cs.vertcat(*term_terms)
            ny_e = int(ocp.model.cost_y_expr_e.shape[0])
            ocp.cost.W_e = np.eye(ny_e)
            ocp.cost.yref_e = np.zeros(ny_e)

    # -------- CONSTRAINTS --------
    def _build_constraints(self, ocp: AcadosOcp) -> None:
        stages = {
            'h_0': ('_0', 'lh_0', 'uh_0', 'idxsh_0'),
            'h'  : ('',   'lh',   'uh',   'idxsh'),
            'h_e': ('_e', 'lh_e', 'uh_e', 'idxsh_e'),
        }
        for tag, (sfx, lh_name, uh_name, idxs_name) in stages.items():
            exprs = self.spec._constr[tag]
            if not exprs:
                continue
            expr = cs.vertcat(*exprs)
            setattr(ocp.model, f'con_h_expr{sfx}', expr)
            # bounds
            lh = np.array(self.spec._constr_lb[tag], dtype=float)
            uh = np.array(self.spec._constr_ub[tag], dtype=float)
            setattr(ocp.constraints, lh_name, lh)
            setattr(ocp.constraints, uh_name, uh)
            # slacks
            idxs = np.array(self.spec._slack_idx[tag], dtype=np.int64)
            if idxs.size > 0:
                # de-duplicate and sort for renderer sanity
                idxs = np.unique(idxs)
                setattr(ocp.constraints, idxs_name, idxs)
                Z = float(self.spec._slack_Zl[tag]); z = float(self.spec._slack_zl[tag])
                nsl = int(idxs.size)
                setattr(ocp.cost, f'Zl{sfx}', np.ones(nsl) * Z)
                setattr(ocp.cost, f'Zu{sfx}', np.ones(nsl) * Z)
                setattr(ocp.cost, f'zl{sfx}', np.ones(nsl) * z)
                setattr(ocp.cost, f'zu{sfx}', np.ones(nsl) * z)

        # Variable bounds
        for var in ('x', 'u'):
            lb, ub, idx = self.spec._var_bounds[var]
            if len(lb) == 0:
                continue
            idx_arr = np.array(idx, dtype=np.int64)
            lb_arr = np.array(lb, dtype=float)
            ub_arr = np.array(ub, dtype=float)
            setattr(ocp.constraints, f'idxb{var}', idx_arr)
            setattr(ocp.constraints, f'lb{var}', lb_arr)
            setattr(ocp.constraints, f'ub{var}', ub_arr)
            if var == 'x':
                # acados uses separate fields for initial and terminal state bounds.
                # When x0 is free, mirror the state bounds onto stage 0 as well.
                if self.spec._x0_val is None:
                    ocp.constraints.idxbx_0 = idx_arr
                    ocp.constraints.lbx_0   = lb_arr
                    ocp.constraints.ubx_0   = ub_arr
                ocp.constraints.idxbx_e = idx_arr
                ocp.constraints.lbx_e   = lb_arr
                ocp.constraints.ubx_e   = ub_arr

        # Initial-state equality via idxbx_0/lbx_0/ubx_0
        if self.spec._x0_val is not None:
            nx = int(ocp.model.x.shape[0])  # use already-built model
            x0 = np.asarray(self.spec._x0_val, dtype=float).reshape(-1)
            if x0.size != nx:
                raise ValueError(f"spec.set_x0 length {x0.size} != nx {nx}")
            ocp.constraints.x0 = x0
            if self.N == 0:
                # A zero-horizon problem has only a terminal stage. Acados
                # ignores stage-0 fields here, including constraints.x0.
                ocp.constraints.idxbx_e = np.arange(nx)
                ocp.constraints.lbx_e = x0.copy()
                ocp.constraints.ubx_e = x0.copy()

    # -------- PREFLIGHT CHECKS --------
    @staticmethod
    def _assert_vec(name, arr, n):
        a = np.asarray(arr, dtype=float).reshape(-1)
        if a.size != n:
            raise ValueError(f"{name} must have length {n}, got {a.size}")
        if not np.all(np.isfinite(a)):
            raise ValueError(f"{name} contains non-finite entries")

    @staticmethod
    def _assert_mat(name, M, n):
        A = np.asarray(M, dtype=float)
        if A.shape != (n, n):
            raise ValueError(f"{name} must be shape {(n, n)}, got {A.shape}")
        if not np.all(np.isfinite(A)):
            raise ValueError(f"{name} contains non-finite entries")

    def _preflight(self, ocp: AcadosOcp):
        # Acados represents an absent expression as []; a zero-horizon problem
        # can have only a terminal cost. Validate only configured NLS costs.
        if ocp.cost.cost_type == 'NONLINEAR_LS':
            ny = int(getattr(ocp.model, 'cost_y_expr').shape[0])
            if ny <= 0:
                raise ValueError("NONLINEAR_LS path cost set but ny == 0.")
            self._assert_mat("W", ocp.cost.W, ny)
            self._assert_vec("yref", ocp.cost.yref, ny)

        # TERMINAL COST
        if ocp.cost.cost_type_e == 'NONLINEAR_LS':
            ny_e = int(getattr(ocp.model, 'cost_y_expr_e').shape[0])
            if ny_e <= 0:
                raise ValueError("NONLINEAR_LS terminal cost set but ny_e == 0.")
            self._assert_mat("W_e", ocp.cost.W_e, ny_e)
            self._assert_vec("yref_e", ocp.cost.yref_e, ny_e)

        # CONSTRAINTS
        def check_con(tag_label: str, sfx: str):
            expr_obj = getattr(ocp.model, f'con_h_expr{sfx}', None)
            if expr_obj is None:
                return

            # Accept SX/MX or list of SX/MX
            if isinstance(expr_obj, list):
                if len(expr_obj) == 0:
                    return
                expr_cs = cs.vertcat(*expr_obj)
                nh = int(expr_cs.shape[0])
            else:
                # assume CasADi SX/MX
                nh = int(expr_obj.shape[0])

            lh = getattr(ocp.constraints, f'lh{sfx}')
            uh = getattr(ocp.constraints, f'uh{sfx}')
            self._assert_vec(f"l{tag_label}", lh, nh)
            self._assert_vec(f"u{tag_label}", uh, nh)

            idxs_name = {'': 'idxsh', '_0': 'idxsh_0', '_e': 'idxsh_e'}[sfx]
            idxs = getattr(ocp.constraints, idxs_name, None)
            if idxs is not None:
                idxs = np.asarray(idxs, dtype=np.int64).reshape(-1)
                if idxs.size > 0:
                    if (idxs.min() < 0) or (idxs.max() >= nh):
                        raise ValueError(f"{idxs_name} has indices outside 0..{nh - 1}")
                    nsl = int(idxs.size)
                    for nm in (f"Zl{sfx}", f"Zu{sfx}", f"zl{sfx}", f"zu{sfx}"):
                        v = getattr(ocp.cost, nm)
                        self._assert_vec(nm, v, nsl)

        check_con('h (path)', '')
        check_con('h_0 (initial)', '_0')
        check_con('h_e (terminal)', '_e')

        # VARIABLE BOUNDS
        for var in ('x', 'u'):
            idxb = getattr(ocp.constraints, f'idxb{var}', None)
            lb = getattr(ocp.constraints, f'lb{var}', None)
            ub = getattr(ocp.constraints, f'ub{var}', None)
            if idxb is None:
                continue
            idxb = np.asarray(idxb, dtype=np.int64).reshape(-1)
            self._assert_vec(f"lb{var}", lb, idxb.size)
            self._assert_vec(f"ub{var}", ub, idxb.size)
            nv = int(getattr(ocp.model, var).shape[0])
            if (idxb.size > 0) and ((idxb.min() < 0) or (idxb.max() >= nv)):
                raise ValueError(f"idxb{var} has indices outside 0..{nv - 1}")

        # INITIAL STATE BOUNDS / EQUALITY
        if self.spec._x0_val is not None:
            nx = int(ocp.model.x.shape[0])
            self._assert_vec("lbx_0", ocp.constraints.lbx_0, nx)
            self._assert_vec("ubx_0", ocp.constraints.ubx_0, nx)
            idx = np.asarray(ocp.constraints.idxbx_0, dtype=np.int64).reshape(-1)
            if idx.size != nx or (idx.min() != 0) or (idx.max() != nx - 1):
                raise ValueError("idxbx_0 must be range(nx) when using full-state equality")
        else:
            idx = getattr(ocp.constraints, 'idxbx_0', None)
            if idx is not None:
                idx = np.asarray(idx, dtype=np.int64).reshape(-1)
                self._assert_vec("lbx_0", ocp.constraints.lbx_0, idx.size)
                self._assert_vec("ubx_0", ocp.constraints.ubx_0, idx.size)
                nx = int(ocp.model.x.shape[0])
                if (idx.size > 0) and ((idx.min() < 0) or (idx.max() >= nx)):
                    raise ValueError(f"idxbx_0 has indices outside 0..{nx - 1}")

        # PARAMETERS
        np_param = int(ocp.model.p.shape[0])
        if np_param > 0:
            pv = np.asarray(ocp.parameter_values, dtype=float).reshape(-1)
            if pv.size != np_param:
                raise ValueError(f"parameter_values length {pv.size} != np {np_param}")
            if not np.all(np.isfinite(pv)):
                raise ValueError("parameter_values contains non-finite entries")


    # -------- BUILD --------
    def build(self, force_recompile: bool = True,
              solver_config_path: str = "",
              code_gen_path: str = "",
              hpipm_mode: HPIPM_MODE = HPIPM_MODE.balance,
              qp_max_iter: int = 50,
              max_nlp_iters: int = 500,
              tol: float = 1e-4) -> 'AcadosSolver':
        self.model = self._build_model()

        ocp = AcadosOcp()
        ocp.model = self.model
        ocp.solver_options.N_horizon = self.N
        ocp.solver_options.tf = self.tf # Scaled via runtime parameter

        if self.model.__dict__.get('disc_dyn_expr', None) is not None:
            ocp.solver_options.integrator_type = 'DISCRETE'
        elif self.model.__dict__.get('f_expl_expr', None) is not None:
            ocp.solver_options.integrator_type = 'ERK'
        else:
            ocp.solver_options.integrator_type = 'IRK'

        self._build_costs(ocp)
        self._build_constraints(ocp)

        # Parameters
        np_param = int(self.spec._vcat('p').shape[0])
        if np_param > 0 and self.parameter_defaults:
            flat = []
            for name in self.spec.blocks['p'].keys():
                v = np.asarray(list(self.parameter_defaults.get(name, [])), dtype=float).reshape(-1)
                flat.extend(v)
            if len(flat) != np_param:
                raise ValueError(f"parameter_defaults length {len(flat)} != np {np_param}")
            ocp.parameter_values = np.array(flat, dtype=float)
        else:
            ocp.parameter_values = np.zeros(np_param, dtype=float)

        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.qp_solver_cond_N = self.N
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.qp_solver_iter_max = qp_max_iter
        ocp.solver_options.nlp_solver_max_iter = max_nlp_iters
        ocp.solver_options.nlp_solver_warm_start_first_qp = True
        ocp.solver_options.nlp_solver_warm_start_first_qp_from_nlp = True
        ocp.solver_options.nlp_solver_type = 'SQP_WITH_FEASIBLE_QP'

        ocp.solver_options.nlp_qp_tol_strategy = 'ADAPTIVE_CURRENT_RES_JOINT'

        ocp.solver_options.tol = tol
        ocp.solver_options.hpipm_mode = hpipm_mode.value

        ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
        ocp.solver_options.search_direction_mode = 'BYRD_OMOJOKUN'
        ocp.solver_options.allow_direction_mode_switch_to_nominal = False

        ocp.solver_options.collocation_type = 'GAUSS_LEGENDRE'
        ocp.solver_options.sim_method_num_stages  = 1
        ocp.solver_options.sim_method_num_steps   = 1
        ocp.solver_options.sim_method_newton_iter = 3

        # Final sanity pass before codegen
        self._preflight(ocp)

        ocp.code_export_directory = code_gen_path
        json_file = f"{solver_config_path}/{self.name}.json"
        
        solver = AcadosOcpSolver(
            ocp,
            json_file=json_file,
            generate=force_recompile,
            build=force_recompile,
            verbose=False,
        )
        self.ocp = ocp
        return AcadosSolver(self.name, self.spec, self.model, ocp, solver)


class AcadosSolver:
    def __init__(self, name: str, spec: ProblemSpec, model: AcadosModel, ocp: AcadosOcp, solver: AcadosOcpSolver) -> None:
        self.name = name
        self.spec = spec
        self.model = model
        self.ocp = ocp
        self.solver = solver
        self.N = ocp.solver_options.N_horizon
        self._nx = int(model.x.shape[0]) if hasattr(model, 'x') else 0
        self._nu = int(model.u.shape[0]) if hasattr(model, 'u') else 0
        self._np = int(model.p.shape[0]) if hasattr(model, 'p') else 0

        if self._np > 0:
            base = self._param_base_vector()
            self._p_buf = np.tile(base.reshape(1, -1), (self.N + 1, 1))
            
            p0 = np.asarray(self.solver.get(0, 'p'), dtype=float).reshape(-1)
            if p0.size == self._np and np.all(np.isfinite(p0)):
                self._p_buf[:, :] = p0.reshape(1, -1)
        else:
            self._p_buf = np.zeros((self.N + 1, 0), dtype=float)

    def print_status(self, status: int) -> None:
        status_map = {
            0: "SUCCESS — optimal solution found.",
            1: "FAILURE",
            2: "MAXITER — maximum number of iterations reached.",
            3: "MINSTEP — minimum step size in QP solver reached.",
            4: "QP_FAILURE — qp solver failed.",
        }
        meaning = status_map.get(status, "UNKNOWN STATUS CODE (check acados log).")
        print(f"[acados] Solver status = {status}: {meaning}")

    def get_residuals(self):
        """Return (res_stat, res_eq, res_ineq, res_comp) if available, else None."""
        try:
            return self.solver.get_residuals()
        except Exception:
            return None

    def set_initial_state(self, data):
        x0 = self._stack_block("x", data)
        x0_lb = x0.copy()
        x0_ub = x0.copy()
        if self.spec._x0_val is not None:
            self.solver.constraints_set(0, "lbx", x0_lb)
            self.solver.constraints_set(0, "ubx", x0_ub)

    def set_state_constant(self, data: Dict[str, Iterable[float]]) -> None:
        x = self._stack_block('x', data)
        for i in range(self.N + 1):
            self.solver.set(i, 'x', x)

    def set_input_constant(self, data: Dict[str, Iterable[float]]) -> None:
        u = self._stack_block('u', data)
        for i in range(self.N):
            self.solver.set(i, 'u', u)

    # --- parameters ---
    def _param_base_vector(self) -> np.ndarray:
        np_param = self._np
        if np_param == 0:
            return np.zeros(0, dtype=float)
        try:
            base = np.asarray(self.ocp.parameter_values, dtype=float).reshape(-1)
            if base.size == np_param:
                return base.copy()
        except Exception:
            pass
        return np.zeros(np_param, dtype=float)

    def _apply_param_block_updates(self, vec: np.ndarray, updates: dict) -> None:
        order = list(self.spec.blocks['p'].keys())
        for name, arr in updates.items():
            if name not in order:
                raise KeyError(f"Unknown parameter block '{name}'. Known: {order}")
            a = np.asarray(arr, dtype=float).reshape(-1)
            i0, i1 = self.spec.block_slice('p', name)
            if a.size != (i1 - i0):
                raise ValueError(f"Block '{name}' size {a.size} != expected {(i1 - i0)}")
            vec[i0:i1] = a

    def set_parameters_constant(self, data: Dict[str, Iterable[float]]) -> None:
        """
        Set (or partially update) parameters as CONSTANT across the whole horizon, using a dict of blocks.

        This ONLY updates the internal parameter buffer. Call apply_parameters() to commit to acados.
        """
        if self._np == 0:
            return
        for k in range(self.N + 1):
            self._apply_param_block_updates(self._p_buf[k, :], data)

    def set_parameters_trajectory(self, data: Dict[str, object]) -> None:
        """
        Set (or partially update) parameters as a TRAJECTORY across the horizon, using a dict of blocks.

        Supported per-block formats:
        - (dim,)            -> constant block, broadcast to all stages
        - (dim, N+1)        -> full trajectory for that block
        - (N+1,)            -> only allowed if dim == 1 (scalar block time series)

        This ONLY updates the internal parameter buffer. Call apply_parameters() to commit to acados.
        """
        if self._np == 0:
            return
        if not isinstance(data, dict):
            raise TypeError("set_parameters_trajectory expects a dict of parameter blocks")

        def _coerce_block_array(A) -> np.ndarray:
            # CasADi DM/SX/MX -> numpy if possible
            try:
                import casadi
                if isinstance(A, (casadi.DM, casadi.SX, casadi.MX)):
                    A = A.full()
            except Exception:
                pass

            B = np.asarray(A)
            if B.dtype == object:
                B = np.array(A, dtype=float)
            else:
                B = B.astype(float, copy=False)

            if B.ndim > 2:
                B = np.squeeze(B)
            return B

        # normalize each block to (dim, N+1)
        norm_blocks = {}
        for name, A in data.items():
            i0, i1 = self.spec.block_slice('p', name)
            d = i1 - i0
            B = _coerce_block_array(A)

            if B.ndim == 1:
                if B.size == d:
                    B = B.reshape(d, 1).repeat(self.N + 1, axis=1)
                elif B.size == (self.N + 1):
                    if d != 1:
                        raise ValueError(
                            f"Block '{name}' dim={d}, but provided 1D time series length N+1={self.N+1}. "
                            f"Only allowed for scalar blocks."
                        )
                    B = B.reshape(1, self.N + 1)
                else:
                    raise ValueError(
                        f"Block '{name}' 1D size must be {d} (constant) or {self.N+1} (scalar time series). Got {B.size}."
                    )
            elif B.ndim == 2:
                if B.shape != (d, self.N + 1):
                    raise ValueError(f"Block '{name}' must be shape (dim={d}, N+1={self.N+1}), got {B.shape}.")
            else:
                raise ValueError(f"Block '{name}' must be 1D or 2D after squeeze; got shape={B.shape}.")

            norm_blocks[name] = (i0, i1, B)

        # write into buffer (overwrite only provided blocks; leave others unchanged)
        for k in range(self.N + 1):
            for _, (i0, i1, B) in norm_blocks.items():
                self._p_buf[k, i0:i1] = B[:, k]

    def apply_parameters(self) -> None:
        """Commit the internal parameter buffer to acados for all stages 0..N."""
        if self._np == 0:
            return
        for k in range(self.N + 1):
            self.solver.set(k, 'p', self._p_buf[k, :])

    def get_parameter_buffer(self) -> np.ndarray:
        """Return a copy of the internal parameter buffer as shape (N+1, np)."""
        return self._p_buf.copy()

    def _y_blocks(self, terminal: bool) -> List[Tuple[str, int, int]]:
        names: List[str] = []
        for name, (expr, do_path, do_term) in self.spec.costs.items():
            if (terminal and do_term) or ((not terminal) and do_path):
                names.append(name)
        sizes = [int(self.spec.costs[n][0].shape[0]) for n in names]
        slices: List[Tuple[str, int, int]] = []
        start = 0
        for n, sz in zip(names, sizes):
            slices.append((n, start, start + sz))
            start += sz
        return slices

    def set_cost_weight_constant(self, weights: Dict[str, Union[float, Iterable[float]]], terminal: bool = False) -> None:
        yW = np.array(self.solver.get_cost('W_e' if terminal else 'W'), dtype=float)
        blocks = self._y_blocks(terminal)
        for name, i0, i1 in blocks:
            default = self.spec._W_defaults.get(name, [1.0])
            w = np.asarray(_as_1d_floats(weights.get(name, default)), dtype=float)
            if w.size == 1:
                yW[i0:i1, i0:i1] = np.eye(i1 - i0) * float(w)
            elif w.size == (i1 - i0):
                yW[i0:i1, i0:i1] = np.diag(w)
            else:
                raise ValueError(f"Weight size for '{name}' must be 1 or {i1 - i0}.")
        self.solver.cost_set(0 if not terminal else self.N, 'W' if not terminal else 'W_e', yW)
        if not terminal:
            for i in range(self.N):
                self.solver.cost_set(i, 'W', yW)

    def set_ref_constant(self, refs: Dict[str, Iterable[float]], terminal: bool = False) -> None:
        yref = np.array(self.solver.get_cost('yref_e' if terminal else 'yref'), dtype=float)
        blocks = self._y_blocks(terminal)
        for name, i0, i1 in blocks:
            r = np.asarray(list(refs[name]), dtype=float).reshape(-1)
            if r.size != (i1 - i0):
                raise ValueError(f"Ref size for '{name}' must be {i1 - i0}.")
            yref[i0:i1] = r
        if terminal:
            self.solver.cost_set(self.N, 'yref', yref)
        else:
            for i in range(self.N):
                self.solver.cost_set(i, 'yref', yref)

    def set_qp_tol(self, tol: float):
        self.solver.options_set('qp_tol_stat', tol)
        self.solver.options_set('qp_tol_eq', tol)
        self.solver.options_set('qp_tol_ineq', tol)
        self.solver.options_set('qp_tol_comp', tol)

    def set_nlp_tol(self, tol: float):
        self.solver.options_set('tol_eq', tol)
        self.solver.options_set('tol_stat', tol)
        self.solver.options_set('tol_ineq', tol)
        self.solver.options_set('tol_comp', tol)

    def solve(self, nlp_iters: int, first_iteration=True):
        self.solver.options_set('nlp_solver_max_iter', int(nlp_iters))
        status = self.solver.solve()
        
        stats_dict = {
            "#it": [],
            "res_stat": [],
            "res_eq": [],
            "res_ineq": [],
            "res_comp": [],
            "qp1_stat": [],
            "qp1_it": [],
            "qp2_stat": [],
            "qp2_it": [],
            "qp3_stat": [],
            "qp3_it": [],
            "alpha": [],
            "||pi||": [],
            "||lam||": [],
        }

        stat = np.asarray(self.solver.get_stats("statistics"), dtype=float)
        n_stat_rows = len(stats_dict)

        # acados may return a single iterate as a 1D array instead of a 2D matrix.
        # Normalize everything to (n_stats, n_cols) so the column selection below is stable.
        if stat.ndim == 0:
            stat = stat.reshape(1, 1)
        elif stat.ndim == 1:
            if stat.size % n_stat_rows == 0:
                stat = stat.reshape(n_stat_rows, -1)
            else:
                stat = stat.reshape(-1, 1)

        if stat.ndim != 2:
            raise ValueError(f"Unexpected acados statistics shape: {stat.shape}")

        if stat.shape[0] != n_stat_rows and stat.shape[1] == n_stat_rows:
            stat = stat.T

        if stat.shape[0] != n_stat_rows:
            raise ValueError(
                f"Unexpected acados statistics shape {stat.shape}; expected {n_stat_rows} statistic rows."
            )

        n_cols = stat.shape[1]
        col_idxs = [n_cols - 1]
        if first_iteration and n_cols >= 2:
            col_idxs = [n_cols - 2, n_cols - 1]

        qp_keys = {"qp1_stat", "qp1_it", "qp2_stat", "qp2_it", "qp3_stat", "qp3_it"}

        for col_pos, col_idx in enumerate(col_idxs):
            for i, k in enumerate(stats_dict):
                v = stat[i, col_idx]
                if k in {"#it"}:
                    stats_dict[k].append(int(v))
                elif k in qp_keys:
                    if first_iteration and len(col_idxs) == 2 and col_pos == 0:
                        stats_dict[k].append(int(0))
                    else:
                        prev_col_idx = col_idx - 1
                        qp_v = stat[i, prev_col_idx] if prev_col_idx >= 0 else v
                        stats_dict[k].append(int(qp_v))
                else:
                    stats_dict[k].append(float(v))

        stats_dict["time_tot_ms_cpu"] = [float(self.solver.get_stats("time_tot")) * 1000.0]

        return status, stats_dict

    def get_traj(self) -> Dict[str, np.ndarray]:
        X = np.zeros((self._nx, self.N + 1))
        U = np.zeros((self._nu, self.N)) if self._nu > 0 else np.zeros((0, self.N))
        for i in range(self.N + 1):
            X[:, i] = np.array(self.solver.get(i, 'x')).reshape(-1)
        for i in range(self.N):
            if self._nu > 0:
                U[:, i] = np.array(self.solver.get(i, 'u')).reshape(-1)
        return {'x': X, 'u': U}

    def _stack_block(self, field: str, data: Dict[str, Iterable[float]]) -> np.ndarray:
        from faro.ocp.robot_blocks import BlockLayout
        blocks = self.spec.blocks[field]
        if set(data) != set(blocks):
            raise KeyError(f"{field} blocks differ: missing {set(blocks) - set(data)}, unexpected {set(data) - set(blocks)}")
        return BlockLayout({name: int(sym.numel()) for name, sym in blocks.items()}).pack(data)

    def _block_slices(self, field: str):
        """Return an ordered list of (name, i0, i1) for x/u/p blocks."""
        i = 0
        out = []
        for name, sym in self.spec.blocks[field].items():
            n = int(sym.shape[0])
            out.append((name, i, i+n))
            i += n
        return out

    def get_state_block_trajectory(self, name: str):
        """(dim, N+1) array for a single state block."""
        X = self.get_traj()['x']  # (nx, N+1)
        for blk, i0, i1 in self._block_slices('x'):
            if blk == name:
                return X[i0:i1, :]
        raise KeyError(f"Unknown state block '{name}'. Known: {list(self.spec.blocks['x'].keys())}")

    def get_input_block_trajectory(self, name: str):
        """(dim, N) array for a single input block."""
        U = self.get_traj()['u']  # (nu, N)
        for blk, i0, i1 in self._block_slices('u'):
            if blk == name:
                return U[i0:i1, :]
        raise KeyError(f"Unknown input block '{name}'. Known: {list(self.spec.blocks['u'].keys())}")

    def get_parameter_trajectory(self, name: str):
        """(dim, N+1) array for a single parameter block."""
        i0, i1 = self.spec.block_slice('p', name)
        dim = i1 - i0
        traj = np.zeros((dim, self.N + 1))
        for k in range(self.N + 1):
            p = np.asarray(self.solver.get(k, 'p'), dtype=float).reshape(-1)
            traj[:, k] = p[i0:i1]
        return traj

    def get_states_by_block(self):
        """Dict of {block_name: (dim, N+1)} for all state blocks."""
        X = self.get_traj()['x']
        out = {}
        for name, i0, i1 in self._block_slices('x'):
            out[name] = X[i0:i1, :]
        return out

    def get_inputs_by_block(self):
        """Dict of {block_name: (dim, N)} for all input blocks."""
        U = self.get_traj()['u']
        out = {}
        for name, i0, i1 in self._block_slices('u'):
            out[name] = U[i0:i1, :]
        return out
