from __future__ import annotations

from faro.configs.identifiers import robot_name
from faro.ocp.configuration import normalize_pose_trajectory, align_pose_trajectory
from faro.solvers._owned_results import owned_warm_start, validate_owned_warm_start
from faro.solvers._acados_reset import reset_acados_solver

import hashlib
import numpy as np
import casadi as cs
import pinocchio as pin
from pathlib import Path
from dataclasses import asdict, dataclass, replace
from typing import Dict, Any, Tuple, List
from faro.ocp.ocp_description import OCPDescription

from faro.solvers.acados_wrapper import (
    ProblemSpec,
    AcadosBuilder,
    AcadosSolver,
    HPIPM_MODE
)


@dataclass
class AcadosStaticBuildConfig:
    """Build and numerical options for one free configuration, without intervals."""

    name: str = "ik_solver"
    solver_config_path: str = ""
    code_gen_path: str = ""
    force_recompile: bool = True
    hpipm_mode: HPIPM_MODE = HPIPM_MODE.balance
    qp_max_iter: int = 100
    max_nlp_iters: int = 500  # Build-time upper bound for runtime SQP iteration limits.
    tol: float = 1e-4


@dataclass
class AcadosBuildConfig:
    name: str = "g1_obj_solver"
    solver_config_path: str = ""
    code_gen_path: str = ""
    N_horizon: int = 20
    t_total: float = 1.0
    force_recompile: bool = True
    hpipm_mode: HPIPM_MODE = HPIPM_MODE.balance
    qp_max_iter: int = 50
    max_nlp_iters: int = 500  # Build-time upper bound for runtime SQP iteration limits.
    tol: float = 1e-4
    free_init_state: bool = False
    # None selects 1e-4 for quaternion configurations, zero for offset states.
    levenberg_marquardt: float | None = None
    # KSO contact-only SQP warm-up; zero disables the second solver.
    presolve_max_iters: int = 10

    def ensure_build_artifacts(self) -> None:
        solver_config_file = Path(self.solver_config_path) / f"{self.name}.json"
        solver_object_file = Path(self.code_gen_path) / f"acados_solver_{self.name}_model.o"

        solver_config_exists = solver_config_file.is_file()
        solver_object_exists = solver_object_file.is_file()

        if not self.force_recompile and (not solver_config_exists or not solver_object_exists):
            print(
                f"[AcadosBuildConfig] Missing build artifacts for '{self.name}' "
                f"while force_recompile=False. Enabling recompilation."
            )
            self.force_recompile = True

class Acados:
    def __init__(self, ocp: OCPDescription, 
                       build_cfg: AcadosBuildConfig | AcadosStaticBuildConfig,
                       dyn = "implicit"):
        if isinstance(build_cfg, AcadosStaticBuildConfig):
            build_cfg = AcadosBuildConfig(**asdict(build_cfg), N_horizon=0, free_init_state=True)
        if build_cfg.levenberg_marquardt is None:
            build_cfg = replace(build_cfg, levenberg_marquardt=(
                1e-4 if ocp.state_coordinates == 'configuration' else 0.0))
        if not np.isfinite(build_cfg.levenberg_marquardt) or build_cfg.levenberg_marquardt < 0:
            raise ValueError('levenberg_marquardt must be finite and nonnegative.')
        self.ocp = ocp
        fingerprint = hashlib.sha256()
        fingerprint.update(repr((dyn, ocp.state_coordinates, build_cfg.N_horizon, build_cfg.t_total,
                                 build_cfg.free_init_state, build_cfg.hpipm_mode,
                                 build_cfg.qp_max_iter, build_cfg.max_nlp_iters, build_cfg.tol,
                                 build_cfg.levenberg_marquardt)).encode())
        for collection in (ocp.x_dict, ocp.u_dict, ocp.p_dict):
            fingerprint.update(repr([(key, value.shape) for key, value in collection.items()]).encode())
        for function in (ocp.f_impl, ocp.f_expl, ocp.f_discr, ocp.r_path, ocp.r_terminal):
            if function is not None:
                fingerprint.update(function.serialize().encode())
        for constraint in ocp.constraints:
            fingerprint.update(repr((constraint.name, constraint.role, constraint.lb, constraint.ub)).encode())
            fingerprint.update(constraint.fun.serialize().encode())
        for bound in ocp.linear_bounds:
            fingerprint.update(repr(bound).encode())
        suffix = fingerprint.hexdigest()[:16]
        # Both filenames and their compilation directory identify the formulation.
        name = f"{build_cfg.name}_{suffix}"
        self.build_cfg = replace(build_cfg, name=name,
                                 code_gen_path=str(Path(build_cfg.code_gen_path) / name))
        self.build_cfg.ensure_build_artifacts()
        self.N = self.build_cfg.N_horizon

        self.spec = self._build_spec(dyn=dyn)
        self.solver = self._build_solver()

        self._violation_funs = {}
        self.last_violation_details: List[Dict[str, Any]] = []
        self._build_violation_evaluators()

    def _build_spec(self, dyn = "implicit") -> ProblemSpec:
        spec = ProblemSpec()
        for name, state in self.ocp.x_dict.items(): spec.add_states(name, state)
        for name, ctrl in self.ocp.u_dict.items(): spec.add_inputs(name, ctrl)
        for key, val in self.ocp.p_dict.items(): spec.add_parameters(name=key, p=val)

        if all(fun is None for fun in (self.ocp.f_impl, self.ocp.f_expl, self.ocp.f_discr)):
            if self.N != 0:
                raise ValueError("A formulation without dynamics requires zero horizon.")
            # Acados code generation expects a model even with no intervals.
            # Keep that backend-only declaration out of static formulations.
            xdot = cs.SX.sym('xdot', self.ocp.x.size1())
            spec.set_implicit_dynamics(f_impl=xdot, xdot=xdot)
        elif dyn == "implicit":
            xdot = cs.SX.sym('xdot', self.ocp.x.size1())
            F = self.ocp.f_impl(self.ocp.x, xdot, self.ocp.u, self.ocp.p)
            spec.set_implicit_dynamics(f_impl=F, xdot=xdot)
        elif dyn == "explicit":
            f_expl = self.ocp.f_expl(self.ocp.x, self.ocp.u, self.ocp.p)
            spec.set_explicit_dynamics(f_expl=f_expl)
        elif dyn == "discrete":
            f_discr = self.ocp.f_discr(self.ocp.x, self.ocp.u, self.ocp.p)
            spec.set_discrete_dynamics(f_discr)

        if self.ocp.r_path is not None:
            r = self.ocp.r_path(self.ocp.x, self.ocp.u, self.ocp.p)
            spec.add_cost('ls_path', expr=r, at=('path',))
        if self.ocp.r_terminal is not None:
            r_e = self.ocp.r_terminal(self.ocp.x, self.ocp.u, self.ocp.p)
            spec.add_cost('ls_term', expr=r_e, at=('terminal',))

        for c in self.ocp.constraints:
            spec.add_constraint(name=c.name, expr=c.fun(self.ocp.x, self.ocp.u, self.ocp.p),
                                lb=c.lb, ub=c.ub, at=(c.role,))

        for b in self.ocp.linear_bounds:
            spec.add_bounds(b.var_type, lb=b.lb, ub=b.ub, idx=b.indices)

        if not self.build_cfg.free_init_state:
            x0 = np.zeros(self.ocp.x.size1())
            if self.ocp.state_coordinates == "configuration":
                offset = 0
                for state in self.ocp.x_dict.values():
                    x0[offset + 6] = 1.0
                    offset += state.numel()
            spec.set_x0(x0)

        return spec

    def _build_solver(self) -> AcadosSolver:
        cfg = self.build_cfg
        builder = AcadosBuilder(cfg.name, 
                                spec=self.spec, 
                                N=cfg.N_horizon,
                                tf=cfg.t_total)
        
        solver = builder.build(force_recompile=cfg.force_recompile,
                               solver_config_path=cfg.solver_config_path,
                               code_gen_path=cfg.code_gen_path,
                               hpipm_mode=cfg.hpipm_mode, 
                               qp_max_iter=cfg.qp_max_iter, 
                               max_nlp_iters=cfg.max_nlp_iters,
                               tol=cfg.tol,
                               levenberg_marquardt=cfg.levenberg_marquardt)
        
        return solver


    def update_params(self, constant_params: Dict[str, np.ndarray] = None, 
                            trajectory_params: Dict[str, np.ndarray] = None):
        """Unified interface to update parameters."""
        if constant_params:
            self.solver.set_parameters_constant(constant_params)
        if trajectory_params:
            self.solver.set_parameters_trajectory(trajectory_params)
    
    def apply_params(self):
        self.solver.apply_parameters()

    def solve(self, nlp_iters: int, first_iteration=True):
        status, stats_dict = self.solver.solve(
            nlp_iters=nlp_iters, first_iteration=first_iteration)

        cost = self.solver.solver.get_cost()
        return status, stats_dict, cost

    def copy_primal_iterate_from(self, source: Acados) -> None:
        """Transfer raw x/u between matching layouts, resetting destination duals.

        Unlike importing an external initial guess, this does not project or
        sign-align quaternions. Constraint multipliers cannot be transferred
        between formulations with different constraint sets.
        """
        if self.N != source.N or self.ocp.state_coordinates != source.ocp.state_coordinates:
            raise ValueError("Primal transfer requires matching horizons and state coordinates.")
        for field in ('x_dict', 'u_dict'):
            target_layout = [(name, sym.shape) for name, sym in getattr(self.ocp, field).items()]
            source_layout = [(name, sym.shape) for name, sym in getattr(source.ocp, field).items()]
            if target_layout != source_layout:
                raise ValueError(f"Primal transfer requires matching {field} layouts.")
        trajectory = source.solver.get_traj()
        if not all(np.isfinite(value).all() for value in trajectory.values()):
            raise ValueError("Cannot transfer a nonfinite primal iterate.")
        reset_acados_solver(self.solver.solver, horizon=self.N)
        for k in range(self.N + 1):
            self.solver.solver.set(k, 'x', trajectory['x'][:, k])
        for k in range(self.N):
            self.solver.solver.set(k, 'u', trajectory['u'][:, k])

    def compute_detailed_metrics(self) -> Tuple[float, float, float, np.ndarray, List[str]]:
        """Returns (cost, max_violation, sum_violations, sorted_vios, sorted_vio_names)."""
        cost = float(self.solver.solver.get_cost())
        sum_vio = 0.0

        all_vios: List[float] = []
        all_names: List[str] = []
        all_details: List[Dict[str, Any]] = []

        for k in range(self.N + 1):
            x_k = self.solver.solver.get(k, "x")
            u_k = self.solver.solver.get(k, "u") if k < self.N else np.zeros(self.ocp.u.size1())
            p_k = self.solver.solver.get(k, "p")

            if self.ocp.state_coordinates == 'configuration' and k < self.N:
                endpoint = np.asarray(self.ocp.f_discr(x_k, u_k, p_k)).reshape(-1)
                defect = self.solver.solver.get(k + 1, 'x') - endpoint
                for j, value in enumerate(defect):
                    violation = abs(float(value)) if np.isfinite(value) else float('inf')
                    if violation == 0:
                        continue
                    name = f'dynamics[{j}]'
                    sum_vio += violation
                    all_vios.append(violation)
                    all_names.append(f'k={k}:{name}')
                    all_details.append(dict(k=k, name=name, side='eq', value=float(value),
                                            lb=0.0, ub=0.0, bound=0.0, violation=violation))

            stage_type = "path"
            if k == self.N:
                stage_type = "terminal"
            elif k == 0:
                stage_type = "initial"

            if stage_type not in self._violation_funs:
                continue

            names, fun, lb, ub = self._violation_funs[stage_type]
            val = np.array(fun(x_k, u_k, p_k)).flatten()

            vio_lb = np.maximum(lb - val, 0.0)
            vio_ub = np.maximum(val - ub, 0.0)
            vio = np.maximum(vio_lb, vio_ub)

            sum_vio += float(np.sum(vio_lb) + np.sum(vio_ub))

            if vio.size == 0:
                continue

            for j, v in enumerate(vio):
                v = float(v)
                if v <= 0.0:
                    continue 
                if vio_lb[j] >= vio_ub[j]:
                    side = "lb"
                    bound = float(lb[j])
                else:
                    side = "ub"
                    bound = float(ub[j])

                all_vios.append(v)
                all_names.append(f"k={k}:{names[j]}")
                all_details.append({
                    "k": k,
                    "name": names[j],
                    "side": side,
                    "value": float(val[j]),
                    "lb": float(lb[j]),
                    "ub": float(ub[j]),
                    "bound": bound,
                    "violation": v,
                })

        if all_vios:
            order = np.argsort(-np.asarray(all_vios))
            sorted_vios = np.asarray(all_vios)[order]
            sorted_names = [all_names[i] for i in order]
            self.last_violation_details = [all_details[i] for i in order]
            max_vio = float(sorted_vios[0])
        else:
            sorted_vios = np.asarray([], dtype=float)
            sorted_names = []
            self.last_violation_details = []
            max_vio = 0.0

        return cost, max_vio, float(sum_vio), sorted_vios, sorted_names

    def _build_violation_evaluators(self):
            """Lazily builds CasADi functions to check constraints, handling broadcasting."""
            x, u, p = self.ocp.x, self.ocp.u, self.ocp.p
            
            for role in ['initial', 'path', 'terminal']:
                names_flat, exprs, lbs, ubs = [], [], [], []
                for c in self.ocp.constraints:
                    if c.role == role:
                        fun = c.fun
                        
                        n_out = fun.size1_out(0)
                        
                        lb_arr = np.array(c.lb, dtype=float).flatten()
                        ub_arr = np.array(c.ub, dtype=float).flatten()
                        
                        # Broadcast scalar bounds to match the expression size.
                        if lb_arr.size == 1 and n_out > 1:
                            lb_arr = np.full(n_out, lb_arr[0])
                        if ub_arr.size == 1 and n_out > 1:
                            ub_arr = np.full(n_out, ub_arr[0])
                            
                        if lb_arr.size != n_out:
                            raise ValueError(f"Constraint '{c.name}' dim {n_out} mismatch with lb dim {lb_arr.size}")

                        for i in range(n_out):
                            names_flat.append(f"{role}:{c.name}[{i}]")

                        exprs.append(fun(x, u, p))
                        lbs.append(lb_arr)
                        ubs.append(ub_arr)
                
                if exprs:
                    self._violation_funs[role] = (
                        names_flat,
                        cs.Function(f'vio_{role}', [x, u, p], [cs.vertcat(*exprs)]),
                        np.concatenate(lbs),
                        np.concatenate(ubs)
                    )

    def set_initial_guess(self, x0_dict, u0_dict):
        """Restore owned trajectories without changing initial-state bounds."""
        states, controls = validate_owned_warm_start(
            x0_dict, u0_dict, self.ocp.x_dict, self.ocp.u_dict, self.N + 1, self.N)
        if self.ocp.state_coordinates == "configuration":
            states = {name: normalize_pose_trajectory(value, name=f"states.{name}")
                      for name, value in states.items()}
            controls = {name: normalize_pose_trajectory(value, name=f"controls.{name}")
                        for name, value in controls.items()}
            initial = getattr(getattr(self, 'spec', None), '_x0_val', None)
            offset = 0
            for name, value in states.items():
                if initial is not None:
                    value[:1] = align_pose_trajectory(value[:1], initial[None, offset:offset + value.shape[1]])
                for k in range(1, len(value)):
                    value[k:k + 1] = align_pose_trajectory(value[k:k + 1], value[k - 1:k])
                offset += value.shape[1]
            # Each next-configuration control uses its endpoint state's lift.
            pairs = [(robot_name(rid, 'q_rob'), robot_name(rid, 'q_next_rob'))
                     for rid in x0_dict['robots']]
            pairs.extend((name, name) for name in x0_dict['objects'])
            for state_name, control_name in pairs:
                if control_name in controls:
                    controls[control_name] = align_pose_trajectory(controls[control_name], states[state_name][1:])
        x = np.hstack(list(states.values())) if states else np.zeros((self.N + 1, 0))
        u = np.hstack(list(controls.values())) if controls else np.zeros((self.N, 0))
        for k in range(self.N + 1):
            self.solver.solver.set(k, "x", x[k])
        for k in range(self.N):
            self.solver.solver.set(k, "u", u[k])

    def get_results(self, robot_cfgs, env_cfgs, resolved,
                    rob_state_name="rob_dx") -> Dict[str, Any]:
        """
        Retrieve physical configurations, retracting only offset-coordinate OCPs.
        Configuration states are copied verbatim, including any norm residual.
        Returns pure data (numpy arrays), agnostic of Acados.
        """
        states = self.solver.get_states_by_block()
        inputs = self.solver.get_inputs_by_block()

        def physical_states(name, reference):
            if self.ocp.state_coordinates == "configuration":
                return states[name].T.copy()
            return self._retract_manifold(states[name], reference).T

        robot_xs = {
            rid: physical_states(robot_name(rid, rob_state_name), resolved.robots[rid].x)
            for rid in robot_cfgs
        }
        
        obj_xs = {}
        for name, cfg in env_cfgs.items():
            if name in states:
                obj_xs[name] = physical_states(name, resolved.env_objects[name].x)

        return {
            **owned_warm_start(states, inputs, robot_cfgs, env_cfgs, rob_state_name),
            "robot_xs": robot_xs,
            "obj_xs": obj_xs,
            "states": states,
            "inputs": inputs,
        }

    def _retract_manifold(self, states, x0):
        """Internal helper for Manifold retraction."""
        integrate = pin.liegroups.SE3().integrate
        base_ref = np.array(x0[:7])
        X_full = []
        for k in range(states.shape[1]):
            q_base = np.array(integrate(base_ref, states[:6, k])).reshape(7)
            rest = states[6:, k]
            X_full.append(np.concatenate([q_base, rest]))
        return np.stack(X_full, axis=1)
