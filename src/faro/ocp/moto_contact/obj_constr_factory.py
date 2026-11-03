import casadi
import pinocchio.casadi as cpin

from dataclasses import dataclass
from typing import Optional
from abc import ABC, abstractmethod

from faro.ocp.moto_contact.force_constr_funcs import \
                               sliding_patch_friction, \
                               zero_wrench_impact, \
                               obj_patch_env_force_constr

from faro.ocp.moto_contact.kin_constr_funcs import \
                             plane_bounded_plane_res_param

@dataclass
class ObjConstraintVars:
    """Wb_obj is body [moment, force] at q_i_obj; V_i_obj is a WORLD twist."""
    q_i_obj: casadi.SX
    Wb_obj:  casadi.SX
    p_i_obj: casadi.SX
    R_i_obj: casadi.SX
    V_i_obj: casadi.SX
    p_n_obj: casadi.SX
    R_n_obj: casadi.SX
    p_bnds:  casadi.SX
    active:  casadi.SX
    impact:  casadi.SX

@dataclass
class EnvConstraintVars:
    fp_p: Optional[casadi.SX] = None
    fp_R: Optional[casadi.SX] = None
    fp_V: Optional[casadi.SX] = None
    fp_pn: Optional[casadi.SX] = None
    fp_Rn: Optional[casadi.SX] = None
    fp_bnds: Optional[casadi.SX] = None
    mu: Optional[casadi.SX] = None
    containment_margin: casadi.SX = 0.05

class BaseContactConstraint(ABC):
    def __init__(self):
        self.has_kin_eqs = False
        self.has_kin_ineqs = False
        self.has_ctrl_eqs = False
        self.has_ctrl_ineqs = False

    @abstractmethod
    def get_kin_eqs(self, vars: ObjConstraintVars,
                    env_vars: EnvConstraintVars) -> casadi.SX:
        """
        Return a CasADi SX vector that must be driven to 0.
        """
        pass

    @abstractmethod
    def get_kin_ineqs(self, vars: ObjConstraintVars,
                      env_vars: EnvConstraintVars) -> casadi.SX:
        """
        Return a CasADi SX vector g(x) such that g(x) >= 0.
        """
        pass

    @abstractmethod
    def get_ctrl_eqs(self, vars: ObjConstraintVars,
                     env_vars: EnvConstraintVars) -> casadi.SX:
        """
        Return a CasADi SX vector that must be driven to 0.
        """
        pass

    @abstractmethod
    def get_ctrl_ineqs(self, vars: ObjConstraintVars,
                       env_vars: EnvConstraintVars) -> casadi.SX:
        """
        Return a CasADi SX vector g(x) such that g(x) >= 0.
        """
        pass


class ObjectPatchEnvPatchConstraint(BaseContactConstraint):
    def __init__(self,):
        super().__init__()

        self.has_kin_eqs = True
        self.has_kin_ineqs = True
        self.has_ctrl_eqs = True
        self.has_ctrl_ineqs = True

        self.stick_bounded_plane_func = plane_bounded_plane_res_param(parameterized_margin=True)

        self.stick_fric_func = obj_patch_env_force_constr()

        self.zero_wrench_impact = zero_wrench_impact()
        self.sliding_fric_func = sliding_patch_friction()

    def get_kin_constraints(self, vars: ObjConstraintVars,
                    env_vars: EnvConstraintVars):
        stick_plane_res, stick_plane_ineq = self.stick_bounded_plane_func(
            vars.p_i_obj,
            vars.R_i_obj,
            vars.p_n_obj,
            vars.R_n_obj,
            vars.p_bnds,
            env_vars.fp_p,
            env_vars.fp_R,
            env_vars.fp_pn,
            env_vars.fp_Rn,
            env_vars.fp_bnds,
            vars.active,
            env_vars.containment_margin
        )

        return stick_plane_res, stick_plane_ineq

    def get_kin_eqs(self, vars: ObjConstraintVars, env_vars: EnvConstraintVars):
        return self.get_kin_constraints(vars, env_vars)[0]

    def get_kin_ineqs(self, vars: ObjConstraintVars, env_vars: EnvConstraintVars):
        return self.get_kin_constraints(vars, env_vars)[1]

    def get_sliding_kin_eqs(self, vars: ObjConstraintVars, env_vars: EnvConstraintVars):
        # Keep normal contact and tilt; in-plane translation and yaw may change.
        return self.get_kin_constraints(vars, env_vars)[0][2:5]

    def get_ctrl_eqs(self, vars: ObjConstraintVars,
                     env_vars: EnvConstraintVars) -> casadi.SX:
        return self.get_wrench_eqs(vars, env_vars)

    def get_wrench_eqs(self, vars: ObjConstraintVars, env_vars: EnvConstraintVars):
        return self.zero_wrench_impact(vars.Wb_obj, vars.active, vars.impact)

    def get_sliding_constraints(self, vars: ObjConstraintVars, env_vars: EnvConstraintVars):
        rotation = cpin.XYZQUATToSE3(casadi.SX(vars.q_i_obj)).rotation
        return self.sliding_fric_func(vars.p_i_obj, env_vars.fp_R,
            vars.V_i_obj, env_vars.fp_V, rotation @ vars.Wb_obj[3:], env_vars.mu)

    def get_ctrl_ineqs(self, vars: ObjConstraintVars,
                       env_vars: EnvConstraintVars) -> casadi.SX:
        return self.get_wrench_ineqs(vars, env_vars)

    def get_wrench_ineqs(self, vars: ObjConstraintVars, env_vars: EnvConstraintVars):
        return self.stick_fric_func(vars.q_i_obj,
                                     vars.p_i_obj,
                                     vars.R_i_obj,
                                     vars.p_bnds,
                                     env_vars.fp_p,
                                     env_vars.fp_R,
                                     env_vars.fp_bnds,
                                     vars.Wb_obj,
                                     env_vars.mu,
                                     vars.active)
