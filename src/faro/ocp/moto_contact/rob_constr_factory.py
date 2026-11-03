import casadi

from dataclasses import dataclass
from abc import ABC, abstractmethod

from faro.ocp.moto_contact.force_constr_funcs import (
    sliding_patch_friction,
    zero_wrench_int_switch,
    ee_patch_env_force_constr,
)

from faro.ocp.moto_contact.kin_constr_funcs import plane_bounded_plane_res_param


@dataclass
class RobConstraintVars:
    """V_i_ee is WORLD; W_rob is world-aligned [force, moment] about p_i_ee."""
    p_i_ee: casadi.SX
    R_i_ee: casadi.SX
    V_i_ee: casadi.SX
    p_n_ee: casadi.SX
    R_n_ee: casadi.SX
    p_bnds: casadi.SX
    W_rob: casadi.SX
    active: casadi.SX
    is_int_switch: casadi.SX

@dataclass
class EnvConstraintVars:
    """Target geometry in world coordinates and WORLD spatial twist fp_V."""
    fp_p: casadi.SX
    fp_R: casadi.SX
    fp_V: casadi.SX
    fp_pn: casadi.SX
    fp_Rn: casadi.SX
    fp_bnds: casadi.SX
    mu: casadi.SX
    containment_margin: casadi.SX = 0.05


class BaseContactConstraint(ABC):
    def __init__(self):
        self.has_kin_eqs = False
        self.has_kin_ineqs = False
        self.has_ctrl_eqs = False
        self.has_ctrl_ineqs = False

    @abstractmethod
    def get_kin_eqs(self, vars: RobConstraintVars,
                    env_vars: EnvConstraintVars) -> casadi.SX:
        """
        Return a CasADi SX vector that must be driven to 0.
        """
        pass

    @abstractmethod
    def get_kin_ineqs(self, vars: RobConstraintVars,
                      env_vars: EnvConstraintVars) -> casadi.SX:
        """
        Return a CasADi SX vector g(x) such that g(x) >= 0.
        """
        pass

    @abstractmethod
    def get_ctrl_eqs(self, vars: RobConstraintVars,
                     env_vars: EnvConstraintVars) -> casadi.SX:
        """
        Return a CasADi SX vector that must be driven to 0.
        """
        pass

    @abstractmethod
    def get_ctrl_ineqs(self, vars: RobConstraintVars,
                       env_vars: EnvConstraintVars) -> casadi.SX:
        """
        Return a CasADi SX vector g(x) such that g(x) >= 0.
        """
        pass

class RobotPatchEnvPatchConstraint(BaseContactConstraint):
    def __init__(self):
        super().__init__()

        self.has_kin_eqs    = True
        self.has_kin_ineqs  = True
        self.has_ctrl_eqs   = True
        self.has_ctrl_ineqs = True
        
        self.stick_bounded_plane_func = plane_bounded_plane_res_param(parameterized_margin=True)

        self.stick_fric_func = ee_patch_env_force_constr()

        self.zero_wrench_int_switch = zero_wrench_int_switch()
        self.sliding_fric_func = sliding_patch_friction()

    def get_kin_constraints(self, vars: RobConstraintVars,
                    env_vars: EnvConstraintVars):

        stick_plane_res, stick_plane_ineq = self.stick_bounded_plane_func(
            vars.p_i_ee,
            vars.R_i_ee,
            vars.p_n_ee,
            vars.R_n_ee,
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

    def get_kin_eqs(self, vars: RobConstraintVars, env_vars: EnvConstraintVars):
        return self.get_kin_constraints(vars, env_vars)[0]

    def get_kin_ineqs(self, vars: RobConstraintVars, env_vars: EnvConstraintVars):
        return self.get_kin_constraints(vars, env_vars)[1]

    def get_sliding_kin_eqs(self, vars: RobConstraintVars, env_vars: EnvConstraintVars):
        # Keep normal contact and tilt; in-plane translation and yaw may change.
        return self.get_kin_constraints(vars, env_vars)[0][2:5]
            
    def get_ctrl_eqs(self, vars: RobConstraintVars,
                     env_vars: EnvConstraintVars) -> casadi.SX:
        return self.get_wrench_eqs(vars, env_vars)

    def get_wrench_eqs(self, vars: RobConstraintVars, env_vars: EnvConstraintVars):
        return self.zero_wrench_int_switch(vars.W_rob, vars.is_int_switch)

    def get_sliding_constraints(self, vars: RobConstraintVars, env_vars: EnvConstraintVars):
        return self.sliding_fric_func(vars.p_i_ee, env_vars.fp_R,
            vars.V_i_ee, env_vars.fp_V, vars.W_rob[:3], env_vars.mu)

    def get_ctrl_ineqs(self, vars: RobConstraintVars,
                    env_vars: EnvConstraintVars) -> casadi.SX:
        return self.get_wrench_ineqs(vars, env_vars)

    def get_wrench_ineqs(self, vars: RobConstraintVars, env_vars: EnvConstraintVars):
        return self.stick_fric_func(vars.p_i_ee,
                                     vars.R_i_ee,
                                     vars.p_bnds,
                                     env_vars.fp_p,
                                     env_vars.fp_R,
                                     env_vars.fp_bnds,
                                     vars.W_rob,
                                     env_vars.mu,
                                     vars.active)
