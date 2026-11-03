import casadi
from dataclasses import dataclass
from abc import ABC, abstractmethod
                               
from faro.ocp.ik_contact.kin_constr_funcs import plane_bounded_plane_res_param


@dataclass
class RobConstraintVars:
    p_i_ee: casadi.SX
    R_i_ee: casadi.SX
    p_bnds: casadi.SX
    active: casadi.SX


@dataclass
class EnvConstraintVars:
    fp_p: casadi.SX
    fp_R: casadi.SX
    fp_bnds: casadi.SX


class BaseContactConstraint(ABC):
    def __init__(self):
        self.has_kin_eqs = False
        self.has_kin_ineqs = False

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


class RobotPatchEnvPatchConstraint(BaseContactConstraint):
    def __init__(self):
        super().__init__()

        self.has_kin_eqs    = True
        self.has_kin_ineqs  = True
        
        self.stick_bounded_plane_func = plane_bounded_plane_res_param()

    def get_kin_constraints(self, vars: RobConstraintVars,
                    env_vars: EnvConstraintVars):

        stick_plane_res, stick_plane_ineq = self.stick_bounded_plane_func(
            vars.p_i_ee,
            vars.R_i_ee,
            vars.p_bnds,
            env_vars.fp_p,
            env_vars.fp_R,
            env_vars.fp_bnds,
            vars.active,
        )

        return stick_plane_res, stick_plane_ineq

    def get_kin_eqs(self, vars: RobConstraintVars, env_vars: EnvConstraintVars):
        return self.get_kin_constraints(vars, env_vars)[0]

    def get_kin_ineqs(self, vars: RobConstraintVars, env_vars: EnvConstraintVars):
        return self.get_kin_constraints(vars, env_vars)[1]
