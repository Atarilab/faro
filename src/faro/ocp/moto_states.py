"""Physical configuration states with MOTO tangent-space increments."""

from dataclasses import dataclass

import casadi as cs
import moto
import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin

from faro.ocp.configuration import normalize_pose_trajectory


@dataclass(frozen=True)
class StateLayout:
    """Storage slices; configuration increments have ``nv`` entries."""

    nq: int
    nv: int
    centroidal: bool = False

    @property
    def q(self):
        return slice(0, self.nq)

    @property
    def v(self):
        return slice(self.nq, self.nq + self.nv)

    @property
    def r(self):
        return slice(self.nq + self.nv, self.nq + self.nv + 3)

    @property
    def h(self):
        return slice(self.nq + self.nv + 3, self.nq + self.nv + 9)

    @property
    def dim(self):
        return self.nq + self.nv + (9 if self.centroidal else 0)

    @property
    def tdim(self):
        return 2 * self.nv + (9 if self.centroidal else 0)


def configuration_states(name, model=None):
    """Create current/next robot configurations, or SE(3) poses without a model."""
    if not hasattr(moto, "casadi_manifold"):
        raise RuntimeError(
            "TO requires Moto casadi_manifold; rebuild and install extern/moto "
            "in the active environment following install.md."
        )
    nq, nv = (model.nq, model.nv) if model is not None else (7, 6)
    base = cs.SX.sym(f"{name}_base", nq)
    other = cs.SX.sym(f"{name}_other", nq)
    step = cs.SX.sym(f"{name}_step", nv)
    if model is None:
        group = cpin.liegroups.SE3()
        integrated = group.integrate(base, step)
        delta = group.difference(base, other)
        default = np.array([0., 0., 0., 0., 0., 0., 1.])
    else:
        cmodel = cpin.Model(model)
        integrated = cpin.integrate(cmodel, base, step)
        delta = cpin.difference(cmodel, base, other)
        default = pin.neutral(model)
    return moto.casadi_manifold.create(
        name, base, step, integrated, other, delta, default
    )


def difference(configuration, reference, value):
    """Tangent displacement from reference to value (Pinocchio argument order)."""
    return configuration.symbolic_difference(value, reference)
