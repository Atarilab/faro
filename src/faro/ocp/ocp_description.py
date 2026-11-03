from dataclasses import dataclass, field
from typing import List, Optional, Dict, Literal
import casadi as cs
import numpy as np

ConstraintRole = Literal["initial", "path", "terminal", "stage"]

@dataclass
class ConstraintSpec:
    name: str
    fun: cs.Function          # h(x, u, p)
    lb: np.ndarray
    ub: np.ndarray
    role: ConstraintRole      # 'initial' | 'path' | 'terminal' | 'stage'


class ConstraintBuilder:
    """Wrap expressions once while preserving separate specs for stage roles."""

    def __init__(self, x, u, p):
        self.inputs = [x, u, p]
        self.specs = []
        self._functions = {}

    def _add(self, name, expr, lb, ub, role):
        # Names can repeat for different terminal expressions. Only reuse a
        # function when both its name and its expression object are identical.
        key = (name, id(expr))
        if key not in self._functions:
            self._functions[key] = (
                expr, cs.Function(name, self.inputs, [expr]).expand()
            )
        self.specs.append(ConstraintSpec(
            name=name, fun=self._functions[key][1],
            lb=np.array(lb, dtype=float), ub=np.array(ub, dtype=float), role=role,
        ))

    def equalities(self, expressions, role):
        for name, expr in expressions.items():
            self._add(name, expr, np.zeros(expr.shape[0]), np.zeros(expr.shape[0]), role)

    def inequalities(self, expressions, role):
        for constraint in expressions.values():
            self._add(constraint["name"], constraint["expr"],
                      constraint["lb"], constraint["ub"], role)

@dataclass
class LinearBound:
    name: str
    var_type: Literal["x", "u"]
    indices: List[int]          # indices in x or u vector
    lb: np.ndarray
    ub: np.ndarray

@dataclass
class OCPDescription:
    # symbols (stage-wise)
    x: cs.SX                  # state symbol
    u: cs.SX                  # input symbol
    p: cs.SX                  # parameter symbol
    x_dict: Dict[str, cs.SX]
    u_dict: Dict[str, cs.SX]
    p_dict: Dict[str, cs.SX]

    f_expl: Optional[cs.Function]   # f(x, u, p) = xdot, if explicit
    f_impl: Optional[cs.Function]   # f(x, xdot, u, p) = 0, if implicit
    f_discr: Optional[cs.Function]  # x_next

    r_path: Optional[cs.Function]   # r(x, u, p) for stage
    r_terminal: Optional[cs.Function]  # r_e(x, p) for terminal

    constraints: List[ConstraintSpec]
    linear_bounds: List[LinearBound] = field(default_factory=list)
