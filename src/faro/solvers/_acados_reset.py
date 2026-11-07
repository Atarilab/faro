"""Reset Acados iterates, including constraint-heavy zero-horizon problems."""
from ctypes import c_void_p

import numpy as np


def reset_acados_solver(solver, *, horizon):
    """Clear primal/dual variables and QP memory without changing parameters.

    The bundled generated reset function sizes its scratch buffer with each
    terminal constraint once, but reads two multipliers per constraint. For
    N=0 this can read beyond the buffer. Use dimensioned public setters and
    the same Acados C QP-reset routine for a static problem instead.
    """
    if horizon != 0:
        solver.reset()
        return
    # A terminal-only problem has no controls, algebraics, or dynamics duals.
    for field in ('x', 'sl', 'su', 'lam'):
        values = solver.get(0, field)
        if values.size:
            solver.set(0, field, np.zeros_like(values))
    reset_qp = solver.acados_lib.ocp_nlp_solver_reset_qp_memory
    reset_qp.argtypes = [c_void_p, c_void_p, c_void_p]
    reset_qp.restype = None
    reset_qp(solver.nlp_solver, solver.nlp_in, solver.nlp_out)
