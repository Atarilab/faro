import casadi
import numpy as np


def logmap_1(R11, R12, R13, R21, R22, R23, R31, R32, R33):
    cond = (R33 > R22) * (R33 > R11)
    # R33 is the largest diagonal, a=3, b=1, c=2
    W = R21 - R12
    Q1 = 2.0 + 2.0 * R33
    Q2 = R31 + R13
    Q3 = R23 + R32
    r = casadi.sqrt(casadi.fmax(Q1, 1e-16))
    one_over_r = 1 / r
    norm = casadi.sqrt(Q1*Q1 + Q2*Q2 + Q3*Q3 + W*W + 1e-16)
    sgn_w = casadi.if_else(W < 0, -1.0, 1.0)
    mag = np.pi - (2 * sgn_w * W) / norm
    scale = 0.5 * one_over_r * mag
    omega = sgn_w * scale * casadi.vertcat(Q2, Q3, Q1)
    return omega, cond


def logmap_2(R11, R12, R13, R21, R22, R23, R31, R32, R33):
    cond = (R22 > R11)
    # R22 is the largest diagonal, a=2, b=3, c=1
    W = R13 - R31
    Q1 = 2.0 + 2.0 * R22
    Q2 = R23 + R32
    Q3 = R12 + R21
    r = casadi.sqrt(casadi.fmax(Q1, 1e-16))
    one_over_r = 1 / r
    norm = casadi.sqrt(Q1*Q1 + Q2*Q2 + Q3*Q3 + W*W + 1e-16)
    sgn_w = casadi.if_else(W < 0, -1.0, 1.0)
    mag = np.pi - (2 * sgn_w * W) / norm
    scale = 0.5 * one_over_r * mag
    omega = sgn_w * scale * casadi.vertcat(Q3, Q1, Q2)
    return omega, cond


def logmap_3(R11, R12, R13, R21, R22, R23, R31, R32, R33):
    cond = None
    # R11 is the largest diagonal, a=1, b=2, c=3
    W = R32 - R23
    Q1 = 2.0 + 2.0 * R11
    Q2 = R12 + R21
    Q3 = R31 + R13
    r = casadi.sqrt(casadi.fmax(Q1, 1e-16))
    one_over_r = 1 / r
    norm = casadi.sqrt(Q1*Q1 + Q2*Q2 + Q3*Q3 + W*W + 1e-16)
    sgn_w = casadi.if_else(W < 0, -1.0, 1.0)
    mag = np.pi - (2 * sgn_w * W) / norm
    scale = 0.5 * one_over_r * mag
    omega = sgn_w * scale * casadi.vertcat(Q1, Q2, Q3)
    return omega, cond


def logmap_almost_general_case(R11, R12, R13, R21, R22, R23, R31, R32, R33):
    tr = R11 + R22 + R33
    tr_3 = tr - 3.0  # could be non-negative if the matrix is off orthogonal
    magnitude = 0.5 - tr_3 / 12.0 + tr_3*tr_3/60.0
    omega = magnitude * casadi.vertcat(R32 - R23, R13 - R31, R21 - R12)
    return omega


def logmap_general_case(R11, R12, R13, R21, R22, R23, R31, R32, R33):
    # this is the normal case -1 < trace < 3
    tr = R11 + R22 + R33
    # CasADi also differentiates inactive branches. Supply an interior acos
    # argument there so identity, half turns and off-orthogonal SQP trials
    # never evaluate the unused singular expression.
    selected = casadi.logic_and(tr + 1.0 >= 1e-3, tr - 3.0 < -1e-6)
    cosine = casadi.if_else(selected, (tr - 1.0) / 2, 0.0)
    theta = casadi.acos(cosine)
    magnitude = theta / (2.0 * casadi.sin(theta))
    omega = magnitude * casadi.vertcat(R32 - R23, R13 - R31, R21 - R12)
    return omega


def gtsam_logmap(R):
    """GTSAM-based SO(3) log with finite inactive CasADi branches.

    Keep the small-angle series and dominant-diagonal near-pi branches from
    the original adaptation. At an exact half turn choose a nonzero sign,
    matching GTSAM's W < 0 convention; sign(0) would erase the rotation error.
    The shortest rotation vector still has its unavoidable branch cut at pi.

    Reference: https://github.com/borglab/gtsam/blob/4.2/gtsam/geometry/SO3.cpp
    """
    # Contact selectors use a zero matrix for an inactive target.
    R = casadi.if_else(casadi.sumsqr(R) > 1e-24, R, casadi.SX.eye(3))
    R11 = R[0, 0]
    R12 = R[0, 1]
    R13 = R[0, 2]
    R21 = R[1, 0]
    R22 = R[1, 1]
    R23 = R[1, 2]
    R31 = R[2, 0]
    R32 = R[2, 1]
    R33 = R[2, 2]

    # Get trace(R)
    tr = R11 + R22 + R33

    logmap1_val, logmap1_constraint = logmap_1(
        R11=R11, R12=R12, R13=R13, R21=R21, R22=R22, R23=R23, R31=R31, R32=R32, R33=R33
    )
    logmap2_val, logmap2_constraint = logmap_2(
        R11=R11, R12=R12, R13=R13, R21=R21, R22=R22, R23=R23, R31=R31, R32=R32, R33=R33
    )
    logmap3_val, _ = logmap_3(
        R11=R11, R12=R12, R13=R13, R21=R21, R22=R22, R23=R23, R31=R31, R32=R32, R33=R33
    )
    logmap_general_case_val = logmap_general_case(R11=R11, R12=R12, R13=R13, R21=R21, R22=R22, R23=R23, R31=R31, R32=R32, R33=R33)
    logmap_almost_general_case_val = logmap_almost_general_case(R11=R11, R12=R12, R13=R13, R21=R21, R22=R22, R23=R23, R31=R31, R32=R32, R33=R33)
    omega = casadi.if_else(
        tr + 1.0 < 1e-3,
        casadi.if_else(
            logmap1_constraint,
            logmap1_val,
            casadi.if_else(logmap2_constraint, logmap2_val, logmap3_val),
        ),
        casadi.if_else(
            tr - 3 < -1e-6,
            logmap_general_case_val,
            logmap_almost_general_case_val,
        ),
    )
    return omega


def quaternion_rotation(quaternion):
    """Polynomial XYZW-to-rotation map; evaluate raw values without normalization."""
    x, y, z, w = (quaternion[i] for i in range(4))
    return casadi.vertcat(
        casadi.horzcat(1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)),
        casadi.horzcat(2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)),
        casadi.horzcat(2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)),
    )


def pose_error(q, reference):
    """World translation and target-frame rotation error for XYZ + XYZW poses."""
    rotation = quaternion_rotation(q[3:7])
    reference_rotation = quaternion_rotation(reference[3:7])
    return casadi.vertcat(q[:3] - reference[:3],
                         gtsam_logmap(reference_rotation.T @ rotation))
