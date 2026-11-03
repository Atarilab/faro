import casadi
import numpy as np

def logmap_1(R11, R12, R13, R21, R22, R23, R31, R32, R33):
    cond = (R33 > R22) * (R33 > R11)
    # R33 is the largest diagonal, a=3, b=1, c=2
    W = R21 - R12
    Q1 = 2.0 + 2.0 * R33
    Q2 = R31 + R13
    Q3 = R23 + R32
    r = casadi.sqrt(Q1 + 1e-16)
    one_over_r = 1 / r
    norm = casadi.sqrt(Q1*Q1 + Q2*Q2 + Q3*Q3 + W*W + 1e-16)
    sgn_w = casadi.sign(W)
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
    r = casadi.sqrt(Q1 + 1e-16)
    one_over_r = 1 / r
    norm = casadi.sqrt(Q1*Q1 + Q2*Q2 + Q3*Q3 + W*W + 1e-16)
    sgn_w = casadi.sign(W)
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
    r = casadi.sqrt(Q1 + 1e-16)
    one_over_r = 1 / r
    norm = casadi.sqrt(Q1*Q1 + Q2*Q2 + Q3*Q3 + W*W + 1e-16)
    sgn_w = casadi.sign(W)
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
    theta = casadi.acos((tr - 1.0) / 2)
    magnitude = theta / (2.0 * casadi.sin(theta))
    omega = magnitude * casadi.vertcat(R32 - R23, R13 - R31, R21 - R12)
    return omega


def gtsam_logmap(R):
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


def pose_error(q, reference):
    """World translation and target-frame rotation error for XYZ + XYZW poses."""
    import pinocchio.casadi as cpin

    rotation = cpin.XYZQUATToSE3(q[:7]).rotation
    reference_rotation = cpin.XYZQUATToSE3(reference[:7]).rotation
    return casadi.vertcat(q[:3] - reference[:3],
                         gtsam_logmap(reference_rotation.T @ rotation))
