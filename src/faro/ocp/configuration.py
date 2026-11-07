"""Physical XYZ/XYZW configurations and their tangent-space errors."""

import casadi as cs
import numpy as np

from faro.ocp.utils import gtsam_logmap, quaternion_rotation


def configuration_difference(reference, value):
    """Local tangent displacement, invariant to q/-q, without normalization.

    Use the custom SO(3) log and inverse left Jacobian for the free-flyer,
    followed by scalar joint differences (the layout required by RobotConfig).
    Inputs represent unit quaternions. KSO enforces this through
    initialization and unit-norm constraints; trial values are evaluated as
    supplied.
    """
    reference_rotation = quaternion_rotation(reference[3:7])
    rotation = quaternion_rotation(value[3:7])
    omega = gtsam_logmap(reference_rotation.T @ rotation)
    translation = reference_rotation.T @ (value[:3] - reference[:3])

    # J_left(omega)^-1 = I - [omega]/2 + a*[omega]^2. The half-angle
    # expression stays regular at pi; its series avoids cancellation at zero.
    theta2 = cs.dot(omega, omega)
    small = theta2 < 1e-6
    safe_theta2 = cs.if_else(small, 1.0, theta2)
    half_theta = 0.5 * cs.sqrt(safe_theta2)
    a = cs.if_else(
        small,
        1.0 / 12.0 + theta2 / 720.0 + theta2 * theta2 / 30240.0,
        (1.0 - half_theta * cs.cos(half_theta) / cs.sin(half_theta)) / safe_theta2,
    )
    cross = cs.cross(omega, translation)
    linear = translation - 0.5 * cross + a * cs.cross(omega, cross)
    return cs.vertcat(linear, omega, value[7:] - reference[7:])


def normalize_pose_trajectory(values, *, name):
    """Prepare initial guesses with unit xyzw quaternions, preserving sign."""
    values = np.array(values, dtype=float, copy=True)
    quaternions = values[:, 3:7]
    # Scaling first also avoids overflow for finite but very large inputs.
    scales = np.max(np.abs(quaternions), axis=1)
    if np.any(scales == 0):
        raise ValueError(f"{name}: zero quaternion in physical state trajectory")
    quaternions /= scales[:, None]
    quaternions /= np.linalg.norm(quaternions, axis=1)[:, None]
    return values


def align_pose_trajectory(values, reference):
    """Align copied trajectory quaternion signs with corresponding references."""
    values = np.array(values, dtype=float, copy=True)
    flip = np.sum(values[:, 3:7] * reference[:, 3:7], axis=1) < 0
    values[flip, 3:7] *= -1
    return values
