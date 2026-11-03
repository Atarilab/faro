import math
from typing import Tuple

import numpy as np
import casadi as ca

def get_plane_features(plane_size):

    Rvec = np.eye(3).reshape(-1, order='F')

    xybnds = np.asarray(plane_size, dtype=float).reshape(2,)
    bnds = np.array([xybnds[0], xybnds[1],
                     xybnds[0], xybnds[1]], dtype=float)

    plane_features = np.hstack((np.zeros(3),
                                Rvec,
                                bnds))
    return plane_features

def plane_features_func():
    plane_size = ca.SX.sym("plane_size", 2, 1)

    Rvec = ca.reshape(ca.DM.eye(3), 9, 1)

    xybnds = plane_size
    bnds = ca.vertcat(xybnds[0], xybnds[1], xybnds[0], xybnds[1])

    plane_features = ca.vertcat(ca.SX.zeros(3, 1), Rvec, bnds)

    return ca.Function(
        "plane_features_func",
        [plane_size],
        [plane_features],
        ["plane_size"],
        ["plane_features"],
    )

# This order defines local face IDs (1..6).
BOX_FACES = ("top", "left", "front", "right", "back", "bottom")
BOX_FACE_ROTATIONS = np.array([
    np.eye(3),
    [[-1, 0, 0], [0, 0, 1], [0, 1, 0]],
    [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
    [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
    [[0, 0, -1], [-1, 0, 0], [0, 1, 0]],
    [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
], dtype=float)


def get_box_features(box_size):
    size = np.asarray(box_size)
    positions = np.column_stack([r[:, 2] * size for r in BOX_FACE_ROTATIONS])
    rotations = np.column_stack([r.reshape(9, order="F") for r in BOX_FACE_ROTATIONS])
    bounds = np.column_stack([np.abs(r.T @ size)[:2] for r in BOX_FACE_ROTATIONS])
    return np.vstack((positions, rotations, bounds, bounds))


def box_features_func():
    size = ca.SX.sym("box_size", 3)
    rotations = [ca.DM(r) for r in BOX_FACE_ROTATIONS]
    positions = ca.horzcat(*(r[:, 2] * size for r in rotations))
    bounds = ca.horzcat(*(ca.fabs(r.T @ size)[:2] for r in rotations))
    features = ca.vertcat(positions, ca.horzcat(*(ca.vec(r) for r in rotations)), bounds, bounds)
    return ca.Function("box_features_fun", [size], [features], ["box_size"], ["box_features"])

def box_inertia(half_lengths, mass):
    hx, hy, hz = half_lengths
    
    Ixx = (1.0 / 3.0) * mass * (hy**2 + hz**2)
    Iyy = (1.0 / 3.0) * mass * (hx**2 + hz**2)
    Izz = (1.0 / 3.0) * mass * (hx**2 + hy**2)
    
    return [Ixx, Iyy, Izz]

Extents = Tuple[float, float, float, float]

def _size_from_extents(ext: Extents) -> Tuple[float, float]:
    """Convert (posx, posy, negx, negy) extents to (width, height)."""
    posx, posy, negx, negy = ext
    return posx + negx, posy + negy


def _can_fit_rotated(
    inner_w: float,
    inner_h: float,
    outer_w: float,
    outer_h: float,
    steps: int = 720,
    eps: float = 1e-9,
) -> bool:
    """
    Check if a rectangle (inner_w x inner_h) can fit inside
    (outer_w x outer_h) for some rotation angle in [0, pi/2].

    Returns whether any sampled angle fits.
    """
    # quick axis-aligned checks (0° and 90°)
    if (inner_w <= outer_w + eps and inner_h <= outer_h + eps):
        return True
    if (inner_h <= outer_w + eps and inner_w <= outer_h + eps):
        return True

    # sweep angles between 0 and 90 degrees
    for i in range(steps + 1):
        theta = (math.pi / 2) * i / steps
        c = math.cos(theta)
        s = math.sin(theta)

        # bounding box size of rotated inner rectangle
        w_rot = inner_w * c + inner_h * s
        h_rot = inner_w * s + inner_h * c

        if w_rot <= outer_w + eps and h_rot <= outer_h + eps:
            return True

    return False


def rectangle_containment(
    extA: Extents,
    extB: Extents,
    steps: int = 720,
    eps: float = 1e-9,
) -> bool:
    """Whether either rectangle contains the other at a sampled rotation.

    Extents are (posx, posy, negx, negy). Mutually fitting rectangles are
    rejected if either has a width or height at most 1e-12.
    """
    wA, hA = _size_from_extents(extA)
    wB, hB = _size_from_extents(extB)

    A_in_B = _can_fit_rotated(wA, hA, wB, hB, steps=steps, eps=eps)
    B_in_A = _can_fit_rotated(wB, hB, wA, hA, steps=steps, eps=eps)

    if A_in_B and B_in_A:
        return not (wA <= 1e-12 or hA <= 1e-12 or wB <= 1e-12 or hB <= 1e-12)
    return A_in_B or B_in_A
