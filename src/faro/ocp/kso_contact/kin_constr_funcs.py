from functools import lru_cache

import casadi
import pinocchio.casadi as cpin
from faro.ocp.utils import gtsam_logmap

@lru_cache(maxsize=1)
def plane_bounded_plane_res_param():
    R_flip = casadi.SX.zeros(3,3)
    R_flip[0,0] = 1; R_flip[1,1] = -1; R_flip[2,2] = -1

    p = casadi.SX.sym("p", 3)
    R_ = casadi.SX.sym("R", 9)
    R = casadi.reshape(R_, 3, 3)
    p_bnds = casadi.SX.sym("p_bounds", 4)   # (posx, posy, negx, negy)

    fp_p = casadi.SX.sym("fp_p", 3)
    fp_R_ = casadi.SX.sym("fp_R", 9)
    fp_R = casadi.reshape(fp_R_, 3, 3)
    fp_bnds = casadi.SX.sym("fp_bnds", 4)   # (posx, posy, negx, negy)

    active = casadi.SX.sym("active", 1)

    # P = patch, E = target patch
    T_patch = cpin.SE3(R, p)
    T_env_patch_ = cpin.SE3(fp_R, fp_p)
    T_env_patch = cpin.SE3(fp_R @ R_flip, fp_p)

    T_E_P = T_env_patch.inverse() * T_patch         # flipped normal for orientation constraints
    T_E_P_ = T_env_patch_.inverse() * T_patch       # no flip, used for z and bounds
    T_P_E_ = T_patch.inverse() * T_env_patch_       # reverse transform (env -> patch)


    # --- Patch bounds in its own frame (as SE3 points, z = 0) ---
    fr_bnd_p = cpin.SE3(casadi.SX_eye(3), casadi.vertcat(p_bnds[0],  p_bnds[1], 0))
    fl_bnd_p = cpin.SE3(casadi.SX_eye(3), casadi.vertcat(p_bnds[0], -p_bnds[3], 0))
    rr_bnd_p = cpin.SE3(casadi.SX_eye(3), casadi.vertcat(-p_bnds[2],  p_bnds[1], 0))
    rl_bnd_p = cpin.SE3(casadi.SX_eye(3), casadi.vertcat(-p_bnds[2], -p_bnds[3], 0))

    patch_boundary_points = [fr_bnd_p, fl_bnd_p, rr_bnd_p, rl_bnd_p]

    # --- Env bounds in its own frame (as SE3 points, z = 0) ---
    fr_bnd_fp = cpin.SE3(casadi.SX_eye(3), casadi.vertcat(fp_bnds[0],  fp_bnds[1], 0))
    fl_bnd_fp = cpin.SE3(casadi.SX_eye(3), casadi.vertcat(fp_bnds[0], -fp_bnds[3], 0))
    rr_bnd_fp = cpin.SE3(casadi.SX_eye(3), casadi.vertcat(-fp_bnds[2],  fp_bnds[1], 0))
    rl_bnd_fp = cpin.SE3(casadi.SX_eye(3), casadi.vertcat(-fp_bnds[2], -fp_bnds[3], 0))

    env_boundary_points = [fr_bnd_fp, fl_bnd_fp, rr_bnd_fp, rl_bnd_fp]

    # Case 1: patch inside env
    bf_x_p_in_env = []
    bf_y_p_in_env = []

    for boundary_point in patch_boundary_points:
        patch_bnd_in_env = T_E_P_ * boundary_point
        bf_x_p_in_env.append(patch_bnd_in_env.translation[0])
        bf_y_p_in_env.append(patch_bnd_in_env.translation[1])

    inequalities_p_in_env_ = []
    for x in bf_x_p_in_env:
        inequalities_p_in_env_.append(active * (-x + fp_bnds[0]))
        inequalities_p_in_env_.append(active * (x - (-fp_bnds[2])))

    for y in bf_y_p_in_env:
        inequalities_p_in_env_.append(active * (-y + fp_bnds[1]))
        inequalities_p_in_env_.append(active * (y - (-fp_bnds[3])))

    inequalities_p_in_env = casadi.vertcat(*inequalities_p_in_env_)

    # Case 2: env inside patch (roles swapped)
    bf_x_env_in_p = []
    bf_y_env_in_p = []

    for boundary_point in env_boundary_points:
        env_bnd_in_patch = T_P_E_ * boundary_point
        bf_x_env_in_p.append(env_bnd_in_patch.translation[0])
        bf_y_env_in_p.append(env_bnd_in_patch.translation[1])

    inequalities_env_in_p_ = []
    for x in bf_x_env_in_p:
        inequalities_env_in_p_.append(active * (-x + p_bnds[0]))
        inequalities_env_in_p_.append(active * (x - (-p_bnds[2])))

    for y in bf_y_env_in_p:
        inequalities_env_in_p_.append(active * (-y + p_bnds[1]))
        inequalities_env_in_p_.append(active * (y - (-p_bnds[3])))

    inequalities_env_in_p = casadi.vertcat(*inequalities_env_in_p_)

    sum_p   = casadi.sum1(p_bnds)
    sum_fp  = casadi.sum1(fp_bnds)
    patch_is_smaller = sum_p <= sum_fp

    inequalities = casadi.if_else(patch_is_smaller,
                                  inequalities_p_in_env,
                                  inequalities_env_in_p)

    equalities = casadi.vertcat(
        active * (T_E_P_.translation[2]),
        active * gtsam_logmap(T_E_P.rotation)[:2],
    )

    return casadi.Function(
        "plane_to_bounded_fix_plane_residual_param",
        [p, R_, p_bnds,
         fp_p, fp_R_,fp_bnds, active],
        [equalities, inequalities],
        ["p", "R", "p_bnds",
         "fp_p", "fp_R_", "fp_bnds", "active"],
        ["equalities", "inequalities"],
    )

@lru_cache(maxsize=1)
def relative_position_repeat():
    p = casadi.SX.sym("p", 3)
    R_ = casadi.SX.sym("R", 9)
    R = casadi.reshape(R_, 3, 3)

    p_next = casadi.SX.sym("p_next", 3)
    R_next_ = casadi.SX.sym("R_next_", 9)
    R_next = casadi.reshape(R_next_, 3, 3)

    fp_p = casadi.SX.sym("fp_p", 3)
    fp_R_ = casadi.SX.sym("fp_R", 9)
    fp_R = casadi.reshape(fp_R_, 3, 3)

    fp_p_next = casadi.SX.sym("fp_p_next", 3)
    fp_R_next_ = casadi.SX.sym("fp_R_next_", 9)
    fp_R_next = casadi.reshape(fp_R_next_, 3, 3)

    repeat = casadi.SX.sym("repeat", 1)

    T_patch = cpin.SE3(R, p)
    T_patch_next = cpin.SE3(R_next, p_next)
    
    T_env_patch = cpin.SE3(fp_R, fp_p)
    T_env_patch_next = cpin.SE3(fp_R_next, fp_p_next)

    T_rel = T_env_patch.inverse() * T_patch
    T_rel_next = T_env_patch_next.inverse() * T_patch_next

    equalities = casadi.vertcat(
        repeat * (T_rel_next.translation - T_rel.translation),
        repeat * gtsam_logmap(T_rel_next.rotation.T @ T_rel.rotation),
    )

    return casadi.Function(
        "relative_position_repeat_res",
        [p, R_, p_next, R_next_,
         fp_p, fp_R_, fp_p_next, fp_R_next_, repeat],
        [equalities],
        ["p", "R_", "p_next", "R_next_",
         "fp_p", "fp_R_", "fp_p_next", "fp_R_next_", "repeat"],
        ["equalities"],
    )
