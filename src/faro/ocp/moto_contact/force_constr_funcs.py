from functools import lru_cache

import casadi
import pinocchio.casadi as cpin


@lru_cache(maxsize=1)
def linearized_friction_cone():
    """Four-facet friction pyramid in contact axes; inequalities are >= 0.

    The bounds are affine in force for fixed mu, including at zero force.
    This square pyramid is an outer approximation of the circular cone.
    """
    force = casadi.SX.sym("force", 3)
    mu = casadi.SX.sym("mu")
    fx, fy, fz = force[0], force[1], force[2]
    ineqs = casadi.vertcat(
        mu * fz - fx,  # +x facet: fx <= mu * fz.
        mu * fz + fx,  # -x facet: fx >= -mu * fz.
        mu * fz - fy,  # +y facet: fy <= mu * fz.
        mu * fz + fy,  # -y facet: fy >= -mu * fz.
        fz,           # Unilateral contact: normal force >= 0.
    )
    return casadi.Function("linearized_friction_cone", [force, mu], [ineqs],
                           ["force", "mu"], ["ineqs"])


@lru_cache(maxsize=1)
def sliding_patch_friction():
    """Saturated Coulomb friction paired with the shared linear friction cone.

    Both velocities are WORLD spatial twists (linear part at the world origin).
    Shift their difference to the source patch before rotating into the target.

    Tangential force saturates the circular Coulomb cone in target axes.
    The shared wrench cone supplies normal-force, CoP and torsional bounds.
    Direction constraints are bilinear and saturation is quadratic.
    Inequalities use >= 0. The squared-speed row is redundant but explicitly
    permits zero speed and every tangential direction, without a speed norm.
    """
    p = casadi.SX.sym("p", 3)
    R_env = casadi.SX.sym("R_env", 9)
    V = casadi.SX.sym("V", 6)
    V_env = casadi.SX.sym("V_env", 6)
    force = casadi.SX.sym("force", 3)
    mu = casadi.SX.sym("mu")
    R = casadi.reshape(R_env, 3, 3)  # Target axes in WORLD.
    relative = V - V_env  # Relative WORLD spatial twist.
    velocity = R.T @ (relative[:3] + casadi.cross(relative[3:], p))  # Point slip in target axes.
    v_t = velocity[:2]  # Tangential slip: exclude the normal component.
    f = R.T @ force
    f_t, f_n = f[:2], f[2]  # Tangential and normal force in target axes.

    eqs = casadi.vertcat(
        v_t[0] * f_t[1] - v_t[1] * f_t[0],  # Collinearity: force is parallel to slip.
        casadi.dot(f_t, f_t) - (mu * f_n) ** 2,  # Saturation: the force lies on the friction cone.
    )

    ineqs = casadi.vertcat(
        -casadi.dot(f_t, v_t),  # Opposition: friction does no positive work.
        casadi.dot(v_t, v_t),  # Nonnegative squared speed: allows rest and any slip direction.
    )
    return casadi.Function("sliding_patch_friction",
        [p, R_env, V, V_env, force, mu], [eqs, ineqs],
        ["p", "R_env", "V", "V_env", "force", "mu"],
        ["eqs", "ineqs"])


@lru_cache(maxsize=1)
def zero_wrench_int_switch() -> casadi.Function:
    wrench = casadi.SX.sym("f_contact", 6)
    is_int_switch = casadi.SX.sym("is_int_switch", 1)

    return casadi.Function(
        "ee_free",
        [wrench, is_int_switch], [is_int_switch * wrench],
        ["wrench", "is_int_switch"], ["zero_wrench_int_switch"],
    )

@lru_cache(maxsize=1)
def zero_wrench_impact() -> casadi.Function:
    wrench = casadi.SX.sym("f_contact", 6)
    active = casadi.SX.sym("active", 1)
    impact = casadi.SX.sym("impact", 1)

    return casadi.Function(
        "ee_free",
        [wrench, active, impact], [(1 - impact) * (1 - active) * wrench],
        ["wrench", "active", "impact"], ["zero_wrench"],
    )

@lru_cache(maxsize=1)
def ee_patch_env_force_constr():
    # Set to False to use the eight rectangular yaw inequalities.
    use_simple_torsion = True

    p_patch = casadi.SX.sym("p", 3)
    R_patch_ = casadi.SX.sym("R", 9)
    R_patch = casadi.reshape(R_patch_, 3, 3)
    patch_bnds = casadi.SX.sym("p_bounds", 4)  # (posx, posy, negx, negy)

    fp_p = casadi.SX.sym("fp_p", 3)
    fp_R_ = casadi.SX.sym("fp_R", 9)
    fp_R = casadi.reshape(fp_R_, 3, 3)
    fp_bnds = casadi.SX.sym("fp_bnds", 4)      # (posx, posy, negx, negy)

    W_ee_ = casadi.SX.sym("object_body_wrench", 6)

    mu = casadi.SX.sym("mu", 1)
    active = casadi.SX.sym("active", 1)

    # pin.Force() convention is [f, m]
    W_ee = cpin.Force(linear=W_ee_[:3], angular=W_ee_[3:])

    R_flip = casadi.SX.zeros(3, 3)
    R_flip[0,0] = 1; R_flip[1,1] = -1; R_flip[2,2] = -1

    # Patch contact frame: origin at 0, orientation R_patch @ R_flip
    patch_SE3 = cpin.SE3(R_patch @ R_flip, casadi.SX.zeros(3))

    # Target frame relative to the EE origin, with world rotation fp_R
    patch_fp_SE3 = cpin.SE3(fp_R, fp_p - p_patch)

    # Wrench in patch frame
    W_patch = patch_SE3.inverse() * W_ee
    fx_patch = W_patch.linear[0]
    fy_patch = W_patch.linear[1]
    fz_patch = W_patch.linear[2]
    mx_patch = W_patch.angular[0]
    my_patch = W_patch.angular[1]
    mz_patch = W_patch.angular[2]

    # Wrench in fp frame
    W_fp = patch_fp_SE3.inverse() * W_ee
    fx_fp = W_fp.linear[0]
    fy_fp = W_fp.linear[1]
    fz_fp = W_fp.linear[2]
    mx_fp = W_fp.angular[0]
    my_fp = W_fp.angular[1]
    mz_fp = W_fp.angular[2]

    # Decide which box/frame is active based on sum of extents
    sum_patch = casadi.sum1(patch_bnds)
    sum_fp    = casadi.sum1(fp_bnds)
    fp_is_smaller = sum_fp <= sum_patch

    # Select wrench components in the corresponding frame
    fx = casadi.if_else(fp_is_smaller, fx_fp, fx_patch)
    fy = casadi.if_else(fp_is_smaller, fy_fp, fy_patch)
    fz = casadi.if_else(fp_is_smaller, fz_fp, fz_patch)
    mx = casadi.if_else(fp_is_smaller, mx_fp, mx_patch)
    my = casadi.if_else(fp_is_smaller, my_fp, my_patch)
    mz = casadi.if_else(fp_is_smaller, mz_fp, mz_patch)

    # Select bounds corresponding to chosen frame
    # Flipping the source y/z axes swaps its positive/negative y bounds.
    eff_bnds = casadi.if_else(fp_is_smaller, fp_bnds, patch_bnds[[0, 3, 2, 1]])
    bx_pos = eff_bnds[0]
    by_pos = eff_bnds[1]
    bx_neg = eff_bnds[2]
    by_neg = eff_bnds[3]

    # Sticking and sliding contacts share the friction and CoP bounds.
    friction = active * linearized_friction_cone()(casadi.vertcat(fx, fy, fz), mu)

    if use_simple_torsion:
        # Simplified yaw bounds in the selected contact frame.
        torsion = casadi.vertcat(
            active * (mu * fz - mz),
            active * (mu * fz + mz),
        )
    else:
        # Center and half-extents of the selected, possibly asymmetric rectangle.
        cx = 0.5 * (bx_pos - bx_neg)
        cy = 0.5 * (by_pos - by_neg)
        X = 0.5 * (bx_pos + bx_neg)
        Y = 0.5 * (by_pos + by_neg)

        # Shift moments to the rectangle center: m_center = m_origin - c x f.
        mx_c = mx - cy * fz
        my_c = my + cx * fz
        mz_c = mz - cx * fy + cy * fx

        # Expand the full rectangular yaw bounds into eight affine inequalities.
        yaw_ineqs = []
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                yaw_ineqs.append(
                    active * (
                        mu * (X + Y) * fz
                        - mz_c
                        - sx * (Y * fx + mu * mx_c)
                        - sy * (X * fy + mu * my_c)
                    )
                )
                yaw_ineqs.append(
                    active * (
                        mu * (X + Y) * fz
                        + mz_c
                        - sx * (Y * fx - mu * mx_c)
                        - sy * (X * fy - mu * my_c)
                    )
                )
        torsion = casadi.vertcat(*yaw_ineqs)

    ineqs = casadi.vertcat(
        friction[:4],
        torsion,

        active * (my + fz * bx_pos),       # my ≥ -fz*bx_pos
        active * (fz * bx_neg - my),       # my ≤  fz*bx_neg
        active * (mx + fz * by_neg),       # mx ≥ -fz*by_neg
        active * (fz * by_pos - mx),       # mx ≤  fz*by_pos

        friction[4],                     # fz ≥ 0
    )

    return casadi.Function(
        "ee_patch_env_force_constr",
        [p_patch, R_patch_, patch_bnds,
         fp_p, fp_R_, fp_bnds,
         W_ee_, mu, active],
        [ineqs],
        ["p_patch", "R_patch_", "patch_bnds",
         "fp_p", "fp_R_", "fp_bnds",
         "W_ee_", "mu", "active"],
        ["ineqs"],
    )

@lru_cache(maxsize=1)
def obj_patch_env_force_constr():
    # Set to False to use the eight rectangular yaw inequalities.
    use_simple_torsion = True

    q_obj = casadi.SX.sym("q_obj", 7)

    p_patch = casadi.SX.sym("p", 3)
    R_patch_ = casadi.SX.sym("R", 9)
    R_patch = casadi.reshape(R_patch_, 3, 3)
    patch_bnds = casadi.SX.sym("p_bounds", 4)  # (posx, posy, negx, negy)

    # fixed plane pose + bounds
    fp_p = casadi.SX.sym("fp_p", 3)
    fp_R_ = casadi.SX.sym("fp_R", 9)
    fp_R = casadi.reshape(fp_R_, 3, 3)
    fp_bnds = casadi.SX.sym("fp_bnds", 4)      # (posx, posy, negx, negy)

    # Body wrench in dynamics convention [body moment, body force]
    Wb_obj = casadi.SX.sym("object_body_wrench", 6)

    mu = casadi.SX.sym("mu", 1)
    active = casadi.SX.sym("active", 1)

    # pin.Force() convention is [f, m]
    Wb_obj_pin = cpin.Force(linear=Wb_obj[3:], angular=Wb_obj[:3])

    R_flip = casadi.SX.zeros(3, 3)
    R_flip[0,0] = 1; R_flip[1,1] = -1; R_flip[2,2] = -1

    # Object pose
    obj_SE3 = cpin.XYZQUATToSE3(q_obj)

    # Contact frames (z pointing out of plane via R_flip)
    patch_contact_SE3 = cpin.SE3(R_patch @ R_flip, p_patch)
    fp_contact_SE3    = cpin.SE3(fp_R, fp_p)

    # Wrench in patch contact frame
    W_patch = patch_contact_SE3.inverse() * obj_SE3 * Wb_obj_pin
    fx_patch = W_patch.linear[0]
    fy_patch = W_patch.linear[1]
    fz_patch = W_patch.linear[2]
    mx_patch = W_patch.angular[0]
    my_patch = W_patch.angular[1]
    mz_patch = W_patch.angular[2]

    # Wrench in fp contact frame
    W_fp = fp_contact_SE3.inverse() * obj_SE3 * Wb_obj_pin
    fx_fp = W_fp.linear[0]
    fy_fp = W_fp.linear[1]
    fz_fp = W_fp.linear[2]
    mx_fp = W_fp.angular[0]
    my_fp = W_fp.angular[1]
    mz_fp = W_fp.angular[2]

    # Decide which box is active (smaller extents)
    sum_patch = casadi.sum1(patch_bnds)
    sum_fp    = casadi.sum1(fp_bnds)
    fp_is_smaller = sum_fp <= sum_patch

    # Select wrench components in the corresponding frame
    fx = casadi.if_else(fp_is_smaller, fx_fp, fx_patch)
    fy = casadi.if_else(fp_is_smaller, fy_fp, fy_patch)
    fz = casadi.if_else(fp_is_smaller, fz_fp, fz_patch)
    mx = casadi.if_else(fp_is_smaller, mx_fp, mx_patch)
    my = casadi.if_else(fp_is_smaller, my_fp, my_patch)
    mz = casadi.if_else(fp_is_smaller, mz_fp, mz_patch)

    # Select bounds corresponding to the chosen frame
    # Flipping the source y/z axes swaps its positive/negative y bounds.
    eff_bnds = casadi.if_else(fp_is_smaller, fp_bnds, patch_bnds[[0, 3, 2, 1]])
    bx_pos = eff_bnds[0]
    by_pos = eff_bnds[1]
    bx_neg = eff_bnds[2]
    by_neg = eff_bnds[3]

    # Sticking and sliding contacts share the friction and CoP bounds.
    friction = active * linearized_friction_cone()(casadi.vertcat(fx, fy, fz), mu)

    if use_simple_torsion:
        # Simplified yaw bounds in the selected contact frame.
        torsion = casadi.vertcat(
            active * (mu * fz - mz),
            active * (mu * fz + mz),
        )
    else:
        # Center and half-extents of the selected, possibly asymmetric rectangle.
        cx = 0.5 * (bx_pos - bx_neg)
        cy = 0.5 * (by_pos - by_neg)
        X = 0.5 * (bx_pos + bx_neg)
        Y = 0.5 * (by_pos + by_neg)

        # Shift moments to the rectangle center: m_center = m_origin - c x f.
        mx_c = mx - cy * fz
        my_c = my + cx * fz
        mz_c = mz - cx * fy + cy * fx

        # Expand the full rectangular yaw bounds into eight affine inequalities.
        yaw_ineqs = []
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                yaw_ineqs.append(
                    active * (
                        mu * (X + Y) * fz
                        - mz_c
                        - sx * (Y * fx + mu * mx_c)
                        - sy * (X * fy + mu * my_c)
                    )
                )
                yaw_ineqs.append(
                    active * (
                        mu * (X + Y) * fz
                        + mz_c
                        - sx * (Y * fx - mu * mx_c)
                        - sy * (X * fy - mu * my_c)
                    )
                )
        torsion = casadi.vertcat(*yaw_ineqs)

    ineqs = casadi.vertcat(
        friction[:4],
        torsion,

        active * (my + fz * bx_pos),       # my ≥ -fz*bx_pos
        active * (fz * bx_neg - my),       # my ≤  fz*bx_neg
        active * (mx + fz * by_neg),       # mx ≥ -fz*by_neg
        active * (fz * by_pos - mx),       # mx ≤  fz*by_pos

        friction[4],                     # fz ≥ 0
    )

    return casadi.Function(
        "obj_patch_env_force_constr",
        [q_obj, p_patch, R_patch_, patch_bnds,
         fp_p, fp_R_, fp_bnds,
         Wb_obj, mu, active],
        [ineqs],
        ["q_obj", "p_patch", "R_patch_", "patch_bnds",
         "fp_p", "fp_R_", "fp_bnds",
         "Wb_obj", "mu", "active"],
        ["ineqs"],
    )
