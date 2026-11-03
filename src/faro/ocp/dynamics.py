import casadi
import pinocchio.casadi as cpin

def kinematic_centroidal_momentum(model, data):
    """Return [linear, angular] momentum, world-aligned about the robot CoM."""
    q = casadi.SX.sym("q", model.nq)
    v = casadi.SX.sym("v", model.nv)

    kin_cent_mom = cpin.computeCentroidalMomentum(model, data, q, v)

    h = casadi.vertcat(
        kin_cent_mom.linear,
        kin_cent_mom.angular,
    )

    return casadi.Function("kinematic_centroidal_momentum",
                           [q, v], [h],
                           ["q", "v"], ["kin_cent_mom"])

def centroidal_dynamics(rob_model, rob_data, ee_names):
    """Sum world-aligned [force, moment] wrenches about their supplied points.

    Each moment is shifted from its supplied point to the CoM at q exactly
    once. The points must match the wrench origins, including across TO steps.
    """
    q = casadi.SX.sym("q", rob_model.nq)
    r = casadi.SX.sym("r", 3)
    mass = casadi.SX.sym("mass", 1)

    wrenches  = [casadi.SX.sym(f"f_{frame}", 6) for frame in ee_names]
    ee_points = [casadi.SX.sym(f"p_{frame}", 3) for frame in ee_names]

    com = cpin.centerOfMass(rob_model, rob_data, q)

    dl, dk = casadi.SX.zeros(3), casadi.SX.zeros(3)
    for w, p in zip(wrenches, ee_points):
        dl += w[0:3]
        dk += w[3:6] + casadi.cross(p - com, w[0:3])
    
    hdot = casadi.vertcat(dl, dk)
    hdot[:3] += mass * rob_model.gravity981

    return casadi.Function("centroidal_dynamics", 
            [q, r, mass] + wrenches + ee_points, 
            [hdot], 
            ["q", "r", "mass"] + ee_names + [frame + "_point" for frame in ee_names],
            ["hdot"])

def single_rigid_body_dynamics():
    # Body twist dot [lin vel dot, ang vel dot]
    #   -  order based on JointModelFreeFlyer convention
    Vb_dot = casadi.SX.sym("Body Twist_dot", 6)

    # Body twist [lin vel, ang vel]
    #   - based on JointModelFreeFlyer convention
    Vb = casadi.SX.sym("Vb", 6)

    mass = casadi.SX.sym("mass", 1)
    inertia = casadi.SX.sym("inertia", 3)

    # Spatial inertia matrix
    Ib = casadi.diag(inertia)
    mI = mass * casadi.SX_eye(3)

    Gb = casadi.blockcat([[Ib, casadi.SX.zeros(3, 3)], 
                          [casadi.SX.zeros(3, 3), mI]])

    # Twist-wrench dynamics have reverse order of twist
    #   - this is based on the Modern Robotics book
    Vb_dot_anglin = casadi.vertcat(Vb_dot[3:], Vb_dot[:3])
    Vb_anglin = casadi.vertcat(Vb[3:], Vb[:3])

    # Lie bracket of body twist (twist order: [ang vel, lin vel])
    skew_Vb_lin = cpin.skew(Vb[:3])
    skew_Vb_ang = cpin.skew(Vb[3:])
    ad_Vb_anglin = casadi.blockcat([[skew_Vb_ang, casadi.SX.zeros(3, 3)], 
                                    [skew_Vb_lin, skew_Vb_ang]])

    # Body wrench in dynamics convention [body moment, body force]
    W_body = Gb @ Vb_dot_anglin - ad_Vb_anglin.T @ Gb @ Vb_anglin

    return casadi.Function("single_rigid_body_dynamics", 
                           [Vb, Vb_dot, mass, inertia], 
                           [W_body], 
                           ["Vb", "vd", "mass", "inertia"], 
                           ["W_body"])

def ext_Wobj_body(rob_obj_forces, obj_obj_forces):
    """Sum incoming reactions and gravity in object-body [moment, force] order.

    Robot inputs are world-aligned [force, moment] at their EE positions;
    object inputs are body-local [moment, force] at their body origins.
    """
    
    ext_ee_names = list(rob_obj_forces.keys())
    ext_forces = [casadi.SX.sym(f"f_{frame}", 6) for frame in ext_ee_names]
    ext_Rs = [casadi.SX.sym(f"R_{frame}", 9) for frame in ext_ee_names]
    ext_ps = [casadi.SX.sym(f"p_{frame}", 3) for frame in ext_ee_names]

    ext_obj_names = list(obj_obj_forces.keys())
    ext_obj_forces = [casadi.SX.sym(f"f_{obj}", 6) for obj in ext_obj_names]
    ext_obj_qs = [casadi.SX.sym(f"q_{obj}", 7) for obj in ext_obj_names]

    q_obj = casadi.SX.sym("q", 7)
    obj_mass = casadi.SX.sym("obj_mass", 1)

    # Body wrench in dynamics convention [body moment, body force]
    Wbody_ext = casadi.SX.zeros(6)
    Tobj_base = cpin.XYZQUATToSE3(q_obj)
    TobjR_base = cpin.SE3(Tobj_base.rotation, casadi.SX.zeros(3))

    for f, p in zip(ext_forces, ext_ps):
        fee_lwa = cpin.Force(linear = -f[:3], angular = -f[3:])
        # Already world-aligned: only translate the moment to the world origin.
        # Rotation inputs remain in the function signature for existing callers.
        fee_W = cpin.SE3(casadi.SX.eye(3), p) * fee_lwa
        W_i_obj = Tobj_base.inverse() * fee_W

        Wbody_ext[:3] += W_i_obj.angular
        Wbody_ext[3:] += W_i_obj.linear

    for f, q in zip(ext_obj_forces, ext_obj_qs):
        T_ext_obj = cpin.XYZQUATToSE3(q)

        Wb_obj = cpin.Force(linear = -f[3:], angular = -f[:3])
        Wb_obj_base = Tobj_base.inverse() * T_ext_obj * Wb_obj

        Wbody_ext[:3] += Wb_obj_base.angular
        Wbody_ext[3:] += Wb_obj_base.linear

    mg = obj_mass * 9.81 * casadi.vertcat(0, 0, 1)
    W_grav = cpin.Force(linear = -mg, angular = casadi.SX.zeros(3))
    W_grav_obj = TobjR_base.inverse() * W_grav

    Wbody_ext[:3] += W_grav_obj.angular
    Wbody_ext[3:] += W_grav_obj.linear

    return casadi.Function("ext_Wobj_body", 
                           [q_obj, obj_mass] + \
                            ext_forces + ext_Rs + ext_ps + \
                            ext_obj_forces + ext_obj_qs,
                           [Wbody_ext],
                           ["q_obj", "obj_mass"] + ext_ee_names + \
                           [s + "R" for s in ext_ee_names] + \
                           [s + "p" for s in ext_ee_names] + \
                           ext_obj_names + \
                           [s + "q" for s in ext_obj_names],
                           ["Wbody_ext"])

def get_joint_torques(model, data, frame_names):
    """RNEA with world-aligned [force, moment] inputs at the EE origins at q."""
    q = casadi.SX.sym("q", model.nq)
    v = casadi.SX.sym("v", model.nv)
    vd = casadi.SX.sym("joint_acc", model.nv)
    
    wrenches = [casadi.SX.sym(f"w_{frame}", 6) for frame in frame_names]

    cpin.framesForwardKinematics(model, data, q)
    
    f_ext = [cpin.Force(casadi.SX.zeros(6)) for _ in range(model.njoints)]

    if frame_names is not None:
        for frame_name, w_lwa in zip(frame_names, wrenches):
            frame_id = model.getFrameId(frame_name)
            joint_id = model.frames[frame_id].parentJoint
            
            T_joint_ee = model.frames[frame_id].placement

            R_ee_w = data.oMf[model.getFrameId(frame_name)].rotation.T
            T_R_ee_w = cpin.SE3(R_ee_w, casadi.SX.zeros(3))

            W_locWA = cpin.Force(linear=w_lwa[:3], angular=w_lwa[3:])
            W_loc = T_R_ee_w * W_locWA

            f_ext[joint_id] += T_joint_ee * W_loc

    tau_full = cpin.rnea(model, data, q, v, vd, f_ext)

    tau_base = tau_full[:6]
    tau_joints = tau_full[6:]

    return casadi.Function("get_torques", 
                       [q, v, vd] + wrenches, 
                       [tau_base, tau_joints], 
                       ["q", "v", "vd"] + frame_names, 
                       ["tau_base", "tau_joints"])
