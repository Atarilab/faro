import casadi
import pinocchio as pin
import pinocchio.casadi as cpin


class RobotKinematics:
    """Build requested frames together; evaluate once per symbolic configuration."""

    def __init__(self, model, frames, velocity_frames=None):
        self.frame_names = tuple(frames)
        self.velocity_frame_names = tuple(velocity_frames or {})
        data = model.createData()
        q = casadi.SX.sym("q", model.nq)
        cpin.framesForwardKinematics(model, data, q)
        poses = [data.oMf[model.getFrameId(frame)] for frame in frames.values()]
        rotations = casadi.horzcat(*[casadi.vec(pose.rotation) for pose in poses])
        positions = casadi.horzcat(*[pose.translation for pose in poses])
        self._poses = casadi.Function("robot_frame_poses", [q], [rotations, positions])

        if velocity_frames:
            v = casadi.SX.sym("v", model.nv)
            cpin.forwardKinematics(model, data, q, v)
            rotations, positions, velocities = [], [], []
            for frame in velocity_frames.values():
                frame_id = model.getFrameId(frame)
                cpin.updateFramePlacement(model, data, frame_id)
                pose = data.oMf[frame_id]
                rotations.append(casadi.vec(pose.rotation))
                positions.append(pose.translation)
                velocities.append(cpin.getFrameVelocity(model, data, frame_id, pin.WORLD).vector)
            self._states = casadi.Function(
                "robot_frame_states", [q, v],
                [casadi.horzcat(*rotations), casadi.horzcat(*positions),
                 casadi.horzcat(*velocities)],
            )

    def poses(self, q):
        rotations, positions = self._poses(q)
        return {
            name: (rotations[:, i], positions[:, i])
            for i, name in enumerate(self.frame_names)
        }

    def states(self, q, v):
        """EE poses and WORLD twists [linear at world origin, angular]."""
        if not self.velocity_frame_names:
            return {}
        rotations, positions, velocities = self._states(q, v)
        return {
            name: (rotations[:, i], positions[:, i], velocities[:, i])
            for i, name in enumerate(self.velocity_frame_names)
        }


def rob_com(model, data):
    q = casadi.SX.sym("q", model.nq)
    cpin.framesForwardKinematics(model, data, q)
    com = cpin.centerOfMass(model, data, q)
    return casadi.Function(
            "rob_com",
            [q], [com],
            ["q"], ["com"]
            )

def obj_state_to_face_q():
    q = casadi.SX.sym("q", 7)

    p_obj_face = casadi.SX.sym("p_obj_face", 3)
    R_obj_face = casadi.SX.sym("R_obj_face", 9)

    Tobj = cpin.XYZQUATToSE3(q)
    T_obj_face = cpin.SE3(casadi.reshape(R_obj_face, 3, 3),
                          p_obj_face)

    T_face = Tobj * T_obj_face
    R = T_face.rotation
    p = T_face.translation

    return casadi.Function(
            "obj_state_to_face_qv",
            [q, p_obj_face, R_obj_face], [casadi.vec(R), p],
            ["q", "p_obj_face", "R_obj_face"], ["R", "p"]
            )

def obj_state_to_face_qv():
    """Face pose and WORLD twist from an object body-local [linear, angular] twist."""
    q = casadi.SX.sym("q", 7)
    v = casadi.SX.sym("v", 6)

    p_obj_face = casadi.SX.sym("p_obj_face", 3)
    R_obj_face = casadi.SX.sym("R_obj_face", 9)

    Tobj = cpin.XYZQUATToSE3(q)
    T_obj_face = cpin.SE3(casadi.reshape(R_obj_face, 3, 3),
                          p_obj_face)

    T_face = Tobj * T_obj_face
    R = T_face.rotation
    p = T_face.translation

    # Every rigidly attached face has the same WORLD spatial twist. Its
    # physical point velocity is V[:3] + cross(V[3:], face_world_position).
    v_face_World = Tobj.action @ v

    return casadi.Function(
            "obj_state_to_face_qv",
            [q, v, p_obj_face, R_obj_face], [casadi.vec(R), p, v_face_World],
            ["q", "v", "p_obj_face", "R_obj_face"], ["R", "p", "v_face_World"]
            )

def q_integrate():
    q = casadi.SX.sym("q", 7)
    dq = casadi.SX.sym("dq", 6)

    q_next = cpin.liegroups.SE3().integrate(q, dq)

    return casadi.Function("qv_state_integrate",
                           [q, dq], [q_next],
                           ["q", "dq"], ["q_next"])    

def qrob_integrate(model):
    q = casadi.SX.sym("q", model.nq)
    dq = casadi.SX.sym("dq", model.nv)

    q_next = casadi.vertcat(
        cpin.liegroups.SE3().integrate(q[:7], dq[:6]),
        dq[6:]
    )

    return casadi.Function("qrob_integrate",
                           [q, dq], [q_next],
                           ["q", "dq"], ["q_next"])
