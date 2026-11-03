import math

import mujoco
import numpy as np


class SceneGeomManager:
    """Append debug geometry directly to viewer.user_scn."""

    def __init__(self, viewer):
        self.viewer = viewer
        self.clear()

    def clear(self):
        self.viewer.user_scn.ngeom = 0

    def _add_geom(self, geom_type, size, pos, R, rgba):
        scene = self.viewer.user_scn
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            type=geom_type,
            size=np.asarray(size, dtype=float).reshape(3),
            pos=np.asarray(pos, dtype=float),
            mat=np.asarray(R, dtype=float).reshape(9),
            rgba=rgba,
        )
        scene.ngeom += 1

    def add_arrow(self, origin, vec,
                  length_scale=0.1,
                  radius=0.015,
                  rgba=(0.2, 0.8, 0.2, 0.8),
                  flip=False):
        origin = np.asarray(origin, dtype=float)
        vec = np.asarray(vec, dtype=float)
        magnitude = np.linalg.norm(vec)
        length = magnitude * float(length_scale)
        if length < 1e-9:
            return

        direction = vec / (magnitude + 1e-12)
        if flip:
            origin = origin + direction * (length / 2.0)
            direction = -direction

        quat, rotation = np.empty(4), np.empty(9)
        mujoco.mju_quatZ2Vec(quat, direction)
        mujoco.mju_quat2Mat(rotation, quat)
        self._add_geom(mujoco.mjtGeom.mjGEOM_ARROW,
                       [radius, radius, length], origin, rotation, rgba)

    def add_box(self, center, half_lengths, R=np.eye(3),
                rgba=(0.5, 0.5, 0.5, 1.0)):
        self._add_geom(mujoco.mjtGeom.mjGEOM_BOX, half_lengths, center, R, rgba)

    def add_capsule(self, center, half_length, radius, R=np.eye(3), rgba=(0.5, 0.5, 0.5, 1.0)):
        """Draw along local Z; half_length excludes the spherical end caps."""
        self._add_geom(mujoco.mjtGeom.mjGEOM_CAPSULE,
                       [radius, half_length, 0], center, R, rgba)

    def add_rect_patch_box(self,
                           frame_pos,
                           frame_R,
                           bounds,
                           thickness=1e-3,
                           rgba=(0.3, 0.3, 0.9, 0.3)):
        """Draw a patch with local [x+, y+, x-, y-] bounds in a world frame.

        frame_pos is (3,), frame_R is (3,3). Return the box's world center.
        """
        frame_pos = np.asarray(frame_pos, dtype=float)
        frame_R = np.asarray(frame_R, dtype=float).reshape(3, 3)
        x_plus, y_plus, x_minus, y_minus = [float(b) for b in bounds]
        half_lengths = [(x_plus + x_minus) / 2, (y_plus + y_minus) / 2, thickness]
        offset_local = [(x_plus - x_minus) / 2, (y_plus - y_minus) / 2, 0]
        center_world = frame_pos + frame_R @ offset_local
        self.add_box(center_world, half_lengths, frame_R, rgba=rgba)
        return center_world


def plot_joint_dynamics(model, q_traj, v_traj, tau_traj, com_traj, dt,
                        v_lims, tau_lims, *, endpoint_controls=False):
    """
    Assumptions:
      - q_traj: (T, nq) with floating base first; actuated q is q[:, 7:].
      - v_traj: (T, nv) with floating base first; actuated v is v[:, 6:].
      - tau_traj: (T-1, nj) control trajectory (one less timestep).
      - dt: scalar timestep or (T-1,) interval durations; dt[k] spans
        state knots k and k+1, including any time scaling.
      - endpoint_controls: pair tau[k] with state/time k+1 instead of k.
      - com_traj: optional (T, 3) CoM trajectory to compare against Pinocchio.
    """
    import matplotlib.pyplot as plt
    import pinocchio as pin

    q_traj = np.asarray(q_traj)
    v_traj = np.asarray(v_traj)
    tau = np.asarray(tau_traj)

    # Actuated parts for joint plots
    q = q_traj[:, 7:]
    v = v_traj[:, 6:]

    T, nj = q.shape
    step_ts = np.asarray(dt, dtype=float)
    if step_ts.ndim > 1 or (step_ts.ndim == 1 and step_ts.shape != (T - 1,)):
        raise ValueError(f"dt must be a scalar or have shape ({T - 1},), got {step_ts.shape}")
    if not np.all(np.isfinite(step_ts)) or np.any(step_ts < 0):
        raise ValueError("dt must contain finite nonnegative interval durations")
    if step_ts.ndim == 0:
        step_ts = np.full(T - 1, float(step_ts))
    t_x = np.r_[0., np.cumsum(step_ts)]  # state knots for q, v, com
    t_u = t_x[1:] if endpoint_controls else t_x[:-1]

    joint_names = list(getattr(model, "names", []))[-nj:] or [f"Joint {i}" for i in range(nj)]

    q_lims = np.stack([model.lowerPositionLimit[-nj:],
                        model.upperPositionLimit[-nj:]], axis=1)

    n_cols = math.ceil(math.sqrt(nj))
    n_rows = math.ceil(nj / n_cols)

    def joint_axes(title, ylab, row_height=2.5):
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * row_height),
            constrained_layout=True,
        )
        fig.suptitle(title, fontsize=16)
        axes = np.atleast_1d(axes).ravel()
        for i, ax in enumerate(axes[:nj]):
            ax.set_title(joint_names[i], fontsize=10)
            if i % n_cols == 0:
                ax.set_ylabel(ylab)
            ax.grid(True, ls=":", alpha=0.6)
        for ax in axes[nj:]:
            ax.axis("off")
        return axes[:nj]

    def plot_grid(title, ylab, time, data, lims, lim_type):
        for i, ax in enumerate(joint_axes(title, ylab)):
            ax.plot(time, data[:, i], lw=1.5)
            if lim_type == "bounds":
                ax.axhline(lims[i, 0], ls="--", alpha=0.5)
                ax.axhline(lims[i, 1], ls="--", alpha=0.5)
            else:
                limit = lims[i]
                if np.isfinite(limit) and limit < 1e10:
                    ax.axhline(+limit, ls="--", alpha=0.5)
                    ax.axhline(-limit, ls="--", alpha=0.5)
            if i >= nj - n_cols:
                ax.set_xlabel("Time [s]")

    plot_grid("Joint Positions", "Rad",     t_x, q,   q_lims,   "bounds")
    plot_grid("Joint Velocities", "Rad/s",  t_x, v,   v_lims,   "symmetric")
    plot_grid("Joint Torques / Inputs", "Nm", t_u, tau, tau_lims, "symmetric")

    # Match torque samples (T-1) with velocities on the control grid.
    v_u = v[1:, :] if endpoint_controls else v[:-1, :]
    for i, ax in enumerate(joint_axes("Torque-Speed Curves", "Torque [Nm]", 3.0)):
        vmax = float(v_lims[i])
        taumax = float(tau_lims[i])

        # Trajectory in (velocity, torque) space
        ax.plot(v_u[:, i], tau[:, i], lw=1.5)
        ax.scatter(v_u[:, i], tau[:, i], s=8, alpha=0.7)

        # Plot linear torque-speed envelope if limits are finite
        if np.isfinite(vmax) and np.isfinite(taumax) and vmax > 0 and taumax < 1e10:
            vv = np.linspace(-vmax, vmax, 200)
            tau_env = taumax * (1.0 - np.abs(vv) / vmax)

            ax.plot(vv,  tau_env, ls="--", alpha=0.8)
            ax.plot(vv, -tau_env, ls="--", alpha=0.8)

            ax.axvline(+vmax, ls=":", alpha=0.5)
            ax.axvline(-vmax, ls=":", alpha=0.5)

        ax.set_xlabel("Velocity [rad/s]")

    com_arg = None if com_traj is None else np.asarray(com_traj)
    if com_arg is not None and com_arg.shape != (T, 3):
        raise ValueError(f"com_traj must have shape (T,3) = ({T},3), got {com_arg.shape}")

    data_pin = model.createData()
    com_pin = np.zeros((T, 3))
    for k in range(T):
        # pin.centerOfMass returns a 3D vector (and also updates data_pin.com[0])
        com_pin[k, :] = np.asarray(pin.centerOfMass(model, data_pin, q_traj[k])).reshape(3)

    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True, constrained_layout=True)
    fig.suptitle("Center of Mass: Solver Centroidal Com vs Pinocchio", fontsize=16)

    labels = ["CoM x [m]", "CoM y [m]", "CoM z [m]"]
    for i in range(3):
        ax = axes[i]
        ax.plot(t_x, com_pin[:, i], lw=2.0, label="Pinocchio")
        if com_arg is not None:
            ax.plot(t_x, com_arg[:, i], lw=1.5, ls="--", label="Solver")
        ax.set_ylabel(labels[i])
        ax.grid(True, ls=":", alpha=0.6)
        if i == 0:
            ax.legend(loc="best")

    axes[-1].set_xlabel("Time [s]")
    plt.show()


def plot_timescale_traj(traj, start_idx=0, end_idx=None):
    """Plot scale[k] over the interval from state knot k to state knot k+1."""
    import matplotlib.pyplot as plt

    y = np.asarray(traj).reshape(-1)
    T = y.shape[0]

    if end_idx is None:
        end_idx = T
    start_idx = int(start_idx)
    end_idx = int(end_idx)

    if not (0 <= start_idx < end_idx <= T):
        raise ValueError(f"Invalid slice [{start_idx}:{end_idx}] for traj length {T}")

    y = y[start_idx:end_idx]
    x = np.arange(start_idx, end_idx)

    x_step = np.r_[x, end_idx]
    y_step = np.r_[y, y[-1]]

    plt.figure(figsize=(9, 3.5), constrained_layout=True)
    plt.step(x_step, y_step, where="post", label="Time scaling")
    plt.plot(x, y, linestyle="None", marker="o", markersize=3.0)
    plt.axhline(1.0, ls="--", alpha=0.7, label="Nominal (1.0)")

    plt.title("Discretization Scaling Factor Trajectory")
    plt.xlabel("State knot index (scale applies to following interval)")
    plt.ylabel("Time scaling factor")
    plt.grid(True, ls=":", alpha=0.6)
    plt.legend(loc="best")
    plt.show()
