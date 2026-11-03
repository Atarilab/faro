import time

import mujoco
import mujoco.viewer
import numpy as np

from kinevis.vis_utils import SceneGeomManager


def _center_of_pressure(wrench, bounds):
    """Return the local CoP, clamped to [x+, y+, x-, y-] patch bounds."""
    fz, mx, my = wrench[2:5]
    if abs(fz) < 1e-6:
        return np.zeros(3)
    x_plus, y_plus, x_minus, y_minus = bounds
    return np.array([
        np.clip(-my / fz, -x_minus, x_plus),
        np.clip(mx / fz, -y_minus, y_plus),
        0.0,
    ])


class KinematicRobotVisualizer:
    """Play trajectories; show_forces sets initial arrow visibility (G toggles)."""

    def __init__(self,
                 model: mujoco.MjModel,
                 transparent_robot: bool = False,
                 robot_alpha: float = 0.2,
                 show_forces: bool = True):
        self.model = model
        self.data = mujoco.MjData(model)
        self.viewer = None
        self.geom: SceneGeomManager | None = None
        self.transparent_robot = transparent_robot
        self.robot_alpha = float(robot_alpha)
        self._patch_wrenches = []
        self._site_capsules = []
        self._forces_on = bool(show_forces)

    def add_patch_wrench(self,
                         R_traj,
                         p_traj,
                         wrench_local_traj,
                         bounds_local_traj,
                         rgba=(0.9, 0.3, 0.3, 0.4),
                         length_scale=0.05,
                         flip=True):
        """Register a contact trajectory with a common length T.

        R_traj (T,3,3) and p_traj (T,3) locate the patch frame in world.
        wrench_local_traj (T,6) contains local [fx, fy, fz, mx, my, mz].
        bounds_local_traj (T,4) contains local [x+, y+, x-, y-] extents.
        Active contacts draw a patch box and a force arrow at the clamped CoP.
        """
        W = np.asarray(wrench_local_traj, dtype=float)
        R = np.asarray(R_traj, dtype=float)
        p = np.asarray(p_traj, dtype=float)
        B = np.asarray(bounds_local_traj, dtype=float)

        assert W.ndim == 2 and W.shape[1] == 6
        assert R.ndim == 3 and R.shape[1:] == (3, 3)
        assert p.ndim == 2 and p.shape[1] == 3
        assert B.ndim == 2 and B.shape[1] == 4
        assert W.shape[0] == R.shape[0] == p.shape[0] == B.shape[0]

        self._patch_wrenches.append(dict(
            R=R,
            p=p,
            wrenches_local=W,
            bounds_traj=B,
            rgba=rgba,
            length_scale=float(length_scale),
            flip=bool(flip),
        ))

    def add_site_capsule(self, site_name, length, radius, rgba=(0.9, 0.6, 0.2, 0.5)):
        """Attach a capsule along a site's local Z axis.

        length is the distance between the two sphere centers.
        """
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id == -1:
            raise ValueError(f"Site '{site_name}' not found in model.")
        self._site_capsules.append(dict(
            site_id=site_id,
            length=float(length),
            radius=float(radius),
            rgba=rgba,
        ))

    def _apply_robot_transparency(self):
        """Scale positive alpha values on all model geoms when enabled."""
        if self.transparent_robot:
            alpha = self.model.geom_rgba[:, 3]
            alpha[alpha > 0] *= self.robot_alpha

    def _initialize_viewer(self, key_callback=None):
        self._apply_robot_transparency()
        self.viewer = mujoco.viewer.launch_passive(
            self.model, self.data, show_left_ui=False, show_right_ui=False,
            key_callback=key_callback,
        )
        scn = self.viewer.user_scn
        scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        scn.flags[mujoco.mjtRndFlag.mjRND_FOG] = 0
        scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = 0
        self.viewer.opt.sitegroup[0] = 0
        self.geom = SceneGeomManager(self.viewer)
        self.viewer.sync()

    def _draw_patch_wrenches(self, k):
        for cfg in self._patch_wrenches:
            if k >= len(cfg["R"]):
                continue
            R, p = cfg["R"][k], cfg["p"][k]
            wrench, bounds = cfg["wrenches_local"][k], cfg["bounds_traj"][k]
            force = wrench[:3]
            if np.linalg.norm(force) < 1e-6:
                continue

            self.geom.add_rect_patch_box(p, R, bounds, rgba=cfg["rgba"])
            if self._forces_on:
                self.geom.add_arrow(
                    origin=p + R @ _center_of_pressure(wrench, bounds),
                    vec=R @ force,
                    length_scale=cfg["length_scale"],
                    rgba=cfg["rgba"],
                    flip=cfg["flip"],
                )

    def _draw_site_capsules(self):
        for cfg in self._site_capsules:
            site_id = cfg["site_id"]
            self.geom.add_capsule(
                center=self.data.site_xpos[site_id],
                half_length=cfg["length"] / 2.0,
                radius=cfg["radius"],
                R=self.data.site_xmat[site_id].reshape(3, 3),
                rgba=cfg["rgba"],
            )

    def _render_frame(self, k=None):
        mujoco.mj_fwdPosition(self.model, self.data)
        self.geom.clear()
        if k is not None:
            self._draw_patch_wrenches(k)
        self._draw_site_capsules()
        self.viewer.sync()

    def visualize_trajectory(self, qpos_trajectory, dt_trajectory):
        """Loop over qpos (T,nq) using finite, nonnegative durations.

        With dt (T-1,), dt[k] times the interval from qpos[k] to qpos[k+1].
        The terminal state is rendered, then playback wraps without an extra
        interval. Legacy dt (T,) supplies one duration per frame, including
        the terminal frame before wrapping. Rendering time counts toward each
        duration; intervals shorter than rendering time advance immediately.
        MuJoCo's data.time follows the cumulative timestamp of the current knot.

        Space pauses/resumes, arrows step while paused, and G toggles forces.
        Contact trajectories may be shorter; their overlays stop at the end.
        """
        qpos_trajectory = np.asarray(qpos_trajectory, dtype=float)
        dt_trajectory = np.asarray(dt_trajectory, dtype=float)
        if (qpos_trajectory.ndim != 2 or qpos_trajectory.shape[0] == 0
                or qpos_trajectory.shape[1] != self.model.nq):
            raise ValueError(f"qpos_trajectory must have nonempty shape (T, {self.model.nq}).")
        T = len(qpos_trajectory)
        if dt_trajectory.ndim != 1 or len(dt_trajectory) not in (T - 1, T):
            raise ValueError("dt_trajectory must contain T-1 interval durations or T frame durations.")
        if not np.all(np.isfinite(dt_trajectory)) or np.any(dt_trajectory < 0):
            raise ValueError("dt_trajectory durations must be finite and nonnegative.")
        knot_times = np.concatenate(([0.], np.cumsum(dt_trajectory[:T - 1])))

        paused = False
        k = 0

        def key_cb(keycode):
            nonlocal paused, k
            if keycode == 32:  # space
                paused = not paused
            elif keycode == 71:  # G
                self._forces_on = not self._forces_on
            elif paused:
                if keycode == 262:  # right
                    k = min(k + 1, T - 1)
                elif keycode == 263:  # left
                    k = max(k - 1, 0)

        self._initialize_viewer(key_callback=key_cb)
        while self.viewer.is_running():
            frame_start = time.perf_counter()
            self.data.qpos[:] = qpos_trajectory[k]
            self.data.time = float(knot_times[k])
            self._render_frame(k=k)
            if not paused:
                if k < len(dt_trajectory):
                    remaining = float(dt_trajectory[k]) - (time.perf_counter() - frame_start)
                    if remaining > 0:
                        time.sleep(remaining)
                k = (k + 1) % T
            else:
                time.sleep(0.01)
