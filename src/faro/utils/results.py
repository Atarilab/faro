import xml.etree.ElementTree as ET
import json

from faro.configs.identifiers import robot_name, interface_name
from pathlib import Path

from faro.configs.objects import Plane

import numpy as np
import pinocchio as pin


def upsample_results_hermite(
    model,
    results,
    time,
    frequency,
    append_terminal=False,
):
    """
    Resample result state trajectories with Hermite interpolation.

    The returned results dict is a shallow copy of ``results`` with
    ``robot_xs`` and ``obj_xs`` replaced by trajectories sampled on one
    uniform grid at ``frequency`` Hz. For each desired sample time, the
    neighboring original knots are found and that interval is interpolated.

    The final original knot is included only if it lies on the uniform grid.
    Set ``append_terminal=True`` to append it even when that makes the last
    time step non-uniform.

    Returns:
        upsampled_results, upsampled_time
    """
    time = _as_time_vector(time)
    frequency = float(frequency)
    if frequency <= 0.0:
        raise ValueError("frequency must be positive.")

    upsampled_time = _make_uniform_upsampled_time(
        time,
        1.0 / frequency,
        append_terminal=append_terminal,
    )

    upsampled_results = dict(results)
    if set(model) != set(results["robot_xs"]):
        raise ValueError("Robot model IDs must match the result trajectories.")
    upsampled_results["robot_xs"] = {
        rid: _upsample_robot_xs(pm, results["robot_xs"][rid], time, upsampled_time)
        for rid, pm in model.items()
    }
    upsampled_results["obj_xs"] = {
        name: _upsample_obj_xs(obj_xs, time, upsampled_time)
        for name, obj_xs in results["obj_xs"].items()
    }

    return upsampled_results, upsampled_time


def plot_interpolated_trajectory(
    model,
    results,
    time,
    frequency,
    append_terminal=False,
    object_names=None,
    show=True,
):
    """
    Interpolate results and plot the resulting robot/object trajectories.

    Returns:
        interpolated_results, interpolated_time, fig, axes
    """
    interpolated_results, interpolated_time = upsample_results_hermite(
        model,
        results,
        time,
        frequency,
        append_terminal=append_terminal,
    )
    fig, axes = plot_results_trajectory(
        model,
        interpolated_results,
        interpolated_time,
        object_names=object_names,
        show=show,
    )
    return interpolated_results, interpolated_time, fig, axes


def plot_results_trajectory(
    model,
    results,
    time,
    object_names=None,
    show=True,
):
    """
    Plot robot and object state trajectories already sampled on ``time``.

    Returns:
        fig, axes
    """
    import matplotlib.pyplot as plt

    time = _as_time_vector(time, min_samples=1)
    plots = []
    for robot_id, pm in model.items():
        robot_xs = np.asarray(results["robot_xs"][robot_id], dtype=float)
        if robot_xs.shape[0] != len(time):
            raise ValueError("Robot trajectories and time must have the same length.")
        if robot_xs.shape[1] < pm.nq + pm.nv:
            raise ValueError("robot_xs must contain at least q and v columns.")
        q = robot_xs[:, :pm.nq]
        v = robot_xs[:, pm.nq:pm.nq + pm.nv]
        plots.extend([
            (f"{robot_id} base position", q[:, :3], ["x", "y", "z"]),
            (f"{robot_id} base velocity", v[:, :6], ["vx", "vy", "vz", "wx", "wy", "wz"]),
        ])
        n_joints = max(0, pm.nq - 7)
        if n_joints:
            names = _joint_names(pm, n_joints)
            plots.extend([
                (f"{robot_id} joint position", q[:, 7:pm.nq], names),
                (f"{robot_id} joint velocity", v[:, 6:pm.nv], names),
            ])

    obj_xs = results.get("obj_xs", {})
    if object_names is None:
        object_names = list(obj_xs.keys())

    for name in object_names:
        obj_x = np.asarray(obj_xs[name], dtype=float)
        if obj_x.shape[0] != len(time):
            raise ValueError(f"obj_xs[{name!r}] and time must have the same length.")
        if obj_x.shape[1] < 13:
            raise ValueError(f"obj_xs[{name!r}] must contain at least q and v columns.")

        plots.extend([
            (f"{name} position", obj_x[:, :3], ["x", "y", "z"]),
            (
                f"{name} velocity",
                obj_x[:, 7:13],
                ["vx", "vy", "vz", "wx", "wy", "wz"],
            ),
        ])

    fig, axes = plt.subplots(
        len(plots),
        1,
        figsize=(10, max(3.0, 2.2 * len(plots))),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    for ax, (title, values, labels) in zip(axes, plots):
        _plot_trajectory_block(ax, time, values, labels)
        ax.set_title(title)

    axes[-1].set_xlabel("Time [s]")
    if show:
        plt.show()

    return fig, axes


def _upsample_robot_xs(
    model,
    robot_xs,
    time,
    upsampled_time,
):
    robot_xs = np.asarray(robot_xs, dtype=float)
    if robot_xs.shape[0] != len(time):
        raise ValueError("robot_xs and time must have the same length.")
    if robot_xs.shape[1] < model.nq + model.nv:
        raise ValueError("robot_xs must contain at least q and v columns.")

    q = robot_xs[:, :model.nq]
    v = robot_xs[:, model.nq:model.nq + model.nv]

    q_up = np.empty((len(upsampled_time), model.nq))
    v_up = np.empty((len(upsampled_time), model.nv))
    interval_ids = _interval_ids(time, upsampled_time)

    for interval_id in np.unique(interval_ids):
        out_ids = np.flatnonzero(interval_ids == interval_id)
        i = int(interval_id)

        base_interp = make_se3_body_hermite(
            pin.XYZQUATToSE3(q[i, :7]),
            v[i, :6],
            pin.XYZQUATToSE3(q[i + 1, :7]),
            v[i + 1, :6],
            time[i],
            time[i + 1],
        )
        joint_interp = make_euclidean_hermite(
            q[i, 7:model.nq],
            v[i, 6:model.nv],
            q[i + 1, 7:model.nq],
            v[i + 1, 6:model.nv],
            time[i],
            time[i + 1],
        )

        for out_id in out_ids:
            tau = upsampled_time[out_id]
            T_tau, V_tau = base_interp(tau)
            joint_q, joint_v = joint_interp(tau)
            q_up[out_id, :7] = pin.SE3ToXYZQUAT(T_tau)
            v_up[out_id, :6] = V_tau
            q_up[out_id, 7:model.nq] = joint_q
            v_up[out_id, 6:model.nv] = joint_v

    trailing = robot_xs[:, model.nq + model.nv:]
    robot_xs_up = np.hstack((q_up, v_up))
    if trailing.shape[1] == 0:
        return robot_xs_up

    trailing_up = _linear_interp_columns(trailing, time, upsampled_time)
    return np.hstack((robot_xs_up, trailing_up))


def _upsample_obj_xs(
    obj_xs,
    time,
    upsampled_time,
):
    obj_xs = np.asarray(obj_xs, dtype=float)
    if obj_xs.shape[0] != len(time):
        raise ValueError("obj_xs and time must have the same length.")
    if obj_xs.shape[1] < 13:
        raise ValueError("obj_xs must contain at least q and v columns.")

    q = obj_xs[:, :7]
    v = obj_xs[:, 7:13]

    q_up = np.empty((len(upsampled_time), 7))
    v_up = np.empty((len(upsampled_time), 6))
    interval_ids = _interval_ids(time, upsampled_time)

    for interval_id in np.unique(interval_ids):
        out_ids = np.flatnonzero(interval_ids == interval_id)
        i = int(interval_id)
        interp = make_se3_body_hermite(
            pin.XYZQUATToSE3(q[i, :7]),
            v[i, :6],
            pin.XYZQUATToSE3(q[i + 1, :7]),
            v[i + 1, :6],
            time[i],
            time[i + 1],
        )

        for out_id in out_ids:
            T_tau, V_tau = interp(upsampled_time[out_id])
            q_up[out_id] = pin.SE3ToXYZQUAT(T_tau)
            v_up[out_id] = V_tau

    trailing = obj_xs[:, 13:]
    obj_xs_up = np.hstack((q_up, v_up))
    if trailing.shape[1] == 0:
        return obj_xs_up

    trailing_up = _linear_interp_columns(trailing, time, upsampled_time)
    return np.hstack((obj_xs_up, trailing_up))


def _interval_ids(time, query_time):
    ids = np.searchsorted(time, query_time, side="right") - 1
    return np.clip(ids, 0, len(time) - 2)


def _make_uniform_upsampled_time(time, dt, append_terminal=False):
    tol = 1e-12 * max(1.0, time[-1] - time[0])
    duration = time[-1] - time[0]
    n_steps = int(np.floor((duration + tol) / dt))
    sampled_time = time[0] + dt * np.arange(n_steps + 1)

    if abs(sampled_time[-1] - time[-1]) <= tol:
        sampled_time[-1] = time[-1]

    if append_terminal and time[-1] - sampled_time[-1] > tol:
        sampled_time = np.concatenate((sampled_time, [time[-1]]))

    return sampled_time


def _as_time_vector(time, min_samples=2):
    time = np.asarray(time, dtype=float)
    if time.ndim != 1:
        if time.ndim == 2 and 1 in time.shape:
            time = time.reshape(-1)
        else:
            raise ValueError("time must be a 1D array.")
    if len(time) < min_samples:
        raise ValueError(f"time must contain at least {min_samples} samples.")
    if not np.all(np.diff(time) > 0.0):
        raise ValueError("time must be strictly increasing.")
    return time


def _linear_interp_columns(values, time, query_time):
    return np.column_stack([
        np.interp(query_time, time, values[:, j])
        for j in range(values.shape[1])
    ])


def _joint_names(model, n_joints):
    if n_joints <= 0:
        return []
    names = list(getattr(model, "names", []))
    if len(names) >= n_joints:
        return names[-n_joints:]
    return [f"joint_{i}" for i in range(n_joints)]


def _plot_trajectory_block(ax, time, values, labels):
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    labels = labels or [f"{i}" for i in range(values.shape[1])]

    for i in range(values.shape[1]):
        label = labels[i] if i < len(labels) else f"{i}"
        ax.plot(time, values[:, i], lw=1.2, label=label)

    ax.grid(True, ls=":", alpha=0.6)
    if values.shape[1] <= 12:
        ax.legend(loc="best", fontsize=8, ncols=min(values.shape[1], 4))


def pin_to_mj(
    scene,
    results_,
    time_,
    cms,
    num_modes,
    nodes_per_mode,
    frequency=50,
):
    models = {rid: cfg.pin_model for rid, cfg in scene.robots.items()}
    original_time = _as_time_vector(time_)
    results, time = upsample_results_hermite(models, results_, original_time, frequency)
    traj = {
        "schema_version": np.array(3 if scene.include_robot_ees else 2, dtype=np.int64),
        "robot_ids": np.array(list(scene.robots), dtype=str),
        "object_names": np.array(list(scene.env_objects), dtype=str),
        "time": time,
    }
    for rid, model in models.items():
        q = results["robot_xs"][rid][:, :model.nq].copy()
        v = results["robot_xs"][rid][:, model.nq:model.nq + model.nv].copy()
        for i in range(len(q)):
            v[i, :3] = pin.XYZQUATToSE3(q[i, :7]).rotation @ v[i, :3]
        fields = {
            "dof_pos": q[:, 7:], "dof_vel": v[:, 6:],
            "root_pos": q[:, :3], "root_rot": q[:, [6, 3, 4, 5]],
            "root_lin_vel": v[:, :3], "root_ang_vel": v[:, 3:6],
        }
        traj.update({robot_name(rid, field): value for field, value in fields.items()})

    for obj_name in scene.env_objects:
        if obj_name not in results["obj_xs"]:
            raise KeyError(f"Missing object trajectory for '{obj_name}'.")

        q_obj = results["obj_xs"][obj_name][:, :7].copy()
        v_obj = results["obj_xs"][obj_name][:, 7:13].copy()

        for i in range(q_obj.shape[0]):
            R_obj_base = pin.XYZQUATToSE3(q_obj[i, :7]).rotation
            v_obj[i, :3] = R_obj_base @ v_obj[i, :3]

        q_obj[:, [3,4,5,6]] = q_obj[:, [6,3,4,5]]

        traj[f"{obj_name}_pos"] = q_obj[:, :3]
        traj[f"{obj_name}_rot"] = q_obj[:, 3:7]
        traj[f"{obj_name}_lin_vel"] = v_obj[:, :3]
        traj[f"{obj_name}_ang_vel"] = v_obj[:, 3:6]

    contact_id_nodes = _contact_ids_on_solver_nodes(
        scene,
        cms,
        num_modes,
        nodes_per_mode,
    )
    traj["contact_interface_names"] = np.array([json.dumps(key) for key in scene.active_patches], dtype=str)
    if scene.include_robot_ees:
        traj['contact_target_names'] = np.array(
            [json.dumps(key) for key in _scene_contact_targets(scene)], dtype=str)
    traj["contact_id"] = _sample_contact_ids(
        contact_id_nodes,
        original_time,
        time,
    )

    return traj


def save_scene_xml(scene, resolved, path):
    contact_ids = _scene_contact_ids(scene)

    root = ET.Element(
        "scene",
        {
            "name": Path(path).stem,
            "schema_version": "2" if scene.include_robot_ees else "1",
        },
    )
    geometries = ET.SubElement(root, "geometries")

    for obj_name, obj_cfg in scene.fixed_env_objects.items():
        obj_state = resolved.fixed_env_objects[obj_name]
        attrs = _scene_xml_geometry_attrs(
            obj_name,
            obj_cfg,
            obj_state.q,
            contact_ids[obj_name],
            kind="fixed",
        )
        ET.SubElement(geometries, "geometry", attrs)

    for obj_name, obj_cfg in scene.env_objects.items():
        obj_state = resolved.env_objects[obj_name]
        attrs = _scene_xml_geometry_attrs(
            obj_name,
            obj_cfg,
            obj_state.q,
            contact_ids[obj_name],
            kind="movable",
        )
        attrs["mass"] = _format_float(obj_cfg.mass)
        ET.SubElement(geometries, "geometry", attrs)

    if scene.include_robot_ees:
        targets = ET.SubElement(root, 'contact_targets')
        for contact_id, key in enumerate(_scene_contact_targets(scene), 1):
            attrs = dict(contact_id=str(contact_id), kind=key[0])
            if key[0] == 'obj':
                attrs['name'] = key[1]
            else:
                attrs.update(robot_id=key[1], patch=key[2])
            ET.SubElement(targets, 'target', attrs)

    if hasattr(ET, "indent"):
        ET.indent(root)

    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return contact_ids


def _scene_contact_ids(scene):
    contact_ids = {}
    next_id = 1

    for obj_name in scene.fixed_env_objects:
        contact_ids[obj_name] = next_id
        next_id += 1

    for obj_name in scene.env_objects:
        contact_ids[obj_name] = next_id
        next_id += 1

    for key in scene.features.ee_ids:
        contact_ids[key] = next_id
        next_id += 1

    return contact_ids


def _scene_contact_targets(scene):
    """Target table order is the one-based export contact ID contract."""
    return [*(('obj', name) for name in scene.objects), *scene.features.ee_ids]


def _scene_xml_geometry_attrs(obj_name, obj_cfg, q, contact_id, kind):
    attrs = {
        "contact_id": str(int(contact_id)),
        "name": obj_name,
        "kind": kind,
        "type": ("plane" if isinstance(obj_cfg, Plane) else "box"),
        "pos": _format_vec(q[:3]),
        "quat": _format_vec(_pin_quat_xyzw_to_wxyz(q[3:7])),
    }

    attrs["half_size"] = _format_vec(obj_cfg.half_extents)

    return attrs


def _pin_quat_xyzw_to_wxyz(quat_xyzw):
    quat_xyzw = np.asarray(quat_xyzw, dtype=float)
    return quat_xyzw[[3, 0, 1, 2]]


def _format_float(value):
    return f"{float(value):.12g}"


def _format_vec(values):
    return " ".join(_format_float(v) for v in values)


def _contact_ids_on_solver_nodes(scene, cms, num_modes, nodes_per_mode):
    num_modes = int(num_modes)
    nodes_per_mode = int(nodes_per_mode)
    if num_modes < 1:
        raise ValueError("num_modes must be positive.")
    if nodes_per_mode < 1:
        raise ValueError("nodes_per_mode must be positive.")

    interface_names = list(scene.active_patches)
    contact_ids = _scene_contact_ids(scene)
    out = np.zeros(
        (num_modes * nodes_per_mode + 1, len(interface_names)),
        dtype=np.int64,
    )

    for col, interface_name in enumerate(interface_names):
        mode_contacts = list(cms.get(interface_name, [()] * num_modes))
        if len(mode_contacts) != num_modes:
            raise ValueError(
                f"{interface_name}: expected {num_modes} contact modes, "
                f"got {len(mode_contacts)}."
            )

        is_movable_object = interface_name[0] == "obj"
        for mode_id, contact in enumerate(mode_contacts):
            start, end = _contact_segment_bounds(
                mode_id,
                num_modes,
                nodes_per_mode,
            )
            out[start:end, col] = _contact_tuple_to_contact_id(
                scene,
                contact_ids,
                contact,
                is_movable_object,
                context=f"{interface_name}[mode={mode_id}]",
            )

    return out


def _contact_segment_bounds(i, num_modes, nodes_per_mode):
    start = int(i * nodes_per_mode)
    if i == num_modes - 1:
        end = int((i + 1) * nodes_per_mode + 1)
    else:
        end = int((i + 1) * nodes_per_mode)
    return start, end


def _contact_tuple_to_contact_id(
    scene,
    contact_ids,
    contact,
    is_movable_object,
    context,
):
    contact = tuple(contact)
    if len(contact) == 0:
        return 0

    if is_movable_object:
        if len(contact) != 3:
            raise ValueError(f"{context}: expected () or (self_feat_id, env_feat_id, contact_type).")
        env_feat_id = int(contact[1])
    else:
        if len(contact) != 2:
            raise ValueError(f"{context}: expected () or (env_feat_id, contact_type).")
        env_feat_id = int(contact[0])

    if env_feat_id == 0:
        return 0

    owner = scene.features.owner(env_feat_id)
    return int(contact_ids[owner[1] if owner[0] == 'obj' else owner])



def _sample_contact_ids(contact_id_nodes, original_time, query_time):
    if contact_id_nodes.shape[0] != len(original_time):
        raise ValueError(
            "contact node trajectory length must match the original time grid."
        )

    ids = np.searchsorted(original_time, query_time, side="right") - 1
    ids = np.clip(ids, 0, contact_id_nodes.shape[0] - 1)
    return contact_id_nodes[ids].astype(np.int64, copy=False)


def make_se3_body_hermite(T0, V0_body, T1, V1_body, t0, t1):
    """
    Return interp(tau) for SE(3) pose + body twist Hermite interpolation.

    Args:
        T0, T1:
            pin.SE3 endpoint poses.
        V0_body, V1_body:
            Endpoint body twists as 6-vectors in Pinocchio order:
            [linear, angular].
        t0, t1:
            Endpoint times.

    Returns:
        interp(tau) -> T_tau, V_tau_body
    """
    h = float(t1 - t0)

    V0_body = np.asarray(V0_body, dtype=float).reshape(6)
    V1_body = np.asarray(V1_body, dtype=float).reshape(6)

    M01 = T0.inverse() * T1
    A = pin.log6(M01).vector

    m0 = h * V0_body
    m1 = h * (pin.Jlog6(M01) @ V1_body)

    def interp(tau):
        s = (float(tau) - t0) / h

        H10 = s**3 - 2*s**2 + s
        H01 = -2*s**3 + 3*s**2
        H11 = s**3 - s**2

        eta = H10*m0 + H01*A + H11*m1

        dH10 = 3*s**2 - 4*s + 1
        dH01 = -6*s**2 + 6*s
        dH11 = 3*s**2 - 2*s

        deta_ds = dH10*m0 + dH01*A + dH11*m1

        T_tau = T0 * pin.exp6(eta)

        V_tau_body = pin.Jexp6(pin.Motion(eta)) @ (deta_ds / h)

        return T_tau, V_tau_body

    return interp

def make_euclidean_hermite(x0, v0, x1, v1, t0, t1):
    """
    Return interp(tau) for Euclidean position + velocity Hermite interpolation.

    Args:
        x0, x1:
            Endpoint positions, scalar or vector.
        v0, v1:
            Endpoint velocities, same shape as x0/x1.
        t0, t1:
            Endpoint times.

    Returns:
        interp(tau) -> x_tau, v_tau
    """
    h = float(t1 - t0)

    x0 = np.asarray(x0, dtype=float)
    x1 = np.asarray(x1, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    v1 = np.asarray(v1, dtype=float)

    m0 = h * v0
    m1 = h * v1

    def interp(tau):
        s = (float(tau) - t0) / h

        H00 = 2*s**3 - 3*s**2 + 1
        H10 = s**3 - 2*s**2 + s
        H01 = -2*s**3 + 3*s**2
        H11 = s**3 - s**2

        x_tau = H00*x0 + H10*m0 + H01*x1 + H11*m1

        dH00 = 6*s**2 - 6*s
        dH10 = 3*s**2 - 4*s + 1
        dH01 = -6*s**2 + 6*s
        dH11 = 3*s**2 - 2*s

        v_tau = (dH00*x0 + dH10*m0 + dH01*x1 + dH11*m1) / h

        return x_tau, v_tau

    return interp


_ROBOT_ARCHIVE_FIELDS = ("dof_pos", "dof_vel", "root_pos", "root_rot", "root_lin_vel", "root_ang_vel")


def load_trajectory_archive(scene, path):
    """Load versioned or historical single-robot .npz without pickle."""
    with np.load(path, allow_pickle=False) as stored:
        trajectory = {key: stored[key] for key in stored.files}
    return normalize_loaded_trajectory(scene, trajectory)


def normalize_loaded_trajectory(scene, trajectory):
    trajectory = dict(trajectory)
    version = int(np.asarray(trajectory.get("schema_version", 1)).item())
    if version == 1:
        if len(scene.robots) != 1:
            raise ValueError("Legacy single-robot trajectories are ambiguous in a multi-robot scene.")
        rid = next(iter(scene.robots))
        for field in _ROBOT_ARCHIVE_FIELDS:
            if field in trajectory:
                trajectory[robot_name(rid, field)] = trajectory.pop(field)
        trajectory["robot_ids"] = np.array([rid], dtype=str)
        trajectory["object_names"] = np.array(list(scene.env_objects), dtype=str)
        trajectory["schema_version"] = np.array(2, dtype=np.int64)
        if "contact_interface_names" in trajectory:
            from faro.configs.identifiers import import_legacy_contacts
            keys = import_legacy_contacts(scene, {str(key): () for key in trajectory["contact_interface_names"]})
            trajectory["contact_interface_names"] = np.array([json.dumps(key) for key in keys], dtype=str)
    elif version not in (2, 3):
        raise ValueError(f"Unsupported trajectory schema version {version}.")
    if version == 3:
        expected = _scene_contact_targets(scene)
        actual = [tuple(json.loads(key)) for key in trajectory.get('contact_target_names', [])]
        if actual != expected:
            raise ValueError("Trajectory contact target IDs/order must match the scene.")
        ids = np.asarray(trajectory['contact_id'])
        if not np.issubdtype(ids.dtype, np.integer) or np.any(ids < 0) or np.any(ids > len(expected)):
            raise ValueError("Invalid trajectory contact target ID.")
    if tuple(trajectory["robot_ids"]) != tuple(scene.robots):
        raise ValueError("Trajectory robot IDs/order must match the scene.")
    if tuple(trajectory["object_names"]) != tuple(scene.env_objects):
        raise ValueError("Trajectory object names/order must match the scene.")
    time = _as_time_vector(trajectory["time"], min_samples=1)
    for rid, cfg in scene.robots.items():
        dimensions = (cfg.nj, cfg.nj, 3, 4, 3, 3)
        for field, dimension in zip(_ROBOT_ARCHIVE_FIELDS, dimensions):
            name = robot_name(rid, field)
            if name not in trajectory and field in ("dof_vel", "root_lin_vel", "root_ang_vel"):
                trajectory[name] = np.zeros((len(time), dimension))
            value = np.asarray(trajectory[name], dtype=float)
            if value.shape != (len(time), dimension) or not np.all(np.isfinite(value)):
                raise ValueError(f"Invalid archive array {name!r}.")
    for obj in scene.env_objects:
        for field, dimension in (("pos", 3), ("rot", 4), ("lin_vel", 3), ("ang_vel", 3)):
            if f"{obj}_{field}" not in trajectory and field in ("lin_vel", "ang_vel"):
                trajectory[f"{obj}_{field}"] = np.zeros((len(time), dimension))
            value = np.asarray(trajectory[f"{obj}_{field}"], dtype=float)
            if value.shape != (len(time), dimension) or not np.all(np.isfinite(value)):
                raise ValueError(f"Invalid object archive array {obj}_{field}.")
    return trajectory


def trajectory_to_qpos(scene, model, trajectory):
    """Pack a normalized exported trajectory using compiled joint addresses."""
    from faro.configs.mujoco import configurations_to_qpos
    trajectory = normalize_loaded_trajectory(scene, trajectory)
    robot_qs = {}
    for rid in scene.robots:
        q = np.hstack([trajectory[robot_name(rid, field)] for field in ("root_pos", "root_rot", "dof_pos")])
        q[:, 3:7] = q[:, [4, 5, 6, 3]]
        robot_qs[rid] = q
    obj_qs = {}
    for name in scene.env_objects:
        q = np.hstack((trajectory[f"{name}_pos"], trajectory[f"{name}_rot"]))
        q[:, 3:7] = q[:, [4, 5, 6, 3]]
        obj_qs[name] = q
    return configurations_to_qpos(scene, model, robot_qs, obj_qs)
