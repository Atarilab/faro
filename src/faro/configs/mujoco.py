"""Build a visualization model for an explicit scene state."""
import mujoco
import numpy as np
import pinocchio as pin

from faro.configs.identifiers import solver_name

from faro.configs.objects import Plane


def _make_visualization_spec(scene):
    """Shared model settings and a soft gray background for all viewers."""
    spec = mujoco.MjSpec()
    spec.option.integrator = next(iter(scene.robots.values())).mj_spec.option.integrator
    spec.add_texture(
        name="skybox",
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        width=256,
        height=1536,
        rgb1=[0.35, 0.52, 0.72],  # soft blue overhead
        rgb2=[0.98, 0.98, 0.99],  # light neutral horizon
    )
    return spec


def _add_classic_groundplane_material(spec, prefix="") -> str:
    tex = spec.add_texture(
        name=prefix + "groundplane_tex",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        width=300,
        height=300,
        rgb1=[0.2, 0.3, 0.4],
        rgb2=[0.1, 0.2, 0.3],
        mark=mujoco.mjtMark.mjMARK_EDGE,
        markrgb=[0.8, 0.8, 0.8],
    )
    mat = spec.add_material(
        name=prefix + "groundplane_mat",
        texrepeat=[5, 5],
        texuniform=True,
        reflectance=0.0,
    )
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
    return mat.name


def _add_state(spec, scene, state, robots, objects, *,
               fixed_environment, prefix="", alpha=1.0):
    # copy() preserves assets and never mutates the robot's source spec.
    for robot_id in robots:
        robot_spec = scene.robots[robot_id].mj_spec.copy()
        if alpha != 1.0:
            _scale_spec_alpha(robot_spec, alpha)
        spec.attach(robot_spec, prefix=prefix + robot_prefix(robot_id),
                    frame=spec.worldbody.add_frame())
    for name in objects:
        cfg = scene.env_objects[name]
        rgba = np.array(cfg.color, dtype=float, copy=True)
        rgba[3] *= alpha
        body = spec.worldbody.add_body(name=f"{prefix}{name}-base")
        body.add_freejoint(name=f"{prefix}{name}-floating_base_joint")
        body.add_geom(
            name=f"{prefix}{name}-{name}_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
            size=cfg.half_extents, mass=cfg.mass, rgba=rgba,
        )
        # Face sites share the contact geometry's position and orientation.
        for i, feature in enumerate(cfg.features.T, 1):
            quat = np.empty(4)
            mujoco.mju_mat2Quat(quat, feature[3:12].reshape(3, 3, order="F").ravel())
            body.add_site(name=f"{prefix}{name}-face{i}", pos=feature[:3], quat=quat,
                          size=(0.01, 0.01, 0.01), rgba=(0, 0, 1, 1))

    if fixed_environment:
        material = _add_classic_groundplane_material(spec, prefix)
        for name, cfg in scene.fixed_env_objects.items():
            q = state.fixed_env_objects[name].q
            plane = isinstance(cfg, Plane)
            spec.worldbody.add_geom(
                name=prefix + name,
                type=mujoco.mjtGeom.mjGEOM_PLANE if plane else mujoco.mjtGeom.mjGEOM_BOX,
                pos=q[:3], quat=q[[6, 3, 4, 5]],
                size=(*cfg.half_extents, 0.1) if plane else cfg.half_extents,
                material=material if plane else "",
                rgba=(1, 1, 1, 1) if plane else cfg.color,
            )


def _scale_spec_alpha(spec, alpha):
    """Fade copied geometry without overriding material-provided colors."""
    for material in spec.materials:
        material.rgba[3] *= alpha
    for geom in spec.geoms:
        # MuJoCo uses material RGBA only when geom RGBA has its default value.
        if geom.material and np.array_equal(geom.rgba, [0.5, 0.5, 0.5, 1.0]):
            continue
        geom.rgba[3] *= alpha


def build_mujoco(scene, state):
    """Build a model and its initial qpos from one resolved scene state."""
    spec = _make_visualization_spec(scene)
    _add_state(spec, scene, state, scene.robots, scene.env_objects,
                fixed_environment=True)
    model = spec.compile()
    qpos, _ = pack_trajectory(
        scene, model,
        {rid: value.x[None, :] for rid, value in state.robots.items()},
        {name: value.x[None, :] for name, value in state.env_objects.items()},
    )
    return model, qpos[0]


def build_mujoco_comparison(scene, state, comparison_state, *,
                            robots=None, objects=None, comparison_alpha=0.25):
    """Overlay two resolved states in a single visualization model.

    The initial state retains its normal appearance. ``robots`` and ``objects``
    select translucent comparison bodies; None selects all and () selects none.
    comparison_alpha scales their original opacity. Fixed objects must have
    the same poses and are drawn once, unfaded.
    Each copy uses an ``initial__`` or ``comparison__`` name prefix.
    """
    comparison_alpha = float(comparison_alpha)
    if not np.isfinite(comparison_alpha) or not 0 <= comparison_alpha <= 1:
        raise ValueError("comparison_alpha must be finite and between 0 and 1.")
    robots = _select_owners(robots, scene.robots, "robot")
    objects = _select_owners(objects, scene.env_objects, "movable object")
    for name in scene.fixed_env_objects:
        first = pin.XYZQUATToSE3(state.fixed_env_objects[name].q)
        second = pin.XYZQUATToSE3(comparison_state.fixed_env_objects[name].q)
        if not np.allclose(first.homogeneous, second.homogeneous, atol=1e-8, rtol=0):
            raise ValueError(f"Fixed-object poses must match for comparison: {name!r}.")

    layers = (
        ("initial__", state, scene.robots, scene.env_objects, 1.0, True),
        ("comparison__", comparison_state, robots, objects, comparison_alpha, False),
    )
    spec = _make_visualization_spec(scene)
    for prefix, resolved, robot_ids, object_names, alpha, fixed in layers:
        _add_state(spec, scene, resolved, robot_ids, object_names,
                    fixed_environment=fixed, prefix=prefix, alpha=alpha)
    model = spec.compile()
    qpos = model.qpos0[None, :].copy()
    qvel = np.zeros((1, model.nv))
    for prefix, resolved, robot_ids, object_names, _, _ in layers:
        _pack_trajectories(
            scene, model,
            {rid: resolved.robots[rid].q[None, :] for rid in robot_ids},
            {name: resolved.env_objects[name].q[None, :] for name in object_names},
            qpos, qvel, prefix=prefix,
        )
    return model, qpos[0]


def _select_owners(selection, owners, kind):
    selected = set(owners if selection is None else selection)
    if unknown := selected - owners.keys():
        raise ValueError(f"Unknown comparison {kind} names: {sorted(unknown)}")
    return tuple(name for name in owners if name in selected)


def robot_prefix(robot_id):
    """MuJoCo attachment prefix for one instance's local model names."""
    return solver_name("robot", robot_id) + "__"


def pack_trajectory(scene, model, robot_xs, obj_xs):
    """Pack Pinocchio states using compiled MuJoCo joint addresses.

    Robot free-joint linear velocity is rotated to world coordinates; angular
    velocity retains MuJoCo's local body convention. Configurations use wxyz.
    The source models remain untouched and their joint order need not match
    MuJoCo's compiled traversal order.
    """
    if set(robot_xs) != set(scene.robots):
        raise ValueError("Robot trajectory IDs must match the scene.")
    n = len(next(iter(robot_xs.values())))
    qpos = np.tile(model.qpos0, (n, 1))
    qvel = np.zeros((n, model.nv))
    _pack_trajectories(scene, model, robot_xs,
                       {name: obj_xs[name] for name in scene.env_objects}, qpos, qvel)
    return qpos, qvel


def _pack_trajectories(scene, model, robot_xs, obj_xs, qpos, qvel, *, prefix=""):
    """Fill named owners in preallocated arrays, optionally under a namespace."""
    n = len(qpos)
    for robot_id, values in robot_xs.items():
        cfg = scene.robots[robot_id]
        states = np.asarray(values, dtype=float)
        pm = cfg.pin_model
        if states.ndim != 2 or len(states) != n or states.shape[1] < pm.nq:
            raise ValueError(f"Invalid trajectory for robot {robot_id!r}.")
        has_velocity = states.shape[1] >= pm.nq + pm.nv
        for jid in range(1, pm.njoints):
            pj = pm.joints[jid]
            mj = model.joint(prefix + robot_prefix(robot_id) + pm.names[jid])
            qa, va = int(mj.qposadr[0]), int(mj.dofadr[0])
            q = states[:, pj.idx_q:pj.idx_q + pj.nq]
            if pj.nq == 7:
                qpos[:, qa:qa + 7] = q[:, [0, 1, 2, 6, 3, 4, 5]]
            else:
                qpos[:, qa:qa + pj.nq] = q
            if has_velocity:
                v = states[:, pm.nq + pj.idx_v:pm.nq + pj.idx_v + pj.nv].copy()
                if pj.nq == 7:
                    for k in range(n):
                        v[k, :3] = pin.XYZQUATToSE3(q[k]).rotation @ v[k, :3]
                qvel[:, va:va + pj.nv] = v
    for name, values in obj_xs.items():
        states = np.asarray(values, dtype=float)
        if states.ndim != 2 or len(states) != n or states.shape[1] < 7:
            raise ValueError(f"Invalid trajectory for object {name!r}.")
        joint = model.joint(f"{prefix}{name}-floating_base_joint")
        qa, va = int(joint.qposadr[0]), int(joint.dofadr[0])
        qpos[:, qa:qa + 7] = states[:, [0, 1, 2, 6, 3, 4, 5]]
        if states.shape[1] >= 13:
            v = states[:, 7:13].copy()
            for k in range(n):
                v[k, :3] = pin.XYZQUATToSE3(states[k, :7]).rotation @ v[k, :3]
            qvel[:, va:va + 6] = v


def configurations_to_qpos(scene, model, robot_qs, obj_qs):
    """Assemble trajectories of configurations for every robot and object."""
    return pack_trajectory(scene, model, robot_qs, obj_qs)[0]
