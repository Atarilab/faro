"""Ownership views for flat CasADi solver blocks (without changing coordinates)."""
from faro.configs.identifiers import robot_name, interface_name


def owned_warm_start(states, inputs, robot_cfgs, env_cfgs, state_field):
    """Expose backend coordinates as time-major arrays with explicit ownership."""
    state = {
        "robots": {rid: states[robot_name(rid, state_field)].T.copy() for rid in robot_cfgs},
        "objects": {name: states[name].T.copy() for name in env_cfgs if name in states},
    }
    controls = {"robots": {}, "objects": {}}
    wrenches = {}
    for rid, cfg in robot_cfgs.items():
        robot_controls = {}
        for field, label in (("q_next_rob", "next_configuration"),
                             ("dq_next_rob", "next_configuration"), ("vdot_rob", "acceleration")):
            key = robot_name(rid, field)
            if key in inputs:
                robot_controls[label] = inputs[key].T.copy()
        wrenches[rid] = {patch: inputs[interface_name(("ee", rid, patch))].T.copy()
                         for patch in cfg.patch_ees if interface_name(("ee", rid, patch)) in inputs}
        if wrenches[rid]:
            robot_controls["wrenches"] = wrenches[rid]
        controls["robots"][rid] = robot_controls
    for name in env_cfgs:
        key = name if name in inputs else name + "_W"
        if key in inputs:
            controls["objects"][name] = inputs[key].T.copy()
    if "t_scale" in inputs:
        controls["t_scale"] = inputs["t_scale"].reshape(-1).copy()
    object_wrenches = {name: inputs[name + "_W"].T.copy()
                       for name in env_cfgs if name + "_W" in inputs}
    return {"x0_dict": state, "u0_dict": controls, "rob_Ws": wrenches,
            "obj_Ws": object_wrenches}


def validate_owned_warm_start(x0_dict, u0_dict, x_symbols, u_symbols,
                             state_rows, control_rows):
    """Validate owned trajectories and return flat blocks in backend ordering.

    Robot IDs and local patch names remain explicit throughout conversion. The
    complete generated block set must match the OCP, so missing owners cannot be
    silently dropped. No solver storage is touched by this function.
    """
    from collections.abc import Mapping
    import numpy as np

    def mapping(value, label, keys=None):
        if not isinstance(value, Mapping):
            raise ValueError(f"{label} must be a mapping")
        if keys is not None and set(value) != set(keys):
            raise ValueError(f"{label} keys differ: missing {set(keys) - set(value)}, unexpected {set(value) - set(keys)}")
        return value

    def array(value, rows, dimension, label):
        result = np.asarray(value, dtype=float)
        if result.shape != (rows, dimension):
            raise ValueError(f"{label}: expected {(rows, dimension)}, got {result.shape}")
        if not np.isfinite(result).all():
            raise ValueError(f"{label} must contain only finite values")
        return result.copy()

    mapping(x0_dict, "states", ("robots", "objects"))
    control_sections = ("robots", "objects", "t_scale") if "t_scale" in u_symbols else ("robots", "objects")
    mapping(u0_dict, "controls", control_sections)
    robots = mapping(x0_dict["robots"], "states.robots")
    objects = mapping(x0_dict["objects"], "states.objects")
    mapping(u0_dict["robots"], "controls.robots", robots)
    controlled_objects = {name for name in objects if name in u_symbols or name + "_W" in u_symbols}
    mapping(u0_dict["objects"], "controls.objects", controlled_objects)
    x_blocks, u_blocks = {}, {}
    for rid, values in robots.items():
        if not isinstance(rid, str) or not rid:
            raise ValueError(f"Invalid robot instance ID {rid!r}")
        matches = [robot_name(rid, field) for field in ("q_rob", "dq_rob", "rob_dx")
                   if robot_name(rid, field) in x_symbols]
        if len(matches) != 1:
            raise ValueError(f"No unambiguous robot state block for {rid!r}")
        key = matches[0]
        x_blocks[key] = array(values, state_rows, int(x_symbols[key].numel()), f"states.robots.{rid}")
        control = mapping(u0_dict["robots"][rid], f"controls.robots.{rid}")
        for label, value in control.items():
            if label == "wrenches":
                patches = mapping(value, f"controls.robots.{rid}.wrenches")
                if not patches:
                    raise ValueError(f"controls.robots.{rid}.wrenches must contain declared patch blocks")
                for patch, wrench in patches.items():
                    if not isinstance(patch, str) or not patch:
                        raise ValueError(f"Invalid patch name {patch!r}")
                    key = interface_name(("ee", rid, patch))
                    if key not in u_symbols:
                        raise ValueError(f"Unknown wrench interface {(rid, patch)!r}")
                    u_blocks[key] = array(wrench, control_rows, int(u_symbols[key].numel()), f"controls.robots.{rid}.wrenches.{patch}")
            else:
                fields = {"next_configuration": ("q_next_rob" if robot_name(rid, "q_next_rob") in u_symbols
                                                  else "dq_next_rob"),
                          "acceleration": "vdot_rob"}
                if label not in fields:
                    raise ValueError(f"Unknown robot control field {label!r} for {rid!r}")
                key = robot_name(rid, fields[label])
                if key not in u_symbols:
                    raise ValueError(f"Robot {rid!r} has no {label!r} control block")
                u_blocks[key] = array(value, control_rows, int(u_symbols[key].numel()), f"controls.robots.{rid}.{label}")
    for name, values in objects.items():
        if not isinstance(name, str) or name not in x_symbols:
            raise ValueError(f"Unknown object state {name!r}")
        if name in x_blocks:
            raise ValueError(f"Duplicate state block {name!r}")
        x_blocks[name] = array(values, state_rows, int(x_symbols[name].numel()), f"states.objects.{name}")
        if name not in controlled_objects:
            continue
        key = name if name in u_symbols else name + "_W"
        if key in u_blocks:
            raise ValueError(f"Duplicate control block {key!r}")
        u_blocks[key] = array(u0_dict["objects"][name], control_rows, int(u_symbols[key].numel()), f"controls.objects.{name}")
    if "t_scale" in u_symbols:
        value = np.asarray(u0_dict["t_scale"], dtype=float)
        if int(u_symbols["t_scale"].numel()) != 1:
            raise ValueError("The shared t_scale control must be scalar")
        if value.shape != (control_rows,) or not np.isfinite(value).all():
            raise ValueError(f"controls.t_scale: expected {control_rows} finite values")
        u_blocks["t_scale"] = value[:, None].copy()
    mapping(x_blocks, "flat state blocks", x_symbols)
    mapping(u_blocks, "flat control blocks", u_symbols)
    return ({name: x_blocks[name] for name in x_symbols},
            {name: u_blocks[name] for name in u_symbols})
