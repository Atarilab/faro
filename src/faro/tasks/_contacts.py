"""Contact parameter selection and geometry checks shared by tasks."""

import numpy as np
from faro.configs.identifiers import interface_name

from faro.utils.contact_type import encode_contact_schedule_from_tuples
from faro.utils.contact_mode import contact_mode_union
from faro.utils.geom_utils import rectangle_containment


def interface_param_keys(scene, include_switches=False):
    for key in scene.robot_interfaces:
        name = interface_name(key)
        yield f"{name}_env_feat_oh"
        yield f"{name}_type_oh"
        if include_switches:
            yield f"{name}_is_int_switch"
    for name in scene.env_objects:
        yield f"{name}_self_feat_oh"
        yield f"{name}_env_feat_oh"
        yield f"{name}_type_oh"
        if include_switches:
            yield f"{name}_is_int_switch"


def select_interface_params(scene, trajectory_params, include_switches=False):
    return {key: np.array(trajectory_params[key])
            for key in interface_param_keys(scene, include_switches)}


def ik_interface_params(scene, contact_mode):
    """Encode one already-assembled contact mode."""
    mode = dict.fromkeys(scene.active_patches, ())
    mode.update(contact_mode)
    encoded = select_interface_params(scene, encode_contact_schedule_from_tuples(
        scene, {key: [value] for key, value in mode.items()}, 1, 0))
    return {key: value for key, value in encoded.items() if not key.endswith('_type_oh')}


def kso_interface_params(scene, tuple_dict, num_modes, *, hard_initial, terminal_endpoint):
    """Encode modes on either side of KSO's transition states.

    Unsuffixed parameters describe the outgoing mode (also the mode governing
    motion to the next state); ``*_incoming_oh`` describes the incoming mode.
    A missing endpoint side is FREE. Each transition must have at most one
    active geometry per source interface.
    """
    complete = {key: [()] * num_modes for key in scene.active_patches}
    complete.update(tuple_dict)
    encoded = encode_contact_schedule_from_tuples(
        scene, complete, num_modes, nodes_per_mode=1,
    )
    for i in range(num_modes - 1):
        contact_mode_union({key: values[i] for key, values in complete.items()},
                           {key: values[i + 1] for key, values in complete.items()})
    # Column num_modes below is FREE, not the encoder's repeated last mode.
    incoming = ([num_modes] if hard_initial else []) + list(range(num_modes - 1))
    outgoing = ([0] if hard_initial else []) + list(range(1, num_modes))
    if terminal_endpoint:
        incoming.append(num_modes - 1)
        outgoing.append(num_modes)
    if not incoming:
        raise ValueError("KSO requires at least one transition or an endpoint.")
    params = {}
    for key in interface_param_keys(scene):
        modes = encoded[key][:, :num_modes]
        modes = np.column_stack((modes, np.zeros(modes.shape[0])))
        params[key] = modes[:, outgoing]
        params[key.removesuffix('_oh') + '_incoming_oh'] = modes[:, incoming]
    return params


def validate_interface_containment(scene, object_features,
                                   interface_one_hots, self_interface_ohs, mode_begins,
                                   *, use_object_sizes=False):
    sizes = {name: cfg.half_extents for name, cfg in scene.objects.items()}
    env_features = object_features.local_features(sizes)
    for interface in scene.active_patches:
        prefix = interface_name(interface)
        name = f"{prefix}_env_feat_oh"
        if name not in interface_one_hots:
            continue
        oh_traj = interface_one_hots[name]
        self_oh_traj = None
        if interface[0] == "obj":
            self_oh_traj = self_interface_ohs[f"{prefix}_self_feat_oh"]

        for mode_begin in mode_begins:
            env_interface = oh_traj[:, mode_begin]

            env_feat = (env_features @ env_interface).full().flatten().tolist()

            if interface[0] == "ee":
                bnds = scene.robot_interfaces[interface]
            else:
                self_feat = self_oh_traj[:, mode_begin]
                # Zero-size self bounds skip object containment checks.
                self_feature = object_features.local_feature(
                    interface[1], sizes[interface[1]] if use_object_sizes else np.zeros(3), self_feat
                )
                bnds = np.asarray(self_feature[12:16]).reshape(-1)

            if sum(env_feat[-4:]) < 1e-3 or sum(bnds) < 1e-3:
                continue

            if rectangle_containment(env_feat[-4:], bnds):
                continue
            else:
                raise ValueError(f"Patch {interface!r} and interface {env_interface} cannot contain eachother")
