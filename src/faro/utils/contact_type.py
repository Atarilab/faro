from __future__ import annotations

from typing import Dict, Tuple, Any, Mapping, List, Union
import numpy as np

from faro.configs.identifiers import interface_name

# =========================
# Contact type infrastructure
# =========================

CONTACT_TYPE_UNILATERAL = 1
CONTACT_TYPE_SLIDING    = 2
CONTACT_TYPE_BILATERAL  = 3

CONTACT_TYPE_ORDER = [CONTACT_TYPE_UNILATERAL,
                      CONTACT_TYPE_SLIDING,
                      CONTACT_TYPE_BILATERAL]

CONTACT_TYPE_TO_COL = {ct: j for j, ct in enumerate(CONTACT_TYPE_ORDER)}
NUM_CONTACT_TYPES = len(CONTACT_TYPE_ORDER)

# Type 1 is sticking unilateral contact; type 2 permits tangential slip.
# Bilateral contact remains reserved, without a solver implementation.
SUPPORTED_TYPES = {CONTACT_TYPE_UNILATERAL, CONTACT_TYPE_SLIDING}

def require_supported_contact_type(ct: int, context: str):
    if ct not in SUPPORTED_TYPES:
        raise ValueError(f"{context}: unsupported contact type {ct}")

# =========================
# Shared segmentation helper
# =========================

def _segment_bounds(i: int, num_modes: int, nodes_per_mode: int) -> Tuple[int, int]:
    """Return a mode's half-open sample range, including the terminal sample in the last mode."""
    start = int(i * nodes_per_mode)
    if i == num_modes - 1:
        end = int((i + 1) * nodes_per_mode + 1)
    else:
        end = int((i + 1) * nodes_per_mode)
    return start, end


def _compute_interface_switch_from_blocks(
    *blocks: np.ndarray,
    shift_right: bool = False
) -> np.ndarray:
    """
    Returns a float array of shape (N+1,).

    If shift_right=False:
        switch[t] = 1 iff timestep t differs from timestep t+1
        switch[-1] = 0

    If shift_right=True:
        switch[t] = 1 iff timestep t differs from timestep t-1
        switch[0] = 0

    Each block is expected in row-major timestep format, e.g. (N+1, d_i).
    """
    if len(blocks) == 0:
        raise ValueError("At least one block must be provided.")

    num_steps = blocks[0].shape[0]
    for b in blocks:
        if b.shape[0] != num_steps:
            raise ValueError("All blocks must have the same number of timesteps.")

    interface_mat = np.concatenate(blocks, axis=1)
    switch = np.zeros((num_steps,), dtype=np.float64)

    if num_steps > 1:
        diffs = np.any(interface_mat[1:] != interface_mat[:-1], axis=1).astype(np.float64)

        if shift_right:
            switch[1:] = diffs   # prepend 0
        else:
            switch[:-1] = diffs  # append 0

    return switch

FREE = ()
TupleEE  = Union[Tuple[()], Tuple[int, int]]      # () or (env_id, type)
TupleObj = Union[Tuple[()], Tuple[int, int, int]] # () or (self_id, env_id, type)

def _parse_obj_tuple_strict(item: TupleObj, context: str) -> Tuple[int, int, int]:
    """
    Accepts:
      - ()                           -> FREE
      - (self_id, env_id, ct)        -> active object contact
    Returns: (self_id, env_id, ct) with ints.
    """
    if item == FREE:
        # FREE: no active contact; use numeric sentinels
        return 0, 0, CONTACT_TYPE_UNILATERAL

    if not (isinstance(item, tuple) and len(item) == 3):
        raise TypeError(f"{context}: expected () or (self_id,env_id,type), got {item}")

    self_id, env_id, ct = item
    require_supported_contact_type(int(ct), context=context)
    return int(self_id), int(env_id), int(ct)

def _parse_ee_tuple_strict(item: TupleEE, context: str) -> Tuple[int, int]:
    """
    Accepts:
      - ()                -> FREE
      - (env_id, ct)      -> active EE contact
    Returns: (env_id, ct) with ints.
    """
    if item == FREE:
        return 0, CONTACT_TYPE_UNILATERAL

    if not (isinstance(item, tuple) and len(item) == 2):
        raise TypeError(f"{context}: expected () or (env_id,type), got {item}")

    env_id, ct = item
    require_supported_contact_type(int(ct), context=context)
    return int(env_id), int(ct)

def encode_contact_schedule_from_tuples(
    scene: Any,
    tuple_dict: Mapping[tuple, List[Union[TupleEE, TupleObj]]],
    num_modes: int,
    nodes_per_mode: int,
    extend_contact_into_release: bool = False
) -> Dict[str, np.ndarray]:
    """
    Output (flat dict, solver-ready):
      '{name}_env_feat_oh'   : (scene.total_env_features, N+1)
      '{name}_type_oh'       : (NUM_CONTACT_TYPES, N+1)
      '{obj}_self_feat_oh'   : (self_dim, N+1)   # objects only
      '{name}_is_int_switch' : (N+1,) float      # 1 at interface changes

    EE switches compare with the previous sample; object switches compare
    with the next sample. Only changes in contact geometry count as an
    interface switch: sticking/sliding changes retain contact and loading.

    If extend_contact_into_release=True:
      optional legacy sample extension (KSO and the IK edge oracle use unions).
      when any interface transitions active -> free across a mode boundary,
      the first sample of the release interval keeps the previous active contact
      encoding. This effectively extends the contact into the release by one sample.
    """
    E = int(scene.total_env_features)
    N = int(num_modes * nodes_per_mode)

    if num_modes < 1 or nodes_per_mode < 0 or (nodes_per_mode == 0 and num_modes != 1):
        raise ValueError("num_modes must be positive; zero nodes_per_mode is only valid for one static mode.")
    unknown = set(tuple_dict) - set(scene.active_patches)
    if unknown:
        raise KeyError(f"Unknown contact interfaces: {unknown}")
    from faro.utils.contact_validation import validate_contact_mode, validate_contact_transition
    previous = None
    for key, values in tuple_dict.items():
        if len(values) != num_modes:
            raise ValueError(f"{key}: expected {num_modes} modes, got {len(values)}")
    for i in range(num_modes):
        mode = {}
        for key, values in tuple_dict.items():
            value = values[i]
            if key[0] == 'ee':
                _parse_ee_tuple_strict(value, context=f"{key}[mode={i}]")
            else:
                _parse_obj_tuple_strict(value, context=f"{key}[mode={i}]")
            # Zero sentinels are retained for compatibility with historical schedules.
            mode[key] = value if value and value[-2] != 0 else ()
        validate_contact_mode(scene, mode)
        if previous is not None:
            validate_contact_transition(scene, previous, mode)
        previous = mode
    ee_names = list(scene.robot_interfaces)
    obj_names = list(scene.env_objects)

    # ---- Build object schedules/mats first
    obj_env_mat: Dict[str, np.ndarray] = {}
    obj_self_mat: Dict[str, np.ndarray] = {}
    obj_type_mat: Dict[str, np.ndarray] = {}

    for obj_name in obj_names:
        if ("obj", obj_name) not in tuple_dict:
            continue
        tup_list = tuple_dict[("obj", obj_name)]
        if len(tup_list) != num_modes:
            raise ValueError(f"{obj_name}: expected {num_modes} modes, got {len(tup_list)}")

        self_feat_dim = scene.objects[obj_name].num_features

        self_ids = [0] * num_modes
        env_ids = [0] * num_modes
        cts = [CONTACT_TYPE_UNILATERAL] * num_modes
        free = [True] * num_modes

        for i, item in enumerate(tup_list):
            sid, eid, ct = _parse_obj_tuple_strict(item, context=f"{obj_name}[mode={i}]")
            self_ids[i] = sid
            env_ids[i] = eid
            cts[i] = ct
            free[i] = (eid == 0 or sid == 0)

            if eid != 0 and not (1 <= eid <= E):
                raise ValueError(f"{obj_name}[mode={i}]: env_id={eid} out of range 1..{E}")
            if sid != 0 and not (1 <= sid <= self_feat_dim):
                raise ValueError(f"{obj_name}[mode={i}]: self_id={sid} out of range 1..{self_feat_dim}")

        env_mat = np.zeros((N + 1, E), dtype=float)
        self_mat = np.zeros((N + 1, self_feat_dim), dtype=float)
        type_mat = np.zeros((N + 1, NUM_CONTACT_TYPES), dtype=float)

        for i in range(num_modes):
            start, end = _segment_bounds(i, num_modes, nodes_per_mode)
            if free[i]:
                continue
            env_mat[start:end, env_ids[i] - 1] = 1.0
            self_mat[start:end, self_ids[i] - 1] = 1.0
            type_mat[start:end, CONTACT_TYPE_TO_COL[cts[i]]] = 1.0

        if extend_contact_into_release:
            for i in range(1, num_modes):
                if (not free[i - 1]) and free[i]:
                    start, end = _segment_bounds(i, num_modes, nodes_per_mode)
                    if start < end:
                        env_mat[start, :] = 0.0
                        self_mat[start, :] = 0.0
                        type_mat[start, :] = 0.0

                        env_mat[start, env_ids[i - 1] - 1] = 1.0
                        self_mat[start, self_ids[i - 1] - 1] = 1.0
                        type_mat[start, CONTACT_TYPE_TO_COL[cts[i - 1]]] = 1.0

        obj_env_mat[obj_name] = env_mat
        obj_self_mat[obj_name] = self_mat
        obj_type_mat[obj_name] = type_mat

    # ---- Build EE schedules/mats
    ee_env_mat: Dict[str, np.ndarray] = {}
    ee_type_mat: Dict[str, np.ndarray] = {}

    for ee_name in ee_names:
        if ee_name not in tuple_dict:
            continue
        tup_list = tuple_dict[ee_name]
        if len(tup_list) != num_modes:
            raise ValueError(f"{ee_name}: expected {num_modes} modes, got {len(tup_list)}")

        env_ids = [0] * num_modes
        cts = [CONTACT_TYPE_UNILATERAL] * num_modes
        free = [True] * num_modes

        for i, item in enumerate(tup_list):
            eid, ct = _parse_ee_tuple_strict(item, context=f"{ee_name}[mode={i}]")
            env_ids[i] = eid
            cts[i] = ct
            free[i] = (eid == 0)

            if eid != 0 and not (1 <= eid <= E):
                raise ValueError(f"{ee_name}[mode={i}]: env_id={eid} out of range 1..{E}")

        env_mat = np.zeros((N + 1, E), dtype=float)
        type_mat = np.zeros((N + 1, NUM_CONTACT_TYPES), dtype=float)

        for i in range(num_modes):
            start, end = _segment_bounds(i, num_modes, nodes_per_mode)
            if env_ids[i] == 0:
                continue
            env_mat[start:end, env_ids[i] - 1] = 1.0
            type_mat[start:end, CONTACT_TYPE_TO_COL[cts[i]]] = 1.0

        if extend_contact_into_release:
            for i in range(1, num_modes):
                if (not free[i - 1]) and free[i]:
                    start, end = _segment_bounds(i, num_modes, nodes_per_mode)
                    if start < end:
                        env_mat[start, :] = 0.0
                        type_mat[start, :] = 0.0

                        env_mat[start, env_ids[i - 1] - 1] = 1.0
                        type_mat[start, CONTACT_TYPE_TO_COL[cts[i - 1]]] = 1.0

        ee_env_mat[ee_name] = env_mat
        ee_type_mat[ee_name] = type_mat

    # ---- Pack outputs (transpose to solver format)
    traj_params: Dict[str, np.ndarray] = {}

    # EEs present in tuple_dict
    for ee_key in ee_env_mat.keys():
        ee_name = interface_name(ee_key)
        traj_params[f"{ee_name}_env_feat_oh"] = ee_env_mat[ee_key].T
        traj_params[f"{ee_name}_type_oh"] = ee_type_mat[ee_key].T
        traj_params[f"{ee_name}_is_int_switch"] = _compute_interface_switch_from_blocks(
            ee_env_mat[ee_key],
            shift_right=True
        )

    # Objects present in tuple_dict
    for obj_name in obj_env_mat.keys():
        traj_params[f"{obj_name}_env_feat_oh"] = obj_env_mat[obj_name].T
        traj_params[f"{obj_name}_self_feat_oh"] = obj_self_mat[obj_name].T
        traj_params[f"{obj_name}_type_oh"] = obj_type_mat[obj_name].T
        traj_params[f"{obj_name}_is_int_switch"] = _compute_interface_switch_from_blocks(
            obj_env_mat[obj_name],
            obj_self_mat[obj_name],
        )

    return traj_params


def interval_time_factors(scene, interface_one_hots):
    """Return shared step factors, shortening a simultaneous switch only once.

    Indices follow the TO transition convention: each switch flag applies to
    the interval beginning at that solver node. The terminal entry is retained
    so this can also be used with the solver's parameter trajectory.
    """
    values = list(interface_one_hots.values())
    if not values:
        raise ValueError("A contact parameter trajectory is required.")
    n = np.asarray(values[0]).shape[-1]
    switches = np.zeros(n, dtype=bool)
    for robot_id, robot in scene.robots.items():
        for patch in robot.switch_contacts:
            name = interface_name(("ee", robot_id, patch))
            flag = np.asarray(interface_one_hots[f"{name}_is_int_switch"]).reshape(-1)
            if len(flag) != n:
                raise ValueError("Contact switch trajectories must have equal length.")
            switches |= flag >= 1.0
    return np.where(switches, 0.1, 1.0)
