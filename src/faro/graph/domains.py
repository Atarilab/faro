from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from faro.graph.ct_types import InterfaceKey
from faro.graph.modespace import ModeSpace
from faro.utils.contact_type import SUPPORTED_TYPES


@dataclass(frozen=True)
class AllowedEE:
    env_ids: Sequence[int]
    contact_types: Sequence[int]


@dataclass(frozen=True)
class AllowedObj:
    self_to_env_ids: Mapping[int, Sequence[int]]
    contact_types: Sequence[int]


@dataclass(frozen=True)
class AllowedSpec:
    ee: Mapping[InterfaceKey, AllowedEE]
    obj: Mapping[InterfaceKey, AllowedObj]


def build_modespace_from_allowed(scene: Any, allowed: AllowedSpec) -> ModeSpace:
    admitted_by_env: Mapping[int, Tuple[int, ...]] = scene.features.admitted
    valid_env_ids = set(scene.features.by_id)

    def _check_env_id(env_id: int, ctx: str) -> None:
        if env_id not in valid_env_ids:
            raise ValueError(f"{ctx}: invalid env_id={env_id}.")
        if int(env_id) not in admitted_by_env:
            raise ValueError(
                f"{ctx}: env_id={env_id} missing from scene.features.admitted (should be covered)."
            )

    def _effective_cts(*ct_sets: Sequence[int]) -> Tuple[int, ...]:
        if len(ct_sets) == 0:
            return ()
        seed = [int(x) for x in ct_sets[0]]
        others = [set(int(x) for x in seq) for seq in ct_sets[1:]]
        out: list[int] = []
        seen: set[int] = set()
        for ct in seed:
            if ct in seen or ct not in SUPPORTED_TYPES:
                continue
            if all(ct in s for s in others):
                out.append(ct)
                seen.add(ct)
        return tuple(out)

    def _dedupe_preserve_order(vals: Sequence[tuple]) -> list[tuple]:
        out: list[tuple] = []
        seen: set[tuple] = set()
        for v in vals:
            # normalize to plain int tuples defensively
            vv = tuple(int(x) for x in v)
            if vv in seen:
                continue
            seen.add(vv)
            out.append(vv)
        return out

    keys: list[InterfaceKey] = []
    per_key_values: Dict[InterfaceKey, list[tuple]] = {}

    # -----------------
    # EEs (explicit)
    # -----------------
    for k in scene.robot_interfaces:
        keys.append(k)

        if k not in allowed.ee:
            raise KeyError(f"AllowedSpec missing EE key {k}")

        spec = allowed.ee[k]
        env_ids = [int(e) for e in spec.env_ids]
        allowed_cts = [int(ct) for ct in spec.contact_types]

        vals: list[tuple] = []
        for env_id in env_ids:
            _check_env_id(env_id, ctx=f"EE {k}")
            if scene.features.owner(env_id) == k:
                continue
            effective_cts = _effective_cts(allowed_cts, admitted_by_env[int(env_id)])
            for ct in effective_cts:
                vals.append((env_id, ct))

        per_key_values[k] = _dedupe_preserve_order(vals)

    # -----------------
    # Objects (explicit)
    # -----------------
    for obj_name, obj_cfg in scene.env_objects.items():
        k = ("obj", obj_name)
        keys.append(k)

        if k not in allowed.obj:
            raise KeyError(f"AllowedSpec missing OBJ key {k}")

        spec = allowed.obj[k]
        num_faces = int(obj_cfg.num_features)
        allowed_cts = [int(ct) for ct in spec.contact_types]

        vals: list[tuple] = []
        for self_id_raw, env_ids_raw in spec.self_to_env_ids.items():
            self_id = int(self_id_raw)
            if self_id < 1 or self_id > num_faces:
                raise ValueError(f"OBJ {k}: invalid self_id {self_id}, expected 1..{num_faces}")

            self_env_id = scene.faces.ids_by_object[obj_name][self_id - 1]

            env_ids = [int(e) for e in env_ids_raw]
            for env_id in env_ids:
                _check_env_id(env_id, ctx=f"OBJ {k} (target env feature)")

                if scene.features.owner(env_id) == ('obj', obj_name):
                    continue

                effective_cts = _effective_cts(
                    allowed_cts,
                    admitted_by_env[self_env_id],
                    admitted_by_env[int(env_id)],
                )
                for ct in effective_cts:
                    vals.append((self_id, env_id, ct))

        per_key_values[k] = _dedupe_preserve_order(vals)

    return ModeSpace(keys=keys, per_key_values=per_key_values)
