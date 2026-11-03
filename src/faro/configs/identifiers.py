"""Instance-owned identifiers shared by authoring, optimization and rendering.

Local model frame names are never renamed or parsed to infer ownership.
"""
from __future__ import annotations

from hashlib import sha256

InterfaceKey = tuple[str, str, str] | tuple[str, str]


def solver_name(*parts: str) -> str:
    """Unambiguous component encoding with bounded native artifact filenames.

    SHA-256 bounds long compound names; ownership lives in the scene/index,
    never in a decoded symbol name. Leave room for native derivative suffixes.
    """
    encoded = "n_" + "_".join(str(part).encode("utf-8").hex() for part in parts)
    return encoded if len(encoded) <= 120 else "n_" + sha256(encoded.encode("ascii")).hexdigest()


def robot_name(robot_id: str, field: str) -> str:
    return solver_name("robot", robot_id, field)


def interface_name(key: InterfaceKey) -> str:
    """Parameter prefix; retain object prefixes and namespace robot patches."""
    if len(key) == 2 and key[0] == "obj":
        return key[1]
    if len(key) == 3 and key[0] == "ee":
        return solver_name(*key)
    raise ValueError(f"Invalid contact interface: {key!r}")


def capsule_name(robot_id: str, capsule: str) -> str:
    return solver_name("capsule", robot_id, capsule)


def import_legacy_contacts(scene, contacts):
    """Normalize historical file keys at the import boundary only."""
    out = {}
    for key, values in contacts.items():
        if isinstance(key, tuple) and key in scene.active_patches:
            canonical = key
        elif isinstance(key, str) and key in scene.env_objects:
            canonical = ("obj", key)
        elif isinstance(key, str) or (isinstance(key, tuple) and len(key) == 2 and key[0] == "ee"):
            if len(scene.robots) != 1:
                raise ValueError("Legacy robot contact keys are ambiguous in a multi-robot scene.")
            robot_id = next(iter(scene.robots))
            canonical = ("ee", robot_id, key if isinstance(key, str) else key[1])
        else:
            canonical = key
        if canonical not in scene.active_patches:
            raise ValueError(f"Unknown contact interface {key!r}.")
        if canonical in out:
            raise ValueError(f"Duplicate contact interface {canonical!r}.")
        out[canonical] = values
    return out
