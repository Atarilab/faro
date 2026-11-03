"""The three supported scene objects. Dimensions are half lengths in metres."""
from dataclasses import dataclass, field
from functools import cached_property
from types import MappingProxyType
from typing import Mapping

import coal
import numpy as np

from faro.utils.geom_utils import BOX_FACES, box_inertia, get_box_features, get_plane_features


def _validate(obj, dimensions, faces):
    extents = tuple(float(x) for x in obj.half_extents)
    if len(extents) != dimensions or not all(np.isfinite(x) and x > 0 for x in extents):
        raise ValueError(f"half_extents must contain {dimensions} positive finite lengths.")
    color = tuple(float(x) for x in obj.color)
    if len(color) != 4 or not all(0 <= x <= 1 for x in color):
        raise ValueError("color must contain four RGBA values in [0, 1].")
    permissions = {face: tuple(cts) for face, cts in obj.face_contact_types.items()}
    if unknown := permissions.keys() - set(faces):
        raise ValueError(f"Unknown faces: {sorted(unknown)}")
    object.__setattr__(obj, 'half_extents', extents)
    object.__setattr__(obj, 'color', color)
    object.__setattr__(obj, 'contact_types', tuple(obj.contact_types))
    object.__setattr__(obj, 'face_contact_types', MappingProxyType(permissions))


@dataclass(frozen=True)
class Plane:
    """A bounded contact surface with infinite halfspace collision geometry."""
    half_extents: tuple[float, float] = (10.0, 10.0)
    color: tuple[float, ...] = (0.8, 0.8, 0.8, 1.0)
    contact_types: tuple[int, ...] = (1,)
    face_contact_types: Mapping[str, tuple[int, ...]] = field(default_factory=dict)

    def __post_init__(self):
        _validate(self, 2, ('surface',))

    num_features = 1

    @cached_property
    def features(self):
        return get_plane_features(self.half_extents).reshape(16, 1)

    @cached_property
    def coal_shape(self):
        return coal.Halfspace(0, 0, 1, 0)


@dataclass(frozen=True)
class FixedBox:
    half_extents: tuple[float, float, float]
    color: tuple[float, ...] = (0.8, 0.8, 0.8, 1.0)
    contact_types: tuple[int, ...] = (1,)
    # Additional admitted types, unioned with contact_types on each named face.
    face_contact_types: Mapping[str, tuple[int, ...]] = field(default_factory=dict)

    def __post_init__(self):
        _validate(self, 3, BOX_FACES)

    num_features = 6

    @cached_property
    def features(self):
        return get_box_features(self.half_extents)

    @cached_property
    def coal_shape(self):
        return coal.Box(*(2 * x for x in self.half_extents))


@dataclass(frozen=True)
class MovableBox(FixedBox):
    mass: float = 1.0
    color: tuple[float, ...] = (0.0, 1.0, 0.0, 0.2)

    def __post_init__(self):
        super().__post_init__()
        if not np.isfinite(self.mass) or self.mass <= 0:
            raise ValueError("mass must be positive and finite.")

    @property
    def inertia(self):
        return box_inertia(self.half_extents, self.mass)
