from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import coal

@dataclass(frozen=True)
class CapsuleSpec:
    frame: str
    parent_frame: str
    axis: np.ndarray         # axis in local frame (3,)
    length: float
    radius: float

    def coal_shape(self) -> coal.Capsule:
        return coal.Capsule(float(self.radius), float(self.length))


@dataclass(frozen=True)
class RobotCollisionModel:
    """
    Robot-only collision model (independent of scene and horizon):
      - capsule specs
      - self collision active pairs (capsule name pairs)
      - which capsules collide against env (static vs movable)
    """
    capsules: Dict[str, CapsuleSpec]
    self_collision_pairs: List[Tuple[str, str]]

    @staticmethod
    def build_unique_pairs(self_collision_map: Dict[str, List[str]]) -> List[Tuple[str, str]]:
        uniq = set()
        for a, bs in self_collision_map.items():
            for b in bs:
                if a == b:
                    continue
                uniq.add(tuple(sorted((a, b))))
        return sorted(uniq)
