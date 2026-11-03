from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Set, Tuple

@dataclass(frozen=True)
class CollisionRules:
    """
    Declarative collision policy. Scene compilation turns this into an index of pairs.
    """
    enable_self_coll: bool = True
    enable_robot_robot: bool = True
    enable_mv_box_mv_box: bool = True
    enable_mv_box_st_box: bool = True
    enable_robot_mv_box: bool = True
    enable_robot_st_box: bool = True
    enable_robot_plane: bool = True
    enable_mv_box_plane: bool = True

    # Optional explicit ignore list (pair names at the *entity* level, not geom IDs).
    # Robot endpoints use capsule_name(robot_id, local_name); objects use their names.
    ignore_pairs: Optional[Set[Tuple[str, str]]] = None
