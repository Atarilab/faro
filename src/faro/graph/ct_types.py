from __future__ import annotations

from typing import Dict, Hashable, Tuple, Union

# FREE sentinel
FREE: Tuple[()] = ()

# Keys:
#   ("ee", robot_id, patch_name)
#   ("obj", obj_name)
InterfaceKey = Tuple[Hashable, ...]

# Values:
EEContact = Tuple[int, int]          # (env_id, contact_type)
ObjContact = Tuple[int, int, int]    # (self_id, env_id, contact_type)
InterfaceValue = Union[Tuple[()], EEContact, ObjContact]

# Human-friendly representation for nodes (decoded form)
ModeDict = Dict[InterfaceKey, InterfaceValue]


EncodedNode = tuple[int, ...]
EncodedPath = tuple[EncodedNode, ...]
GoalAlternatives = tuple[tuple[tuple[int, int], ...], ...]
