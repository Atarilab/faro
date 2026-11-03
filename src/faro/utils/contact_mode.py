"""Contact-mode unions with one geometry per source interface."""
from typing import Mapping

from faro.configs.identifiers import InterfaceKey
from faro.utils.contact_type import FREE, TupleEE, TupleObj, require_supported_contact_type


ContactMode = Mapping[InterfaceKey, TupleEE | TupleObj]


def contact_mode_union(left: ContactMode, right: ContactMode) -> dict[InterfaceKey, TupleEE | TupleObj]:
    """Combine per-interface requirements; FREE adds nothing.

    Different active geometries on the same source interface are unsupported.
    Sticking/sliding changes retain the left contact's shared geometry.
    Neither input is modified; missing interfaces are FREE.
    """
    result = {}
    for key in dict.fromkeys((*left, *right)):
        a, b = left.get(key, FREE), right.get(key, FREE)
        for contact in (a, b):
            if contact:
                require_supported_contact_type(contact[-1], context=str(key))
        if a and b and a[:-1] != b[:-1]:
            raise ValueError(f"Contact union for {key!r} has conflicting active geometries; "
                             "release through FREE before changing geometry.")
        result[key] = a or b
    return result
