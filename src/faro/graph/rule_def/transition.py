from faro.graph.ct_types import FREE
from faro.graph.rule_def._contacts import ContactRelations, has_reciprocal_contacts


def tr_no_reciprocal_object_contacts_across_transition(scene, ms):
    """Reciprocal swaps must pass through a mode with neither direction active."""
    contacts = ContactRelations(scene, ms)
    return lambda _ms, a, b: not has_reciprocal_contacts(
        (contacts.pairs(a) | contacts.pairs(b)) if scene.include_robot_ees
        else (contacts.object_contacts(a) | contacts.object_contacts(b)))


def tr_grounded_object_cannot_release_without_robot_contact(scene, ms):
    """A grounded object may release only if a robot EE touches it in the child.

    EE-target scenes also follow robot support chains. Simultaneous touch/release
    is valid, with robot contact authored in either direction.
    """
    contacts = ContactRelations(scene, ms)

    def rule(_ms, parent, child):
        for name, i in contacts.object_indices.items():
            releasing = contacts.value(parent, i) != FREE and contacts.value(child, i) == FREE
            if releasing and contacts.object_is_grounded(parent, name):
                touched = contacts.object_touched_by_robot(child, name)
                if not touched:
                    return False
        return True

    return rule
