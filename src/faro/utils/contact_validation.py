"""Physical contact identity shared by authoring, schedules and graph rules."""


def contact_pair(scene, source, value):
    """Directed endpoints; object-object pairs retain body-level identity."""
    target, local_id = scene.features.by_id[value[-2]]
    if source[0] == 'obj' and target[0] == 'obj':
        return source, target
    a = source if source[0] == 'ee' else ('face', source[1], value[0])
    b = target if target[0] == 'ee' else ('face', target[1], local_id)
    return a, b


def contact_pairs(scene, mode):
    return {contact_pair(scene, key, value)
            for key, value in mode.items() if value}


def has_reverse(pairs):
    return any((b, a) in pairs for a, b in pairs)


def validate_contact_target(scene, source, value):
    if not value:
        return
    target = scene.features.owner(value[-2])
    if target == source:
        raise ValueError(f"Contact interface {source!r} cannot target itself.")


def validate_contact_mode(scene, mode):
    for key, value in mode.items():
        validate_contact_target(scene, key, value)
    if scene.include_robot_ees and has_reverse(contact_pairs(scene, mode)):
        raise ValueError("Reciprocal contacts represent the same contact twice.")


def validate_contact_transition(scene, previous, current):
    if scene.include_robot_ees and has_reverse(
            contact_pairs(scene, previous) | contact_pairs(scene, current)):
        raise ValueError("Contact direction reversal must pass through FREE.")
