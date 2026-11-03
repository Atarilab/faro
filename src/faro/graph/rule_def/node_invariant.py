from faro.graph.rule_def._contacts import ContactRelations, has_reciprocal_contacts


def inv_no_reciprocal_object_contacts(scene, ms):
    """Reject reciprocal object contacts, and EE contacts when enabled."""
    contacts = ContactRelations(scene, ms)
    return lambda _ms, node: not has_reciprocal_contacts((contacts.pairs(node) if scene.include_robot_ees else contacts.object_contacts(node)))


def inv_robot_is_grounded(scene, ms):
    """Every robot must have a support chain ending at fixed environment.

    FREE supports and cycles do not ground the robot.
    """
    contacts = ContactRelations(scene, ms)
    if scene.include_robot_ees:
        robot_bodies = {('robot', rid) for rid in scene.robots}
        return lambda _ms, node: robot_bodies <= contacts.grounded_bodies(node)
    return lambda _ms, node: all(
        any(contacts.target_is_grounded(node, target)
            for target in contacts.ee_targets(node, robot_id))
        for robot_id in scene.robots)
