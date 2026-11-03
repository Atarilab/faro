"""Contact relationships shared by node and transition rules."""
from faro.graph.ct_types import FREE
from faro.utils.contact_validation import contact_pairs


class ContactRelations:
    def __init__(self, scene, ms):
        self.scene = scene
        self.ms = ms
        self.object_indices = {name: ms.key_to_i['obj', name] for name in scene.env_objects
                               if ('obj', name) in ms.key_to_i}
        self.ee_indices = [i for i, key in enumerate(ms.keys) if key[0] == 'ee']

    def value(self, node, i):
        return self.ms.domains[i][node[i]]

    def object_contacts(self, node):
        contacts = set()
        for name, i in self.object_indices.items():
            value = self.value(node, i)
            if value != FREE:
                target = self.scene.features.owner(int(value[1]))
                if target[0] == 'obj' and target[1] in self.scene.env_objects and target[1] != name:
                    contacts.add((name, target[1]))
        return contacts

    def target_is_grounded(self, node, env_id, visiting=frozenset()):
        owner = self.scene.features.owner(int(env_id))
        if self.scene.include_robot_ees:
            return self.body_is_grounded(node, self.body(owner))
        name = owner[1]
        if name in self.scene.fixed_env_objects:
            return True
        return self.object_is_grounded(node, name, visiting)

    def object_is_grounded(self, node, name, visiting=frozenset()):
        if self.scene.include_robot_ees:
            return self.body_is_grounded(node, ('obj', name))
        if name in visiting or name not in self.object_indices:
            return False
        value = self.value(node, self.object_indices[name])
        return value != FREE and self.target_is_grounded(node, value[1], visiting | {name})

    def mode(self, node):
        return {key: self.value(node, i) for i, key in enumerate(self.ms.keys)}

    def pairs(self, node):
        return contact_pairs(self.scene, self.mode(node))

    @staticmethod
    def body(owner):
        return ('robot', owner[1]) if owner[0] == 'ee' else owner

    def body_is_grounded(self, node, body):
        return body in self.grounded_bodies(node)

    def grounded_bodies(self, node):
        """Bodies connected to fixed environment, found in one graph traversal."""
        adjacency = {}
        for source, value in self.mode(node).items():
            if not value:
                continue
            a = self.body(source)
            b = self.body(self.scene.features.owner(int(value[-2])))
            adjacency.setdefault(a, set()).add(b)
            adjacency.setdefault(b, set()).add(a)
        visited = set()
        pending = [('obj', name) for name in self.scene.fixed_env_objects]
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            pending.extend(adjacency.get(current, set()) - visited)
        return visited

    def object_touched_by_robot(self, node, name):
        for source, value in self.mode(node).items():
            if not value:
                continue
            target = self.scene.features.owner(int(value[-2]))
            if ((source[0] == 'ee' and target == ('obj', name)) or
                    (source == ('obj', name) and target[0] == 'ee')):
                return True
        return False

    def ee_targets(self, node, robot_id=None):
        indices = self.ee_indices if robot_id is None else [
            i for i in self.ee_indices if self.ms.keys[i][1] == robot_id]
        return [int(value[0]) for i in indices
                if (value := self.value(node, i)) != FREE]


def has_reciprocal_contacts(contacts):
    return any((b, a) in contacts for a, b in contacts)
