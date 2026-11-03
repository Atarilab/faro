"""Shared indexing and geometry operations for environment contact features."""

from collections.abc import Mapping

import casadi

from faro.configs.objects import Plane
from faro.ocp.kinematics import obj_state_to_face_q, obj_state_to_face_qv
from faro.utils.geom_utils import box_features_func, plane_features_func


class ObjectFeatures:
    """Keep feature ownership, local geometry, and world transforms separate."""

    def __init__(self, scene):
        self.scene = scene
        self.configs = scene.objects
        self.object_names = scene.env_obj_order
        self.object_indices = {name: i for i, name in enumerate(self.object_names)}
        self.feature_slices = scene.faces.slices
        self.total_features = scene.total_env_features
        self.feature_to_object = casadi.DM.zeros(len(self.object_names), self.total_features)
        for i, name in enumerate(self.object_names):
            self.feature_to_object[i, self.feature_slices[name]] = 1

        self._box_features = box_features_func()
        self._plane_features = plane_features_func()
        self._face_q = obj_state_to_face_q()
        self._face_qv = obj_state_to_face_qv()

    def local_features(self, sizes_by_name: Mapping[str, object]):
        """Return all local feature columns in global feature-ID order."""
        columns = [
            self._local_features_for_object(name, sizes_by_name[name])
            for name in self.object_names
        ]
        columns.extend(casadi.vertcat(casadi.DM.zeros(3), casadi.vec(casadi.DM.eye(3)),
                                     self.scene.robot_interfaces[key])
                       for key in self.scene.features.ee_ids)
        return casadi.horzcat(*columns) if columns else casadi.DM.zeros(16, 0)

    def world_features(self, sizes, object_states, ee_states, *, velocity=False):
        """Transform each feature before selection; EE frames already are world frames.

        ee_states maps owned EE keys to (R, p[, WORLD spatial velocity]).
        Object velocity states are body-frame [linear, angular] twists.
        """
        local = self.local_features(sizes)
        columns = []
        for feature_id, (owner, _) in self.scene.features.by_id.items():
            feature = local[:, feature_id - 1]
            if owner[0] == 'ee':
                state = ee_states[owner]
                R, p = state[:2]
                V = state[2] if velocity else None
                bounds = feature[12:16]
            elif velocity:
                R, p, bounds, V = self.face_pose_velocity(object_states[owner[1]], feature)
            else:
                R, p, bounds = self.face_pose(object_states[owner[1]], feature)
            parts = [R, p, bounds]
            if velocity:
                parts.append(V)
            columns.append(casadi.vertcat(*parts))
        return casadi.horzcat(*columns) if columns else casadi.DM.zeros(22 if velocity else 16, 0)

    @staticmethod
    def select(world_features, selector):
        value = world_features @ selector
        result = (value[:9], value[9:12], value[12:16])
        return (*result, value[16:22]) if value.numel() == 22 else result

    def local_feature(self, name: str, size, selector):
        """Select one local feature (or a linear combination) for an object."""
        return self._local_features_for_object(name, size) @ selector

    def object_weights(self, selector):
        """Map a global feature selector to its object ownership weights."""
        return self.feature_to_object @ selector

    def object_weight(self, name: str, selector):
        """Return the ownership weight for one object."""
        return self.object_weights(selector)[self.object_indices[name]]

    def stack_states(self, states_by_name: Mapping[str, object]):
        """Stack object states as columns in global object order."""
        return casadi.horzcat(*(states_by_name[name] for name in self.object_names))

    def face_pose(self, q, local_feature, *, flip: bool = False):
        """Transform local feature geometry by an object pose."""
        rotation, position = self._face_q(
            q[:7], local_feature[:3], local_feature[3:12]
        )
        bounds = local_feature[12:16]
        return self._apply_flip(rotation, position, bounds, flip)

    def face_pose_velocity(self, x, local_feature, *, flip: bool = False):
        """Transform local geometry by an object pose and spatial velocity."""
        rotation, position, velocity = self._face_qv(
            x[:7], x[7:13], local_feature[:3], local_feature[3:12]
        )
        bounds = local_feature[12:16]
        rotation, position, bounds = self._apply_flip(
            rotation, position, bounds, flip
        )
        return rotation, position, bounds, velocity

    def _local_features_for_object(self, name: str, size):
        if isinstance(self.configs[name], Plane):
            return self._plane_features(size)
        return self._box_features(size)

    @staticmethod
    def _apply_flip(rotation, position, bounds, flip: bool):
        if not flip:
            return rotation, position, bounds

        flip_rotation = casadi.diag(casadi.DM([1, -1, -1]))
        rotation = casadi.reshape(rotation, 3, 3) @ flip_rotation
        bounds = bounds[[0, 3, 2, 1]]
        return casadi.vec(rotation), position, bounds
