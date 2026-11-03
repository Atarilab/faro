"""Scene-owned collision pairs, independent of trajectory horizon."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping

from faro.configs.objects import Plane, FixedBox
from faro.configs.identifiers import capsule_name, solver_name
from faro.collision.collision_rules import CollisionRules


@dataclass(frozen=True)
class PairDef:
    name: str
    name_i: str
    name_j: str
    robot_i: str | None = None
    robot_j: str | None = None
    local_i: str | None = None
    local_j: str | None = None


@dataclass(frozen=True)
class CollisionIndex:
    groups: Mapping[str, dict[str, PairDef]]
    pair_names: tuple[str, ...]

    @staticmethod
    def from_scene(scene, collision_rules: CollisionRules | None = None):
        rules = scene.collision_rules if collision_rules is None else collision_rules
        groups = {name: {} for name in (
            'self_coll', 'robot_robot', 'mv_box_mv_box', 'mv_box_st_box',
            'robot_mv_box', 'robot_st_box', 'robot_plane', 'mv_box_plane')}
        ignore = rules.ignore_pairs or set()

        def add(group, a, b, ri=None, rj=None, li=None, lj=None):
            if not getattr(rules, 'enable_' + group):
                return
            if (a, b) in ignore or (b, a) in ignore:
                return
            name = solver_name('collision', group, a, b)
            groups[group][name] = PairDef(name, a, b, ri, rj, li, lj)

        planes = [n for n, c in scene.fixed_env_objects.items() if isinstance(c, Plane)]
        boxes = [n for n, c in scene.fixed_env_objects.items() if isinstance(c, FixedBox)]
        moving = tuple(scene.env_objects)
        for rid, cfg in scene.robots.items():
            for a, b in cfg.collision_model.self_collision_pairs:
                add('self_coll', capsule_name(rid, a), capsule_name(rid, b), rid, rid, a, b)
            for cap in cfg.collision_model.capsules:
                for group, objects in (('robot_mv_box', moving), ('robot_st_box', boxes), ('robot_plane', planes)):
                    for obj in objects:
                        add(group, capsule_name(rid, cap), obj, rid, None, cap, obj)
        for (ri, ci), (rj, cj) in combinations(scene.robots.items(), 2):
            for a in ci.collision_model.capsules:
                for b in cj.collision_model.capsules:
                    add('robot_robot', capsule_name(ri, a), capsule_name(rj, b), ri, rj, a, b)
        for a, b in combinations(moving, 2):
            add('mv_box_mv_box', a, b, li=a, lj=b)
        for a in moving:
            for group, objects in (('mv_box_st_box', boxes), ('mv_box_plane', planes)):
                for b in objects:
                    add(group, a, b, li=a, lj=b)
        # Group order is stable; per-group insertion follows scene/model order.
        return CollisionIndex(groups, tuple(name for pairs in groups.values() for name in pairs))
