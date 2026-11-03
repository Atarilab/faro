"""Load explicit presets using scene module IDs, e.g. ``faro.box_pushing``.

IDs are relative to ``scenes``, e.g. ``example_scene`` or ``faro.box_pushing``.
Scene families use dots, just like Python imports.
"""
import importlib


def _load(scene_name, module_name):
    return importlib.import_module(f"scenes.{scene_name}.{module_name}")


def _lookup(scene_name, module_name, mapping_name, name):
    table = getattr(_load(scene_name, module_name), mapping_name)
    try:
        return table[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown {module_name} preset '{name}'. Available: {sorted(table)}"
        ) from exc


def make_scene(scene_name: str):
    return _load(scene_name, "scene").make_scene()


def get_initial_condition(scene_name: str, ic_name: str = "default", *, scene):
    return _lookup(scene_name, "initial_conditions", "INITIAL_CONDITIONS", ic_name)(scene)


def get_domains(scene_name: str, domain_name: str = "default", *, scene):
    return _lookup(scene_name, "domains", "DOMAINS", domain_name)(scene)


def get_discrete_goal(scene_name: str, goal_name: str, *, scene):
    return _lookup(scene_name, "goals", "DISCRETE_GOALS", goal_name)(scene)


def get_terminal_goal(scene_name: str, goal_name: str, *, scene, resolved_state):
    return _lookup(scene_name, "goals", "TERMINAL_GOALS", goal_name)(scene, resolved_state)


def get_outcome_scorer(scene_name: str, scorer_name: str, *, scene, resolved_state):
    return _lookup(scene_name, "goals", "OUTCOME_SCORERS", scorer_name)(scene, resolved_state)


def get_experiment(scene_name: str, experiment_name: str):
    return _lookup(scene_name, "experiments", "EXPERIMENTS", experiment_name)


def list_experiments(scene_name: str) -> list[str]:
    return sorted(_load(scene_name, "experiments").EXPERIMENTS)


def make_scene_with_state(scene_name: str, state_name: str = "default"):
    """Load a scene and resolve a preset; visualization builds MuJoCo separately."""
    scene = make_scene(scene_name)
    spec = get_initial_condition(scene_name, state_name, scene=scene)
    return scene, scene.resolve(spec)
