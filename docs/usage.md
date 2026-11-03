# Usage and architecture

FARO optimizes motions for an explicit sequence of contacts between robots,
movable objects, and the environment. Search, human input, or an external
generator supplies that sequence; the continuous solvers optimize its geometry,
forces, and timing. This is the authoritative overview for contributors and
coding agents. Keep it aligned with changes to the contracts below.

Follow [installation](../install.md), then run from the repository root:

```bash
conda activate faro_env
python examples/pushing/box_push.py
```

The [pushing example](../examples/pushing/box_push.py) shows the trajectory
optimization (TO) lifecycle. [Inverse kinematics (IK)](../scripts/main_ik.py),
[kinematic sequence optimization (KSO)](../scripts/main_kso.py), and
[two-box ascent](../evaluations/human_input/climb_2/ctrl_ascent.py) are useful
starting points for individual stages and longer schedules. Batch experiments
are described in the [evaluation guide](../evaluations/README.md).

| Code | Responsibility |
| --- | --- |
| [`src/scenes`](../src/scenes), [`configs`](../src/faro/configs) | Scene presets, initial placements, robot models, contact interfaces, and geometry. |
| [`tasks`](../src/faro/tasks) | Public solve lifecycle: initialization, parameters, collision updates, results, and diagnostics. |
| [`ocp/formulation`](../src/faro/ocp/formulation), [`ocp`](../src/faro/ocp) | IK/KSO/TO equations, contact constraints, dynamics, and coordinate layouts. |
| [`solvers`](../src/faro/solvers), [`cost`](../src/faro/cost) | Native solver adapters and formulation-specific regularization weights. IK/KSO use Acados; TO uses Moto. |
| [`graph`](../src/faro/graph), [`search`](../src/faro/search), [`evaluators`](../src/faro/evaluators) | Symbolic contact choices and rules, path search, and adapters to feasibility solves. |
| [`collision`](../src/faro/collision) | Collision pair selection and Coal distance queries used by the optimization. |
| [`kinevis`](../src/kinevis), [`configs/mujoco.py`](../src/faro/configs/mujoco.py) | MuJoCo model assembly and trajectory playback. Playback is kinematic, not a dynamics validation. |

Load a preset with `scenes.common.registry.make_scene_with_state(scene_id,
"default")`, which returns a `Scene` and `ResolvedSceneState`. Scene modules
separate geometry (`scene.py`), placements (`initial_conditions.py`), and goals
(`goals.py`); search presets also supply `domains.py` and `experiments.py`.
Configure robots and collision rules before constructing tasks: formulations
and indices capture their structure.

For TO, use `build_solver(cfg)` → `initialize_solver(cms)` →
`update_mode_oh_via_tup(cms)` → `set_fixed_params(weights=...)` → `solve()` →
`get_results(coll=True)`. Generate weights with that stage's
`make_ocp_weights(scene.robots)`. Changing FREE phases or contact types requires
calling `initialize_solver` again because they determine which variables and
constraints exist. Changing model structure or horizon requires rebuilding.
First use can spend several minutes compiling native kernels; subsequent runs
can reuse generated artifacts.

The following are architectural contracts. Changes to them require coordinated
updates to formulations, adapters, search, and result consumers.

1. **Explicit ownership and stable IDs.** Robot interfaces are
   `('ee', robot_id, local_patch_name)`; movable-object interfaces are
   `('obj', object_name)`. Keep ownership explicit even for one robot; use
   [identifier helpers](../src/faro/configs/identifiers.py) for solver symbols.
   Global target IDs are one-based, with fixed objects before movable objects.
   Box-local faces are `top, left, front, right, back, bottom` (IDs 1–6).
   Prefer `scene.face_id(name, face)` when authoring: reordering objects can
   invalidate saved numeric schedules. With `include_robot_ees=True`, EE targets
   are appended and available through `scene.ee_feature_id(robot_id, patch)`.

2. **One outgoing contact per interface.** A mode maps each interface to `()`
   (FREE), `(target_id, type)` for an EE, or
   `(self_face_id, target_id, type)` for an object. A sequence maps interfaces to
   equal-length lists of modes; TO requires every `scene.active_patches` key.
   Type `1` is sticking unilateral contact, `2` is sliding, and `3` is reserved
   and unsupported. Changing active geometry must pass through FREE for the
   transition-union checks; changing type on the same geometry can be direct.
   Represent each physical contact once, with reactions routed to its target.
   A FREE interface can still receive incoming contacts from other sources.
   Reciprocal contacts and direct direction reversals are rejected when EE
   targets are enabled. See [contact encoding](../src/faro/utils/contact_type.py)
   and [mode unions](../src/faro/utils/contact_mode.py).

3. **Supported geometry and robot topology.** Scene objects are `Plane`,
   `FixedBox`, or `MovableBox`; dimensions are half-extents in metres. A plane's
   contact patch is bounded, but its collision shape is an infinite halfspace.
   Robots require a leading free-flyer followed by scalar joints, with actuator
   limits keyed by joint name. Derive dimensions from the model; G1 is a preset.
   New object shapes or joint topologies require changes across geometry,
   dynamics, collision, and rendering as well as configuration.

4. **Coordinates and wrench conventions.** Physical configurations use world
   XYZ and a unit XYZW quaternion, followed by robot joints. MuJoCo conversion
   to WXYZ belongs at the visualization boundary. TO stores robot states as
   `[q, v, CoM, centroidal_momentum]` and object states as `[q, v]`; configuration
   increments have dimension `nv`, not `nq`. IK/KSO use base-pose offsets and
   absolute robot joint coordinates internally; warm starts are backend-specific.
   Robot wrenches are world-aligned `[force, moment]` about the EE origin; object
   wrenches are body-local `[moment, force]` about the object origin. Preserve
   these distinctions and use the [wrench routing](../src/faro/ocp/contact_wrenches.py)
   and [state layout](../src/faro/ocp/moto_states.py) helpers.

5. **Different feasibility stages.** IK solves one free configuration; the
   edge oracle solves the union of adjacent contact modes. KSO solves transition
   configurations and enforces contact continuity between them. For `M` modes,
   it has `M - 1` transition states, plus a fixed initial state unless
   `free_init_state=True`, and a terminal state when a KSO target is active.
   Neither stage establishes dynamic feasibility. TO couples robot centroidal
   dynamics, inverse-dynamics actuation limits, and rigid-body object dynamics.
   Preserve this distinction when changing filters or interpreting success.

6. **TO interval placement and time.** There are
   `N = num_modes * nodes_per_mode` intervals, `N + 1` states, and `N` controls.
   Interval `k` connects `x[k]` to `x[k+1]`; its contact parameters belong to `k`,
   while its wrenches and collision constraints use the endpoint. Integration
   is implicit on the configuration manifold. The shared `t_scale[k]` multiplies
   nominal duration; configured robot contact switches shorten an interval by
   `0.1` once, even for simultaneous switches. Use `get_step_ts(results)` and
   `get_ts(results)` for physical time. Endpoint placement, switch indexing,
   and integration must be changed together.

7. **Collision checks are iterative approximations.** Robot capsules, boxes,
   and halfspaces supply nearest points and normals; the solver uses local
   signed-distance approximations refreshed during solving. `CollisionRules`
   and contact-dependent exemptions determine which pairs are constrained.
   Preserve this update loop and inspect collision metrics after a solve;
   sampled feasibility does not establish continuous collision clearance.

8. **Search feasibility and goals have separate meanings.** The symbolic graph
   describes allowed contact transitions; its node/edge oracle cache shares
   geometric checks. Sequence costs and TO results depend on the entire path,
   so they cannot be cached by final mode alone. `DiscreteGoal` matches contacts;
   `TerminalGoal` applies geometric targets as `soft`, `hard`, or `off`
   independently in KSO and TO; `OutcomeScorer` ranks results. Regularization
   weights and ranking scores do not impose terminal constraints. See
   [goal definitions](../src/faro/search/goal.py).

For TO, always request `get_results(coll=True)` for the final collision solve;
the default returns the presolve when one exists. `solve()` returns `0` for
primal feasibility below `feasibility_tol`, `1` for exhausted iterations, and
`2` for solver failure; `0` does not imply full optimality. Check constraint and
collision diagnostics, not only the animation. `max_nlp_iters` counts outer
collision updates, while `collision_update_batch_size` controls inner SQP work.
`get_solve_time()` excludes construction and native initialization but includes
collision updates inside the solve; it is not end-to-end runtime.

Extend scenes, goals, domains, weights, and build configurations through their
existing interfaces. Edit equations in `ocp` and backend behavior in `solvers`;
generated artifacts are disposable. Use the pinned dependencies and native
build procedure in [installation](../install.md), and rebuild after intentional
dependency changes. Update this overview whenever a contract changes.
