# CPDOT Python Reproduction

This is a standalone Python reproduction of selected core algorithmic ideas in
the CPDOT C++/ROS codebase. It is intentionally not a full ROS/Gazebo rewrite,
and it is still not an exact reproduction of every ROS, IRIS, IPOPT, and
visualization component in the paper code.

Implemented pieces:

- 2D obstacle map with point, segment, and polygon collision checks.
- Topological PRM using the CPDOT guard/connector roadmap idea from
  `formation_planner/topo_prm.cpp`.
- Homotopy-style path pruning by checking visibility between corresponding
  samples of two paths.
- Multi-robot formation rollout along selected guide paths, with per-robot PRM
  seed paths used as a practical fallback for the standalone demo.
- Simplified trajectory smoothing with obstacle, segment-safety, smoothness, and
  formation-shape penalties.
- Flexible sheet forward kinematics ported from
  `formation_planner/forward_kinematics.cpp`.
- C++-style path resampling/time profiling and a SciPy-backed `SolveFm`
  counterpart for the joint formation penalty NLP.
- Hybrid A* coarse path planning from `CoarsePathPlanner`, including 3D grid
  nodes, 2D DP heuristic, kinematic vehicle expansion, disc-box collision
  checking, costs, and homotopy half-space filtering.
- DecompROS-style safe flight corridor generation from
  `Environment::generateSFC`.
- `Plan_fm` core warm-start loop around joint `SolveFm`, operating on Python
  `FullStates` guesses.
- Static visualization and optional animation for the fast demo path.

## Fidelity status

The current code has two tracks: source-aligned algorithm primitives and a fast
standalone demo. The source-aligned pieces are the ones to use when comparing
against CPDOT core algorithms.

Aligned with the C++ code:

- The topological PRM keeps the guard/connector graph structure, visibility
  tests, shortcutting, and homotopy-style pruning from
  `formation_planner/topo_prm.cpp`.
- The flexible sheet forward kinematics follows the C++ taut-subset
  enumeration, rank checks, KKT solve, feasibility checks, and force-closure
  test from `formation_planner/forward_kinematics.cpp`.
- The standalone demo now defaults to five robots, matching the flexible
  formation demo paths and `num_robot_ = 5` setting used by the C++ source.
- `TrajectoryPoint` and `FullStates` mirror the C++ optimizer state containers
  and provide the base for porting `SolveFm`.
- `FormationNLPProblem` now mirrors the C++ formation IPOPT variable layout,
  initial vector packing, objective, infeasibility residuals, and variable
  bounds.
- `solve_fm` solves the same bound-constrained penalty objective used by C++
  `LightweightProblem::SolveFm`, then unpacks results using the C++
  `ConvertVectorToJointStates` layout.
- `resample_path_to_full_states` ports C++ `FormationPlanner::ResamplePath`,
  including time profile, velocity, steering, acceleration, and steering-rate
  fields.
- `CoarsePathPlanner` ports the C++ Hybrid A* body: `Node3d`/`Node2d`
  discretization, 8-neighbor 2D heuristic, steering primitive expansion,
  gear/steering penalties, vehicle disc collision checks, and
  `CheckHomotopyConstraints`.
- `FormationPlanner.plan_coarse_full_states` reproduces the C++ coarse-guess
  block before `Plan_fm`: per-robot coarse planning, path resampling, selecting
  the maximum-`tf` trajectory, and aligning all robot trajectories to the same
  `tf`/`nfe`.
- `generate_sfc` ports the DecompROS 2D ellipsoid corridor path used by
  `Environment::generateSFC`: obstacle edge interpolation, local bounding boxes,
  ellipsoid shrinkage around obstacle points, closest separating hyperplanes,
  and `[a_x, a_y, b]` half-space output.
- `FormationPlanner.plan_fm_from_guess` reproduces the core `Plan_fm` sequence
  after coarse path generation: per-robot SFC construction, height-constraint
  extraction, `GenerateDesiredRP` radius updates, repeated `SolveFm` calls, and
  infeasibility/radius checks.
- Formation similarity uses the same ring-adjacency normalized Laplacian shape
  metric used by the C++ reporting code.

Current mismatches found by source review:

- The Python `solve_fm` uses SciPy finite-difference optimization, not
  IPOPT/ADOL-C sparse derivatives. The objective, bounds, packing, and residuals
  match the C++ structure, but convergence behavior is not identical.
- The C++ `Plan_fm` function currently has a source-level early
  `return false` at `warm_start == 5`, making its later height-tightening branch
  unreachable as written. Python exposes this through
  `enforce_cpp_early_return=True`, but defaults to running the intended
  refinement branch.
- The C++ `SolveFm` safe-corridor residual appears to use `cos(theta)` in the
  first disc's y-coordinate expression; Python keeps the algorithmically
  consistent `GetDiscPositions` formula used elsewhere in the C++ source.
- The C++ coarse planner's OMPL Dubins/Reeds-Shepp one-shot connector is not
  available in the current Python environment. Python does not substitute a
  different connector; it runs the same Hybrid A* expansion and leaves one-shot
  disabled unless a compatible connector is supplied later.
- The standalone demo still defaults to the fast XY smoother for portability;
  the source-aligned `plan_fm_from_guess` path is available separately and is
  covered by tests.
- The command-line staged interface, YAML scene loading, metrics JSON output,
  and unicycle/diff-drive simulation from the implementation guide are not yet
  complete.

Near-term implementation plan:

1. Add or vendor a Python-compatible OMPL Dubins/Reeds-Shepp connector for the
   C++ one-shot path, without replacing it with a different shortcut algorithm.
2. Wire `plan_coarse_full_states -> plan_fm_from_guess` into a source-aligned CLI
   path separate from the fast demo.
3. Add fixture-based comparisons against selected C++ YAML trajectories and
   corridor half-spaces.
4. Decide whether to keep documenting the two C++ source-level issues above or
   preserve them behind strict compatibility flags only.

## Environment

Create and activate the isolated Conda environment:

```bash
cd /home/lyq/CPDOT/cpdot_python
conda env create -f environment.yml
conda activate cpdot-py
```

If the environment already exists:

```bash
conda env update -f environment.yml --prune
conda activate cpdot-py
```

The pinned package versions are also mirrored in `requirements.txt` for pip-only
setups, but Conda is the recommended path here.

## Acceptance

Run the tests first:

```bash
conda run -n cpdot-py python -m pytest tests
```

Then run the demo:

```bash
conda run -n cpdot-py python main.py
```

The demo should print metrics including a nonzero `topo_path_count` and
`robot_collision_count: 0.0000`. By default, each run uses a new randomized
obstacle seed and prints `scene_seed`; pass `--scene-seed N` to reproduce a
specific scene exactly. It never overwrites previous figures: the first run
writes

```text
outputs/cpdot_result.png
```

and later runs write paths such as:

```text
outputs/cpdot_result_001.png
outputs/cpdot_result_002.png
```

Optional:

```bash
conda run -n cpdot-py python main.py --show
conda run -n cpdot-py python main.py --animate
conda run -n cpdot-py python main.py --scene-seed 42
```

The default demo keeps the CPDOT structure but still uses a lightweight Python
smoother for runtime. The source-level NLP reproduction is available through
`cpdot_py.generate_sfc`, `cpdot_py.solve_fm`,
`cpdot_py.FormationNLPProblem`, and
`cpdot_py.FormationPlanner.plan_fm_from_guess`.
