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
- Static visualization and optional animation.

## Fidelity status

The current code should be treated as a runnable simplified reproduction, not a
paper-equivalent implementation.

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
- Formation similarity uses the same ring-adjacency normalized Laplacian shape
  metric used by the C++ reporting code.

Current mismatches found by source review:

- The Python `solve_fm` uses SciPy finite-difference optimization, not
  IPOPT/ADOL-C sparse derivatives. The objective, bounds, packing, and residuals
  match the C++ structure, but convergence behavior is not identical.
- The C++ planner generates safe corridors around each robot path and couples
  all robots in one formation optimization. Python can evaluate corridor
  half-spaces in `FormationNLPProblem`, but it does not yet port
  `Environment::generateSFC`/IRIS corridor construction.
- The C++ `Plan_fm` loop iteratively tightens height-derived inter-robot
  distance constraints with `GenerateHeightCons` and `GenerateDesiredRP`.
  Python has height evaluation and obstacle height constraints, but the full
  warm-start loop is not yet wired into the demo.
- The standalone demo still defaults to the fast XY smoother for portability;
  the new `solve_fm` path is available as a source-code reproduction primitive
  and is covered by tests.
- The command-line staged interface, YAML scene loading, metrics JSON output,
  and unicycle/diff-drive simulation from the implementation guide are not yet
  complete.

Near-term implementation plan:

1. Port safe-flight-corridor generation from `Environment::generateSFC` or add a
   Python half-space corridor builder with the same data shape.
2. Wire `resample_path_to_full_states` into robot seed generation so `SolveFm`
   receives full `(x, y, theta, v, phi, a, omega)` guesses.
3. Implement the C++ `Plan_fm` warm-start loop around `solve_fm`, including
   `GenerateDesiredRP` height tightening.
4. Add a CLI flag to run the full joint NLP path separately from the fast demo.

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
`cpdot_py.solve_fm` and the `FormationNLPProblem` class.
