# CPDOT Python Reproduction

This is a standalone Python reproduction of the core algorithmic ideas in the
CPDOT C++/ROS codebase. It is intentionally not a full ROS/Gazebo rewrite.

Implemented pieces:

- 2D obstacle map with point, segment, and polygon collision checks.
- Topological PRM using the CPDOT guard/connector roadmap idea from
  `formation_planner/topo_prm.cpp`.
- Homotopy-style path pruning by checking visibility between corresponding
  samples of two paths.
- Multi-robot formation rollout along a selected guide path.
- Simplified nonlinear trajectory optimization with obstacle, smoothness, and
  formation-shape penalties.
- Flexible sheet forward kinematics ported from
  `formation_planner/forward_kinematics.cpp`.
- Static visualization and optional animation.

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
python -m pytest tests
```

Then run the demo:

```bash
python main.py
```

The demo should print metrics including a nonzero `topo_path_count` and
`robot_collision_count: 0.0000`, and it writes:

```text
outputs/cpdot_result.png
```

Optional:

```bash
python main.py --show
python main.py --animate
```

The implementation keeps the CPDOT structure but simplifies the final Ipopt
optimal-control problem into a SciPy penalty optimization. This keeps the demo
portable while preserving the useful behavior: several topology candidates,
formation-preserving paths, obstacle penalties, and sheet-height evaluation.
