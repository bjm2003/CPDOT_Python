"""C++ baseline diff regression for the CasADi+IPOPT NLP backend.

The CPDOT C++ tree ships warm-start NLP solutions in
``cpp_fixtures/flexible_formation/{N}/traj_{N}R1000.yaml``
(for ``N`` robots, ``R`` robot index). These come from ``Plan_fm`` in
``formation_planner.cpp:852`` after the first refinement iteration. They
include ``x/y/theta/v/phi/a/omega/t`` per state, so they are the closest
thing to a published reference solution that ships with the repo.

This test set checks that:
1. The fixtures load with the schema we expect (sanity).
2. Re-using the C++ first/last states as Python ``Constraints`` and feeding
   the C++ solution back in as a guess, the Python IPOPT backend (a) does
   not blow up, (b) keeps ``tf`` close to the C++ ``tf``, and (c) keeps
   each robot's xy trajectory within a tolerance of the C++ trajectory.

The strict every-step RMSE diff against the original 9942-step controller
``traj_real*.yaml`` is intentionally out of scope: that is a controller
output, not an NLP output, and reproducing it requires the C++
trajectory_tracking stack.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cpdot_py import (
    PlannerConfig,
    cpp_warmstart_endpoints,
    cpp_warmstart_xy_tensor,
    full_states_to_xy_tensor,
    load_cpp_warmstart_set,
    solve_fm,
)
from cpdot_py.states import Constraints

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_ROOT = REPO_ROOT / "cpp_fixtures/flexible_formation"


@pytest.mark.parametrize("num_robots,expected_nfe", [(3, 77), (5, 93)])
def test_cpp_warmstart_fixture_metadata_is_consistent(num_robots, expected_nfe):
    fixture_dir = FIXTURE_ROOT / str(num_robots)
    states = load_cpp_warmstart_set(fixture_dir, num_robots, warm_start=1)
    assert len(states) == num_robots
    nfe_seen = {len(full.states) for full in states}
    assert nfe_seen == {expected_nfe}
    tf_set = {round(full.tf, 6) for full in states}
    assert len(tf_set) == 1, f"per-robot tf disagrees in {fixture_dir}: {tf_set}"
    assert next(iter(tf_set)) > 0


def test_cpp_warmstart_n5_first_frame_is_regular_polygon_around_corner():
    fixture_dir = FIXTURE_ROOT / "5"
    xy = cpp_warmstart_xy_tensor(fixture_dir, 5, warm_start=1)
    first = xy[0]
    centre = first.mean(axis=0)
    radii = np.linalg.norm(first - centre, axis=1)
    np.testing.assert_allclose(radii, radii[0], atol=1e-6)
    expected_radius = 4.05 / np.sqrt(3.0)
    np.testing.assert_allclose(radii[0], expected_radius, atol=5e-3)


@pytest.mark.parametrize("num_robots", [3, 5])
def test_python_ipopt_keeps_xy_close_to_cpp_warmstart_when_using_it_as_guess(num_robots):
    fixture_dir = FIXTURE_ROOT / str(num_robots)
    starts, goals, tf_cpp = cpp_warmstart_endpoints(fixture_dir, num_robots, warm_start=1)
    cpp_full_states = load_cpp_warmstart_set(fixture_dir, num_robots, warm_start=1)
    cpp_xy = cpp_warmstart_xy_tensor(fixture_dir, num_robots, warm_start=1)

    profile = [Constraints(start=starts[i], goal=goals[i]) for i in range(num_robots)]
    config = PlannerConfig()

    sol = solve_fm(
        profile,
        cpp_full_states,
        config=config,
        method="ipopt",
        maxiter=10,
    )
    assert np.isfinite(sol.objective)
    assert np.isfinite(sol.infeasibility)
    py_xy = full_states_to_xy_tensor(sol.states)
    assert py_xy.shape == cpp_xy.shape

    # The Python IPOPT solver re-optimizes the same NFE samples; with only
    # 10 iterations and no corridor/height constraints it should stay close
    # to the C++ guess in xy. We use a generous bound that still catches
    # divergence (e.g. NaN explosion or sign flip).
    per_robot_max_diff = np.linalg.norm(py_xy - cpp_xy, axis=2).max(axis=0)
    assert np.all(per_robot_max_diff < 5.0), (
        f"Python xy diverged > 5m from C++ baseline: {per_robot_max_diff}"
    )

    # Time should not jump catastrophically. The C++ baseline uses tf~103s;
    # Python's bound only enforces ``lower=0.1``. Allow ±50% bracket for the
    # smoke check.
    py_tf = float(sol.vector[0])
    assert 0.5 * tf_cpp <= py_tf <= 1.5 * tf_cpp, (
        f"Python tf={py_tf:.3f} outside [{0.5*tf_cpp:.1f}, {1.5*tf_cpp:.1f}]"
    )
