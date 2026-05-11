"""Verify the CasADi/IPOPT backend converges on simple problems.

These tests intentionally use small finite-element counts (and short
horizons) so they run in seconds. Their purpose is to detect regressions in
the CasADi-IPOPT plumbing - that the solver returns a feasible solution and
that the result schema matches the scipy backend - rather than to enforce
specific cost values.

For the long-form C++ baseline diff (M2 in the iteration plan) see
``tests/test_cpp_baseline_diff.py``.
"""

from __future__ import annotations

import numpy as np

from cpdot_py import (
    FormationPlanner,
    Map2D,
    PlannerConfig,
    full_states_to_xy_tensor,
    solve_fm,
)
from cpdot_py.optimizer import (
    CarLikeNLPProblem,
    DiffDriveNLPProblem,
    CarLikeReplanNLPProblem,
)
from cpdot_py.states import Constraints, FullStates, TrajectoryPoint


def test_carlike_nlp_ipopt_returns_feasible_solution_on_static_guess():
    states = [TrajectoryPoint(0.0, 0.0, 0.0)] + [
        TrajectoryPoint(float(i) * 0.5, 0.0, 0.0) for i in range(1, 5)
    ]
    guess = FullStates(tf=2.0, states=states)
    profile = Constraints(states[0], states[-1])
    problem = CarLikeNLPProblem(profile, guess, w_inf=1e3)
    sol = problem.solve(start=states[0], goal=states[-1], method="ipopt", maxiter=50)
    assert isinstance(sol.vector, np.ndarray)
    assert sol.vector.shape == (problem.nvar,)
    assert sol.iterations >= 0
    assert np.isfinite(sol.objective)
    assert np.isfinite(sol.infeasibility)


def test_diff_drive_nlp_ipopt_keeps_fixed_time_bound():
    states = [TrajectoryPoint(0.0, 0.0, 0.0)] + [
        TrajectoryPoint(float(i) * 0.4, 0.0, 0.0) for i in range(1, 5)
    ]
    guess = FullStates(tf=1.5, states=states)
    profile = Constraints(states[0], states[-1])
    problem = DiffDriveNLPProblem(profile, guess, w_inf=1e3)
    sol = problem.solve(start=states[0], goal=states[-1], method="ipopt", maxiter=50)
    assert abs(sol.vector[0] - 1.5) < 1e-6
    assert np.isfinite(sol.objective)


def test_replan_nlp_ipopt_keeps_fixed_time_bound():
    states = [TrajectoryPoint(0.0, 0.0, 0.0)] + [
        TrajectoryPoint(float(i) * 0.4, 0.0, 0.0) for i in range(1, 5)
    ]
    guess = FullStates(tf=1.5, states=states)
    profile = Constraints(states[0], states[-1])
    problem = CarLikeReplanNLPProblem(profile, guess, w_inf=1e3)
    sol = problem.solve(start=states[0], goal=states[-1], method="ipopt", maxiter=50)
    assert abs(sol.vector[0] - 1.5) < 1e-6
    assert np.isfinite(sol.objective)


def test_formation_nlp_ipopt_returns_full_solution_schema():
    scene = Map2D(20, 14, [], (5, 7), (13, 7))
    formation = FormationPlanner(scene, robot_count=5)
    starts, goals = (
        [TrajectoryPoint(*scene.start + offset) for offset in formation.desired_offsets],
        [TrajectoryPoint(*scene.goal + offset) for offset in formation.desired_offsets],
    )
    config = PlannerConfig(min_nfe=4, xy_resolution=0.5, grid_xy_resolution=1.0, step_size=0.2)
    guess = formation.plan_coarse_full_states(
        starts,
        goals,
        hyperparam_sets=None,
        config=config,
        max_search_time=3.0,
        max_expansions=15000,
    )
    assert len(guess) == 5
    profile = [Constraints(start=starts[i], goal=goals[i]) for i in range(5)]
    sol = solve_fm(
        profile,
        guess,
        config=config,
        method="ipopt",
        maxiter=20,
    )
    assert sol.vector.ndim == 1
    assert len(sol.states) == 5
    assert isinstance(sol.infeasibility_terms, dict)
    assert sol.iterations >= 0
    assert np.isfinite(sol.objective)
    trajectory = full_states_to_xy_tensor(sol.states)
    assert trajectory.shape[1] == 5
    assert trajectory.shape[2] == 2


def test_plan_fm_from_guess_with_ipopt_returns_completed_loop():
    scene = Map2D(20, 14, [], (5, 7), (13, 7))
    formation = FormationPlanner(scene, robot_count=5)
    starts, goals = (
        [TrajectoryPoint(*scene.start + offset) for offset in formation.desired_offsets],
        [TrajectoryPoint(*scene.goal + offset) for offset in formation.desired_offsets],
    )
    config = PlannerConfig(min_nfe=4, xy_resolution=0.5, grid_xy_resolution=1.0, step_size=0.2)
    guess = formation.plan_coarse_full_states(
        starts,
        goals,
        hyperparam_sets=None,
        config=config,
        max_search_time=3.0,
        max_expansions=15000,
    )
    result = formation.plan_fm_from_guess(
        guess,
        config=config,
        max_warm_start=1,
        initial_warm_starts=1,
        solver_maxiter=20,
        solver_method="ipopt",
    )
    assert len(result.states) == 5
    assert len(result.solve_history) == 1
    assert set(result.solve_history[0].infeasibility_terms) == {
        "initial_terminal",
        "dynamics",
        "terminal_error",
        "edge_distance",
        "topology",
        "sfc",
    }
