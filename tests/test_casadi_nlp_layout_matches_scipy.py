"""Verify the CasADi NLP backend reproduces the scipy expressions exactly.

The CasADi expressions in :mod:`cpdot_py.optimizer_casadi` are a 1:1 port of
the NumPy expressions in :mod:`cpdot_py.optimizer`. Drift between the two
would silently change the objective surface that IPOPT sees, so this test
evaluates each CasADi MX objective at the initial guess and asserts equality
with the scipy ``eval_objective`` implementation.
"""

from __future__ import annotations

import casadi as ca
import numpy as np

from cpdot_py.optimizer import (
    CarLikeNLPProblem,
    CarLikeReplanNLPProblem,
    DiffDriveNLPProblem,
    FormationNLPProblem,
)
from cpdot_py.optimizer_casadi import (
    _carlike_objective,
    _diff_drive_objective,
    _formation_objective,
    _replan_objective,
)
from cpdot_py.states import Constraints, FullStates, TrajectoryPoint


def _eval_casadi_objective(builder, problem, vector=None):
    x_sym = ca.MX.sym("x", problem.nvar)
    expr = builder(problem, x_sym)
    func = ca.Function("obj", [x_sym], [expr])
    if vector is None:
        vector = problem.clipped_initial_guess()
    return float(func(vector).full().ravel()[0])


def test_casadi_carlike_objective_matches_scipy_at_initial_guess():
    states = [
        TrajectoryPoint(1.0, 2.0, 0.0),
        TrajectoryPoint(1.0, 2.0, 0.0),
        TrajectoryPoint(1.0, 2.0, 0.0),
    ]
    guess = FullStates(tf=1.0, states=states)
    corridor_lb = np.full((3, 8), -10.0)
    corridor_ub = np.full((3, 8), 10.0)
    profile = Constraints(states[0], states[-1], corridor_lb=corridor_lb, corridor_ub=corridor_ub)
    problem = CarLikeNLPProblem(profile, guess, w_inf=1e3)

    vector = problem.pack_initial_guess()
    source_vertices = problem._active_residual_vertices(1.0, 2.0, 0.0)
    for i, value in enumerate(source_vertices):
        vector[problem.idx_vertex(i, 0)] = value
    scipy_value = problem.eval_objective(vector)

    x_sym = ca.MX.sym("x", problem.nvar)
    expr = _carlike_objective(problem, x_sym)
    casadi_value = float(ca.Function("obj", [x_sym], [expr])(vector).full().ravel()[0])

    assert abs(casadi_value - scipy_value) < 1e-7


def test_casadi_diff_drive_objective_matches_scipy_at_initial_guess():
    states = [
        TrajectoryPoint(1.0, 2.0, 0.0),
        TrajectoryPoint(1.0, 2.0, 0.0),
        TrajectoryPoint(1.0, 2.0, 0.0),
    ]
    guess = FullStates(tf=1.5, states=states)
    profile = Constraints(states[0], states[-1])
    problem = DiffDriveNLPProblem(profile, guess, w_inf=1e3)

    vector = problem.pack_initial_guess()
    scipy_value = problem.eval_objective(vector)
    casadi_value = _eval_casadi_objective(_diff_drive_objective, problem, vector)
    assert abs(casadi_value - scipy_value) < 1e-7


def test_casadi_replan_objective_matches_scipy_at_initial_guess():
    states = [
        TrajectoryPoint(1.0, 2.0, 0.0),
        TrajectoryPoint(1.0, 2.0, 0.0),
        TrajectoryPoint(1.0, 2.0, 0.0),
    ]
    guess = FullStates(tf=1.5, states=states)
    profile = Constraints(states[0], states[-1])
    problem = CarLikeReplanNLPProblem(profile, guess, w_inf=1e3)

    vector = problem.pack_initial_guess()
    scipy_value = problem.eval_objective(vector)
    casadi_value = _eval_casadi_objective(_replan_objective, problem, vector)
    assert abs(casadi_value - scipy_value) < 1e-7


def test_casadi_formation_objective_matches_scipy_at_initial_guess():
    points = np.array(
        [
            [1.0, 0.0],
            [0.30901699, 0.95105652],
            [-0.80901699, 0.58778525],
            [-0.80901699, -0.58778525],
            [0.30901699, -0.95105652],
        ]
    )
    guess: list[FullStates] = []
    profile: list[Constraints] = []
    for px, py in points:
        state = TrajectoryPoint(float(px), float(py))
        full = FullStates(tf=1.0, states=[state, state, state])
        guess.append(full)
        profile.append(Constraints(start=state, goal=state))

    problem = FormationNLPProblem(profile, guess, w_inf=10.0)
    vector = problem.pack_initial_guess()
    scipy_value = problem.eval_objective(vector)
    casadi_value = _eval_casadi_objective(_formation_objective, problem, vector)
    assert abs(casadi_value - scipy_value) < 1e-7


def test_casadi_carlike_objective_perturbed_guess_agrees_within_tolerance():
    """CasADi vs scipy must agree away from the static optimum, not just at it."""
    rng = np.random.default_rng(11)
    states = [
        TrajectoryPoint(0.0, 0.0, 0.0),
        TrajectoryPoint(0.5, 0.0, 0.0),
        TrajectoryPoint(1.0, 0.0, 0.0),
        TrajectoryPoint(1.5, 0.0, 0.0),
    ]
    guess = FullStates(tf=2.0, states=states)
    corridor_lb = np.full((4, 8), -10.0)
    corridor_ub = np.full((4, 8), 10.0)
    profile = Constraints(states[0], states[-1], corridor_lb=corridor_lb, corridor_ub=corridor_ub)
    problem = CarLikeNLPProblem(profile, guess, w_inf=1e3)

    base = problem.pack_initial_guess()
    perturbed = base + rng.normal(0.0, 0.05, size=base.shape)
    perturbed[0] = max(perturbed[0], 0.2)
    scipy_value = problem.eval_objective(perturbed)
    x_sym = ca.MX.sym("x", problem.nvar)
    expr = _carlike_objective(problem, x_sym)
    casadi_value = float(ca.Function("obj", [x_sym], [expr])(perturbed).full().ravel()[0])
    assert abs(casadi_value - scipy_value) / max(1.0, abs(scipy_value)) < 1e-7
