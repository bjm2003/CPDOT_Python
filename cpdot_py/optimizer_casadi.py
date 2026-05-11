"""CasADi + IPOPT backend for the four CPDOT NLP problems.

This module provides a numerically-strong solver path to complement the
``scipy.optimize`` backend in :mod:`optimizer`. The goal is to match the
``IPOPT + ADOL-C`` solver chain that the C++ reference uses in
``lightweight_nlp_problem.cpp``, while reusing the variable layout, bounds,
and result schema already defined on the four NLP classes.

Contract:
- The CasADi expressions in this module are a 1:1 translation of the NumPy
  expressions in ``optimizer.py``. Any logic change must be mirrored in both
  files; the test ``test_casadi_nlp_layout_matches_scipy`` enforces that the
  objective values agree at the initial guess.
- IPOPT options reproduce ``lightweight_nlp_problem.cpp:1969-1972``:
  ``print_level=0``, ``bound_relax_factor=0.0``, ``linear_solver=mumps``,
  ``max_iter=config.opti_inner_iter_max``.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import casadi as ca
import numpy as np

from .states import NVAR, FullStates, TrajectoryPoint

if TYPE_CHECKING:
    from .optimizer import (
        CarLikeNLPProblem,
        CarLikeReplanNLPProblem,
        DiffDriveNLPProblem,
        FormationNLPProblem,
        FormationNLPSolution,
        SingleRobotNLPSolution,
    )


def _ipopt_options(max_iter: int, *, print_level: int = 0) -> dict:
    """Mirror the IPOPT setup at ``lightweight_nlp_problem.cpp:1969-1972``."""
    return {
        "ipopt.print_level": int(print_level),
        "ipopt.max_iter": int(max_iter),
        "ipopt.bound_relax_factor": 0.0,
        "ipopt.linear_solver": "mumps",
        "print_time": False,
    }


def _ipopt_status(solver: ca.Function) -> tuple[bool, str, int]:
    stats = solver.stats()
    return_status = str(stats.get("return_status", "Unknown"))
    success = bool(stats.get("success", return_status == "Solve_Succeeded"))
    iterations = int(stats.get("iter_count", -1))
    return success, return_status, iterations


# ---------------------------------------------------------------------------
# CarLikeNLPProblem (C++ ``LiomIPOPTInterface``)
# ---------------------------------------------------------------------------


def _carlike_objective(problem: "CarLikeNLPProblem", x: ca.MX) -> ca.MX:
    p = problem
    veh = p.config.vehicle
    dt = x[0] / p.nfe
    start = p.profile.start
    goal = p.profile.goal

    def state(var: int, row: int) -> ca.MX:
        return x[p.idx_state(var, row)]

    def vert(var: int, row: int) -> ca.MX:
        return x[p.idx_vertex(var, row)]

    infs = (
        (state(0, 0) - start.x - dt * start.v * float(np.cos(start.theta))) ** 2
        + (state(1, 0) - start.y - dt * start.v * float(np.sin(start.theta))) ** 2
        + (
            state(2, 0)
            - start.theta
            - dt * start.v * float(np.tan(start.phi)) / veh.wheel_base
        )
        ** 2
        + (state(3, 0) - start.v - dt * start.a) ** 2
        + (state(4, 0) - start.phi - dt * start.omega) ** 2
    )

    for row in range(1, p.nrows):
        prev = row - 1
        infs += (
            (state(0, row) - state(0, prev) - dt * state(3, prev) * ca.cos(state(2, prev))) ** 2
            + (state(1, row) - state(1, prev) - dt * state(3, prev) * ca.sin(state(2, prev))) ** 2
            + (
                state(2, row)
                - state(2, prev)
                - dt * state(3, prev) * ca.tan(state(4, prev)) / veh.wheel_base
            )
            ** 2
            + (state(3, row) - state(3, prev) - dt * state(5, prev)) ** 2
            + (state(4, row) - state(4, prev) - dt * state(6, prev)) ** 2
        )

    last = p.nrows - 1
    infs += (
        (goal.x - state(0, last) - dt * state(3, last) * ca.cos(state(2, last))) ** 2
        + (goal.y - state(1, last) - dt * state(3, last) * ca.sin(state(2, last))) ** 2
        + (
            x[p.idx_terminal_theta]
            - state(2, last)
            - dt * state(3, last) * ca.tan(state(4, last)) / veh.wheel_base
        )
        ** 2
        + (goal.v - state(3, last) - dt * state(5, last)) ** 2
        + (goal.phi - state(4, last) - dt * state(6, last)) ** 2
        + (float(np.sin(goal.theta)) - ca.sin(x[p.idx_terminal_theta])) ** 2
        + (float(np.cos(goal.theta)) - ca.cos(x[p.idx_terminal_theta])) ** 2
    )

    # Vertex residuals - mirror CarLikeNLPProblem._active_residual_vertices.
    rear_diag = float(np.hypot(veh.rear_hang_length, veh.width / 2.0))
    rear_angle = float(np.arctan2(veh.width / 2.0, veh.rear_hang_length))
    long_x = 1.2 + 2.0
    long_y = 1.2 + veh.offset
    long_diag = float(np.hypot(long_x, long_y))
    long_angle = float(np.arctan2(long_y, long_x))
    for row in range(p.nrows):
        xx = state(0, row)
        yy = state(1, row)
        th = state(2, row)
        rear_x = xx - rear_diag * ca.cos(th - rear_angle)
        rear_y = yy - rear_diag * ca.sin(th - rear_angle)
        expected = [
            rear_x,
            rear_y,
            rear_x + long_x * ca.cos(th),
            rear_y + long_y * ca.sin(th),
            rear_x + long_diag * ca.cos(th - long_angle),
            rear_y + long_diag * ca.sin(th - long_angle),
            rear_x + long_y * ca.cos(np.pi / 2.0 - th),
            rear_y - long_y * ca.sin(np.pi / 2.0 - th),
        ]
        for var in range(p.vert_nvar):
            infs += (vert(var, row) - expected[var]) ** 2

    obj = p.config.opti_t * x[0]
    for row in range(p.nrows):
        obj += p.config.opti_w_diff_drive * state(5, row) ** 2
        obj += p.config.opti_w_diff_drive * state(6, row) ** 2
    for row in range(1, p.nrows):
        obj += p.config.opti_w_diff_drive * state(3, row) ** 2
    return obj + p.w_inf * infs


# ---------------------------------------------------------------------------
# DiffDriveNLPProblem (C++ ``LiomIPOPTInterface_diff_drive``)
# ---------------------------------------------------------------------------


def _diff_drive_objective(problem: "DiffDriveNLPProblem", x: ca.MX) -> ca.MX:
    p = problem
    dt = x[0] / p.nfe
    first = p.guess.states[0]
    last_state = p.guess.states[-1]

    def state(var: int, row: int) -> ca.MX:
        return x[p.idx_state(var, row)]

    infs = (
        (state(0, 0) - first.x - dt * first.v * float(np.cos(first.theta))) ** 2
        + (state(1, 0) - first.y - dt * first.v * float(np.sin(first.theta))) ** 2
        + (state(2, 0) - first.theta - dt * first.omega) ** 2
        + (state(3, 0) - first.v - dt * first.a) ** 2
        + (state(6, 0) - first.omega - dt * first.phi) ** 2
    )

    for row in range(1, p.nrows):
        prev = row - 1
        infs += (
            (state(0, row) - state(0, prev) - dt * state(3, prev) * ca.cos(state(2, prev))) ** 2
            + (state(1, row) - state(1, prev) - dt * state(3, prev) * ca.sin(state(2, prev))) ** 2
            + (state(2, row) - state(2, prev) - dt * state(6, prev)) ** 2
            + (state(3, row) - state(3, prev) - dt * state(5, prev)) ** 2
            + (state(6, row) - state(6, prev) - dt * state(4, prev)) ** 2
        )

    last = p.nrows - 1
    infs += (
        (last_state.x - state(0, last) - dt * state(3, last) * ca.cos(state(2, last))) ** 2
        + (last_state.y - state(1, last) - dt * state(3, last) * ca.sin(state(2, last))) ** 2
        + (x[p.idx_terminal_theta] - state(2, last) - dt * state(6, last)) ** 2
        + (last_state.v - state(3, last) - dt * state(5, last)) ** 2
        + (last_state.omega - state(6, last) - dt * state(4, last)) ** 2
        + (float(np.sin(last_state.theta)) - ca.sin(x[p.idx_terminal_theta])) ** 2
        + (float(np.cos(last_state.theta)) - ca.cos(x[p.idx_terminal_theta])) ** 2
    )

    obj = x[0]
    for row in range(p.nrows):
        obj += p.config.opti_w_err * (state(0, row) - p.guess.states[row].x) ** 2
        obj += p.config.opti_w_err * (state(1, row) - p.guess.states[row].y) ** 2
    return obj + p.w_inf * infs


# ---------------------------------------------------------------------------
# CarLikeReplanNLPProblem (C++ ``LiomIPOPTInterface_car_like_replan``)
# ---------------------------------------------------------------------------


def _replan_objective(problem: "CarLikeReplanNLPProblem", x: ca.MX) -> ca.MX:
    p = problem
    veh = p.config.vehicle
    dt = x[0] / p.nfe
    first = p.guess.states[0]
    last_state = p.guess.states[-1]

    def state(var: int, row: int) -> ca.MX:
        return x[p.idx_state(var, row)]

    infs = (
        (state(0, 0) - first.x - dt * first.v * float(np.cos(first.theta))) ** 2
        + (state(1, 0) - first.y - dt * first.v * float(np.sin(first.theta))) ** 2
        + (
            state(2, 0)
            - first.theta
            - dt * first.v * float(np.tan(first.phi)) / veh.wheel_base
        )
        ** 2
        + (state(3, 0) - first.v - dt * first.a) ** 2
        + (state(4, 0) - first.phi - dt * first.omega) ** 2
    )

    for row in range(1, p.nrows):
        prev = row - 1
        infs += (
            (state(0, row) - state(0, prev) - dt * state(3, prev) * ca.cos(state(2, prev))) ** 2
            + (state(1, row) - state(1, prev) - dt * state(3, prev) * ca.sin(state(2, prev))) ** 2
            + (
                state(2, row)
                - state(2, prev)
                - dt * state(3, prev) * ca.tan(state(4, prev)) / veh.wheel_base
            )
            ** 2
            + (state(3, row) - state(3, prev) - dt * state(5, prev)) ** 2
            + (state(4, row) - state(4, prev) - dt * state(6, prev)) ** 2
        )

    last = p.nrows - 1
    infs += (
        (last_state.x - state(0, last) - dt * state(3, last) * ca.cos(state(2, last))) ** 2
        + (last_state.y - state(1, last) - dt * state(3, last) * ca.sin(state(2, last))) ** 2
        + (
            x[p.idx_terminal_theta]
            - state(2, last)
            - dt * state(3, last) * ca.tan(state(4, last)) / veh.wheel_base
        )
        ** 2
        + (last_state.v - state(3, last) - dt * state(5, last)) ** 2
        + (last_state.phi - state(4, last) - dt * state(6, last)) ** 2
        + (float(np.sin(last_state.theta)) - ca.sin(x[p.idx_terminal_theta])) ** 2
        + (float(np.cos(last_state.theta)) - ca.cos(x[p.idx_terminal_theta])) ** 2
    )

    obj = x[0]
    for row in range(p.nrows):
        obj += p.config.opti_w_x * (state(0, row) - p.guess.states[row].x) ** 2
        obj += p.config.opti_w_y * (state(1, row) - p.guess.states[row].y) ** 2
    return obj + p.w_inf * infs


# ---------------------------------------------------------------------------
# FormationNLPProblem (C++ ``LiomIPOPTInterfaceFm``)
# ---------------------------------------------------------------------------


def _formation_objective(problem: "FormationNLPProblem", x: ca.MX) -> ca.MX:
    p = problem
    veh = p.config.vehicle
    dt = x[0] / p.nfe

    def state(robot: int, var: int, row: int) -> ca.MX:
        return x[p.idx_state(robot, var, row)]

    def edge(robot: int, row: int) -> ca.MX:
        return x[p.idx_edge_distance(robot, row)]

    def topo(idx: int, row: int) -> ca.MX:
        return x[p.idx_topology(idx, row)]

    def slack(idx: int) -> ca.MX:
        return x[p.idx_sfc(idx)]

    # Aggregated terminal mean error.
    terminal_x = ca.MX(0.0)
    terminal_y = ca.MX(0.0)
    target_x = 0.0
    target_y = 0.0
    for robot in range(p.robot_count):
        terminal_x += state(robot, 0, p.nrows - 1)
        terminal_y += state(robot, 1, p.nrows - 1)
        target_x += p.profile[robot].goal.x
        target_y += p.profile[robot].goal.y
    terminal_mean_error = (
        (terminal_x - target_x) / p.robot_count
    ) ** 2 + ((terminal_y - target_y) / p.robot_count) ** 2

    infs = ca.MX(0.0)
    for robot in range(p.robot_count):
        start = p.profile[robot].start
        goal = p.profile[robot].goal
        infs += (
            (state(robot, 0, 0) - start.x - dt * start.v * float(np.cos(start.theta))) ** 2
            + (state(robot, 1, 0) - start.y - dt * start.v * float(np.sin(start.theta))) ** 2
            + (
                state(robot, 2, 0)
                - start.theta
                - dt * start.v * float(np.tan(start.phi)) / veh.wheel_base
            )
            ** 2
            + (state(robot, 3, 0) - start.v - dt * start.a) ** 2
            + (state(robot, 4, 0) - start.phi - dt * start.omega) ** 2
        )

        last = p.nrows - 1
        infs += (
            (goal.v - state(robot, 3, last)) ** 2
            + (goal.phi - state(robot, 4, last)) ** 2
            + (goal.a - state(robot, 5, last)) ** 2
            + (goal.omega - state(robot, 6, last)) ** 2
        )

        for row in range(1, p.nrows):
            prev = row - 1
            infs += (
                (
                    state(robot, 0, row)
                    - state(robot, 0, prev)
                    - dt * state(robot, 3, prev) * ca.cos(state(robot, 2, prev))
                )
                ** 2
                + (
                    state(robot, 1, row)
                    - state(robot, 1, prev)
                    - dt * state(robot, 3, prev) * ca.sin(state(robot, 2, prev))
                )
                ** 2
                + (
                    state(robot, 2, row)
                    - state(robot, 2, prev)
                    - dt
                    * state(robot, 3, prev)
                    * ca.tan(state(robot, 4, prev))
                    / veh.wheel_base
                )
                ** 2
                + (
                    state(robot, 3, row)
                    - state(robot, 3, prev)
                    - dt * state(robot, 5, prev)
                )
                ** 2
                + (
                    state(robot, 4, row)
                    - state(robot, 4, prev)
                    - dt * state(robot, 6, prev)
                )
                ** 2
            )

    infs += (x[p.idx_terminal_error] - terminal_mean_error) ** 2

    for row in range(p.nrows):
        for robot in range(p.robot_count):
            next_robot = (robot + 1) % p.robot_count
            dx = state(next_robot, 0, row) - state(robot, 0, row)
            dy = state(next_robot, 1, row) - state(robot, 1, row)
            infs += (edge(robot, row) - (dx * dx + dy * dy)) ** 2

    for row in range(p.nrows):
        topo_index = 0
        for robot in range(p.robot_count):
            xk = state(robot, 0, row)
            yk = state(robot, 1, row)
            for q in range(p.robot_count):
                if q == robot or (q + 1) % p.robot_count == robot:
                    continue
                next_q = (q + 1) % p.robot_count
                xp = state(q, 0, row)
                yp = state(q, 1, row)
                xnp = state(next_q, 0, row)
                ynp = state(next_q, 1, row)
                expr = (xk - xp) * (yp - ynp) + (yk - yp) * (xnp - xp)
                infs += (topo(topo_index, row) - expr) ** 2
                topo_index += 1

    slack_index = 0
    for row in range(p.nrows):
        for robot in range(p.robot_count):
            theta = state(robot, 2, row)
            for coeff in p.config.vehicle.disc_coefficients:
                px = state(robot, 0, row) + coeff * ca.cos(theta)
                py = state(robot, 1, row) + coeff * ca.sin(theta)
                for halfspace in p.corridor_cons[robot][row]:
                    expr = halfspace[0] * px + halfspace[1] * py - halfspace[2]
                    infs += (slack(slack_index) - expr) ** 2
                    slack_index += 1

    obj = p.config.opti_t * x[0]
    for robot in range(p.robot_count):
        for row in range(p.nrows):
            obj += (
                p.config.opti_w_a * state(robot, 5, row) ** 2
                + p.config.opti_w_omega * state(robot, 6, row) ** 2
            )
    return obj + p.w_inf * infs


# ---------------------------------------------------------------------------
# Public entry points used by ``optimizer.py`` when ``method == "ipopt"``.
# ---------------------------------------------------------------------------


def _solve_single_robot(
    problem,
    objective_builder,
    *,
    start: TrajectoryPoint,
    goal: TrajectoryPoint,
    maxiter: int,
    print_level: int,
) -> "SingleRobotNLPSolution":
    from .optimizer import SingleRobotNLPSolution

    n = problem.nvar
    x_sym = ca.MX.sym("x", n)
    obj_expr = objective_builder(problem, x_sym)
    nlp = {"x": x_sym, "f": obj_expr}
    solver = ca.nlpsol("solver", "ipopt", nlp, _ipopt_options(maxiter, print_level=print_level))

    lower, upper = problem.bounds()
    x0 = problem.clipped_initial_guess()
    started = time.perf_counter()
    result = solver(x0=x0, lbx=lower, ubx=upper)
    elapsed = time.perf_counter() - started
    success, status, iters = _ipopt_status(solver)

    vector = np.asarray(result["x"]).ravel()
    return SingleRobotNLPSolution(
        state=problem.unpack_solution(vector, start, goal),
        vector=vector,
        objective=problem.eval_objective(vector),
        infeasibility=problem.eval_infeasibility(vector),
        solve_time=elapsed,
        scipy_success=success,
        scipy_message=status,
        iterations=iters,
    )


def solve_car_like_ipopt(
    problem: "CarLikeNLPProblem",
    *,
    start: TrajectoryPoint,
    goal: TrajectoryPoint,
    maxiter: int = 100,
    print_level: int = 0,
) -> "SingleRobotNLPSolution":
    return _solve_single_robot(
        problem,
        _carlike_objective,
        start=start,
        goal=goal,
        maxiter=maxiter,
        print_level=print_level,
    )


def solve_diff_drive_ipopt(
    problem: "DiffDriveNLPProblem",
    *,
    start: TrajectoryPoint,
    goal: TrajectoryPoint,
    maxiter: int = 100,
    print_level: int = 0,
) -> "SingleRobotNLPSolution":
    return _solve_single_robot(
        problem,
        _diff_drive_objective,
        start=start,
        goal=goal,
        maxiter=maxiter,
        print_level=print_level,
    )


def solve_replan_ipopt(
    problem: "CarLikeReplanNLPProblem",
    *,
    start: TrajectoryPoint,
    goal: TrajectoryPoint,
    maxiter: int = 100,
    print_level: int = 0,
) -> "SingleRobotNLPSolution":
    return _solve_single_robot(
        problem,
        _replan_objective,
        start=start,
        goal=goal,
        maxiter=maxiter,
        print_level=print_level,
    )


def solve_formation_ipopt(
    problem: "FormationNLPProblem",
    *,
    maxiter: int = 100,
    print_level: int = 0,
) -> "FormationNLPSolution":
    from .optimizer import FormationNLPSolution

    n = problem.nvar
    x_sym = ca.MX.sym("x", n)
    obj_expr = _formation_objective(problem, x_sym)
    nlp = {"x": x_sym, "f": obj_expr}
    solver = ca.nlpsol("solver", "ipopt", nlp, _ipopt_options(maxiter, print_level=print_level))

    lower, upper = problem.bounds()
    x0 = problem.clipped_initial_guess()
    started = time.perf_counter()
    result = solver(x0=x0, lbx=lower, ubx=upper)
    elapsed = time.perf_counter() - started
    success, status, iters = _ipopt_status(solver)

    vector = np.asarray(result["x"]).ravel()
    return FormationNLPSolution(
        states=problem.unpack_solution(vector),
        vector=vector,
        objective=problem.eval_objective(vector),
        infeasibility=problem.eval_infeasibility(vector),
        infeasibility_terms=problem.eval_infeasibility_terms(vector),
        solve_time=elapsed,
        scipy_success=success,
        scipy_message=status,
        iterations=iters,
    )
