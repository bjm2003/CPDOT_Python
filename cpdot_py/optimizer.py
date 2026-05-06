"""Joint formation NLP structures ported from CPDOT's IPOPT problem."""

from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np
from scipy.optimize import Bounds, minimize

from .states import NVAR, Constraints, FullStates, TrajectoryPoint


@dataclass
class VehicleModel:
    """Subset of C++ ``VehicleModel`` needed by ``LiomIPOPTInterfaceFm``."""

    offset: float = 3.0
    front_hang_length: float = 0.165
    wheel_base: float = 0.65
    rear_hang_length: float = 0.165
    width: float = 0.605
    max_velocity: float = 1.0
    min_velocity: float = -1.0
    max_acceleration: float = 1.0
    phi_max: float = 0.69
    phi_min: float = 0.69
    omega_max: float = 0.2
    n_disc: int = 2
    disc_radius: float = 0.0
    disc_coefficients: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.initialize_discs()

    def initialize_discs(self) -> None:
        length = self.wheel_base + self.rear_hang_length + self.front_hang_length
        self.disc_radius = 0.5 * float(np.hypot(length / self.n_disc, self.width))
        self.disc_coefficients = [
            (2.0 * (i + 1) - 1.0) / (2.0 * self.n_disc) * length - self.rear_hang_length
            for i in range(self.n_disc)
        ]

    def disc_positions(self, x: float, y: float, theta: float) -> np.ndarray:
        out = []
        for coeff in self.disc_coefficients:
            out.extend([x + coeff * np.cos(theta), y + coeff * np.sin(theta)])
        return np.asarray(out, dtype=float)


@dataclass
class PlannerConfig:
    """Subset of C++ ``PlannerConfig`` used by the formation NLP."""

    xy_resolution: float = 0.5
    theta_resolution: float = 0.1
    step_size: float = 0.2
    next_node_num: int = 6
    grid_xy_resolution: float = 1.0
    forward_penalty: float = 0.5
    backward_penalty: float = 1.0
    gear_change_penalty: float = 5.0
    steering_penalty: float = 0.5
    steering_change_penalty: float = 1.0
    min_nfe: int = 20
    time_step: float = 0.5
    opti_t: float = 1.0
    opti_w_a: float = 1.0
    opti_w_omega: float = 1.0
    opti_w_penalty0: float = 1e4
    opti_varepsilon_tol: float = 1e-4
    vehicle: VehicleModel = field(default_factory=VehicleModel)


@dataclass(frozen=True)
class VVCMConstants:
    """Constants from C++ ``VVCM`` used by the formation NLP bounds."""

    formation_radius: float = 4.05 / np.sqrt(3.0)
    radius_inc: float = 0.2
    xv2t: float = 1.2
    zr: float = 2.2

    @property
    def xv2(self) -> float:
        return float(self.formation_radius * np.sqrt(3.0))


@dataclass
class FormationNLPSolution:
    """Result of the Python ``SolveFm`` counterpart."""

    states: list[FullStates]
    vector: np.ndarray
    objective: float
    infeasibility: float
    solve_time: float
    scipy_success: bool
    scipy_message: str
    iterations: int


class FormationNLPProblem:
    """Python layout/evaluator for C++ ``LiomIPOPTInterfaceFm``.

    This class intentionally mirrors the C++ indexing scheme. It is the
    accuracy-critical bridge before wiring the problem to a numerical optimizer.
    """

    def __init__(
        self,
        profile: list[Constraints],
        guess: list[FullStates],
        config: PlannerConfig | None = None,
        corridor_cons: list[list[list[list[float]]]] | None = None,
        height_cons: list[float] | None = None,
        w_inf: float = 1e4,
    ):
        if not guess:
            raise ValueError("guess must contain at least one robot trajectory")
        self.profile = profile
        self.guess = guess
        self.config = PlannerConfig() if config is None else config
        self.corridor_cons = corridor_cons or [[[] for _ in range(len(guess[0].states) - 1)] for _ in guess]
        self.height_cons = height_cons or [-1.0 for _ in range(len(guess[0].states))]
        self.w_inf = float(w_inf)
        self.robot_count = len(guess)
        self.nfe = len(guess[0].states)
        self.nrows = self.nfe - 1
        if len(profile) != self.robot_count:
            raise ValueError("profile and guess must have the same robot count")
        if any(len(item.states) != self.nfe for item in guess):
            raise ValueError("all robot guesses must have the same number of states")

        self.add_var = self.robot_count + self.robot_count * (self.robot_count - 2)
        self.ncols = NVAR * self.robot_count + self.add_var
        self.num_sfc_cons = sum(len(self.corridor_cons[j][r]) for j in range(self.robot_count) for r in range(self.nrows))
        self.nvar = 1 + self.nrows * self.ncols + 2 * self.num_sfc_cons + 1
        self.vvcm = VVCMConstants()

    def idx_state(self, robot: int, var: int, row: int) -> int:
        return 1 + (var + NVAR * robot) * self.nrows + row

    def idx_edge_distance(self, robot: int, row: int) -> int:
        return 1 + (NVAR * self.robot_count + robot) * self.nrows + row

    def idx_topology(self, topo_index: int, row: int) -> int:
        return 1 + (NVAR * self.robot_count + self.robot_count + topo_index) * self.nrows + row

    def idx_sfc(self, slack_index: int) -> int:
        return 1 + (NVAR * self.robot_count + self.add_var) * self.nrows + slack_index

    @property
    def idx_terminal_error(self) -> int:
        return self.nvar - 1

    def pack_initial_guess(self) -> np.ndarray:
        """Build C++ ``get_starting_point`` style variable vector."""
        x = np.zeros(self.nvar, dtype=float)
        x[0] = self.guess[0].tf
        x[self.idx_terminal_error] = sum(
            (self.profile[j].goal.x - self.guess[j].states[-1].x) ** 2
            + (self.profile[j].goal.y - self.guess[j].states[-1].y) ** 2
            + (self.profile[j].goal.theta - self.guess[j].states[-1].theta) ** 2
            for j in range(self.robot_count)
        )

        for row in range(self.nrows):
            state_index = row + 1
            for robot, traj in enumerate(self.guess):
                vec = traj.states[state_index].as_vector()
                for var in range(NVAR):
                    x[self.idx_state(robot, var, row)] = vec[var]

        for row in range(self.nrows):
            state_index = row + 1
            for robot in range(self.robot_count):
                current = self.guess[robot].states[state_index]
                following = self.guess[(robot + 1) % self.robot_count].states[state_index]
                x[self.idx_edge_distance(robot, row)] = (following.x - current.x) ** 2 + (following.y - current.y) ** 2

            topo_index = 0
            for robot in range(self.robot_count):
                current = self.guess[robot].states[state_index]
                for p in range(self.robot_count):
                    if p == robot or (p + 1) % self.robot_count == robot:
                        continue
                    prev = self.guess[p].states[state_index]
                    next_state = self.guess[(p + 1) % self.robot_count].states[state_index]
                    x[self.idx_topology(topo_index, row)] = (current.x - prev.x) * (prev.y - next_state.y) + (
                        current.y - prev.y
                    ) * (next_state.x - prev.x)
                    topo_index += 1

        slack_index = 0
        for row in range(self.nrows):
            state_index = row + 1
            for robot, traj in enumerate(self.guess):
                state = traj.states[state_index]
                discs = self.config.vehicle.disc_positions(state.x, state.y, state.theta)
                for disc in range(self.config.vehicle.n_disc):
                    px, py = discs[2 * disc], discs[2 * disc + 1]
                    for halfspace in self.corridor_cons[robot][row]:
                        x[self.idx_sfc(slack_index)] = halfspace[0] * px + halfspace[1] * py - halfspace[2]
                        slack_index += 1
        return x

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return C++ ``get_bounds_info`` style lower and upper bounds."""
        lower = np.full(self.nvar, -np.inf, dtype=float)
        upper = np.full(self.nvar, np.inf, dtype=float)

        lower[0] = 0.1
        lower[self.idx_terminal_error] = 1e-4
        upper[self.idx_terminal_error] = 12.0

        for row in range(self.nrows):
            for robot in range(self.robot_count):
                lower[self.idx_state(robot, 3, row)] = self.config.vehicle.min_velocity
                lower[self.idx_state(robot, 4, row)] = -self.config.vehicle.phi_min
                lower[self.idx_state(robot, 5, row)] = -self.config.vehicle.max_acceleration
                lower[self.idx_state(robot, 6, row)] = -self.config.vehicle.omega_max

                upper[self.idx_state(robot, 3, row)] = self.config.vehicle.max_velocity
                upper[self.idx_state(robot, 4, row)] = self.config.vehicle.phi_max
                upper[self.idx_state(robot, 5, row)] = self.config.vehicle.max_acceleration
                upper[self.idx_state(robot, 6, row)] = self.config.vehicle.omega_max

        for row in range(self.nrows):
            height = self.height_cons[row + 1]
            edge_lower = self.vvcm.xv2t * self.vvcm.xv2t if height == -1 else 3.0 * height * height
            edge_upper = self.vvcm.xv2 * self.vvcm.xv2
            for robot in range(self.robot_count):
                lower[self.idx_edge_distance(robot, row)] = edge_lower
                upper[self.idx_edge_distance(robot, row)] = edge_upper
            for topo_index in range(self.robot_count * (self.robot_count - 2)):
                lower[self.idx_topology(topo_index, row)] = 0.0

        for slack_index in range(2 * self.num_sfc_cons):
            upper[self.idx_sfc(slack_index)] = 0.0

        return lower, upper

    def eval_infeasibility(self, x: np.ndarray) -> float:
        """Evaluate the softened constraints from C++ ``eval_infeasibility``."""
        x = np.asarray(x, dtype=float)
        dt = x[0] / self.nfe
        infeasibility = 0.0
        terminal_x = 0.0
        terminal_y = 0.0
        target_x = 0.0
        target_y = 0.0
        for robot in range(self.robot_count):
            terminal_x += x[self.idx_state(robot, 0, self.nrows - 1)]
            terminal_y += x[self.idx_state(robot, 1, self.nrows - 1)]
            target_x += self.profile[robot].goal.x
            target_y += self.profile[robot].goal.y
        terminal_mean_error = ((terminal_x - target_x) / self.robot_count) ** 2 + (
            (terminal_y - target_y) / self.robot_count
        ) ** 2

        for robot in range(self.robot_count):
            start = self.profile[robot].start
            goal = self.profile[robot].goal
            infeasibility += (
                (x[self.idx_state(robot, 0, 0)] - start.x - dt * start.v * np.cos(start.theta)) ** 2
                + (x[self.idx_state(robot, 1, 0)] - start.y - dt * start.v * np.sin(start.theta)) ** 2
                + (
                    x[self.idx_state(robot, 2, 0)]
                    - start.theta
                    - dt * start.v * np.tan(start.phi) / self.config.vehicle.wheel_base
                )
                ** 2
                + (x[self.idx_state(robot, 3, 0)] - start.v - dt * start.a) ** 2
                + (x[self.idx_state(robot, 4, 0)] - start.phi - dt * start.omega) ** 2
            )

            last = self.nrows - 1
            infeasibility += (
                (goal.v - x[self.idx_state(robot, 3, last)]) ** 2
                + (goal.phi - x[self.idx_state(robot, 4, last)]) ** 2
                + (goal.a - x[self.idx_state(robot, 5, last)]) ** 2
                + (goal.omega - x[self.idx_state(robot, 6, last)]) ** 2
            )

            for row in range(1, self.nrows):
                prev = row - 1
                infeasibility += (
                    (
                        x[self.idx_state(robot, 0, row)]
                        - x[self.idx_state(robot, 0, prev)]
                        - dt * x[self.idx_state(robot, 3, prev)] * np.cos(x[self.idx_state(robot, 2, prev)])
                    )
                    ** 2
                    + (
                        x[self.idx_state(robot, 1, row)]
                        - x[self.idx_state(robot, 1, prev)]
                        - dt * x[self.idx_state(robot, 3, prev)] * np.sin(x[self.idx_state(robot, 2, prev)])
                    )
                    ** 2
                    + (
                        x[self.idx_state(robot, 2, row)]
                        - x[self.idx_state(robot, 2, prev)]
                        - dt
                        * x[self.idx_state(robot, 3, prev)]
                        * np.tan(x[self.idx_state(robot, 4, prev)])
                        / self.config.vehicle.wheel_base
                    )
                    ** 2
                    + (
                        x[self.idx_state(robot, 3, row)]
                        - x[self.idx_state(robot, 3, prev)]
                        - dt * x[self.idx_state(robot, 5, prev)]
                    )
                    ** 2
                    + (
                        x[self.idx_state(robot, 4, row)]
                        - x[self.idx_state(robot, 4, prev)]
                        - dt * x[self.idx_state(robot, 6, prev)]
                    )
                    ** 2
                )

        infeasibility += (x[self.idx_terminal_error] - terminal_mean_error) ** 2

        for row in range(self.nrows):
            for robot in range(self.robot_count):
                next_robot = (robot + 1) % self.robot_count
                dx = x[self.idx_state(next_robot, 0, row)] - x[self.idx_state(robot, 0, row)]
                dy = x[self.idx_state(next_robot, 1, row)] - x[self.idx_state(robot, 1, row)]
                infeasibility += (x[self.idx_edge_distance(robot, row)] - (dx * dx + dy * dy)) ** 2

        for row in range(self.nrows):
            topo_index = 0
            for robot in range(self.robot_count):
                xk = x[self.idx_state(robot, 0, row)]
                yk = x[self.idx_state(robot, 1, row)]
                for p in range(self.robot_count):
                    if p == robot or (p + 1) % self.robot_count == robot:
                        continue
                    next_p = (p + 1) % self.robot_count
                    xp = x[self.idx_state(p, 0, row)]
                    yp = x[self.idx_state(p, 1, row)]
                    xnp = x[self.idx_state(next_p, 0, row)]
                    ynp = x[self.idx_state(next_p, 1, row)]
                    expr = (xk - xp) * (yp - ynp) + (yk - yp) * (xnp - xp)
                    infeasibility += (x[self.idx_topology(topo_index, row)] - expr) ** 2
                    topo_index += 1

        slack_index = 0
        for row in range(self.nrows):
            for robot in range(self.robot_count):
                theta = x[self.idx_state(robot, 2, row)]
                for coeff in self.config.vehicle.disc_coefficients:
                    px = x[self.idx_state(robot, 0, row)] + coeff * np.cos(theta)
                    py = x[self.idx_state(robot, 1, row)] + coeff * np.sin(theta)
                    for halfspace in self.corridor_cons[robot][row]:
                        expr = halfspace[0] * px + halfspace[1] * py - halfspace[2]
                        infeasibility += (x[self.idx_sfc(slack_index)] - expr) ** 2
                        slack_index += 1

        return float(infeasibility)

    def eval_objective(self, x: np.ndarray) -> float:
        """Evaluate C++ ``eval_obj`` terms currently present in SolveFm."""
        x = np.asarray(x, dtype=float)
        obj = self.config.opti_t * x[0]
        for robot in range(self.robot_count):
            for row in range(self.nrows):
                obj += (
                    self.config.opti_w_a * x[self.idx_state(robot, 5, row)] ** 2
                    + self.config.opti_w_omega * x[self.idx_state(robot, 6, row)] ** 2
                )
        obj += self.w_inf * self.eval_infeasibility(x)
        return float(obj)

    def unpack_solution(self, x: np.ndarray) -> list[FullStates]:
        """Convert an NLP vector to C++ ``ConvertVectorToJointStates`` layout."""
        x = np.asarray(x, dtype=float)
        result: list[FullStates] = []
        for robot in range(self.robot_count):
            states = [self.profile[robot].start]
            for row in range(self.nrows):
                vec = np.array([x[self.idx_state(robot, var, row)] for var in range(NVAR)], dtype=float)
                states.append(TrajectoryPoint.from_vector(vec))
            result.append(FullStates(tf=float(x[0]), states=states))
        return result

    def clipped_initial_guess(self) -> np.ndarray:
        """Return a bound-compatible starting vector for SciPy optimizers."""
        x0 = self.pack_initial_guess()
        lower, upper = self.bounds()
        finite_lower = np.isfinite(lower)
        finite_upper = np.isfinite(upper)
        x0[finite_lower] = np.maximum(x0[finite_lower], lower[finite_lower])
        x0[finite_upper] = np.minimum(x0[finite_upper], upper[finite_upper])
        return x0

    def solve(
        self,
        *,
        method: str = "L-BFGS-B",
        maxiter: int = 200,
        ftol: float = 1e-9,
    ) -> FormationNLPSolution:
        """Solve the C++ ``SolveFm`` penalty NLP with SciPy.

        The C++ implementation gives IPOPT no explicit constraints for this
        problem (`m = 0`); kinematics, formation, topology, and safe-corridor
        relations are folded into ``w_inf * eval_infeasibility`` plus variable
        bounds. This method preserves that structure and uses SciPy as the
        portable Python optimizer.
        """
        lower, upper = self.bounds()
        x0 = self.clipped_initial_guess()
        start = time.perf_counter()
        opt = minimize(
            self.eval_objective,
            x0,
            method=method,
            bounds=Bounds(lower, upper),
            options={"maxiter": maxiter, "ftol": ftol},
        )
        elapsed = time.perf_counter() - start
        vector = np.asarray(opt.x, dtype=float)
        return FormationNLPSolution(
            states=self.unpack_solution(vector),
            vector=vector,
            objective=self.eval_objective(vector),
            infeasibility=self.eval_infeasibility(vector),
            solve_time=elapsed,
            scipy_success=bool(opt.success),
            scipy_message=str(opt.message),
            iterations=int(getattr(opt, "nit", -1)),
        )


def solve_fm(
    profile: list[Constraints],
    guess: list[FullStates],
    *,
    config: PlannerConfig | None = None,
    corridor_cons: list[list[list[list[float]]]] | None = None,
    height_cons: list[float] | None = None,
    w_inf: float = 1e4,
    method: str = "L-BFGS-B",
    maxiter: int = 200,
) -> FormationNLPSolution:
    """Python equivalent of C++ ``LightweightProblem::SolveFm``."""
    problem = FormationNLPProblem(
        profile,
        guess,
        config=config,
        corridor_cons=corridor_cons,
        height_cons=height_cons,
        w_inf=w_inf,
    )
    return problem.solve(method=method, maxiter=maxiter)
