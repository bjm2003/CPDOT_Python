"""Joint formation NLP structures ported from CPDOT's IPOPT problem."""

from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np
from scipy.optimize import Bounds, least_squares, minimize

from .states import NVAR, Constraints, FullStates, TrajectoryPoint


@dataclass
class VehicleModel:
    """Subset of C++ ``VehicleModel`` needed by CPDOT NLP interfaces."""

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
    vertices: int = 4
    min_vel_diff: float = -1.0
    max_vel_diff: float = 2.0
    omg_acc_diff: float = 2.5
    max_acc_diff: float = 1.0
    omg_max_diff: float = 1.5
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

    def formation_centre(self, x: float, y: float, theta: float) -> np.ndarray:
        """Port C++ ``VehicleModel::GetFormationCentre``."""
        offset_angle = float(np.arctan2(self.offset / 2.0, 1.0))
        d_fc = float(np.hypot(self.offset / 2.0, 1.0))
        return np.asarray([x + d_fc * np.cos(theta - offset_angle), y + d_fc * np.sin(theta - offset_angle)])

    def vertex_positions(self, x: float, y: float, theta: float, scale: float = 1.0) -> np.ndarray:
        """Port the active C++ ``GetVertexPositions(..., 1.0)`` geometry."""
        rear_diag = float(np.hypot(self.rear_hang_length, self.width / 2.0))
        rear_angle = float(np.arctan2(self.width / 2.0, self.rear_hang_length))
        if scale > 1.0:
            long_x = 1.2 + 3.0
            long_y = 1.2 + 3.0
        else:
            long_x = 1.2 + scale * 3.0
            long_y = 1.2 + scale * self.offset
        long_diag = float(np.hypot(long_x, long_y))
        long_angle = float(np.arctan2(1.2 + self.offset, 1.2 + 3.0))
        pts = [
            (
                x - rear_diag * np.cos(theta - rear_angle),
                y - rear_diag * np.sin(theta - rear_angle),
            ),
            (
                x - rear_diag * np.cos(theta - rear_angle) + long_x * np.cos(theta),
                y - rear_diag * np.sin(theta - rear_angle) + long_y * np.sin(theta),
            ),
            (
                x - rear_diag * np.cos(theta - rear_angle) + long_diag * np.cos(theta - long_angle),
                y - rear_diag * np.sin(theta - rear_angle) + long_diag * np.sin(theta - long_angle),
            ),
            (
                x - rear_diag * np.cos(theta - rear_angle) + long_y * np.cos(np.pi / 2.0 - theta),
                y - rear_diag * np.sin(theta - rear_angle) - long_y * np.sin(np.pi / 2.0 - theta),
            ),
        ]
        return np.asarray([value for point in pts for value in point], dtype=float)


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
    corridor_max_iter: int = 1000
    corridor_incremental_limit: float = 20.0
    opti_t: float = 1.0
    opti_w_phi: float = 1.0
    opti_w_a: float = 1.0
    factor_a: float = 0.9
    factor_b: float = 1.1
    opti_w_omega: float = 1.0
    opti_w_diff_drive: float = 0.05
    opti_w_err: float = 1.0
    opti_w_x: float = 1.0
    opti_w_y: float = 1.0
    opti_w_theta: float = 1.0
    opti_inner_iter_max: int = 100
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
    infeasibility_terms: dict[str, float]
    solve_time: float
    scipy_success: bool
    scipy_message: str
    iterations: int


@dataclass
class SingleRobotNLPSolution:
    """Result of C++ single-robot ``LightweightProblem`` counterparts."""

    state: FullStates
    vector: np.ndarray
    objective: float
    infeasibility: float
    solve_time: float
    scipy_success: bool
    scipy_message: str
    iterations: int


class SingleRobotNLPProblemBase:
    """Shared C++ layout for single-robot ``LightweightProblem`` TNLPs."""

    def __init__(
        self,
        profile: Constraints,
        guess: FullStates,
        *,
        config: PlannerConfig | None = None,
        pre_sol: FullStates | None = None,
        w_inf: float = 1e4,
    ):
        if len(guess.states) < 3:
            raise ValueError("single-robot NLP guesses must contain at least three states")
        self.profile = profile
        self.guess = guess
        self.pre_sol = pre_sol or FullStates()
        self.config = PlannerConfig() if config is None else config
        self.w_inf = float(w_inf)
        self.nfe = len(guess.states)
        self.nrows = self.nfe - 2
        self.ncols = NVAR
        self.nvar = 1 + self.nrows * self.ncols + 1

    @property
    def idx_terminal_theta(self) -> int:
        return self.nvar - 1

    def idx_state(self, var: int, row: int) -> int:
        return 1 + var * self.nrows + row

    def pack_initial_guess(self) -> np.ndarray:
        source = self.pre_sol if self.pre_sol.states else self.guess
        if len(source.states) != self.nfe:
            raise ValueError("pre_sol must have the same number of states as guess")
        x = np.zeros(self.nvar, dtype=float)
        x[0] = source.tf
        x[self.idx_terminal_theta] = source.states[-1].theta
        for row in range(self.nrows):
            vec = source.states[row + 1].as_vector()
            for var in range(NVAR):
                x[self.idx_state(var, row)] = vec[var]
        return x

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lower = np.full(self.nvar, -np.inf, dtype=float)
        upper = np.full(self.nvar, np.inf, dtype=float)
        return lower, upper

    def clipped_initial_guess(self) -> np.ndarray:
        x0 = self.pack_initial_guess()
        lower, upper = self.bounds()
        finite_lower = np.isfinite(lower)
        finite_upper = np.isfinite(upper)
        x0[finite_lower] = np.maximum(x0[finite_lower], lower[finite_lower])
        x0[finite_upper] = np.minimum(x0[finite_upper], upper[finite_upper])
        return x0

    def unpack_solution(
        self,
        x: np.ndarray,
        start: TrajectoryPoint,
        goal: TrajectoryPoint,
    ) -> FullStates:
        x = np.asarray(x, dtype=float)
        states = [start]
        for row in range(self.nrows):
            vec = np.array([x[self.idx_state(var, row)] for var in range(NVAR)], dtype=float)
            states.append(TrajectoryPoint.from_vector(vec))
        states.append(goal)
        return FullStates(tf=float(x[0]), states=states)

    def eval_infeasibility(self, x: np.ndarray) -> float:
        raise NotImplementedError

    def eval_objective(self, x: np.ndarray) -> float:
        raise NotImplementedError

    def solve(
        self,
        *,
        start: TrajectoryPoint,
        goal: TrajectoryPoint,
        method: str = "L-BFGS-B",
        maxiter: int = 200,
        ftol: float = 1e-9,
    ) -> SingleRobotNLPSolution:
        if method == "ipopt":
            return self._solve_ipopt(start=start, goal=goal, maxiter=maxiter)
        lower, upper = self.bounds()
        x0 = self.clipped_initial_guess()
        start_time = time.perf_counter()
        opt = minimize(
            self.eval_objective,
            x0,
            method=method,
            bounds=Bounds(lower, upper),
            options={"maxiter": maxiter, "ftol": ftol},
        )
        elapsed = time.perf_counter() - start_time
        vector = np.asarray(opt.x, dtype=float)
        return SingleRobotNLPSolution(
            state=self.unpack_solution(vector, start, goal),
            vector=vector,
            objective=self.eval_objective(vector),
            infeasibility=self.eval_infeasibility(vector),
            solve_time=elapsed,
            scipy_success=bool(opt.success),
            scipy_message=str(opt.message),
            iterations=int(getattr(opt, "nit", -1)),
        )

    def _solve_ipopt(
        self,
        *,
        start: TrajectoryPoint,
        goal: TrajectoryPoint,
        maxiter: int,
    ) -> SingleRobotNLPSolution:
        from . import optimizer_casadi

        if isinstance(self, CarLikeNLPProblem):
            return optimizer_casadi.solve_car_like_ipopt(
                self, start=start, goal=goal, maxiter=maxiter
            )
        if isinstance(self, DiffDriveNLPProblem):
            return optimizer_casadi.solve_diff_drive_ipopt(
                self, start=start, goal=goal, maxiter=maxiter
            )
        if isinstance(self, CarLikeReplanNLPProblem):
            return optimizer_casadi.solve_replan_ipopt(
                self, start=start, goal=goal, maxiter=maxiter
            )
        raise TypeError(f"ipopt backend does not support {type(self).__name__}")


class CarLikeNLPProblem(SingleRobotNLPProblemBase):
    """Python layout/evaluator for C++ ``LiomIPOPTInterface``."""

    def __init__(
        self,
        profile: Constraints,
        guess: FullStates,
        *,
        config: PlannerConfig | None = None,
        w_inf: float = 1e4,
    ):
        super().__init__(profile, guess, config=config, w_inf=w_inf)
        self.vert_nvar = self.config.vehicle.vertices * 2
        self.ncols = NVAR + self.vert_nvar
        self.nvar = 1 + self.nrows * self.ncols + 1

    def idx_vertex(self, vertex_var: int, row: int) -> int:
        return 1 + (NVAR + vertex_var) * self.nrows + row

    def _active_residual_vertices(self, x: float, y: float, theta: float) -> np.ndarray:
        """Return the vertex formula active in C++ ``eval_infeasibility``."""
        vehicle = self.config.vehicle
        rear_diag = float(np.hypot(vehicle.rear_hang_length, vehicle.width / 2.0))
        rear_angle = float(np.arctan2(vehicle.width / 2.0, vehicle.rear_hang_length))
        rear_x = x - rear_diag * np.cos(theta - rear_angle)
        rear_y = y - rear_diag * np.sin(theta - rear_angle)
        long_x = 1.2 + 2.0
        long_y = 1.2 + vehicle.offset
        long_diag = float(np.hypot(long_x, long_y))
        long_angle = float(np.arctan2(long_y, long_x))
        pts = [
            (rear_x, rear_y),
            (rear_x + long_x * np.cos(theta), rear_y + long_y * np.sin(theta)),
            (rear_x + long_diag * np.cos(theta - long_angle), rear_y + long_diag * np.sin(theta - long_angle)),
            (rear_x + long_y * np.cos(np.pi / 2.0 - theta), rear_y - long_y * np.sin(np.pi / 2.0 - theta)),
        ]
        return np.asarray([value for point in pts for value in point], dtype=float)

    def pack_initial_guess(self) -> np.ndarray:
        x = np.zeros(self.nvar, dtype=float)
        x[0] = self.guess.tf
        x[self.idx_terminal_theta] = self.profile.goal.theta
        for row in range(self.nrows):
            state = self.guess.states[row + 1]
            vec = state.as_vector()
            for var in range(NVAR):
                x[self.idx_state(var, row)] = vec[var]
            vertices = self.config.vehicle.vertex_positions(state.x, state.y, state.theta, 1.0)
            for var in range(self.vert_nvar):
                x[self.idx_vertex(var, row)] = vertices[var]
        return x

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lower = np.full(self.nvar, -np.inf, dtype=float)
        upper = np.full(self.nvar, np.inf, dtype=float)
        lower[0] = 0.1
        for row in range(self.nrows):
            lower[self.idx_state(3, row)] = self.config.vehicle.min_velocity
            lower[self.idx_state(4, row)] = -self.config.vehicle.phi_min
            lower[self.idx_state(5, row)] = -self.config.vehicle.max_acceleration
            lower[self.idx_state(6, row)] = -self.config.vehicle.omega_max
            upper[self.idx_state(3, row)] = self.config.vehicle.max_velocity
            upper[self.idx_state(4, row)] = self.config.vehicle.phi_max
            upper[self.idx_state(5, row)] = self.config.vehicle.max_acceleration
            upper[self.idx_state(6, row)] = self.config.vehicle.omega_max

        if self.profile.corridor_lb is not None:
            lb = np.asarray(self.profile.corridor_lb, dtype=float)
            for row in range(self.nrows):
                for var in range(min(self.vert_nvar, lb.shape[1])):
                    lower[self.idx_vertex(var, row)] = lb[row + 1, var]
        if self.profile.corridor_ub is not None:
            ub = np.asarray(self.profile.corridor_ub, dtype=float)
            for row in range(self.nrows):
                for var in range(min(self.vert_nvar, ub.shape[1])):
                    upper[self.idx_vertex(var, row)] = ub[row + 1, var]
        return lower, upper

    def eval_infeasibility(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        dt = x[0] / self.nfe
        start = self.profile.start
        goal = self.profile.goal
        infs = (
            (x[self.idx_state(0, 0)] - start.x - dt * start.v * np.cos(start.theta)) ** 2
            + (x[self.idx_state(1, 0)] - start.y - dt * start.v * np.sin(start.theta)) ** 2
            + (
                x[self.idx_state(2, 0)]
                - start.theta
                - dt * start.v * np.tan(start.phi) / self.config.vehicle.wheel_base
            )
            ** 2
            + (x[self.idx_state(3, 0)] - start.v - dt * start.a) ** 2
            + (x[self.idx_state(4, 0)] - start.phi - dt * start.omega) ** 2
        )
        for row in range(1, self.nrows):
            prev = row - 1
            infs += (
                (
                    x[self.idx_state(0, row)]
                    - x[self.idx_state(0, prev)]
                    - dt * x[self.idx_state(3, prev)] * np.cos(x[self.idx_state(2, prev)])
                )
                ** 2
                + (
                    x[self.idx_state(1, row)]
                    - x[self.idx_state(1, prev)]
                    - dt * x[self.idx_state(3, prev)] * np.sin(x[self.idx_state(2, prev)])
                )
                ** 2
                + (
                    x[self.idx_state(2, row)]
                    - x[self.idx_state(2, prev)]
                    - dt
                    * x[self.idx_state(3, prev)]
                    * np.tan(x[self.idx_state(4, prev)])
                    / self.config.vehicle.wheel_base
                )
                ** 2
                + (x[self.idx_state(3, row)] - x[self.idx_state(3, prev)] - dt * x[self.idx_state(5, prev)]) ** 2
                + (x[self.idx_state(4, row)] - x[self.idx_state(4, prev)] - dt * x[self.idx_state(6, prev)]) ** 2
            )

        last = self.nrows - 1
        infs += (
            (
                goal.x
                - x[self.idx_state(0, last)]
                - dt * x[self.idx_state(3, last)] * np.cos(x[self.idx_state(2, last)])
            )
            ** 2
            + (
                goal.y
                - x[self.idx_state(1, last)]
                - dt * x[self.idx_state(3, last)] * np.sin(x[self.idx_state(2, last)])
            )
            ** 2
            + (
                x[self.idx_terminal_theta]
                - x[self.idx_state(2, last)]
                - dt
                * x[self.idx_state(3, last)]
                * np.tan(x[self.idx_state(4, last)])
                / self.config.vehicle.wheel_base
            )
            ** 2
            + (goal.v - x[self.idx_state(3, last)] - dt * x[self.idx_state(5, last)]) ** 2
            + (goal.phi - x[self.idx_state(4, last)] - dt * x[self.idx_state(6, last)]) ** 2
            + (np.sin(goal.theta) - np.sin(x[self.idx_terminal_theta])) ** 2
            + (np.cos(goal.theta) - np.cos(x[self.idx_terminal_theta])) ** 2
        )
        for row in range(self.nrows):
            expected = self._active_residual_vertices(
                x[self.idx_state(0, row)],
                x[self.idx_state(1, row)],
                x[self.idx_state(2, row)],
            )
            for var in range(self.vert_nvar):
                infs += (x[self.idx_vertex(var, row)] - expected[var]) ** 2
        return float(infs)

    def eval_objective(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        obj = self.config.opti_t * x[0]
        for row in range(self.nrows):
            obj += self.config.opti_w_diff_drive * x[self.idx_state(5, row)] ** 2
            obj += self.config.opti_w_diff_drive * x[self.idx_state(6, row)] ** 2
        for row in range(1, self.nrows):
            obj += self.config.opti_w_diff_drive * x[self.idx_state(3, row)] ** 2
        obj += self.w_inf * self.eval_infeasibility(x)
        return float(obj)


class DiffDriveNLPProblem(SingleRobotNLPProblemBase):
    """Python layout/evaluator for C++ ``LiomIPOPTInterface_diff_drive``."""

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lower = np.full(self.nvar, -np.inf, dtype=float)
        upper = np.full(self.nvar, np.inf, dtype=float)
        lower[0] = self.guess.tf
        upper[0] = self.guess.tf
        for row in range(self.nrows):
            lower[self.idx_state(3, row)] = self.config.vehicle.min_vel_diff
            lower[self.idx_state(4, row)] = -self.config.vehicle.omg_acc_diff
            lower[self.idx_state(5, row)] = -self.config.vehicle.max_acc_diff
            lower[self.idx_state(6, row)] = -self.config.vehicle.omg_max_diff
            upper[self.idx_state(3, row)] = self.config.vehicle.max_vel_diff
            upper[self.idx_state(4, row)] = self.config.vehicle.omg_acc_diff
            upper[self.idx_state(5, row)] = self.config.vehicle.max_acc_diff
            upper[self.idx_state(6, row)] = self.config.vehicle.omg_max_diff
        return lower, upper

    def eval_infeasibility(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        dt = x[0] / self.nfe
        first = self.guess.states[0]
        last_state = self.guess.states[-1]
        infs = (
            (x[self.idx_state(0, 0)] - first.x - dt * first.v * np.cos(first.theta)) ** 2
            + (x[self.idx_state(1, 0)] - first.y - dt * first.v * np.sin(first.theta)) ** 2
            + (x[self.idx_state(2, 0)] - first.theta - dt * first.omega) ** 2
            + (x[self.idx_state(3, 0)] - first.v - dt * first.a) ** 2
            + (x[self.idx_state(6, 0)] - first.omega - dt * first.phi) ** 2
        )
        for row in range(1, self.nrows):
            prev = row - 1
            infs += (
                (
                    x[self.idx_state(0, row)]
                    - x[self.idx_state(0, prev)]
                    - dt * x[self.idx_state(3, prev)] * np.cos(x[self.idx_state(2, prev)])
                )
                ** 2
                + (
                    x[self.idx_state(1, row)]
                    - x[self.idx_state(1, prev)]
                    - dt * x[self.idx_state(3, prev)] * np.sin(x[self.idx_state(2, prev)])
                )
                ** 2
                + (x[self.idx_state(2, row)] - x[self.idx_state(2, prev)] - dt * x[self.idx_state(6, prev)]) ** 2
                + (x[self.idx_state(3, row)] - x[self.idx_state(3, prev)] - dt * x[self.idx_state(5, prev)]) ** 2
                + (x[self.idx_state(6, row)] - x[self.idx_state(6, prev)] - dt * x[self.idx_state(4, prev)]) ** 2
            )
        last = self.nrows - 1
        infs += (
            (
                last_state.x
                - x[self.idx_state(0, last)]
                - dt * x[self.idx_state(3, last)] * np.cos(x[self.idx_state(2, last)])
            )
            ** 2
            + (
                last_state.y
                - x[self.idx_state(1, last)]
                - dt * x[self.idx_state(3, last)] * np.sin(x[self.idx_state(2, last)])
            )
            ** 2
            + (x[self.idx_terminal_theta] - x[self.idx_state(2, last)] - dt * x[self.idx_state(6, last)]) ** 2
            + (last_state.v - x[self.idx_state(3, last)] - dt * x[self.idx_state(5, last)]) ** 2
            + (last_state.omega - x[self.idx_state(6, last)] - dt * x[self.idx_state(4, last)]) ** 2
            + (np.sin(last_state.theta) - np.sin(x[self.idx_terminal_theta])) ** 2
            + (np.cos(last_state.theta) - np.cos(x[self.idx_terminal_theta])) ** 2
        )
        return float(infs)

    def eval_objective(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        obj = x[0]
        for row in range(self.nrows):
            obj += self.config.opti_w_err * (x[self.idx_state(0, row)] - self.guess.states[row].x) ** 2
            obj += self.config.opti_w_err * (x[self.idx_state(1, row)] - self.guess.states[row].y) ** 2
        obj += self.w_inf * self.eval_infeasibility(x)
        return float(obj)


class CarLikeReplanNLPProblem(SingleRobotNLPProblemBase):
    """Python layout/evaluator for C++ ``LiomIPOPTInterface_car_like_replan``."""

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lower = np.full(self.nvar, -np.inf, dtype=float)
        upper = np.full(self.nvar, np.inf, dtype=float)
        lower[0] = self.guess.tf
        upper[0] = self.guess.tf
        for row in range(self.nrows):
            lower[self.idx_state(3, row)] = self.config.vehicle.min_velocity
            lower[self.idx_state(4, row)] = -self.config.vehicle.phi_max
            lower[self.idx_state(5, row)] = -self.config.vehicle.max_acceleration
            lower[self.idx_state(6, row)] = -self.config.vehicle.omega_max
            upper[self.idx_state(3, row)] = self.config.vehicle.max_velocity
            upper[self.idx_state(4, row)] = self.config.vehicle.phi_max
            upper[self.idx_state(5, row)] = self.config.vehicle.max_acceleration
            upper[self.idx_state(6, row)] = self.config.vehicle.omega_max
        return lower, upper

    def eval_infeasibility(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        dt = x[0] / self.nfe
        first = self.guess.states[0]
        last_state = self.guess.states[-1]
        infs = (
            (x[self.idx_state(0, 0)] - first.x - dt * first.v * np.cos(first.theta)) ** 2
            + (x[self.idx_state(1, 0)] - first.y - dt * first.v * np.sin(first.theta)) ** 2
            + (
                x[self.idx_state(2, 0)]
                - first.theta
                - dt * first.v * np.tan(first.phi) / self.config.vehicle.wheel_base
            )
            ** 2
            + (x[self.idx_state(3, 0)] - first.v - dt * first.a) ** 2
            + (x[self.idx_state(4, 0)] - first.phi - dt * first.omega) ** 2
        )
        for row in range(1, self.nrows):
            prev = row - 1
            infs += (
                (
                    x[self.idx_state(0, row)]
                    - x[self.idx_state(0, prev)]
                    - dt * x[self.idx_state(3, prev)] * np.cos(x[self.idx_state(2, prev)])
                )
                ** 2
                + (
                    x[self.idx_state(1, row)]
                    - x[self.idx_state(1, prev)]
                    - dt * x[self.idx_state(3, prev)] * np.sin(x[self.idx_state(2, prev)])
                )
                ** 2
                + (
                    x[self.idx_state(2, row)]
                    - x[self.idx_state(2, prev)]
                    - dt
                    * x[self.idx_state(3, prev)]
                    * np.tan(x[self.idx_state(4, prev)])
                    / self.config.vehicle.wheel_base
                )
                ** 2
                + (x[self.idx_state(3, row)] - x[self.idx_state(3, prev)] - dt * x[self.idx_state(5, prev)]) ** 2
                + (x[self.idx_state(4, row)] - x[self.idx_state(4, prev)] - dt * x[self.idx_state(6, prev)]) ** 2
            )
        last = self.nrows - 1
        infs += (
            (
                last_state.x
                - x[self.idx_state(0, last)]
                - dt * x[self.idx_state(3, last)] * np.cos(x[self.idx_state(2, last)])
            )
            ** 2
            + (
                last_state.y
                - x[self.idx_state(1, last)]
                - dt * x[self.idx_state(3, last)] * np.sin(x[self.idx_state(2, last)])
            )
            ** 2
            + (x[self.idx_terminal_theta] - x[self.idx_state(2, last)] - dt * x[self.idx_state(6, last)]) ** 2
            + (last_state.v - x[self.idx_state(3, last)] - dt * x[self.idx_state(5, last)]) ** 2
            + (last_state.phi - x[self.idx_state(4, last)] - dt * x[self.idx_state(6, last)]) ** 2
            + (np.sin(last_state.theta) - np.sin(x[self.idx_terminal_theta])) ** 2
            + (np.cos(last_state.theta) - np.cos(x[self.idx_terminal_theta])) ** 2
        )
        return float(infs)

    def eval_objective(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        obj = x[0]
        for row in range(self.nrows):
            obj += self.config.opti_w_x * (x[self.idx_state(0, row)] - self.guess.states[row].x) ** 2
            obj += self.config.opti_w_y * (x[self.idx_state(1, row)] - self.guess.states[row].y) ** 2
        obj += self.w_inf * self.eval_infeasibility(x)
        return float(obj)


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

    def eval_infeasibility_terms(self, x: np.ndarray) -> dict[str, float]:
        """Return grouped C++ ``eval_infeasibility`` residual terms."""
        x = np.asarray(x, dtype=float)
        dt = x[0] / self.nfe
        terms = {
            "initial_terminal": 0.0,
            "dynamics": 0.0,
            "terminal_error": 0.0,
            "edge_distance": 0.0,
            "topology": 0.0,
            "sfc": 0.0,
        }
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
            terms["initial_terminal"] += (
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
            terms["initial_terminal"] += (
                (goal.v - x[self.idx_state(robot, 3, last)]) ** 2
                + (goal.phi - x[self.idx_state(robot, 4, last)]) ** 2
                + (goal.a - x[self.idx_state(robot, 5, last)]) ** 2
                + (goal.omega - x[self.idx_state(robot, 6, last)]) ** 2
            )

            for row in range(1, self.nrows):
                prev = row - 1
                terms["dynamics"] += (
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

        terms["terminal_error"] += (x[self.idx_terminal_error] - terminal_mean_error) ** 2

        for row in range(self.nrows):
            for robot in range(self.robot_count):
                next_robot = (robot + 1) % self.robot_count
                dx = x[self.idx_state(next_robot, 0, row)] - x[self.idx_state(robot, 0, row)]
                dy = x[self.idx_state(next_robot, 1, row)] - x[self.idx_state(robot, 1, row)]
                terms["edge_distance"] += (x[self.idx_edge_distance(robot, row)] - (dx * dx + dy * dy)) ** 2

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
                    terms["topology"] += (x[self.idx_topology(topo_index, row)] - expr) ** 2
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
                        terms["sfc"] += (x[self.idx_sfc(slack_index)] - expr) ** 2
                        slack_index += 1

        return {key: float(value) for key, value in terms.items()}

    def eval_infeasibility(self, x: np.ndarray) -> float:
        """Evaluate the softened constraints from C++ ``eval_infeasibility``."""
        return float(sum(self.eval_infeasibility_terms(x).values()))

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

    def _project_auxiliary_variables(self, x: np.ndarray) -> np.ndarray:
        """Project non-state slack variables to their best bound-compatible values.

        For fixed states, the optimal C++ auxiliary variables are simply the
        residual expressions clipped to their bounds. This preserves the full
        C++ vector layout while avoiding numerical finite differences over
        thousands of independent slack variables in the reduced Python backend.
        """
        x = np.asarray(x, dtype=float).copy()
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
        x[self.idx_terminal_error] = float(np.clip(terminal_mean_error, 1e-4, 12.0))

        for row in range(self.nrows):
            height = self.height_cons[row + 1]
            edge_lower = self.vvcm.xv2t * self.vvcm.xv2t if height == -1 else 3.0 * height * height
            edge_upper = self.vvcm.xv2 * self.vvcm.xv2
            for robot in range(self.robot_count):
                next_robot = (robot + 1) % self.robot_count
                dx = x[self.idx_state(next_robot, 0, row)] - x[self.idx_state(robot, 0, row)]
                dy = x[self.idx_state(next_robot, 1, row)] - x[self.idx_state(robot, 1, row)]
                x[self.idx_edge_distance(robot, row)] = float(np.clip(dx * dx + dy * dy, edge_lower, edge_upper))

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
                    x[self.idx_topology(topo_index, row)] = max(float(expr), 0.0)
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
                        x[self.idx_sfc(slack_index)] = min(float(expr), 0.0)
                        slack_index += 1
        return x

    def _xy_array_from_vector(self, x: np.ndarray) -> np.ndarray:
        xy = np.zeros((self.nrows, self.robot_count, 2), dtype=float)
        for row in range(self.nrows):
            for robot in range(self.robot_count):
                xy[row, robot, 0] = x[self.idx_state(robot, 0, row)]
                xy[row, robot, 1] = x[self.idx_state(robot, 1, row)]
        return xy

    def _write_xy_array_to_vector(self, x: np.ndarray, xy: np.ndarray) -> np.ndarray:
        out = np.asarray(x, dtype=float).copy()
        for row in range(self.nrows):
            for robot in range(self.robot_count):
                out[self.idx_state(robot, 0, row)] = xy[row, robot, 0]
                out[self.idx_state(robot, 1, row)] = xy[row, robot, 1]
        return out

    def solve_reduced_lsq(self, *, max_nfev: int = 20) -> FormationNLPSolution:
        """State-only least-squares backend for large Python ``SolveFm`` runs.

        This is not a replacement for the C++ IPOPT/ADOL-C backend. It keeps the
        C++ vector layout and residual definitions, but optimizes only XY state
        variables and analytically projects auxiliary slack variables.
        """
        base = self.clipped_initial_guess()
        base_xy = self._xy_array_from_vector(base)
        scale = max(float(np.max(np.abs(base_xy))), 1.0)
        y0 = (base_xy / scale).reshape(-1)

        edge_lowers = np.zeros(self.nrows, dtype=float)
        edge_upper = self.vvcm.xv2 * self.vvcm.xv2
        for row in range(self.nrows):
            height = self.height_cons[row + 1]
            edge_lowers[row] = self.vvcm.xv2t * self.vvcm.xv2t if height == -1 else 3.0 * height * height

        def residual(y: np.ndarray) -> np.ndarray:
            xy = y.reshape(self.nrows, self.robot_count, 2) * scale
            parts: list[float] = []
            # Keep the reduced optimizer near the kinematic coarse guess while
            # allowing enough motion to repair formation/topology violations.
            parts.extend((0.03 * (xy - base_xy)).reshape(-1))
            goal_xy = np.asarray([profile.goal.xy() for profile in self.profile], dtype=float)
            parts.extend((2.0 * (xy[-1] - goal_xy)).reshape(-1))
            if self.nrows > 2:
                parts.extend((0.08 * (xy[2:] - 2.0 * xy[1:-1] + xy[:-2])).reshape(-1))

            for row in range(self.nrows):
                for robot in range(self.robot_count):
                    nxt = (robot + 1) % self.robot_count
                    diff = xy[row, nxt] - xy[row, robot]
                    d2 = float(np.dot(diff, diff))
                    if d2 < edge_lowers[row]:
                        parts.append(0.5 * (edge_lowers[row] - d2))
                    elif d2 > edge_upper:
                        parts.append(0.5 * (d2 - edge_upper))
                    else:
                        parts.append(0.0)

                for robot in range(self.robot_count):
                    xk, yk = xy[row, robot]
                    for p in range(self.robot_count):
                        if p == robot or (p + 1) % self.robot_count == robot:
                            continue
                        next_p = (p + 1) % self.robot_count
                        xp, yp = xy[row, p]
                        xnp, ynp = xy[row, next_p]
                        expr = (xk - xp) * (yp - ynp) + (yk - yp) * (xnp - xp)
                        parts.append(0.35 * min(0.0, float(expr)))

                for robot in range(self.robot_count):
                    theta = base[self.idx_state(robot, 2, row)]
                    for coeff in self.config.vehicle.disc_coefficients:
                        px = xy[row, robot, 0] + coeff * np.cos(theta)
                        py = xy[row, robot, 1] + coeff * np.sin(theta)
                        for halfspace in self.corridor_cons[robot][row]:
                            expr = halfspace[0] * px + halfspace[1] * py - halfspace[2]
                            parts.append(0.5 * max(0.0, float(expr)))
            return np.asarray(parts, dtype=float)

        start = time.perf_counter()
        opt = least_squares(
            residual,
            y0,
            method="trf",
            max_nfev=max(1, max_nfev),
            ftol=1e-6,
            xtol=1e-6,
            gtol=1e-6,
        )
        elapsed = time.perf_counter() - start
        vector = self._write_xy_array_to_vector(base, opt.x.reshape(self.nrows, self.robot_count, 2) * scale)
        vector = self._project_auxiliary_variables(vector)
        return FormationNLPSolution(
            states=self.unpack_solution(vector),
            vector=vector,
            objective=self.eval_objective(vector),
            infeasibility=self.eval_infeasibility(vector),
            infeasibility_terms=self.eval_infeasibility_terms(vector),
            solve_time=elapsed,
            scipy_success=bool(opt.success),
            scipy_message=str(opt.message),
            iterations=int(opt.nfev),
        )

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
        if method == "reduced-lsq":
            return self.solve_reduced_lsq(max_nfev=maxiter)
        if method == "ipopt":
            from . import optimizer_casadi

            return optimizer_casadi.solve_formation_ipopt(self, maxiter=maxiter)
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
            infeasibility_terms=self.eval_infeasibility_terms(vector),
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


def solve(
    profile: Constraints,
    guess: FullStates,
    *,
    config: PlannerConfig | None = None,
    w_inf: float = 1e4,
    method: str = "L-BFGS-B",
    maxiter: int = 200,
) -> SingleRobotNLPSolution:
    """Python equivalent of C++ ``LightweightProblem::Solve``."""
    problem = CarLikeNLPProblem(profile, guess, config=config, w_inf=w_inf)
    return problem.solve(
        start=profile.start,
        goal=profile.goal,
        method=method,
        maxiter=maxiter,
    )


def solve_diff_drive(
    profile: Constraints,
    guess: FullStates,
    pre_sol: FullStates | None = None,
    *,
    config: PlannerConfig | None = None,
    w_inf: float = 1e4,
    method: str = "L-BFGS-B",
    maxiter: int = 200,
) -> SingleRobotNLPSolution:
    """Python equivalent of C++ ``LightweightProblem::Solve_diff_drive``."""
    problem = DiffDriveNLPProblem(profile, guess, pre_sol=pre_sol, config=config, w_inf=w_inf)
    return problem.solve(
        start=guess.states[0],
        goal=guess.states[-1],
        method=method,
        maxiter=maxiter,
    )


def solve_replan(
    profile: Constraints,
    guess: FullStates,
    pre_sol: FullStates | None = None,
    *,
    config: PlannerConfig | None = None,
    w_inf: float = 1e4,
    method: str = "L-BFGS-B",
    maxiter: int = 200,
) -> SingleRobotNLPSolution:
    """Python equivalent of C++ ``LightweightProblem::Solve_replan``."""
    problem = CarLikeReplanNLPProblem(profile, guess, pre_sol=pre_sol, config=config, w_inf=w_inf)
    return problem.solve(
        start=guess.states[0],
        goal=guess.states[-1],
        method=method,
        maxiter=maxiter,
    )
