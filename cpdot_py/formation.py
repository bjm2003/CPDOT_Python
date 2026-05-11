"""Formation generation and simplified trajectory optimization."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import time

import numpy as np

from .coarse_path_planner import CoarsePathPlanner, Pose2D, poses_to_array
from .coarse_path_planner import axis_aligned_box
from .env import CircleObstacle, Map2D, PolygonObstacle, RectangleObstacle
from .forward_kinematics import ForwardKinematics
from .geometry import headings_from_path, resample_polyline
from .optimizer import (
    FormationNLPSolution,
    PlannerConfig,
    SingleRobotNLPSolution,
    VVCMConstants,
    solve as solve_single_robot,
    solve_diff_drive,
    solve_fm,
    solve_replan,
)
from .sfc import generate_sfc
from .states import CPDOT_FORMATION_ROBOTS, Constraints, FullStates, TrajectoryPoint
from .topo_prm import TopologyPRM


def regular_polygon(radius: float, count: int, phase: float = 0.0) -> np.ndarray:
    """Return regular polygon vertices centered at the origin."""
    angles = phase + np.arange(count) * 2.0 * np.pi / count
    return radius * np.column_stack([np.cos(angles), np.sin(angles)])


def normalize_angle(angle: float) -> float:
    """Normalize an angle to ``[-pi, pi)``."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def generate_optimal_time_profile_segment(
    stations: np.ndarray,
    start_time: float = 0.0,
    config: PlannerConfig | None = None,
) -> np.ndarray:
    """Port C++ ``GenerateOptimalTimeProfileSegment``.

    ``stations`` are cumulative path lengths. The returned vector gives the
    timestamp assigned to each station under bounded acceleration and velocity.
    """
    cfg = PlannerConfig() if config is None else config
    stations = np.asarray(stations, dtype=float)
    if stations.ndim != 1 or len(stations) == 0:
        raise ValueError("stations must be a non-empty 1D array")
    if len(stations) == 1:
        return np.asarray([start_time], dtype=float)

    max_accel = cfg.vehicle.max_acceleration
    max_decel = -cfg.vehicle.max_acceleration
    max_velocity = cfg.vehicle.max_velocity
    min_velocity = -cfg.vehicle.max_velocity

    accel_idx = 0
    decel_idx = len(stations) - 1
    velocity = 0.0
    profile = np.zeros(len(stations), dtype=float)
    for i in range(len(stations) - 1):
        ds = max(float(stations[i + 1] - stations[i]), 0.0)
        profile[i] = velocity
        velocity = np.sqrt(max(velocity * velocity + 2.0 * max_accel * ds, 0.0))
        velocity = float(np.clip(velocity, min_velocity, max_velocity))
        if velocity >= max_velocity:
            accel_idx = i + 1
            break

    velocity = 0.0
    for i in range(len(stations) - 1, accel_idx, -1):
        ds = max(float(stations[i] - stations[i - 1]), 0.0)
        profile[i] = velocity
        velocity = np.sqrt(max(velocity * velocity - 2.0 * max_decel * ds, 0.0))
        velocity = float(np.clip(velocity, min_velocity, max_velocity))
        if velocity >= max_velocity:
            decel_idx = i
            break

    profile[accel_idx:decel_idx] = max_velocity
    time_profile = np.full(len(stations), float(start_time), dtype=float)
    for i in range(1, len(stations)):
        ds = float(stations[i] - stations[i - 1])
        if ds <= 1e-9:
            time_profile[i] = time_profile[i - 1]
        else:
            if profile[i] < 1e-6:
                fallback = min(max_velocity, np.sqrt(max(2.0 * max_accel * ds, 0.0)))
                speed = max(0.5 * (profile[i - 1] + profile[i]), fallback, 1e-6)
            else:
                speed = profile[i]
            time_profile[i] = time_profile[i - 1] + ds / speed
    return time_profile


def resample_path_to_full_states(
    path: np.ndarray,
    *,
    step_num: int = 0,
    ratio: bool = False,
    config: PlannerConfig | None = None,
) -> FullStates:
    """Port C++ ``FormationPlanner::ResamplePath`` for standalone paths.

    ``path`` accepts ``Nx2`` points or ``Nx3`` poses ``(x, y, theta)``. The
    output includes interpolated pose, velocity, steering angle, acceleration,
    and steering-rate fields used by ``SolveFm``.
    """
    cfg = PlannerConfig() if config is None else config
    arr = np.asarray(path, dtype=float)
    if arr.ndim != 2 or arr.shape[1] not in (2, 3) or len(arr) < 2:
        raise ValueError("path must have shape Nx2 or Nx3 with at least two samples")

    xy = arr[:, :2]
    theta = np.unwrap(arr[:, 2] if arr.shape[1] == 3 else headings_from_path(xy))
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    stations = np.concatenate([[0.0], np.cumsum(seg)])
    if stations[-1] <= 1e-9:
        return FullStates(tf=0.1, states=[TrajectoryPoint(float(xy[0, 0]), float(xy[0, 1]), float(theta[0]))] * 2)

    gears = np.ones(len(xy), dtype=int)
    for i in range(1, len(xy)):
        tracking_angle = np.arctan2(xy[i, 1] - xy[i - 1, 1], xy[i, 0] - xy[i - 1, 0])
        gears[i] = 1 if abs(normalize_angle(tracking_angle - theta[i])) < np.pi / 2.0 else -1
    gears[0] = gears[1]

    time_profile = np.zeros(len(xy), dtype=float)
    last_idx = 0
    start_time = 0.0
    for i in range(len(gears)):
        if i == len(gears) - 1 or gears[i + 1] != gears[i]:
            local = stations[last_idx : i + 1]
            local = local - local[0]
            profile = generate_optimal_time_profile_segment(local, start_time, cfg)
            time_profile[last_idx : i + 1] = profile
            start_time = float(profile[-1])
            last_idx = i

    total_time = max(float(time_profile[-1]), 0.1)
    nfe = int(step_num) if ratio else max(cfg.min_nfe, int(total_time / cfg.time_step))
    nfe = max(nfe, 2)
    ticks = np.linspace(float(time_profile[0]), total_time, nfe)

    interp_x = np.interp(ticks, time_profile, xy[:, 0])
    interp_y = np.interp(ticks, time_profile, xy[:, 1])
    interp_theta = np.unwrap(np.interp(ticks, time_profile, theta))
    states = [
        TrajectoryPoint(x=float(x), y=float(y), theta=float(th))
        for x, y, th in zip(interp_x, interp_y, interp_theta)
    ]

    dt = max(float(ticks[1] - ticks[0]), 1e-9)
    for i in range(nfe - 1):
        dx = states[i + 1].x - states[i].x
        dy = states[i + 1].y - states[i].y
        tracking_angle = np.arctan2(dy, dx)
        gear = abs(normalize_angle(tracking_angle - states[i].theta)) < np.pi / 2.0
        velocity = float(np.hypot(dx, dy) / dt)
        states[i].v = float(np.clip(velocity if gear else -velocity, -cfg.vehicle.max_velocity, cfg.vehicle.max_velocity))
        denom = states[i].v * dt
        if abs(denom) > 1e-9:
            phi = np.arctan((states[i + 1].theta - states[i].theta) * cfg.vehicle.wheel_base / denom)
            states[i].phi = float(np.clip(phi, -cfg.vehicle.phi_min, cfg.vehicle.phi_max))

    for i in range(nfe - 1):
        states[i].a = float(
            np.clip((states[i + 1].v - states[i].v) / dt, -cfg.vehicle.max_acceleration, cfg.vehicle.max_acceleration)
        )
        states[i].omega = float(
            np.clip((states[i + 1].phi - states[i].phi) / dt, -cfg.vehicle.omega_max, cfg.vehicle.omega_max)
        )
    return FullStates(tf=total_time, states=states)


def full_states_to_xy_tensor(states: list[FullStates]) -> np.ndarray:
    """Convert C++ joint ``FullStates`` to ``T x R x 2``."""
    if not states:
        raise ValueError("states must not be empty")
    nfe = len(states[0].states)
    if any(len(item.states) != nfe for item in states):
        raise ValueError("all robot trajectories must have the same length")
    out = np.zeros((nfe, len(states), 2), dtype=float)
    for robot, full in enumerate(states):
        out[:, robot, :] = full.xy_array()
    return out


def xy_tensor_to_full_states(
    trajectory: np.ndarray,
    *,
    config: PlannerConfig | None = None,
) -> list[FullStates]:
    """Convert ``T x R x 2`` paths to C++ ``FullStates`` guesses."""
    arr = np.asarray(trajectory, dtype=float)
    if arr.ndim != 3 or arr.shape[2] != 2:
        raise ValueError("trajectory must have shape T x R x 2")
    return [
        resample_path_to_full_states(arr[:, robot, :], step_num=arr.shape[0], ratio=True, config=config)
        for robot in range(arr.shape[1])
    ]


def generate_desired_rp(height_cons: np.ndarray, height_cons_set: np.ndarray) -> np.ndarray:
    """Port C++ ``GenerateDesiredRP`` height-to-radius update."""
    vvcm = VVCMConstants()
    out = np.asarray(height_cons_set, dtype=float).copy()
    for i, height in enumerate(np.asarray(height_cons, dtype=float)):
        if height == -1:
            out[i] = -1.0
        elif out[i] == -1:
            out[i] = vvcm.xv2t
        else:
            out[i] += vvcm.radius_inc
    return out


@dataclass
class PlanFmResult:
    """Python result container for the C++ ``Plan_fm`` core loop."""

    states: list[FullStates]
    best_states: list[FullStates] | None
    corridor_cons: list[list[list[list[float]]]]
    height_cons: np.ndarray
    height_cons_set: np.ndarray
    solve_history: list[FormationNLPSolution]
    warm_start: int
    success: bool
    reason: str


@dataclass
class PlanResult:
    """Result container for C++ ``FormationPlanner::Plan`` counterpart."""

    state: FullStates
    guess: FullStates
    solution: SingleRobotNLPSolution | None
    coarse_time: float
    solve_time: float
    infeasibility: float
    success: bool
    reason: str


@dataclass
class ReplanResult:
    """Result container for C++ heterogeneous replan wrappers."""

    state: FullStates
    solution: SingleRobotNLPSolution
    infeasibility: float
    solve_time: float
    max_error: float
    avg_error: float
    success: bool
    reason: str


@dataclass
class FormationPlanner:
    """Plan and optimize a flexible multi-robot formation along a guide path."""

    map2d: Map2D
    robot_count: int = CPDOT_FORMATION_ROBOTS
    formation_radius: float = 4.05 / np.sqrt(3.0)
    sheet_radius: float = 4.05 / np.sqrt(3.0)
    robot_clearance: float = 0.18
    obstacle_weight: float = 160.0
    formation_weight: float = 35.0
    reference_weight: float = 0.45
    smooth_weight: float = 9.0
    bound_weight: float = 300.0

    def __post_init__(self):
        self.desired_offsets = regular_polygon(self.formation_radius, self.robot_count)
        self.sheet_vertices = regular_polygon(self.sheet_radius, self.robot_count)
        self.fk = ForwardKinematics(self.sheet_vertices)
        self.desired_distances = np.linalg.norm(
            self.desired_offsets[:, None, :] - self.desired_offsets[None, :, :], axis=2
        )

    def initial_trajectory(self, guide_path: np.ndarray, steps: int = 45) -> np.ndarray:
        """Lift a center guide path into robot trajectories using rotated offsets."""
        centers = resample_polyline(guide_path, steps)
        headings = headings_from_path(centers)
        traj = np.zeros((steps, self.robot_count, 2), dtype=float)
        for t, (center, theta) in enumerate(zip(centers, headings)):
            rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            traj[t] = center + self.desired_offsets @ rot.T
        return traj

    def generate_guess_from_path(
        self,
        path: list[Pose2D] | np.ndarray,
        start: TrajectoryPoint,
        step_num: int = 0,
        ratio: bool = False,
        config: PlannerConfig | None = None,
    ) -> FullStates:
        """Port C++ ``FormationPlanner::GenerateGuessFromPath``.

        The source prunes a global pose path at the waypoint closest to the
        current start state, then calls ``ResamplePath`` on the remaining path.
        """
        arr = poses_to_array(path) if isinstance(path, list) else np.asarray(path, dtype=float)
        if arr.ndim != 2 or arr.shape[1] not in (2, 3) or len(arr) == 0:
            raise ValueError("path must be a non-empty Nx2 or Nx3 path")
        distances = np.linalg.norm(arr[:, :2] - start.xy(), axis=1)
        closest_index = int(np.argmin(distances))
        pruned = arr[closest_index:]
        if len(pruned) == 1:
            point = pruned[0]
            theta = float(point[2]) if pruned.shape[1] == 3 else start.theta
            pruned = np.vstack([point[:2], point[:2]])
            pruned = np.column_stack([pruned, [theta, theta]])
        return resample_path_to_full_states(pruned, step_num=step_num, ratio=ratio, config=config)

    @staticmethod
    def stitch_previous_solution(solution: FullStates, start: TrajectoryPoint) -> FullStates:
        """Port C++ ``FormationPlanner::StitchPreviousSolution``."""
        if not solution.states:
            return FullStates()
        distances = [float(np.hypot(state.x - start.x, state.y - start.y)) for state in solution.states]
        closest_index = int(np.argmin(distances))
        dt = solution.tf / len(solution.states)
        return FullStates(
            tf=float(solution.tf - closest_index * dt),
            states=list(solution.states[closest_index:]),
        )

    def check_guess_feasibility(
        self,
        guess: FullStates,
        config: PlannerConfig | None = None,
    ) -> bool:
        """Port C++ ``FormationPlanner::CheckGuessFeasibility``.

        The C++ implementation delegates each state to
        ``Environment::CheckPoseCollision``. Here we mirror the existing
        Python Hybrid A* collision model: the vehicle is approximated by its
        configured discs, each checked as an axis-aligned box.
        """
        if not guess.states:
            return False
        cfg = PlannerConfig() if config is None else config
        wh = cfg.vehicle.disc_radius * 2.0
        for state in guess.states:
            discs = cfg.vehicle.disc_positions(state.x, state.y, state.theta)
            for i in range(cfg.vehicle.n_disc):
                center = np.array([discs[2 * i], discs[2 * i + 1]], dtype=float)
                if self.map2d.polygon_collides(axis_aligned_box(center, wh, wh)):
                    return False
        return True

    def _generate_corridor_box(
        self,
        center: np.ndarray,
        radius: float,
        config: PlannerConfig,
    ) -> tuple[float, float, float, float] | None:
        """Port the AABox expansion behavior used by C++ ``GenerateCorridorBox``."""
        center = np.asarray(center, dtype=float)
        ri = float(radius)

        def collides(cx: float, cy: float, xmin_extra: float = 0.0, xmax_extra: float = 0.0,
                     ymin_extra: float = 0.0, ymax_extra: float = 0.0) -> bool:
            box_center = np.array(
                [
                    cx + 0.5 * (xmax_extra - xmin_extra),
                    cy + 0.5 * (ymax_extra - ymin_extra),
                ],
                dtype=float,
            )
            width = 2.0 * ri + xmin_extra + xmax_extra
            height = 2.0 * ri + ymin_extra + ymax_extra
            return self.map2d.polygon_collides(axis_aligned_box(box_center, width, height))

        x, y = float(center[0]), float(center[1])
        if collides(x, y):
            found = False
            inc = 4
            while inc < config.corridor_max_iter:
                iteration = inc // 4
                edge = inc % 4
                real_x, real_y = x, y
                if edge == 0:
                    real_x = x - iteration * 0.05
                elif edge == 1:
                    real_x = x + iteration * 0.05
                elif edge == 2:
                    real_y = y - iteration * 0.05
                else:
                    real_y = y + iteration * 0.05
                inc += 1
                if not collides(real_x, real_y):
                    x, y = real_x, real_y
                    found = True
                    break
            if not found:
                return None

        incremental = [0.0, 0.0, 0.0, 0.0]
        blocked = [False, False, False, False]
        step = ri * 0.2
        inc = 4
        while not all(blocked) and inc < config.corridor_max_iter:
            iteration = inc // 4
            edge = inc % 4
            inc += 1
            if blocked[edge]:
                continue
            incremental[edge] = iteration * step
            if (
                collides(x, y, incremental[0], incremental[1], incremental[2], incremental[3])
                or incremental[edge] >= config.corridor_incremental_limit
            ):
                incremental[edge] -= step
                blocked[edge] = True
        if inc >= config.corridor_max_iter:
            return None
        return x - incremental[0], y - incremental[2], x + incremental[1], y + incremental[3]

    def _build_vertex_corridor_constraints(
        self,
        guess: FullStates,
        config: PlannerConfig,
        *,
        radius: float = 0.3,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Build C++ ``Plan`` style repeated vertex corridor bounds."""
        vertices_nvar = config.vehicle.vertices * 2
        lb = np.full((len(guess.states), vertices_nvar), -np.inf, dtype=float)
        ub = np.full((len(guess.states), vertices_nvar), np.inf, dtype=float)
        for row, state in enumerate(guess.states):
            centre = config.vehicle.formation_centre(state.x, state.y, state.theta)
            box = self._generate_corridor_box(centre, radius, config)
            if box is None:
                return None
            xmin, ymin, xmax, ymax = box
            for vertex in range(config.vehicle.vertices):
                lb[row, 2 * vertex] = xmin
                lb[row, 2 * vertex + 1] = ymin
                ub[row, 2 * vertex] = xmax
                ub[row, 2 * vertex + 1] = ymax
        return lb, ub

    def plan_single(
        self,
        prev_sol: FullStates,
        start: TrajectoryPoint,
        goal: TrajectoryPoint,
        *,
        config: PlannerConfig | None = None,
        hyperparam_set: list[list[list[float]]] | None = None,
        max_search_time: float = 30.0,
        max_expansions: int = 200000,
        solver_maxiter: int = 200,
        solver_method: str = "L-BFGS-B",
    ) -> PlanResult:
        """Reproduce C++ ``FormationPlanner::Plan`` without ROS visualization."""
        cfg = PlannerConfig() if config is None else config
        hyperparam_set = hyperparam_set or []
        guess = self.stitch_previous_solution(prev_sol, start)
        coarse_time = 0.0
        if len(guess.states) < 3 or not self.check_guess_feasibility(guess, cfg):
            coarse_start = time.perf_counter()
            planner = CoarsePathPlanner(
                self.map2d,
                cfg,
                max_search_time=max_search_time,
                max_expansions=max_expansions,
            )
            path = planner.plan(Pose2D(start.x, start.y, start.theta), Pose2D(goal.x, goal.y, goal.theta), hyperparam_set)
            coarse_time = time.perf_counter() - coarse_start
            if not path:
                return PlanResult(guess, guess, None, coarse_time, 0.0, np.inf, False, "coarse_path_failed")
            guess = self.generate_guess_from_path(path, start, config=cfg)

        corridor = self._build_vertex_corridor_constraints(guess, cfg)
        if corridor is None:
            return PlanResult(guess, guess, None, coarse_time, 0.0, np.inf, False, "corridor_box_failed")
        corridor_lb, corridor_ub = corridor
        constraints = Constraints(start=start, goal=goal, corridor_lb=corridor_lb, corridor_ub=corridor_ub)
        solution = solve_single_robot(
            constraints,
            guess,
            config=cfg,
            w_inf=cfg.opti_w_penalty0,
            maxiter=solver_maxiter,
            method=solver_method,
        )
        reason = (
            "infeasibility_above_tolerance"
            if solution.infeasibility > cfg.opti_varepsilon_tol
            else "completed"
        )
        return PlanResult(
            solution.state,
            guess,
            solution,
            coarse_time,
            solution.solve_time,
            solution.infeasibility,
            True,
            reason,
        )

    @staticmethod
    def _tracking_errors(reference: FullStates, result: FullStates) -> tuple[float, float]:
        errors = [
            float(np.hypot(ref.x - state.x, ref.y - state.y))
            for ref, state in zip(reference.states, result.states)
        ]
        if not errors:
            return 0.0, 0.0
        return max(errors), float(sum(errors) / len(errors))

    def plan_diff_drive(
        self,
        guess: FullStates,
        prev_sol: FullStates,
        start: TrajectoryPoint,
        goal: TrajectoryPoint,
        *,
        config: PlannerConfig | None = None,
        solver_maxiter: int = 200,
        solver_method: str = "L-BFGS-B",
    ) -> ReplanResult:
        """Reproduce C++ ``FormationPlanner::Plan_diff_drive`` wrapper."""
        cfg = PlannerConfig() if config is None else config
        constraints = Constraints(start=start, goal=goal)
        solution = solve_diff_drive(
            constraints,
            guess,
            prev_sol,
            config=cfg,
            w_inf=cfg.opti_w_penalty0,
            maxiter=solver_maxiter,
            method=solver_method,
        )
        max_error, avg_error = self._tracking_errors(guess, solution.state)
        reason = (
            "infeasibility_above_tolerance"
            if solution.infeasibility > cfg.opti_varepsilon_tol
            else "completed"
        )
        return ReplanResult(
            solution.state,
            solution,
            solution.infeasibility,
            solution.solve_time,
            max_error,
            avg_error,
            True,
            reason,
        )

    def plan_car_like_replan(
        self,
        guess: FullStates,
        prev_sol: FullStates,
        *,
        config: PlannerConfig | None = None,
        solver_maxiter: int = 200,
        solver_method: str = "L-BFGS-B",
    ) -> ReplanResult:
        """Reproduce C++ ``FormationPlanner::Plan_car_like_replan`` wrapper."""
        cfg = PlannerConfig() if config is None else config
        constraints = Constraints(start=guess.states[0], goal=guess.states[-1])
        solution = solve_replan(
            constraints,
            guess,
            prev_sol,
            config=cfg,
            w_inf=cfg.opti_w_penalty0,
            maxiter=solver_maxiter,
            method=solver_method,
        )
        max_error, avg_error = self._tracking_errors(guess, solution.state)
        if solution.infeasibility > 0.1:
            reason = "trajectory_needs_refinement"
        elif solution.infeasibility > cfg.opti_varepsilon_tol:
            reason = "infeasibility_above_tolerance"
        else:
            reason = "completed"
        return ReplanResult(
            solution.state,
            solution,
            solution.infeasibility,
            solution.solve_time,
            max_error,
            avg_error,
            True,
            reason,
        )

    @staticmethod
    def plan_car_like(
        traj_lead: FullStates,
        offset: float,
        config: PlannerConfig | None = None,
    ) -> FullStates:
        """Port C++ ``FormationPlanner::Plan_car_like`` follower generation."""
        cfg = PlannerConfig() if config is None else config
        if not traj_lead.states:
            return FullStates(tf=traj_lead.tf)
        dt = traj_lead.tf / len(traj_lead.states)
        follower_states: list[TrajectoryPoint] = []
        for lead in traj_lead.states:
            follower_states.append(
                TrajectoryPoint(
                    x=float(lead.x + offset * np.sin(lead.theta)),
                    y=float(lead.y - offset * np.cos(lead.theta)),
                    theta=lead.theta,
                    v=float((1.0 + offset * np.tan(lead.phi) / cfg.vehicle.wheel_base) * lead.v),
                    phi=float(
                        np.arctan(
                            cfg.vehicle.wheel_base
                            * np.tan(lead.phi)
                            / (cfg.vehicle.wheel_base + offset * lead.phi)
                        )
                    ),
                )
            )
        for i in range(1, len(follower_states) - 1):
            follower_states[i].a = float((traj_lead.states[i].v - traj_lead.states[i - 1].v) / dt)
            follower_states[i].omega = float((traj_lead.states[i].phi - traj_lead.states[i - 1].phi) / dt)
        follower_states[0].a = traj_lead.states[0].a
        follower_states[0].omega = traj_lead.states[0].omega
        follower_states[-1].a = traj_lead.states[-1].a
        follower_states[-1].omega = traj_lead.states[-1].omega
        return FullStates(tf=traj_lead.tf, states=follower_states)

    @staticmethod
    def check_car_kinematic(
        current_result: FullStates,
        offset_car: list[float],
        config: PlannerConfig | None = None,
    ) -> bool:
        """Port C++ ``FormationPlanner::CheckCarKinematic`` velocity check."""
        cfg = PlannerConfig() if config is None else config
        for offset in offset_car:
            for state in current_result.states:
                if state.phi >= 0.0:
                    current_ref_vel = state.v * (1.0 + offset * state.phi / cfg.vehicle.wheel_base)
                    if current_ref_vel > cfg.vehicle.max_velocity:
                        return False
        return True

    @staticmethod
    def check_diff_drive_kinematic(
        current_result: FullStates,
        offset: list[float],
        re_phi: list[float],
        config: PlannerConfig | None = None,
    ) -> bool:
        """Port C++ ``FormationPlanner::CheckDiffDriveKinematic``."""
        cfg = PlannerConfig() if config is None else config
        if len(offset) != len(re_phi):
            raise ValueError("offset and re_phi must have the same length")
        if len(current_result.states) < 2:
            return True
        dt = current_result.tf / len(current_result.states)
        for offset_value, re_phi_value in zip(offset, re_phi):
            for i in range(1, len(current_result.states)):
                prev = current_result.states[i - 1]
                curr = current_result.states[i]
                prev_point = np.array(
                    [
                        prev.x + offset_value * np.cos(prev.theta - re_phi_value),
                        prev.y + offset_value * np.sin(prev.theta - re_phi_value),
                    ],
                    dtype=float,
                )
                curr_point = np.array(
                    [
                        curr.x + offset_value * np.cos(curr.theta - re_phi_value),
                        curr.y + offset_value * np.sin(curr.theta - re_phi_value),
                    ],
                    dtype=float,
                )
                if float(np.linalg.norm(curr_point - prev_point) / dt) > cfg.vehicle.max_velocity:
                    return False
        return True

    def generate_corridor_constraints(
        self,
        guess: list[FullStates],
        *,
        bbox_width: float = 5.0,
    ) -> list[list[list[list[float]]]]:
        """Port the per-robot ``env_->generateSFC`` call from C++ ``Plan_fm``."""
        corridors: list[list[list[list[float]]]] = []
        for full in guess:
            path = full.xy_array()
            hpolys, _, _ = generate_sfc(path, self.map2d, bbox_width=bbox_width)
            corridors.append(hpolys)
        return corridors

    def plan_coarse_full_states(
        self,
        start_set: list[TrajectoryPoint],
        goal_set: list[TrajectoryPoint],
        *,
        hyperparam_sets: list[list[list[list[float]]]] | None = None,
        config: PlannerConfig | None = None,
        max_search_time: float = 30.0,
        max_expansions: int = 200000,
        enable_oneshot: bool = False,
    ) -> list[FullStates]:
        """Port the coarse-guess generation block in C++ ``Plan_fm``."""
        cfg = PlannerConfig() if config is None else config
        if len(start_set) != len(goal_set):
            raise ValueError("start_set and goal_set must have the same length")
        if hyperparam_sets is None:
            hyperparam_sets = [[] for _ in start_set]
        if len(hyperparam_sets) != len(start_set):
            raise ValueError("hyperparam_sets must match robot count")

        initial_plan_set: list[np.ndarray] = []
        guess: list[FullStates] = []
        for start, goal, hyper in zip(start_set, goal_set, hyperparam_sets):
            planner = CoarsePathPlanner(
                self.map2d,
                cfg,
                enable_oneshot=enable_oneshot,
                max_search_time=max_search_time,
                max_expansions=max_expansions,
            )
            path = planner.plan(
                Pose2D(start.x, start.y, start.theta),
                Pose2D(goal.x, goal.y, goal.theta),
                hyper,
            )
            if not path:
                raise RuntimeError("coarse path planner failed")
            path_array = poses_to_array(path)
            initial_plan_set.append(path_array)
            guess.append(resample_path_to_full_states(path_array, config=cfg))

        tf_max = max(full.tf for full in guess)
        tf_max_ind = int(np.argmax([full.tf for full in guess]))
        target_steps = len(guess[tf_max_ind].states)
        aligned: list[FullStates] = []
        for i, full in enumerate(guess):
            if i == tf_max_ind:
                aligned_full = full
            else:
                aligned_full = resample_path_to_full_states(
                    initial_plan_set[i],
                    step_num=target_steps,
                    ratio=True,
                    config=cfg,
                )
                ratio_value = aligned_full.tf / max(tf_max, 1e-9)
                for state in aligned_full.states:
                    state.v *= ratio_value
                    state.a *= ratio_value
                    state.omega *= ratio_value
            aligned_full.tf = tf_max
            aligned.append(aligned_full)
        return aligned

    def generate_height_cons(self, guess: list[FullStates]) -> np.ndarray:
        """Port C++ ``FormationPlanner::GenerateHeightCons``."""
        trajectory = full_states_to_xy_tensor(guess)
        return self.obstacle_height_constraints(trajectory)

    def derive_heights_from_full_states(self, states: list[FullStates]) -> np.ndarray:
        """Port C++ ``DeriveHeight`` for joint ``FullStates``."""
        return self.derive_heights(full_states_to_xy_tensor(states))

    def check_height_cons(self, states: list[FullStates], height_cons: np.ndarray) -> tuple[bool, np.ndarray]:
        """Port C++ ``CheckHeightCons``."""
        heights = self.derive_heights_from_full_states(states)
        for i, constraint in enumerate(np.asarray(height_cons, dtype=float)):
            if constraint != -1 and (not np.isfinite(heights[i]) or constraint >= heights[i]):
                return False, heights
        return True, heights

    def plan_fm_from_guess(
        self,
        guess: list[FullStates],
        *,
        profile: list[Constraints] | None = None,
        config: PlannerConfig | None = None,
        bbox_width: float = 5.0,
        max_warm_start: int = 15,
        initial_warm_starts: int = 5,
        solver_maxiter: int = 200,
        solver_method: str = "L-BFGS-B",
        enforce_cpp_early_return: bool = False,
    ) -> PlanFmResult:
        """Reproduce the core C++ ``Plan_fm`` loop from existing guesses.

        Coarse path generation remains outside this function, just as the C++
        method delegates it to ``coarse_path_planner_`` before building safe
        corridors and solving ``SolveFm``.
        """
        if not guess:
            raise ValueError("guess must not be empty")
        cfg = PlannerConfig() if config is None else config
        if profile is None:
            profile = [Constraints(start=full.states[0], goal=full.states[-1]) for full in guess]
        corridor_cons = self.generate_corridor_constraints(guess, bbox_width=bbox_width)
        height_cons = self.generate_height_cons(guess)
        vvcm = VVCMConstants()
        height_cons_set = np.full(len(guess[0].states), vvcm.xv2t, dtype=float)
        result = guess
        current_guess = guess
        best: list[FullStates] | None = None
        history: list[FormationNLPSolution] = []
        warm_start = 0

        while warm_start < max_warm_start:
            if warm_start < initial_warm_starts:
                solution = solve_fm(
                    profile,
                    current_guess,
                    config=cfg,
                    corridor_cons=corridor_cons,
                    height_cons=height_cons_set.tolist(),
                    w_inf=cfg.opti_w_penalty0,
                    method=solver_method,
                    maxiter=solver_maxiter,
                )
                history.append(solution)
                result = solution.states
                if solution.infeasibility > 1.0:
                    return PlanFmResult(
                        result,
                        best,
                        corridor_cons,
                        height_cons,
                        height_cons_set,
                        history,
                        warm_start,
                        False,
                        "infeasibility_above_cpp_initial_threshold",
                    )
                current_guess = result
                height_cons = self.generate_height_cons(result)
                if best is None or best[0].tf > result[0].tf:
                    best = result
                warm_start += 1
                continue

            if enforce_cpp_early_return and warm_start == initial_warm_starts:
                return PlanFmResult(
                    best or result,
                    best,
                    corridor_cons,
                    height_cons,
                    height_cons_set,
                    history,
                    warm_start,
                    False,
                    "cpp_source_returns_false_at_warm_start_5",
                )

            height_ok, _ = self.check_height_cons(result, height_cons)
            if not height_ok:
                height_cons = self.generate_height_cons(current_guess)
                height_cons_set = generate_desired_rp(height_cons, height_cons_set)
            elif best is None or best[0].tf > result[0].tf:
                best = result

            solution = solve_fm(
                profile,
                current_guess,
                config=cfg,
                corridor_cons=corridor_cons,
                height_cons=height_cons_set.tolist(),
                w_inf=cfg.opti_w_penalty0,
                method=solver_method,
                maxiter=solver_maxiter,
            )
            history.append(solution)
            result = solution.states
            current_guess = result
            finite_radius = height_cons_set[height_cons_set != -1]
            if len(finite_radius) and np.max(finite_radius) > vvcm.formation_radius:
                return PlanFmResult(
                    result,
                    best,
                    corridor_cons,
                    height_cons,
                    height_cons_set,
                    history,
                    warm_start,
                    False,
                    "height_radius_exceeds_formation_radius",
                )
            if solution.infeasibility > 0.5:
                return PlanFmResult(
                    result,
                    best,
                    corridor_cons,
                    height_cons,
                    height_cons_set,
                    history,
                    warm_start,
                    False,
                    "infeasibility_above_cpp_refinement_threshold",
                )
            warm_start += 1

        return PlanFmResult(
            best or result,
            best,
            corridor_cons,
            height_cons,
            height_cons_set,
            history,
            warm_start,
            best is not None,
            "completed_warm_start_loop",
        )

    def plan_individual_trajectories(
        self,
        reference: np.ndarray,
        *,
        max_samples: int = 400,
        clearance: float | None = None,
        resolution: float = 0.45,
        seed: int = 31,
        return_paths: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, list[list[np.ndarray]]]:
        """Plan a collision-free seed path for each robot.

        The lifted formation path is a useful shape reference, but it can place
        individual robots inside obstacles when the centerline passes through a
        tight homotopy class. Seeding each robot with its own PRM path gives the
        smoother a feasible trajectory to refine instead of asking it to escape
        from collisions.
        """
        reference = np.asarray(reference, dtype=float)
        steps = reference.shape[0]
        clearance = self.robot_clearance if clearance is None else clearance
        diag = float(np.hypot(self.map2d.width, self.map2d.height))
        sample_inflate = (0.65 * diag, 0.45 * diag)
        trajectories = []
        topo_paths_set: list[list[np.ndarray]] = []
        for robot in range(self.robot_count):
            start = reference[0, robot]
            goal = reference[-1, robot]
            if self.map2d.segment_is_collision_free(start, goal, clearance):
                paths = [np.asarray([start, goal], dtype=float)]
            else:
                visibility_path = self._visibility_graph_path(start, goal, clearance)
                paths = [visibility_path] if visibility_path is not None else []
                for attempt in range(3 if not paths else 0):
                    prm = TopologyPRM(
                        self.map2d,
                        max_samples=max_samples * (2**attempt),
                        sample_inflate=sample_inflate,
                        clearance=clearance,
                        resolution=resolution,
                        max_raw_paths=12,
                        reserve_num=4,
                        seed=seed + robot + 97 * attempt,
                    )
                    paths = prm.find_topo_paths(start, goal)
                    if paths:
                        break
            topo_paths_set.append(paths)
            if paths:
                chosen = min(paths, key=lambda path: self._path_score(path, reference[:, robot]))
                trajectories.append(resample_polyline(chosen, steps))
            else:
                trajectories.append(reference[:, robot])
        result = np.stack(trajectories, axis=1)
        if return_paths:
            return result, topo_paths_set
        return result

    def _visibility_graph_path(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        clearance: float,
    ) -> np.ndarray | None:
        """Find a short collision-free seed path through obstacle-corner waypoints."""
        waypoints = [np.asarray(start, dtype=float), np.asarray(goal, dtype=float)]
        offset = clearance + 0.35
        for obs in self.map2d.obstacles:
            if isinstance(obs, CircleObstacle):
                angles = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
                candidates = obs.center + (obs.radius + offset) * np.column_stack(
                    [np.cos(angles), np.sin(angles)]
                )
            elif isinstance(obs, (PolygonObstacle, RectangleObstacle)):
                polygon = obs.polygon()
                center = polygon.mean(axis=0)
                directions = polygon - center
                norms = np.linalg.norm(directions, axis=1, keepdims=True)
                candidates = polygon + offset * directions / np.maximum(norms, 1e-9)
            else:
                continue
            for point in candidates:
                if not self.map2d.is_collision(point, clearance):
                    waypoints.append(np.asarray(point, dtype=float))

        n = len(waypoints)
        adjacency: list[list[tuple[float, int]]] = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if self.map2d.segment_is_collision_free(waypoints[i], waypoints[j], clearance):
                    length = float(np.linalg.norm(waypoints[i] - waypoints[j]))
                    adjacency[i].append((length, j))
                    adjacency[j].append((length, i))

        dist = [np.inf] * n
        parent = [-1] * n
        dist[0] = 0.0
        heap = [(0.0, 0)]
        while heap:
            current_dist, node = heapq.heappop(heap)
            if current_dist > dist[node]:
                continue
            if node == 1:
                break
            for edge_len, nb in adjacency[node]:
                candidate = current_dist + edge_len
                if candidate < dist[nb]:
                    dist[nb] = candidate
                    parent[nb] = node
                    heapq.heappush(heap, (candidate, nb))

        if not np.isfinite(dist[1]):
            return None
        path_ids = []
        node = 1
        while node != -1:
            path_ids.append(node)
            node = parent[node]
        path_ids.reverse()
        return np.asarray([waypoints[i] for i in path_ids], dtype=float)

    def optimize(self, initial: np.ndarray, maxiter: int = 180) -> np.ndarray:
        """Smooth paths with explicit potential-field updates.

        This is the portable counterpart of the C++ Ipopt optimization. It uses
        the same practical ingredients, but updates positions directly instead
        of solving the full optimal-control NLP.
        """
        traj = np.asarray(initial, dtype=float).copy()
        ref = traj.copy()
        lr = 0.003
        max_step = 0.08
        for _ in range(maxiter):
            old = traj.copy()
            grad = np.zeros_like(traj)

            grad[1:-1] += self.reference_weight * (traj[1:-1] - ref[1:-1])
            grad[1:-1] += self.smooth_weight * (
                2.0 * traj[1:-1] - traj[:-2] - traj[2:]
            )

            for t in range(1, len(traj) - 1):
                points = traj[t]
                for i in range(self.robot_count):
                    for j in range(i + 1, self.robot_count):
                        delta = points[i] - points[j]
                        dist = float(np.linalg.norm(delta))
                        if dist < 1e-9:
                            continue
                        err = dist - self.desired_distances[i, j]
                        g = self.formation_weight * err * delta / dist
                        grad[t, i] += g
                        grad[t, j] -= g
                for i in range(self.robot_count):
                    clearance, direction = self.clearance_gradient(points[i])
                    violation = self.robot_clearance - clearance
                    if violation > 0:
                        grad[t, i] -= self.obstacle_weight * violation * direction

            step = -lr * grad
            norms = np.linalg.norm(step[1:-1], axis=2, keepdims=True)
            scale = np.minimum(1.0, max_step / np.maximum(norms, 1e-9))
            step[1:-1] *= scale

            for t in range(1, len(traj) - 1):
                for i in range(self.robot_count):
                    proposed = old[t, i] + step[t, i]
                    traj[t, i] = self._accepted_step(old, proposed, t, i)
        return traj

    def _accepted_step(self, old: np.ndarray, proposed: np.ndarray, t: int, robot: int) -> np.ndarray:
        """Backtrack a robot update until point and adjacent motion segments are safe."""
        base = old[t, robot]
        delta = proposed - base
        for scale in (1.0, 0.5, 0.25, 0.125):
            candidate = base + scale * delta
            if self._motion_is_safe(old, candidate, t, robot):
                return candidate
        return base

    def _motion_is_safe(self, old: np.ndarray, point: np.ndarray, t: int, robot: int) -> bool:
        if self.map2d.is_collision(point, self.robot_clearance):
            return False
        previous = old[t - 1, robot]
        following = old[t + 1, robot]
        clearance = self.robot_clearance
        return (
            self.map2d.segment_is_collision_free(previous, point, clearance)
            and self.map2d.segment_is_collision_free(point, following, clearance)
        )

    def _path_score(self, path: np.ndarray, reference: np.ndarray) -> float:
        sampled = resample_polyline(path, len(reference))
        length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
        deviation = float(np.linalg.norm(sampled - reference, axis=1).mean())
        return length + 0.4 * deviation

    def clearance_gradient(self, point: np.ndarray) -> tuple[float, np.ndarray]:
        """Approximate signed clearance and outward gradient at a point."""
        candidates: list[tuple[float, np.ndarray]] = []
        x, y = point
        candidates.extend(
            [
                (x, np.array([1.0, 0.0])),
                (y, np.array([0.0, 1.0])),
                (self.map2d.width - x, np.array([-1.0, 0.0])),
                (self.map2d.height - y, np.array([0.0, -1.0])),
            ]
        )
        for obs in self.map2d.obstacles:
            if isinstance(obs, CircleObstacle):
                vec = point - obs.center
                norm = float(np.linalg.norm(vec))
                direction = vec / max(norm, 1e-9)
                candidates.append((norm - obs.radius, direction))
            elif isinstance(obs, PolygonObstacle):
                poly = obs.polygon()
                center = poly.mean(axis=0)
                best_dist = np.inf
                best_dir = point - center
                for a, b in zip(poly, np.roll(poly, -1, axis=0)):
                    ab = b - a
                    tau = np.clip(np.dot(point - a, ab) / max(np.dot(ab, ab), 1e-9), 0.0, 1.0)
                    nearest = a + tau * ab
                    vec = point - nearest
                    dist = float(np.linalg.norm(vec))
                    if dist < best_dist:
                        best_dist = dist
                        best_dir = vec
                norm = float(np.linalg.norm(best_dir))
                direction = best_dir / max(norm, 1e-9)
                signed = obs.distance(point)
                candidates.append((signed, direction))
        return min(candidates, key=lambda item: item[0])

    def obstacle_penalty(self, points: np.ndarray) -> float:
        """Quadratic penalty for robot-obstacle and formation-polygon collisions."""
        penalty = 0.0
        for point in points:
            clearance = self.map2d.clearance(point)
            violation = self.robot_clearance - clearance
            if violation > 0:
                penalty += self.obstacle_weight * violation * violation
        if self.map2d.polygon_collides(points):
            penalty += self.obstacle_weight * 2.0
        return float(penalty)

    def bound_penalty(self, points: np.ndarray) -> float:
        penalty = 0.0
        for x, y in points:
            for v in (-x, -y, x - self.map2d.width, y - self.map2d.height):
                if v > 0:
                    penalty += self.bound_weight * v * v
        return float(penalty)

    def derive_heights(self, trajectory: np.ndarray) -> np.ndarray:
        """Compute minimum feasible object height along a formation trajectory."""
        heights = []
        for points in trajectory:
            solutions = self.fk.solve(points)
            if not solutions:
                heights.append(np.nan)
            else:
                heights.append(min(float(s["object_xyz"][2]) for s in solutions))
        return np.asarray(heights)

    def obstacle_height_constraints(self, trajectory: np.ndarray) -> np.ndarray:
        """Map obstacle intersections of the robot polygon to height constraints."""
        constraints = []
        for points in trajectory:
            height = self.map2d.obstacle_height_under_polygon(points)
            constraints.append(-1.0 if height is None else float(height))
        return np.asarray(constraints)
