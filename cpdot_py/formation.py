"""Formation generation and simplified trajectory optimization."""

from __future__ import annotations

from dataclasses import dataclass
import heapq

import numpy as np

from .env import CircleObstacle, Map2D, PolygonObstacle, RectangleObstacle
from .forward_kinematics import ForwardKinematics
from .geometry import headings_from_path, resample_polyline
from .optimizer import FormationNLPSolution, PlannerConfig, VVCMConstants, solve_fm
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
