"""Hybrid A* coarse path planner ported from CPDOT C++."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import time

import numpy as np

from .env import Map2D
from .geometry import polygons_intersect
from .optimizer import PlannerConfig

INF = float("inf")
UNDEFINED_INDEX = -1


def normalize_angle(angle: float) -> float:
    """Match C++ ``math::NormalizeAngle`` behavior."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


@dataclass
class Pose2D:
    """Python counterpart of C++ ``math::Pose``."""

    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0

    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)

    def distance_to(self, other: "Pose2D | np.ndarray") -> float:
        point = other.xy() if isinstance(other, Pose2D) else np.asarray(other, dtype=float)
        return float(np.hypot(self.x - point[0], self.y - point[1]))


@dataclass
class Node3D:
    """C++ ``CoarsePathPlanner::Node3d`` equivalent."""

    pose: Pose2D
    origin: np.ndarray
    config: PlannerConfig
    is_forward: bool = True
    is_closed: bool = False
    steering: float = 0.0
    pre_index: int = UNDEFINED_INDEX
    g_cost: float = INF
    f_cost: float = INF

    def __post_init__(self) -> None:
        self.x_grid = int(np.floor((self.pose.x - self.origin[0]) / self.config.xy_resolution))
        self.y_grid = int(np.floor((self.pose.y - self.origin[1]) / self.config.xy_resolution))
        self.theta_grid = int(np.floor((self.pose.theta - (-np.pi)) / self.config.theta_resolution))
        self.index = (self.x_grid, self.y_grid, self.theta_grid)

    def set_cost(self, g_cost: float, h_cost: float) -> None:
        self.g_cost = float(g_cost)
        self.f_cost = float(g_cost + h_cost)


@dataclass
class Node2D:
    """C++ ``CoarsePathPlanner::Node2d`` equivalent."""

    x_grid: int
    y_grid: int
    pre_index: tuple[int, int] | None = None
    f_cost: float = INF
    is_closed: bool = False

    @classmethod
    def from_pose(cls, pose: Pose2D, origin: np.ndarray, config: PlannerConfig) -> "Node2D":
        return cls(
            int(np.floor((pose.x - origin[0]) / config.grid_xy_resolution)),
            int(np.floor((pose.y - origin[1]) / config.grid_xy_resolution)),
        )

    @property
    def index(self) -> tuple[int, int]:
        return (self.x_grid, self.y_grid)

    def box(self, origin: np.ndarray, config: PlannerConfig) -> np.ndarray:
        center = np.array(
            [
                origin[0] + config.grid_xy_resolution * self.x_grid,
                origin[1] + config.grid_xy_resolution * self.y_grid,
            ],
            dtype=float,
        )
        half = config.vehicle.disc_radius
        return axis_aligned_box(center, 2.0 * half, 2.0 * half)


def axis_aligned_box(center: np.ndarray, width: float, height: float) -> np.ndarray:
    """Return axis-aligned box vertices."""
    cx, cy = center
    hw, hh = width / 2.0, height / 2.0
    return np.array(
        [
            [cx - hw, cy - hh],
            [cx + hw, cy - hh],
            [cx + hw, cy + hh],
            [cx - hw, cy + hh],
        ],
        dtype=float,
    )


class CoarsePathPlanner:
    """Port of C++ ``formation_planner::CoarsePathPlanner``.

    The Hybrid A* search, 2D DP heuristic, kinematic expansion, costs, collision
    checks, and homotopy constraints follow the C++ code. The OMPL
    Dubins/Reeds-Shepp one-shot connector is exposed as an explicit unsupported
    hook unless a caller supplies a compatible implementation.
    """

    grid_directions = (
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    )
    grid_direction_costs = (
        np.sqrt(2.0),
        1.0,
        np.sqrt(2.0),
        1.0,
        1.0,
        np.sqrt(2.0),
        1.0,
        np.sqrt(2.0),
    )

    def __init__(
        self,
        map2d: Map2D,
        config: PlannerConfig | None = None,
        *,
        enable_oneshot: bool = False,
        max_search_time: float = 30.0,
        max_expansions: int = 200000,
    ):
        self.map = map2d
        self.config = PlannerConfig() if config is None else config
        self.enable_oneshot = enable_oneshot
        self.max_search_time = max_search_time
        self.max_expansions = max_expansions
        self.origin = np.zeros(2, dtype=float)
        self.is_forward_only = False
        self.forward_num = 1
        self.open_heap: list[tuple[float, int, tuple[int, int, int]]] = []
        self.open_set: dict[tuple[int, int, int], Node3D] = {}
        self.grid_heap: list[tuple[float, int, tuple[int, int]]] = []
        self.grid_open_set: dict[tuple[int, int], Node2D] = {}
        self._push_counter = 0

    def plan(
        self,
        start: Pose2D,
        goal: Pose2D,
        hyperparam_set: list[list[list[float]]] | None = None,
    ) -> list[Pose2D]:
        """Port C++ ``Plan``."""
        hyperparam_set = [] if hyperparam_set is None else hyperparam_set
        self.origin = 0.5 * (start.xy() + goal.xy())
        self.is_forward_only = self.config.vehicle.min_velocity >= 0.0
        forward_distance = self.config.xy_resolution * np.sqrt(2.0)
        self.forward_num = max(1, int(np.ceil(forward_distance / self.config.step_size)))

        self.grid_heap = []
        self.grid_open_set = {}
        grid_goal = Node2D.from_pose(goal, self.origin, self.config)
        grid_goal.f_cost = 0.0
        self._push_grid(grid_goal)
        self.grid_open_set[grid_goal.index] = grid_goal

        self.open_heap = []
        self.open_set = {}
        start_node = Node3D(start, self.origin, self.config)
        goal_node = Node3D(goal, self.origin, self.config)
        start_node.set_cost(0.0, self.estimate_heuristic_cost(start_node))
        self._push_open(start_node)
        self.open_set[start_node.index] = start_node

        dist_start_to_goal = max(start.distance_to(goal), 1e-9)
        oneshot_index: tuple[int, int, int] | None = None
        oneshot_path: list[Pose2D] = []
        walked_node_count = 0
        started = time.perf_counter()

        while self.open_heap:
            if time.perf_counter() - started > self.max_search_time:
                return []
            if walked_node_count > self.max_expansions:
                return []

            _, _, node_index = heapq.heappop(self.open_heap)
            node = self.open_set[node_index]
            if node.is_closed:
                continue
            node.is_closed = True

            if node_index == goal_node.index:
                goal_node.pre_index = node.pre_index
                break

            walked_node_count += 1
            scaled_heu_cost = (node.f_cost - node.g_cost) / dist_start_to_goal
            oneshot_freq = int(2 + scaled_heu_cost * (100 - 2))
            should_check = oneshot_freq < 1 or walked_node_count % max(oneshot_freq, 1) == 0
            if self.enable_oneshot and should_check:
                path = self.check_oneshot_path(node, goal_node, hyperparam_set)
                if path:
                    oneshot_index = node.index
                    oneshot_path = path
                    break

            next_node_num = self.config.next_node_num // 2 if self.is_forward_only else self.config.next_node_num
            for i in range(next_node_num):
                next_node = self.expand_next_node(node, i, hyperparam_set)
                if next_node is None:
                    continue
                opened = self.open_set.get(next_node.index)
                if opened is not None and opened.is_closed:
                    continue
                next_node.set_cost(
                    node.g_cost + self.evaluate_expand_cost(node, next_node),
                    self.estimate_heuristic_cost(next_node),
                )
                if opened is None:
                    self.open_set[next_node.index] = next_node
                    self._push_open(next_node)
                elif next_node.g_cost < opened.g_cost:
                    opened.g_cost = next_node.g_cost
                    opened.f_cost = next_node.f_cost
                    opened.pre_index = node.index
                    opened.is_forward = next_node.is_forward
                    opened.steering = next_node.steering
                    self._push_open(opened)

        if oneshot_index is not None:
            return self.traverse_path(oneshot_index) + oneshot_path
        if goal_node.pre_index != UNDEFINED_INDEX:
            return self.traverse_path(goal_node.index)
        return []

    def _push_open(self, node: Node3D) -> None:
        self._push_counter += 1
        heapq.heappush(self.open_heap, (node.f_cost, self._push_counter, node.index))

    def _push_grid(self, node: Node2D) -> None:
        self._push_counter += 1
        heapq.heappush(self.grid_heap, (node.f_cost, self._push_counter, node.index))

    def traverse_path(self, node_index: tuple[int, int, int]) -> list[Pose2D]:
        """Port C++ ``TraversePath``."""
        result: list[Pose2D] = []
        if node_index not in self.open_set:
            return result
        while True:
            node = self.open_set[node_index]
            pre_index = node.pre_index
            if pre_index == UNDEFINED_INDEX or pre_index not in self.open_set:
                break
            pre_node = self.open_set[pre_index]
            path = self.generate_kinematic_path(pre_node.pose, node.is_forward, node.steering)
            for pose in reversed(path[1:]):
                result.append(pose)
            node_index = pre_index
        result.reverse()
        return result

    def generate_kinematic_path(self, pose: Pose2D, is_forward: bool, steering: float) -> list[Pose2D]:
        """Port C++ ``GenerateKinematicPath``."""
        path: list[Pose2D] = []
        step_size = self.config.step_size if is_forward else -self.config.step_size
        current = Pose2D(pose.x, pose.y, pose.theta)
        for _ in range(self.forward_num):
            path.append(Pose2D(current.x, current.y, current.theta))
            current = Pose2D(
                current.x + step_size * np.cos(current.theta),
                current.y + step_size * np.sin(current.theta),
                normalize_angle(current.theta + step_size * np.tan(steering) / self.config.vehicle.wheel_base),
            )
        path.append(current)
        return path

    def expand_next_node(
        self,
        node: Node3D,
        next_index: int,
        hyperparam_set: list[list[list[float]]],
    ) -> Node3D | None:
        """Port C++ ``ExpandNextNode``."""
        is_forward = next_index < float(self.config.next_node_num) / 2.0
        node_res = float(self.config.next_node_num) / 2.0 - 1.0
        if is_forward:
            steering = -self.config.vehicle.phi_max + 2.0 * self.config.vehicle.phi_max / node_res * next_index
        else:
            index = next_index - self.config.next_node_num // 2
            steering = -self.config.vehicle.phi_min + 2.0 * self.config.vehicle.phi_max / node_res * index

        path = self.generate_kinematic_path(node.pose, is_forward, steering)
        for pose in path:
            if self.check_pose_collision(pose):
                return None
        if not self.check_homotopy_constraints(path[-1], hyperparam_set):
            return None

        next_node = Node3D(path[-1], self.origin, self.config)
        next_node.is_forward = is_forward
        next_node.steering = steering
        next_node.pre_index = node.index
        return next_node

    def evaluate_expand_cost(self, parent: Node3D, node: Node3D) -> float:
        """Port C++ ``EvaluateExpandCost``."""
        if node.is_forward:
            cost = self.forward_num * self.config.step_size * self.config.forward_penalty
        else:
            cost = self.forward_num * self.config.step_size * self.config.backward_penalty
        if parent.is_forward != node.is_forward:
            cost += self.config.gear_change_penalty
        cost += self.config.steering_penalty * abs(node.steering)
        cost += self.config.steering_change_penalty * abs(node.steering - parent.steering)
        return float(cost)

    def estimate_heuristic_cost(self, node: Node3D) -> float:
        """Port C++ ``EstimateHeuristicCost``."""
        return self.calculate_2d_cost(node)

    def calculate_2d_cost(self, node_3d: Node3D) -> float:
        """Port C++ dynamic-programming 2D heuristic."""
        node_2d = Node2D.from_pose(node_3d.pose, self.origin, self.config)
        dp_node = self.grid_open_set.get(node_2d.index)
        if dp_node is not None and dp_node.is_closed:
            return dp_node.f_cost

        if self.check_box_collision(node_2d.box(self.origin, self.config)):
            return INF

        while self.grid_heap:
            _, _, node_index = heapq.heappop(self.grid_heap)
            node = self.grid_open_set[node_index]
            if node.is_closed:
                continue
            node.is_closed = True
            if node_index == node_2d.index:
                return node.f_cost

            for (dx, dy), direction_cost in zip(self.grid_directions, self.grid_direction_costs):
                expand = Node2D(node.x_grid + dx, node.y_grid + dy, pre_index=node.index)
                opened = self.grid_open_set.get(expand.index)
                if opened is not None and opened.is_closed:
                    continue
                if self.check_box_collision(expand.box(self.origin, self.config)):
                    if opened is not None:
                        opened.is_closed = True
                    else:
                        expand.is_closed = True
                        self.grid_open_set[expand.index] = expand
                    continue
                expand.f_cost = node.f_cost + direction_cost * self.config.grid_xy_resolution
                if opened is None:
                    self.grid_open_set[expand.index] = expand
                    self._push_grid(expand)
                elif expand.f_cost < opened.f_cost:
                    opened.f_cost = expand.f_cost
                    opened.pre_index = node.index
                    self._push_grid(opened)
        return INF

    def check_oneshot_path(
        self,
        node: Node3D,
        goal: Node3D,
        hyperparam_set: list[list[list[float]]],
    ) -> list[Pose2D]:
        """Port C++ ``CheckOneshotPath`` when an OMPL connector is supplied.

        CPDOT uses OMPL's Dubins/Reeds-Shepp implementation here. The current
        Python environment has no equivalent library installed, so this method
        returns no connection instead of substituting a different planner.
        """
        _ = (node, goal, hyperparam_set)
        return []

    def check_pose_collision(self, pose: Pose2D) -> bool:
        """Port C++ ``Environment::CheckPoseCollision`` using vehicle discs."""
        discs = self.config.vehicle.disc_positions(pose.x, pose.y, pose.theta)
        wh = self.config.vehicle.disc_radius * 2.0
        for i in range(self.config.vehicle.n_disc):
            center = np.array([discs[2 * i], discs[2 * i + 1]], dtype=float)
            if self.check_box_collision(axis_aligned_box(center, wh, wh)):
                return True
        return False

    def check_box_collision(self, box: np.ndarray) -> bool:
        """Port C++ ``Environment::CheckBoxCollision`` for static polygons."""
        if any(not self.map.is_in_bounds(vertex) for vertex in box):
            return True
        return any(polygons_intersect(box, obstacle.polygon()) for obstacle in self.map.obstacles)

    @staticmethod
    def check_homotopy_constraints(pose: Pose2D, hyperparam_set: list[list[list[float]]]) -> bool:
        """Port C++ ``Environment::CheckHomotopyConstraints``."""
        if not hyperparam_set:
            return True
        for polygon_constraints in hyperparam_set:
            violated = False
            for a, b, c in polygon_constraints:
                if pose.x * a + pose.y * b - c > 0.0:
                    violated = True
                    break
            if not violated:
                return True
        return False


def poses_to_array(path: list[Pose2D]) -> np.ndarray:
    """Convert pose list to an ``Nx3`` array."""
    return np.asarray([[pose.x, pose.y, pose.theta] for pose in path], dtype=float)
