"""Homotopy combination utilities ported from CPDOT ``IdentifyHomotopy``."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from .env import Map2D
from .geometry import polygons_intersect, resample_polyline
from .sfc import generate_sfc

DEFAULT_INTER_ROBOT_DISTANCE_LIMIT = 4.05
INTER_ROBOT_DISTANCE_TOLERANCE = 1e-9


@dataclass
class Pathset:
    """C++ ``Pathset`` counterpart: path length plus 100-point path."""

    path_length: float
    path: np.ndarray


@dataclass
class CombinationResult:
    """Result of C++ ``CalCombination`` filtering and sorting."""

    paths_sets: list[list[Pathset]]
    combinations: list[list[int]]
    filter_sort_time: float
    safety_costs: list[float]
    length_costs: list[float]
    homotopy_costs: list[float]


def path_length(path: np.ndarray) -> float:
    """Port ``calPathLength``."""
    arr = np.asarray(path, dtype=float)
    if len(arr) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(arr, axis=0), axis=1).sum())


def discretize_path(path: np.ndarray, point_count: int = 100) -> np.ndarray:
    """Port ``discretizePath``."""
    return resample_polyline(np.asarray(path, dtype=float), point_count)


def find_two_smallest(values: list[float]) -> list[int]:
    """Port C++ ``findTwoSmallest``.

    Despite the source name, the active implementation returns all indices
    sorted by value, not only two indices.
    """
    return sorted(range(len(values)), key=lambda idx: values[idx])


def find_smallest_indices(values: list[float], count: int) -> list[int]:
    """Port C++ ``findSmallestIndices``."""
    if count <= 0:
        return []
    return sorted(range(len(values)), key=lambda idx: values[idx])[: min(count, len(values))]


def eval_path_length(raw_paths: list[np.ndarray], num_selected_path: int) -> list[int]:
    """Port C++ ``evalPathLength`` path-length ranking."""
    lengths = [path_length(path) for path in raw_paths]
    return find_smallest_indices(lengths, num_selected_path)


def generate_combinations(robot_count: int, max_index: int) -> list[list[int]]:
    """Port C++ ``generateCombinations(k, n)`` order."""
    if robot_count <= 0 or max_index < 0:
        return []
    if max_index == 0:
        return [[0 for _ in range(robot_count)]]
    current = [0 for _ in range(robot_count)]
    result: list[list[int]] = []
    done = False
    while not done:
        result.append(current.copy())
        for i in range(robot_count):
            if current[i] < max_index:
                current[i] += 1
                break
            if i == robot_count - 1:
                done = True
            current[i] = 0
    return result


def rewire_path(path: np.ndarray, point1: np.ndarray, point2: np.ndarray, map2d: Map2D) -> np.ndarray:
    """Port C++ ``RewiretPath``."""
    arr = np.asarray(path, dtype=float)
    start_index = 0
    end_index = len(arr) - 1
    for i, point in enumerate(arr):
        if not map2d.segment_is_collision_free(point, point1):
            start_index = max(i - 1, 0)
            break
    for j in range(len(arr) - 1, -1, -1):
        if not map2d.segment_is_collision_free(arr[j], point2):
            end_index = min(j + 1, len(arr) - 1)
            break
    if start_index >= end_index:
        return np.vstack([point1, arr[end_index:], point2])
    return np.vstack([point1, arr[start_index : end_index + 1], point2])


def calculate_signed_distance(points: np.ndarray) -> bool:
    """Port ``calculateSignedDistance`` orientation/topology check."""
    num_robot = len(points)
    for k in range(num_robot):
        for p in range(num_robot):
            if p != k and (p + 1) % num_robot != k:
                np1 = (p + 1) % num_robot
                infeasibility = (points[k, 0] - points[p, 0]) * (points[p, 1] - points[np1, 1]) + (
                    points[k, 1] - points[p, 1]
                ) * (points[np1, 0] - points[p, 0])
                if infeasibility <= 0.0:
                    return False
    return True


def beyond_height_cons(paths_sets: list[list[Pathset]], combination: list[int], map2d: Map2D, zr: float = 2.2) -> bool:
    """Port ``BeyondHeightCons``."""
    for i in range(len(paths_sets[0][0].path)):
        polygon = np.asarray([paths_sets[j][combination[j]].path[i] for j in range(len(combination))], dtype=float)
        for obstacle in map2d.obstacles:
            if obstacle.height > zr and polygons_intersect(polygon, obstacle.polygon()):
                return True
    return False


def beyond_inter_distance_cons(
    paths_sets: list[list[Pathset]],
    combination: list[int],
    *,
    preserve_cpp_bug: bool = False,
    distance_limit: float = DEFAULT_INTER_ROBOT_DISTANCE_LIMIT,
) -> bool:
    """Port ``BeyondInterdisCons``.

    The C++ source indexes the next robot with ``combination[j]`` and returns
    ``true`` after the loop, which removes nearly all combinations. The default
    keeps the intended adjacent-robot distance check using the VVCM edge limit
    ``xv2`` (4.05m); strict compatibility can reproduce the source-level
    behavior.
    """
    for i in range(len(paths_sets[0][0].path)):
        for j in range(len(combination)):
            current = paths_sets[j][combination[j]].path[i]
            next_choice = combination[j] if preserve_cpp_bug else combination[(j + 1) % len(combination)]
            following = paths_sets[(j + 1) % len(combination)][next_choice].path[i]
            inter_dis = abs(current[1] - following[1]) if preserve_cpp_bug else float(np.linalg.norm(current - following))
            limit = 3.0 if preserve_cpp_bug else distance_limit
            if inter_dis > limit + INTER_ROBOT_DISTANCE_TOLERANCE:
                return True
    return True if preserve_cpp_bug else False


def cal_length_set(paths_sets: list[list[Pathset]], combination: list[int]) -> float:
    """Port ``CalLengthSet``."""
    return float(sum(paths_sets[i][combination[i]].path_length for i in range(len(combination))) / len(paths_sets))


def cal_turning_set(paths_sets: list[list[Pathset]], combination: list[int]) -> float:
    """Port C++ ``CalTurningSet`` source behavior.

    The C++ helper truncates direction deltas to ``int`` and returns the
    integer quotient ``turnCount / combination.size()``. The function is present
    in the source but its cost term is commented out in ``CalCombination``.
    """
    if not combination:
        return 0.0
    turn_count = 0
    sample_count = len(paths_sets[0][0].path)
    for i in range(1, sample_count - 1):
        for robot, selected in enumerate(combination):
            path = paths_sets[robot][selected].path
            dir1_x = int(path[i, 0] - path[i - 1, 0])
            dir1_y = int(path[i, 1] - path[i - 1, 1])
            len1 = float(np.sqrt(dir1_x * dir1_x + dir1_y * dir1_y))
            norm_dir1_x = np.nan if len1 == 0.0 else dir1_x / len1
            norm_dir1_y = np.nan if len1 == 0.0 else dir1_y / len1

            dir2_x = int(path[i + 1, 0] - path[i, 0])
            dir2_y = int(path[i + 1, 1] - path[i, 1])
            len2 = float(np.sqrt(dir2_x * dir2_x + dir2_y * dir2_y))
            norm_dir2_x = np.nan if len2 == 0.0 else dir2_x / len2
            norm_dir2_y = np.nan if len2 == 0.0 else dir2_y / len2

            if norm_dir1_x != norm_dir2_x or norm_dir1_y != norm_dir2_y:
                turn_count += 1
    return float(turn_count // len(combination))


def cal_homotopy_set(combination: list[int]) -> float:
    """Port ``CalHomotopySet``."""
    return 1.0 if combination and all(item == combination[0] for item in combination[1:]) else 0.0


def cal_safety_set(paths_sets: list[list[Pathset]], combination: list[int]) -> float:
    """Port ``CalSafetySet``."""
    violations = 0
    for i in range(len(paths_sets[0][0].path)):
        points = np.asarray([paths_sets[j][combination[j]].path[i] for j in range(len(combination))], dtype=float)
        if not calculate_signed_distance(points):
            violations += 1
    return float(violations)


def normalize(values: list[float]) -> list[float]:
    """Port C++ ``Normalizer``."""
    if not values:
        return []
    max_value = max(values)
    min_value = min(values)
    value_range = max_value - min_value
    if value_range == 0:
        return [0.0 for _ in values]
    return [(value - min_value) / value_range for value in values]


def calculate_cost(arrays: list[list[float]]) -> list[float]:
    """Port C++ ``calculateCost``."""
    if not arrays:
        return []
    cost = [0.0 for _ in arrays[0]]
    for arr in arrays:
        for i, value in enumerate(normalize(arr)):
            cost[i] += value
    return cost


def cal_combination(
    raw_paths_set: list[list[np.ndarray]],
    map2d: Map2D,
    *,
    selected_path_limit: int = 8,
    preserve_cpp_bugs: bool = False,
) -> CombinationResult:
    """Port C++ ``CalCombination`` filtering and sorting."""
    if not raw_paths_set or any(len(paths) == 0 for paths in raw_paths_set):
        raise ValueError("raw_paths_set must contain at least one path per robot")
    start_time = time.perf_counter()
    min_path_num = min(len(paths) for paths in raw_paths_set)
    num_selected_path = min(selected_path_limit, min_path_num)
    paths_sets: list[list[Pathset]] = []
    for raw_paths in raw_paths_set:
        order = eval_path_length(raw_paths, num_selected_path)
        paths_sets.append([Pathset(path_length(raw_paths[idx]), discretize_path(raw_paths[idx], 100)) for idx in order])

    combinations = generate_combinations(len(raw_paths_set), num_selected_path - 1)
    filtered: list[list[int]] = []
    for combination in combinations:
        if beyond_height_cons(paths_sets, combination, map2d):
            continue
        if beyond_inter_distance_cons(paths_sets, combination, preserve_cpp_bug=preserve_cpp_bugs):
            continue
        filtered.append(combination)

    safety_costs = [cal_safety_set(paths_sets, combination) for combination in filtered]
    length_costs = [cal_length_set(paths_sets, combination) for combination in filtered]
    homotopy_costs = [cal_homotopy_set(combination) for combination in filtered]
    cost_set = calculate_cost([safety_costs, length_costs, homotopy_costs])
    sorted_indices = sorted(range(len(cost_set)), key=lambda idx: cost_set[idx])
    if preserve_cpp_bugs:
        sorted_combinations = [
            filtered[sorted_indices[sorted_indices[i]]] for i in range(len(sorted_indices))
        ]
    else:
        sorted_combinations = [filtered[i] for i in sorted_indices]
    return CombinationResult(
        paths_sets=paths_sets,
        combinations=sorted_combinations,
        filter_sort_time=time.perf_counter() - start_time,
        safety_costs=safety_costs,
        length_costs=length_costs,
        homotopy_costs=homotopy_costs,
    )


def cal_corridors(
    paths_sets: list[list[Pathset]],
    combination: list[int],
    map2d: Map2D,
    *,
    bbox_width: float = 3.0,
    preserve_cpp_cumulative_polys: bool = False,
) -> list[list[list[list[float]]]]:
    """Port C++ ``CalCorridors`` to per-robot half-space sets."""
    hyperparam_sets: list[list[list[list[float]]]] = []
    cumulative: list[list[list[float]]] = []
    for robot, selected in enumerate(combination):
        hyperparam_set, _, _ = generate_sfc(paths_sets[robot][selected].path, map2d, bbox_width=bbox_width)
        if preserve_cpp_cumulative_polys:
            cumulative.extend(hyperparam_set)
            hyperparam_sets.append(list(cumulative))
        else:
            hyperparam_sets.append(hyperparam_set)
    return hyperparam_sets
