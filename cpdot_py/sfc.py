"""Safe flight corridor generation ported from CPDOT DecompROS utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .env import Map2D

EPS = 1e-9


@dataclass
class Hyperplane2D:
    """Point-normal half-space with outward normal, matching DecompROS."""

    point: np.ndarray
    normal: np.ndarray

    def signed_dist(self, point: np.ndarray) -> float:
        return float(np.dot(self.normal, point - self.point))

    @property
    def b(self) -> float:
        return float(np.dot(self.normal, self.point))


@dataclass
class Ellipsoid2D:
    """2D ellipsoid ``||C^-1 (x-d)|| <= 1`` from DecompROS."""

    c_matrix: np.ndarray
    center: np.ndarray

    def dist(self, point: np.ndarray) -> float:
        return float(np.linalg.norm(np.linalg.solve(self.c_matrix, point - self.center)))

    def points_inside(self, points: np.ndarray) -> np.ndarray:
        if len(points) == 0:
            return points.reshape(0, 2)
        mask = np.array([self.dist(point) <= 1.0 for point in points], dtype=bool)
        return points[mask]

    def closest_point(self, points: np.ndarray) -> np.ndarray:
        distances = np.array([self.dist(point) for point in points], dtype=float)
        return np.asarray(points[int(np.argmin(distances))], dtype=float)

    def closest_hyperplane(self, points: np.ndarray) -> Hyperplane2D:
        closest = self.closest_point(points)
        inv_c = np.linalg.inv(self.c_matrix)
        normal = inv_c @ inv_c.T @ (closest - self.center)
        norm = float(np.linalg.norm(normal))
        if norm <= EPS:
            normal = closest - self.center
            norm = float(np.linalg.norm(normal))
        return Hyperplane2D(closest, normal / max(norm, EPS))


def vec2_to_rotation(vector: np.ndarray) -> np.ndarray:
    """Return the C++ ``vec2_to_rotation`` matrix."""
    yaw = float(np.arctan2(vector[1], vector[0]))
    return np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]], dtype=float)


def local_bbox_hyperplanes(p1: np.ndarray, p2: np.ndarray, bbox: np.ndarray) -> list[Hyperplane2D]:
    """Build DecompROS local bounding-box virtual walls around a segment."""
    direction = np.asarray(p2, dtype=float) - np.asarray(p1, dtype=float)
    norm = float(np.linalg.norm(direction))
    if norm <= EPS:
        direction = np.array([1.0, 0.0])
    else:
        direction = direction / norm
    direction_h = np.array([direction[1], -direction[0]], dtype=float)
    if np.linalg.norm(direction_h) <= EPS:
        direction_h = np.array([-1.0, 0.0], dtype=float)
    direction_h = direction_h / max(float(np.linalg.norm(direction_h)), EPS)

    pp1 = p1 + direction_h * bbox[1]
    pp2 = p1 - direction_h * bbox[1]
    pp3 = p2 + direction * bbox[0]
    pp4 = p1 - direction * bbox[0]
    return [
        Hyperplane2D(pp1, direction_h),
        Hyperplane2D(pp2, -direction_h),
        Hyperplane2D(pp3, direction),
        Hyperplane2D(pp4, -direction),
    ]


def points_inside_polyhedron(points: np.ndarray, hyperplanes: list[Hyperplane2D]) -> np.ndarray:
    """Return points satisfying all DecompROS half-space inequalities."""
    if len(points) == 0:
        return points.reshape(0, 2)
    mask = []
    for point in points:
        mask.append(all(plane.signed_dist(point) <= EPS for plane in hyperplanes))
    return points[np.asarray(mask, dtype=bool)]


def find_line_ellipsoid(p1: np.ndarray, p2: np.ndarray, obs: np.ndarray, offset_x: float = 0.0) -> Ellipsoid2D:
    """Port ``LineSegment<2>::find_ellipsoid``."""
    focal = float(np.linalg.norm(p1 - p2)) / 2.0
    focal = max(focal, EPS)
    c_matrix = focal * np.eye(2)
    axes = np.full(2, focal, dtype=float)
    c_matrix[0, 0] += offset_x
    axes[0] += offset_x

    if axes[0] > 0:
        ratio = axes[1] / axes[0]
        axes *= ratio
        c_matrix *= ratio

    rotation = vec2_to_rotation(p2 - p1)
    ellipsoid = Ellipsoid2D(rotation @ c_matrix @ rotation.T, 0.5 * (p1 + p2))
    obs_inside = ellipsoid.points_inside(obs)

    while len(obs_inside) > 0:
        closest = ellipsoid.closest_point(obs_inside)
        point_body = rotation.T @ (closest - ellipsoid.center)
        denom = 1.0 - (point_body[0] / max(axes[0], EPS)) ** 2
        if point_body[0] < axes[0] and denom > EPS:
            axes[1] = abs(point_body[1]) / np.sqrt(denom)
        new_c = np.eye(2)
        new_c[0, 0] = axes[0]
        new_c[1, 1] = max(axes[1], EPS)
        ellipsoid.c_matrix = rotation @ new_c @ rotation.T
        obs_inside = np.asarray(
            [point for point in obs_inside if 1.0 - ellipsoid.dist(point) > EPS],
            dtype=float,
        ).reshape(-1, 2)

    return ellipsoid


def line_segment_decomp(
    p1: np.ndarray,
    p2: np.ndarray,
    obs: np.ndarray,
    local_bbox: np.ndarray,
    offset_x: float = 0.0,
) -> list[Hyperplane2D]:
    """Port DecompROS ``LineSegment<2>::dilate`` to half-spaces."""
    bbox_planes = local_bbox_hyperplanes(p1, p2, local_bbox)
    local_obs = points_inside_polyhedron(obs, bbox_planes)
    ellipsoid = find_line_ellipsoid(p1, p2, local_obs, offset_x)

    planes: list[Hyperplane2D] = []
    obs_remain = local_obs
    while len(obs_remain) > 0:
        plane = ellipsoid.closest_hyperplane(obs_remain)
        planes.append(plane)
        obs_remain = np.asarray(
            [point for point in obs_remain if plane.signed_dist(point) < 0.0],
            dtype=float,
        ).reshape(-1, 2)
    planes.extend(bbox_planes)
    return planes


def interpolate_points(start: np.ndarray, end: np.ndarray, num_points: int) -> np.ndarray:
    """Port ``Environment::interpolatePoints``."""
    step = (end - start) / float(num_points + 1)
    return np.asarray([start + i * step for i in range(num_points + 2)], dtype=float)


def obstacle_point_cloud(map2d: Map2D, edge_samples: int = 20) -> np.ndarray:
    """Build the obstacle point cloud used by C++ ``Environment::generateSFC``."""
    points = [
        np.array([60.0, 60.0]),
        np.array([-60.0, 60.0]),
        np.array([-60.0, -60.0]),
        np.array([60.0, -60.0]),
    ]
    for obstacle in map2d.obstacles:
        polygon = obstacle.polygon()
        for i in range(len(polygon)):
            points.extend(interpolate_points(polygon[i], polygon[(i + 1) % len(polygon)], edge_samples))
    return np.asarray(points, dtype=float)


def polyhedron_vertices(planes: list[Hyperplane2D]) -> np.ndarray:
    """Port ``cal_vertices`` for 2D half-space visualization/testing."""
    vertices = []
    for i, plane_i in enumerate(planes):
        vi = np.array([-plane_i.normal[1], plane_i.normal[0]], dtype=float)
        for plane_j in planes[i + 1 :]:
            vj = np.array([-plane_j.normal[1], plane_j.normal[0]], dtype=float)
            a = np.array([[-vi[1], vi[0]], [-vj[1], vj[0]]], dtype=float)
            rhs = np.array(
                [
                    a[0, 0] * plane_i.point[0] + a[0, 1] * plane_i.point[1],
                    a[1, 0] * plane_j.point[0] + a[1, 1] * plane_j.point[1],
                ],
                dtype=float,
            )
            det = float(np.linalg.det(a))
            if abs(det) <= EPS:
                continue
            point = np.linalg.solve(a, rhs)
            if all(plane.signed_dist(point) <= 1e-7 for plane in planes):
                vertices.append(point)
    if not vertices:
        return np.empty((0, 2), dtype=float)
    unique = np.unique(np.round(np.asarray(vertices), decimals=10), axis=0)
    center = unique.mean(axis=0)
    order = np.argsort(np.arctan2(unique[:, 1] - center[1], unique[:, 0] - center[0]))
    return unique[order]


def generate_sfc(
    path: np.ndarray,
    map2d: Map2D,
    bbox_width: float = 5.0,
    edge_samples: int = 20,
) -> tuple[list[list[list[float]]], list[tuple[np.ndarray, np.ndarray]], list[np.ndarray]]:
    """Port C++ ``Environment::generateSFC``.

    Returns ``(hyperparam_set, key_points, corridor_vertices)``. Each
    half-space is represented as ``[a_x, a_y, b]`` and is satisfied when
    ``a_x * x + a_y * y - b <= 0``.
    """
    arr = np.asarray(path, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) < 2:
        raise ValueError("path must be an Nx2 array with at least two points")

    obs = obstacle_point_cloud(map2d, edge_samples=edge_samples)
    bbox = np.array([bbox_width, bbox_width], dtype=float)
    hyperparam_set: list[list[list[float]]] = []
    key_points: list[tuple[np.ndarray, np.ndarray]] = []
    vertices_set: list[np.ndarray] = []

    for idx in range(len(arr)):
        if idx < len(arr) - 1:
            next_point = arr[idx + 1]
            if np.allclose(arr[idx], next_point) and vertices_set:
                hyperparam_set.append(hyperparam_set[-1])
                vertices_set.append(vertices_set[-1])
                key_points.append((arr[idx], next_point))
                continue
        else:
            next_point = arr[idx] + np.array([0.1, 0.1])

        planes = line_segment_decomp(arr[idx], next_point, obs, bbox)
        hyperparam_set.append([[float(p.normal[0]), float(p.normal[1]), p.b] for p in planes])
        key_points.append((arr[idx], next_point))
        vertices_set.append(polyhedron_vertices(planes))

    return hyperparam_set, key_points, vertices_set
