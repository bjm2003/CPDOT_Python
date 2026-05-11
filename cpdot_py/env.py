"""2D map and obstacle primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .geometry import (
    AABB,
    as_point,
    oriented_box,
    point_in_polygon,
    polygon_edges,
    polygons_intersect,
    segment_distance,
    segments_intersect,
)


class Obstacle:
    """Base obstacle interface."""

    height: float = 0.0

    def contains(self, point: Iterable[float], clearance: float = 0.0) -> bool:
        raise NotImplementedError

    def distance(self, point: Iterable[float]) -> float:
        raise NotImplementedError

    def intersects_segment(
        self, p1: Iterable[float], p2: Iterable[float], clearance: float = 0.0
    ) -> bool:
        raise NotImplementedError

    def polygon(self) -> np.ndarray:
        raise NotImplementedError


@dataclass
class CircleObstacle(Obstacle):
    """Circular obstacle."""

    center: Sequence[float]
    radius: float
    height: float = 0.0
    facets: int = 48

    def __post_init__(self):
        self.center = as_point(self.center)

    def contains(self, point: Iterable[float], clearance: float = 0.0) -> bool:
        return np.linalg.norm(as_point(point) - self.center) <= self.radius + clearance

    def distance(self, point: Iterable[float]) -> float:
        return float(np.linalg.norm(as_point(point) - self.center) - self.radius)

    def intersects_segment(
        self, p1: Iterable[float], p2: Iterable[float], clearance: float = 0.0
    ) -> bool:
        return segment_distance(self.center, as_point(p1), as_point(p2)) <= self.radius + clearance

    def polygon(self) -> np.ndarray:
        angles = np.linspace(0.0, 2.0 * np.pi, self.facets, endpoint=False)
        return self.center + self.radius * np.column_stack([np.cos(angles), np.sin(angles)])


@dataclass
class PolygonObstacle(Obstacle):
    """Simple polygon obstacle."""

    vertices: Sequence[Sequence[float]]
    height: float = 0.0

    def __post_init__(self):
        self.vertices = np.asarray(self.vertices, dtype=float)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 2 or len(self.vertices) < 3:
            raise ValueError("polygon obstacle needs at least three 2D vertices")

    def contains(self, point: Iterable[float], clearance: float = 0.0) -> bool:
        point = as_point(point)
        if point_in_polygon(point, self.vertices):
            return True
        return clearance > 0.0 and self.distance(point) <= clearance

    def distance(self, point: Iterable[float]) -> float:
        point = as_point(point)
        edge_dist = min(segment_distance(point, a, b) for a, b in polygon_edges(self.vertices))
        return -edge_dist if point_in_polygon(point, self.vertices) else edge_dist

    def intersects_segment(
        self, p1: Iterable[float], p2: Iterable[float], clearance: float = 0.0
    ) -> bool:
        a = as_point(p1)
        b = as_point(p2)
        if self.contains(a, clearance) or self.contains(b, clearance):
            return True
        if clearance > 0.0 and min(segment_distance(v, a, b) for v in self.vertices) <= clearance:
            return True
        return any(segments_intersect(a, b, u, v) for u, v in polygon_edges(self.vertices))

    def polygon(self) -> np.ndarray:
        return np.array(self.vertices, copy=True)


class RectangleObstacle(PolygonObstacle):
    """Axis-aligned rectangle obstacle."""

    def __init__(self, center: Sequence[float], width: float, height: float, obs_height: float = 0.0):
        cx, cy = as_point(center)
        hw, hh = width / 2.0, height / 2.0
        super().__init__(
            [(cx - hw, cy - hh), (cx + hw, cy - hh), (cx + hw, cy + hh), (cx - hw, cy + hh)],
            height=obs_height,
        )
        self.center = np.array([cx, cy])
        self.width = width
        self.rect_height = height


@dataclass
class Map2D:
    """Bounded 2D world with static obstacles."""

    width: float
    height: float
    obstacles: list[Obstacle]
    start: Sequence[float]
    goal: Sequence[float]

    def __post_init__(self):
        self.bounds = AABB(0.0, 0.0, float(self.width), float(self.height))
        self.start = as_point(self.start)
        self.goal = as_point(self.goal)

    def is_in_bounds(self, point: Iterable[float], clearance: float = 0.0) -> bool:
        return self.bounds.contains(as_point(point), clearance)

    def is_collision(self, point: Iterable[float], clearance: float = 0.0) -> bool:
        point = as_point(point)
        return not self.is_in_bounds(point, clearance) or any(
            obs.contains(point, clearance) for obs in self.obstacles
        )

    def clearance(self, point: Iterable[float]) -> float:
        """Signed clearance to the closest obstacle and boundary."""
        point = as_point(point)
        bound_clearance = min(point[0], point[1], self.width - point[0], self.height - point[1])
        if not self.obstacles:
            return float(bound_clearance)
        return float(min(bound_clearance, min(obs.distance(point) for obs in self.obstacles)))

    def segment_is_collision_free(
        self, p1: Iterable[float], p2: Iterable[float], clearance: float = 0.0
    ) -> bool:
        p1 = as_point(p1)
        p2 = as_point(p2)
        if not self.is_in_bounds(p1, clearance) or not self.is_in_bounds(p2, clearance):
            return False
        return not any(obs.intersects_segment(p1, p2, clearance) for obs in self.obstacles)

    def polygon_collides(self, polygon: np.ndarray, clearance: float = 0.0) -> bool:
        """Return true if a polygon intersects any obstacle or leaves the map."""
        if any(not self.is_in_bounds(v, clearance) for v in polygon):
            return True
        return any(polygons_intersect(polygon, obs.polygon()) for obs in self.obstacles)

    def vertex_box_collides(self, centre: Iterable[float], vehicle_offset: float = 3.0) -> bool:
        """Port C++ ``Environment::CheckVerticeCollision`` box check."""
        box = oriented_box(as_point(centre), 0.0, 2.0 + 1.2, vehicle_offset + 1.2)
        return self.polygon_collides(box)

    def spatial_envelope_collides(self, centre: Iterable[float], vehicle_offset: float = 3.0) -> bool:
        """Port C++ ``Environment::CheckSpatialEnvelopes`` box check."""
        box = oriented_box(
            as_point(centre),
            0.0,
            np.sqrt(2.0) * (2.0 + 1.2),
            np.sqrt(2.0) * (vehicle_offset + 1.2),
        )
        return self.polygon_collides(box)

    def obstacle_height_under_polygon(self, polygon: np.ndarray) -> float | None:
        """Return max obstacle height intersected by a formation polygon."""
        heights = [obs.height for obs in self.obstacles if polygons_intersect(polygon, obs.polygon())]
        return max(heights) if heights else None
