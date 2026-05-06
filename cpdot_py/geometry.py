"""Small 2D geometry helpers used by the Python CPDOT reproduction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

EPS = 1e-9


def as_point(point: Iterable[float]) -> np.ndarray:
    """Return *point* as a float numpy vector with shape ``(2,)``."""
    arr = np.asarray(point, dtype=float)
    if arr.shape != (2,):
        raise ValueError(f"expected a 2D point, got shape {arr.shape}")
    return arr


def cross(a: np.ndarray, b: np.ndarray) -> float:
    """2D scalar cross product."""
    return float(a[0] * b[1] - a[1] * b[0])


def segment_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance from a point to a segment."""
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= EPS:
        return float(np.linalg.norm(point - a))
    t = float(np.clip(np.dot(point - a, ab) / denom, 0.0, 1.0))
    return float(np.linalg.norm(point - (a + t * ab)))


def segments_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    """Return true when closed segments AB and CD intersect."""

    def orient(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
        return cross(q - p, r - p)

    def on_segment(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> bool:
        return (
            min(p[0], r[0]) - EPS <= q[0] <= max(p[0], r[0]) + EPS
            and min(p[1], r[1]) - EPS <= q[1] <= max(p[1], r[1]) + EPS
            and abs(orient(p, q, r)) <= EPS
        )

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)

    if o1 * o2 < -EPS and o3 * o4 < -EPS:
        return True
    return (
        on_segment(a, c, b)
        or on_segment(a, d, b)
        or on_segment(c, a, d)
        or on_segment(c, b, d)
    )


def point_in_polygon(point: np.ndarray, vertices: np.ndarray) -> bool:
    """Ray-casting point-in-polygon test for simple polygons."""
    x, y = point
    inside = False
    n = len(vertices)
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        if segment_distance(point, vertices[i], vertices[(i + 1) % n]) <= EPS:
            return True
        crosses = (y1 > y) != (y2 > y)
        if crosses:
            x_inter = (x2 - x1) * (y - y1) / (y2 - y1 + EPS) + x1
            if x_inter > x:
                inside = not inside
    return inside


def polygon_edges(vertices: np.ndarray):
    """Yield closed polygon edges."""
    for i in range(len(vertices)):
        yield vertices[i], vertices[(i + 1) % len(vertices)]


def polygons_intersect(poly_a: np.ndarray, poly_b: np.ndarray) -> bool:
    """Return true if two simple closed polygons overlap or touch."""
    for a1, a2 in polygon_edges(poly_a):
        for b1, b2 in polygon_edges(poly_b):
            if segments_intersect(a1, a2, b1, b2):
                return True
    return point_in_polygon(poly_a[0], poly_b) or point_in_polygon(poly_b[0], poly_a)


def resample_polyline(path: np.ndarray, count: int) -> np.ndarray:
    """Sample ``count`` evenly spaced points along a polyline."""
    path = np.asarray(path, dtype=float)
    if len(path) == 0:
        raise ValueError("cannot resample an empty path")
    if len(path) == 1 or count <= 1:
        return np.repeat(path[:1], max(count, 1), axis=0)

    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    total = float(seg.sum())
    if total <= EPS:
        return np.repeat(path[:1], count, axis=0)
    stations = np.concatenate([[0.0], np.cumsum(seg)])
    targets = np.linspace(0.0, total, count)
    out = np.zeros((count, 2), dtype=float)
    for i, s in enumerate(targets):
        j = int(np.searchsorted(stations, s, side="right") - 1)
        j = min(max(j, 0), len(seg) - 1)
        local = (s - stations[j]) / max(seg[j], EPS)
        out[i] = (1.0 - local) * path[j] + local * path[j + 1]
    return out


def headings_from_path(path: np.ndarray) -> np.ndarray:
    """Compute a heading angle at each path sample."""
    delta = np.gradient(np.asarray(path, dtype=float), axis=0)
    return np.arctan2(delta[:, 1], delta[:, 0])


@dataclass(frozen=True)
class AABB:
    """Axis-aligned map bounds."""

    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def contains(self, point: np.ndarray, clearance: float = 0.0) -> bool:
        return (
            self.xmin + clearance <= point[0] <= self.xmax - clearance
            and self.ymin + clearance <= point[1] <= self.ymax - clearance
        )
