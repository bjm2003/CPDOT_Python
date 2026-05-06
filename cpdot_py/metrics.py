"""Experiment metrics for CPDOT Python demos."""

from __future__ import annotations

import numpy as np


def path_length(path: np.ndarray) -> float:
    """Polyline length."""
    if len(path) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())


def normalized_laplacian(distance_matrix: np.ndarray) -> np.ndarray:
    """Normalized graph Laplacian for a weighted adjacency matrix."""
    weights = np.asarray(distance_matrix, dtype=float)
    np.fill_diagonal(weights, 0.0)
    degree = weights.sum(axis=1)
    inv_sqrt = np.zeros_like(degree)
    mask = degree > 1e-12
    inv_sqrt[mask] = 1.0 / np.sqrt(degree[mask])
    return np.eye(len(weights)) - inv_sqrt[:, None] * weights * inv_sqrt[None, :]


def ring_adjacency(points: np.ndarray, edge_length: float | None = None) -> np.ndarray:
    """Return CPDOT's ring adjacency used for formation similarity."""
    points = np.asarray(points, dtype=float)
    count = len(points)
    weights = np.zeros((count, count), dtype=float)
    for i in range(count):
        for j in ((i + 1) % count, (i - 1) % count):
            if edge_length is None:
                weights[i, j] = float(np.linalg.norm(points[i] - points[j]))
            else:
                weights[i, j] = float(edge_length)
    return weights


def formation_similarity(trajectory: np.ndarray, desired_offsets: np.ndarray) -> tuple[float, float]:
    """Return max and average CPDOT ring-Laplacian shape error."""
    desired_edge = float(np.linalg.norm(desired_offsets[1] - desired_offsets[0]))
    l_des = normalized_laplacian(ring_adjacency(desired_offsets, desired_edge))
    errors = []
    for points in trajectory:
        current = ring_adjacency(points)
        errors.append(float(np.linalg.norm(normalized_laplacian(current) - l_des, ord="fro")))
    return float(np.max(errors)), float(np.mean(errors))


def collision_count(map2d, trajectory: np.ndarray, clearance: float = 0.0) -> int:
    """Count colliding robot states and unsafe motion segments."""
    state_collisions = sum(map2d.is_collision(point, clearance) for step in trajectory for point in step)
    segment_collisions = 0
    for robot in range(trajectory.shape[1]):
        segment_collisions += sum(
            not map2d.segment_is_collision_free(a, b, clearance)
            for a, b in zip(trajectory[:-1, robot], trajectory[1:, robot])
        )
    return int(state_collisions + segment_collisions)
