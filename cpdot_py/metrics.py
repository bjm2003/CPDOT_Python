"""Experiment metrics for CPDOT Python demos."""

from __future__ import annotations

import numpy as np


def path_length(path: np.ndarray) -> float:
    """Polyline length."""
    if len(path) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())


def normalized_laplacian(distance_matrix: np.ndarray) -> np.ndarray:
    """Normalized graph Laplacian for a weighted complete graph."""
    weights = np.asarray(distance_matrix, dtype=float)
    np.fill_diagonal(weights, 0.0)
    degree = weights.sum(axis=1)
    inv_sqrt = np.zeros_like(degree)
    mask = degree > 1e-12
    inv_sqrt[mask] = 1.0 / np.sqrt(degree[mask])
    return np.eye(len(weights)) - inv_sqrt[:, None] * weights * inv_sqrt[None, :]


def formation_similarity(trajectory: np.ndarray, desired_offsets: np.ndarray) -> tuple[float, float]:
    """Return max and average Laplacian shape error."""
    desired = np.linalg.norm(desired_offsets[:, None, :] - desired_offsets[None, :, :], axis=2)
    l_des = normalized_laplacian(desired)
    errors = []
    for points in trajectory:
        current = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
        errors.append(float(np.linalg.norm(normalized_laplacian(current) - l_des, ord="fro")))
    return float(np.max(errors)), float(np.mean(errors))


def collision_count(map2d, trajectory: np.ndarray, clearance: float = 0.0) -> int:
    """Count colliding robot states."""
    return int(sum(map2d.is_collision(point, clearance) for step in trajectory for point in step))
