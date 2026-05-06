"""Formation generation and simplified trajectory optimization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .env import CircleObstacle, Map2D, PolygonObstacle
from .forward_kinematics import ForwardKinematics
from .geometry import headings_from_path, resample_polyline


def regular_polygon(radius: float, count: int, phase: float = 0.0) -> np.ndarray:
    """Return regular polygon vertices centered at the origin."""
    angles = phase + np.arange(count) * 2.0 * np.pi / count
    return radius * np.column_stack([np.cos(angles), np.sin(angles)])


@dataclass
class FormationPlanner:
    """Plan and optimize a flexible multi-robot formation along a guide path."""

    map2d: Map2D
    robot_count: int = 4
    formation_radius: float = 1.2
    sheet_radius: float = 2.35
    robot_clearance: float = 0.18
    obstacle_weight: float = 160.0
    formation_weight: float = 35.0
    reference_weight: float = 0.45
    smooth_weight: float = 9.0
    bound_weight: float = 300.0

    def __post_init__(self):
        self.desired_offsets = regular_polygon(self.formation_radius, self.robot_count, phase=np.pi / 4.0)
        self.sheet_vertices = regular_polygon(self.sheet_radius, self.robot_count, phase=np.pi / 4.0)
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

    def optimize(self, initial: np.ndarray, maxiter: int = 180) -> np.ndarray:
        """Smooth paths with explicit potential-field updates.

        This is the portable counterpart of the C++ Ipopt optimization. It uses
        the same practical ingredients, but updates positions directly instead
        of solving the full optimal-control NLP.
        """
        traj = np.asarray(initial, dtype=float).copy()
        ref = traj.copy()
        lr = 0.018
        for _ in range(maxiter):
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

            traj[1:-1] -= lr * grad[1:-1]
            traj[:, :, 0] = np.clip(traj[:, :, 0], 0.05, self.map2d.width - 0.05)
            traj[:, :, 1] = np.clip(traj[:, :, 1], 0.05, self.map2d.height - 0.05)
        return traj

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
