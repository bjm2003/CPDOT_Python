"""Flexible-sheet forward kinematics ported from CPDOT C++."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .geometry import point_in_polygon


@dataclass
class ForwardKinematics:
    """Solve CPDOT's quasi-static sheet/object forward kinematics.

    ``vertices_initial`` are sheet attachment coordinates in the undeformed
    sheet frame. ``robot_positions`` are current robot positions in world XY.
    The solver enumerates taut cable subsets, solves the KKT system from the
    C++ code, then checks inequality feasibility and force closure.
    """

    vertices_initial: np.ndarray
    zr: float = 2.2

    def __post_init__(self):
        self.vertices_initial = np.asarray(self.vertices_initial, dtype=float)

    def solve(self, robot_positions: np.ndarray):
        """Return feasible object poses and taut robot index sets."""
        robots = np.asarray(robot_positions, dtype=float)
        if not self.formation_feasible(robots):
            return []

        out = []
        n = len(robots)
        for taut_num in range(3, n + 1):
            for taut in combinations(range(n), taut_num):
                ordered = list(taut) + [i for i in range(n) if i not in taut]
                i1 = ordered[0]
                xi1, yi1 = robots[i1]
                xvi1, yvi1 = self.vertices_initial[i1]
                c = np.array([-2 * xi1, -2 * yi1, 2 * xvi1, 2 * yvi1], dtype=float)

                rows = []
                rhs = []
                for ij in ordered[1:]:
                    xvij, yvij = self.vertices_initial[ij]
                    xij, yij = robots[ij]
                    rows.append([xij - xi1, yij - yi1, xvi1 - xvij, yvi1 - yvij])
                    rhs.append(
                        0.5
                        * (
                            xvi1 * xvi1
                            + yvi1 * yvi1
                            - xvij * xvij
                            - yvij * yvij
                            - xi1 * xi1
                            - yi1 * yi1
                            + xij * xij
                            + yij * yij
                        )
                    )
                aij = np.asarray(rows, dtype=float)
                bij = np.asarray(rhs, dtype=float)
                a1, b1, a2, b2 = self.create_matrices(aij, bij, taut_num)
                if not self.rank_equality(a1, b1):
                    continue
                a11, b11 = self.full_row_rank(a1, b1)
                x = self.solve_kkt(c, a11, b11)
                feasible, zo = self.check_feasibility(x, a2, b2, xi1, yi1, xvi1, yvi1)
                if feasible and self.force_closure(x[:2], robots[list(taut)]):
                    out.append({"object_xyz": np.array([x[0], x[1], zo]), "sheet_xy": x[2:4], "taut": tuple(taut)})
        return out

    def formation_feasible(self, robots: np.ndarray) -> bool:
        """Current robot distances must be shorter than undeformed sheet distances."""
        n = len(robots)
        for i in range(n):
            for j in range(i + 1, n):
                dist_r = np.linalg.norm(robots[i] - robots[j])
                dist_v = np.linalg.norm(self.vertices_initial[i] - self.vertices_initial[j])
                if dist_r >= dist_v:
                    return False
        return True

    @staticmethod
    def create_matrices(aij: np.ndarray, bij: np.ndarray, k: int):
        a1 = aij[: k - 1]
        b1 = bij[: k - 1]
        a2 = aij[k - 1 :]
        b2 = bij[k - 1 :]
        return a1, b1, a2, b2

    @staticmethod
    def rank_equality(a1: np.ndarray, b1: np.ndarray) -> bool:
        if a1.size == 0:
            return True
        abar = np.column_stack([a1, b1])
        return np.linalg.matrix_rank(a1, tol=1e-9) == np.linalg.matrix_rank(abar, tol=1e-9)

    @staticmethod
    def full_row_rank(a1: np.ndarray, b1: np.ndarray):
        if a1.size == 0:
            return a1, b1
        independent = []
        rank = 0
        for i in range(a1.shape[0]):
            trial = independent + [i]
            trial_rank = np.linalg.matrix_rank(a1[trial], tol=1e-9)
            if trial_rank > rank:
                independent.append(i)
                rank = trial_rank
        return a1[independent], b1[independent]

    @staticmethod
    def solve_kkt(c: np.ndarray, a11: np.ndarray, b11: np.ndarray) -> np.ndarray:
        h = np.diag([2.0, 2.0, -2.0, -2.0])
        if a11.size == 0:
            return -np.linalg.solve(h, c)
        kkt = np.block([[h, a11.T], [a11, np.zeros((a11.shape[0], a11.shape[0]))]])
        rhs = np.concatenate([-c, b11])
        sol = np.linalg.lstsq(kkt, rhs, rcond=None)[0]
        return sol[:4]

    def check_feasibility(
        self,
        x: np.ndarray,
        a2: np.ndarray,
        b2: np.ndarray,
        xi1: float,
        yi1: float,
        xvi1: float,
        yvi1: float,
    ) -> tuple[bool, float]:
        if a2.size and np.any(a2 @ x <= b2 + 1e-9):
            return False, np.nan
        xo, yo, xvo, yvo = x
        fx = (xi1 - xo) ** 2 + (yi1 - yo) ** 2 - (xvi1 - xvo) ** 2 - (yvi1 - yvo) ** 2
        if fx >= -1e-9:
            return False, np.nan
        return True, float(self.zr - np.sqrt(-fx))

    @staticmethod
    def force_closure(object_xy: np.ndarray, robot_polygon: np.ndarray) -> bool:
        if len(robot_polygon) < 3:
            return False
        center = robot_polygon.mean(axis=0)
        order = np.argsort(np.arctan2(robot_polygon[:, 1] - center[1], robot_polygon[:, 0] - center[0]))
        return point_in_polygon(object_xy, robot_polygon[order])
