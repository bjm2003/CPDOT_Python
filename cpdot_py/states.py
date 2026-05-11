"""Trajectory state containers aligned with CPDOT C++ optimizer types."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

NVAR = 7
CPDOT_FORMATION_ROBOTS = 3


@dataclass
class TrajectoryPoint:
    """Python counterpart of C++ ``formation_planner::TrajectoryPoint``."""

    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    v: float = 0.0
    phi: float = 0.0
    a: float = 0.0
    omega: float = 0.0

    def as_vector(self) -> np.ndarray:
        return np.array([self.x, self.y, self.theta, self.v, self.phi, self.a, self.omega], dtype=float)

    @classmethod
    def from_vector(cls, values: np.ndarray) -> "TrajectoryPoint":
        arr = np.asarray(values, dtype=float)
        if arr.shape != (NVAR,):
            raise ValueError(f"expected a vector with shape ({NVAR},), got {arr.shape}")
        return cls(*arr.tolist())

    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)


@dataclass
class FullStates:
    """Python counterpart of C++ ``formation_planner::FullStates``."""

    tf: float = 0.0
    states: list[TrajectoryPoint] = field(default_factory=list)

    def xy_array(self) -> np.ndarray:
        return np.array([state.xy() for state in self.states], dtype=float)

    @classmethod
    def from_xy_path(cls, path: np.ndarray, tf: float = 0.0) -> "FullStates":
        arr = np.asarray(path, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"expected an Nx2 path, got shape {arr.shape}")
        return cls(tf=tf, states=[TrajectoryPoint(x=float(x), y=float(y)) for x, y in arr])


@dataclass
class Constraints:
    """Python counterpart of C++ ``formation_planner::Constraints``."""

    start: TrajectoryPoint
    goal: TrajectoryPoint
    corridor_lb: np.ndarray | None = None
    corridor_ub: np.ndarray | None = None
