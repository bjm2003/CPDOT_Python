"""Read-only helpers for CPDOT C++ YAML trajectory fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml


def load_cpp_xy_trajectory(path: str | Path) -> np.ndarray:
    """Load a C++ trajectory YAML with ``x`` and ``y`` arrays as ``N x 2``."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "x" not in data or "y" not in data:
        raise ValueError(f"{path} is not a CPDOT x/y trajectory YAML")
    x = np.asarray(data["x"], dtype=float)
    y = np.asarray(data["y"], dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError(f"{path} has invalid x/y trajectory dimensions")
    return np.column_stack([x, y])


def load_cpp_formation_trajectory(directory: str | Path, robot_count: int, *, prefix: str = "traj_real") -> np.ndarray:
    """Load C++ per-robot YAML trajectories as ``T x R x 2``."""
    directory = Path(directory)
    trajectories = [
        load_cpp_xy_trajectory(directory / f"{prefix}{robot_count}{robot}.yaml")
        for robot in range(robot_count)
    ]
    lengths = {len(traj) for traj in trajectories}
    if len(lengths) != 1:
        raise ValueError(f"robot trajectory lengths do not match in {directory}")
    out = np.zeros((len(trajectories[0]), robot_count, 2), dtype=float)
    for robot, trajectory in enumerate(trajectories):
        out[:, robot, :] = trajectory
    return out


def load_cpp_time_steps(path: str | Path) -> np.ndarray:
    """Load a C++ time-step YAML written as a single nested vector."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    arr = np.asarray(data, dtype=float)
    if arr.ndim == 2 and arr.shape[0] == 1:
        return arr[0]
    if arr.ndim == 1:
        return arr
    raise ValueError(f"{path} is not a CPDOT time-step vector")
