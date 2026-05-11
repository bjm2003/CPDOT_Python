"""Read-only helpers for CPDOT C++ YAML trajectory fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from .states import FullStates, TrajectoryPoint


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


def _parse_state_dict(states_dict: dict) -> FullStates:
    """Build a :class:`FullStates` from ``state1, state2, ...`` keyed YAML."""
    keys = sorted(states_dict.keys(), key=lambda key: int(key.replace("state", "")))
    points: list[TrajectoryPoint] = []
    last_t = 0.0
    for key in keys:
        entry = states_dict[key]
        points.append(
            TrajectoryPoint(
                x=float(entry["x"]),
                y=float(entry["y"]),
                theta=float(entry["theta"]),
                v=float(entry["v"]),
                phi=float(entry["phi"]),
                a=float(entry["a"]),
                omega=float(entry["omega"]),
            )
        )
        last_t = float(entry.get("t", last_t))
    return FullStates(tf=last_t, states=points)


def load_cpp_nlp_solution(path: str | Path) -> FullStates:
    """Load a C++ NLP-style ``traj_NRJ.yaml`` with per-state full kinematics.

    These files are written by ``formation_planner.cpp::Plan_fm`` (see line 852)
    when a warm-start iteration improves the solution. Each YAML key is
    ``stateK`` (1-indexed) with ``x/y/theta/v/phi/a/omega/t`` fields.
    """
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not any(k.startswith("state") for k in data):
        raise ValueError(f"{path} is not a CPDOT NLP-style solution YAML")
    return _parse_state_dict(data)


def load_cpp_warmstart_set(
    directory: str | Path,
    robot_count: int,
    *,
    warm_start: int = 1,
) -> list[FullStates]:
    """Load all per-robot ``traj_NR{warm_start}000.yaml`` solutions.

    Returns ``num_robot`` :class:`FullStates`, each with the same number of
    NFE samples. ``warm_start=1`` corresponds to the C++ ``Plan_fm`` writeout
    at the first refinement iteration (file name suffix ``1000``).
    """
    directory = Path(directory)
    suffix = f"{int(warm_start) * 1000}"
    out: list[FullStates] = []
    for robot in range(robot_count):
        path = directory / f"traj_{robot_count}{robot}{suffix}.yaml"
        out.append(load_cpp_nlp_solution(path))
    nfe_lengths = {len(full.states) for full in out}
    if len(nfe_lengths) != 1:
        raise ValueError(f"NLP solution lengths differ in {directory}: {nfe_lengths}")
    return out


def cpp_warmstart_xy_tensor(
    directory: str | Path,
    robot_count: int,
    *,
    warm_start: int = 1,
) -> np.ndarray:
    """Stack a CPDOT NLP warm-start solution into a ``T x R x 2`` array."""
    full_states = load_cpp_warmstart_set(directory, robot_count, warm_start=warm_start)
    nfe = len(full_states[0].states)
    out = np.zeros((nfe, robot_count, 2), dtype=float)
    for robot, full in enumerate(full_states):
        for t, point in enumerate(full.states):
            out[t, robot, 0] = point.x
            out[t, robot, 1] = point.y
    return out


def cpp_warmstart_endpoints(
    directory: str | Path,
    robot_count: int,
    *,
    warm_start: int = 1,
) -> tuple[list[TrajectoryPoint], list[TrajectoryPoint], float]:
    """Return ``(starts, goals, tf)`` from a C++ NLP warm-start fixture.

    Useful for re-creating the same boundary conditions in a Python plan
    without manually transcribing them. The returned ``starts``/``goals`` are
    the first and last :class:`TrajectoryPoint` of each robot's YAML, and
    ``tf`` is the maximum terminal ``t`` across robots.
    """
    full_states = load_cpp_warmstart_set(directory, robot_count, warm_start=warm_start)
    starts = [full.states[0] for full in full_states]
    goals = [full.states[-1] for full in full_states]
    tf = max(full.tf for full in full_states)
    return starts, goals, float(tf)
