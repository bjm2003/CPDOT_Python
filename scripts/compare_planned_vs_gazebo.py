#!/usr/bin/env python3
"""Compare a recorded Gazebo odometry CSV against an offline cpdot_py plan.

The CSV is produced by ``cpdot_ros/scripts/record_gazebo_trajectory.py`` and
has columns ``t,robot,x,y,theta``. The reference plan is obtained either
from a ``FormationPlanner.plan_fm_from_guess`` call (when ``--mode source``
is given) or from a JSON Lines file produced by ``run_experiments.py``
(``--plan-jsonl path/to/exp.jsonl``); in the JSON Lines case we expect each
row to also contain a ``trajectory`` key, but the typical use is to compute
the plan in-process here so the script stays self-contained.

Per-robot and aggregate RMSE are printed and (optionally) saved to JSON::

    {
      "per_robot": [{"robot": 0, "position_rmse": 0.12, "heading_rmse": 0.05}, ...],
      "position_rmse": 0.18,
      "heading_rmse": 0.07,
      "samples": 1234
    }

Acceptance thresholds documented in the M5 plan: position RMSE < 0.2 m,
heading RMSE < 0.1 rad (Gazebo physics vs simplified NLP model).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cpdot_py import (  # noqa: E402
    FormationPlanner,
    Map2D,
    PlannerConfig,
    full_states_to_xy_tensor,
)
from cpdot_py.optimizer import VVCMConstants  # noqa: E402
from cpdot_py.states import TrajectoryPoint  # noqa: E402


def _read_odom_csv(path: Path) -> dict[int, np.ndarray]:
    """Load a recorder CSV; return ``{robot_idx: array shape (T, 4)}`` where
    columns are ``[t, x, y, theta]``."""
    rows_by_robot: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows_by_robot[int(row["robot"])].append(
                (float(row["t"]), float(row["x"]), float(row["y"]), float(row["theta"]))
            )
    out: dict[int, np.ndarray] = {}
    for robot, rows in rows_by_robot.items():
        arr = np.array(sorted(rows), dtype=float)
        out[robot] = arr
    return out


def _full_states_to_arrays(full_states_list, *, tf: float) -> dict[int, np.ndarray]:
    """Convert ``[FullStates]`` to per-robot ``[(t, x, y, theta), ...]``.

    The plan reuses ``full.tf`` for absolute time so it can be aligned to
    the Gazebo CSV (which starts at t=0 too).
    """
    out: dict[int, np.ndarray] = {}
    for robot, full in enumerate(full_states_list):
        n = len(full.states)
        if n < 2:
            out[robot] = np.empty((0, 4))
            continue
        dt = tf / (n - 1)
        rows = [
            (i * dt, point.x, point.y, point.theta)
            for i, point in enumerate(full.states)
        ]
        out[robot] = np.asarray(rows, dtype=float)
    return out


def _interp_plan_at(plan_arr: np.ndarray, ts: np.ndarray) -> np.ndarray:
    """Linear interpolate a plan ``(T, 4)`` at the given query times.

    Returns ``(len(ts), 3)`` -> ``[x, y, theta]``. Heading is interpolated as
    a 2D unit vector then converted back via ``atan2`` to handle wrap.
    """
    plan_t = plan_arr[:, 0]
    x = np.interp(ts, plan_t, plan_arr[:, 1])
    y = np.interp(ts, plan_t, plan_arr[:, 2])
    sx = np.interp(ts, plan_t, np.sin(plan_arr[:, 3]))
    cx = np.interp(ts, plan_t, np.cos(plan_arr[:, 3]))
    theta = np.arctan2(sx, cx)
    return np.column_stack([x, y, theta])


def _wrap_angle(diff: np.ndarray) -> np.ndarray:
    return (diff + np.pi) % (2.0 * np.pi) - np.pi


def compute_rmse(odom: dict[int, np.ndarray], plan: dict[int, np.ndarray]) -> dict:
    per_robot: list[dict] = []
    pos_sq_all: list[float] = []
    heading_sq_all: list[float] = []
    sample_count = 0
    for robot, plan_arr in plan.items():
        if robot not in odom or len(odom[robot]) == 0 or len(plan_arr) == 0:
            per_robot.append({"robot": robot, "position_rmse": None, "heading_rmse": None, "samples": 0})
            continue
        odom_arr = odom[robot]
        ts = odom_arr[:, 0]
        # Restrict to overlap of the two time ranges.
        mask = (ts >= plan_arr[0, 0]) & (ts <= plan_arr[-1, 0])
        ts = ts[mask]
        if len(ts) == 0:
            per_robot.append({"robot": robot, "position_rmse": None, "heading_rmse": None, "samples": 0})
            continue
        odom_xyt = odom_arr[mask, 1:]
        plan_xyt = _interp_plan_at(plan_arr, ts)
        dxy = odom_xyt[:, :2] - plan_xyt[:, :2]
        dtheta = _wrap_angle(odom_xyt[:, 2] - plan_xyt[:, 2])
        pos_sq = np.sum(dxy * dxy, axis=1)
        heading_sq = dtheta * dtheta
        per_robot.append({
            "robot": robot,
            "position_rmse": float(np.sqrt(pos_sq.mean())),
            "heading_rmse": float(np.sqrt(heading_sq.mean())),
            "samples": int(len(ts)),
        })
        pos_sq_all.extend(pos_sq.tolist())
        heading_sq_all.extend(heading_sq.tolist())
        sample_count += len(ts)
    summary = {
        "per_robot": per_robot,
        "position_rmse": float(math.sqrt(statistics.mean(pos_sq_all))) if pos_sq_all else None,
        "heading_rmse": float(math.sqrt(statistics.mean(heading_sq_all))) if heading_sq_all else None,
        "samples": sample_count,
    }
    return summary


def _regular_polygon(centre_x: float, centre_y: float, theta: float, radius: float, n: int) -> list[TrajectoryPoint]:
    return [
        TrajectoryPoint(
            x=centre_x + radius * np.cos(2.0 * np.pi * i / n),
            y=centre_y + radius * np.sin(2.0 * np.pi * i / n),
            theta=theta,
        )
        for i in range(n)
    ]


def _plan_in_process(args) -> dict[int, np.ndarray]:
    vvcm = VVCMConstants()
    n = args.n_robots
    # Internal Map2D is shifted to non-negative origin (matches the ROS node).
    mid_x = 0.5 * (args.start_x + args.goal_x)
    mid_y = 0.5 * (args.start_y + args.goal_y)
    shift_x = 0.5 * args.map_width - mid_x
    shift_y = 0.5 * args.map_height - mid_y
    sx = args.start_x + shift_x
    sy = args.start_y + shift_y
    gx = args.goal_x + shift_x
    gy = args.goal_y + shift_y
    scene = Map2D(args.map_width, args.map_height, [], (sx, sy), (gx, gy))
    formation = FormationPlanner(scene, robot_count=n)
    starts = _regular_polygon(sx, sy, 0.0, vvcm.formation_radius, n)
    goals = _regular_polygon(gx, gy, 0.0, vvcm.formation_radius, n)
    config = PlannerConfig(min_nfe=args.min_nfe)
    guess = formation.plan_coarse_full_states(
        starts, goals, hyperparam_sets=None, config=config,
        max_search_time=args.coarse_time, max_expansions=200000,
    )
    result = formation.plan_fm_from_guess(
        guess, config=config,
        max_warm_start=args.warm_starts,
        initial_warm_starts=min(1, args.warm_starts),
        solver_maxiter=args.maxiter,
        solver_method=args.solver_method,
    )
    plan_arr = _full_states_to_arrays(result.states, tf=float(result.states[0].tf))
    # Unshift back to client frame.
    for arr in plan_arr.values():
        if len(arr):
            arr[:, 1] -= shift_x
            arr[:, 2] -= shift_y
    return plan_arr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--odom-csv", type=Path, required=True,
                        help="CSV produced by record_gazebo_trajectory.py")
    parser.add_argument("--n-robots", type=int, default=3)
    parser.add_argument("--start-x", type=float, default=-15.0)
    parser.add_argument("--start-y", type=float, default=0.0)
    parser.add_argument("--goal-x", type=float, default=15.0)
    parser.add_argument("--goal-y", type=float, default=0.0)
    parser.add_argument("--map-width", type=float, default=80.0)
    parser.add_argument("--map-height", type=float, default=60.0)
    parser.add_argument("--solver-method", choices=["L-BFGS-B", "reduced-lsq", "ipopt"], default="ipopt")
    parser.add_argument("--maxiter", type=int, default=50)
    parser.add_argument("--warm-starts", type=int, default=1)
    parser.add_argument("--coarse-time", type=float, default=15.0)
    parser.add_argument("--min-nfe", type=int, default=12)
    parser.add_argument("--summary-json", type=Path, default=None,
                        help="optional path to dump the summary as JSON")
    parser.add_argument("--position-rmse-threshold", type=float, default=0.2)
    parser.add_argument("--heading-rmse-threshold", type=float, default=0.1)
    args = parser.parse_args()

    odom = _read_odom_csv(args.odom_csv)
    print(f"Loaded {sum(len(v) for v in odom.values())} odom rows for {len(odom)} robots")
    plan = _plan_in_process(args)
    print(f"Plan ready: tf={plan[0][-1, 0]:.3f}s, {len(plan)} robots, "
          f"{sum(len(v) for v in plan.values())} plan samples total")

    summary = compute_rmse(odom, plan)
    print(json.dumps(summary, indent=2))
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"\nSaved summary to {args.summary_json}")

    pos_ok = summary["position_rmse"] is not None and summary["position_rmse"] < args.position_rmse_threshold
    heading_ok = summary["heading_rmse"] is not None and summary["heading_rmse"] < args.heading_rmse_threshold
    if pos_ok and heading_ok:
        print(f"PASS: position_rmse={summary['position_rmse']:.3f} < {args.position_rmse_threshold}, "
              f"heading_rmse={summary['heading_rmse']:.3f} < {args.heading_rmse_threshold}")
        sys.exit(0)
    print(f"FAIL: position_rmse={summary['position_rmse']}, heading_rmse={summary['heading_rmse']} "
          f"(thresholds {args.position_rmse_threshold} / {args.heading_rmse_threshold})")
    sys.exit(1)


if __name__ == "__main__":
    main()
