"""Unit tests for ``scripts/compare_planned_vs_gazebo.py``.

These do not require ROS or Gazebo: they synthesize a CSV-style odom array
and a plan array, then call the public RMSE helpers directly. The goal is to
catch regressions in the time-alignment / angle-wrap / interpolation logic
before running the full M5 deployment pipeline.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "compare_planned_vs_gazebo.py"


def _load_compare_module():
    spec = importlib.util.spec_from_file_location("_compare_module", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_compare_module"] = module
    spec.loader.exec_module(module)
    return module


COMPARE = _load_compare_module()


def _odom_for_robot(robot: int, ts: np.ndarray, x: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return np.column_stack([ts, x, y, theta])


def test_compute_rmse_zero_when_plan_matches_odom():
    ts = np.linspace(0.0, 5.0, 50)
    plan = {
        0: _odom_for_robot(0, ts, ts, np.zeros_like(ts), np.zeros_like(ts)),
        1: _odom_for_robot(1, ts, np.zeros_like(ts), ts, np.full_like(ts, 0.5)),
    }
    odom = {robot: arr.copy() for robot, arr in plan.items()}
    summary = COMPARE.compute_rmse(odom, plan)
    assert summary["position_rmse"] is not None and summary["position_rmse"] < 1e-9
    assert summary["heading_rmse"] is not None and summary["heading_rmse"] < 1e-9
    assert summary["samples"] > 0


def test_compute_rmse_constant_position_offset():
    ts = np.linspace(0.0, 5.0, 50)
    plan = {0: _odom_for_robot(0, ts, ts, np.zeros_like(ts), np.zeros_like(ts))}
    # Odom is shifted by (0.3, 0.4) -> sqrt(0.25) = 0.5 m position RMSE.
    odom = {0: _odom_for_robot(0, ts, ts + 0.3, np.full_like(ts, 0.4), np.zeros_like(ts))}
    summary = COMPARE.compute_rmse(odom, plan)
    assert math.isclose(summary["position_rmse"], 0.5, rel_tol=1e-6)
    assert math.isclose(summary["heading_rmse"], 0.0, abs_tol=1e-9)


def test_compute_rmse_heading_wrap():
    ts = np.linspace(0.0, 4.0, 40)
    # Plan heading near +pi, odom heading near -pi -> wrapped diff should be small.
    plan = {0: _odom_for_robot(0, ts, np.zeros_like(ts), np.zeros_like(ts), np.full_like(ts, math.pi - 0.05))}
    odom = {0: _odom_for_robot(0, ts, np.zeros_like(ts), np.zeros_like(ts), np.full_like(ts, -math.pi + 0.05))}
    summary = COMPARE.compute_rmse(odom, plan)
    assert summary["heading_rmse"] is not None
    assert summary["heading_rmse"] < 0.2  # wrapped diff = 0.1, not 2*pi-0.1


def test_compute_rmse_only_uses_overlap_when_odom_extends_past_plan():
    plan_ts = np.linspace(0.0, 3.0, 30)
    odom_ts = np.linspace(0.0, 8.0, 80)  # odom keeps recording past plan tf
    plan = {0: _odom_for_robot(0, plan_ts, plan_ts, np.zeros_like(plan_ts), np.zeros_like(plan_ts))}
    odom = {0: _odom_for_robot(0, odom_ts, odom_ts, np.zeros_like(odom_ts), np.zeros_like(odom_ts))}
    summary = COMPARE.compute_rmse(odom, plan)
    # Odom = (t, t, 0, 0); plan interp = (t, 0, 0). diff in x = 0 because both
    # equal t. So pos_rmse should be ~0.
    assert summary["position_rmse"] < 1e-6
    assert summary["samples"] == int((odom_ts <= plan_ts[-1]).sum())


def test_compute_rmse_handles_missing_robot():
    ts = np.linspace(0.0, 2.0, 20)
    plan = {0: _odom_for_robot(0, ts, ts, np.zeros_like(ts), np.zeros_like(ts)),
            1: _odom_for_robot(1, ts, ts, np.zeros_like(ts), np.zeros_like(ts))}
    odom = {0: plan[0].copy()}  # robot 1 missing from odom
    summary = COMPARE.compute_rmse(odom, plan)
    by_robot = {entry["robot"]: entry for entry in summary["per_robot"]}
    assert by_robot[1]["position_rmse"] is None and by_robot[1]["samples"] == 0
    assert by_robot[0]["position_rmse"] is not None and by_robot[0]["position_rmse"] < 1e-9


def test_read_odom_csv_round_trips_with_recorder_format(tmp_path):
    csv_path = tmp_path / "fake_run.csv"
    csv_path.write_text(
        "t,robot,x,y,theta\n"
        "0.000000,0,1.0,2.0,0.0\n"
        "0.100000,0,1.1,2.0,0.0\n"
        "0.000000,1,5.0,6.0,1.57\n"
        "0.100000,1,5.1,6.0,1.57\n",
        encoding="utf-8",
    )
    odom = COMPARE._read_odom_csv(csv_path)
    assert sorted(odom) == [0, 1]
    assert odom[0].shape == (2, 4)
    assert math.isclose(odom[1][0, 3], 1.57, abs_tol=1e-6)
