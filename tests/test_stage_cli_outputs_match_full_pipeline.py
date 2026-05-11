"""Stage CLI parity tests for ``main.run_source_aligned_demo``.

The new ``--source-stage`` option lets the user stop the source pipeline at
any intermediate point and dump the artefact to ``outputs/source_stage_X.npz``.
These tests exercise the boundary cases (each stage shorter than the next)
and check that the npz contains the expected keys + that the data agrees
with what a single-pass run produces.

The fixture monkeypatches ``main.build_scene`` to return a small empty map
so ``cal_combination`` always finds a feasible homotopy class and the suite
stays fast (< 30s wall-clock).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import main as main_module

from cpdot_py import Map2D

from main import (
    SOURCE_STAGES,
    run_source_aligned_demo,
)


@pytest.fixture
def open_scene(monkeypatch):
    """Replace ``main.build_scene`` so all source-mode tests share a small
    obstacle-free map and an open formation pose."""
    def _factory(_seed, _scene):
        return Map2D(20, 14, [], (5, 7), (13, 7))

    monkeypatch.setattr(main_module, "build_scene", _factory)
    return _factory


def _open_scene_args(tmp_path: Path, *, stage: str) -> SimpleNamespace:
    """Build the smallest legal SimpleNamespace for ``run_source_aligned_demo``."""
    return SimpleNamespace(
        mode="source",
        samples=80,
        seed=7,
        scene_seed=0,
        scene="source",
        robots=5,
        maxiter=0,
        output_dir=str(tmp_path),
        show=False,
        animate=False,
        source_xy_resolution=0.5,
        source_theta_resolution=0.1,
        source_step_size=0.2,
        source_grid_resolution=1.0,
        source_min_nfe=4,
        source_coarse_time=5.0,
        source_max_expansions=20000,
        source_enable_oneshot=False,
        source_warm_starts=1,
        source_initial_warm_starts=1,
        source_solver_maxiter=0,
        source_solver_method="L-BFGS-B",
        source_topology_attempts=1,
        source_topology_paths=1,
        source_topology_bbox=8.0,
        source_strict_homotopy_bugs=False,
        source_strict_cpp_early_return=False,
        source_stage=stage,
    )


def test_stage_topo_writes_npz_with_centre_paths(open_scene, tmp_path):
    args = _open_scene_args(tmp_path, stage="topo")
    metrics = run_source_aligned_demo(args)
    assert metrics["source_stage"] == "topo"
    assert metrics["source_topology_path_count"] >= 1
    assert "stage_npz" in metrics and "source_coarse_tf" not in metrics

    data = np.load(metrics["stage_npz"], allow_pickle=True)
    assert int(data["topo_path_count"]) == int(metrics["source_topology_path_count"])
    for i in range(int(data["topo_path_count"])):
        path = data[f"topo_path_{i}"]
        assert path.ndim == 2 and path.shape[1] == 2 and path.shape[0] >= 2


def test_stage_combo_writes_combinations_and_costs(open_scene, tmp_path):
    args = _open_scene_args(tmp_path, stage="combo")
    metrics = run_source_aligned_demo(args)
    assert metrics["source_stage"] == "combo"
    assert "source_topology_first_combination_sum" in metrics

    data = np.load(metrics["stage_npz"], allow_pickle=True)
    assert data["combinations"].ndim == 2
    assert data["first_combination"].shape == (5,)  # one per robot
    assert data["safety_costs"].ndim == 1
    assert data["length_costs"].shape == data["safety_costs"].shape
    assert data["homotopy_costs"].shape == data["safety_costs"].shape


def test_stage_corridor_writes_per_robot_halfspaces(open_scene, tmp_path):
    args = _open_scene_args(tmp_path, stage="corridor")
    metrics = run_source_aligned_demo(args)
    assert metrics["source_stage"] == "corridor"

    data = np.load(metrics["stage_npz"], allow_pickle=True)
    n = int(data["robot_count"])
    assert n == 5
    for r in range(n):
        count = int(data[f"robot_{r}_corridor_count"])
        assert count >= 1
        sample = data[f"robot_{r}_corridor_0"]
        assert sample.ndim == 2 and sample.shape[1] == 3  # [a_x, a_y, b]


def test_stage_coarse_writes_full_states_for_each_robot(open_scene, tmp_path):
    args = _open_scene_args(tmp_path, stage="coarse")
    metrics = run_source_aligned_demo(args)
    assert metrics["source_stage"] == "coarse"
    assert metrics["source_coarse_tf"] > 0
    assert "source_result_tf" not in metrics  # plan stage skipped

    data = np.load(metrics["stage_npz"], allow_pickle=True)
    assert float(data["coarse_tf"]) == metrics["source_coarse_tf"]
    for r in range(5):
        xy = data[f"coarse_{r}_xy"]
        assert xy.ndim == 2 and xy.shape[1] == 2 and xy.shape[0] >= 3


def test_stage_plan_persists_solution_states_and_skips_figure(open_scene, tmp_path):
    args = _open_scene_args(tmp_path, stage="plan")
    metrics = run_source_aligned_demo(args)
    assert metrics["source_stage"] == "plan"
    assert metrics["source_result_tf"] > 0
    assert "figure_path" not in metrics

    data = np.load(metrics["stage_npz"], allow_pickle=True)
    assert float(data["plan_tf"]) == metrics["source_result_tf"]
    for r in range(5):
        plan_xy = data[f"plan_{r}_xy"]
        assert plan_xy.ndim == 2 and plan_xy.shape[1] == 2
    assert "height_cons_set" in data


def test_stage_full_keeps_default_behaviour_with_figure_path(open_scene, tmp_path):
    args = _open_scene_args(tmp_path, stage="full")
    metrics = run_source_aligned_demo(args)
    assert metrics["source_stage"] == "full"
    assert "figure_path" in metrics
    assert "stage_npz" not in metrics
    assert metrics["source_result_tf"] > 0


def test_stage_topo_and_full_agree_on_topology_path_count(open_scene, tmp_path):
    """Stopping early must not change earlier-stage data."""
    args_full = _open_scene_args(tmp_path, stage="full")
    args_topo = _open_scene_args(tmp_path, stage="topo")
    full_metrics = run_source_aligned_demo(args_full)
    topo_metrics = run_source_aligned_demo(args_topo)
    assert full_metrics["source_topology_path_count"] == topo_metrics["source_topology_path_count"]


def test_invalid_stage_raises_value_error(open_scene, tmp_path):
    args = _open_scene_args(tmp_path, stage="not-a-stage")
    with pytest.raises(ValueError, match="not-a-stage"):
        run_source_aligned_demo(args)


def test_source_stages_constant_lists_expected_pipeline_order():
    assert SOURCE_STAGES == ("topo", "combo", "corridor", "coarse", "plan", "full")
