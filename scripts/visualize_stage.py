#!/usr/bin/env python3
"""Visualize the per-stage artefacts produced by ``main.py --source-stage X``.

Reads ``outputs/source_stage_{stage}.npz`` (or the path given via
``--npz``) and produces a side-by-side matplotlib figure showing what that
stage contributed to the pipeline:

- ``topo``     : centre TopologyPRM paths over the scene
- ``combo``    : per-robot rewired raw paths (the input to cal_combination)
- ``corridor`` : robot-0 corridor halfspace polygons
- ``coarse``   : Hybrid A* per-robot xy paths
- ``plan``     : final Plan_fm IPOPT solution xy paths overlaid on the coarse guess

Use this to eyeball whether each stage is doing what you expect, or to
compare scipy vs IPOPT outputs by re-running the same stage with a
different ``--source-solver-method`` and visualising both npz files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save-to-file by default; allow --show for GUI
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cpdot_py.visualization import plot_map  # noqa: E402
from main import build_scene  # noqa: E402


def _detect_stage(data) -> str:
    if "stage" in data.files:
        return str(data["stage"].item() if data["stage"].shape == () else data["stage"][0])
    if "topo_path_count" in data.files:
        return "topo"
    if "combinations" in data.files:
        return "combo"
    if "robot_count" in data.files and "robot_0_corridor_0" in data.files:
        return "corridor"
    if "coarse_tf" in data.files and "plan_tf" not in data.files:
        return "coarse"
    if "plan_tf" in data.files:
        return "plan"
    raise ValueError("could not detect stage from npz contents")


def _plot_topo(data, ax) -> None:
    n = int(data["topo_path_count"])
    for i in range(n):
        path = data[f"topo_path_{i}"]
        ax.plot(path[:, 0], path[:, 1], "-", lw=2.0, alpha=0.85, label=f"path {i}")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"Stage 'topo': {n} centre TopologyPRM paths")


def _plot_combo(data, ax) -> None:
    combos = data["combinations"]
    first = data["first_combination"]
    n = int(data["topo_path_count"]) if "topo_path_count" in data.files else 0
    for i in range(n):
        path = data[f"topo_path_{i}"]
        ax.plot(path[:, 0], path[:, 1], "-", lw=1.5, alpha=0.4, color="#888888")
    ax.set_title(
        f"Stage 'combo': {combos.shape[0]} ranked combinations | first={first.tolist()}"
    )


def _plot_corridor(data, ax) -> None:
    n_robots = int(data["robot_count"])
    colors = plt.cm.viridis(np.linspace(0, 1, n_robots))
    for r in range(n_robots):
        count = int(data[f"robot_{r}_corridor_count"])
        for c in range(count):
            halfspaces = data[f"robot_{r}_corridor_{c}"]
            # halfspaces[i] = [a, b, c] s.t. ax+by-c<=0; intersection is a polygon
            poly = _polygon_from_halfspaces(halfspaces)
            if poly is None:
                continue
            ax.fill(poly[:, 0], poly[:, 1], color=colors[r], alpha=0.06,
                    edgecolor=colors[r], lw=0.5)
    ax.set_title(f"Stage 'corridor': halfspaces × {n_robots} robots")


def _polygon_from_halfspaces(halfspaces: np.ndarray):
    """Compute the polygon vertices (CCW) of the half-space intersection.

    Naive O(M^2) intersection of every pair of edges, filtered by all
    constraints. Adequate for the corridor counts we ship.
    """
    pts = []
    m = len(halfspaces)
    for i in range(m):
        a1, b1, c1 = halfspaces[i]
        for j in range(i + 1, m):
            a2, b2, c2 = halfspaces[j]
            det = a1 * b2 - a2 * b1
            if abs(det) < 1e-9:
                continue
            x = (c1 * b2 - c2 * b1) / det
            y = (a1 * c2 - a2 * c1) / det
            if all(a * x + b * y - c <= 1e-6 for a, b, c in halfspaces):
                pts.append((x, y))
    if not pts:
        return None
    arr = np.unique(np.round(np.asarray(pts), 6), axis=0)
    centre = arr.mean(axis=0)
    order = np.argsort(np.arctan2(arr[:, 1] - centre[1], arr[:, 0] - centre[0]))
    return arr[order]


def _plot_xy_set(data, ax, prefix: str, label: str, *, lw: float = 2.0, alpha: float = 0.95) -> None:
    robot = 0
    while f"{prefix}_{robot}_xy" in data.files:
        xy = data[f"{prefix}_{robot}_xy"]
        ax.plot(xy[:, 0], xy[:, 1], "-", lw=lw, alpha=alpha,
                label=f"{label} robot {robot}" if robot == 0 else None)
        ax.scatter(xy[0, 0], xy[0, 1], c="#15803d", s=30, zorder=4)
        ax.scatter(xy[-1, 0], xy[-1, 1], c="#b91c1c", s=30, marker="*", zorder=4)
        robot += 1


def _plot_coarse(data, ax) -> None:
    _plot_xy_set(data, ax, "coarse", "coarse")
    tf = float(data["coarse_tf"])
    ax.set_title(f"Stage 'coarse': Hybrid A* paths, tf={tf:.2f}s")
    ax.legend(loc="upper right", fontsize=8)


def _plot_plan(data, ax) -> None:
    if "coarse_0_xy" in data.files:
        _plot_xy_set(data, ax, "coarse", "guess", lw=1.0, alpha=0.4)
    _plot_xy_set(data, ax, "plan", "plan", lw=2.5, alpha=0.95)
    plan_tf = float(data["plan_tf"])
    ax.set_title(f"Stage 'plan': IPOPT solution, tf={plan_tf:.2f}s")
    ax.legend(loc="upper right", fontsize=8)


_STAGE_PLOTTERS = {
    "topo": _plot_topo,
    "combo": _plot_combo,
    "corridor": _plot_corridor,
    "coarse": _plot_coarse,
    "plan": _plot_plan,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, required=True,
                        help="path to outputs/source_stage_X.npz")
    parser.add_argument("--scene", choices=["source", "compact"], default="source")
    parser.add_argument("--scene-seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None,
                        help="png to save (default <npz>.png)")
    parser.add_argument("--show", action="store_true",
                        help="open the figure in a GUI window")
    args = parser.parse_args()

    if args.show:
        matplotlib.use("TkAgg", force=True)

    data = np.load(args.npz, allow_pickle=True)
    stage = _detect_stage(data)
    print(f"Detected stage: {stage}")
    plotter = _STAGE_PLOTTERS.get(stage)
    if plotter is None:
        raise ValueError(f"no plotter for stage={stage}")

    scene = build_scene(args.scene_seed, args.scene)
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_map(scene, ax=ax)
    plotter(data, ax)
    fig.tight_layout()

    output = args.output or args.npz.with_suffix(".png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140, bbox_inches="tight")
    print(f"Saved {output}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
