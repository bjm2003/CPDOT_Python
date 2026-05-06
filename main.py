"""Run a standalone CPDOT Python reproduction demo."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cpdot_py import CircleObstacle, FormationPlanner, Map2D, RectangleObstacle, TopologyPRM
from cpdot_py.geometry import resample_polyline
from cpdot_py.metrics import collision_count, formation_similarity, path_length
from cpdot_py.visualization import animate_result, plot_map, plot_result


def build_scene() -> Map2D:
    """Create a compact 2D scene with multiple homotopy classes."""
    obstacles = [
        RectangleObstacle(center=(7.2, 5.6), width=2.1, height=6.0, obs_height=0.7),
        RectangleObstacle(center=(12.6, 6.4), width=2.0, height=5.0, obs_height=0.9),
        CircleObstacle(center=(10.0, 2.6), radius=1.1, height=0.5),
        CircleObstacle(center=(10.2, 9.6), radius=1.0, height=0.6),
    ]
    return Map2D(width=20.0, height=12.0, obstacles=obstacles, start=(1.6, 2.0), goal=(18.4, 10.2))


def run_demo(args: argparse.Namespace) -> dict[str, float]:
    """Plan topological guide paths, optimize a formation, and save figures."""
    scene = build_scene()
    prm = TopologyPRM(
        scene,
        max_samples=args.samples,
        sample_inflate=(11.0, 4.8),
        clearance=0.2,
        resolution=0.35,
        seed=args.seed,
    )
    topo_paths = prm.find_topo_paths(scene.start, scene.goal, rectangle_ratio=1.0)
    if not topo_paths:
        raise RuntimeError("TopologyPRM did not find a path; increase --samples or adjust the scene")

    guide = resample_polyline(topo_paths[0], args.steps)
    formation = FormationPlanner(scene, robot_count=args.robots)
    initial = formation.initial_trajectory(guide, args.steps)
    optimized = formation.optimize(initial, maxiter=args.maxiter)
    heights = formation.derive_heights(optimized)
    height_constraints = formation.obstacle_height_constraints(optimized)
    e_max, e_avg = formation_similarity(optimized, formation.desired_offsets)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_result(scene, topo_paths, optimized, output_dir / "cpdot_result.png")
    if args.animate:
        animate_result(scene, optimized, output_dir / "cpdot_animation.gif")
    if args.show:
        import matplotlib.pyplot as plt

        ax = plot_map(scene)
        for path in topo_paths:
            ax.plot(path[:, 0], path[:, 1], "--", alpha=0.5)
        for r in range(args.robots):
            ax.plot(optimized[:, r, 0], optimized[:, r, 1], lw=2.0)
        plt.show()

    finite_heights = heights[np.isfinite(heights)]
    metrics = {
        "topo_path_count": float(len(topo_paths)),
        "guide_length": path_length(topo_paths[0]),
        "robot_collision_count": float(collision_count(scene, optimized, clearance=0.03)),
        "formation_error_max": e_max,
        "formation_error_avg": e_avg,
        "height_min": float(np.min(finite_heights)) if len(finite_heights) else float("nan"),
        "height_avg": float(np.mean(finite_heights)) if len(finite_heights) else float("nan"),
        "height_constraints_hit": float(np.sum(height_constraints >= 0.0)),
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1800)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--robots", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--maxiter", type=int, default=60)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--animate", action="store_true")
    args = parser.parse_args()

    metrics = run_demo(args)
    print("CPDOT Python demo metrics")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    print(f"Saved figure: {Path(args.output_dir) / 'cpdot_result.png'}")


if __name__ == "__main__":
    main()
