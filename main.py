"""Run a standalone CPDOT Python reproduction demo."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cpdot_py import (
    CircleObstacle,
    FormationPlanner,
    Map2D,
    PlannerConfig,
    PolygonObstacle,
    RectangleObstacle,
    TopologyPRM,
    TrajectoryPoint,
    cal_combination,
    cal_corridors,
    full_states_to_xy_tensor,
    rewire_path,
)
from cpdot_py.states import CPDOT_FORMATION_ROBOTS
from cpdot_py.geometry import resample_polyline
from cpdot_py.metrics import collision_count, formation_similarity, path_length
from cpdot_py.visualization import animate_result, plot_map, plot_result


def build_scene(scene_seed: int | None = 0, scene: str = "source") -> Map2D:
    """Create a CPDOT-style 2D scene."""
    if scene == "compact":
        return build_compact_scene(scene_seed)
    if scene == "source":
        return build_source_scene(scene_seed)
    raise ValueError(f"unknown scene {scene!r}")


def build_compact_scene(scene_seed: int | None = 0) -> Map2D:
    """Create the original compact smoke-test scene."""
    rng = np.random.default_rng(scene_seed)
    rect1_center = (7.2 + rng.uniform(-0.15, 0.15), 5.6 + rng.uniform(-0.08, 0.08))
    rect2_center = (12.6 + rng.uniform(-0.15, 0.15), 6.4 + rng.uniform(-0.08, 0.08))
    circle1_center = (10.0 + rng.uniform(-0.18, 0.18), 2.6 + rng.uniform(-0.06, 0.06))
    circle2_center = (10.2 + rng.uniform(-0.18, 0.18), 9.6 + rng.uniform(-0.06, 0.06))
    obstacles = [
        RectangleObstacle(
            center=rect1_center,
            width=2.1 + rng.uniform(-0.08, 0.08),
            height=6.0 + rng.uniform(-0.12, 0.12),
            obs_height=0.7,
        ),
        RectangleObstacle(
            center=rect2_center,
            width=2.0 + rng.uniform(-0.08, 0.08),
            height=5.0 + rng.uniform(-0.12, 0.12),
            obs_height=0.9,
        ),
        CircleObstacle(center=circle1_center, radius=1.1 + rng.uniform(-0.04, 0.04), height=0.5),
        CircleObstacle(center=circle2_center, radius=1.0 + rng.uniform(-0.04, 0.04), height=0.6),
    ]
    return Map2D(width=20.0, height=12.0, obstacles=obstacles, start=(1.6, 2.0), goal=(18.4, 10.2))


def build_source_scene(scene_seed: int | None = 0) -> Map2D:
    """Create the active planning scene used by C++ ``topologic_test.cpp``."""
    rng = np.random.default_rng(scene_seed)
    shift = np.array([30.0, 17.0])
    jitter_scale = 0.0 if scene_seed == 0 else 0.12
    obstacle_specs = [
        ((-5.0, 3.0), 0.0, False, 0.1),
        ((-5.0, -3.0), 0.0, False, 0.1),
        ((6.0, 0.0), 0.23, True, 0.1),
    ]

    def rotated_rectangle(center: tuple[float, float], angle: float, inflated: bool) -> np.ndarray:
        width, height = (4.0, 1.0) if inflated else (6.0, 3.0)
        local = np.array(
            [
                [-width / 2.0, height / 2.0],
                [-width / 2.0, -height / 2.0],
                [width / 2.0, -height / 2.0],
                [width / 2.0, height / 2.0],
            ]
        )
        c, s = np.cos(angle), np.sin(angle)
        rot = np.array([[c, -s], [s, c]])
        return local @ rot.T + np.asarray(center, dtype=float)

    obstacles = []
    for center, angle, inflated, height in obstacle_specs:
        center_jitter = rng.normal(0.0, jitter_scale, size=2)
        angle_jitter = float(rng.normal(0.0, 0.03 * jitter_scale))
        poly = rotated_rectangle(tuple(np.asarray(center) + center_jitter), angle + angle_jitter, inflated)
        inflated_poly = np.array(
            [
                [poly[0, 0] - 0.5, poly[0, 1] + 0.5],
                [poly[1, 0] - 0.5, poly[1, 1] - 0.5],
                [poly[2, 0] + 0.5, poly[2, 1] - 0.5],
                [poly[3, 0] + 0.5, poly[3, 1] + 0.5],
            ]
        )
        obstacles.append(PolygonObstacle(inflated_poly + shift, height=height))
    return Map2D(width=60.0, height=34.0, obstacles=obstacles, start=(15.0, 17.0), goal=(45.0, 17.0))


def run_demo(args: argparse.Namespace) -> dict[str, float | str]:
    """Plan topological guide paths, optimize a formation, and save figures."""
    scene_seed = args.scene_seed
    if scene_seed is None:
        scene_seed = int(np.random.SeedSequence().entropy) % (2**32)
    scene = build_scene(scene_seed, args.scene)
    distance = float(np.linalg.norm(scene.goal - scene.start))
    center_clearance = 0.2
    if scene.segment_is_collision_free(scene.start, scene.goal, center_clearance):
        topo_paths = [np.asarray([scene.start, scene.goal], dtype=float)]
    else:
        prm = TopologyPRM(
            scene,
            max_samples=args.samples,
            sample_inflate=(max(11.0, 0.85 * distance), 0.28 * scene.height),
            clearance=center_clearance,
            resolution=0.35,
            max_raw_paths=16,
            reserve_num=5,
            seed=args.seed,
        )
        topo_paths = prm.find_topo_paths(scene.start, scene.goal, rectangle_ratio=1.0)
    if not topo_paths:
        raise RuntimeError("TopologyPRM did not find a path; increase --samples or adjust the scene")

    formation = FormationPlanner(scene, robot_count=args.robots)
    guide_path = min(topo_paths, key=lambda path: guide_score(scene, formation, path, args.steps))
    guide = resample_polyline(guide_path, args.steps)
    reference = formation.initial_trajectory(guide, args.steps)
    initial, robot_topo_paths = formation.plan_individual_trajectories(
        reference,
        max_samples=args.robot_samples,
        resolution=args.robot_resolution,
        seed=args.seed + 100,
        return_paths=True,
    )
    optimized = formation.optimize(initial, maxiter=args.maxiter)
    initial_collisions = collision_count(scene, initial, clearance=0.03)
    optimized_collisions = collision_count(scene, optimized, clearance=0.03)
    used_seed_trajectory = optimized_collisions > initial_collisions
    if used_seed_trajectory:
        optimized = initial
        optimized_collisions = initial_collisions
    heights = formation.derive_heights(optimized)
    height_constraints = formation.obstacle_height_constraints(optimized)
    e_max, e_avg = formation_similarity(optimized, formation.desired_offsets)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = next_available_path(output_dir / "cpdot_result.png")
    plot_result(
        scene,
        topo_paths,
        optimized,
        figure_path,
        selected_guide=guide_path,
        seed_trajectory=initial,
        robot_topo_paths=robot_topo_paths,
    )
    if args.animate:
        animate_result(scene, optimized, next_available_path(output_dir / "cpdot_animation.gif"))
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
        "robot_topo_candidate_count": float(sum(len(paths) for paths in robot_topo_paths)),
        "guide_length": path_length(guide_path),
        "robot_collision_count": float(optimized_collisions),
        "formation_error_max": e_max,
        "formation_error_avg": e_avg,
        "height_min": float(np.min(finite_heights)) if len(finite_heights) else float("nan"),
        "height_avg": float(np.mean(finite_heights)) if len(finite_heights) else float("nan"),
        "height_constraints_hit": float(np.sum(height_constraints >= 0.0)),
        "scene_seed": float(scene_seed),
        "used_seed_trajectory": float(used_seed_trajectory),
    }
    metrics["figure_path"] = str(figure_path)
    return metrics


def source_aligned_robot_states(
    scene: Map2D,
    formation: FormationPlanner,
) -> tuple[list[TrajectoryPoint], list[TrajectoryPoint]]:
    """Port the C++ regular-polygon start/goal set generation."""
    starts = []
    goals = []
    for offset in formation.desired_offsets:
        start_xy = scene.start + offset
        goal_xy = scene.goal + offset
        starts.append(TrajectoryPoint(float(start_xy[0]), float(start_xy[1]), 0.0))
        goals.append(TrajectoryPoint(float(goal_xy[0]), float(goal_xy[1]), 0.0))
    return starts, goals


def source_aligned_homotopy_constraints(
    scene: Map2D,
    start_set: list[TrajectoryPoint],
    goal_set: list[TrajectoryPoint],
    args: argparse.Namespace,
) -> tuple[list[list[list[list[float]]]], list[np.ndarray], list[int]]:
    """Port the C++ center-topology, rewire, combination, and corridor block."""
    distance = float(np.linalg.norm(scene.goal - scene.start))
    ratio = 1.0
    topo_paths: list[np.ndarray] = []
    for _ in range(args.source_topology_attempts):
        prm = TopologyPRM(
            scene,
            max_samples=args.samples,
            sample_inflate=(max(11.0, 0.85 * distance), 0.28 * scene.height),
            clearance=0.2,
            resolution=0.35,
            max_raw_paths=16,
            reserve_num=args.source_topology_paths,
            seed=args.seed,
        )
        topo_paths = prm.find_topo_paths(scene.start, scene.goal, rectangle_ratio=ratio)
        if topo_paths:
            break
        ratio *= 1.5
    if not topo_paths:
        raise RuntimeError("source topology PRM did not find center paths")

    raw_paths_set: list[list[np.ndarray]] = []
    for start, goal in zip(start_set, goal_set):
        robot_paths = []
        start_xy = np.asarray([start.x, start.y], dtype=float)
        goal_xy = np.asarray([goal.x, goal.y], dtype=float)
        for path in topo_paths:
            robot_paths.append(rewire_path(resample_polyline(path, 100), start_xy, goal_xy, scene))
        raw_paths_set.append(robot_paths)

    combination_result = cal_combination(
        raw_paths_set,
        scene,
        selected_path_limit=args.source_topology_paths,
        preserve_cpp_bugs=args.source_strict_homotopy_bugs,
    )
    if not combination_result.combinations:
        raise RuntimeError("source homotopy combination filtering removed all candidates")
    combination = combination_result.combinations[0]
    hyperparam_sets = cal_corridors(
        combination_result.paths_sets,
        combination,
        scene,
        bbox_width=args.source_topology_bbox,
        preserve_cpp_cumulative_polys=args.source_strict_homotopy_bugs,
    )
    return hyperparam_sets, topo_paths, combination


def run_source_aligned_demo(args: argparse.Namespace) -> dict[str, float | str]:
    """Run the reproduced CPDOT core chain: coarse path -> SFC -> Plan_fm."""
    scene_seed = args.scene_seed
    if scene_seed is None:
        scene_seed = int(np.random.SeedSequence().entropy) % (2**32)
    scene = build_scene(scene_seed, args.scene)
    formation = FormationPlanner(scene, robot_count=args.robots)
    config = PlannerConfig(
        xy_resolution=args.source_xy_resolution,
        theta_resolution=args.source_theta_resolution,
        step_size=args.source_step_size,
        grid_xy_resolution=args.source_grid_resolution,
        min_nfe=args.source_min_nfe,
    )
    start_set, goal_set = source_aligned_robot_states(scene, formation)
    hyperparam_sets, topo_paths, combination = source_aligned_homotopy_constraints(scene, start_set, goal_set, args)
    guess = formation.plan_coarse_full_states(
        start_set,
        goal_set,
        hyperparam_sets=hyperparam_sets,
        config=config,
        max_search_time=args.source_coarse_time,
        max_expansions=args.source_max_expansions,
        enable_oneshot=args.source_enable_oneshot,
    )
    result = formation.plan_fm_from_guess(
        guess,
        config=config,
        max_warm_start=args.source_warm_starts,
        initial_warm_starts=min(args.source_initial_warm_starts, args.source_warm_starts),
        solver_maxiter=args.source_solver_maxiter,
        enforce_cpp_early_return=args.source_strict_cpp_early_return,
    )
    trajectory = full_states_to_xy_tensor(result.states)
    seed_trajectory = full_states_to_xy_tensor(guess)
    robot_collisions = collision_count(scene, trajectory, clearance=0.03)
    heights = formation.derive_heights_from_full_states(result.states)
    e_max, e_avg = formation_similarity(trajectory, formation.desired_offsets)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = next_available_path(output_dir / "cpdot_source_result.png")
    center_guide = np.asarray([scene.start, scene.goal], dtype=float)
    plot_result(
        scene,
        topo_paths,
        trajectory,
        figure_path,
        selected_guide=center_guide,
        seed_trajectory=seed_trajectory,
    )
    if args.animate:
        animate_result(scene, trajectory, next_available_path(output_dir / "cpdot_source_animation.gif"))
    if args.show:
        import matplotlib.pyplot as plt

        ax = plot_map(scene)
        ax.plot(center_guide[:, 0], center_guide[:, 1], "--", alpha=0.5)
        for r in range(args.robots):
            ax.plot(trajectory[:, r, 0], trajectory[:, r, 1], lw=2.0)
        plt.show()

    finite_heights = heights[np.isfinite(heights)]
    finite_radii = result.height_cons_set[result.height_cons_set != -1]
    metrics = {
        "mode": "source",
        "source_success": float(result.success),
        "source_reason": result.reason,
        "source_warm_start": float(result.warm_start),
        "source_solve_count": float(len(result.solve_history)),
        "source_topology_path_count": float(len(topo_paths)),
        "source_topology_first_combination_sum": float(sum(combination)),
        "source_coarse_tf": float(guess[0].tf),
        "source_result_tf": float(result.states[0].tf),
        "source_radius_max": float(np.max(finite_radii)) if len(finite_radii) else float("nan"),
        "robot_collision_count": float(robot_collisions),
        "formation_error_max": e_max,
        "formation_error_avg": e_avg,
        "height_min": float(np.min(finite_heights)) if len(finite_heights) else float("nan"),
        "height_avg": float(np.mean(finite_heights)) if len(finite_heights) else float("nan"),
        "height_constraints_hit": float(np.sum(result.height_cons >= 0.0)),
        "scene_seed": float(scene_seed),
        "figure_path": str(figure_path),
    }
    return metrics


def guide_score(scene: Map2D, formation: FormationPlanner, path: np.ndarray, steps: int) -> float:
    """Prefer guide paths whose lifted formation is close to feasible."""
    reference = formation.initial_trajectory(resample_polyline(path, steps), steps)
    robot_collisions = collision_count(scene, reference, clearance=0.03)
    polygon_collisions = sum(scene.polygon_collides(points, clearance=0.03) for points in reference)
    return path_length(path) + 25.0 * robot_collisions + 5.0 * polygon_collisions


def next_available_path(path: Path) -> Path:
    """Return a path that does not overwrite an existing output file."""
    if not path.exists():
        return path
    for index in range(1, 10000):
        candidate = path.with_name(f"{path.stem}_{index:03d}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"could not allocate a non-overwriting output path for {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fast", "source"], default="fast")
    parser.add_argument("--samples", type=int, default=1800)
    parser.add_argument("--robot-samples", type=int, default=450)
    parser.add_argument("--robot-resolution", type=float, default=0.65)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--robots", type=int, default=CPDOT_FORMATION_ROBOTS)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--scene-seed", type=int, default=None)
    parser.add_argument("--maxiter", type=int, default=60)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--scene", choices=["source", "compact"], default="source")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--animate", action="store_true")
    parser.add_argument("--source-xy-resolution", type=float, default=0.5)
    parser.add_argument("--source-theta-resolution", type=float, default=0.1)
    parser.add_argument("--source-step-size", type=float, default=0.2)
    parser.add_argument("--source-grid-resolution", type=float, default=1.0)
    parser.add_argument("--source-min-nfe", type=int, default=20)
    parser.add_argument("--source-coarse-time", type=float, default=30.0)
    parser.add_argument("--source-max-expansions", type=int, default=200000)
    parser.add_argument("--source-enable-oneshot", action="store_true")
    parser.add_argument("--source-warm-starts", type=int, default=15)
    parser.add_argument("--source-initial-warm-starts", type=int, default=5)
    parser.add_argument("--source-solver-maxiter", type=int, default=200)
    parser.add_argument("--source-topology-attempts", type=int, default=4)
    parser.add_argument("--source-topology-paths", type=int, default=5)
    parser.add_argument("--source-topology-bbox", type=float, default=3.0)
    parser.add_argument("--source-strict-homotopy-bugs", action="store_true")
    parser.add_argument("--source-strict-cpp-early-return", action="store_true")
    args = parser.parse_args()

    metrics = run_source_aligned_demo(args) if args.mode == "source" else run_demo(args)
    print("CPDOT Python demo metrics")
    for key, value in metrics.items():
        if key.endswith("_seed") and isinstance(value, float):
            print(f"  {key}: {int(value)}")
        elif isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        elif isinstance(value, str) and key != "figure_path":
            print(f"  {key}: {value}")
    print(f"Saved figure: {metrics['figure_path']}")


if __name__ == "__main__":
    main()
