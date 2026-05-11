#!/usr/bin/env python3
"""Batch-run the CPDOT source-aligned chain across seeds, robot counts, and scenes.

Outputs a JSON Lines file (one record per experiment) plus a summary CSV.
Skips visualization to keep the run cheap.

Example::

    conda run -n cpdot-py python scripts/run_experiments.py \
        --seeds 5 --robots 3 --scenes source --solver-method ipopt \
        --output outputs/experiments.jsonl

The resulting ``outputs/experiments.jsonl`` is one JSON object per line::

    {"seed": 0, "n_robots": 3, "scene": "source", "tf": 79.43, ...}
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

# Make ``main`` importable when running from cpdot_python/
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cpdot_py import (  # noqa: E402
    FormationPlanner,
    PlannerConfig,
    full_states_to_xy_tensor,
)
from cpdot_py.metrics import collision_count, formation_similarity  # noqa: E402
from main import (  # noqa: E402
    build_scene,
    source_aligned_homotopy_constraints,
    source_aligned_robot_states,
)


def _build_args_namespace(seed: int, n_robots: int, scene: str, solver_method: str, *,
                           samples: int, maxiter: int, warm_starts: int,
                           min_nfe: int, coarse_time: float,
                           initial_warm_starts: int) -> SimpleNamespace:
    return SimpleNamespace(
        mode="source",
        samples=samples,
        seed=seed,
        scene_seed=seed,
        scene=scene,
        robots=n_robots,
        source_xy_resolution=0.5,
        source_theta_resolution=0.1,
        source_step_size=0.2,
        source_grid_resolution=1.0,
        source_min_nfe=min_nfe,
        source_coarse_time=coarse_time,
        source_max_expansions=200000,
        source_enable_oneshot=False,
        source_warm_starts=warm_starts,
        source_initial_warm_starts=min(initial_warm_starts, warm_starts),
        source_solver_maxiter=maxiter,
        source_solver_method=solver_method,
        source_topology_attempts=4,
        source_topology_paths=5,
        source_topology_bbox=3.0,
        source_strict_homotopy_bugs=False,
        source_strict_cpp_early_return=False,
    )


def run_one_experiment(seed: int, n_robots: int, scene_name: str, *,
                        solver_method: str, samples: int, maxiter: int,
                        warm_starts: int, min_nfe: int, coarse_time: float,
                        initial_warm_starts: int) -> dict:
    """Run one source-aligned NLP plan and return aggregated metrics."""
    args = _build_args_namespace(
        seed,
        n_robots,
        scene_name,
        solver_method,
        samples=samples,
        maxiter=maxiter,
        warm_starts=warm_starts,
        min_nfe=min_nfe,
        coarse_time=coarse_time,
        initial_warm_starts=initial_warm_starts,
    )
    started = time.perf_counter()
    record = {
        "seed": seed,
        "n_robots": n_robots,
        "scene": scene_name,
        "solver_method": solver_method,
        "samples": samples,
        "maxiter": maxiter,
        "warm_starts": warm_starts,
        "success": False,
        "error": None,
    }
    try:
        scene = build_scene(args.scene_seed, args.scene)
        formation = FormationPlanner(scene, robot_count=args.robots)
        config = PlannerConfig(
            xy_resolution=args.source_xy_resolution,
            theta_resolution=args.source_theta_resolution,
            step_size=args.source_step_size,
            grid_xy_resolution=args.source_grid_resolution,
            min_nfe=args.source_min_nfe,
        )
        starts, goals = source_aligned_robot_states(scene, formation)
        hyperparam_sets, topo_paths, combination = source_aligned_homotopy_constraints(
            scene, starts, goals, args
        )
        guess = formation.plan_coarse_full_states(
            starts,
            goals,
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
            solver_method=args.source_solver_method,
            enforce_cpp_early_return=args.source_strict_cpp_early_return,
        )
        trajectory = full_states_to_xy_tensor(result.states)
        e_max, e_avg = formation_similarity(trajectory, formation.desired_offsets)
        robot_collisions = collision_count(scene, trajectory, clearance=0.03)
        heights = formation.derive_heights_from_full_states(result.states)
        finite_radii = result.height_cons_set[result.height_cons_set != -1]
        finite_heights = heights[np.isfinite(heights)]
        final = result.solve_history[-1] if result.solve_history else None

        record.update(
            success=bool(result.success),
            reason=result.reason,
            warm_start=int(result.warm_start),
            solve_count=len(result.solve_history),
            tf_coarse=float(guess[0].tf),
            tf=float(result.states[0].tf),
            objective=float(final.objective) if final is not None else None,
            infeasibility=float(final.infeasibility) if final is not None else None,
            iterations=int(final.iterations) if final is not None else -1,
            solver_status=final.scipy_message if final is not None else "not_run",
            robot_collisions=int(robot_collisions),
            formation_error_max=float(e_max),
            formation_error_avg=float(e_avg),
            radius_max=float(np.max(finite_radii)) if len(finite_radii) else None,
            height_avg=float(np.mean(finite_heights)) if len(finite_heights) else None,
            topology_path_count=len(topo_paths),
            topology_first_combination_sum=int(sum(combination)),
            wall_time=float(time.perf_counter() - started),
        )
    except Exception as exc:  # pragma: no cover - surface failures to JSON
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["wall_time"] = float(time.perf_counter() - started)
    return record


def _sanitize(value):
    """Convert NaN/Inf floats to None so the dict serialises to strict JSON."""
    if isinstance(value, float):
        if not np.isfinite(value):
            return None
    return value


def _summarize(records: list[dict]) -> dict:
    successful = [r for r in records if r.get("success")]
    if not successful:
        return {"total": len(records), "succeeded": 0}
    return {
        "total": len(records),
        "succeeded": len(successful),
        "success_rate": len(successful) / len(records),
        "tf_mean": statistics.mean(r["tf"] for r in successful),
        "tf_stdev": statistics.pstdev(r["tf"] for r in successful) if len(successful) > 1 else 0.0,
        "infeasibility_mean": statistics.mean(r["infeasibility"] for r in successful),
        "formation_error_max_mean": statistics.mean(r["formation_error_max"] for r in successful),
        "wall_time_mean": statistics.mean(r["wall_time"] for r in successful),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5,
                        help="how many random seeds to run (0..N-1)")
    parser.add_argument("--robots", type=int, nargs="+", default=[3],
                        help="robot counts to evaluate")
    parser.add_argument("--scenes", type=str, nargs="+", default=["source"],
                        choices=["source", "compact"],
                        help="scene types to use (defined in main.build_scene)")
    parser.add_argument("--solver-method", choices=["L-BFGS-B", "reduced-lsq", "ipopt"],
                        default="ipopt")
    parser.add_argument("--samples", type=int, default=1800,
                        help="TopologyPRM max_samples")
    parser.add_argument("--maxiter", type=int, default=200,
                        help="solver max iterations")
    parser.add_argument("--warm-starts", type=int, default=15,
                        help="Plan_fm max warm-start iterations")
    parser.add_argument("--initial-warm-starts", type=int, default=5,
                        help="number of initial-guess warm-starts inside Plan_fm")
    parser.add_argument("--min-nfe", type=int, default=20,
                        help="PlannerConfig.min_nfe (lowering speeds up batch runs)")
    parser.add_argument("--coarse-time", type=float, default=30.0,
                        help="Hybrid A* time budget per seed in seconds")
    parser.add_argument("--output", type=str, default="outputs/experiments.jsonl",
                        help="JSON Lines file for individual records")
    parser.add_argument("--summary", type=str, default="outputs/experiments_summary.json",
                        help="overall summary JSON path")
    parser.add_argument("--verbose", action="store_true",
                        help="print one line per finished experiment")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    with output_path.open("w", encoding="utf-8") as fh:
        for scene in args.scenes:
            for n_robots in args.robots:
                for seed in range(args.seeds):
                    record = run_one_experiment(
                        seed,
                        n_robots,
                        scene,
                        solver_method=args.solver_method,
                        samples=args.samples,
                        maxiter=args.maxiter,
                        warm_starts=args.warm_starts,
                        min_nfe=args.min_nfe,
                        coarse_time=args.coarse_time,
                        initial_warm_starts=args.initial_warm_starts,
                    )
                    record = {key: _sanitize(value) for key, value in record.items()}
                    fh.write(json.dumps(record) + "\n")
                    fh.flush()
                    records.append(record)
                    if args.verbose:
                        ok = "✓" if record.get("success") else "✗"
                        tf = record.get("tf")
                        infeas = record.get("infeasibility")
                        err = record.get("error") or ""
                        if tf is None:
                            tf_str = "n/a"
                        else:
                            tf_str = f"{tf:.2f}"
                        if infeas is None:
                            infeas_str = "n/a"
                        else:
                            infeas_str = f"{infeas:.4f}"
                        print(
                            f"{ok} seed={seed:2d} n={n_robots} scene={scene:8s} "
                            f"tf={tf_str:>7s} infeas={infeas_str:>9s} "
                            f"wall={record['wall_time']:6.1f}s {err}"
                        )

    summary = _summarize(records)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {len(records)} records to {output_path}")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
