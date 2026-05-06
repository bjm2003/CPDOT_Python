from pathlib import Path

import numpy as np

from cpdot_py import (
    CircleObstacle,
    CoarsePathPlanner,
    FormationPlanner,
    Map2D,
    Pose2D,
    RectangleObstacle,
    TopologyPRM,
    cal_combination,
    cal_corridors,
    generate_desired_rp,
    generate_optimal_time_profile_segment,
    generate_sfc,
    load_cpp_formation_trajectory,
    load_cpp_time_steps,
    poses_to_array,
    resample_path_to_full_states,
    rewire_path,
    solve_fm,
    xy_tensor_to_full_states,
)
from cpdot_py.forward_kinematics import ForwardKinematics
from cpdot_py.coarse_path_planner import Node3D, OptionalDubinsReedsSheppConnector
from cpdot_py.geometry import resample_polyline
from cpdot_py.metrics import collision_count, formation_similarity, ring_adjacency
from cpdot_py.optimizer import FormationNLPProblem, PlannerConfig
from cpdot_py.states import Constraints, FullStates, TrajectoryPoint
from main import build_scene, next_available_path, source_aligned_robot_states


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_collision_segment_circle():
    scene = Map2D(10, 10, [CircleObstacle((5, 5), 1.0)], (1, 1), (9, 9))
    assert scene.is_collision((5, 5))
    assert not scene.segment_is_collision_free((1, 5), (9, 5))
    assert scene.segment_is_collision_free((1, 1), (2, 1))


def test_collision_count_includes_motion_segments():
    scene = Map2D(10, 10, [CircleObstacle((5, 5), 1.0)], (1, 1), (9, 9))
    trajectory = np.array([[[1.0, 5.0]], [[9.0, 5.0]]])
    assert collision_count(scene, trajectory) == 1


def test_topology_prm_finds_path():
    scene = Map2D(12, 8, [RectangleObstacle((6, 4), 1.5, 5.0)], (1, 1), (11, 7))
    prm = TopologyPRM(scene, max_samples=1200, sample_inflate=(7, 4), seed=3)
    paths = prm.find_topo_paths(scene.start, scene.goal)
    assert paths
    assert paths[0].shape[1] == 2
    for a, b in zip(paths[0][:-1], paths[0][1:]):
        assert scene.segment_is_collision_free(a, b, prm.clearance)


def test_forward_kinematics_returns_solution_for_contracting_square():
    sheet = np.array([[2, 0], [0, 2], [-2, 0], [0, -2]], dtype=float)
    robots = 0.55 * sheet + np.array([3.0, 2.0])
    fk = ForwardKinematics(sheet)
    solutions = fk.solve(robots)
    assert solutions
    assert np.isfinite(solutions[0]["object_xyz"]).all()


def test_formation_initial_shape():
    scene = Map2D(12, 8, [], (1, 1), (11, 7))
    guide = resample_polyline(np.array([scene.start, scene.goal]), 10)
    planner = FormationPlanner(scene, robot_count=4)
    traj = planner.initial_trajectory(guide, 10)
    assert traj.shape == (10, 4, 2)
    d0 = np.linalg.norm(traj[0, 0] - traj[0, 1])
    d1 = np.linalg.norm(planner.desired_offsets[0] - planner.desired_offsets[1])
    assert abs(d0 - d1) < 1e-9


def test_formation_similarity_uses_ring_adjacency():
    points = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    weights = ring_adjacency(points)
    assert np.count_nonzero(weights) == 8
    assert weights[0, 2] == 0.0
    err_max, err_avg = formation_similarity(points[None, :, :], points)
    assert err_max == 0.0
    assert err_avg == 0.0


def test_trajectory_state_vector_roundtrip():
    state = TrajectoryPoint(1.0, 2.0, 0.3, 0.4, 0.5, 0.6, 0.7)
    assert TrajectoryPoint.from_vector(state.as_vector()) == state
    path = np.array([[1.0, 2.0], [3.0, 4.0]])
    full = FullStates.from_xy_path(path, tf=1.5)
    assert full.tf == 1.5
    np.testing.assert_allclose(full.xy_array(), path)


def test_next_available_path_does_not_overwrite(tmp_path):
    first = tmp_path / "cpdot_result.png"
    first.write_text("existing", encoding="utf-8")
    assert next_available_path(first) == tmp_path / "cpdot_result_001.png"
    (tmp_path / "cpdot_result_001.png").write_text("existing", encoding="utf-8")
    assert next_available_path(first) == tmp_path / "cpdot_result_002.png"


def test_scene_seed_controls_obstacle_variation():
    scene_a = build_scene(scene_seed=11)
    scene_b = build_scene(scene_seed=11)
    scene_c = build_scene(scene_seed=12)
    assert np.allclose(scene_a.obstacles[0].polygon(), scene_b.obstacles[0].polygon())
    assert not np.allclose(scene_a.obstacles[0].polygon(), scene_c.obstacles[0].polygon())
    assert scene_a.width == 60.0
    assert scene_a.height == 34.0
    assert np.allclose(scene_a.start, [15.0, 17.0])


def test_formation_nlp_pack_and_infeasibility_for_static_guess():
    points = np.array(
        [
            [1.0, 0.0],
            [0.30901699, 0.95105652],
            [-0.80901699, 0.58778525],
            [-0.80901699, -0.58778525],
            [0.30901699, -0.95105652],
        ]
    )
    guess = []
    profile = []
    for x, y in points:
        state = TrajectoryPoint(float(x), float(y))
        full = FullStates(tf=1.0, states=[state, state, state])
        guess.append(full)
        profile.append(Constraints(start=state, goal=state))

    problem = FormationNLPProblem(profile, guess, w_inf=10.0)
    vector = problem.pack_initial_guess()
    assert vector.shape == (problem.nvar,)
    assert problem.eval_infeasibility(vector) < 1e-24
    assert abs(problem.eval_objective(vector) - 1.0) < 1e-12


def test_formation_nlp_bounds_match_cpp_layout():
    points = 1.1 * np.array(
        [
            [1.0, 0.0],
            [0.30901699, 0.95105652],
            [-0.80901699, 0.58778525],
            [-0.80901699, -0.58778525],
            [0.30901699, -0.95105652],
        ]
    )
    guess = []
    profile = []
    for x, y in points:
        state = TrajectoryPoint(float(x), float(y))
        full = FullStates(tf=1.0, states=[state, state, state])
        guess.append(full)
        profile.append(Constraints(start=state, goal=state))

    problem = FormationNLPProblem(profile, guess, height_cons=[-1.0, 0.7, -1.0])
    lower, upper = problem.bounds()
    assert lower[0] == 0.1
    assert np.isposinf(upper[0])
    assert lower[problem.idx_terminal_error] == 1e-4
    assert upper[problem.idx_terminal_error] == 12.0
    assert lower[problem.idx_state(0, 3, 0)] == -1.0
    assert upper[problem.idx_state(0, 3, 0)] == 1.0
    assert lower[problem.idx_state(0, 4, 0)] == -0.69
    assert upper[problem.idx_state(0, 4, 0)] == 0.69
    assert lower[problem.idx_edge_distance(0, 0)] == 3.0 * 0.7 * 0.7
    assert lower[problem.idx_edge_distance(0, 1)] == 1.2 * 1.2
    assert upper[problem.idx_edge_distance(0, 0)] == 4.05 * 4.05
    assert lower[problem.idx_topology(0, 0)] == 0.0
    assert np.isposinf(upper[problem.idx_topology(0, 0)])


def test_cpp_style_time_profile_and_resample_path():
    config = PlannerConfig(min_nfe=6, time_step=0.5)
    stations = np.array([0.0, 0.5, 1.5, 2.0])
    profile = generate_optimal_time_profile_segment(stations, config=config)
    assert np.all(np.diff(profile) >= 0.0)
    assert profile[-1] > profile[0]

    full = resample_path_to_full_states(np.array([[0.0, 0.0], [2.0, 0.0]]), config=config)
    assert len(full.states) == 6
    assert full.tf > 0.0
    assert full.states[0].x == 0.0
    assert full.states[-1].x == 2.0
    assert all(abs(state.v) <= config.vehicle.max_velocity for state in full.states)


def test_solve_fm_wrapper_returns_joint_states_for_static_guess():
    points = np.array(
        [
            [1.0, 0.0],
            [-0.5, 0.8660254],
            [-0.5, -0.8660254],
        ]
    )
    guess = []
    profile = []
    for x, y in points:
        state = TrajectoryPoint(float(x), float(y))
        full = FullStates(tf=1.0, states=[state, state, state])
        guess.append(full)
        profile.append(Constraints(start=state, goal=state))

    solution = solve_fm(profile, guess, w_inf=10.0, maxiter=3)
    assert len(solution.states) == 3
    assert len(solution.states[0].states) == 3
    assert solution.solve_time >= 0.0
    assert solution.infeasibility < 1e-5
    np.testing.assert_allclose(solution.states[0].states[0].xy(), points[0])


def test_generate_sfc_contains_path_samples():
    scene = Map2D(12, 8, [RectangleObstacle((6, 4), 1.5, 4.0)], (1, 1), (11, 7))
    path = np.array([[2.0, 1.0], [4.0, 1.2], [9.0, 6.5]])
    hpolys, key_points, vertices = generate_sfc(path, scene, bbox_width=3.0)
    assert len(hpolys) == len(path)
    assert len(key_points) == len(path)
    assert len(vertices) == len(path)
    for point, halfspaces in zip(path, hpolys):
        residuals = [a * point[0] + b * point[1] - c for a, b, c in halfspaces]
        assert max(residuals) <= 1e-7


def test_height_radius_update_matches_cpp_generate_desired_rp():
    current = np.array([1.2, 1.2, -1.0])
    heights = np.array([-1.0, 0.8, 0.5])
    updated = generate_desired_rp(heights, current)
    np.testing.assert_allclose(updated, [-1.0, 1.4, 1.2])


def test_plan_fm_from_guess_runs_cpp_core_loop_smoke():
    scene = Map2D(10, 10, [], (1, 1), (9, 9))
    points = np.array(
        [
            [3.0, 2.0],
            [2.0, 3.7320508],
            [1.0, 2.0],
        ]
    )
    trajectory = np.repeat(points[None, :, :], 3, axis=0)
    guess = xy_tensor_to_full_states(trajectory)
    planner = FormationPlanner(scene, robot_count=3)
    result = planner.plan_fm_from_guess(guess, max_warm_start=1, initial_warm_starts=1, solver_maxiter=0)
    assert result.solve_history
    assert result.warm_start == 1
    assert len(result.states) == 3
    assert len(result.corridor_cons) == 3


def test_coarse_path_planner_generates_kinematic_path_in_empty_map():
    scene = Map2D(12, 8, [], (1, 1), (11, 1))
    config = PlannerConfig(xy_resolution=0.5, grid_xy_resolution=1.0, step_size=0.2)
    planner = CoarsePathPlanner(scene, config=config, max_search_time=5.0, max_expansions=20000)
    path = planner.plan(Pose2D(1.0, 1.0, 0.0), Pose2D(5.0, 1.0, 0.0))
    assert path
    arr = poses_to_array(path)
    assert arr.shape[1] == 3
    assert np.linalg.norm(arr[-1, :2] - np.array([5.0, 1.0])) < 0.8
    assert all(not planner.check_pose_collision(Pose2D(*pose)) for pose in arr)


def test_coarse_path_planner_respects_homotopy_halfspaces():
    scene = Map2D(12, 8, [], (1, 1), (11, 7))
    planner = CoarsePathPlanner(scene, max_search_time=5.0, max_expansions=20000)
    hyper = [[[1.0, 0.0, 6.0], [-1.0, 0.0, -0.5], [0.0, 1.0, 8.0], [0.0, -1.0, 0.0]]]
    path = planner.plan(Pose2D(1.0, 1.0, 0.0), Pose2D(5.0, 1.0, 0.0), hyper)
    assert path
    assert all(pose.x <= 6.0 + 1e-9 for pose in path)


def test_coarse_path_planner_routes_vehicle_discs_around_obstacle():
    scene = Map2D(14, 8, [RectangleObstacle((6, 1), 1.2, 3.0)], (1, 1), (11, 1))
    config = PlannerConfig(xy_resolution=0.5, grid_xy_resolution=1.0, step_size=0.2)
    planner = CoarsePathPlanner(scene, config=config, max_search_time=10.0, max_expansions=60000)
    path = planner.plan(Pose2D(1.0, 1.0, 0.0), Pose2D(11.0, 1.0, 0.0))
    assert path
    assert np.linalg.norm(path[-1].xy() - np.array([11.0, 1.0])) < 0.8
    assert all(not planner.check_pose_collision(pose) for pose in path)


def test_optional_oneshot_connector_reports_missing_bindings():
    connector = OptionalDubinsReedsSheppConnector()
    connector.dubins = None
    connector.reeds_shepp = None
    assert connector.available(True) is False
    assert connector.available(False) is False


def test_coarse_path_planner_uses_supplied_oneshot_connector():
    class TestConnector:
        def available(self, forward_only):
            return True

        def generate(self, start, goal, *, turning_radius, step_size, forward_only):
            return [Pose2D(start.x, start.y, start.theta), Pose2D(goal.x, goal.y, goal.theta)]

    scene = Map2D(12, 8, [], (1, 1), (11, 1))
    config = PlannerConfig(min_nfe=4)
    planner = CoarsePathPlanner(scene, config=config, enable_oneshot=True, oneshot_connector=TestConnector())
    planner.origin = np.array([3.0, 1.0])
    planner.is_forward_only = False
    start = Node3D(Pose2D(1.0, 1.0, 0.0), planner.origin, config)
    goal = Node3D(Pose2D(5.0, 1.0, 0.0), planner.origin, config)
    path = planner.check_oneshot_path(start, goal, [])
    assert len(path) == 2
    np.testing.assert_allclose(path[-1].xy(), [5.0, 1.0])


def test_formation_planner_generates_aligned_coarse_full_states():
    scene = Map2D(14, 8, [], (1, 1), (11, 7))
    planner = FormationPlanner(scene, robot_count=2)
    starts = [TrajectoryPoint(1.0, 1.0, 0.0), TrajectoryPoint(1.0, 2.0, 0.0)]
    goals = [TrajectoryPoint(5.0, 1.0, 0.0), TrajectoryPoint(5.0, 2.0, 0.0)]
    config = PlannerConfig(xy_resolution=0.5, grid_xy_resolution=1.0, step_size=0.2, min_nfe=6)
    guesses = planner.plan_coarse_full_states(starts, goals, config=config, max_search_time=5.0)
    assert len(guesses) == 2
    assert len(guesses[0].states) == len(guesses[1].states)
    assert guesses[0].tf == guesses[1].tf
    np.testing.assert_allclose(guesses[0].states[0].xy(), [1.0, 1.0], atol=0.5)
    assert np.linalg.norm(guesses[0].states[-1].xy() - np.array([5.0, 1.0])) < 0.8


def test_source_aligned_robot_states_match_cpp_regular_polygon():
    scene = Map2D(60, 34, [], (15, 17), (45, 17))
    planner = FormationPlanner(scene, robot_count=5)
    starts, goals = source_aligned_robot_states(scene, planner)
    assert len(starts) == 5
    assert len(goals) == 5
    for index, offset in enumerate(planner.desired_offsets):
        np.testing.assert_allclose(starts[index].xy(), scene.start + offset)
        np.testing.assert_allclose(goals[index].xy(), scene.goal + offset)
        assert starts[index].theta == 0.0
        assert goals[index].theta == 0.0


def test_cpp_flexible_formation_fixture_matches_python_regular_polygon():
    fixture_dir = REPO_ROOT / "src/CPDOT/formation_planner/traj_result/flexible_formation/5"
    trajectory = load_cpp_formation_trajectory(fixture_dir, 5)
    time_steps = load_cpp_time_steps(fixture_dir / "time_step.yaml")
    assert trajectory.shape == (9942, 5, 2)
    assert len(time_steps) == trajectory.shape[0] + 1
    assert np.isfinite(trajectory).all()
    assert np.all(np.diff(time_steps) > 0.0)

    first_frame = trajectory[0]
    center = first_frame.mean(axis=0)
    expected = center + FormationPlanner(Map2D(80, 80, [], center, center), robot_count=5).desired_offsets
    np.testing.assert_allclose(first_frame, expected, atol=1e-3)
    e_max, e_avg = formation_similarity(first_frame[None, :, :], expected - center)
    assert e_max < 1e-3
    assert e_avg < 1e-3


def test_homotopy_rewire_and_combination_follow_cpp_flow():
    scene = Map2D(20, 12, [], (2, 6), (18, 6))
    planner = FormationPlanner(scene, robot_count=5)
    starts, goals = source_aligned_robot_states(scene, planner)
    center_paths = [
        np.array([[2.0, 6.0], [18.0, 6.0]]),
        np.array([[2.0, 6.0], [10.0, 8.0], [18.0, 6.0]]),
    ]
    raw_paths_set = []
    for start, goal in zip(starts, goals):
        robot_paths = [
            rewire_path(resample_polyline(path, 100), start.xy(), goal.xy(), scene)
            for path in center_paths
        ]
        raw_paths_set.append(robot_paths)

    result = cal_combination(raw_paths_set, scene, selected_path_limit=2)
    assert result.combinations
    assert all(len(combination) == 5 for combination in result.combinations)
    assert [0, 0, 0, 0, 0] in result.combinations
    assert len(result.paths_sets) == 5
    for robot in range(5):
        np.testing.assert_allclose(result.paths_sets[robot][0].path[0], starts[robot].xy())
        np.testing.assert_allclose(result.paths_sets[robot][0].path[-1], goals[robot].xy())


def test_cal_corridors_builds_halfspaces_for_selected_combination():
    scene = Map2D(20, 12, [], (2, 6), (18, 6))
    planner = FormationPlanner(scene, robot_count=5)
    starts, goals = source_aligned_robot_states(scene, planner)
    raw_paths_set = []
    center_path = np.array([[2.0, 6.0], [18.0, 6.0]])
    for start, goal in zip(starts, goals):
        raw_paths_set.append([rewire_path(resample_polyline(center_path, 100), start.xy(), goal.xy(), scene)])

    result = cal_combination(raw_paths_set, scene, selected_path_limit=1)
    corridors = cal_corridors(result.paths_sets, result.combinations[0], scene, bbox_width=3.0)
    assert len(corridors) == 5
    assert all(len(robot_corridors) == 100 for robot_corridors in corridors)
    for point in result.paths_sets[0][0].path[::20]:
        assert any(all(a * point[0] + b * point[1] - c <= 1e-7 for a, b, c in corridor) for corridor in corridors[0])
