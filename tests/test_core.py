import numpy as np

from cpdot_py import (
    CircleObstacle,
    FormationPlanner,
    Map2D,
    RectangleObstacle,
    TopologyPRM,
    generate_optimal_time_profile_segment,
    resample_path_to_full_states,
    solve_fm,
)
from cpdot_py.forward_kinematics import ForwardKinematics
from cpdot_py.geometry import resample_polyline
from cpdot_py.metrics import collision_count, formation_similarity, ring_adjacency
from cpdot_py.optimizer import FormationNLPProblem, PlannerConfig
from cpdot_py.states import Constraints, FullStates, TrajectoryPoint
from main import build_scene, next_available_path


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
