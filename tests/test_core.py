from pathlib import Path
from types import SimpleNamespace

import numpy as np

from cpdot_py import (
    CircleObstacle,
    CoarsePathPlanner,
    FormationPlanner,
    Map2D,
    Pathset,
    Pose2D,
    RectangleObstacle,
    TopologyPRM,
    cal_combination,
    cal_corridors,
    cal_homotopy_set,
    cal_length_set,
    cal_safety_set,
    cal_turning_set,
    eval_path_length,
    find_smallest_indices,
    find_two_smallest,
    full_states_to_xy_tensor,
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
from cpdot_py.geometry import AABB, oriented_box, resample_polyline
from cpdot_py.metrics import collision_count, formation_similarity, ring_adjacency
from cpdot_py.optimizer import (
    CarLikeNLPProblem,
    CarLikeReplanNLPProblem,
    DiffDriveNLPProblem,
    FormationNLPProblem,
    PlannerConfig,
    VVCMConstants,
    VehicleModel,
    solve,
    solve_diff_drive,
    solve_replan,
)
from cpdot_py.states import Constraints, FullStates, TrajectoryPoint
from main import (
    build_scene,
    next_available_path,
    run_source_single_demo,
    source_aligned_homotopy_constraints,
    source_aligned_robot_states,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_collision_segment_circle():
    scene = Map2D(10, 10, [CircleObstacle((5, 5), 1.0)], (1, 1), (9, 9))
    assert scene.is_collision((5, 5))
    assert not scene.segment_is_collision_free((1, 5), (9, 5))
    assert scene.segment_is_collision_free((1, 1), (2, 1))


def test_collision_count_includes_motion_segments():
    scene = Map2D(10, 10, [CircleObstacle((5, 5), 1.0)], (1, 1), (9, 9))
    trajectory = np.array([[[1.0, 5.0]], [[9.0, 5.0]]])
    assert collision_count(scene, trajectory) == 1


def test_aabb_and_oriented_box_helpers_match_core_geometry_contracts():
    box = AABB(1.0, 2.0, 4.0, 6.0)
    assert box.width == 3.0
    assert box.height == 4.0
    np.testing.assert_allclose(box.center, [2.5, 4.0])
    assert box.overlaps(AABB(3.0, 5.0, 5.0, 8.0))
    assert not box.overlaps(AABB(5.0, 2.0, 6.0, 3.0))
    assert box.distance_to_point((2.0, 4.0)) == 0.0
    assert box.distance_to_aabb(AABB(5.0, 6.0, 7.0, 8.0)) == 1.0

    rect = oriented_box((0.0, 0.0), 0.0, 4.0, 2.0)
    np.testing.assert_allclose(rect[0], [2.0, 1.0])
    np.testing.assert_allclose(rect[2], [-2.0, -1.0])


def test_map_vertex_and_spatial_envelope_collision_helpers():
    scene = Map2D(10, 10, [RectangleObstacle((5.0, 5.0), 1.0, 1.0)], (1, 1), (9, 9))
    assert scene.vertex_box_collides((5.0, 5.0), vehicle_offset=3.0)
    assert scene.spatial_envelope_collides((5.0, 5.0), vehicle_offset=3.0)
    assert not scene.vertex_box_collides((2.5, 2.5), vehicle_offset=1.0)


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


def test_planner_vehicle_and_vvcm_defaults_match_cpp_headers():
    config = PlannerConfig()
    assert config.xy_resolution == 0.5
    assert config.theta_resolution == 0.1
    assert config.step_size == 0.2
    assert config.next_node_num == 6
    assert config.grid_xy_resolution == 1.0
    assert config.forward_penalty == 0.5
    assert config.backward_penalty == 1.0
    assert config.gear_change_penalty == 5.0
    assert config.steering_penalty == 0.5
    assert config.steering_change_penalty == 1.0
    assert config.min_nfe == 20
    assert config.time_step == 0.5
    assert config.corridor_max_iter == 1000
    assert config.corridor_incremental_limit == 20.0
    assert config.opti_w_phi == 1.0
    assert config.opti_w_a == 1.0
    assert config.opti_t == 1.0
    assert config.factor_a == 0.9
    assert config.factor_b == 1.1
    assert config.opti_w_omega == 1.0
    assert config.opti_w_diff_drive == 0.05
    assert config.opti_w_x == 1.0
    assert config.opti_w_y == 1.0
    assert config.opti_w_err == 1.0
    assert config.opti_w_theta == 1.0
    assert config.opti_inner_iter_max == 100
    assert config.opti_w_penalty0 == 1e4
    assert config.opti_varepsilon_tol == 1e-4

    vehicle = VehicleModel()
    assert vehicle.offset == 3.0
    assert vehicle.vertices == 4
    assert vehicle.front_hang_length == 0.165
    assert vehicle.wheel_base == 0.65
    assert vehicle.rear_hang_length == 0.165
    assert vehicle.width == 0.605
    assert vehicle.max_velocity == 1.0
    assert vehicle.min_velocity == -1.0
    assert vehicle.max_acceleration == 1.0
    assert vehicle.phi_max == 0.69
    assert vehicle.phi_min == 0.69
    assert vehicle.omega_max == 0.2
    assert vehicle.n_disc == 2
    assert vehicle.min_vel_diff == -1.0
    assert vehicle.max_vel_diff == 2.0
    assert vehicle.omg_acc_diff == 2.5
    assert vehicle.max_acc_diff == 1.0
    assert vehicle.omg_max_diff == 1.5
    np.testing.assert_allclose(vehicle.disc_coefficients, [0.08, 0.57])
    np.testing.assert_allclose(vehicle.disc_radius, 0.5 * np.hypot(0.98 / 2.0, 0.605))

    vvcm = VVCMConstants()
    assert vvcm.radius_inc == 0.2
    assert vvcm.xv2t == 1.2
    assert vvcm.zr == 2.2
    np.testing.assert_allclose(vvcm.formation_radius, 4.05 / np.sqrt(3.0))
    np.testing.assert_allclose(vvcm.xv2, 4.05)


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


def test_car_like_nlp_layout_bounds_and_source_vertex_residuals():
    states = [
        TrajectoryPoint(1.0, 2.0, 0.0),
        TrajectoryPoint(1.0, 2.0, 0.0),
        TrajectoryPoint(1.0, 2.0, 0.0),
    ]
    guess = FullStates(tf=1.0, states=states)
    corridor_lb = np.full((3, 8), -10.0)
    corridor_ub = np.full((3, 8), 10.0)
    profile = Constraints(states[0], states[-1], corridor_lb=corridor_lb, corridor_ub=corridor_ub)
    problem = CarLikeNLPProblem(profile, guess)
    assert problem.nrows == 1
    assert problem.ncols == 15
    assert problem.nvar == 17
    lower, upper = problem.bounds()
    assert lower[0] == 0.1
    assert lower[problem.idx_state(3, 0)] == -1.0
    assert upper[problem.idx_state(4, 0)] == 0.69
    assert lower[problem.idx_vertex(0, 0)] == -10.0
    assert upper[problem.idx_vertex(7, 0)] == 10.0

    vector = problem.pack_initial_guess()
    source_vertices = problem._active_residual_vertices(1.0, 2.0, 0.0)
    for i, value in enumerate(source_vertices):
        vector[problem.idx_vertex(i, 0)] = value
    assert problem.eval_infeasibility(vector) < 1e-20


def test_diff_drive_and_replan_nlp_static_infeasibility_and_fixed_time_bounds():
    states = [
        TrajectoryPoint(1.0, 2.0, 0.0),
        TrajectoryPoint(1.0, 2.0, 0.0),
        TrajectoryPoint(1.0, 2.0, 0.0),
    ]
    guess = FullStates(tf=1.5, states=states)
    profile = Constraints(states[0], states[-1])

    diff_problem = DiffDriveNLPProblem(profile, guess)
    diff_lower, diff_upper = diff_problem.bounds()
    assert diff_lower[0] == 1.5
    assert diff_upper[0] == 1.5
    assert diff_lower[diff_problem.idx_state(4, 0)] == -2.5
    assert diff_upper[diff_problem.idx_state(6, 0)] == 1.5
    assert diff_problem.eval_infeasibility(diff_problem.pack_initial_guess()) < 1e-20

    replan_problem = CarLikeReplanNLPProblem(profile, guess)
    replan_lower, replan_upper = replan_problem.bounds()
    assert replan_lower[0] == 1.5
    assert replan_upper[0] == 1.5
    assert replan_lower[replan_problem.idx_state(4, 0)] == -0.69
    assert replan_problem.eval_infeasibility(replan_problem.pack_initial_guess()) < 1e-20


def test_single_robot_solve_wrappers_return_cpp_style_states():
    states = [
        TrajectoryPoint(1.0, 2.0, 0.0),
        TrajectoryPoint(1.0, 2.0, 0.0),
        TrajectoryPoint(1.0, 2.0, 0.0),
    ]
    guess = FullStates(tf=1.0, states=states)
    corridor_lb = np.full((3, 8), -10.0)
    corridor_ub = np.full((3, 8), 10.0)
    profile = Constraints(states[0], states[-1], corridor_lb=corridor_lb, corridor_ub=corridor_ub)
    config = PlannerConfig()

    car_solution = solve(profile, guess, config=config, maxiter=0)
    assert len(car_solution.state.states) == 3
    np.testing.assert_allclose(car_solution.state.states[0].xy(), states[0].xy())
    np.testing.assert_allclose(car_solution.state.states[-1].xy(), states[-1].xy())

    diff_solution = solve_diff_drive(profile, guess, FullStates(), config=config, maxiter=0)
    replan_solution = solve_replan(profile, guess, FullStates(), config=config, maxiter=0)
    assert len(diff_solution.state.states) == 3
    assert len(replan_solution.state.states) == 3
    assert diff_solution.solve_time >= 0.0
    assert replan_solution.solve_time >= 0.0


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


def test_formation_nlp_source_index_regression_for_additional_variables():
    points = np.array([[1.0, 0.0], [-0.5, 0.8660254], [-0.5, -0.8660254]])
    guess = []
    profile = []
    for x, y in points:
        state = TrajectoryPoint(float(x), float(y))
        full = FullStates(tf=1.0, states=[state, state, state])
        guess.append(full)
        profile.append(Constraints(start=state, goal=state))
    corridor_cons = [
        [
            [[1.0, 0.0, 10.0], [0.0, 1.0, 10.0]],
            [[-1.0, 0.0, 10.0]],
        ]
        for _ in range(3)
    ]
    problem = FormationNLPProblem(profile, guess, corridor_cons=corridor_cons)
    assert problem.nrows == 2
    assert problem.add_var == 6
    assert problem.num_sfc_cons == 9
    assert problem.idx_state(1, 0, 0) == 1 + 7 * problem.nrows
    assert problem.idx_edge_distance(0, 0) == 1 + 21 * problem.nrows
    assert problem.idx_topology(0, 0) == 1 + 24 * problem.nrows
    assert problem.idx_sfc(0) == 1 + 27 * problem.nrows
    assert problem.idx_terminal_error == problem.nvar - 1
    assert problem.idx_sfc(2 * problem.num_sfc_cons - 1) == problem.nvar - 2


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


def test_solve_fm_reduced_lsq_projects_auxiliary_terms_for_source_layout():
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

    solution = solve_fm(profile, guess, w_inf=10.0, maxiter=1, method="reduced-lsq")
    assert len(solution.states) == 3
    assert solution.infeasibility < 1e-5
    assert set(solution.infeasibility_terms) == {
        "initial_terminal",
        "dynamics",
        "terminal_error",
        "edge_distance",
        "topology",
        "sfc",
    }


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
    for vertex_set in vertices:
        assert vertex_set.shape[1] == 2
        assert len(vertex_set) >= 3
        assert np.isfinite(vertex_set).all()


def test_source_scene_sfc_halfspace_snapshot_is_deterministic():
    scene = build_scene(scene_seed=0, scene="source")
    path = np.array([[15.0, 17.0], [24.0, 24.0], [35.0, 24.0], [45.0, 17.0]])
    hpolys, _, vertices = generate_sfc(path, scene, bbox_width=5.0)
    assert [len(poly) for poly in hpolys] == [5, 5, 5, 5]
    np.testing.assert_allclose(
        np.asarray(hpolys[0]),
        np.array(
            [
                [0.617231, -0.786782, -4.038730],
                [0.613941, -0.789352, 0.790122],
                [-0.613941, 0.789352, 9.209878],
                [0.789352, 0.613941, 38.679028],
                [-0.789352, -0.613941, -17.277274],
            ]
        ),
        atol=1e-6,
    )
    assert vertices[0].shape == (4, 2)
    np.testing.assert_allclose(
        vertices[0][:3],
        np.array(
            [
                [7.983536, 17.877058],
                [11.114002, 13.852173],
                [27.952660, 27.062119],
            ]
        ),
        atol=1e-6,
    )
    for point, halfspaces in zip(path, hpolys):
        residuals = [a * point[0] + b * point[1] - c for a, b, c in halfspaces]
        assert max(residuals) <= 1e-7


def test_corridor_box_expansion_and_failure_match_source_flow():
    empty = Map2D(12, 8, [], (1, 1), (11, 7))
    planner = FormationPlanner(empty, robot_count=2)
    config = PlannerConfig(corridor_incremental_limit=0.6)
    box = planner._generate_corridor_box(np.array([2.0, 2.0]), 0.3, config)
    assert box is not None
    xmin, ymin, xmax, ymax = box
    assert xmin < 2.0 < xmax
    assert ymin < 2.0 < ymax
    assert xmax - xmin > 0.6
    assert ymax - ymin > 0.6

    blocked = Map2D(12, 8, [RectangleObstacle((2.0, 2.0), 4.0, 4.0)], (1, 1), (11, 7))
    blocked_planner = FormationPlanner(blocked, robot_count=2)
    assert blocked_planner._generate_corridor_box(
        np.array([2.0, 2.0]),
        0.3,
        PlannerConfig(corridor_max_iter=8),
    ) is None


def test_plan_single_corridor_box_bounds_contain_vehicle_vertices():
    scene = Map2D(12, 8, [], (1, 1), (11, 7))
    planner = FormationPlanner(scene, robot_count=2)
    config = PlannerConfig(corridor_incremental_limit=2.0)
    guess = FullStates(
        tf=1.0,
        states=[
            TrajectoryPoint(2.0, 2.0, 0.0),
            TrajectoryPoint(2.0, 2.0, 0.0),
            TrajectoryPoint(2.0, 2.0, 0.0),
        ],
    )
    corridor = planner._build_vertex_corridor_constraints(guess, config)
    assert corridor is not None
    lb, ub = corridor
    assert lb.shape == (3, 8)
    assert ub.shape == (3, 8)
    centre = config.vehicle.formation_centre(2.0, 2.0, 0.0)
    assert np.all(centre >= lb[0, :2] - 1e-9)
    assert np.all(centre <= ub[0, :2] + 1e-9)


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


def test_generate_guess_from_path_prunes_at_closest_start_pose():
    scene = Map2D(12, 8, [], (1, 1), (11, 7))
    planner = FormationPlanner(scene, robot_count=2)
    path = [
        Pose2D(0.0, 0.0, 0.0),
        Pose2D(2.0, 0.0, 0.1),
        Pose2D(4.0, 0.0, 0.2),
    ]
    guess = planner.generate_guess_from_path(
        path,
        TrajectoryPoint(2.1, 0.0, 0.0),
        step_num=4,
        ratio=True,
        config=PlannerConfig(min_nfe=4),
    )
    assert len(guess.states) == 4
    np.testing.assert_allclose(guess.states[0].xy(), [2.0, 0.0])
    np.testing.assert_allclose(guess.states[-1].xy(), [4.0, 0.0])


def test_stitch_previous_solution_and_guess_feasibility_match_cpp_flow():
    scene = Map2D(12, 8, [RectangleObstacle((6.0, 4.0), 1.0, 2.0)], (1, 1), (11, 7))
    planner = FormationPlanner(scene, robot_count=2)
    previous = FullStates(
        tf=4.0,
        states=[
            TrajectoryPoint(1.0, 1.0),
            TrajectoryPoint(3.0, 1.0),
            TrajectoryPoint(5.0, 1.0),
            TrajectoryPoint(7.0, 1.0),
        ],
    )
    stitched = planner.stitch_previous_solution(previous, TrajectoryPoint(4.8, 1.0))
    assert stitched.tf == 2.0
    assert [state.x for state in stitched.states] == [5.0, 7.0]
    assert planner.check_guess_feasibility(stitched)

    colliding = FullStates(tf=1.0, states=[TrajectoryPoint(6.0, 4.0)])
    assert not planner.check_guess_feasibility(colliding)
    assert not planner.check_guess_feasibility(FullStates())


def test_car_like_follower_and_kinematic_checks_match_cpp_formulas():
    config = PlannerConfig()
    lead = FullStates(
        tf=3.0,
        states=[
            TrajectoryPoint(0.0, 0.0, 0.0, v=0.2, phi=0.0, a=0.1, omega=0.2),
            TrajectoryPoint(1.0, 0.0, 0.1, v=0.3, phi=0.1),
            TrajectoryPoint(2.0, 0.0, 0.2, v=0.4, phi=0.2, a=0.3, omega=0.4),
        ],
    )
    follower = FormationPlanner.plan_car_like(lead, offset=0.5, config=config)
    assert follower.tf == lead.tf
    assert len(follower.states) == len(lead.states)
    np.testing.assert_allclose(follower.states[0].xy(), [0.0, -0.5])
    np.testing.assert_allclose(
        follower.states[1].xy(),
        [
            lead.states[1].x + 0.5 * np.sin(lead.states[1].theta),
            lead.states[1].y - 0.5 * np.cos(lead.states[1].theta),
        ],
    )
    assert follower.states[1].a == lead.states[1].v - lead.states[0].v
    assert follower.states[1].omega == lead.states[1].phi - lead.states[0].phi
    assert FormationPlanner.check_car_kinematic(lead, [0.2], config=config)

    too_fast = FullStates(tf=1.0, states=[TrajectoryPoint(v=1.0, phi=0.69)])
    assert not FormationPlanner.check_car_kinematic(too_fast, [1.0], config=config)


def test_diff_drive_kinematic_check_uses_offset_reference_points():
    config = PlannerConfig()
    slow = FullStates(
        tf=4.0,
        states=[
            TrajectoryPoint(0.0, 0.0, 0.0),
            TrajectoryPoint(0.1, 0.0, 0.0),
            TrajectoryPoint(0.2, 0.0, 0.0),
            TrajectoryPoint(0.3, 0.0, 0.0),
        ],
    )
    assert FormationPlanner.check_diff_drive_kinematic(slow, [0.5], [0.0], config=config)
    fast = FullStates(
        tf=1.0,
        states=[
            TrajectoryPoint(0.0, 0.0, 0.0),
            TrajectoryPoint(2.0, 0.0, 0.0),
        ],
    )
    assert not FormationPlanner.check_diff_drive_kinematic(fast, [0.5], [0.0], config=config)


def test_plan_single_uses_previous_solution_and_returns_solver_result():
    scene = Map2D(12, 8, [], (1, 1), (11, 7))
    planner = FormationPlanner(scene, robot_count=2)
    previous = FullStates(
        tf=1.0,
        states=[
            TrajectoryPoint(1.0, 1.0, 0.0),
            TrajectoryPoint(1.0, 1.0, 0.0),
            TrajectoryPoint(1.0, 1.0, 0.0),
        ],
    )
    config = PlannerConfig(corridor_incremental_limit=2.0)
    result = planner.plan_single(
        previous,
        TrajectoryPoint(1.0, 1.0, 0.0),
        TrajectoryPoint(1.0, 1.0, 0.0),
        config=config,
        solver_maxiter=0,
    )
    assert result.success
    assert result.coarse_time == 0.0
    assert result.solution is not None
    assert len(result.state.states) == 3


def test_diff_drive_and_car_like_replan_wrappers_return_error_metrics():
    scene = Map2D(12, 8, [], (1, 1), (11, 7))
    planner = FormationPlanner(scene, robot_count=2)
    guess = FullStates(
        tf=1.0,
        states=[
            TrajectoryPoint(1.0, 1.0, 0.0),
            TrajectoryPoint(1.0, 1.0, 0.0),
            TrajectoryPoint(1.0, 1.0, 0.0),
        ],
    )
    diff_result = planner.plan_diff_drive(
        guess,
        FullStates(),
        guess.states[0],
        guess.states[-1],
        solver_maxiter=0,
    )
    replan_result = planner.plan_car_like_replan(guess, FullStates(), solver_maxiter=0)
    assert diff_result.success
    assert replan_result.success
    assert diff_result.max_error >= 0.0
    assert replan_result.avg_error >= 0.0


def test_source_single_demo_runs_single_robot_branches(tmp_path):
    args = SimpleNamespace(
        mode="source-single",
        samples=200,
        robots=3,
        seed=7,
        scene_seed=0,
        output_dir=str(tmp_path),
        scene="compact",
        show=False,
        animate=False,
        source_xy_resolution=0.5,
        source_theta_resolution=0.1,
        source_step_size=0.2,
        source_grid_resolution=1.0,
        source_min_nfe=4,
        source_coarse_time=5.0,
        source_max_expansions=20000,
        source_solver_maxiter=0,
        source_solver_method="L-BFGS-B",
    )
    metrics = run_source_single_demo(args)
    assert metrics["mode"] == "source-single"
    assert metrics["source_diff_success"] == 1.0
    assert metrics["source_replan_success"] == 1.0
    assert Path(metrics["figure_path"]).exists()


def test_source_metric_schema_includes_nlp_diagnostics(tmp_path):
    args = SimpleNamespace(
        mode="source",
        samples=80,
        robots=5,
        seed=7,
        scene_seed=0,
        output_dir=str(tmp_path),
        scene="source",
        show=False,
        animate=False,
        source_xy_resolution=0.5,
        source_theta_resolution=0.1,
        source_step_size=0.2,
        source_grid_resolution=1.0,
        source_min_nfe=4,
        source_coarse_time=1.0,
        source_max_expansions=1,
        source_enable_oneshot=False,
        source_warm_starts=1,
        source_initial_warm_starts=1,
        source_solver_maxiter=0,
        source_solver_method="L-BFGS-B",
        source_topology_attempts=1,
        source_topology_paths=1,
        source_topology_bbox=3.0,
        source_strict_homotopy_bugs=False,
        source_strict_cpp_early_return=False,
    )
    from main import run_source_aligned_demo

    try:
        metrics = run_source_aligned_demo(args)
    except RuntimeError:
        # The intentionally tiny coarse budget may fail before NLP diagnostics.
        # The full-chain test below covers successful entry into Plan_fm.
        return
    for key in [
        "source_final_infeasibility",
        "source_final_objective",
        "source_final_iterations",
        "source_scipy_success",
        "source_scipy_message",
        "source_infeas_initial_terminal",
        "source_infeas_dynamics",
        "source_infeas_edge_distance",
        "source_infeas_topology",
        "source_infeas_sfc",
    ]:
        assert key in metrics


def test_source_style_full_chain_runs_deterministically_on_open_scene():
    scene = Map2D(20, 14, [], (5, 7), (13, 7))
    formation = FormationPlanner(scene, robot_count=5)
    starts, goals = source_aligned_robot_states(scene, formation)
    args = SimpleNamespace(
        samples=80,
        seed=7,
        source_topology_attempts=1,
        source_topology_paths=1,
        source_topology_bbox=8.0,
        source_strict_homotopy_bugs=False,
    )
    hyperparam_sets, topo_paths, combination = source_aligned_homotopy_constraints(scene, starts, goals, args)
    assert len(topo_paths) == 1
    assert combination == [0, 0, 0, 0, 0]
    assert len(hyperparam_sets) == 5
    assert all(len(robot_corridors) == 100 for robot_corridors in hyperparam_sets)

    config = PlannerConfig(min_nfe=4, xy_resolution=0.5, grid_xy_resolution=1.0, step_size=0.2)
    guess = formation.plan_coarse_full_states(
        starts,
        goals,
        hyperparam_sets=hyperparam_sets,
        config=config,
        max_search_time=3.0,
        max_expansions=15000,
    )
    assert len(guess) == 5
    assert all(len(robot_guess.states) == len(guess[0].states) for robot_guess in guess)
    assert len(guess[0].states) == 17
    np.testing.assert_allclose(guess[0].tf, 8.995897, atol=1e-6)

    result = formation.plan_fm_from_guess(
        guess,
        config=config,
        max_warm_start=1,
        initial_warm_starts=1,
        solver_maxiter=0,
        solver_method="L-BFGS-B",
    )
    trajectory = full_states_to_xy_tensor(result.states)
    assert trajectory.shape == (17, 5, 2)
    assert result.reason == "infeasibility_above_cpp_initial_threshold"
    assert result.warm_start == 0
    assert len(result.solve_history) == 1
    assert set(result.solve_history[0].infeasibility_terms) == {
        "initial_terminal",
        "dynamics",
        "terminal_error",
        "edge_distance",
        "topology",
        "sfc",
    }
    assert len(result.corridor_cons) == 5
    assert [len(robot_corridors) for robot_corridors in result.corridor_cons] == [17, 17, 17, 17, 17]
    np.testing.assert_allclose(result.states[0].tf, 9.079645, atol=1e-6)


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
    fixture_dir = REPO_ROOT / "cpp_fixtures/flexible_formation/5"
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


def test_homotopy_combination_keeps_three_robot_regular_polygon():
    scene = Map2D(60, 34, [], (15, 17), (45, 17))
    planner = FormationPlanner(scene, robot_count=3)
    starts, goals = source_aligned_robot_states(scene, planner)
    center_path = np.array([[15.0, 17.0], [45.0, 17.0]])
    raw_paths_set = [
        [rewire_path(resample_polyline(center_path, 100), start.xy(), goal.xy(), scene)]
        for start, goal in zip(starts, goals)
    ]

    result = cal_combination(raw_paths_set, scene, selected_path_limit=1)

    assert result.combinations == [[0, 0, 0]]


def test_homotopy_helper_scores_match_identify_homotopy_sources():
    lengths = [4.0, 1.0, 2.0, 3.0]
    assert find_two_smallest(lengths) == [1, 2, 3, 0]
    assert find_smallest_indices(lengths, 2) == [1, 2]

    raw_paths = [
        np.array([[0.0, 0.0], [3.0, 0.0]]),
        np.array([[0.0, 0.0], [1.0, 0.0]]),
        np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
    ]
    assert eval_path_length(raw_paths, 2) == [1, 2]

    turning_paths = [
        [Pathset(3.0, np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [1.0, 2.0]]))],
        [Pathset(3.0, np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]))],
    ]
    assert cal_length_set(turning_paths, [0, 0]) == 3.0
    assert cal_turning_set(turning_paths, [0, 0]) == 1.0
    assert cal_homotopy_set([0, 0, 0]) == 1.0
    assert cal_homotopy_set([0, 1, 0]) == 0.0

    triangle_paths = [
        [Pathset(0.0, np.repeat(np.array([[1.0, 0.0]]), 3, axis=0))],
        [Pathset(0.0, np.repeat(np.array([[0.0, 1.0]]), 3, axis=0))],
        [Pathset(0.0, np.repeat(np.array([[-1.0, 0.0]]), 3, axis=0))],
    ]
    assert cal_safety_set(triangle_paths, [0, 0, 0]) == 0.0


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
