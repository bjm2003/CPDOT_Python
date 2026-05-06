import numpy as np

from cpdot_py import CircleObstacle, FormationPlanner, Map2D, RectangleObstacle, TopologyPRM
from cpdot_py.forward_kinematics import ForwardKinematics
from cpdot_py.geometry import resample_polyline


def test_collision_segment_circle():
    scene = Map2D(10, 10, [CircleObstacle((5, 5), 1.0)], (1, 1), (9, 9))
    assert scene.is_collision((5, 5))
    assert not scene.segment_is_collision_free((1, 5), (9, 5))
    assert scene.segment_is_collision_free((1, 1), (2, 1))


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
