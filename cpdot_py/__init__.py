"""Lightweight Python reproduction of the core CPDOT planning ideas."""

from .coarse_path_planner import CoarsePathPlanner, Pose2D, poses_to_array
from .env import CircleObstacle, Map2D, PolygonObstacle, RectangleObstacle
from .formation import (
    FormationPlanner,
    PlanFmResult,
    full_states_to_xy_tensor,
    generate_desired_rp,
    generate_optimal_time_profile_segment,
    regular_polygon,
    resample_path_to_full_states,
    xy_tensor_to_full_states,
)
from .forward_kinematics import ForwardKinematics
from .homotopy import CombinationResult, Pathset, cal_combination, cal_corridors, rewire_path
from .optimizer import FormationNLPProblem, FormationNLPSolution, PlannerConfig, VehicleModel, solve_fm
from .sfc import generate_sfc
from .states import CPDOT_FORMATION_ROBOTS, Constraints, FullStates, TrajectoryPoint
from .topo_prm import TopologyPRM

__all__ = [
    "CircleObstacle",
    "CPDOT_FORMATION_ROBOTS",
    "CoarsePathPlanner",
    "CombinationResult",
    "Constraints",
    "FormationPlanner",
    "FormationNLPProblem",
    "FormationNLPSolution",
    "ForwardKinematics",
    "FullStates",
    "Map2D",
    "Pathset",
    "PlanFmResult",
    "PolygonObstacle",
    "PlannerConfig",
    "Pose2D",
    "RectangleObstacle",
    "TopologyPRM",
    "TrajectoryPoint",
    "VehicleModel",
    "cal_combination",
    "cal_corridors",
    "full_states_to_xy_tensor",
    "generate_desired_rp",
    "generate_optimal_time_profile_segment",
    "generate_sfc",
    "poses_to_array",
    "regular_polygon",
    "resample_path_to_full_states",
    "rewire_path",
    "solve_fm",
    "xy_tensor_to_full_states",
]
