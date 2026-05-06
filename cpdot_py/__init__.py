"""Lightweight Python reproduction of the core CPDOT planning ideas."""

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
from .optimizer import FormationNLPProblem, FormationNLPSolution, PlannerConfig, VehicleModel, solve_fm
from .sfc import generate_sfc
from .states import CPDOT_FORMATION_ROBOTS, Constraints, FullStates, TrajectoryPoint
from .topo_prm import TopologyPRM

__all__ = [
    "CircleObstacle",
    "CPDOT_FORMATION_ROBOTS",
    "Constraints",
    "FormationPlanner",
    "FormationNLPProblem",
    "FormationNLPSolution",
    "ForwardKinematics",
    "FullStates",
    "Map2D",
    "PlanFmResult",
    "PolygonObstacle",
    "PlannerConfig",
    "RectangleObstacle",
    "TopologyPRM",
    "TrajectoryPoint",
    "VehicleModel",
    "full_states_to_xy_tensor",
    "generate_desired_rp",
    "generate_optimal_time_profile_segment",
    "generate_sfc",
    "regular_polygon",
    "resample_path_to_full_states",
    "solve_fm",
    "xy_tensor_to_full_states",
]
