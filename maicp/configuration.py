"""Load and validate the case study configurations.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from math import isfinite
from maicp.ranks import binomial_calibration_rank


# ---------------------------------------------------------------------------
# Primary experiment switches
# ---------------------------------------------------------------------------
EXPERIMENT_PROFILE = "paper"  # "smoke" or "paper"
NUM_SUBTEAMS = 4
UGVS_PER_SUBTEAM = 5
UAVS_PER_SUBTEAM = 5
UAV_DYNAMICS = "double_integrator"  # "velocity_controlled" or "double_integrator"
SENSITIVITY_MODE = "configured"  # "analytic", "configured", or "empirical"
RUN_SENSITIVITY_DIAGNOSTICS = False

# Formation margins measure acceleration error in m/s². Tracking margins are dimensionless.
FORMATION_INITIAL_MARGIN_UGV = 0.2
FORMATION_INITIAL_MARGIN_UAV = 0.3
TRACKING_INITIAL_MARGIN_UGV = 4.0
TRACKING_INITIAL_MARGIN_UAV = 2.0

# Use zero for automatic worker selection and one for serial execution.
MISSION_WORKERS = 0

# Full QP histories for every robot at every time step are useful for traced
# debugging waves but are too large for routine calibration and paper runs.
RECORD_FULL_QP_DIAGNOSTICS = False

# Formation obstacle safety uses a robust HOCBF quadratic program with hard constraints at each robot.
# Wang remains an explicit regression/comparison option.
COLLISION_CONTROLLER = "hocbf"  # "hocbf", "wang", "mestres", or "projection"
HOCBF_LAMBDA = 0.5

# Acceleration and speed limits for the planar double integrator.
WANG_MAX_ACCEL_UGV = 2.0
WANG_MAX_ACCEL_UAV = 4.0
WANG_MAX_SPEED_UGV = 8.0
WANG_MAX_SPEED_UAV = 14.0
WANG_OPTIMAL_DECAY_CBF = True
WANG_CBF_GAMMA_NOMINAL = 1.0
WANG_CBF_DECAY_WEIGHT = 10.0
WANG_CBF_DECAY_MULTIPLIER_MIN = 0.1
WANG_CBF_DECAY_MULTIPLIER_MAX = 10.0
WANG_POSITION_KP_UGV = 0.8
WANG_POSITION_KP_UAV = 1.0
WANG_VELOCITY_KD_UGV = 1.2
WANG_VELOCITY_KD_UAV = 1.5

# Mission and communication geometry
MAP_HALF_EXTENT = 48.0
QUADRANT_ANCHOR_DISTANCE = 29.0
DEPOT_PAD_DISTANCE = 4.0
TASK_COMMUNICATION_RANGE = 30.0
UPDATE_COMMUNICATION_RANGE = 12.0
HOME_TOLERANCE = 0.8
HOME_DWELL_STEPS = 3

# Parameters for computing distributed order statistics.
PINBALL_GAMMA_0 = 0.05
PINBALL_EXPONENT = 0.75

# The upper margin limit defines the controller's operating range. Feasibility depends on the calibration batch.
COLLISION_CBF_MARGIN_MAX = 4.0
COLLISION_OPTIMIZER_ITERATIONS = 3
COLLISION_OPTIMIZER_STEP = 0.05
COLLISION_OPTIMIZER_REGULARIZATION = 0.001
# Scale every synthetic field frequency, preserving all term amplitudes.
COLLISION_ERROR_FIELD_FREQUENCY_SCALE = 1.5
COLLISION_INITIAL_POSITION_RADIUS = 0.5
# Formation center estimates use a fixed communication ring.
COLLISION_COMMUNICATION_RANGE = 14.5
COLLISION_MOTION_NOISE_SIGMA_UGV = 0.01
COLLISION_MOTION_NOISE_SIGMA_UAV = 0.01
COLLISION_MOTION_NOISE_MAX_UGV = 0.03
COLLISION_MOTION_NOISE_MAX_UAV = 0.03
# The deployment gain scales the true neighbor disturbance. The predictor was trained at unit gain.
COLLISION_NEIGHBOR_RADIUS = 6.0
COLLISION_NEIGHBOR_GAIN = 2.0
COLLISION_SCENARIO = "formation_obstacles_v1"
COLLISION_FORMATION_AGENT_RADIUS = 0.45
COLLISION_FORMATION_ROBOTS_PER_CLASS = 3
FORMATION_OBSTACLE_COUNT = 6
# Select the obstacle sampling distribution at mission initialization.
FORMATION_OBSTACLE_LAYOUT = "random"
FORMATION_GATE_RADIUS = 1.4
FORMATION_GATE_PENETRATION = 0.3
FORMATION_GATE_JITTER_M = 0.3
# The disturbance from nearby robots depends on obstacle exposure in the other class.
FORMATION_OBSTACLE_DISTURBANCE_MODE = "density"
# Scale the difference between the learned and true dynamics for this mission.
MISSION_DYNAMICS_SEVERITY_SIGMA = 1.0

# Formation reference and center consensus parameters.
FORMATION_CENTER_START = (7.0, 7.0)
FORMATION_CENTER_GOAL = (25.0, 21.0)
FORMATION_CENTER_DURATION_S = 50.0
# Each formation mission starts from its prescribed slots and runs for a fixed horizon.
FORMATION_OUTBOUND_DEADLINE_S = 50.0
FORMATION_MISSION_STEPS = 250
FORMATION_RADIUS = 4.0
FORMATION_CONSENSUS_ROUNDS = 1
FORMATION_CENTER_VARIANT = "distributed"  # distributed, exact, or prescribed
FORMATION_GOAL_GAIN_FRACTION = 0.5


class Platform(StrEnum):
    UGV = "ugv"
    UAV = "uav"


class UAVDynamics(StrEnum):
    VELOCITY_CONTROLLED = "velocity_controlled"
    DOUBLE_INTEGRATOR = "double_integrator"


class SensitivityMode(StrEnum):
    ANALYTIC = "analytic"
    CONFIGURED = "configured"
    EMPIRICAL = "empirical"


class CollisionController(StrEnum):
    """Choice of collision avoidance controller."""

    HOCBF = "hocbf"
    WANG = "wang"
    MESTRES = "mestres"
    PROJECTION = "projection"


@dataclass(frozen=True)
class ClassConfig:
    platform: Platform
    robots_per_mission: int
    alpha: float
    delta_cal: float
    violations: int
    scale: float
    weight: float
    initial_margin: float
    margin_min: float
    margin_max: float
    tracking_beta: float
    configured_kappa: float

    @property
    def mission_rank(self) -> int:
        return self.robots_per_mission - self.violations


@dataclass(frozen=True)
class ProfileConfig:
    name: str
    outer_replicates: int
    rounds: int
    calibration_missions: int
    deployment_missions: int
    mission_pinball_steps: int
    threshold_pinball_steps: int
    simulation_steps: int
    tracking_admm_steps: int
    ugv_alpha: float
    uav_alpha: float
    delta_cal: float


@dataclass(frozen=True)
class ExperimentConfig:
    profile: ProfileConfig
    classes: dict[Platform, ClassConfig]
    num_subteams: int
    uav_dynamics: UAVDynamics
    sensitivity_mode: SensitivityMode
    collision_controller: CollisionController
    map_half_extent: float
    quadrant_anchor_distance: float
    depot_pad_distance: float
    task_communication_range: float
    update_communication_range: float
    home_tolerance: float
    home_dwell_steps: int
    pinball_gamma_0: float
    pinball_exponent: float
    seed: int = 27
    proximal_weight: float = 0.25
    margin_consensus_tolerance: float = 1e-7
    collision_cbf_margin_max: float = COLLISION_CBF_MARGIN_MAX
    collision_optimizer_iterations: int = COLLISION_OPTIMIZER_ITERATIONS
    collision_optimizer_step: float = COLLISION_OPTIMIZER_STEP
    collision_optimizer_regularization: float = COLLISION_OPTIMIZER_REGULARIZATION
    collision_error_field_frequency_scale: float = COLLISION_ERROR_FIELD_FREQUENCY_SCALE
    collision_initial_position_radius: float = COLLISION_INITIAL_POSITION_RADIUS
    collision_communication_range: float = COLLISION_COMMUNICATION_RANGE
    collision_motion_noise_sigma_ugv: float = COLLISION_MOTION_NOISE_SIGMA_UGV
    collision_motion_noise_sigma_uav: float = COLLISION_MOTION_NOISE_SIGMA_UAV
    collision_motion_noise_max_ugv: float = COLLISION_MOTION_NOISE_MAX_UGV
    collision_motion_noise_max_uav: float = COLLISION_MOTION_NOISE_MAX_UAV
    collision_neighbor_radius: float = COLLISION_NEIGHBOR_RADIUS
    collision_neighbor_gain: float = COLLISION_NEIGHBOR_GAIN
    collision_scenario: str = COLLISION_SCENARIO
    collision_formation_agent_radius: float = COLLISION_FORMATION_AGENT_RADIUS
    formation_pillar_positions: tuple[tuple[float, float], ...] | None = None
    formation_obstacle_count: int = FORMATION_OBSTACLE_COUNT
    formation_obstacle_layout: str = FORMATION_OBSTACLE_LAYOUT
    formation_gate_radius: float = FORMATION_GATE_RADIUS
    formation_gate_penetration: float = FORMATION_GATE_PENETRATION
    formation_gate_jitter_m: float = FORMATION_GATE_JITTER_M
    obstacle_position_sigma_m: float = 4.0
    mission_dynamics_severity_sigma: float = MISSION_DYNAMICS_SEVERITY_SIGMA
    formation_obstacle_disturbance_max: float = 0.0
    formation_obstacle_disturbance_band_m: float = 1.5
    formation_obstacle_disturbance_mode: str = FORMATION_OBSTACLE_DISTURBANCE_MODE
    hocbf_lambda: float = HOCBF_LAMBDA
    wang_max_accel_ugv: float = WANG_MAX_ACCEL_UGV
    wang_max_accel_uav: float = WANG_MAX_ACCEL_UAV
    wang_max_speed_ugv: float = WANG_MAX_SPEED_UGV
    wang_max_speed_uav: float = WANG_MAX_SPEED_UAV
    wang_optimal_decay_cbf: bool = WANG_OPTIMAL_DECAY_CBF
    wang_cbf_gamma_nominal: float = WANG_CBF_GAMMA_NOMINAL
    wang_cbf_decay_weight: float = WANG_CBF_DECAY_WEIGHT
    wang_cbf_decay_multiplier_min: float = WANG_CBF_DECAY_MULTIPLIER_MIN
    wang_cbf_decay_multiplier_max: float = WANG_CBF_DECAY_MULTIPLIER_MAX
    wang_position_kp_ugv: float = WANG_POSITION_KP_UGV
    wang_position_kp_uav: float = WANG_POSITION_KP_UAV
    wang_velocity_kd_ugv: float = WANG_VELOCITY_KD_UGV
    wang_velocity_kd_uav: float = WANG_VELOCITY_KD_UAV
    mission_workers: int = MISSION_WORKERS
    record_full_qp_diagnostics: bool = RECORD_FULL_QP_DIAGNOSTICS
    run_sensitivity_diagnostics: bool = RUN_SENSITIVITY_DIAGNOSTICS
    formation_center_start: tuple[float, float] = FORMATION_CENTER_START
    formation_center_goal: tuple[float, float] = FORMATION_CENTER_GOAL
    formation_center_duration_s: float = FORMATION_CENTER_DURATION_S
    formation_outbound_deadline_s: float = FORMATION_OUTBOUND_DEADLINE_S
    formation_stop_when_home: bool = False
    formation_radius: float = FORMATION_RADIUS
    center_consensus_rounds: int = FORMATION_CONSENSUS_ROUNDS
    center_variant: str = FORMATION_CENTER_VARIANT
    formation_goal_gain_fraction: float = FORMATION_GOAL_GAIN_FRACTION

    @property
    def wang_max_accel(self) -> dict[Platform, float]:
        """Wang acceleration bounds by class."""

        return {
            Platform.UGV: self.wang_max_accel_ugv,
            Platform.UAV: self.wang_max_accel_uav,
        }

    @property
    def wang_max_speed(self) -> dict[Platform, float]:
        """Wang speed bounds by class."""

        return {
            Platform.UGV: self.wang_max_speed_ugv,
            Platform.UAV: self.wang_max_speed_uav,
        }

    @property
    def wang_position_kp(self) -> dict[Platform, float]:
        """Proportional gains for Wang nominal control by class."""

        return {
            Platform.UGV: self.wang_position_kp_ugv,
            Platform.UAV: self.wang_position_kp_uav,
        }

    @property
    def wang_velocity_kd(self) -> dict[Platform, float]:
        """Derivative gains for Wang nominal control by class."""

        return {
            Platform.UGV: self.wang_velocity_kd_ugv,
            Platform.UAV: self.wang_velocity_kd_uav,
        }

    def validate(self) -> None:
        if self.formation_pillar_positions is not None:
            positions = self.formation_pillar_positions
            if len(positions) != 3 or any(len(p) != 2 or not all(isfinite(float(x)) for x in p) for p in positions):
                raise ValueError("formation_pillar_positions must contain three finite planar positions")
        if not isfinite(float(self.hocbf_lambda)) or self.hocbf_lambda <= 0.0:
            raise ValueError("hocbf_lambda must be finite and positive")
        try:
            CollisionController(self.collision_controller)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "collision_controller must be one of: "
                + ", ".join(controller.value for controller in CollisionController)
            ) from exc
        if self.num_subteams != 4:
            raise ValueError("The quadrant experiment requires exactly four subteams")
        if set(self.classes) != set(Platform):
            raise ValueError("Both UGV and UAV class configurations are required")
        if min(
            self.map_half_extent,
            self.quadrant_anchor_distance,
            self.task_communication_range,
            self.update_communication_range,
            self.home_tolerance,
        ) <= 0.0 or self.home_dwell_steps < 1:
            raise ValueError("Mission geometry and dwell parameters must be positive")
        if self.profile.calibration_missions <= 0 or self.profile.deployment_missions <= 0:
            raise ValueError("Mission sample counts must be positive")
        if self.profile.calibration_missions % self.num_subteams:
            raise ValueError("Calibration mission count must be divisible by NUM_SUBTEAMS")
        if self.profile.deployment_missions % self.num_subteams:
            raise ValueError("Deployment mission count must be divisible by NUM_SUBTEAMS")
        if not 0.5 < self.pinball_exponent <= 1.0:
            raise ValueError("Pinball exponent must make the step sizes square summable")
        if self.pinball_gamma_0 <= 0.0 or self.proximal_weight <= 0.0:
            raise ValueError("Pinball and strongly convex objective weights must be positive")
        if not 0.0 <= self.margin_consensus_tolerance < float("inf"):
            raise ValueError("Margin consensus tolerance must be finite and nonnegative")
        if not 0.0 < self.collision_cbf_margin_max < float("inf"):
            raise ValueError("Collision CBF margin ceiling must be finite and positive")
        if self.collision_optimizer_iterations < 0:
            raise ValueError("Collision optimizer iterations must be nonnegative")
        if not 0.0 < self.collision_optimizer_step < float("inf"):
            raise ValueError("Collision optimizer step must be finite and positive")
        if not 0.0 < self.collision_optimizer_regularization < float("inf"):
            raise ValueError("Collision optimizer regularization must be finite and positive")
        if not 0.0 < self.collision_error_field_frequency_scale < float("inf"):
            raise ValueError("Collision field frequency scale must be finite and positive")
        if not 0.0 <= self.collision_initial_position_radius < float("inf"):
            raise ValueError("Collision initial-position radius must be finite and nonnegative")
        if not 0.0 < self.collision_communication_range < float("inf"):
            raise ValueError("Collision communication range must be finite and positive")
        if not 0.0 < self.collision_neighbor_radius <= self.collision_communication_range:
            raise ValueError("Neighbor radius must be positive and within communication range")
        if not 0.0 <= self.collision_neighbor_gain < float("inf"):
            raise ValueError("Neighbor deployment gain must be finite and nonnegative")
        if self.collision_scenario != COLLISION_SCENARIO:
            raise ValueError(
                f"Collision scenario must be {COLLISION_SCENARIO!r}"
            )
        if not 0.0 < self.collision_formation_agent_radius < float("inf"):
            raise ValueError("Formation agent radius must be finite and positive")
        if (
            isinstance(self.formation_obstacle_count, bool)
            or not isinstance(self.formation_obstacle_count, int)
            or self.formation_obstacle_count < 3
        ):
            raise ValueError("formation_obstacle_count must be an integer >= 3")
        if self.formation_obstacle_layout not in {"random", "slot_gates"}:
            raise ValueError(
                "formation_obstacle_layout must be 'random' or 'slot_gates'"
            )
        if self.formation_obstacle_layout == "slot_gates" and self.formation_obstacle_count != 6:
            raise ValueError("slot_gates requires exactly six formation obstacles")
        if (
            not isfinite(float(self.formation_gate_radius))
            or not 1.0 <= float(self.formation_gate_radius) <= 1.6
        ):
            raise ValueError("formation_gate_radius must lie in [1.0, 1.6]")
        if (
            not isfinite(float(self.formation_gate_penetration))
            or not 0.15 <= float(self.formation_gate_penetration) <= 0.6
        ):
            raise ValueError("formation_gate_penetration must lie in [0.15, 0.6]")
        if (
            not isfinite(float(self.formation_gate_jitter_m))
            or float(self.formation_gate_jitter_m) < 0.0
        ):
            raise ValueError(
                "formation_gate_jitter_m must be finite and nonnegative"
            )
        if not 0.0 <= self.obstacle_position_sigma_m < float("inf"):
            raise ValueError("obstacle_position_sigma_m must be finite and nonnegative")
        if not 0.0 <= self.formation_obstacle_disturbance_max < float("inf"):
            raise ValueError("formation_obstacle_disturbance_max must be finite and nonnegative")
        if not 0.0 < self.formation_obstacle_disturbance_band_m < float("inf"):
            raise ValueError("formation_obstacle_disturbance_band_m must be finite and positive")
        if self.formation_obstacle_disturbance_mode not in {"density", "peer_encounter"}:
            raise ValueError(
                "formation_obstacle_disturbance_mode must be 'density' or 'peer_encounter'"
            )
        if (
            not isfinite(float(self.mission_dynamics_severity_sigma))
            or not 0.0 <= float(self.mission_dynamics_severity_sigma) <= 1.0
        ):
            raise ValueError(
                "mission_dynamics_severity_sigma must be finite and lie in [0, 1]"
            )
        for name, value in (
            ("formation_center_start", self.formation_center_start),
            ("formation_center_goal", self.formation_center_goal),
        ):
            vector = tuple(value)
            if len(vector) != 2 or not all(isfinite(float(component)) for component in vector):
                raise ValueError(f"{name} must be a finite planar endpoint")
        if not isfinite(float(self.formation_center_duration_s)) or self.formation_center_duration_s <= 0.0:
            raise ValueError("formation_center_duration_s must be finite and positive")
        if (not isfinite(float(self.formation_outbound_deadline_s))
                or self.formation_outbound_deadline_s < self.formation_center_duration_s):
            raise ValueError("formation_outbound_deadline_s must be finite and at least the outbound duration")
        if not isfinite(float(self.formation_radius)) or self.formation_radius <= 0.0:
            raise ValueError("formation_radius must be finite and positive")
        if (
            isinstance(self.center_consensus_rounds, bool)
            or not isinstance(self.center_consensus_rounds, int)
            or self.center_consensus_rounds < 1
        ):
            raise ValueError("center_consensus_rounds must be a positive integer")
        if self.center_variant not in {"distributed", "exact", "prescribed"}:
            raise ValueError(
                "center_variant must be distributed, exact, or prescribed"
            )
        if (
            not isfinite(float(self.formation_goal_gain_fraction))
            or self.formation_goal_gain_fraction <= 0.0
            or self.formation_goal_gain_fraction >= 1.0
        ):
            raise ValueError("formation_goal_gain_fraction must lie in (0, 1)")
        for platform, sigma, maximum in (
            (Platform.UGV, self.collision_motion_noise_sigma_ugv,
             self.collision_motion_noise_max_ugv),
            (Platform.UAV, self.collision_motion_noise_sigma_uav,
             self.collision_motion_noise_max_uav),
        ):
            if not 0.0 <= sigma < float("inf"):
                raise ValueError(
                    f"Collision motion-noise sigma must be finite and nonnegative for {platform}"
                )
            if not 0.0 < maximum < float("inf"):
                raise ValueError(
                    f"Collision motion-noise maximum must be finite and positive for {platform}"
                )
        for name, value in (
            ("wang_max_accel_ugv", self.wang_max_accel_ugv),
            ("wang_max_accel_uav", self.wang_max_accel_uav),
            ("wang_max_speed_ugv", self.wang_max_speed_ugv),
            ("wang_max_speed_uav", self.wang_max_speed_uav),
            ("wang_cbf_gamma_nominal", self.wang_cbf_gamma_nominal),
            ("wang_cbf_decay_weight", self.wang_cbf_decay_weight),
            ("wang_position_kp_ugv", self.wang_position_kp_ugv),
            ("wang_position_kp_uav", self.wang_position_kp_uav),
            ("wang_velocity_kd_ugv", self.wang_velocity_kd_ugv),
            ("wang_velocity_kd_uav", self.wang_velocity_kd_uav),
        ):
            if not isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not isinstance(self.wang_optimal_decay_cbf, bool):
            raise ValueError("wang_optimal_decay_cbf must be boolean")
        if not isinstance(self.mission_workers, int) or self.mission_workers < 0:
            raise ValueError("mission_workers must be a nonnegative integer")
        if not isinstance(self.record_full_qp_diagnostics, bool):
            raise ValueError("record_full_qp_diagnostics must be boolean")
        if not isinstance(self.run_sensitivity_diagnostics, bool):
            raise ValueError("run_sensitivity_diagnostics must be boolean")
        if (
            not isfinite(float(self.wang_cbf_decay_multiplier_min))
            or float(self.wang_cbf_decay_multiplier_min) < 0.0
            or not isfinite(float(self.wang_cbf_decay_multiplier_max))
            or float(self.wang_cbf_decay_multiplier_max)
            < float(self.wang_cbf_decay_multiplier_min)
        ):
            raise ValueError(
                "Wang decay multiplier bounds must be finite, nonnegative, and ordered"
            )
        for cls in self.classes.values():
            if not 0.0 < cls.alpha < 1.0:
                raise ValueError(f"alpha must lie in (0,1) for {cls.platform}")
            binomial_calibration_rank(
                self.profile.calibration_missions, cls.alpha, cls.delta_cal
            )
            if not 1 <= cls.mission_rank <= cls.robots_per_mission:
                raise ValueError(f"Invalid mission rank for {cls.platform}")
            if not (
                0.0 < cls.scale
                and 0.0 < cls.weight
                and cls.margin_min <= cls.initial_margin <= cls.margin_max
            ):
                raise ValueError(f"Invalid scale or margin interval for {cls.platform}")


PROFILES = {
    "smoke": ProfileConfig(
        name="smoke",
        outer_replicates=1,
        rounds=2,
        calibration_missions=8,
        deployment_missions=8,
        mission_pinball_steps=5,
        threshold_pinball_steps=5,
        simulation_steps=320,
        tracking_admm_steps=10,
        ugv_alpha=0.50,
        uav_alpha=0.50,
        delta_cal=0.99,
    ),
    "paper": ProfileConfig(
        name="paper",
        outer_replicates=1,
        rounds=5,
        calibration_missions=200,
        deployment_missions=200,
        mission_pinball_steps=100,
        threshold_pinball_steps=100,
        simulation_steps=320,
        tracking_admm_steps=50,
        ugv_alpha=0.10,
        uav_alpha=0.10,
        delta_cal=0.99,
    ),
}


def build_config(
    profile_name: str | None = None,
    sensitivity_mode: str | SensitivityMode | None = None,
    uav_dynamics: str | UAVDynamics | None = None,
    collision_controller: str | CollisionController | None = None,
    *,
    controller: str | CollisionController | None = None,
    collision_scenario: str | None = None,
    formation_obstacle_count: int | None = None,
    formation_obstacle_layout: str | None = None,
    formation_gate_radius: float | None = None,
    formation_gate_penetration: float | None = None,
    formation_gate_jitter_m: float | None = None,
    formation_obstacle_disturbance_mode: str | None = None,
) -> ExperimentConfig:
    if collision_controller is not None and controller is not None:
        if CollisionController(collision_controller) is not CollisionController(controller):
            raise ValueError("Specify only one conflicting collision-controller override")
    selected_controller = (
        collision_controller
        if collision_controller is not None
        else controller
    )
    selected_profile = EXPERIMENT_PROFILE if profile_name is None else profile_name
    if selected_profile not in PROFILES:
        raise ValueError(f"Unknown experiment profile: {selected_profile!r}")
    profile = PROFILES[selected_profile]
    # Tracking uses five robots per class. Formation uses three.
    scenario_explicit = collision_scenario is not None
    scenario = COLLISION_SCENARIO if collision_scenario is None else collision_scenario
    if scenario != COLLISION_SCENARIO:
        raise ValueError(
            f"Unknown collision scenario {scenario!r}; expected {COLLISION_SCENARIO!r}"
        )
    if scenario_explicit:
        profile = replace(profile, simulation_steps=FORMATION_MISSION_STEPS)
    ugv_count = (
        COLLISION_FORMATION_ROBOTS_PER_CLASS
        if scenario_explicit
        else UGVS_PER_SUBTEAM
    )
    uav_count = (
        COLLISION_FORMATION_ROBOTS_PER_CLASS
        if scenario_explicit
        else UAVS_PER_SUBTEAM
    )
    classes = {
        Platform.UGV: ClassConfig(
            platform=Platform.UGV,
            robots_per_mission=ugv_count,
            alpha=profile.ugv_alpha,
            delta_cal=profile.delta_cal,
            violations=1,
            scale=1.0,
            weight=1.0,
            initial_margin=(
                FORMATION_INITIAL_MARGIN_UGV
                if scenario_explicit else TRACKING_INITIAL_MARGIN_UGV
            ),
            margin_min=0.0,
            margin_max=100.0,
            tracking_beta=0.30 if scenario_explicit else 0.020,
            configured_kappa=0.3,
        ),
        Platform.UAV: ClassConfig(
            platform=Platform.UAV,
            robots_per_mission=uav_count,
            alpha=profile.uav_alpha,
            delta_cal=profile.delta_cal,
            violations=0 if scenario_explicit else 1,
            scale=1.0,
            weight=1.0,
            initial_margin=(
                FORMATION_INITIAL_MARGIN_UAV
                if scenario_explicit else TRACKING_INITIAL_MARGIN_UAV
            ),
            margin_min=0.0,
            margin_max=100.0,
            tracking_beta=0.20 if scenario_explicit else 0.050,
            configured_kappa=0.3,
        ),
    }
    config = ExperimentConfig(
        profile=profile,
        classes=classes,
        num_subteams=NUM_SUBTEAMS,
        uav_dynamics=UAVDynamics(UAV_DYNAMICS if uav_dynamics is None else uav_dynamics),
        sensitivity_mode=SensitivityMode(SENSITIVITY_MODE if sensitivity_mode is None else sensitivity_mode),
        collision_controller=CollisionController(
            COLLISION_CONTROLLER if selected_controller is None else selected_controller
        ),
        collision_scenario=scenario,
        formation_obstacle_count=(
            FORMATION_OBSTACLE_COUNT
            if formation_obstacle_count is None
            else formation_obstacle_count
        ),
        formation_obstacle_layout=(
            FORMATION_OBSTACLE_LAYOUT
            if formation_obstacle_layout is None
            else formation_obstacle_layout
        ),
        formation_gate_radius=(
            FORMATION_GATE_RADIUS
            if formation_gate_radius is None
            else formation_gate_radius
        ),
        formation_gate_penetration=(
            FORMATION_GATE_PENETRATION
            if formation_gate_penetration is None
            else formation_gate_penetration
        ),
        formation_gate_jitter_m=(
            FORMATION_GATE_JITTER_M
            if formation_gate_jitter_m is None
            else formation_gate_jitter_m
        ),
        formation_obstacle_disturbance_mode=(
            FORMATION_OBSTACLE_DISTURBANCE_MODE
            if formation_obstacle_disturbance_mode is None
            else formation_obstacle_disturbance_mode
        ),
        map_half_extent=MAP_HALF_EXTENT,
        quadrant_anchor_distance=QUADRANT_ANCHOR_DISTANCE,
        depot_pad_distance=DEPOT_PAD_DISTANCE,
        task_communication_range=TASK_COMMUNICATION_RANGE,
        update_communication_range=UPDATE_COMMUNICATION_RANGE,
        home_tolerance=HOME_TOLERANCE,
        home_dwell_steps=HOME_DWELL_STEPS,
        pinball_gamma_0=PINBALL_GAMMA_0,
        pinball_exponent=PINBALL_EXPONENT,
    )
    config.validate()
    return config


def load_config(case):
    from pathlib import Path
    import json

    values = json.loads((Path(__file__).resolve().parent.parent / "experiments" / case / "config.json").read_text())
    values["classes"] = {
        Platform(key): ClassConfig(**{**value, "platform": Platform(key)})
        for key, value in values["classes"].items()
    }
    values["profile"] = ProfileConfig(**values["profile"])
    values["uav_dynamics"] = UAVDynamics(values["uav_dynamics"])
    values["sensitivity_mode"] = SensitivityMode(values["sensitivity_mode"])
    values["collision_controller"] = CollisionController(values["collision_controller"])
    for key in ("formation_center_start", "formation_center_goal"):
        values[key] = tuple(values[key])
    config = ExperimentConfig(**values)
    config.validate()
    return config


