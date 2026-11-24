"""Formation control with obstacle avoidance for UAV and UGV subteams."""

from __future__ import annotations

import hashlib
import math
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from experiments.communication import GraphSnapshot
from experiments.formation.consensus import (
    DynamicAverageConsensus,
    alternating_ring_neighbors,
    nominal_formation_control,
)
from experiments.formation.disturbances import (
    _keyed_rng,
    collision_mission_seed,
    evaluate_canonical_class_field,
    generate_collision_mission_realization,
    obstacle_encounter_acceleration,
)
from experiments.formation.dynamics import (
    WangVehicleState,
    _as_vector,
    propagate_true_wang_dynamics,
    wang_acceleration_model_error,
    wang_allocation_weights,
    wang_b,
    wang_h,
    wang_barrier,
    wang_barrier_rhs,
    wang_obstacle_residuals,
    wang_pair_residuals,
)
from experiments.formation.geometry import (
    build_formation_plan,
    obstacles,
    offsets,
    quadratic_segment_circle_clearance,
    quadratic_segment_circle_first_violation,
    reference,
    sample_formation_phase,
)
from experiments.formation.qp import WangControlStep as WangControllerStep
from experiments.formation.qp import (
    WangQPDiagnostics,
    WangWall,
    effective_pair_decay,
    obstacle_robust_tightening,
    pair_robust_tightening,
    wall_B_hat,
    wall_B_true,
    wall_clearance,
)
from experiments.formation.sensitivity import unavailable_analytic_sensitivity
from experiments.formation.workers import CircularObstacle, CollisionAvoidanceExperiment
from experiments.mission import (
    CanonicalMissionFrame,
    build_distance_graph,
    build_mission_plans,
    canonical_mission_frame,
    robot_platforms,
)
from maicp.configuration import CollisionController, ExperimentConfig, Platform
from maicp.score_stores import LocalCalibrationStores
from maicp.trace import WaveTraceRecorder
from maicp.types import MissionPlan, MissionResult

Array = np.ndarray


def _connect_terminal_update_graph(
    graph: GraphSnapshot[str], positions: Mapping[str, Array],
) -> tuple[GraphSnapshot[str], list[dict[str, object]]]:
    """Bridge terminal components by their nearest pair of physical robots.

    These are logical CP links, allowed beyond the physical radio range.
    Preserve all range edges and add exactly one edge per component merge.
    """
    if graph.is_connected:
        return graph, []
    adjacency = {robot: set(graph[robot]) for robot in graph}
    parents = {robot: robot for robot in graph}

    def root(robot: str) -> str:
        while parents[robot] != robot:
            parents[robot] = parents[parents[robot]]
            robot = parents[robot]
        return robot

    def merge(left: str, right: str) -> bool:
        a, b = root(left), root(right)
        if a == b:
            return False
        parents[b] = a
        return True

    for left, right in sorted(graph.edges):
        merge(left, right)
    nodes = sorted(graph)
    candidates = sorted(
        (float(np.linalg.norm(positions[left] - positions[right])), left, right)
        for index, left in enumerate(nodes) for right in nodes[index + 1:]
    )
    added = []
    for distance, left, right in candidates:
        if merge(left, right):
            adjacency[left].add(right)
            adjacency[right].add(left)
            added.append({"robots": [left, right], "distance_m": distance})
    return GraphSnapshot(adjacency, require_connected=True), added

FORMATION_SCENARIO = "formation_obstacles_v1"


def sample_dynamics_severity_multiplier(
    plan: MissionPlan,
    sigma: float,
) -> float:
    """Sample one multiplier for the mission from its random stream.

    The stream is keyed only by :func:`collision_mission_seed` and a dedicated
    source label, so paired margins, reflected subteam frames, and center
    feedback variants cannot shift this draw or any other mission randomness.
    """

    sigma = float(sigma)
    if not np.isfinite(sigma) or not 0.0 <= sigma <= 1.0:
        raise ValueError("mission dynamics severity sigma must lie in [0, 1]")
    mission_seed = collision_mission_seed(plan)
    rng = _keyed_rng(mission_seed, "mission-dynamics-severity")
    return float(rng.uniform(1.0 - sigma, 1.0 + sigma))


def _audit_float(value: Any) -> float | None:
    """Convert a diagnostic value to a finite JSON number or None."""

    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _obstacle_applies(obstacle: CircularObstacle, platform: Platform) -> bool:
    """Return whether one shared catalog entry constrains ``platform``.

    A pillar with no assigned platform extends through both operating layers.
    """

    owner = getattr(obstacle, "platform", None)
    return owner is None or owner is platform


def _formation_reference(
    config: ExperimentConfig,
    time_s: float,
    frame: CanonicalMissionFrame | None = None,
    *,
    return_start_s: float | None = None,
    return_origin: Array | None = None,
) -> tuple[Array, Array, Array]:
    """Get reflected ``(c_ref, v_ref, a_ref)`` from formation geometry."""

    # ``reference`` owns the canonical route and applies a point
    # transform to the center and vector transforms to both derivatives.
    kwargs = {} if return_start_s is None else {
        "return_start_s": return_start_s, "return_origin": return_origin
    }
    value = reference(float(time_s), frame=frame, config=config, **kwargs)
    if not isinstance(value, Sequence) or len(value) != 3:
        raise TypeError("reference must return (center, velocity, acceleration)")
    return tuple(
        _as_vector(component, name=f"formation reference[{index}]")
        for index, component in enumerate(value)
    )  # type: ignore[return-value]


def _formation_offset_map(
    plan: MissionPlan,
    config: ExperimentConfig,
    formation_phase_rad: float | None = None,
) -> dict[str, Array]:
    """Return the offset assigned to each robot for this mission."""
    value = offsets(plan, config, formation_phase_rad=formation_phase_rad)
    if not isinstance(value, Mapping):
        raise TypeError("offsets must return a mapping")
    return {
        str(robot_id): _as_vector(offset, name=f"formation offset[{robot_id}]")
        for robot_id, offset in value.items()
    }


def _formation_obstacles(
    plan: MissionPlan,
    config: ExperimentConfig,
    formation_phase_rad: float | None = None,
) -> list[CircularObstacle]:
    """Resolve one shared, reflected obstacle catalog for a subteam."""
    value = obstacles(
        plan,
        config,
        canonical_mission_frame(plan.subteam_id),
        formation_phase_rad=formation_phase_rad,
    )
    obstacle_catalog = list(value)
    if not obstacle_catalog:
        raise ValueError("formation_obstacles_v1 must provide shared pillars")
    return obstacle_catalog


def _parameter(
    config: ExperimentConfig,
    stem: str,
    platform: Platform,
    default: float,
) -> float:
    """Read a Wang parameter across current and anticipated config layouts."""

    suffixes = (platform.value, platform.name.lower())
    names = tuple(f"wang_{stem}_{suffix}" for suffix in suffixes)
    for name in names:
        if hasattr(config, name):
            return float(getattr(config, name))
    mapping = getattr(config, f"wang_{stem}", None)
    if isinstance(mapping, Mapping):
        for key in (platform, platform.value, platform.name.lower()):
            if key in mapping:
                return float(mapping[key])
    wang = getattr(config, "wang", None)
    if wang is not None:
        for name in names:
            if hasattr(wang, name):
                return float(getattr(wang, name))
        mapping = getattr(wang, stem, None)
        if isinstance(mapping, Mapping):
            for key in (platform, platform.value, platform.name.lower()):
                if key in mapping:
                    return float(mapping[key])
    return float(default)


def _gamma(config: ExperimentConfig) -> float:
    return float(config.wang_cbf_gamma_nominal)


def _coerce_margin_mapping(
    margins: Mapping[Platform | str, float],
) -> tuple[dict[Platform, float] | None, dict[str, float] | None]:
    """Read class margins or individual robot margins and reject a mixture of key types."""

    if not isinstance(margins, Mapping) or not margins:
        raise ValueError("Wang margins must be a nonempty mapping")
    class_margins: dict[Platform, float] = {}
    robot_margins: dict[str, float] = {}
    for raw_key, raw_value in margins.items():
        platform_key: Platform | None = None
        if isinstance(raw_key, Platform):
            platform_key = raw_key
        elif isinstance(raw_key, str):
            normalized = raw_key.strip().lower()
            for platform in Platform:
                if normalized in {platform.value.lower(), platform.name.lower()}:
                    platform_key = platform
                    break
        if platform_key is not None:
            class_margins[platform_key] = float(raw_value)
        elif isinstance(raw_key, str):
            robot_margins[raw_key] = float(raw_value)
        else:
            raise TypeError(
                "Wang margin keys must be Platform values, class names, or robot IDs"
            )
    if class_margins and robot_margins:
        raise ValueError("Wang margins cannot mix class and robot keys")
    if class_margins:
        if set(class_margins) != set(Platform):
            missing = sorted(platform.value for platform in set(Platform) - set(class_margins))
            raise ValueError(f"Wang class margins are missing: {missing}")
        return class_margins, None
    return None, robot_margins


def _resolve_robot_margins(
    margins: Mapping[Platform | str, float],
    platforms: Mapping[str, Platform],
    config: ExperimentConfig,
) -> tuple[dict[str, float], dict[Platform, float] | None, str]:
    """Resolve one validated robustness margin for every robot in a wave."""

    class_margins, input_robot_margins = _coerce_margin_mapping(margins)
    if class_margins is not None:
        robot_margins = {
            robot_id: float(class_margins[platform])
            for robot_id, platform in platforms.items()
        }
        margin_mode = "class"
    else:
        assert input_robot_margins is not None
        expected = set(platforms)
        supplied = set(input_robot_margins)
        if supplied != expected:
            missing = sorted(expected - supplied)
            unknown = sorted(supplied - expected)
            details = []
            if missing:
                details.append(f"missing robot margins: {missing}")
            if unknown:
                details.append(f"unknown robot margins: {unknown}")
            raise ValueError("; ".join(details))
        robot_margins = {
            robot_id: float(input_robot_margins[robot_id]) for robot_id in platforms
        }
        margin_mode = "robot"

    for robot_id, value in robot_margins.items():
        platform = platforms[robot_id]
        cls = config.classes[platform]
        if not np.isfinite(value) or not cls.margin_min <= value <= cls.margin_max:
            raise ValueError(
                f"{robot_id} ({platform.value}) margin lies outside its Wang operational box"
            )
    return robot_margins, class_margins, margin_mode


class WangCollisionExperiment(CollisionAvoidanceExperiment):
    """Wang collision experiment with a fixed graph and planar acceleration dynamics."""

    controller_paper_fidelity = False
    supports_robot_margins = True

    def __init__(self, config: ExperimentConfig, *, position_models=None, neighbor_models=None):
        if str(config.collision_scenario) != FORMATION_SCENARIO:
            raise ValueError(
                "WangCollisionExperiment requires the formation_obstacles_v1 scenario"
            )
        classes = {
            platform: replace(
                class_config,
                margin_max=min(class_config.margin_max, config.collision_cbf_margin_max),
            )
            for platform, class_config in config.classes.items()
        }
        bounded_config = replace(config, classes=classes)
        bounded_config.validate()
        super().__init__(bounded_config)
        self.use_hocbf = config.collision_controller == CollisionController.HOCBF
        self.hocbf_lambda = float(config.hocbf_lambda)
        self.agent_radius = float(
            getattr(config, "collision_formation_agent_radius", self.agent_radius)
        )
        if position_models is None and neighbor_models is None:
            from pathlib import Path

            from experiments.formation.models import load_models
            artifact = Path(__file__).with_name("models.json")
            position_models, neighbor_models = load_models(artifact)
        if position_models is None or neighbor_models is None:
            raise ValueError("Provide both position and neighbor models")
        self.models = position_models
        self.neighbor_models = dict(neighbor_models)
        self.gamma = _gamma(config)
        self.optimal_decay = self.use_hocbf or bool(getattr(config, "wang_optimal_decay_cbf", True))
        self.decay_weight = float(getattr(config, "wang_cbf_decay_weight", 10.0))
        self.decay_multiplier_min = float(
            getattr(config, "wang_cbf_decay_multiplier_min", 0.0)
        )
        self.decay_multiplier_max = float(
            getattr(config, "wang_cbf_decay_multiplier_max", 10.0)
        )
        if self.use_hocbf:
            self.decay_weight = 10.0
            self.decay_multiplier_min = 0.0
            self.decay_multiplier_max = 10.0
        self.acceleration_limits = {
            Platform.UGV: _parameter(config, "max_accel", Platform.UGV, 2.0),
            Platform.UAV: _parameter(config, "max_accel", Platform.UAV, 4.0),
        }
        self.speed_limits = {
            Platform.UGV: _parameter(config, "max_speed", Platform.UGV, 8.0),
            Platform.UAV: _parameter(config, "max_speed", Platform.UAV, 12.0),
        }
        self.position_gains = {
            Platform.UGV: _parameter(config, "position_kp", Platform.UGV, 1.4),
            Platform.UAV: _parameter(config, "position_kp", Platform.UAV, 1.8),
        }
        self.velocity_gains = {
            Platform.UGV: _parameter(config, "velocity_kd", Platform.UGV, 2.0),
            Platform.UAV: _parameter(config, "velocity_kd", Platform.UAV, 2.2),
        }
        self.formation_position_gains = {
            platform: _parameter(config, "formation_kp", platform, self.position_gains[platform])
            for platform in Platform
        }
        goal_fraction = float(getattr(config, "formation_goal_gain_fraction", 0.5))
        self.formation_goal_gains = {
            platform: _parameter(
                config,
                "formation_goal_kp",
                platform,
                goal_fraction * self.formation_position_gains[platform],
            )
            for platform in Platform
        }
        configured_variant = str(config.center_variant)
        self.center_feedback_variant = (
            "production" if configured_variant == "distributed" else configured_variant
        )
        if self.center_feedback_variant not in {
            "production",
            "exact",
            "prescribed",
        }:
            raise ValueError(
                "wang_center_feedback_variant must be production, exact, or prescribed"
            )
        self.consensus_rounds = int(config.center_consensus_rounds)
        if self.consensus_rounds < 1:
            raise ValueError("Wang center consensus rounds must be positive")
        for platform in Platform:
            values = (
                self.acceleration_limits[platform],
                self.speed_limits[platform],
                self.position_gains[platform],
                self.velocity_gains[platform],
            )
            if not all(np.isfinite(value) and value > 0.0 for value in values):
                raise ValueError(f"Invalid Wang design parameters for {platform.value}")
        if not np.isfinite(self.gamma) or self.gamma < 0.0:
            raise ValueError("Wang CBF gamma must be finite and nonnegative")
        if not np.isfinite(self.decay_weight) or self.decay_weight <= 0.0:
            raise ValueError("Wang decay weight must be finite and positive")
        if (
            not np.isfinite(self.decay_multiplier_min)
            or self.decay_multiplier_min < 0.0
            or not np.isfinite(self.decay_multiplier_max)
            or self.decay_multiplier_max < self.decay_multiplier_min
        ):
            raise ValueError("Wang decay multiplier bounds must be finite, nonnegative, and ordered")

    @property
    def controller_name(self) -> str:
        return "hocbf" if self.use_hocbf else "wang"

    @property
    def sensitivity_horizon(self) -> float:
        return self.dt * self.config.profile.simulation_steps

    @staticmethod
    def _state_snapshot(states: Mapping[str, WangVehicleState]) -> dict[str, dict[str, Array | float | str]]:
        return {
            robot_id: {
                "platform": state.platform.value,
                "position": state.position.copy(),
                "velocity": state.velocity.copy(),
            }
            for robot_id, state in states.items()
        }

    def _learned_callback(
        self,
        frame: CanonicalMissionFrame,
        platform: Platform,
        position: Array,
    ) -> Array:
        return evaluate_canonical_class_field(
            self.models,
            platform,
            position,
            frame,
            learned=True,
        )

    def _make_controller(
        self,
        graph: GraphSnapshot[str],
        obstacles: Mapping[str, Sequence[CircularObstacle]],
        walls: Mapping[Platform, Sequence[WangWall]],
        pair_edges: Sequence[tuple[str, str]],
        platforms: Mapping[str, Platform],
        margins: Mapping[str, float],
        frame: CanonicalMissionFrame,
    ) -> Any:
        del graph
        from experiments.formation.qp import WangController, WangObstacle

        selected_obstacles = {
            robot_id: tuple(
                WangObstacle(
                    center=np.asarray(obstacle.center, dtype=float),
                    radius=float(obstacle.radius),
                    # Both layers use the same obstacle identifiers.
                    obstacle_id=str(
                        getattr(obstacle, "obstacle_id", None) or f"pillar:{index}"
                    ),
                )
                for index, obstacle in enumerate(obstacle_values)
                if _obstacle_applies(obstacle, platforms[robot_id])
            )
            for robot_id, obstacle_values in obstacles.items()
        }
        selected_walls = {
            robot_id: tuple(
                WangWall(wall.normal, wall.offset, f"{robot_id}:{wall.wall_id}")
                for wall in walls[platforms[robot_id]]
            )
            for robot_id in selected_obstacles
        }
        if self.use_hocbf:
            from experiments.formation.control import RobotLocalHOCBFController
            return RobotLocalHOCBFController(
                obstacles=selected_obstacles,
                acceleration_limits=self.acceleration_limits,
                # Each local solve obtains only its own margin from its
                # WangState. Do not expose the peer CP margin mapping.
                margins=None,
                lambda1=self.hocbf_lambda,
                lambda2=self.hocbf_lambda,
                decay_weight=self.decay_weight,
                omega_min=0.0,
                omega_max=10.0,
            )
        return WangController(
            pair_edges=pair_edges,
            robot_platforms={
                robot_id: platforms[robot_id] for robot_id in selected_obstacles
            },
            acceleration_limits=self.acceleration_limits,
            gammas=self.gamma,
            margins=margins,
            safety_radii=self.agent_radius,
            learned_model=lambda state: self._learned_callback(
                frame, state.platform, state.position
            ),
            obstacles=selected_obstacles,
            walls=selected_walls,
            optimal_decay=self.optimal_decay,
            decay_weight=self.decay_weight,
            decay_multiplier_min=self.decay_multiplier_min,
            decay_multiplier_max=self.decay_multiplier_max,
        )

    def _controller_states(
        self,
        controller: Any,
        robot_ids: Sequence[str],
        states: Mapping[str, WangVehicleState],
        margins: Mapping[str, float],
    ) -> Mapping[str, Any]:
        """Adapt mutable plant states to the dedicated controller packet type."""

        del controller
        from experiments.formation.qp import WangState
        return {
            robot_id: WangState(
                robot_id=robot_id,
                platform=states[robot_id].platform,
                position=states[robot_id].position,
                velocity=states[robot_id].velocity,
                acceleration_limit=self.acceleration_limits[states[robot_id].platform],
                radius=self.agent_radius,
                margin=margins[robot_id],
                max_speed=self.speed_limits[states[robot_id].platform],
            )
            for robot_id in robot_ids
        }

    def _obstacles(
        self,
        plan,
        formation_phase_rad: float | None = None,
    ) -> list[CircularObstacle]:
        return _formation_obstacles(plan, self.config, formation_phase_rad)

    def _neighbor_fields(self, plans, positions, *, learned):
        """Pool relative position predictions from the state before the update.

        Both classes contribute to the learned interaction model for the receiving class. This
        synthetic coupling uses planar separation, independently of the
        physical collision layers. Subteams remain independent mission draws.
        The fixed peer count preserves density effects outside the radius.
        """
        fields = {}
        for plan in plans:
            frame = canonical_mission_frame(plan.subteam_id)
            peer_ids = tuple(sorted(plan.home_positions))
            peer_points = np.asarray([frame.point_to_local(positions[rid]) for rid in peer_ids])
            peer_index = {rid: index for index, rid in enumerate(peer_ids)}
            for platform in Platform:
                ids = tuple(sorted(plan.robot_ids[platform]))
                own_indices = [peer_index[rid] for rid in ids]
                relative = peer_points[None, :, :] - peer_points[own_indices, None, :]
                model = self.neighbor_models[platform]
                if learned:
                    pair_values = model.predict(relative, radius=self.config.collision_neighbor_radius)
                else:
                    pair_values = model.true_field(
                        relative, radius=self.config.collision_neighbor_radius,
                        gain=self.config.collision_neighbor_gain,
                    )
                pair_values[np.arange(len(ids)), own_indices] = 0.0
                pooled = np.sum(pair_values, axis=1) / max(1, len(peer_ids) - 1)
                fields.update({rid: frame.vector_to_global(value) for rid, value in zip(ids, pooled, strict=True)})
        return fields

    @property
    def experiment_metadata(self) -> dict[str, object]:
        return {
            "controller": self.controller_name,
            "controller_source_equations": (
                "robot-local robust relative-degree-two HOCBF with bounded optimal decay in the second condition"
                if self.use_hocbf else
                "Wang et al. (ACC 2016), Eq. (8) geometry and Strategy B, "
                "with linear class-K optimal decay"
            ),
            "controller_strategy": "local_obstacles_only" if self.use_hocbf else "B",
            "hocbf_lambda1": self.hocbf_lambda if self.use_hocbf else None,
            "hocbf_lambda2": self.hocbf_lambda if self.use_hocbf else None,
            "hocbf_omega_bounds": [0.0, 10.0] if self.use_hocbf else None,
            "hocbf_decay_penalty_weight": self.decay_weight if self.use_hocbf else None,
            "collision_scenario": FORMATION_SCENARIO,
            "agent_radius_m": self.agent_radius,
            "formation_reference": {
                "center_start": list(np.asarray(self.config.formation_center_start, dtype=float)),
                "center_goal": list(np.asarray(self.config.formation_center_goal, dtype=float)),
                "duration_s": float(self.config.formation_center_duration_s),
                "profile": "quintic_zero_velocity_zero_acceleration",
                "terminal_behavior": "hold_destination",
                "return_required": False,
                "return_leg_enabled": False,
                "waypoint_arrival_rule": "terminal slots within position/speed tolerances; diagnostic only",
                "arrival_position_tolerance_m": self.config.home_tolerance,
                "arrival_velocity_tolerance_m_s": 0.5,
                "mission_horizon_s": self.config.profile.simulation_steps * self.dt,
                "termination_policy": "fixed_time_limit",
            },
            "center_variant": self.config.center_variant,
            "center_feedback_variant": self.center_feedback_variant,
            "center_consensus_rounds": self.consensus_rounds,
            "center_consensus_graph": "alternating_ugv_uav_ring",
            "center_consensus_reliability_assumption": (
                "fixed connected synchronous logical ring selected at mission "
                "initialization; links are assumed reliable for the scored interval"
            ),
            "neighbor_interaction": {
                "radius_m": self.config.collision_neighbor_radius,
                "training_gain": 1.0,
                "deployment_gain": self.config.collision_neighbor_gain,
                "pooling": "sum over all other UAV and UGV subteam robots / fixed total peer count",
                "cross_class_interaction": True,
                "geometry": "synthetic planar coupling; altitude separates collision bodies only",
                "distance_metric": "horizontal Euclidean distance only",
                "vertical_class_separation_m": 12.0,
                "source_class_dependence": "same pair law for either source class; receiver class selects the model",
                "snapshot": "synchronous pre-control positions in canonical mission coordinates",
                "distribution_shift": "offline training gain 1; pilot, calibration and evaluation share deployment gain",
                "models": {p.value: self.neighbor_models[p].report for p in Platform},
                "kappa": "configured assumption, not certified for the joint-state feedback system",
            },
            "collision_error_field_frequency_scale": self.config.collision_error_field_frequency_scale,
            "shared_obstacle_catalog": True,
            "formation_obstacle_layout": self.config.formation_obstacle_layout,
            "formation_gate_parameters": {
                "radius": self.config.formation_gate_radius,
                "penetration": self.config.formation_gate_penetration,
                "along_route_jitter_m": self.config.formation_gate_jitter_m,
            } if self.config.formation_obstacle_layout == "slot_gates" else None,
            "obstacle_position_sigma_m": self.config.obstacle_position_sigma_m,
            "mission_dynamics_severity_sigma": self.config.mission_dynamics_severity_sigma,
            "obstacle_encounter_disturbance": {
                "law": (
                    "bounded_inward_opposite_class_"
                    f"{self.config.formation_obstacle_disturbance_mode}_v1"
                ),
                "mode": self.config.formation_obstacle_disturbance_mode,
                "maximum_acceleration_m_s2": self.config.formation_obstacle_disturbance_max,
                "band_m": self.config.formation_obstacle_disturbance_band_m,
                "strength_distribution": "mission-keyed uniform[0.75,1.0]",
                "controller_knows_field": False,
            },
            "mission_dynamics_severity": (
                "per-subteam a_m ~ Uniform(1-sigma, 1+sigma), applied only to "
                "the learned-to-true position and neighbor model-error components"
            ),
            "obstacle_layer_semantics": (
                "platform=None pillars apply to UGV and UAV footprints; UGV-UAV "
                "overlap is not a physical contact because of vertical separation"
            ),
            "continuous_time_safety_theorem_claimed": False,
            "plant": ("p_dot=v; v_dot=u+d_class(p)+d_neighbors(p_i,{p_j})+d_encounter+w"
                      if self.config.formation_obstacle_disturbance_max > 0 else
                      "p_dot=v; v_dot=u+d_class(p)+d_neighbors(p_i,{p_j})+w"),
            "plant_state": "planar_second_order_for_both_UGV_and_UAV",
            "model_units": "acceleration",
            "robustness_margin_units": "acceleration",
            "robot_nonconformity_score": ("max_t ||d_class(p_i)+d_neighbors(p_i,{p_j})+d_encounter+w_i-d_hat_class(p_i)-d_hat_neighbors(p_i,{p_j})||_2"
                                          if self.config.formation_obstacle_disturbance_max > 0 else
                                          "max_t ||d_class(p_i)+d_neighbors(p_i,{p_j})+w_i-d_hat_class(p_i)-d_hat_neighbors(p_i,{p_j})||_2"),
            "pair_robust_tightening": None if self.use_hocbf else "||p_i-p_j||_2 * (r_C(i)+r_C(j))",
            "obstacle_robust_tightening": "2 ||p_i-p_o||_2 * r_C(i)" if self.use_hocbf else "||p_i-p_o||_2 * r_C(i)",
            "wall_robust_tightening": None if self.use_hocbf else "r_C(i) for unit wall normals",
            "wang_gamma": self.gamma,
            "wang_optimal_decay_cbf": self.optimal_decay,
            "wang_decay_weight": self.decay_weight,
            "wang_decay_multiplier_bounds": [
                self.decay_multiplier_min,
                self.decay_multiplier_max,
            ],
            "wang_acceleration_limits": {
                platform.value: self.acceleration_limits[platform] for platform in Platform
            },
            "wang_speed_limits": {platform.value: self.speed_limits[platform] for platform in Platform},
            "wang_position_gains": {platform.value: self.position_gains[platform] for platform in Platform},
            "wang_velocity_gains": {platform.value: self.velocity_gains[platform] for platform in Platform},
            "collision_initial_position_radius": self.config.collision_initial_position_radius,
            "collision_communication_range": self.config.collision_communication_range,
            "update_communication_range": self.config.update_communication_range,
            "terminal_update_graph_policy": "home_reset_physical_range",
            "terminal_update_long_range_links_assumed_available": True,
            "collision_motion_noise": {
                Platform.UGV.value: {
                    "sigma": self.config.collision_motion_noise_sigma_ugv,
                    "maximum": self.config.collision_motion_noise_max_ugv,
                },
                Platform.UAV.value: {
                    "sigma": self.config.collision_motion_noise_sigma_uav,
                    "maximum": self.config.collision_motion_noise_max_uav,
                },
            },
            "class_models": {
                platform.value: {
                    "train_mse": self.models.report(platform).train_mse,
                    "validation_mse": self.models.report(platform).validation_mse,
                    "nn_spectral_lipschitz_bound": self.models.report(platform).prediction_lipschitz,
                    "frequency_scale": self.models.report(platform).frequency_scale,
                    "true_disturbance_magnitude_bound": self.models.report(platform).true_field_bounds.M,
                    "true_disturbance_lipschitz_bound": self.models.report(platform).true_field_bounds.L,
                    "error_field_magnitude_bound": self.models.report(platform).error_bounds.M,
                    "error_field_lipschitz_bound": self.models.report(platform).error_bounds.L,
                    "architecture": [2, *self.models.report(platform).hidden_layers, 2],
                    "training_seed": self.models.report(platform).train_seed,
                    "validation_seed": self.models.report(platform).validation_seed,
                    "frozen": True,
                    "bounds_scope": "position-only component; excludes neighbor interaction",
                    "units": "acceleration",
                }
                for platform in Platform
            },
        }

    def analytic_kappa(self, context: object):
        """Report that the acceleration score has no analytic sensitivity bound here."""

        return unavailable_analytic_sensitivity(
            context,
            "analytic sensitivity for Wang acceleration-error score not implemented",
        )

    def run_wave(
        self,
        margins: Mapping[Platform | str, float],
        first_mission_id: int,
        wave_seed: int,
        trace_recorder: WaveTraceRecorder | None = None,
        local_calibration: LocalCalibrationStores | None = None,
        *,
        subteam_id: int | None = None,
    ) -> tuple[list[MissionResult], GraphSnapshot[str], dict[str, Platform]]:
        """Simulate all four subteams or replay a single subteam with its original seed."""
        record_full_qp_diagnostics = bool(
            getattr(self.config, "record_full_qp_diagnostics", False)
            or trace_recorder is not None
        )

        base_plans = build_mission_plans(self.config, first_mission_id, wave_seed)
        if subteam_id is not None:
            base_plans = tuple(plan for plan in base_plans if plan.subteam_id == subteam_id)
            if not base_plans:
                raise ValueError(f"Unknown formation subteam: {subteam_id}")
        formation_phase_rad = {
            plan.subteam_id: sample_formation_phase(plan)
            for plan in base_plans
        }
        plans = tuple(
            build_formation_plan(
                plan,
                self.config,
                agent_radius=self.agent_radius,
                formation_phase_rad=formation_phase_rad[plan.subteam_id],
            )
            for plan in base_plans
        )
        # Sample once at mission setup. Both classes share these static objects
        # for all controller solves, collision audits, and trace frames.
        mission_obstacles = {
            plan.subteam_id: self._obstacles(
                plan,
                formation_phase_rad=formation_phase_rad[plan.subteam_id],
            )
            for plan in plans
        }
        mission_walls = {
            plan.subteam_id: {platform: () for platform in Platform}
            for plan in plans
        }
        realizations = {
            plan.subteam_id: generate_collision_mission_realization(
                self.config,
                plan,
                formation_phase_rad=formation_phase_rad[plan.subteam_id],
            )
            for plan in plans
        }
        dynamics_severity_multiplier = {
            plan.subteam_id: sample_dynamics_severity_multiplier(
                plan,
                self.config.mission_dynamics_severity_sigma,
            )
            for plan in plans
        }
        encounter_amplitude = self.config.formation_obstacle_disturbance_max
        encounter_strength = {
            plan.subteam_id: float(_keyed_rng(collision_mission_seed(plan),
                "obstacle-encounter-strength-v1").uniform(0.75, 1.0))
            for plan in plans
        } if encounter_amplitude > 0 else {}
        platforms = robot_platforms(plans)
        robot_margins, class_margins, margin_mode = _resolve_robot_margins(
            margins, platforms, self.config
        )
        offsets_by_subteam = {
            plan.subteam_id: _formation_offset_map(
                plan,
                self.config,
                formation_phase_rad=formation_phase_rad[plan.subteam_id],
            )
            for plan in plans
        }
        for plan in plans:
            offsets = offsets_by_subteam[plan.subteam_id]
            ordered_ids = tuple(plan.home_positions)
            if set(offsets) != set(ordered_ids):
                raise ValueError("Formation geometry must provide one offset per robot")
            offset_sum = np.sum(
                np.asarray([offsets[robot_id] for robot_id in ordered_ids], dtype=float),
                axis=0,
            )
            if not np.allclose(offset_sum, np.zeros(2), rtol=0.0, atol=1e-9):
                raise ValueError("Formation offsets must sum to zero within each subteam")
        states: dict[str, WangVehicleState] = {}
        for plan in plans:
            realization = realizations[plan.subteam_id]
            frame = realization.frame
            for robot_id in plan.home_positions:
                states[robot_id] = WangVehicleState.from_plan(
                    robot_id,
                    platforms[robot_id],
                    plan.home_positions[robot_id],
                    plan.waypoints[robot_id],
                )
                offset = offsets_by_subteam[plan.subteam_id].get(robot_id)
                if offset is None:
                    center_start, _, _ = _formation_reference(self.config, 0.0, frame)
                    offset = states[robot_id].home - center_start
                else:
                    # Reflect formation offsets into the subteam's quadrant before computing center observations.
                    offset = frame.vector_to_global(offset)
                states[robot_id].formation_offset = _as_vector(
                    offset, name=f"formation offset[{robot_id}]"
                )
                states[robot_id].position = (
                    states[robot_id].home + realization.initial_perturbations[robot_id]
                )
                # Wang mode explicitly starts every class at zero velocity.
                states[robot_id].velocity = np.zeros(2)

        # The shared realization object hashes phase, offsets, jitter and the
        # complete process noise arrays. Formation also samples pillars and
        # severity, so include their realized values and the initial states in
        # the pairing certificate before any control step that uses the margin.
        exogenous_digests = {}
        for plan in plans:
            value = hashlib.blake2b(digest_size=16)
            value.update(realizations[plan.subteam_id].digest.encode("ascii"))
            value.update(np.asarray([dynamics_severity_multiplier[plan.subteam_id]], dtype=np.float64).tobytes())
            if encounter_amplitude > 0:
                value.update(np.asarray([encounter_strength[plan.subteam_id]], dtype=np.float64).tobytes())
            for obstacle in mission_obstacles[plan.subteam_id]:
                value.update(str(obstacle.obstacle_id).encode("utf-8"))
                value.update(np.asarray([*obstacle.center, obstacle.radius], dtype=np.float64).tobytes())
            for robot_id in sorted(plan.home_positions):
                value.update(robot_id.encode("utf-8"))
                value.update(np.asarray([*states[robot_id].position, *states[robot_id].velocity], dtype=np.float64).tobytes())
            exogenous_digests[plan.subteam_id] = value.hexdigest()

        def ring_order(plan: MissionPlan) -> tuple[str, ...]:
            """Return the shared alternating UGV/UAV protocol order."""

            order = tuple(
                robot_id
                for index in range(max(len(plan.robot_ids[platform]) for platform in Platform))
                for platform in (Platform.UGV, Platform.UAV)
                for robot_id in plan.robot_ids[platform][index:index + 1]
            )
            if len(order) != 6 or len(set(order)) != len(order):
                raise ValueError("formation_obstacles_v1 requires six unique robots per subteam")
            return order

        ring_orders = {plan.subteam_id: ring_order(plan) for plan in plans}
        fixed_graphs = {
            plan.subteam_id: GraphSnapshot.from_adjacency(
                alternating_ring_neighbors(ring_orders[plan.subteam_id]),
                require_connected=True,
            )
            for plan in plans
        }
        # Record contact between robots in the same layer. The QP enforces obstacle clearance.
        pair_edges = {plan.subteam_id: tuple() for plan in plans}

        controllers: dict[int, Any] = {}
        for plan in plans:
            graph = fixed_graphs[plan.subteam_id]
            frame = realizations[plan.subteam_id].frame
            obstacles_by_robot = {
                robot_id: tuple(obstacle for obstacle in mission_obstacles[plan.subteam_id])
                for robot_id in plan.home_positions
            }
            controllers[plan.subteam_id] = self._make_controller(
                graph,
                obstacles_by_robot,
                mission_walls[plan.subteam_id],
                pair_edges[plan.subteam_id],
                platforms,
                robot_margins,
                frame,
            )

        robot_ids = tuple(sorted(states))
        plan_by_robot = {
            robot_id: plan
            for plan in plans
            for robot_id in plan.home_positions
        }
        # Each subteam has a separate ring with alternating UAV and UGV nodes.
        consensus_state: dict[int, dict[str, Array]] = {}
        consensus_workers: dict[int, DynamicAverageConsensus] = {}
        for plan in plans:
            ordered = ring_orders[plan.subteam_id]
            observations = {
                robot_id: states[robot_id].position
                - np.asarray(states[robot_id].formation_offset, dtype=float)
                for robot_id in ordered
            }
            worker = DynamicAverageConsensus(ordered, rounds=self.consensus_rounds)
            consensus_workers[plan.subteam_id] = worker
            consensus_state[plan.subteam_id] = worker.initialize(observations)

        # Dense matched diagnostics are kept in the audit at fixed margins.  The
        # sweep serializer may move these arrays to compressed wave sidecars,
        # but run_wave owns one canonical time alignment for every field.
        trajectory: dict[str, dict[str, list[object]]] = {
            robot_id: {
                name: []
                for name in (
                    "position",
                    "velocity",
                    "consensus",
                    "nominal_control",
                    "applied_control",
                    "formation_term",
                    "goal_term",
                    "center_observation",
                    "true_center",
                    "consensus_error",
                    "formation_error",
                    "center_reference_error",
                    "position_model_error",
                    "neighbor_model_error",
                    "process_noise",
                    "acceleration_error",
                    "true_position_residual",
                    "learned_position_residual",
                    "true_neighbor_residual",
                    "learned_neighbor_residual",
                    "base_true_position_residual",
                    "base_true_neighbor_residual",
                )
            }
            for robot_id in robot_ids
        }
        trajectory_time_s: list[float] = []
        if encounter_amplitude > 0:
            for series in trajectory.values():
                series["obstacle_encounter_error"] = []
        hocbf_fields = (
            "hocbf_h", "hocbf_psi", "hocbf_slack", "hocbf_omega",
            "hocbf_interval_clearance", "hocbf_interval_tau",
            "qp_status_code", "qp_feasible", "fallback_used",
        )
        if self.use_hocbf:
            for series in trajectory.values():
                series.update({name: [] for name in hocbf_fields})
        hocbf_initial_pairs: list[dict[str, object]] = []
        hocbf_min_slack = {plan.subteam_id: float("inf") for plan in plans}
        hocbf_success_count = {plan.subteam_id: 0 for plan in plans}
        score_reconstruction_error = {plan.subteam_id: 0.0 for plan in plans}
        controller_prediction_error = {plan.subteam_id: 0.0 for plan in plans}
        hocbf_status_codes = {"optimal": 1, "infeasible": 0, "exception": -1}
        score_argmax_step = {robot_id: None for robot_id in robot_ids}
        score_argmax_components: dict[str, dict[str, object] | None] = {
            robot_id: None for robot_id in robot_ids
        }
        encounter_diagnostics = {rid: {
            "near_obstacle_constraint_samples": 0,
            "near_uncovered_constraint_samples": 0,
            "maximum_inward_error_near_obstacles": 0.,
            "maximum_added_acceleration": 0.,
            "minimum_true_hocbf_residual_near_obstacles": None,
        } for rid in robot_ids}
        encounter_geometry = {
            rid: (
                np.asarray([o.center for o in mission_obstacles[plan_by_robot[rid].subteam_id]
                            if _obstacle_applies(o, platforms[rid])]).reshape(-1, 2),
                np.asarray([o.radius + self.agent_radius
                            for o in mission_obstacles[plan_by_robot[rid].subteam_id]
                            if _obstacle_applies(o, platforms[rid])]),
                tuple(peer for peer in plan_by_robot[rid].home_positions
                      if platforms[peer] != platforms[rid]),
            ) for rid in robot_ids
        } if encounter_amplitude > 0 else {}
        scores = {
            plan.subteam_id: {
                platform: {robot_id: 0.0 for robot_id in plan.robot_ids[platform]}
                for platform in Platform
            }
            for plan in plans
        }
        global_failures: dict[int, list[str]] = {
            plan.subteam_id: [] for plan in plans
        }
        robot_failure_reasons: dict[int, dict[str, set[str]]] = {
            plan.subteam_id: {
                robot_id: set() for robot_id in plan.home_positions
            }
            for plan in plans
        }
        failed_robot_ids: dict[int, set[str]] = {
            plan.subteam_id: set() for plan in plans
        }

        def mark_robot_failed(
            subteam_id: int, robot_id: str, reason: str
        ) -> None:
            # Record application failures separately from acceleration residuals.
            failed_robot_ids[subteam_id].add(robot_id)
            robot_failure_reasons[subteam_id][robot_id].add(reason)

        minimum_separation = {plan.subteam_id: float("inf") for plan in plans}
        minimum_implemented = {plan.subteam_id: float("inf") for plan in plans}
        minimum_true = {plan.subteam_id: float("inf") for plan in plans}
        maximum_acceleration_error = {plan.subteam_id: 0.0 for plan in plans}
        maximum_pair_tightening = {plan.subteam_id: 0.0 for plan in plans}
        maximum_obstacle_tightening = {plan.subteam_id: 0.0 for plan in plans}
        maximum_wall_tightening = {
            plan.subteam_id: (0.0 if self.use_hocbf else
                             max(float(robot_margins[robot_id])
                                 for robot_id in plan.home_positions))
            for plan in plans
        }
        minimum_wall_clearance = {plan.subteam_id: float("inf") for plan in plans}
        qp_failures = {plan.subteam_id: 0 for plan in plans}
        qp_solves = {plan.subteam_id: 0 for plan in plans}
        minimum_active_sigma = {plan.subteam_id: float("inf") for plan in plans}
        maximum_condition = {plan.subteam_id: 0.0 for plan in plans}
        singular_qps = {plan.subteam_id: 0 for plan in plans}
        ill_conditioned_qps = {plan.subteam_id: 0 for plan in plans}
        maximum_control_norm = {plan.subteam_id: 0.0 for plan in plans}
        maximum_control_inf_norm = {plan.subteam_id: 0.0 for plan in plans}
        maximum_acceleration_slack = {plan.subteam_id: float("inf") for plan in plans}
        minimum_decay_multiplier = {plan.subteam_id: float("inf") for plan in plans}
        maximum_decay_multiplier = {plan.subteam_id: float("-inf") for plan in plans}
        changed_decay_multipliers = {plan.subteam_id: 0 for plan in plans}
        total_decay_multipliers = {plan.subteam_id: 0 for plan in plans}
        total_decay_penalty = {plan.subteam_id: 0.0 for plan in plans}
        collisions = {plan.subteam_id: 0 for plan in plans}
        wall_collisions = {plan.subteam_id: 0 for plan in plans}
        obstacle_collisions = {plan.subteam_id: 0 for plan in plans}
        pair_collisions = {plan.subteam_id: 0 for plan in plans}
        maximum_motion_noise_norm = {plan.subteam_id: 0.0 for plan in plans}
        maximum_speed = {plan.subteam_id: 0.0 for plan in plans}
        speed_violations = {plan.subteam_id: 0 for plan in plans}
        home_snap_occurred = {plan.subteam_id: False for plan in plans}
        qp_diagnostic_records: dict[str, list[dict[str, object]]] = {
            robot_id: [] for robot_id in robot_ids
        }

        # Keep extrema and counts here. Detailed QP histories are optional.
        raw_score_max = {robot_id: None for robot_id in robot_ids}
        raw_score_samples = {robot_id: 0 for robot_id in robot_ids}
        score_attempts = {robot_id: 0 for robot_id in robot_ids}
        score_invalid = {robot_id: False for robot_id in robot_ids}
        robot_qp_failures = {robot_id: 0 for robot_id in robot_ids}
        robot_qp_solves = {robot_id: 0 for robot_id in robot_ids}
        robot_qp_fallbacks = {robot_id: 0 for robot_id in robot_ids}
        robot_qp_feasible_solves = {robot_id: 0 for robot_id in robot_ids}
        robot_min_active_sigma = {robot_id: float("inf") for robot_id in robot_ids}
        robot_max_condition = {robot_id: 0.0 for robot_id in robot_ids}
        robot_singular_qps = {robot_id: 0 for robot_id in robot_ids}
        robot_ill_conditioned_qps = {robot_id: 0 for robot_id in robot_ids}
        robot_min_acceleration_slack = {
            robot_id: float("inf") for robot_id in robot_ids
        }
        robot_max_acceleration_error = {robot_id: None for robot_id in robot_ids}
        robot_min_implemented_residual = {
            robot_id: float("inf") for robot_id in robot_ids
        }
        robot_min_true_residual = {robot_id: float("inf") for robot_id in robot_ids}
        robot_max_pair_tightening = {robot_id: 0.0 for robot_id in robot_ids}
        robot_max_obstacle_tightening = {robot_id: 0.0 for robot_id in robot_ids}
        robot_max_wall_tightening = {
            robot_id: (0.0 if self.use_hocbf else float(robot_margins[robot_id]))
            for robot_id in robot_ids
        }
        robot_min_wall_clearance = {
            robot_id: float("inf") for robot_id in robot_ids
        }
        robot_min_physical_clearance = {
            robot_id: float("inf") for robot_id in robot_ids
        }
        robot_min_separation = {robot_id: float("inf") for robot_id in robot_ids}
        robot_collision_flags = {
            robot_id: {"obstacle": False, "wall": False, "pair": False}
            for robot_id in robot_ids
        }
        robot_collision_events = {
            robot_id: {"obstacle": 0, "wall": 0, "pair": 0}
            for robot_id in robot_ids
        }
        audit_collision_events = {
            plan.subteam_id: {"obstacle": 0, "wall": 0, "pair": 0}
            for plan in plans
        }
        pair_collision_records: dict[int, list[dict[str, object]]] = {
            plan.subteam_id: [] for plan in plans
        }
        obstacle_collision_records: dict[int, list[dict[str, object]]] = {
            plan.subteam_id: [] for plan in plans
        }
        obstacle_step_diagnostics: dict[
            tuple[int, str, str, int], dict[str, object]
        ] = {}
        obstacle_collision_event_index: dict[
            tuple[int, str, str, int], dict[str, object]
        ] = {}
        waypoint_arrival_times = {robot_id: None for robot_id in robot_ids}
        destination_reached = {robot_id: False for robot_id in robot_ids}
        completion_position_tolerance = float(
            getattr(self.config, "formation_completion_position_tolerance", self.config.home_tolerance)
        )
        completion_velocity_tolerance = float(
            getattr(self.config, "formation_completion_velocity_tolerance", 0.5)
        )
        def mission_reference(subteam_id: int, time_s: float) -> tuple[Array, Array, Array]:
            return _formation_reference(
                self.config, time_s, realizations[subteam_id].frame,
            )
        audit_exception_groups: dict[int, list[dict[str, object]]] = {
            plan.subteam_id: [] for plan in plans
        }

        def record_audit_collision(
            subteam_id: int,
            robot_id: str,
            kind: str,
            *,
            mission_event: bool = True,
        ) -> None:
            """Record one physical event and unique robot participation."""

            if kind not in ("obstacle", "wall", "pair"):
                raise ValueError(f"Unknown Wang audit collision kind: {kind}")
            robot_collision_flags[robot_id][kind] = True
            robot_collision_events[robot_id][kind] += 1
            if mission_event:
                audit_collision_events[subteam_id][kind] += 1

        def current_score_status(robot_id: str) -> str:
            if raw_score_samples[robot_id] <= 0:
                return "unavailable"
            if score_invalid[robot_id]:
                return "uncovered"
            raw = raw_score_max[robot_id]
            if raw is not None and raw > float(robot_margins[robot_id]):
                return "uncovered"
            return "covered"

        def record_audit_exception(
            subteam_id: int,
            context: str,
            error: BaseException,
            step: int | None = None,
        ) -> None:
            """Aggregate caught execution exceptions without changing control flow."""

            groups = audit_exception_groups[subteam_id]
            error_type = type(error).__name__
            message = str(error)
            existing = next(
                (
                    item
                    for item in groups
                    if item["context"] == context
                    and item["type"] == error_type
                    and item["message"] == message
                ),
                None,
            )
            if existing is None:
                groups.append(
                    {
                        "context": context,
                        "type": error_type,
                        "message": message,
                        "traceback": traceback.format_exc(),
                        "count": 1,
                        "first_step": step,
                        "last_step": step,
                    }
                )
            else:
                existing["count"] = int(existing["count"]) + 1
                existing["last_step"] = step

        if trace_recorder is not None:
            trace_obstacles = [
                {
                    "subteam_id": plan.subteam_id,
                    "obstacle_id": str(
                        getattr(obstacle, "obstacle_id", None)
                        or f"pillar:{obstacle_index}"
                    ),
                    "center": obstacle.center.tolist(),
                    "radius": obstacle.radius,
                    "platform": (
                        getattr(obstacle.platform, "value", None)
                        if getattr(obstacle, "platform", None) is not None
                        else None
                    ),
                }
                for plan in plans
                for obstacle_index, obstacle in enumerate(
                    mission_obstacles[plan.subteam_id]
                )
            ]
            trace_walls = []
            trace_recorder.begin_wave(
                "collision_avoidance",
                self.config,
                plans,
                fixed_graphs,
                wave_seed,
                self.dt,
                obstacles=trace_obstacles,
            )
            trace_recorder.metadata.update(self.experiment_metadata)
            trace_recorder.metadata["margin_mode"] = margin_mode
            trace_recorder.metadata["robot_margins"] = {
                robot_id: float(value) for robot_id, value in robot_margins.items()
            }
            trace_recorder.metadata["class_margins"] = (
                None
                if class_margins is None
                else {platform.value: float(value) for platform, value in class_margins.items()}
            )
            trace_recorder.metadata["walls"] = trace_walls
            trace_recorder.metadata["controller_paper_fidelity"] = False
            trace_recorder.metadata["task_graph_mode"] = "fixed_at_mission_initialization"
            trace_recorder.metadata["fixed_graph_construction"] = (
                "fixed six-node alternating UGV/UAV logical ring selected at mission "
                "initialization; links are assumed reliable and center consensus uses "
                "self and two-neighbour weight 1/3"
            )
            trace_recorder.metadata["mission_seeds"] = {
                str(plan.subteam_id): realizations[plan.subteam_id].mission_seed
                for plan in plans
            }
            trace_recorder.metadata["formation_phase_rad"] = {
                str(plan.subteam_id): realizations[plan.subteam_id].formation_phase_rad
                for plan in plans
            }
            trace_recorder.metadata["formation_offsets_canonical"] = {
                str(plan.subteam_id): {
                    robot_id: value.tolist()
                    for robot_id, value in realizations[plan.subteam_id]
                    .formation_offsets_canonical.items()
                }
                for plan in plans
            }
            trace_recorder.metadata["mission_dynamics_severity_sigma"] = (
                self.config.mission_dynamics_severity_sigma
            )
            trace_recorder.metadata["dynamics_severity_multiplier"] = {
                str(plan.subteam_id): dynamics_severity_multiplier[plan.subteam_id]
                for plan in plans
            }
            trace_recorder.metadata["exogenous_realization_digests"] = {
                str(plan.subteam_id): exogenous_digests[plan.subteam_id] for plan in plans
            }
            trace_recorder.metadata["fixed_wang_graph_edges"] = {
                str(plan.subteam_id): [list(edge) for edge in sorted(fixed_graphs[plan.subteam_id].edges)]
                for plan in plans
            }
            trace_recorder.metadata["fixed_communication_graph_connected"] = {
                str(plan.subteam_id): bool(fixed_graphs[plan.subteam_id].is_connected)
                for plan in plans
            }
            trace_recorder.metadata["center_consensus_graph"] = "alternating_ugv_uav_ring"
            trace_recorder.metadata["center_consensus_graph_edges"] = dict(
                trace_recorder.metadata["fixed_wang_graph_edges"]
            )
            # Both controllers use the same field names in saved traces.
            trace_recorder.metadata["fixed_cbf_graph_edges"] = dict(
                trace_recorder.metadata["fixed_wang_graph_edges"]
            )
            trace_recorder.metadata["fixed_pair_safety_edges"] = {
                str(plan.subteam_id): [list(edge) for edge in pair_edges[plan.subteam_id]]
                for plan in plans
            }
            trace_recorder.metadata["initial_position_perturbations"] = {
                str(plan.subteam_id): {
                    robot_id: realizations[plan.subteam_id].initial_perturbations[robot_id].tolist()
                    for robot_id in plan.home_positions
                }
                for plan in plans
            }
            trace_recorder.metadata["initial_position_perturbations_canonical"] = {
                str(plan.subteam_id): {
                    robot_id: realizations[plan.subteam_id]
                    .local_initial_perturbations[robot_id].tolist()
                    for robot_id in plan.home_positions
                }
                for plan in plans
            }
            trace_recorder.metadata["canonical_subteam_transforms"] = {
                str(plan.subteam_id): {
                    "frame_id": f"quadrant_reflection_{plan.subteam_id}",
                    "matrix": realizations[plan.subteam_id].frame.matrix.tolist(),
                    "origin": [0.0, 0.0],
                }
                for plan in plans
            }
            trace_recorder.metadata["collision_motion_noise_sigma"] = {
                Platform.UGV.value: self.config.collision_motion_noise_sigma_ugv,
                Platform.UAV.value: self.config.collision_motion_noise_sigma_uav,
            }
            trace_recorder.metadata["collision_motion_noise_max"] = {
                Platform.UGV.value: self.config.collision_motion_noise_max_ugv,
                Platform.UAV.value: self.config.collision_motion_noise_max_uav,
            }
            trace_recorder.metadata["initial_velocity"] = {robot_id: [0.0, 0.0] for robot_id in robot_ids}
            trace_recorder.metadata["agent_radius"] = self.agent_radius
            trace_recorder.metadata["control_step_decay_coefficients"] = [{}]
            trace_recorder.metadata["diagnostic_frame_semantics"] = (
                "positions/velocities are at time[k]; control, fields, noise, QP and "
                "decay coefficients at frame k>0 describe the step from k-1 to k. "
                "Frame 0 has no control step. These fields are diagnostics only."
            )

            def record_frame(
                time: float,
                dynamics_mismatch: Mapping[str, float],
                collision_events: Mapping[str, bool],
                controller_failures: Mapping[str, bool],
                aggregate_residual: Mapping[str, float],
                diagnostics: dict[str, Array] | None = None,
            ) -> None:
                if diagnostics is None:
                    diagnostics = {
                        **{name: np.full((len(robot_ids), 2), np.nan) for name in
                           ("control", "nominal_control", "consensus", "formation_term",
                            "goal_term", "learned_field", "true_field", "process_noise",
                            "learned_neighbor_field", "true_neighbor_field")},
                        **{name: np.full(len(robot_ids), np.nan) for name in
                           ("qp_feasible", "fallback_used", "controller_exception", "acceleration_bound_slack", "scored_active")},
                    }
                phase = []
                for robot_id in robot_ids:
                    state = states[robot_id]
                    phase.append(0 if state.waypoint_index <= 1 else (1 if state.task_active else 2))
                trace_recorder.record_frame(
                    time,
                    positions=np.asarray([states[rid].position for rid in robot_ids]),
                    velocities=np.asarray([states[rid].velocity for rid in robot_ids]),
                    waypoint_index=np.asarray([states[rid].waypoint_index for rid in robot_ids]),
                    **diagnostics,
                    headings=np.zeros(len(robot_ids)),
                    phase=np.asarray(phase, dtype=np.int8),
                    dynamics_mismatch=np.asarray([dynamics_mismatch[rid] for rid in robot_ids]),
                    collision_events=np.asarray([collision_events[rid] for rid in robot_ids], dtype=bool),
                    controller_failures=np.asarray([controller_failures[rid] for rid in robot_ids], dtype=bool),
                    aggregate_cbf_residual=np.asarray([aggregate_residual[rid] for rid in robot_ids]),
                    task_edge_active=np.ones(
                        sum(len(graph.edges) for graph in fixed_graphs.values()), dtype=bool
                    ),
                )

            record_frame(
                0.0,
                {robot_id: float("nan") for robot_id in robot_ids},
                {robot_id: False for robot_id in robot_ids},
                {robot_id: False for robot_id in robot_ids},
                {robot_id: float("nan") for robot_id in robot_ids},
            )
        else:
            record_frame = None

        def append_trajectory_frame(
            *,
            time_s: float,
            nominal_commands: Mapping[str, Array] | None = None,
            applied_commands: Mapping[str, Array] | None = None,
            center_references: Mapping[int, tuple[Array, Array, Array]] | None = None,
            learned_position: Mapping[str, Array] | None = None,
            learned_neighbor: Mapping[str, Array] | None = None,
            true_position: Mapping[str, Array] | None = None,
            true_neighbor: Mapping[str, Array] | None = None,
            obstacle_encounter: Mapping[str, Array] | None = None,
            base_true_position: Mapping[str, Array] | None = None,
            base_true_neighbor: Mapping[str, Array] | None = None,
            motion_noise_values: Mapping[str, Array] | None = None,
            center_values: Mapping[int, Array] | None = None,
            z_values: Mapping[int, Mapping[str, Array]] | None = None,
        ) -> None:
            """Append one common time frame to every robot's audit arrays."""

            nominal_commands = nominal_commands or {}
            applied_commands = applied_commands or {}
            learned_position = learned_position or {}
            learned_neighbor = learned_neighbor or {}
            true_position = true_position or {}
            true_neighbor = true_neighbor or {}
            base_true_position = base_true_position or {}
            base_true_neighbor = base_true_neighbor or {}
            motion_noise_values = motion_noise_values or {}
            center_references = center_references or {}
            center_values = center_values or {}
            z_values = z_values or {}
            trajectory_time_s.append(float(time_s))
            for robot_id in robot_ids:
                state = states[robot_id]
                subteam_id = plan_by_robot[robot_id].subteam_id
                center_ref = center_references.get(subteam_id)
                if center_ref is None:
                    center_ref = mission_reference(subteam_id, time_s)
                center = center_values.get(
                    subteam_id,
                    np.mean(
                        [
                            states[rid].position
                            - np.asarray(states[rid].formation_offset, dtype=float)
                            for rid in plan_by_robot[robot_id].home_positions
                        ],
                        axis=0,
                    ),
                )
                z = z_values.get(subteam_id, {}).get(robot_id, center)
                offset = np.asarray(state.formation_offset, dtype=float)
                learned_p = learned_position.get(robot_id, np.zeros(2))
                learned_n = learned_neighbor.get(robot_id, np.zeros(2))
                true_p = true_position.get(robot_id, np.zeros(2))
                true_n = true_neighbor.get(robot_id, np.zeros(2))
                base_true_p = base_true_position.get(robot_id, np.zeros(2))
                base_true_n = base_true_neighbor.get(robot_id, np.zeros(2))
                noise = motion_noise_values.get(robot_id, np.zeros(2))
                p_model_error = true_p - learned_p
                n_model_error = true_n - learned_n
                total_error = p_model_error + n_model_error + noise
                if encounter_amplitude > 0:
                    added_error = (obstacle_encounter or {}).get(robot_id, np.zeros(2))
                    total_error = total_error + added_error
                    trajectory[robot_id]["obstacle_encounter_error"].append(np.asarray(added_error).tolist())
                c_ref = center_ref[0]
                trajectory[robot_id]["position"].append(state.position.tolist())
                trajectory[robot_id]["velocity"].append(state.velocity.tolist())
                trajectory[robot_id]["consensus"].append(np.asarray(z).tolist())
                trajectory[robot_id]["nominal_control"].append(
                    np.asarray(nominal_commands.get(robot_id, np.zeros(2))).tolist()
                )
                trajectory[robot_id]["applied_control"].append(
                    np.asarray(applied_commands.get(robot_id, np.zeros(2))).tolist()
                )
                trajectory[robot_id]["formation_term"].append(
                    (self.formation_position_gains[state.platform]
                     * (np.asarray(z) + offset - state.position)).tolist()
                )
                trajectory[robot_id]["goal_term"].append(
                    (self.formation_goal_gains[state.platform]
                     * (c_ref - np.asarray(z))).tolist()
                )
                trajectory[robot_id]["center_observation"].append(
                    (state.position - offset).tolist()
                )
                trajectory[robot_id]["true_center"].append(np.asarray(center).tolist())
                trajectory[robot_id]["consensus_error"].append(
                    (np.asarray(z) - np.asarray(center)).tolist()
                )
                trajectory[robot_id]["formation_error"].append(
                    (state.position - (np.asarray(center) + offset)).tolist()
                )
                trajectory[robot_id]["center_reference_error"].append(
                    (np.asarray(center) - c_ref).tolist()
                )
                trajectory[robot_id]["position_model_error"].append(
                    p_model_error.tolist()
                )
                trajectory[robot_id]["neighbor_model_error"].append(
                    n_model_error.tolist()
                )
                trajectory[robot_id]["process_noise"].append(np.asarray(noise).tolist())
                trajectory[robot_id]["acceleration_error"].append(total_error.tolist())
                trajectory[robot_id]["true_position_residual"].append(np.asarray(true_p).tolist())
                trajectory[robot_id]["learned_position_residual"].append(np.asarray(learned_p).tolist())
                trajectory[robot_id]["true_neighbor_residual"].append(np.asarray(true_n).tolist())
                trajectory[robot_id]["learned_neighbor_residual"].append(np.asarray(learned_n).tolist())
                trajectory[robot_id]["base_true_position_residual"].append(
                    np.asarray(base_true_p).tolist()
                )
                trajectory[robot_id]["base_true_neighbor_residual"].append(
                    np.asarray(base_true_n).tolist()
                )

        # Use the initial safety checks only for evaluation.
        mission_initial_obstacle_h: dict[tuple[str, str], float] = {}
        for plan in plans:
            subteam_id = plan.subteam_id
            for robot_id in plan.home_positions:
                state = states[robot_id]
                for wall in mission_walls[subteam_id][state.platform]:
                    clearance = wall_clearance(
                        state.position, self.agent_radius, wall
                    )
                    minimum_wall_clearance[subteam_id] = min(
                        minimum_wall_clearance[subteam_id], clearance
                    )
                    robot_min_wall_clearance[robot_id] = min(
                        robot_min_wall_clearance[robot_id], clearance
                    )
                    robot_min_physical_clearance[robot_id] = min(
                        robot_min_physical_clearance[robot_id], clearance
                    )
                    if clearance <= 0.0:
                        mark_robot_failed(
                            subteam_id, robot_id, "unsafe_initial_wall_state"
                        )
                        record_audit_collision(subteam_id, robot_id, "wall")
                for obstacle in mission_obstacles[subteam_id]:
                    if not _obstacle_applies(obstacle, state.platform):
                        continue
                    obstacle_id = str(obstacle.obstacle_id)
                    if self.use_hocbf:
                        q = state.position - obstacle.center
                        initial_h = float(q @ q - (obstacle.radius + self.agent_radius)**2)
                        initial_psi = float(2.0 * q @ state.velocity + self.hocbf_lambda * initial_h)
                        hocbf_initial_pairs.append({
                            "robot_id": robot_id, "obstacle_id": obstacle_id,
                            "h": initial_h, "psi": initial_psi,
                            "margin": float(robot_margins[robot_id]),
                        })
                    else:
                        try:
                            initial_h = wang_barrier(
                                state.position, obstacle.center, state.velocity, np.zeros(2),
                                self.acceleration_limits[state.platform], 0.0,
                                obstacle.radius + self.agent_radius,
                            )
                        except ValueError:
                            initial_h = float("nan")
                    mission_initial_obstacle_h[robot_id, obstacle_id] = initial_h
                    obstacle_clearance = float(
                        np.linalg.norm(state.position - obstacle.center)
                        - obstacle.radius
                        - self.agent_radius
                    )
                    robot_min_physical_clearance[robot_id] = min(
                        robot_min_physical_clearance[robot_id], obstacle_clearance
                    )
                    if np.linalg.norm(state.position - obstacle.center) <= (
                        obstacle.radius + self.agent_radius
                    ):
                        mark_robot_failed(
                            subteam_id,
                            robot_id,
                            "unsafe_initial_circular_obstacle_state",
                        )
                        record_audit_collision(subteam_id, robot_id, "obstacle")
            for left, right in pair_edges[subteam_id]:
                initial_separation = float(
                    np.linalg.norm(states[left].position - states[right].position)
                )
                robot_min_separation[left] = min(
                    robot_min_separation[left], initial_separation
                )
                robot_min_separation[right] = min(
                    robot_min_separation[right], initial_separation
                )
                pair_clearance = initial_separation - 2.0 * self.agent_radius
                robot_min_physical_clearance[left] = min(
                    robot_min_physical_clearance[left], pair_clearance
                )
                robot_min_physical_clearance[right] = min(
                    robot_min_physical_clearance[right], pair_clearance
                )
                if initial_separation <= 2.0 * self.agent_radius:
                    mark_robot_failed(
                        subteam_id, left, f"unsafe_initial_pair_state:{right}"
                    )
                    mark_robot_failed(
                        subteam_id, right, f"unsafe_initial_pair_state:{left}"
                    )
                    record_audit_collision(subteam_id, left, "pair")
                    record_audit_collision(
                        subteam_id, right, "pair", mission_event=False
                    )
                    pair_collision_records[subteam_id].append(
                        {
                            "step": 0,
                            "time_s": 0.0,
                            "kind": "pair",
                            "robot_ids": [left, right],
                            "endpoints": [
                                {
                                    "robot_id": left,
                                    "platform": platforms[left].value,
                                    "score_status": current_score_status(left),
                                },
                                {
                                    "robot_id": right,
                                    "platform": platforms[right].value,
                                    "score_status": current_score_status(right),
                                },
                            ],
                            "separation": initial_separation,
                            "clearance": pair_clearance,
                        }
                    )

        steps_executed = 0
        for step_index in range(self.config.profile.simulation_steps):
            steps_executed = step_index + 1
            step_mismatch = {robot_id: float("nan") for robot_id in robot_ids}
            step_collision = {robot_id: False for robot_id in robot_ids}
            step_failure = {robot_id: False for robot_id in robot_ids}
            diagnostic_qp = {rid: False for rid in robot_ids}
            diagnostic_fallback = {rid: False for rid in robot_ids}
            diagnostic_exception = {rid: False for rid in robot_ids}
            diagnostic_slack = {rid: float("nan") for rid in robot_ids}
            diagnostic_status = {rid: "exception" for rid in robot_ids}
            hocbf_frame = {
                rid: {name: [] for name in hocbf_fields[:6]} for rid in robot_ids
            }
            step_residual = {robot_id: float("nan") for robot_id in robot_ids}
            current_references = {
                plan.subteam_id: mission_reference(plan.subteam_id, step_index * self.dt)
                for plan in plans
            }
            prior_position = {robot_id: states[robot_id].position.copy() for robot_id in robot_ids}
            prior_velocity = {robot_id: states[robot_id].velocity.copy() for robot_id in robot_ids}
            prior_centers = {
                plan.subteam_id: np.mean(
                    [
                        prior_position[rid]
                        - np.asarray(states[rid].formation_offset, dtype=float)
                        for rid in plan.home_positions
                    ],
                    axis=0,
                )
                for plan in plans
            }
            z_control = {
                plan.subteam_id: {
                    robot_id: consensus_state[plan.subteam_id][robot_id].copy()
                    for robot_id in plan.home_positions
                }
                for plan in plans
            }
            # Evaluation variants replace only the center feedback source.
            # Keep their diagnostic z field aligned with the exact or
            # prescribed value used by the nominal command at this time.
            if self.center_feedback_variant == "exact":
                for plan in plans:
                    consensus_state[plan.subteam_id] = {
                        robot_id: prior_centers[plan.subteam_id].copy()
                        for robot_id in plan.home_positions
                    }
                z_control = {
                    plan.subteam_id: {
                        robot_id: consensus_state[plan.subteam_id][robot_id].copy()
                        for robot_id in plan.home_positions
                    }
                    for plan in plans
                }
            elif self.center_feedback_variant == "prescribed":
                for plan in plans:
                    consensus_state[plan.subteam_id] = {
                        robot_id: current_references[plan.subteam_id][0].copy()
                        for robot_id in plan.home_positions
                    }
                z_control = {
                    plan.subteam_id: {
                        robot_id: consensus_state[plan.subteam_id][robot_id].copy()
                        for robot_id in plan.home_positions
                    }
                    for plan in plans
                }
            nominal: dict[str, Array] = {}
            for plan in plans:
                c_ref, v_ref, a_ref = current_references[plan.subteam_id]
                for robot_id in plan.home_positions:
                    state = states[robot_id]
                    offset = np.asarray(state.formation_offset, dtype=float)
                    if self.center_feedback_variant == "exact":
                        center_feedback = prior_centers[plan.subteam_id]
                    elif self.center_feedback_variant == "prescribed":
                        center_feedback = c_ref
                    else:
                        center_feedback = consensus_state[plan.subteam_id][robot_id]
                    nominal[robot_id] = nominal_formation_control(
                        state.position,
                        state.velocity,
                        offset,
                        center_feedback,
                        c_ref,
                        v_ref,
                        a_ref,
                        formation_gain=self.formation_position_gains[state.platform],
                        goal_gain=self.formation_goal_gains[state.platform],
                        velocity_gain=self.velocity_gains[state.platform],
                    )
            # All robots remain in the fixed scored interval, including a
            # robot with an application failure or a reference that holds the final position.
            scored_active = {robot_id: True for robot_id in robot_ids}
            learned_position = {
                robot_id: evaluate_canonical_class_field(
                    self.models,
                    platforms[robot_id],
                    prior_position[robot_id],
                    realizations[plan_by_robot[robot_id].subteam_id].frame,
                    learned=True,
                )
                for robot_id in robot_ids
            }
            learned_neighbor = self._neighbor_fields(plans, prior_position, learned=True)
            learned = {
                rid: learned_position[rid] + learned_neighbor[rid] for rid in robot_ids
            }
            commands: dict[str, Array] = {}
            step_decay_gamma: dict[str, dict[str, float]] = {
                robot_id: {} for robot_id in robot_ids
            }
            for plan in plans:
                subteam_id = plan.subteam_id
                # Use the same predicted acceleration in the controller and the score.
                controllers[subteam_id].learned_model = {
                    rid: learned[rid] for rid in plan.home_positions
                }
                if self.use_hocbf:
                    controller_prediction_error[subteam_id] = max(
                        controller_prediction_error[subteam_id],
                        max(float(np.linalg.norm(controllers[subteam_id].learned_model[rid]
                                                 - learned_position[rid] - learned_neighbor[rid]))
                            for rid in plan.home_positions),
                    )
                try:
                    outputs = controllers[subteam_id].step(
                        self._controller_states(
                            controllers[subteam_id],
                            plan.home_positions,
                            states,
                            robot_margins,
                        ),
                        {robot_id: nominal[robot_id] for robot_id in plan.home_positions},
                    )
                except (ValueError, FloatingPointError, OverflowError) as exc:
                    record_audit_exception(
                        subteam_id, "controller_step", exc, steps_executed
                    )
                    global_failures[subteam_id].append(
                        f"controller_invalid_state:{type(exc).__name__}"
                    )
                    for robot_id in plan.home_positions:
                        diagnostic_exception[robot_id] = True
                        diagnostic_fallback[robot_id] = True
                        step_failure[robot_id] = True
                        qp_failures[subteam_id] += 1
                        robot_qp_failures[robot_id] += 1
                        robot_qp_fallbacks[robot_id] += 1
                        commands[robot_id] = np.zeros(2)
                        if record_full_qp_diagnostics:
                            qp_diagnostic_records[robot_id].append(
                                {
                                    "feasible": False,
                                    "status": f"controller_exception:{type(exc).__name__}",
                                    "active_constraint_keys": [],
                                    "active_normal_matrix": [],
                                    "min_active_singular_value": float("nan"),
                                    "condition_number": float("inf"),
                                    "control_norm": 0.0,
                                    "acceleration_limit_slack": float("nan"),
                                    "objective": float("nan"),
                                    "optimized_decay_multipliers": {},
                                    "optimized_decay_coefficients": {},
                                    "minimum_decay_multiplier": float("nan"),
                                    "maximum_decay_multiplier": float("nan"),
                                    "fraction_decay_multipliers_changed": float("nan"),
                                    "decay_penalty": float("nan"),
                                }
                            )
                    continue
                for robot_id in plan.home_positions:
                    qp_solves[subteam_id] += 1
                    robot_qp_solves[robot_id] += 1
                    try:
                        output = outputs[robot_id]
                        diagnostics = output.qp_diagnostics
                        command = _as_vector(
                            output.control, name="controller output"
                        )
                    except (KeyError, AttributeError, TypeError, ValueError) as exc:
                        record_audit_exception(
                            subteam_id,
                            f"controller_output:{robot_id}",
                            exc,
                            steps_executed,
                        )
                        mark_robot_failed(
                            subteam_id,
                            robot_id,
                            f"controller_invalid_state:{type(exc).__name__}",
                        )
                        step_failure[robot_id] = True
                        qp_failures[subteam_id] += 1
                        robot_qp_failures[robot_id] += 1
                        robot_qp_fallbacks[robot_id] += 1
                        commands[robot_id] = np.zeros(2)
                        diagnostic_exception[robot_id] = True
                        diagnostic_fallback[robot_id] = True
                        continue
                    diagnostic_qp[robot_id] = bool(diagnostics.feasible)
                    diagnostic_status[robot_id] = str(diagnostics.status)
                    diagnostic_fallback[robot_id] = bool(output.failed or not diagnostics.feasible)
                    diagnostic_slack[robot_id] = float(diagnostics.acceleration_limit_slack)
                    if diagnostics.feasible and not output.failed:
                        robot_qp_feasible_solves[robot_id] += 1
                    if output.failed or not diagnostics.feasible:
                        # Count failed solves per time step and failed robots per mission.
                        robot_qp_failures[robot_id] += 1
                        robot_qp_fallbacks[robot_id] += 1
                    if (
                        robot_id not in failed_robot_ids[subteam_id]
                        and (output.failed or not diagnostics.feasible)
                    ):
                        mark_robot_failed(
                            subteam_id, robot_id, "controller_local_qp_failure"
                        )
                        step_failure[robot_id] = True
                        qp_failures[subteam_id] += 1
                        command = np.zeros(2)
                    commands[robot_id] = command
                    step_decay_gamma[robot_id] = dict(
                        diagnostics.optimized_decay_coefficients
                    )
                    omega_values = tuple(
                        diagnostics.optimized_decay_multipliers.values()
                    )
                    total_decay_multipliers[subteam_id] += len(omega_values)
                    changed_decay_multipliers[subteam_id] += sum(
                        abs(omega - 1.0) > 1e-3 for omega in omega_values
                    )
                    if omega_values:
                        minimum_decay_multiplier[subteam_id] = min(
                            minimum_decay_multiplier[subteam_id], min(omega_values)
                        )
                        maximum_decay_multiplier[subteam_id] = max(
                            maximum_decay_multiplier[subteam_id], max(omega_values)
                        )
                    if np.isfinite(diagnostics.decay_penalty):
                        total_decay_penalty[subteam_id] += diagnostics.decay_penalty
                    if record_full_qp_diagnostics:
                        qp_diagnostic_records[robot_id].append(
                            {
                                "feasible": bool(diagnostics.feasible),
                                "status": str(diagnostics.status),
                                "active_constraint_keys": list(
                                    diagnostics.active_constraint_keys
                                ),
                                "active_normal_matrix": np.asarray(
                                    diagnostics.active_normal_matrix, dtype=float
                                ).tolist(),
                                "min_active_singular_value": float(
                                    diagnostics.min_active_singular_value
                                ),
                                "condition_number": float(
                                    diagnostics.active_condition_number
                                ),
                                "control_norm": float(diagnostics.control_norm),
                                "acceleration_limit_slack": float(
                                    diagnostics.acceleration_limit_slack
                                ),
                                "objective": float(diagnostics.objective),
                                "optimized_decay_multipliers": dict(
                                    diagnostics.optimized_decay_multipliers
                                ),
                                "optimized_decay_coefficients": dict(
                                    diagnostics.optimized_decay_coefficients
                                ),
                                "minimum_decay_multiplier": float(
                                    diagnostics.minimum_decay_multiplier
                                ),
                                "maximum_decay_multiplier": float(
                                    diagnostics.maximum_decay_multiplier
                                ),
                                "fraction_decay_multipliers_changed": float(
                                    diagnostics.fraction_decay_multipliers_changed
                                ),
                                "decay_penalty": float(diagnostics.decay_penalty),
                            }
                        )
                    maximum_control_norm[subteam_id] = max(
                        maximum_control_norm[subteam_id], float(np.linalg.norm(command))
                    )
                    maximum_control_inf_norm[subteam_id] = max(
                        maximum_control_inf_norm[subteam_id], float(np.linalg.norm(command, ord=np.inf))
                    )
                    maximum_acceleration_slack[subteam_id] = min(
                        maximum_acceleration_slack[subteam_id], diagnostics.acceleration_limit_slack
                    )
                    if np.isfinite(diagnostics.acceleration_limit_slack):
                        robot_min_acceleration_slack[robot_id] = min(
                            robot_min_acceleration_slack[robot_id],
                            float(diagnostics.acceleration_limit_slack),
                        )
                    singular = float(diagnostics.min_active_singular_value)
                    condition = float(diagnostics.active_condition_number)
                    if np.isfinite(singular):
                        robot_min_active_sigma[robot_id] = min(
                            robot_min_active_sigma[robot_id], singular
                        )
                        minimum_active_sigma[subteam_id] = min(minimum_active_sigma[subteam_id], singular)
                        if singular <= 1e-12:
                            robot_singular_qps[robot_id] += 1
                            singular_qps[subteam_id] += 1
                    if np.isfinite(condition):
                        robot_max_condition[robot_id] = max(
                            robot_max_condition[robot_id], condition
                        )
                        maximum_condition[subteam_id] = max(maximum_condition[subteam_id], condition)
                        if condition > 1e8:
                            robot_ill_conditioned_qps[robot_id] += 1
                            ill_conditioned_qps[subteam_id] += 1
                    elif np.isinf(condition):
                        robot_max_condition[robot_id] = float("inf")
                        robot_ill_conditioned_qps[robot_id] += 1
                        maximum_condition[subteam_id] = float("inf")
                        ill_conditioned_qps[subteam_id] += 1

            # Truth and process noise enter only after every local controller
            # has selected its input.  Neither appears in a Wang packet/QP.
            base_true_position = {
                robot_id: evaluate_canonical_class_field(
                    self.models,
                    platforms[robot_id],
                    prior_position[robot_id],
                    realizations[plan_by_robot[robot_id].subteam_id].frame,
                    learned=False,
                )
                for robot_id in robot_ids
            }
            base_true_neighbor = self._neighbor_fields(plans, prior_position, learned=False)
            # Keep the learned model visible to the controller while varying
            # only the existing model error term in the physical plant.
            # One scalar is shared by all six robots in each subteam mission.
            true_position = {
                robot_id: learned_position[robot_id]
                + dynamics_severity_multiplier[plan_by_robot[robot_id].subteam_id]
                * (base_true_position[robot_id] - learned_position[robot_id])
                for robot_id in robot_ids
            }
            true_neighbor = {
                robot_id: learned_neighbor[robot_id]
                + dynamics_severity_multiplier[plan_by_robot[robot_id].subteam_id]
                * (base_true_neighbor[robot_id] - learned_neighbor[robot_id])
                for robot_id in robot_ids
            }
            true = {
                rid: true_position[rid] + true_neighbor[rid] for rid in robot_ids
            }
            encounter_field = {}
            if encounter_amplitude > 0:
                for rid in robot_ids:
                    centers, radii, peers = encounter_geometry[rid]
                    added, _ = obstacle_encounter_acceleration(
                        prior_position[rid], [prior_position[peer] for peer in peers],
                        centers, radii, amplitude=encounter_amplitude,
                        band=self.config.formation_obstacle_disturbance_band_m,
                        peer_radius=self.config.collision_neighbor_radius,
                        strength=encounter_strength[plan_by_robot[rid].subteam_id],
                        mode=self.config.formation_obstacle_disturbance_mode)
                    encounter_field[rid] = added
                    true[rid] = true[rid] + added
                    encounter_diagnostics[rid]["maximum_added_acceleration"] = max(
                        encounter_diagnostics[rid]["maximum_added_acceleration"],
                        float(np.linalg.norm(added)))
            motion_noise = {
                robot_id: realizations[plan_by_robot[robot_id].subteam_id].motion_noise[robot_id][step_index]
                for robot_id in robot_ids
            }
            # Store the causal frame at time k: p_k, v_k, z_k and the command
            # selected from those values.  The state after the step is appended
            # once after the final executed interval below.
            append_trajectory_frame(
                time_s=step_index * self.dt,
                nominal_commands=nominal,
                applied_commands=commands,
                center_references=current_references,
                learned_position=learned_position,
                learned_neighbor=learned_neighbor,
                true_position=true_position,
                true_neighbor=true_neighbor,
                obstacle_encounter=encounter_field,
                base_true_position=base_true_position,
                base_true_neighbor=base_true_neighbor,
                motion_noise_values=motion_noise,
                center_values=prior_centers,
                z_values=z_control,
            )
            interval_clearance_by_robot: dict[str, float] = {
                robot_id: float("inf") for robot_id in robot_ids
            }
            for plan in plans:
                subteam_id = plan.subteam_id
                for robot_id in plan.home_positions:
                    acceleration = commands[robot_id] + true[robot_id] + motion_noise[robot_id]
                    for obstacle_index, obstacle in enumerate(mission_obstacles[subteam_id]):
                        if not _obstacle_applies(obstacle, platforms[robot_id]):
                            continue
                        obstacle_id = str(
                            getattr(obstacle, "obstacle_id", None)
                            or f"pillar:{obstacle_index}"
                        )
                        footprint_radius = obstacle.radius + self.agent_radius
                        pre_clearance = float(
                            np.linalg.norm(prior_position[robot_id] - obstacle.center)
                            - footprint_radius
                        )
                        if self.use_hocbf:
                            initial_q = prior_position[robot_id] - obstacle.center
                            initial_h = float(initial_q @ initial_q - footprint_radius**2)
                        else:
                            try:
                                initial_h = wang_barrier(
                                    prior_position[robot_id],
                                    obstacle.center,
                                    prior_velocity[robot_id],
                                    np.zeros(2),
                                    self.acceleration_limits[platforms[robot_id]],
                                    0.0,
                                    footprint_radius,
                                )
                            except ValueError:
                                initial_h = float("nan")
                        interval_min, interval_tau = quadratic_segment_circle_clearance(
                            prior_position[robot_id],
                            prior_velocity[robot_id],
                            acceleration,
                            obstacle.center,
                            footprint_radius,
                            self.dt,
                        )
                        interval_clearance_by_robot[robot_id] = min(
                            interval_clearance_by_robot[robot_id], interval_min
                        )
                        if self.use_hocbf:
                            q = prior_position[robot_id] - obstacle.center
                            v = prior_velocity[robot_id]
                            lam = self.hocbf_lambda
                            h = float(q @ q - footprint_radius**2)
                            psi = float(2.0 * q @ v + lam * h)
                            obstacle_key = f"obstacle:{robot_id}:{obstacle_id}"
                            omega = float(step_decay_gamma[robot_id].get(obstacle_key, lam) / lam)
                            tightening = float(
                                2.0
                                * np.linalg.norm(q)
                                * robot_margins[robot_id]
                            )
                            maximum_obstacle_tightening[subteam_id] = max(
                                maximum_obstacle_tightening[subteam_id], tightening)
                            robot_max_obstacle_tightening[robot_id] = max(
                                robot_max_obstacle_tightening[robot_id], tightening)
                            implemented = float(2.0 * q @ (commands[robot_id] + learned[robot_id])
                                                + 2.0 * v @ v + 2.0 * lam * q @ v
                                                + omega * lam * psi)
                            slack = implemented - tightening
                            reference_residual = float(implemented + 2.0 * q @ (
                                true[robot_id] + motion_noise[robot_id] - learned[robot_id]))
                            if np.linalg.norm(q) - footprint_radius <= self.config.formation_obstacle_disturbance_band_m:
                                ed = encounter_diagnostics[robot_id]
                                error = true[robot_id] + motion_noise[robot_id] - learned[robot_id]
                                ed["near_obstacle_constraint_samples"] += 1
                                ed["near_uncovered_constraint_samples"] += int(np.linalg.norm(error) > robot_margins[robot_id])
                                ed["maximum_inward_error_near_obstacles"] = max(
                                    ed["maximum_inward_error_near_obstacles"],
                                    float(-q @ error / max(np.linalg.norm(q), 1e-12)))
                                old = ed["minimum_true_hocbf_residual_near_obstacles"]
                                ed["minimum_true_hocbf_residual_near_obstacles"] = reference_residual if old is None else min(old, reference_residual)
                            audit_key = (subteam_id, robot_id, obstacle_id, int(steps_executed))
                            obstacle_audit = {
                                "h": h, "psi": psi, "omega": omega,
                                "lambda1": lam, "lambda2": lam,
                                "initial_h": h, "invalid_initial_h": not np.isfinite(h) or h < 0.0,
                                "omega_is_optimized": (diagnostic_qp[robot_id]
                                                       and not diagnostic_fallback[robot_id]
                                                       and obstacle_key in step_decay_gamma[robot_id]),
                                "implemented_residual": implemented,
                                "true_residual": reference_residual,
                                "required_tightening": tightening,
                                "slack": slack, "robust_hocbf_slack": slack,
                                "qp_feasible": bool(diagnostic_qp[robot_id]),
                                "qp_status": diagnostic_status[robot_id],
                                "fallback_used": bool(diagnostic_fallback[robot_id]),
                                "controller_exception": bool(diagnostic_exception[robot_id]),
                                "d_hat": learned[robot_id].tolist(),
                                "epsilon": (true[robot_id] + motion_noise[robot_id] - learned[robot_id]).tolist(),
                                "epsilon_norm": float(np.linalg.norm(true[robot_id] + motion_noise[robot_id] - learned[robot_id])),
                                "score_error_norm": float(np.linalg.norm(true[robot_id] + motion_noise[robot_id] - learned[robot_id])),
                                "margin": float(robot_margins[robot_id]),
                                "interval_minimum_clearance": float(interval_min),
                                "minimum_clearance_tau": float(interval_tau),
                            }
                            obstacle_step_diagnostics[audit_key] = obstacle_audit
                            for name, value in (
                                ("hocbf_h", h), ("hocbf_psi", psi), ("hocbf_slack", slack),
                                ("hocbf_omega", omega), ("hocbf_interval_clearance", interval_min),
                                ("hocbf_interval_tau", interval_tau),
                            ):
                                hocbf_frame[robot_id][name].append(float(value))
                            if diagnostic_qp[robot_id] and not diagnostic_fallback[robot_id]:
                                hocbf_min_slack[subteam_id] = min(hocbf_min_slack[subteam_id], slack)
                                hocbf_success_count[subteam_id] += 1
                                if slack < -1e-7:
                                    mark_robot_failed(subteam_id, robot_id, "robust_hocbf_violation")
                            minimum_implemented[subteam_id] = min(minimum_implemented[subteam_id], implemented)
                            minimum_true[subteam_id] = min(minimum_true[subteam_id], reference_residual)
                            robot_min_implemented_residual[robot_id] = min(robot_min_implemented_residual[robot_id], implemented)
                            robot_min_true_residual[robot_id] = min(robot_min_true_residual[robot_id], reference_residual)
                            step_residual[robot_id] = slack if np.isnan(step_residual[robot_id]) else min(step_residual[robot_id], slack)
                            robot_min_physical_clearance[robot_id] = min(robot_min_physical_clearance[robot_id], interval_min)
                        if interval_min <= 0.0:
                            mark_robot_failed(
                                subteam_id, robot_id, "intersample_obstacle_collision"
                            )
                            record_audit_collision(subteam_id, robot_id, "obstacle")
                            step_collision[robot_id] = True
                            collisions[subteam_id] += 1
                            obstacle_collisions[subteam_id] += 1
                            try:
                                first_violation_tau = quadratic_segment_circle_first_violation(
                                    prior_position[robot_id],
                                    prior_velocity[robot_id],
                                    acceleration,
                                    obstacle.center,
                                    footprint_radius,
                                    self.dt,
                                )
                            except ValueError:
                                first_violation_tau = None
                            event = {
                                "step": int(steps_executed),
                                "step_index": int(step_index),
                                "time_s": float((step_index + interval_tau) * self.dt),
                                "robot_id": robot_id,
                                "obstacle_index": int(obstacle_index),
                                "obstacle_id": obstacle_id,
                                "source": "quadratic_interval",
                                "interval_tau": float(interval_tau),
                                "interval_minimum_clearance": float(interval_min),
                                "minimum_clearance_tau": float(interval_tau),
                                "minimum_clearance_time_s": float(
                                    (step_index + interval_tau) * self.dt
                                ),
                                "first_violation_tau": (
                                    None
                                    if first_violation_tau is None
                                    else float(first_violation_tau)
                                ),
                                "first_violation_time_s": (
                                    None
                                    if first_violation_tau is None
                                    else float(
                                        (step_index + first_violation_tau) * self.dt
                                    )
                                ),
                                "pre_clearance": pre_clearance,
                                "sample_clearance": None,
                                "clearance": float(interval_min),
                                "epsilon": (
                                    true[robot_id]
                                    + motion_noise[robot_id]
                                    - learned[robot_id]
                                ).tolist(),
                                "epsilon_norm": float(
                                    np.linalg.norm(
                                        true[robot_id]
                                        + motion_noise[robot_id]
                                        - learned[robot_id]
                                    )
                                ),
                                "score_error_norm": _audit_float(
                                    np.linalg.norm(
                                        true[robot_id]
                                        + motion_noise[robot_id]
                                        - learned[robot_id]
                                    )
                                ),
                                "initial_h": _audit_float(initial_h),
                                    "invalid_initial_h": (
                                        not np.isfinite(initial_h) or initial_h < 0.0
                                    ),
                                "margin": float(robot_margins[robot_id]),
                                "qp_feasible": bool(diagnostic_qp[robot_id]),
                                "fallback_used": bool(diagnostic_fallback[robot_id]),
                                "controller_exception": bool(
                                    diagnostic_exception[robot_id]
                                ),
                            }
                            obstacle_collision_records[subteam_id].append(event)
                            obstacle_collision_event_index[
                                (subteam_id, robot_id, obstacle_id, int(steps_executed))
                            ] = event
                            if self.use_hocbf:
                                event.update(obstacle_step_diagnostics[audit_key])
            if self.use_hocbf:
                for robot_id in robot_ids:
                    for name, values in hocbf_frame[robot_id].items():
                        trajectory[robot_id][name].append(values)
                    trajectory[robot_id]["qp_status_code"].append(
                        float(hocbf_status_codes.setdefault(
                            diagnostic_status[robot_id], -len(hocbf_status_codes)
                        ))
                    )
                    trajectory[robot_id]["qp_feasible"].append(float(diagnostic_qp[robot_id]))
                    trajectory[robot_id]["fallback_used"].append(float(diagnostic_fallback[robot_id]))
            for plan in plans:
                subteam_id = plan.subteam_id
                for robot_id in plan.home_positions:
                    propagate_true_wang_dynamics(
                        states[robot_id],
                        commands[robot_id],
                        true[robot_id],
                        self.dt,
                        motion_noise=motion_noise[robot_id],
                        home_tolerance=self.config.home_tolerance,
                    )
                    for obstacle_index, obstacle in enumerate(
                        mission_obstacles[subteam_id]
                    ):
                        if not _obstacle_applies(obstacle, platforms[robot_id]):
                            continue
                        obstacle_id = str(
                            getattr(obstacle, "obstacle_id", None)
                            or f"pillar:{obstacle_index}"
                        )
                        event = obstacle_collision_event_index.get(
                            (subteam_id, robot_id, obstacle_id, int(steps_executed))
                        )
                        if event is not None:
                            event["sample_clearance"] = float(
                                np.linalg.norm(states[robot_id].position - obstacle.center)
                                - obstacle.radius
                                - self.agent_radius
                            )
                    if scored_active[robot_id]:
                        maximum_motion_noise_norm[subteam_id] = max(
                            maximum_motion_noise_norm[subteam_id], float(np.linalg.norm(motion_noise[robot_id]))
                        )
                    if np.isfinite(interval_clearance_by_robot[robot_id]):
                        robot_min_physical_clearance[robot_id] = min(
                            robot_min_physical_clearance[robot_id],
                            interval_clearance_by_robot[robot_id],
                        )
                    speed = float(np.linalg.norm(states[robot_id].velocity))
                    maximum_speed[subteam_id] = max(maximum_speed[subteam_id], speed)
                    if speed > self.speed_limits[platforms[robot_id]] + 1e-7:
                        speed_violations[subteam_id] += 1
                    # Score independently of flags for application failures, including
                    # after residual certification has stopped for this robot.
                    if scored_active[robot_id]:
                        score_attempts[robot_id] += 1
                        try:
                            if not all(
                                np.all(np.isfinite(value))
                                for value in (
                                    prior_position[robot_id], prior_velocity[robot_id],
                                    states[robot_id].position, states[robot_id].velocity,
                                )
                            ):
                                raise ValueError("Nonfinite scoring state")
                            mismatch = wang_acceleration_model_error(
                                true[robot_id], motion_noise[robot_id], learned[robot_id]
                            )
                            reconstructed = float(np.linalg.norm(
                                true_position[robot_id] + true_neighbor[robot_id] + motion_noise[robot_id]
                                - learned_position[robot_id] - learned_neighbor[robot_id]
                                + encounter_field.get(robot_id, np.zeros(2))))
                            score_reconstruction_error[subteam_id] = max(
                                score_reconstruction_error[subteam_id], abs(float(mismatch) - reconstructed))
                        except ValueError as exc:
                            record_audit_exception(
                                subteam_id,
                                f"score:{robot_id}",
                                exc,
                                steps_executed,
                            )
                            mismatch = float("nan")
                    else:
                        mismatch = float("nan")
                    step_mismatch[robot_id] = mismatch
                    if scored_active[robot_id]:
                        for event in obstacle_collision_records[subteam_id]:
                            if (
                                event.get("robot_id") == robot_id
                                and event.get("step") == int(steps_executed)
                            ):
                                event["score_error_norm"] = _audit_float(mismatch)
                        if not np.isfinite(mismatch):
                            score_invalid[robot_id] = True
                            mark_robot_failed(
                                subteam_id,
                                robot_id,
                                "nonfinite_acceleration_model_error",
                            )
                            scores[subteam_id][platforms[robot_id]][robot_id] = float("inf")
                        else:
                            raw_score_samples[robot_id] += 1
                            prior_raw_score = raw_score_max[robot_id]
                            raw_score_max[robot_id] = (
                                float(mismatch)
                                if prior_raw_score is None
                                else max(float(prior_raw_score), float(mismatch))
                            )
                            if prior_raw_score is None or float(mismatch) >= float(prior_raw_score):
                                score_argmax_step[robot_id] = int(step_index)
                                score_argmax_components[robot_id] = {
                                    "step": int(step_index),
                                    "control_step": int(steps_executed),
                                    "time_s": float(step_index * self.dt),
                                    "norm": float(mismatch),
                                    "position_model_error": (
                                        true_position[robot_id] - learned_position[robot_id]
                                    ).tolist(),
                                    "neighbor_model_error": (
                                        true_neighbor[robot_id] - learned_neighbor[robot_id]
                                    ).tolist(),
                                    "process_noise": motion_noise[robot_id].tolist(),
                                    "total_acceleration_error": (
                                        true[robot_id] + motion_noise[robot_id] - learned[robot_id]
                                    ).tolist(),
                                }
                            robot_max_acceleration_error[robot_id] = raw_score_max[robot_id]
                            if encounter_amplitude > 0 and score_argmax_step[robot_id] == int(step_index):
                                score_argmax_components[robot_id]["obstacle_encounter_error"] = encounter_field[robot_id].tolist()
                            maximum_acceleration_error[subteam_id] = max(
                                maximum_acceleration_error[subteam_id], mismatch
                            )
                            if np.isfinite(scores[subteam_id][platforms[robot_id]][robot_id]):
                                    scores[subteam_id][platforms[robot_id]][robot_id] = max(
                                    scores[subteam_id][platforms[robot_id]][robot_id], mismatch
                                )

            # The innovation uses only the observations available after this
            # physical step.  Every mixing round reads one immutable snapshot,
            # so robot iteration order cannot affect z[k+1].
            next_observations: dict[int, dict[str, Array]] = {}
            for plan in plans:
                next_observations[plan.subteam_id] = {
                    robot_id: states[robot_id].position
                        - np.asarray(states[robot_id].formation_offset, dtype=float)
                    for robot_id in plan.home_positions
                }
                distributed_estimate = consensus_workers[plan.subteam_id].step(
                    next_observations[plan.subteam_id]
                )
                if self.center_feedback_variant == "production":
                    consensus_state[plan.subteam_id] = distributed_estimate
                elif self.center_feedback_variant == "exact":
                    center = np.mean(tuple(next_observations[plan.subteam_id].values()), axis=0)
                    consensus_state[plan.subteam_id] = {
                        robot_id: center.copy() for robot_id in plan.home_positions
                    }
                else:
                    next_reference = mission_reference(plan.subteam_id, (step_index + 1) * self.dt)[0]
                    consensus_state[plan.subteam_id] = {
                        robot_id: next_reference.copy() for robot_id in plan.home_positions
                    }
            # Hold the center reference at the destination after arrival.
            time_s = steps_executed * self.dt
            for robot_id in robot_ids:
                state = states[robot_id]
                at_destination = (
                    float(np.linalg.norm(state.position - state.waypoints[1]))
                    <= completion_position_tolerance
                    and float(np.linalg.norm(state.velocity))
                    <= completion_velocity_tolerance
                )
                destination_reached[robot_id] = at_destination
                if at_destination and waypoint_arrival_times[robot_id] is None:
                    waypoint_arrival_times[robot_id] = time_s

            def register_residual(
                subteam_id: int,
                participants: tuple[str, ...],
                implemented: float,
                reference: float,
                required_tightening: float,
                kind: str | None = None,
            ) -> None:
                if not np.isfinite(implemented) or not np.isfinite(reference):
                    for participant in participants:
                        mark_robot_failed(
                            subteam_id, participant, "nonfinite_wang_residual"
                        )
                    return
                minimum_implemented[subteam_id] = min(minimum_implemented[subteam_id], implemented)
                minimum_true[subteam_id] = min(minimum_true[subteam_id], reference)
                for participant in participants:
                    robot_min_implemented_residual[participant] = min(
                        robot_min_implemented_residual[participant], implemented
                    )
                    robot_min_true_residual[participant] = min(
                        robot_min_true_residual[participant], reference
                    )
                    if kind == "pair":
                        robot_max_pair_tightening[participant] = max(
                            robot_max_pair_tightening[participant], required_tightening
                        )
                    elif kind == "obstacle":
                        robot_max_obstacle_tightening[participant] = max(
                            robot_max_obstacle_tightening[participant], required_tightening
                        )
                    elif kind == "wall":
                        robot_max_wall_tightening[participant] = max(
                            robot_max_wall_tightening[participant], required_tightening
                        )
                if implemented < required_tightening - 1e-6:
                    for participant in participants:
                        mark_robot_failed(
                            subteam_id, participant, "tightened_wang_violation"
                        )
                for participant in participants:
                    step_residual[participant] = (
                        implemented if np.isnan(step_residual[participant]) else min(step_residual[participant], implemented)
                    )

            for plan in plans:
                subteam_id = plan.subteam_id
                for robot_id in plan.home_positions:
                    if self.use_hocbf:
                        continue
                    if not scored_active[robot_id]:
                        continue
                    vehicle = states[robot_id]
                    for wall in mission_walls[subteam_id][vehicle.platform]:
                        if robot_id in failed_robot_ids[subteam_id]:
                            break
                        wall_key = f"wall:{robot_id}:{wall.wall_id}"
                        residual_gamma = step_decay_gamma[robot_id].get(
                            wall_key, self.gamma
                        )
                        try:
                            implemented = wall_B_hat(
                                prior_position[robot_id],
                                prior_velocity[robot_id],
                                commands[robot_id],
                                self.acceleration_limits[vehicle.platform],
                                self.agent_radius,
                                wall,
                                residual_gamma,
                                learned[robot_id],
                            )
                            reference = wall_B_true(
                                prior_position[robot_id],
                                prior_velocity[robot_id],
                                commands[robot_id],
                                self.acceleration_limits[vehicle.platform],
                                self.agent_radius,
                                wall,
                                residual_gamma,
                                true[robot_id],
                                motion_noise[robot_id],
                            )
                            tightening = float(robot_margins[robot_id])
                        except ValueError as exc:
                            record_audit_exception(
                                subteam_id,
                                f"wall_residual:{robot_id}",
                                exc,
                                steps_executed,
                            )
                            implemented = reference = float("nan")
                            tightening = float("inf")
                        register_residual(
                            subteam_id,
                            (robot_id,),
                            implemented,
                            reference,
                            tightening,
                            "wall",
                        )
                    for obstacle_index, obstacle in enumerate(mission_obstacles[subteam_id]):
                        if not _obstacle_applies(obstacle, vehicle.platform):
                            continue
                        obstacle_id = str(
                            getattr(obstacle, "obstacle_id", None)
                            or f"pillar:{obstacle_index}"
                        )
                        obstacle_key = f"obstacle:{robot_id}:{obstacle_id}"
                        residual_gamma = step_decay_gamma[robot_id].get(
                            obstacle_key, self.gamma
                        )
                        tightening = obstacle_robust_tightening(
                            prior_position[robot_id],
                            obstacle.center,
                            robot_margins[robot_id],
                        )
                        try:
                            initial_h = wang_barrier(
                                prior_position[robot_id],
                                obstacle.center,
                                prior_velocity[robot_id],
                                np.zeros(2),
                                self.acceleration_limits[vehicle.platform],
                                0.0,
                                obstacle.radius + self.agent_radius,
                            )
                        except ValueError:
                            initial_h = float("nan")
                        try:
                            implemented, reference = wang_obstacle_residuals(
                                prior_position[robot_id],
                                prior_velocity[robot_id],
                                commands[robot_id],
                                obstacle.center,
                                learned[robot_id],
                                true[robot_id],
                                self.acceleration_limits[vehicle.platform],
                                obstacle.radius + self.agent_radius,
                                residual_gamma,
                                motion_noise=motion_noise[robot_id],
                            )
                        except ValueError as exc:
                            record_audit_exception(
                                subteam_id,
                                f"obstacle_residual:{robot_id}",
                                exc,
                                steps_executed,
                            )
                            implemented = reference = float("nan")
                        if np.isfinite(tightening):
                            maximum_obstacle_tightening[subteam_id] = max(
                                maximum_obstacle_tightening[subteam_id], tightening
                            )
                        register_residual(
                            subteam_id,
                            (robot_id,),
                            implemented,
                            reference,
                            tightening,
                            "obstacle",
                        )
                        obstacle_audit = {
                            "implemented_residual": _audit_float(implemented),
                            "true_residual": _audit_float(reference),
                            "required_tightening": _audit_float(tightening),
                            "slack": _audit_float(
                                implemented - tightening
                                if np.isfinite(implemented) and np.isfinite(tightening)
                                else float("nan")
                            ),
                            "initial_h": _audit_float(initial_h),
                            "invalid_initial_h": (
                                not np.isfinite(initial_h) or initial_h < 0.0
                            ),
                            "qp_feasible": bool(diagnostic_qp[robot_id]),
                            "fallback_used": bool(diagnostic_fallback[robot_id]),
                            "controller_exception": bool(
                                diagnostic_exception[robot_id]
                            ),
                            "epsilon": (
                                true[robot_id]
                                + motion_noise[robot_id]
                                - learned[robot_id]
                            ).tolist(),
                            "epsilon_norm": float(
                                np.linalg.norm(
                                    true[robot_id]
                                    + motion_noise[robot_id]
                                    - learned[robot_id]
                                )
                            ),
                            "score_error_norm": _audit_float(
                                np.linalg.norm(
                                    true[robot_id]
                                    + motion_noise[robot_id]
                                    - learned[robot_id]
                                )
                            ),
                            "margin": float(robot_margins[robot_id]),
                        }
                        audit_key = (
                            subteam_id,
                            robot_id,
                            obstacle_id,
                            int(steps_executed),
                        )
                        obstacle_step_diagnostics[audit_key] = obstacle_audit
                        event = obstacle_collision_event_index.get(audit_key)
                        if event is not None:
                            event.update(obstacle_audit)
                for left, right in pair_edges[subteam_id]:
                    if (
                        self.use_hocbf
                        or global_failures[subteam_id]
                        or not scored_active[left]
                        or not scored_active[right]
                        or left in failed_robot_ids[subteam_id]
                        or right in failed_robot_ids[subteam_id]
                    ):
                        continue
                    try:
                        eta_left, eta_right = wang_allocation_weights(
                            self.acceleration_limits[platforms[left]],
                            self.acceleration_limits[platforms[right]],
                        )
                        pair_key = f"pair:{left}:{right}"
                        residual_gamma = effective_pair_decay(
                            eta_left,
                            step_decay_gamma[left].get(pair_key, self.gamma),
                            eta_right,
                            step_decay_gamma[right].get(pair_key, self.gamma),
                        )
                        implemented, reference = wang_pair_residuals(
                            prior_position[left],
                            prior_position[right],
                            prior_velocity[left],
                            prior_velocity[right],
                            commands[left],
                            commands[right],
                            learned[left],
                            learned[right],
                            true[left],
                            true[right],
                            self.acceleration_limits[platforms[left]],
                            self.acceleration_limits[platforms[right]],
                            2.0 * self.agent_radius,
                            residual_gamma,
                            motion_noise_i=motion_noise[left],
                            motion_noise_j=motion_noise[right],
                        )
                        tightening = pair_robust_tightening(
                            prior_position[left],
                            prior_position[right],
                            robot_margins[left],
                            robot_margins[right],
                        )
                    except ValueError as exc:
                        record_audit_exception(
                            subteam_id,
                            f"pair_residual:{left}:{right}",
                            exc,
                            steps_executed,
                        )
                        implemented = reference = float("nan")
                        tightening = float("inf")
                    if np.isfinite(tightening):
                        maximum_pair_tightening[subteam_id] = max(
                            maximum_pair_tightening[subteam_id], tightening
                        )
                    register_residual(
                        subteam_id,
                        (left, right),
                        implemented,
                        reference,
                        tightening,
                        "pair",
                    )

                # Check for collisions using the simulated state after the action.
                for robot_id in plan.home_positions:
                    vehicle = states[robot_id]
                    for wall in mission_walls[subteam_id][vehicle.platform]:
                        clearance = wall_clearance(
                            vehicle.position, self.agent_radius, wall
                        )
                        minimum_wall_clearance[subteam_id] = min(
                            minimum_wall_clearance[subteam_id], clearance
                        )
                        robot_min_wall_clearance[robot_id] = min(
                            robot_min_wall_clearance[robot_id], clearance
                        )
                        robot_min_physical_clearance[robot_id] = min(
                            robot_min_physical_clearance[robot_id], clearance
                        )
                        if clearance <= 0.0:
                            mark_robot_failed(subteam_id, robot_id, "wall_collision")
                            record_audit_collision(subteam_id, robot_id, "wall")
                            step_collision[robot_id] = True
                            collisions[subteam_id] += 1
                            wall_collisions[subteam_id] += 1
                    for obstacle_index, obstacle in enumerate(
                        mission_obstacles[subteam_id]
                    ):
                        if not _obstacle_applies(obstacle, vehicle.platform):
                            continue
                        obstacle_id = str(
                            getattr(obstacle, "obstacle_id", None)
                            or f"pillar:{obstacle_index}"
                        )
                        obstacle_clearance = float(
                            np.linalg.norm(vehicle.position - obstacle.center)
                            - obstacle.radius
                            - self.agent_radius
                        )
                        robot_min_physical_clearance[robot_id] = min(
                            robot_min_physical_clearance[robot_id], obstacle_clearance
                        )
                        event_key = (
                            subteam_id,
                            robot_id,
                            obstacle_id,
                            int(steps_executed),
                        )
                        if event_key not in obstacle_collision_event_index and obstacle_clearance <= 0.0:
                            acceleration = (
                                commands[robot_id]
                                + true[robot_id]
                                + motion_noise[robot_id]
                            )
                            interval_min, interval_tau = quadratic_segment_circle_clearance(
                                prior_position[robot_id],
                                prior_velocity[robot_id],
                                acceleration,
                                obstacle.center,
                                obstacle.radius + self.agent_radius,
                                self.dt,
                            )
                            try:
                                first_violation_tau = quadratic_segment_circle_first_violation(
                                    prior_position[robot_id],
                                    prior_velocity[robot_id],
                                    acceleration,
                                    obstacle.center,
                                    obstacle.radius + self.agent_radius,
                                    self.dt,
                                )
                            except ValueError:
                                first_violation_tau = None
                            event = {
                                "step": int(steps_executed),
                                "step_index": int(step_index),
                                "time_s": float(steps_executed * self.dt),
                                "robot_id": robot_id,
                                "obstacle_index": int(obstacle_index),
                                "obstacle_id": obstacle_id,
                                "source": "sampled_endpoint",
                                "interval_tau": float(interval_tau),
                                "interval_minimum_clearance": float(interval_min),
                                "minimum_clearance_tau": float(interval_tau),
                                "minimum_clearance_time_s": float(
                                    (step_index + interval_tau) * self.dt
                                ),
                                "first_violation_tau": (
                                    None
                                    if first_violation_tau is None
                                    else float(first_violation_tau)
                                ),
                                "first_violation_time_s": (
                                    None
                                    if first_violation_tau is None
                                    else float(
                                        (step_index + first_violation_tau) * self.dt
                                    )
                                ),
                                "pre_clearance": float(
                                    np.linalg.norm(
                                        prior_position[robot_id] - obstacle.center
                                    )
                                    - obstacle.radius
                                    - self.agent_radius
                                ),
                                "sample_clearance": obstacle_clearance,
                                "clearance": obstacle_clearance,
                                "epsilon": (
                                    true[robot_id]
                                    + motion_noise[robot_id]
                                    - learned[robot_id]
                                ).tolist(),
                                "epsilon_norm": float(
                                    np.linalg.norm(
                                        true[robot_id]
                                        + motion_noise[robot_id]
                                        - learned[robot_id]
                                    )
                                ),
                                "score_error_norm": _audit_float(
                                    np.linalg.norm(
                                        true[robot_id]
                                        + motion_noise[robot_id]
                                        - learned[robot_id]
                                    )
                                ),
                                "margin": float(robot_margins[robot_id]),
                                "qp_feasible": bool(diagnostic_qp[robot_id]),
                                "fallback_used": bool(diagnostic_fallback[robot_id]),
                                "controller_exception": bool(
                                    diagnostic_exception[robot_id]
                                ),
                            }
                            obstacle_collision_records[subteam_id].append(event)
                            obstacle_collision_event_index[event_key] = event
                            if event_key in obstacle_step_diagnostics:
                                event.update(obstacle_step_diagnostics[event_key])
                            mark_robot_failed(
                                subteam_id, robot_id, "obstacle_collision"
                            )
                            record_audit_collision(subteam_id, robot_id, "obstacle")
                            step_collision[robot_id] = True
                            collisions[subteam_id] += 1
                            obstacle_collisions[subteam_id] += 1
                ids = tuple(plan.home_positions)
                for index, robot_id in enumerate(ids):
                    for other_id in ids[index + 1:]:
                        if platforms[robot_id] is not platforms[other_id]:
                            continue
                        separation = float(np.linalg.norm(states[robot_id].position - states[other_id].position))
                        minimum_separation[subteam_id] = min(minimum_separation[subteam_id], separation)
                        robot_min_separation[robot_id] = min(
                            robot_min_separation[robot_id], separation
                        )
                        robot_min_separation[other_id] = min(
                            robot_min_separation[other_id], separation
                        )
                        pair_clearance = separation - 2.0 * self.agent_radius
                        robot_min_physical_clearance[robot_id] = min(
                            robot_min_physical_clearance[robot_id], pair_clearance
                        )
                        robot_min_physical_clearance[other_id] = min(
                            robot_min_physical_clearance[other_id], pair_clearance
                        )
                        if separation <= 2.0 * self.agent_radius:
                            mark_robot_failed(
                                subteam_id,
                                robot_id,
                                f"agent_collision:{other_id}",
                            )
                            mark_robot_failed(
                                subteam_id,
                                other_id,
                                f"agent_collision:{robot_id}",
                            )
                            record_audit_collision(subteam_id, robot_id, "pair")
                            record_audit_collision(
                                subteam_id, other_id, "pair", mission_event=False
                            )
                            pair_collision_records[subteam_id].append(
                                {
                                    "step": steps_executed,
                                    "time_s": float(steps_executed * self.dt),
                                    "kind": "pair",
                                    "robot_ids": [robot_id, other_id],
                                    "endpoints": [
                                        {
                                            "robot_id": robot_id,
                                            "platform": platforms[robot_id].value,
                                            "score_status": current_score_status(robot_id),
                                        },
                                        {
                                            "robot_id": other_id,
                                            "platform": platforms[other_id].value,
                                            "score_status": current_score_status(other_id),
                                        },
                                    ],
                                    "separation": separation,
                                    "clearance": pair_clearance,
                                }
                            )
                            step_collision[robot_id] = True
                            step_collision[other_id] = True
                            collisions[subteam_id] += 1
                            pair_collisions[subteam_id] += 1

            if trace_recorder is not None:
                trace_recorder.record_task_graph(
                    steps_executed * self.dt,
                    fixed_graphs,
                    invalid_delivery_count=0,
                )
                assert record_frame is not None
                trace_recorder.metadata["control_step_decay_coefficients"].append(step_decay_gamma)
                record_frame(
                    steps_executed * self.dt,
                    step_mismatch,
                    step_collision,
                    step_failure,
                    step_residual,
                    {
                        "control": np.asarray([commands[rid] for rid in robot_ids]),
                        "nominal_control": np.asarray([nominal[rid] for rid in robot_ids]),
                        "consensus": np.asarray(
                            [z_control[plan_by_robot[rid].subteam_id][rid] for rid in robot_ids]
                        ),
                        "formation_term": np.asarray(
                            [
                                self.formation_position_gains[platforms[rid]]
                                * (
                                    z_control[plan_by_robot[rid].subteam_id][rid]
                                    + np.asarray(states[rid].formation_offset, dtype=float)
                                    - prior_position[rid]
                                )
                                for rid in robot_ids
                            ]
                        ),
                        "goal_term": np.asarray(
                            [
                                self.formation_goal_gains[platforms[rid]]
                                * (
                                    current_references[plan_by_robot[rid].subteam_id][0]
                                    - z_control[plan_by_robot[rid].subteam_id][rid]
                                )
                                for rid in robot_ids
                            ]
                        ),
                        "learned_field": np.asarray([learned[rid] for rid in robot_ids]),
                        "true_field": np.asarray([true[rid] for rid in robot_ids]),
                        "learned_neighbor_field": np.asarray([learned_neighbor[rid] for rid in robot_ids]),
                        "true_neighbor_field": np.asarray([true_neighbor[rid] for rid in robot_ids]),
                        "process_noise": np.asarray([motion_noise[rid] for rid in robot_ids]),
                        "qp_feasible": np.asarray([diagnostic_qp[rid] for rid in robot_ids]),
                        "fallback_used": np.asarray([diagnostic_fallback[rid] for rid in robot_ids]),
                        "controller_exception": np.asarray([diagnostic_exception[rid] for rid in robot_ids]),
                        "acceleration_bound_slack": np.asarray([diagnostic_slack[rid] for rid in robot_ids]),
                        "scored_active": np.asarray([scored_active[rid] for rid in robot_ids]),
                    },
                )
        termination_reason = "horizon_exhausted"

        terminal_references = {
            plan.subteam_id: mission_reference(
                plan.subteam_id, steps_executed * self.dt
            )
            for plan in plans
        }
        terminal_centers = {
            plan.subteam_id: np.mean(
                [
                    states[rid].position
                    - np.asarray(states[rid].formation_offset, dtype=float)
                    for rid in plan.home_positions
                ],
                axis=0,
            )
            for plan in plans
        }
        append_trajectory_frame(
            time_s=steps_executed * self.dt,
            center_references=terminal_references,
            center_values=terminal_centers,
            z_values=consensus_state,
        )
        if self.use_hocbf:
            # The terminal state has no solve or following ZOH interval.
            # Pad the terminal entry with NaN to align these arrays with the positions.
            for robot_id in robot_ids:
                for name in hocbf_fields:
                    last = trajectory[robot_id][name][-1]
                    trajectory[robot_id][name].append(
                        [float("nan")] * len(last) if isinstance(last, list) else float("nan")
                    )

        # Retain the scored endpoint before resetting robots for calibration communication.
        update_positions = {
            robot_id: state.home.copy() for robot_id, state in states.items()
        }
        update_graph = build_distance_graph(
            update_positions,
            self.config.update_communication_range,
            require_connected=False,
        )
        update_connectivity = {
            "policy": "home_reset_physical_range",
            "positions_source": "home_positions_after_mission_reset",
            "reset_before_cp_update": True,
            "positions": {rid: value.tolist() for rid, value in update_positions.items()},
            "physical_range_m": self.config.update_communication_range,
            "physical_graph_connected": update_graph.is_connected,
            "effective_graph_connected": update_graph.is_connected,
            "added_links": [],
            "long_range_links_assumed_available": False,
        }
        if trace_recorder is not None:
            trace_recorder.set_update_graph(update_graph)
            trace_recorder.metadata["update_connectivity"] = update_connectivity
            trace_recorder.metadata["termination_reason"] = termination_reason
            trace_recorder.metadata["executed_steps"] = int(steps_executed)
            trace_recorder.metadata["executed_time_s"] = float(steps_executed * self.dt)
            trace_recorder.metadata["maximum_realized_motion_noise_norm"] = {
                str(plan.subteam_id): maximum_motion_noise_norm[plan.subteam_id] for plan in plans
            }
            trace_recorder.metadata["home_snap_occurred"] = {
                str(plan.subteam_id): False for plan in plans
            }
            trace_recorder.metadata["speed_domain_violations"] = {
                str(plan.subteam_id): speed_violations[plan.subteam_id] for plan in plans
            }

        requested_horizon_steps = int(self.config.profile.simulation_steps)
        requested_horizon_seconds = float(requested_horizon_steps * self.dt)
        executed_horizon_seconds = float(steps_executed * self.dt)

        def final_score_complete(robot_id: str) -> bool:
            """Whether the observed raw score spans its intended route interval."""

            if raw_score_samples[robot_id] <= 0 or score_invalid[robot_id]:
                return False
            # Calibration scores cover the fixed physical interval regardless
            # of arrival or distance from home.
            return (raw_score_samples[robot_id] == steps_executed
                    and steps_executed == requested_horizon_steps)

        def final_score_status(robot_id: str) -> str:
            if raw_score_samples[robot_id] <= 0:
                return "unavailable"
            if not final_score_complete(robot_id):
                return "unavailable"
            raw = raw_score_max[robot_id]
            if raw is None or raw > float(robot_margins[robot_id]):
                return "uncovered"
            return "covered"

        def classify_obstacle_event(
            event: dict[str, object],
            robot_id: str,
        ) -> list[str]:
            """Classify one collision event using final score/audit evidence."""

            labels: set[str] = set()
            if not final_score_complete(robot_id):
                labels.add("score_incomplete")
            initial_h = mission_initial_obstacle_h.get(
                (robot_id, str(event.get("obstacle_id", ""))), float("nan")
            )
            event["mission_initial_h"] = _audit_float(initial_h)
            event["step_initial_h"] = event.get("initial_h")
            if not np.isfinite(initial_h) or initial_h < 0.0:
                labels.add("invalid_mission_initial_h")
            if bool(event.get("invalid_initial_h", False)):
                labels.add("invalid_step_initial_h")
            source = str(event.get("source", ""))
            try:
                pre_clearance = float(event.get("pre_clearance", float("nan")))
                sample_clearance = float(event.get("sample_clearance", float("nan")))
                interval_clearance = float(
                    event.get("interval_minimum_clearance", float("nan"))
                )
            except (TypeError, ValueError):
                pre_clearance = sample_clearance = interval_clearance = float("nan")
            if source == "quadratic_interval":
                if (
                    np.isfinite(pre_clearance)
                    and np.isfinite(sample_clearance)
                    and np.isfinite(interval_clearance)
                    and pre_clearance > 0.0
                    and sample_clearance > 0.0
                    and interval_clearance <= 0.0
                ):
                    labels.add("intersample")
                elif np.isfinite(pre_clearance) and pre_clearance <= 0.0:
                    labels.add("initial_penetration")
                elif np.isfinite(sample_clearance) and sample_clearance <= 0.0:
                    labels.add("sampled_endpoint")
                else:
                    labels.add("unresolved")
            elif source == "sampled_endpoint":
                labels.add("sampled_endpoint")
            implemented = event.get("implemented_residual")
            required = event.get("required_tightening")
            reference = event.get("true_residual")
            try:
                implemented_value = float(implemented)
                required_value = float(required)
                reference_value = float(reference)
            except (TypeError, ValueError):
                implemented_value = required_value = reference_value = float("nan")
            if not np.isfinite(implemented_value) or not np.isfinite(reference_value):
                labels.add("unresolved")
            elif np.isfinite(required_value):
                if implemented_value < required_value - 1e-6:
                    labels.add("tightened_inequality_unenforced")
                if abs(implemented_value - reference_value) > required_value + 1e-6:
                    labels.add("robust_inconsistency")
            if not labels:
                labels.add("collision_confirmed")
            return sorted(labels)

        def compact_obstacle_windows(
            records: Sequence[dict[str, object]],
        ) -> list[dict[str, object]]:
            """Collapse contiguous hits while preserving the worst audit row."""

            grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
            for event in records:
                robot_id = str(event.get("robot_id", ""))
                obstacle_id = str(event.get("obstacle_id", ""))
                event["classification"] = classify_obstacle_event(event, robot_id)
                grouped.setdefault((robot_id, obstacle_id), []).append(event)
            windows: list[dict[str, object]] = []
            for (robot_id, obstacle_id), events in grouped.items():
                ordered = sorted(events, key=lambda item: int(item.get("step", 0)))
                episode: list[dict[str, object]] = []

                def flush() -> None:
                    if not episode:
                        return
                    worst = min(
                        episode,
                        key=lambda item: float(
                            item.get("interval_minimum_clearance", float("inf"))
                        ),
                    )
                    labels = sorted(
                        {
                            label
                            for item in episode
                            for label in item.get("classification", [])
                        }
                    )
                    first_violation_times = [
                        float(item["first_violation_time_s"])
                        for item in episode
                        if item.get("first_violation_time_s") is not None
                    ]
                    minimum_sample = [
                        float(item["sample_clearance"])
                        for item in episode
                        if item.get("sample_clearance") is not None
                    ]
                    window: dict[str, object] = {
                        "robot_id": robot_id,
                        "obstacle_id": obstacle_id,
                        "first_step": int(episode[0].get("step", 0)),
                        "last_step": int(episode[-1].get("step", 0)),
                        "first_time_s": _audit_float(episode[0].get("time_s")),
                        "last_time_s": _audit_float(episode[-1].get("time_s")),
                        "event_count": len(episode),
                        "score_status": final_score_status(robot_id),
                        "score_complete": final_score_complete(robot_id),
                        "source": sorted({str(item.get("source", "")) for item in episode}),
                        "classification": labels,
                        "minimum_interval_clearance": _audit_float(
                            worst.get("interval_minimum_clearance")
                        ),
                        "minimum_sample_clearance": (
                            min(minimum_sample) if minimum_sample else None
                        ),
                        "first_violation_time_s": (
                            min(first_violation_times) if first_violation_times else None
                        ),
                        "minimum_clearance_time_s": _audit_float(
                            worst.get("minimum_clearance_time_s")
                        ),
                        "representative_step": int(worst.get("step", 0)),
                        "first_event": dict(episode[0]),
                    }
                    # Promote the representative residual/audit fields so a
                    # window is directly useful without reopening raw traces.
                    for field_name in (
                        "h", "psi", "omega", "omega_is_optimized", "robust_hocbf_slack", "qp_status", "d_hat",
                        "implemented_residual",
                        "true_residual",
                        "required_tightening",
                        "slack",
                        "initial_h",
                        "mission_initial_h",
                        "step_initial_h",
                        "invalid_initial_h",
                        "qp_feasible",
                        "fallback_used",
                        "controller_exception",
                        "epsilon",
                        "epsilon_norm",
                        "score_error_norm",
                        "margin",
                        "pre_clearance",
                        "sample_clearance",
                        "interval_minimum_clearance",
                        "minimum_clearance_tau",
                        "first_violation_tau",
                    ):
                        if field_name in worst:
                            window[field_name] = worst[field_name]
                    windows.append(window)

                previous_step: int | None = None
                for event in ordered:
                    step = int(event.get("step", 0))
                    if previous_step is not None and step > previous_step + 1:
                        flush()
                        episode = []
                    episode.append(event)
                    previous_step = step
                flush()
            return sorted(
                windows,
                key=lambda item: (
                    str(item.get("robot_id", "")),
                    str(item.get("obstacle_id", "")),
                    int(item.get("first_step", 0)),
                ),
            )

        def route_progress(state: WangVehicleState) -> tuple[int, float]:
            plan = plan_by_robot[state.robot_id]
            frame = realizations[plan.subteam_id].frame
            start, _, _ = _formation_reference(self.config, 0.0, frame)
            goal, _, _ = _formation_reference(
                self.config, self.config.formation_center_duration_s, frame
            )
            center_observation = state.position - np.asarray(state.formation_offset, dtype=float)
            direction = goal - start
            denominator = float(direction @ direction)
            progress = 0.0 if denominator <= 0.0 else float(
                np.clip((center_observation - start) @ direction / denominator, 0.0, 1.0)
            )
            completed_segments = int(waypoint_arrival_times[state.robot_id] is not None)
            return completed_segments, progress

        def compact_pair_records(
            records: Sequence[dict[str, object]],
        ) -> list[dict[str, object]]:
            """Collapse repeated timestep hits into one endpoint record per pair."""

            grouped: dict[tuple[str, str], dict[str, object]] = {}
            for event in records:
                pair = tuple(str(robot_id) for robot_id in event["robot_ids"])
                if len(pair) != 2:
                    continue
                key = tuple(sorted(pair))
                if key not in grouped:
                    endpoints = [
                        dict(endpoint) for endpoint in event["endpoints"]  # type: ignore[index]
                    ]
                    grouped[key] = {
                        "kind": "pair",
                        "robot_ids": list(key),
                        "endpoints": endpoints,
                        "event_count": 1,
                        "first_step": int(event["step"]),
                        "last_step": int(event["step"]),
                        "first_time_s": _audit_float(event["time_s"]),
                        "last_time_s": _audit_float(event["time_s"]),
                        "minimum_separation": _audit_float(event["separation"]),
                        "minimum_clearance": _audit_float(event["clearance"]),
                    }
                    continue
                compact = grouped[key]
                compact["event_count"] = int(compact["event_count"]) + 1
                compact["last_step"] = int(event["step"])
                compact["last_time_s"] = _audit_float(event["time_s"])
                separation = _audit_float(event["separation"])
                clearance = _audit_float(event["clearance"])
                if separation is not None:
                    prior_separation = compact["minimum_separation"]
                    compact["minimum_separation"] = (
                        separation
                        if prior_separation is None
                        else min(float(prior_separation), separation)
                    )
                if clearance is not None:
                    prior_clearance = compact["minimum_clearance"]
                    compact["minimum_clearance"] = (
                        clearance
                        if prior_clearance is None
                        else min(float(prior_clearance), clearance)
                    )
            compacted = []
            for compact in grouped.values():
                endpoints = []
                for endpoint in compact["endpoints"]:  # type: ignore[index]
                    endpoint = dict(endpoint)
                    observed = endpoint.get("score_status", "unavailable")
                    robot_id = str(endpoint["robot_id"])
                    endpoint["score_status_at_event"] = observed
                    endpoint["score_coverage_at_event"] = observed
                    endpoint["score_status"] = final_score_status(robot_id)
                    endpoint["score_coverage"] = final_score_status(robot_id)
                    endpoint["score_complete"] = final_score_complete(robot_id)
                    endpoints.append(endpoint)
                compact["endpoints"] = endpoints
                compacted.append(compact)
            return compacted

        results: list[MissionResult] = []
        for plan in plans:
            subteam_id = plan.subteam_id
            returned = False
            task_completed = all(destination_reached[rid] for rid in plan.home_positions)
            if not fixed_graphs[subteam_id].is_connected:
                global_failures[subteam_id].append(
                    "fixed_wang_graph_disconnected"
                )
            failed_by_platform = {
                platform: tuple(
                    robot_id
                    for robot_id in plan.robot_ids[platform]
                    if robot_id in failed_robot_ids[subteam_id]
                )
                for platform in Platform
            }
            failure_budget_exceeded = {
                platform: (
                    len(failed_by_platform[platform])
                    > self.config.classes[platform].violations
                )
                for platform in Platform
            }
            local_reasons = {
                robot_id: tuple(sorted(reasons))
                for robot_id, reasons in robot_failure_reasons[subteam_id].items()
                if reasons
            }
            mission_completion_candidates = [
                waypoint_arrival_times[robot_id]
                for robot_id in plan.home_positions
                if waypoint_arrival_times[robot_id] is not None
            ]
            mission_completion_time_s = (
                _audit_float(max(mission_completion_candidates))
                if task_completed
                and len(mission_completion_candidates) == len(plan.home_positions)
                else None
            )
            robot_records: dict[str, dict[str, object]] = {}
            route_progress_values: list[float] = []
            score_status_counts = {
                "covered": 0,
                "uncovered": 0,
                "unavailable": 0,
            }
            class_score_coverage: dict[str, dict[str, object]] = {}
            for platform in Platform:
                class_robot_ids = tuple(plan.robot_ids[platform])
                class_records: list[dict[str, object]] = []
                for robot_id in class_robot_ids:
                    state = states[robot_id]
                    completed_segments, progress = route_progress(state)
                    route_progress_values.append(progress)
                    status = final_score_status(robot_id)
                    score_status_counts[status] += 1
                    class_records.append(
                        {
                            "robot_id": robot_id,
                            "status": status,
                            "margin": float(robot_margins[robot_id]),
                        }
                    )
                    score = scores[subteam_id][platform][robot_id]
                    robot_records[robot_id] = {
                        "robot_id": robot_id,
                        "platform": platform.value,
                        "margin": float(robot_margins[robot_id]),
                        "production_score": _audit_float(score),
                        "raw_score": _audit_float(raw_score_max[robot_id]),
                        "raw_score_max": _audit_float(raw_score_max[robot_id]),
                        "scored_samples": int(raw_score_samples[robot_id]),
                        "scored_attempts": int(score_attempts[robot_id]),
                        "score_valid": bool(
                            raw_score_samples[robot_id] > 0 and not score_invalid[robot_id]
                        ),
                        "score_complete": final_score_complete(robot_id),
                        "scored_interval_complete": final_score_complete(robot_id),
                        "score_status": status,
                        "score_coverage": status,
                        "application_failed": robot_id in failed_robot_ids[subteam_id],
                        "failure_reasons": sorted(
                            robot_failure_reasons[subteam_id][robot_id]
                        ),
                        "physical_collision": any(
                            robot_collision_flags[robot_id].values()
                        ),
                        "collision_flags": dict(robot_collision_flags[robot_id]),
                        "collision_participation": dict(
                            robot_collision_flags[robot_id]
                        ),
                        "collision_event_counts": dict(
                            robot_collision_events[robot_id]
                        ),
                        "qp_failure_count": int(robot_qp_failures[robot_id]),
                        "qp_solve_count": int(robot_qp_solves[robot_id]),
                        "qp_fallback_count": int(robot_qp_fallbacks[robot_id]),
                        "qp_feasible_solve_count": int(
                            robot_qp_feasible_solves[robot_id]
                        ),
                        "control_effort": _audit_float(state.control_effort),
                        "waypoint_index": int(state.waypoint_index),
                        "waypoints_completed": int(completed_segments),
                        "route_progress": progress,
                        "completed": bool(destination_reached[robot_id]),
                        "destination_reached": bool(destination_reached[robot_id]),
                        "returned_home": False,
                        "return_required": False,
                        "waypoint_arrival_time_s": _audit_float(waypoint_arrival_times[robot_id]),
                        "home_position_error_m": float(np.linalg.norm(state.position - state.home)),
                        "completion_time_s": _audit_float(
                            waypoint_arrival_times[robot_id]
                        ),
                        "elapsed_time_s": executed_horizon_seconds,
                        "minimum_physical_clearance": _audit_float(
                            robot_min_physical_clearance[robot_id]
                        ),
                        "minimum_wall_clearance": _audit_float(
                            robot_min_wall_clearance[robot_id]
                        ),
                        "minimum_same_layer_separation": _audit_float(
                            robot_min_separation[robot_id]
                        ),
                        "maximum_acceleration_error": _audit_float(
                            robot_max_acceleration_error[robot_id]
                        ),
                        "cbf_diagnostics": {
                            "minimum_implemented_wang_residual": _audit_float(
                                robot_min_implemented_residual[robot_id]
                            ),
                            "minimum_true_wang_residual": _audit_float(
                                robot_min_true_residual[robot_id]
                            ),
                            "maximum_pair_robust_tightening": _audit_float(
                                robot_max_pair_tightening[robot_id]
                            ),
                            "maximum_obstacle_robust_tightening": _audit_float(
                                robot_max_obstacle_tightening[robot_id]
                            ),
                            "maximum_wall_robust_tightening": _audit_float(
                                robot_max_wall_tightening[robot_id]
                            ),
                            "minimum_active_set_singular_value": _audit_float(
                                robot_min_active_sigma[robot_id]
                            ),
                            "maximum_active_set_condition_number": _audit_float(
                                robot_max_condition[robot_id]
                            ),
                            "singular_active_set_qp_count": int(
                                robot_singular_qps[robot_id]
                            ),
                            "ill_conditioned_qp_count": int(
                                robot_ill_conditioned_qps[robot_id]
                            ),
                            "minimum_acceleration_limit_slack": _audit_float(
                                robot_min_acceleration_slack[robot_id]
                            ),
                        },
                    }
                covered = sum(item["status"] == "covered" for item in class_records)
                uncovered = sum(
                    item["status"] == "uncovered" for item in class_records
                )
                unavailable = sum(
                    item["status"] == "unavailable" for item in class_records
                )
                configured_k = max(
                    0,
                    len(class_robot_ids) - int(self.config.classes[platform].violations),
                )
                all_robots_covered = (
                    True
                    if class_robot_ids and covered == len(class_robot_ids)
                    else False
                    if uncovered > 0
                    else None
                )
                configured_k_covered = (
                    True
                    if covered >= configured_k
                    else False
                    if uncovered > len(class_robot_ids) - configured_k
                    else None
                )
                class_score_coverage[platform.value] = {
                    "covered_robot_count": int(covered),
                    "uncovered_robot_count": int(uncovered),
                    "unavailable_robot_count": int(unavailable),
                    "robot_count": len(class_robot_ids),
                    "coverage_fraction": (
                        float(covered / len(class_robot_ids))
                        if class_robot_ids
                        else None
                    ),
                    "all_robots_covered": all_robots_covered,
                    "configured_K": configured_k,
                    "configured_K_covered": configured_k_covered,
                }

            collision_event_counts = dict(audit_collision_events[subteam_id])
            collision_participating_counts = {
                kind: sum(
                    bool(robot_collision_flags[robot_id][kind])
                    for robot_id in plan.home_positions
                )
                for kind in ("obstacle", "wall", "pair")
            }
            collision_participating_by_class = {
                platform.value: {
                    kind: sum(
                        bool(robot_collision_flags[robot_id][kind])
                        for robot_id in plan.robot_ids[platform]
                    )
                    for kind in ("obstacle", "wall", "pair")
                }
                for platform in Platform
            }
            collision_events_by_class = {}
            for platform in Platform:
                event_counts = {
                    kind: sum(
                        robot_collision_events[robot_id][kind]
                        for robot_id in plan.robot_ids[platform]
                    )
                    for kind in ("obstacle", "wall", "pair")
                }
                # A contact between two robots in the same layer has two endpoint observations but is
                # one physical event for the mission count for that class.
                event_counts["pair"] //= 2
                collision_events_by_class[platform.value] = event_counts
            finite_clearances = [
                value
                for robot_id in plan.home_positions
                for value in (robot_min_physical_clearance[robot_id],)
                if math.isfinite(value)
            ]
            finite_separations = [
                value
                for robot_id in plan.home_positions
                for value in (robot_min_separation[robot_id],)
                if math.isfinite(value)
            ]
            finite_errors = [
                float(robot_max_acceleration_error[robot_id])
                for robot_id in plan.home_positions
                if robot_max_acceleration_error[robot_id] is not None
            ]
            final_center = terminal_centers[subteam_id]
            final_center_reference = terminal_references[subteam_id][0]
            final_formation_errors = [
                float(
                    np.linalg.norm(
                        states[robot_id].position
                        - (final_center + np.asarray(states[robot_id].formation_offset, dtype=float))
                    )
                )
                for robot_id in plan.home_positions
            ]
            final_center_reference_error = float(
                np.linalg.norm(final_center - final_center_reference)
            )
            final_consensus_errors = [
                float(
                    np.linalg.norm(
                        consensus_state[subteam_id][robot_id] - final_center
                    )
                )
                for robot_id in plan.home_positions
            ]
            obstacle_windows = compact_obstacle_windows(
                obstacle_collision_records[subteam_id]
            )
            fixed_graph = fixed_graphs[subteam_id]
            fixed_graph_edges = [list(edge) for edge in sorted(fixed_graph.edges)]
            mission_fixed_metrics: dict[str, object] = {
                "formation_phase_rad": realizations[subteam_id].formation_phase_rad,
                "formation_offsets_canonical": {
                    robot_id: value.tolist()
                    for robot_id, value in realizations[subteam_id]
                    .formation_offsets_canonical.items()
                },
                "application_failed": bool(global_failures[subteam_id]),
                "application_failure_reasons": sorted(
                    set(global_failures[subteam_id])
                ),
                "communication_failure": False,
                "protocol_failure": False,
                "fixed_communication_graph_connected": bool(fixed_graph.is_connected),
                "fixed_communication_graph_edges": fixed_graph_edges,
                "center_consensus_graph": "alternating_ugv_uav_ring",
                "center_consensus_graph_edges": fixed_graph_edges,
                "returned_home": bool(returned),
                "return_required": False,
                "return_leg_enabled": False,
                "return_start_time_s": None,
                "return_reason": None,
                "outbound_completed": bool(task_completed),
                "destination_reached": bool(task_completed),
                "return_origin_canonical": None,
                "mission_phase": "destination" if task_completed else "outbound",
                "terminal_update_graph_connected": bool(update_graph.is_connected),
                "update_connectivity": update_connectivity,
                "task_completed": bool(task_completed),
                "formation_completed": bool(task_completed),
                "physical_collision": bool(any(collision_event_counts.values())),
                "collision_free": not bool(any(collision_event_counts.values())),
                "collision_event_counts": collision_event_counts,
                "same_layer_contact_count": collision_event_counts["pair"],
                "collision_participating_robot_counts": collision_participating_counts,
                "same_layer_contact_participating_robot_count": collision_participating_counts[
                    "pair"
                ],
                "collision_participating_robot_counts_by_class": collision_participating_by_class,
                "collision_event_counts_by_class": collision_events_by_class,
                "collision_free_by_class": {
                    platform.value: not any(
                        collision_participating_by_class[platform.value].values()
                    )
                    for platform in Platform
                },
                "robot_application_failure_count": len(failed_robot_ids[subteam_id]),
                "local_qp_failure_count": int(
                    sum(robot_qp_failures[robot_id] for robot_id in plan.home_positions)
                ),
                "production_qp_failure_count": int(qp_failures[subteam_id]),
                "local_qp_failure_mission": bool(
                    any(
                        robot_qp_failures[robot_id] > 0
                        for robot_id in plan.home_positions
                    )
                ),
                "control_effort": _audit_float(
                    sum(states[robot_id].control_effort for robot_id in plan.home_positions)
                ),
                "route_progress_min": min(route_progress_values)
                if route_progress_values
                else None,
                "route_progress_mean": (
                    float(sum(route_progress_values) / len(route_progress_values))
                    if route_progress_values
                    else None
                ),
                "route_progress_max": max(route_progress_values)
                if route_progress_values
                else None,
                "formation_restoration_error_max": max(final_formation_errors)
                if final_formation_errors
                else None,
                "formation_restoration_error_mean": (
                    float(sum(final_formation_errors) / len(final_formation_errors))
                    if final_formation_errors
                    else None
                ),
                "center_reference_tracking_error": final_center_reference_error,
                "consensus_error_max": max(final_consensus_errors)
                if final_consensus_errors
                else None,
                "maximum_acceleration_error": _audit_float(
                    max(finite_errors) if finite_errors else None
                ),
                "minimum_physical_clearance": _audit_float(
                    min(finite_clearances) if finite_clearances else None
                ),
                "minimum_same_layer_separation": _audit_float(
                    min(finite_separations) if finite_separations else None
                ),
                "minimum_wall_clearance": _audit_float(
                    minimum_wall_clearance[subteam_id]
                ),
                "minimum_implemented_wang_residual": _audit_float(
                    minimum_implemented[subteam_id]
                ),
                "minimum_true_wang_residual": _audit_float(minimum_true[subteam_id]),
                "maximum_pair_robust_tightening": _audit_float(
                    maximum_pair_tightening[subteam_id]
                ),
                "maximum_obstacle_robust_tightening": _audit_float(
                    maximum_obstacle_tightening[subteam_id]
                ),
                "maximum_wall_robust_tightening": _audit_float(
                    maximum_wall_tightening[subteam_id]
                ),
                "score_status_counts": score_status_counts,
                "score_coverage_by_class": class_score_coverage,
                "all_robots_score_covered": (
                    True
                    if score_status_counts["covered"] == len(plan.home_positions)
                    else False
                    if score_status_counts["uncovered"] > 0
                    else None
                ),
                "all_robots_covered": (
                    True
                    if score_status_counts["covered"] == len(plan.home_positions)
                    else False
                    if score_status_counts["uncovered"] > 0
                    else None
                ),
                "elapsed_time_s": executed_horizon_seconds,
                "termination_reason": termination_reason,
                "completion_time_s": mission_completion_time_s,
                "caught_exception_count": sum(
                    int(item["count"])
                    for item in audit_exception_groups[subteam_id]
                ),
            }
            fixed_margin_audit: dict[str, object] = {
                "schema_version": 1,
                "application": "collision_avoidance",
                "controller": self.controller_name,
                "scenario": self.config.collision_scenario,
                "margin_units": "m/s^2 acceleration",
                "margin_mode": margin_mode,
                "margins": {
                    **(
                        {
                            platform.value: float(value)
                            for platform, value in class_margins.items()
                        }
                        if class_margins is not None
                        else {
                            robot_id: float(robot_margins[robot_id])
                            for robot_id in plan.home_positions
                        }
                    )
                },
                "class_margins": (
                    None
                    if class_margins is None
                    else {
                        platform.value: float(value)
                        for platform, value in class_margins.items()
                    }
                ),
                "robot_margins": {
                    robot_id: float(robot_margins[robot_id])
                    for robot_id in plan.home_positions
                },
                "mission_id": int(plan.mission_id),
                "subteam_id": int(subteam_id),
                "mission_seed": int(realizations[subteam_id].mission_seed),
                "wave_seed": int(wave_seed),
                "obstacle_position_sigma_m": self.config.obstacle_position_sigma_m,
                "mission_dynamics_severity_sigma": self.config.mission_dynamics_severity_sigma,
                "formation_phase_rad": realizations[subteam_id].formation_phase_rad,
                "formation_offsets_canonical": {
                    robot_id: value.tolist()
                    for robot_id, value in realizations[subteam_id]
                    .formation_offsets_canonical.items()
                },
                "dynamics_severity_multiplier": dynamics_severity_multiplier[subteam_id],
                "obstacle_catalog": [
                    {"obstacle_id": str(obstacle.obstacle_id),
                     "center": obstacle.center.tolist(), "radius": obstacle.radius,
                     "platform": getattr(obstacle.platform, "value", None)}
                    for obstacle in mission_obstacles[subteam_id]
                ],
                "requested_horizon": {
                    "steps": requested_horizon_steps,
                    "seconds": requested_horizon_seconds,
                },
                "executed_horizon": {
                    "steps": int(steps_executed),
                    "seconds": executed_horizon_seconds,
                },
                "requested_horizon_steps": requested_horizon_steps,
                "requested_horizon_seconds": requested_horizon_seconds,
                "executed_horizon_steps": int(steps_executed),
                "executed_horizon_seconds": executed_horizon_seconds,
                # Short aliases used by sweep serializers.
                "requested_steps": requested_horizon_steps,
                "executed_steps": int(steps_executed),
                "elapsed_time": executed_horizon_seconds,
                "termination_reason": termination_reason,
                "completed": bool(task_completed),
                "formation_completed": bool(task_completed),
                "communication_failure": False,
                "protocol_failure": False,
                "fixed_communication_graph_connected": bool(fixed_graph.is_connected),
                "fixed_communication_graph_edges": fixed_graph_edges,
                "center_consensus_graph": "alternating_ugv_uav_ring",
                "center_consensus_graph_edges": fixed_graph_edges,
                "returned_home": bool(returned),
                "return_required": False,
                "return_leg_enabled": False,
                "return_start_time_s": mission_fixed_metrics["return_start_time_s"],
                "return_reason": mission_fixed_metrics["return_reason"],
                "outbound_completed": mission_fixed_metrics["outbound_completed"],
                "return_origin_canonical": mission_fixed_metrics["return_origin_canonical"],
                "mission_phase": mission_fixed_metrics["mission_phase"],
                "terminal_update_graph_connected": bool(update_graph.is_connected),
                "update_connectivity": update_connectivity,
                "task_completed": bool(task_completed),
                "completion_time": mission_completion_time_s,
                "completion_time_s": mission_completion_time_s,
                "elapsed_time_s": executed_horizon_seconds,
                "formation_center_variant": self.config.center_variant,
                "center_feedback_variant": self.center_feedback_variant,
                "center_consensus_rounds": self.consensus_rounds,
                "trajectory_time_s": list(trajectory_time_s),
                "trajectory_step": list(range(len(trajectory_time_s))),
                "formation_offsets": {
                    robot_id: np.asarray(states[robot_id].formation_offset, dtype=float).tolist()
                    for robot_id in plan.home_positions
                },
                "trajectory": {
                    robot_id: trajectory[robot_id]
                    for robot_id in plan.home_positions
                },
                "score_argmax_step": {
                    robot_id: score_argmax_step[robot_id]
                    for robot_id in plan.home_positions
                },
                "score_argmax_components": {
                    robot_id: score_argmax_components[robot_id]
                    for robot_id in plan.home_positions
                },
                "mission_initial_obstacle_barriers": {
                    robot_id: {
                        obstacle_id: _audit_float(value)
                        for (rid, obstacle_id), value in mission_initial_obstacle_h.items()
                        if rid == robot_id
                    }
                    for robot_id in plan.home_positions
                },
                "control_effort": mission_fixed_metrics["control_effort"],
                "route_progress": mission_fixed_metrics["route_progress_min"],
                "local_qp_failures": mission_fixed_metrics["local_qp_failure_count"],
                "qp_failure_count": mission_fixed_metrics["local_qp_failure_count"],
                "production_qp_failure_count": mission_fixed_metrics[
                    "production_qp_failure_count"
                ],
                "physical_collision": mission_fixed_metrics["physical_collision"],
                "collision_free": mission_fixed_metrics["collision_free"],
                "collision_events": collision_event_counts["obstacle"]
                + collision_event_counts["wall"]
                + collision_event_counts["pair"],
                "obstacle_collision_events": collision_event_counts["obstacle"],
                "wall_collision_events": collision_event_counts["wall"],
                "pair_collision_events": collision_event_counts["pair"],
                "same_layer_contact_events": collision_event_counts["pair"],
                "exceptions": audit_exception_groups[subteam_id],
                "robots": robot_records,
                "pair_collisions": compact_pair_records(
                    pair_collision_records[subteam_id]
                ),
                "obstacle_collision_windows": obstacle_windows,
                "metrics": mission_fixed_metrics,
                "mission_metrics": mission_fixed_metrics,
            }
            if self.use_hocbf:
                fixed_margin_audit["hocbf"] = {
                    "lambda1": self.hocbf_lambda, "lambda2": self.hocbf_lambda,
                    "omega_min": 0.0, "omega_max": 10.0, "decay_weight": self.decay_weight,
                    "initial_pairs": [pair for pair in hocbf_initial_pairs
                                      if pair["robot_id"] in plan.home_positions],
                    "obstacle_catalog": [
                        {"obstacle_id": str(obstacle.obstacle_id),
                         "center": obstacle.center.tolist(), "radius": obstacle.radius,
                         "platform": getattr(obstacle.platform, "value", None)}
                        for obstacle in mission_obstacles[subteam_id]
                    ],
                    "robot_radius": self.agent_radius,
                    "minimum_successful_qp_slack": _audit_float(hocbf_min_slack[subteam_id]),
                    "successful_constraint_count": hocbf_success_count[subteam_id],
                    "score_reconstruction_max_error": score_reconstruction_error[subteam_id],
                    "controller_prediction_max_error": controller_prediction_error[subteam_id],
                    "status_codes": dict(hocbf_status_codes),
                    "terminal_row": "no solve or ZOH interval; numeric NaN sentinel",
                    "fallback_omega_and_slack": (
                        "hypothetical at omega=1 when no optimized decision exists; "
                        "excluded from successful-QP slack statistics"
                    ),
                }
                fixed_margin_audit["obstacle_encounter"] = {
                    "law": (
                        "bounded_inward_opposite_class_"
                        f"{self.config.formation_obstacle_disturbance_mode}_v1"
                    ),
                    "mode": self.config.formation_obstacle_disturbance_mode,
                    "maximum_acceleration_m_s2": encounter_amplitude,
                    "band_m": self.config.formation_obstacle_disturbance_band_m,
                    "strength": encounter_strength.get(subteam_id),
                    "synthetic_stress_scenario": encounter_amplitude > 0,
                    "robots": {rid: encounter_diagnostics[rid] for rid in plan.home_positions},
                }
            result = MissionResult(
                mission_id=plan.mission_id,
                subteam_id=subteam_id,
                seed=realizations[subteam_id].mission_seed,
                robot_scores=scores[subteam_id],
                mission_graph={node: tuple(sorted(peers)) for node, peers in fixed_graph.items()},
                returned_home=returned,
                update_graph_connected=bool(update_graph.is_connected),
                failed=bool(global_failures[subteam_id]),
                failure_reason=(
                    ";".join(sorted(set(global_failures[subteam_id]))) or None
                ),
                failed_robots=failed_by_platform,
                robot_failure_reasons=local_reasons,
                metrics={
                    "wang_gamma": self.gamma,
                    "wang_optimal_decay_cbf": float(self.optimal_decay),
                    "wang_decay_weight": self.decay_weight,
                    "wang_decay_multiplier_min_config": self.decay_multiplier_min,
                    "wang_decay_multiplier_max_config": self.decay_multiplier_max,
                    "minimum_optimized_decay_multiplier": minimum_decay_multiplier[subteam_id],
                    "maximum_optimized_decay_multiplier": maximum_decay_multiplier[subteam_id],
                    "fraction_decay_multipliers_changed": (
                        changed_decay_multipliers[subteam_id]
                        / total_decay_multipliers[subteam_id]
                        if total_decay_multipliers[subteam_id]
                        else 0.0
                    ),
                    "total_decay_penalty": total_decay_penalty[subteam_id],
                    "wang_acceleration_limit_ugv": self.acceleration_limits[Platform.UGV],
                    "wang_acceleration_limit_uav": self.acceleration_limits[Platform.UAV],
                    "wang_speed_limit_ugv": self.speed_limits[Platform.UGV],
                    "wang_speed_limit_uav": self.speed_limits[Platform.UAV],
                    "wang_position_kp_ugv": self.position_gains[Platform.UGV],
                    "wang_position_kp_uav": self.position_gains[Platform.UAV],
                    "wang_velocity_kd_ugv": self.velocity_gains[Platform.UGV],
                    "wang_velocity_kd_uav": self.velocity_gains[Platform.UAV],
                    "communication_failure": False,
                    "protocol_failure": False,
                    "fixed_communication_graph_connected": bool(
                        fixed_graphs[subteam_id].is_connected
                    ),
                    "fixed_wang_graph_connected": float(fixed_graphs[subteam_id].is_connected),
                    "fixed_wang_graph_edges": len(fixed_graphs[subteam_id].edges),
                    "fixed_pair_safety_edges": len(pair_edges[subteam_id]),
                    "local_qp_failures": qp_failures[subteam_id],
                    "local_qp_solves": qp_solves[subteam_id],
                    "qp_diagnostics_recorded": sum(
                        len(qp_diagnostic_records[robot_id])
                        for robot_id in plan.home_positions
                    ),
                    "minimum_active_set_singular_value_diagnostic": minimum_active_sigma[subteam_id],
                    "maximum_active_set_condition_number_diagnostic": maximum_condition[subteam_id],
                    "singular_active_set_qps_diagnostic": singular_qps[subteam_id],
                    "ill_conditioned_qps_diagnostic": ill_conditioned_qps[subteam_id],
                    "maximum_control_norm": maximum_control_norm[subteam_id],
                    "maximum_control_inf_norm": maximum_control_inf_norm[subteam_id],
                    "minimum_acceleration_limit_slack_diagnostic": maximum_acceleration_slack[subteam_id],
                    "maximum_acceleration_model_error": maximum_acceleration_error[subteam_id],
                    "minimum_implemented_wang_residual": minimum_implemented[subteam_id],
                    "minimum_true_wang_residual": minimum_true[subteam_id],
                    "maximum_pair_robust_tightening": maximum_pair_tightening[subteam_id],
                    "maximum_obstacle_robust_tightening": maximum_obstacle_tightening[subteam_id],
                    "maximum_wall_robust_tightening": maximum_wall_tightening[subteam_id],
                    "formation_restoration_error_max": mission_fixed_metrics[
                        "formation_restoration_error_max"
                    ],
                    "formation_restoration_error_mean": mission_fixed_metrics[
                        "formation_restoration_error_mean"
                    ],
                    "center_reference_tracking_error": mission_fixed_metrics[
                        "center_reference_tracking_error"
                    ],
                    "consensus_error_max": mission_fixed_metrics["consensus_error_max"],
                    "minimum_wall_clearance": minimum_wall_clearance[subteam_id],
                    "minimum_same_layer_separation": minimum_separation[subteam_id],
                    "collision_count": collisions[subteam_id],
                    "wall_collision_count": wall_collisions[subteam_id],
                    "obstacle_collision_count": obstacle_collisions[subteam_id],
                    "pair_collision_count": pair_collisions[subteam_id],
                    "same_layer_contact_count": pair_collisions[subteam_id],
                    "same_layer_contact_participating_robot_count": collision_participating_counts[
                        "pair"
                    ],
                    "ugv_failed_robot_count": len(
                        failed_by_platform[Platform.UGV]
                    ),
                    "uav_failed_robot_count": len(
                        failed_by_platform[Platform.UAV]
                    ),
                    "ugv_failure_budget_exceeded": float(
                        failure_budget_exceeded[Platform.UGV]
                    ),
                    "uav_failure_budget_exceeded": float(
                        failure_budget_exceeded[Platform.UAV]
                    ),
                    "maximum_speed": maximum_speed[subteam_id],
                    "speed_domain_violation_count": speed_violations[subteam_id],
                    "collision_initial_position_radius": self.config.collision_initial_position_radius,
                    "collision_communication_range": self.config.collision_communication_range,
                    "mission_seed": realizations[subteam_id].mission_seed,
                    "mission_dynamics_severity_sigma": self.config.mission_dynamics_severity_sigma,
                    "formation_phase_rad": realizations[subteam_id].formation_phase_rad,
                    "formation_offsets_canonical": {
                        robot_id: value.tolist()
                        for robot_id, value in realizations[subteam_id]
                        .formation_offsets_canonical.items()
                    },
                    "dynamics_severity_multiplier": dynamics_severity_multiplier[subteam_id],
                    "maximum_realized_motion_noise_norm": maximum_motion_noise_norm[subteam_id],
                    "home_snap_occurred": float(home_snap_occurred[subteam_id]),
                    "scored_route_steps": steps_executed,
                    "margin_mode": margin_mode,
                    "robot_margins": {
                        robot_id: float(robot_margins[robot_id])
                        for robot_id in plan.home_positions
                    },
                    "ugv_margin": (
                        None
                        if class_margins is None
                        else float(class_margins[Platform.UGV])
                    ),
                    "uav_margin": (
                        None
                        if class_margins is None
                        else float(class_margins[Platform.UAV])
                    ),
                    "ugv_nn_train_mse": self.models.report(Platform.UGV).train_mse,
                    "uav_nn_train_mse": self.models.report(Platform.UAV).train_mse,
                    "ugv_nn_validation_mse": self.models.report(Platform.UGV).validation_mse,
                    "uav_nn_validation_mse": self.models.report(Platform.UAV).validation_mse,
                },
                audit={
                    "fixed_margin": fixed_margin_audit,
                    "controller": self.controller_name,
                    "controller_strategy": "local_obstacles_only" if self.use_hocbf else "B",
                    "mission_seed": realizations[subteam_id].mission_seed,
                    "mission_dynamics_severity_sigma": self.config.mission_dynamics_severity_sigma,
                    "formation_phase_rad": realizations[subteam_id].formation_phase_rad,
                    "formation_offsets_canonical": {
                        robot_id: value.tolist()
                        for robot_id, value in realizations[subteam_id]
                        .formation_offsets_canonical.items()
                    },
                    "dynamics_severity_multiplier": dynamics_severity_multiplier[subteam_id],
                    "source_plan_seed": plan.seed,
                    "exogenous_realization_digest": exogenous_digests[subteam_id],
                    "collision_initial_position_radius": self.config.collision_initial_position_radius,
                    "collision_motion_noise_sigma": {
                        Platform.UGV.value: self.config.collision_motion_noise_sigma_ugv,
                        Platform.UAV.value: self.config.collision_motion_noise_sigma_uav,
                    },
                    "collision_motion_noise_max": {
                        Platform.UGV.value: self.config.collision_motion_noise_max_ugv,
                        Platform.UAV.value: self.config.collision_motion_noise_max_uav,
                    },
                    "initial_perturbations": {
                        robot_id: realizations[subteam_id].initial_perturbations[robot_id].tolist()
                        for robot_id in plan.home_positions
                    },
                    "initial_perturbations_canonical": {
                        robot_id: realizations[subteam_id].local_initial_perturbations[robot_id].tolist()
                        for robot_id in plan.home_positions
                    },
                    "canonical_subteam_transform": {
                        "frame_id": f"quadrant_reflection_{subteam_id}",
                        "matrix": realizations[subteam_id].frame.matrix.tolist(),
                        "origin": [0.0, 0.0],
                    },
                    "fixed_communication_graph": [
                        list(edge) for edge in sorted(fixed_graphs[subteam_id].edges)
                    ],
                    "fixed_communication_graph_connected": bool(
                        fixed_graphs[subteam_id].is_connected
                    ),
                    "communication_failure": False,
                    "protocol_failure": False,
                    "center_consensus_graph": "alternating_ugv_uav_ring",
                    "fixed_pair_safety_edges": [list(edge) for edge in pair_edges[subteam_id]],
                    "same_layer_pairs_not_safety_edges": [],
                    "cross_layer_noncollision_reason": (
                        "UGVs occupy the ground layer; UAVs occupy the fixed 12 m layer"
                    ),
                    "maximum_realized_motion_noise_norm": maximum_motion_noise_norm[subteam_id],
                    "home_snap_occurred": home_snap_occurred[subteam_id],
                    "speed_domain_violation_count": speed_violations[subteam_id],
                    "maximum_acceleration_model_error": maximum_acceleration_error[subteam_id],
                    "maximum_pair_robust_tightening": maximum_pair_tightening[subteam_id],
                    "maximum_obstacle_robust_tightening": maximum_obstacle_tightening[subteam_id],
                    "maximum_wall_robust_tightening": maximum_wall_tightening[subteam_id],
                    "minimum_wall_clearance": minimum_wall_clearance[subteam_id],
                    "collision_count": collisions[subteam_id],
                    "wall_collision_count": wall_collisions[subteam_id],
                    "obstacle_collision_count": obstacle_collisions[subteam_id],
                    "pair_collision_count": pair_collisions[subteam_id],
                    "failed_robots": {
                        platform.value: list(failed_by_platform[platform])
                        for platform in Platform
                    },
                    "robot_failure_reasons": {
                        robot_id: list(reasons)
                        for robot_id, reasons in local_reasons.items()
                    },
                    "failure_budget_exceeded": {
                        platform.value: failure_budget_exceeded[platform]
                        for platform in Platform
                    },
                    "local_qp_failure_count": qp_failures[subteam_id],
                    "local_qp_solve_count": qp_solves[subteam_id],
                    "minimum_optimized_decay_multiplier": minimum_decay_multiplier[subteam_id],
                    "maximum_optimized_decay_multiplier": maximum_decay_multiplier[subteam_id],
                    "fraction_decay_multipliers_changed": (
                        changed_decay_multipliers[subteam_id]
                        / total_decay_multipliers[subteam_id]
                        if total_decay_multipliers[subteam_id]
                        else 0.0
                    ),
                    "total_decay_penalty": total_decay_penalty[subteam_id],
                    "full_qp_diagnostics_recorded": record_full_qp_diagnostics,
                    "qp_diagnostics": (
                        {
                            robot_id: qp_diagnostic_records[robot_id]
                            for robot_id in plan.home_positions
                        }
                        if record_full_qp_diagnostics
                        else None
                    ),
                    "model_units": "acceleration",
                },
            )
            if local_calibration is not None:
                for platform in Platform:
                    for robot_id, score in scores[subteam_id][platform].items():
                        local_calibration.record_robot_score(plan.mission_id, platform, robot_id, score)
            results.append(result)
        return results, update_graph, platforms


__all__ = [
    "FORMATION_SCENARIO",
    "WangCollisionExperiment",
    "WangControllerStep",
    "WangQPDiagnostics",
    "WangState",
    "WangVehicleState",
    "propagate_true_wang_dynamics",
    "wang_allocation_weights",
    "wang_acceleration_model_error",
    "wang_b",
    "wang_barrier",
    "wang_barrier_rhs",
    "wang_h",
    "wang_obstacle_residuals",
    "wang_pair_residuals",
    "sample_dynamics_severity_multiplier",
]


# The dedicated controller calls its immutable packet ``WangState``.  The
# mutable mission state has the same planar ``(p,v)`` semantics and is exposed
# under that short name for simulation code.
WangState = WangVehicleState


def run(output, resume=False):
    from maicp.configuration import load_config
    from maicp.experiment import run_case_study

    experiment = WangCollisionExperiment(load_config("formation"))
    methods = ('distributed', 'centralized', 'single_agent', 'unshifted', 'fixed', 'nominal')
    run_case_study(experiment, "collision_avoidance", methods, output, resume)
