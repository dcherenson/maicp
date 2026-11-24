"""Four independent target missions sharing a synchronized simulated clock."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import hashlib
import os
from typing import Any

import numpy as np

from maicp.configuration import ExperimentConfig, Platform
from experiments.mission import JointMissionState, build_mission_plans, robot_platforms
from experiments.communication import GraphSnapshot
from maicp.score_stores import LocalCalibrationStores
from maicp.trace import WaveTraceRecorder
from maicp.types import MissionResult

from experiments.tracking.estimator import DRWTTracker, MeasurementPacket
from experiments.tracking.margins import resolve_robot_margins
from experiments.tracking.sensing import (
    CAMERA_FOV_RADIANS,
    CAMERA_MAX_RANGE,
    CAMERA_SIGMA_BEARING,
    LIDAR_FOV_RADIANS,
    LIDAR_MAX_RANGE,
    LIDAR_SIGMA_BEARING,
    LIDAR_SIGMA_RANGE,
    UAV_CROSS_TRACK_FLOOR,
    SensorPacket,
    ugv_lidar_measurement,
    uav_camera_bearing_measurement,
)
from experiments.tracking.uncertainty import (
    TRACKING_SCORE_SEMANTICS,
    conformal_excess_score,
    nominal_radius,
    UncertaintySummary,
    uncertainty_observation,
    validate_xy_covariance,
)


_TRACKING_WORKER_EXPERIMENT: Any | None = None


def _tracker_measurement(packet: SensorPacket) -> object:
    """Convert a physical packet to the DRWT packet type.

    ``MeasurementPacket`` is owned by the tracker module so that all callers
    share one validation boundary.  Keeping this tiny adapter here avoids a
    sensing module dependency on estimator code while allowing sensor unit
    tests to use the richer :class:`SensorPacket` directly.
    """

    return MeasurementPacket(
        visible=packet.visible,
        value=packet.value.copy(),
        H=packet.H.copy(),
        covariance=packet.covariance.copy(),
        sensor_type=packet.sensor_type,
    )


def _initialize_tracking_worker(
    experiment_type: type[Any],
    config: ExperimentConfig,
) -> None:
    """Rebuild one complete experiment inside a mission worker process."""

    global _TRACKING_WORKER_EXPERIMENT
    _TRACKING_WORKER_EXPERIMENT = experiment_type(config)


def _run_tracking_wave_worker(
    margins: Mapping[Platform | str, float],
    first_mission_id: int,
    wave_seed: int,
    collect_local_calibration: bool,
    coverage_radii: Mapping[Platform | str, float] | None,
) -> tuple[
    list[MissionResult],
    dict[str, tuple[str, ...]],
    bool,
    dict[str, Platform],
    LocalCalibrationStores | None,
]:
    """Run one tracking wave and return its local state."""

    if _TRACKING_WORKER_EXPERIMENT is None:  # pragma: no cover - process setup guard
        raise RuntimeError("Tracking mission worker was not initialized")
    local = LocalCalibrationStores() if collect_local_calibration else None
    results, graph, platforms = _TRACKING_WORKER_EXPERIMENT.run_wave(
        dict(margins),
        first_mission_id=first_mission_id,
        wave_seed=wave_seed,
        local_calibration=local,
        coverage_radii=None if coverage_radii is None else dict(coverage_radii),
    )
    graph_payload = {
        node: tuple(sorted(neighbors)) for node, neighbors in graph.items()
    }
    return results, graph_payload, isinstance(graph, GraphSnapshot), platforms, local


@dataclass
class TargetState:
    state: np.ndarray
    center: np.ndarray
    half_width: float = 5.5

    def step(self, dt: float, rng: np.random.Generator) -> None:
        self.state[2:] += rng.normal(0.0, 0.04, size=2)
        speed = np.linalg.norm(self.state[2:])
        if speed > 2.2:
            self.state[2:] *= 2.2 / speed
        self.state[:2] += self.state[2:] * dt
        for axis in (0, 1):
            lower = self.center[axis] - self.half_width
            upper = self.center[axis] + self.half_width
            if self.state[axis] < lower or self.state[axis] > upper:
                self.state[axis] = float(np.clip(self.state[axis], lower, upper))
                self.state[axis + 2] *= -1.0


class TrackingExperiment:
    supports_robot_margins = True
    dt = 0.5

    @staticmethod
    def _keyed_rng(*parts: object) -> np.random.Generator:
        """Seed the random draw from the mission and event identifiers so paired margin settings share disturbances."""

        encoded = "\x1f".join(str(part) for part in parts).encode("utf-8")
        digest = hashlib.blake2b(encoded, digest_size=16).digest()
        seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
        return np.random.default_rng(seed)

    @staticmethod
    def _class_uncertainty_payload(
        summary: UncertaintySummary, radii: Mapping[str, float | None],
    ) -> dict[str, Any]:
        """Report pooled sample diagnostics without inventing a common radius."""
        payload = summary.as_dict()
        values = set(radii.values())
        if len(values) == 1 and next(iter(values)) is not None:
            radius = next(iter(values))
            scale = radius**2 / summary.chi2_threshold
            payload.update(
                coverage_radius=radius, reported_radius=radius,
                covariance_scale=scale, reported_covariance_scale=scale,
                ellipse_axis_scale=float(np.sqrt(scale)),
                reported_ellipse_axis_scale=float(np.sqrt(scale)),
            )
        if len(values) > 1:
            for key in ("coverage_radius", "reported_radius", "covariance_scale",
                        "reported_covariance_scale", "ellipse_axis_scale",
                        "reported_ellipse_axis_scale"):
                payload[key] = None
            payload["reported_radius_range"] = [min(values), max(values)]
            payload["reported_radii_by_robot"] = dict(radii)
        return payload

    @staticmethod
    def _native_position_covariance(covariance: np.ndarray, robot_id: str) -> np.ndarray:
        """Validate an updated 4D estimate covariance and return its XY marginal."""

        value = np.asarray(covariance, dtype=float)
        if value.shape != (4, 4):
            raise ValueError(
                f"Native covariance for {robot_id!r} must have shape (4, 4), got {value.shape}"
            )
        if not np.all(np.isfinite(value)):
            raise ValueError(f"Native covariance for {robot_id!r} must be finite")
        if not np.allclose(value, value.T, rtol=1e-10, atol=1e-12):
            raise ValueError(f"Native covariance for {robot_id!r} must be symmetric")
        value = 0.5 * (value + value.T)
        try:
            np.linalg.cholesky(value)
        except np.linalg.LinAlgError as error:
            raise ValueError(f"Native covariance for {robot_id!r} must be positive definite") from error
        return validate_xy_covariance(
            value[:2, :2],
            name=f"Native XY covariance for {robot_id!r}",
        )

    def __init__(self, config: ExperimentConfig):
        self.config = config
        # Sample sensor severity and persistent biases once per mission.
        self.experiment_metadata = {
            "score_semantics": TRACKING_SCORE_SEMANTICS,
            "sensing_model": "ugv_lidar_range_bearing_uav_camera_bearing_only",
            "sensors": {
                "ugv": {
                    "type": "planar_lidar_range_bearing",
                    "max_range_m": LIDAR_MAX_RANGE,
                    "fov_radians": LIDAR_FOV_RADIANS,
                    "sigma_range_m": LIDAR_SIGMA_RANGE,
                    "sigma_bearing_rad": LIDAR_SIGMA_BEARING,
                    "beta": float(config.classes[Platform.UGV].tracking_beta),
                },
                "uav": {
                    "type": "monocular_bearing_only",
                    "max_range_m": CAMERA_MAX_RANGE,
                    "fov_radians": CAMERA_FOV_RADIANS,
                    "sigma_bearing_rad": CAMERA_SIGMA_BEARING,
                    "covariance_range_source": "pre_update_local_prior_unclipped",
                    "cross_track_floor_m": UAV_CROSS_TRACK_FLOOR,
                    "beta": float(config.classes[Platform.UAV].tracking_beta),
                },
            },
            "task_communication_range": config.task_communication_range,
        }

    def run_wave(
        self,
        margins: Mapping[Platform | str, float],
        first_mission_id: int,
        wave_seed: int,
        trace_recorder: WaveTraceRecorder | None = None,
        local_calibration: LocalCalibrationStores | None = None,
        coverage_radii: Mapping[Platform | str, float] | None = None,
    ) -> tuple[list[MissionResult], dict[str, set[str]], dict[str, Platform]]:
        plans = build_mission_plans(self.config, first_mission_id, wave_seed)
        # Draw UGV severity and persistent calibration values for each robot
        # from semantic keys that exclude robustness margins.  A fixed margin
        # sweep therefore reuses the same physical mission realization.
        sensor_severity: dict[int, dict[str, float]] = {}
        sensor_biases: dict[int, dict[str, dict[str, float]]] = {}
        for plan in plans:
            ugv_severity_rng = self._keyed_rng(
                plan.seed,
                plan.mission_id,
                plan.subteam_id,
                "ugv_sensor_severity",
            )
            sensor_severity[plan.subteam_id] = {
                "lambda_ugv": float(ugv_severity_rng.uniform(0.70, 1.30)),
                # Retain the audit field, with fixed camera angular noise.
                "lambda_uav": 1.0,
            }
            robot_biases: dict[str, dict[str, float]] = {}
            for robot_id in plan.robot_ids[Platform.UGV]:
                range_rng = self._keyed_rng(
                    plan.seed,
                    plan.mission_id,
                    plan.subteam_id,
                    robot_id,
                    "ugv_range_bias",
                )
                bearing_rng = self._keyed_rng(
                    plan.seed,
                    plan.mission_id,
                    plan.subteam_id,
                    robot_id,
                    "ugv_bearing_bias",
                )
                robot_biases[robot_id] = {
                    "range_m": float(range_rng.uniform(-0.15, 0.15)),
                    "bearing_rad": float(bearing_rng.uniform(-0.010, 0.010)),
                }
            for robot_id in plan.robot_ids[Platform.UAV]:
                camera_rng = self._keyed_rng(
                    plan.seed,
                    plan.mission_id,
                    plan.subteam_id,
                    robot_id,
                    "uav_camera_bias",
                )
                robot_biases[robot_id] = {
                    "camera_bearing_rad": float(camera_rng.uniform(-0.015, 0.015)),
                }
            sensor_biases[plan.subteam_id] = robot_biases
        joint = JointMissionState.create(self.config, plans)
        platforms = robot_platforms(plans)
        robot_margins, class_margins = resolve_robot_margins(
            margins, platforms, classes=self.config.classes,
        )
        margin_mode = "class" if class_margins is not None else "robot"
        reporting_margins = None
        if coverage_radii is None:
            robot_coverage_radii = {robot: None for robot in platforms}
            coverage_margin_mode = "native"
        else:
            reporting_margins, class_reporting_margins = resolve_robot_margins(coverage_radii, platforms)
            robot_coverage_radii = {
                robot: nominal_radius(self.config.classes[p].alpha) + reporting_margins[robot]
                for robot, p in platforms.items()
            }
            coverage_margin_mode = "class" if class_reporting_margins is not None else "robot"
        normalized_coverage_radii = {}
        for p in Platform:
            radii = {robot_coverage_radii[robot] for robot in platforms if platforms[robot] == p}
            normalized_coverage_radii[p] = next(iter(radii)) if len(radii) == 1 else None
        reporting_policy = (
            "native_chi_square_ellipse" if coverage_radii is None else
            "robot_calibrated_output_ellipse" if coverage_margin_mode == "robot" else
            "class_calibrated_output_ellipse"
        )
        tracking_window_size = 4
        trackers: dict[int, DRWTTracker] = {}
        targets: dict[int, TargetState] = {}
        scores = {
            plan.subteam_id: {
                platform: {robot_id: 0.0 for robot_id in plan.robot_ids[platform]}
                for platform in Platform
            }
            for plan in plans
        }
        # Keep the numerical score observed by the estimator separate from
        # the production score below.  A failed mission still receives the
        # established infinity sentinel, but the audit at fixed margins retains
        # the finite raw maximum (if any) and its sample count.
        raw_scores: dict[int, dict[Platform, dict[str, float | None]]] = {
            plan.subteam_id: {
                platform: {robot_id: None for robot_id in plan.robot_ids[platform]}
                for platform in Platform
            }
            for plan in plans
        }
        expected_scored_opportunities: dict[int, dict[str, int]] = {
            plan.subteam_id: {robot_id: 0 for robot_id in plan.home_positions}
            for plan in plans
        }
        task_active_timesteps: dict[int, dict[str, int]] = {
            plan.subteam_id: {robot_id: 0 for robot_id in plan.home_positions}
            for plan in plans
        }
        current_drwt_active = {
            robot_id: False for plan in plans for robot_id in plan.home_positions
        }
        covariance_trace = {plan.subteam_id: [] for plan in plans}
        fov_counts = {plan.subteam_id: 0 for plan in plans}
        tracking_errors = {plan.subteam_id: [] for plan in plans}
        tracking_score_samples = {plan.subteam_id: 0 for plan in plans}
        uncertainty_summaries = {
            plan.subteam_id: {
                "classes": {
                    platform.value: UncertaintySummary(
                        self.config.classes[platform].alpha,
                        normalized_coverage_radii[platform],
                    )
                    for platform in Platform
                },
                "robots": {
                    robot_id: UncertaintySummary(
                        self.config.classes[platforms[robot_id]].alpha,
                        robot_coverage_radii[robot_id],
                    )
                    for robot_id in plan.home_positions
                },
            }
            for plan in plans
        }
        physical_task_graph_disconnected = {plan.subteam_id: False for plan in plans}
        physical_task_graph_disconnected_steps = {plan.subteam_id: 0 for plan in plans}
        task_graph_disconnected = {plan.subteam_id: False for plan in plans}
        task_graph_disconnected_steps = {plan.subteam_id: 0 for plan in plans}
        handoff_failed = {plan.subteam_id: False for plan in plans}
        task_graph_updates = {plan.subteam_id: 0 for plan in plans}
        task_graph_edges = {plan.subteam_id: [] for plan in plans}
        observation_history = {
            plan.subteam_id: {
                # [t-T, t] contains T+1 observation opportunities.
                robot_id: deque(maxlen=tracking_window_size + 1)
                for robot_id in plan.home_positions
            }
            for plan in plans
        }

        for plan in plans:
            first_route = np.asarray(plan.waypoints[next(iter(plan.home_positions))])
            sign = np.where(np.mean(first_route[2:7], axis=0) >= 0.0, 1.0, -1.0)
            center = sign * self.config.quadrant_anchor_distance
            # Initialize from the known nominal distribution, independently
            # of the sampled target truth.
            prior_state = np.array([center[0], center[1], -0.8 * sign[0], 0.65 * sign[1]])
            prior_covariance = np.diag([0.45**2, 0.45**2, 0.08**2, 0.08**2])
            target_mean = np.array([center[0], center[1], -0.8 * sign[0], 0.65 * sign[1]])
            initial_rng = self._keyed_rng(
                plan.seed, plan.mission_id, plan.subteam_id, "target_initial"
            )
            target_state = target_mean + np.concatenate(
                [initial_rng.normal(0.0, 0.45, size=2), initial_rng.normal(0.0, 0.08, size=2)]
            )
            targets[plan.subteam_id] = TargetState(target_state, center)
            ids = tuple(plan.home_positions)
            tracker = DRWTTracker(
                robot_ids=ids,
                robot_platforms=platforms,
                neighbors=joint.task_graphs[plan.subteam_id],
                classes=self.config.classes,
                margins=(class_margins if class_margins is not None else
                         {robot: robot_margins[robot] for robot in ids}),
                window_size=tracking_window_size,
                rho=0.8,
                admm_iterations=self.config.profile.tracking_admm_steps,
                dt=self.dt,
            )
            # Initialize the tracker from the nominal prior.
            tracker.initialize(prior_state, prior_covariance=prior_covariance)
            trackers[plan.subteam_id] = tracker

        trace_robot_ids = tuple(sorted(joint.vehicles))
        trace_index = {robot_id: index for index, robot_id in enumerate(trace_robot_ids)}
        trace_subteam = {
            robot_id: plan.subteam_id for plan in plans for robot_id in plan.home_positions
        }
        # Pairing audit. The patrol and sensor inputs are independent
        # of the estimator margins. Hash their realized values without making
        # any additional random draws or including endogenous estimates.
        realization_hashes = {plan.subteam_id: hashlib.sha256() for plan in plans}

        def hash_array(hasher, value):
            array = np.asarray(value, dtype="<f8").copy()
            array[np.isnan(array)] = np.nan  # Flags for missing observations.
            hasher.update(np.asarray(array.shape, dtype="<i8").tobytes())
            hasher.update(array.tobytes())

        for plan in plans:
            hasher = realization_hashes[plan.subteam_id]
            hasher.update(f"tracking-realization-v2:{plan.seed}:{plan.mission_id}:{plan.subteam_id}".encode())
            hash_array(hasher, targets[plan.subteam_id].state)
            hash_array(
                hasher,
                [
                    sensor_severity[plan.subteam_id]["lambda_ugv"],
                    sensor_severity[plan.subteam_id]["lambda_uav"],
                ],
            )
            for robot_id in sorted(plan.home_positions):
                hasher.update(robot_id.encode())
                hash_array(hasher, plan.home_positions[robot_id])
                hash_array(hasher, plan.waypoints[robot_id])
                hash_array(
                    hasher,
                    list(sensor_biases[plan.subteam_id][robot_id].values()),
                )
        if trace_recorder is not None:
            trace_recorder.begin_wave(
                "target_tracking",
                self.config,
                plans,
                joint.task_graphs,
                wave_seed,
                self.dt,
            )
            trace_recorder.metadata["targets"] = [
                {
                    "subteam_id": subteam_id,
                    "center": targets[subteam_id].center.tolist(),
                    "half_width": targets[subteam_id].half_width,
                }
                for subteam_id in sorted(targets)
            ]
            trace_recorder.metadata["score_semantics"] = TRACKING_SCORE_SEMANTICS
            trace_recorder.metadata["sensors"] = {
                **self.experiment_metadata["sensors"],
                "mission_severity": {
                    str(subteam_id): {
                        key: float(value)
                        for key, value in severity.items()
                    }
                    for subteam_id, severity in sorted(sensor_severity.items())
                },
                "persistent_biases": {
                    str(subteam_id): {
                        robot_id: {
                            key: float(value)
                            for key, value in bias.items()
                        }
                        for robot_id, bias in sorted(robot_biases.items())
                    }
                    for subteam_id, robot_biases in sorted(sensor_biases.items())
                },
            }
            trace_recorder.metadata["sensor_severity"] = {
                str(subteam_id): {
                    key: float(value) for key, value in severity.items()
                }
                for subteam_id, severity in sorted(sensor_severity.items())
            }
            trace_recorder.metadata["persistent_sensor_biases"] = {
                str(subteam_id): {
                    robot_id: {
                        key: float(value) for key, value in bias.items()
                    }
                    for robot_id, bias in sorted(robot_biases.items())
                }
                for subteam_id, robot_biases in sorted(sensor_biases.items())
            }
            for robot in trace_recorder.metadata["robots"]:
                subteam_id = int(robot["subteam_id"])
                robot_id = str(robot["robot_id"])
                platform = str(robot["platform"])
                robot["sensor_type"] = (
                    "ugv_lidar" if platform == Platform.UGV.value else "uav_camera_bearing"
                )
                robot["persistent_bias"] = dict(
                    sensor_biases[subteam_id][robot_id]
                )
            trace_recorder.metadata["measurement_schema"] = {
                "ugv": {
                    "sensor_type": "ugv_lidar",
                    "value_shape": [2],
                    "H_shape": [2, 4],
                    "covariance_shape": [2, 2],
                },
                "uav": {
                    "sensor_type": "uav_camera_bearing",
                    "value_shape": [1],
                    "H_shape": [1, 4],
                    "covariance_shape": [1, 1],
                },
            }
            trace_recorder.metadata["sensing"] = {
                "ugv": "planar_lidar_range_bearing",
                "uav": "monocular_bearing_only",
                "truth_usage": "Visibility, physical noise generation, and scoring only.",
                "uav_depth_observation": False,
            }
            trace_recorder.metadata["uncertainty"] = {
                "dimension": 2,
                "policy": reporting_policy,
                "coverage_margin_mode": coverage_margin_mode,
                "coverage_radii_by_robot": dict(robot_coverage_radii),
                "coverage_radii": {
                    platform.value: normalized_coverage_radii[platform]
                    for platform in Platform
                },
                "alpha": {
                    platform.value: float(self.config.classes[platform].alpha)
                    for platform in Platform
                },
                "chi2_threshold": {
                    platform.value: float(
                        uncertainty_summaries[plans[0].subteam_id]["classes"][
                            platform.value
                        ].chi2_threshold
                    )
                    for platform in Platform
                },
            }

            def record_trace_frame(
                time: float,
                target_truth: np.ndarray,
                estimates: np.ndarray,
                covariances: np.ndarray,
                visible: np.ndarray,
                lidar_range: np.ndarray,
                lidar_bearing: np.ndarray,
                camera_bearing: np.ndarray,
                measurement_dimension: np.ndarray,
                ugv_measurement_values: np.ndarray,
                uav_measurement_values: np.ndarray,
                ugv_measurement_H: np.ndarray,
                uav_measurement_H: np.ndarray,
                ugv_measurement_covariances: np.ndarray,
                uav_measurement_covariances: np.ndarray,
                mahalanobis_scores: np.ndarray,
                scored_mask: np.ndarray,
                reported_covariances: np.ndarray,
                covariance_scale: np.ndarray,
            ) -> None:
                phase = []
                for robot_id in trace_robot_ids:
                    vehicle = joint.vehicles[robot_id]
                    if vehicle.waypoint_index <= 1:
                        phase.append(0)
                    elif vehicle.task_active:
                        phase.append(1)
                    else:
                        phase.append(2)
                trace_recorder.record_frame(
                    time,
                    positions=np.asarray([joint.vehicles[rid].position for rid in trace_robot_ids]),
                    headings=np.asarray([joint.vehicles[rid].heading for rid in trace_robot_ids]),
                    phase=np.asarray(phase, dtype=np.int8),
                    drwt_active=np.asarray(
                        [current_drwt_active[rid] for rid in trace_robot_ids], dtype=bool
                    ),
                    target_truth=target_truth,
                    estimates=estimates,
                    covariances=covariances,
                    visible=visible,
                    lidar_range=lidar_range,
                    lidar_bearing=lidar_bearing,
                    camera_bearing=camera_bearing,
                    measurement_dimension=measurement_dimension,
                    ugv_measurement_values=ugv_measurement_values,
                    uav_measurement_values=uav_measurement_values,
                    ugv_measurement_H=ugv_measurement_H,
                    uav_measurement_H=uav_measurement_H,
                    ugv_measurement_covariances=ugv_measurement_covariances,
                    uav_measurement_covariances=uav_measurement_covariances,
                    mahalanobis_scores=mahalanobis_scores,
                    scored_mask=scored_mask,
                    reported_covariances=reported_covariances,
                    covariance_scale=covariance_scale,
                )

            initial_estimates = np.asarray(
                [trackers[trace_subteam[rid]].x_hat[rid][-4:] for rid in trace_robot_ids]
            )
            initial_covariances = np.asarray(
                [
                    trackers[trace_subteam[rid]].P_hat[rid][-4:, -4:]
                    for rid in trace_robot_ids
                ]
            )
            record_trace_frame(
                0.0,
                np.asarray([targets[index].state.copy() for index in range(self.config.num_subteams)]),
                initial_estimates,
                initial_covariances,
                np.zeros(len(trace_robot_ids), dtype=bool),
                np.full(len(trace_robot_ids), np.nan),
                np.full(len(trace_robot_ids), np.nan),
                np.full(len(trace_robot_ids), np.nan),
                np.zeros(len(trace_robot_ids), dtype=np.int8),
                np.full((len(trace_robot_ids), 2), np.nan),
                np.full((len(trace_robot_ids), 1), np.nan),
                np.full((len(trace_robot_ids), 2, 4), np.nan),
                np.full((len(trace_robot_ids), 1, 4), np.nan),
                np.full((len(trace_robot_ids), 2, 2), np.nan),
                np.full((len(trace_robot_ids), 1, 1), np.nan),
                np.full(len(trace_robot_ids), np.nan),
                np.zeros(len(trace_robot_ids), dtype=bool),
                np.full((len(trace_robot_ids), 2, 2), np.nan),
                np.full(len(trace_robot_ids), np.nan),
            )

        dwell = 0
        steps_executed = 0
        termination_reason = "horizon"
        for _ in range(self.config.profile.simulation_steps):
            steps_executed += 1
            frame_visible = {robot_id: False for robot_id in trace_robot_ids}
            frame_lidar_range = {robot_id: np.nan for robot_id in trace_robot_ids}
            frame_lidar_bearing = {robot_id: np.nan for robot_id in trace_robot_ids}
            frame_camera_bearing = {robot_id: np.nan for robot_id in trace_robot_ids}
            frame_measurement_dimension = {
                robot_id: 0 for robot_id in trace_robot_ids
            }
            frame_ugv_measurement_values = {
                robot_id: np.full(2, np.nan) for robot_id in trace_robot_ids
            }
            frame_uav_measurement_values = {
                robot_id: np.full(1, np.nan) for robot_id in trace_robot_ids
            }
            frame_ugv_measurement_H = {
                robot_id: np.full((2, 4), np.nan) for robot_id in trace_robot_ids
            }
            frame_uav_measurement_H = {
                robot_id: np.full((1, 4), np.nan) for robot_id in trace_robot_ids
            }
            frame_ugv_measurement_covariances = {
                robot_id: np.full((2, 2), np.nan) for robot_id in trace_robot_ids
            }
            frame_uav_measurement_covariances = {
                robot_id: np.full((1, 1), np.nan) for robot_id in trace_robot_ids
            }
            frame_estimates: dict[str, np.ndarray] = {}
            frame_covariances: dict[str, np.ndarray] = {}
            frame_mahalanobis_scores = np.full(len(trace_robot_ids), np.nan)
            frame_scored_mask = np.zeros(len(trace_robot_ids), dtype=bool)
            frame_reported_covariances = np.full((len(trace_robot_ids), 2, 2), np.nan)
            frame_covariance_scale = np.full(len(trace_robot_ids), np.nan)

            # Build the communication graph after all targets and vehicles advance.
            for plan in plans:
                target_rng = self._keyed_rng(
                    plan.seed, plan.mission_id, plan.subteam_id, "target_dynamics", steps_executed
                )
                targets[plan.subteam_id].step(self.dt, target_rng)
                for robot_id in plan.home_positions:
                    vehicle = joint.vehicles[robot_id]
                    command = vehicle.desired_planar_input(self.dt)
                    vehicle_rng = self._keyed_rng(
                        plan.seed, plan.mission_id, robot_id, "vehicle_dynamics", steps_executed
                    )
                    vehicle.step(command, self.dt, vehicle_rng)

            current_task_graphs = joint.current_task_graphs(self.config)
            for plan in plans:
                current_graph = current_task_graphs[plan.subteam_id]
                task_graph_updates[plan.subteam_id] += 1
                task_graph_edges[plan.subteam_id].append(len(current_graph.edges))
                if not current_graph.is_connected:
                    # Record connectivity for both the active robots and the full graph.
                    physical_task_graph_disconnected[plan.subteam_id] = True
                    physical_task_graph_disconnected_steps[plan.subteam_id] += 1

            # Every sensor and tracker sees the graph snapshot above.  No
            # graph is rebuilt during the inner ADMM iterations.
            measurements_by_subteam: dict[int, dict[str, object]] = {}
            active_ids_by_subteam: dict[int, tuple[str, ...]] = {}
            for plan in plans:
                measurements: dict[str, object] = {}
                target = targets[plan.subteam_id]
                tracker = trackers[plan.subteam_id]
                for robot_id in plan.home_positions:
                    vehicle = joint.vehicles[robot_id]
                    bias = sensor_biases[plan.subteam_id][robot_id]
                    if vehicle.platform is Platform.UGV:
                        sensor_rng = self._keyed_rng(
                            plan.seed,
                            plan.mission_id,
                            robot_id,
                            "ugv_lidar_noise",
                            steps_executed,
                        )
                        physical_packet = ugv_lidar_measurement(
                            vehicle.position,
                            vehicle.heading,
                            target.state[:2],
                            sensor_rng,
                            severity=sensor_severity[plan.subteam_id]["lambda_ugv"],
                            range_bias=bias["range_m"],
                            bearing_bias=bias["bearing_rad"],
                        )
                    else:
                        # This prior is read before the current physical
                        # tracker update.  It is the only estimate of range
                        # available to the camera covariance calculation.
                        prior_state = tracker.x_hat[robot_id][-4:].copy()
                        sensor_rng = self._keyed_rng(
                            plan.seed,
                            plan.mission_id,
                            robot_id,
                            "uav_camera_noise",
                            steps_executed,
                        )
                        physical_packet = uav_camera_bearing_measurement(
                            vehicle.position,
                            vehicle.heading,
                            target.state[:2],
                            sensor_rng,
                            prior_state=prior_state,
                            camera_bias=bias["camera_bearing_rad"],
                        )
                    measurements[robot_id] = _tracker_measurement(physical_packet)
                    fov_counts[plan.subteam_id] += int(physical_packet.visible)
                    frame_visible[robot_id] = bool(physical_packet.visible)
                    frame_measurement_dimension[robot_id] = (
                        physical_packet.measurement_dimension if physical_packet.visible else 0
                    )
                    if physical_packet.visible:
                        if vehicle.platform is Platform.UGV:
                            frame_lidar_range[robot_id] = float(physical_packet.lidar_range)
                            frame_lidar_bearing[robot_id] = float(physical_packet.lidar_bearing)
                            frame_ugv_measurement_values[robot_id] = physical_packet.value.copy()
                            frame_ugv_measurement_H[robot_id] = physical_packet.H.copy()
                            frame_ugv_measurement_covariances[robot_id] = (
                                physical_packet.covariance.copy()
                            )
                        else:
                            frame_camera_bearing[robot_id] = float(physical_packet.camera_bearing)
                            frame_uav_measurement_values[robot_id] = physical_packet.value.copy()
                            frame_uav_measurement_H[robot_id] = physical_packet.H.copy()
                            frame_uav_measurement_covariances[robot_id] = (
                                physical_packet.covariance.copy()
                            )
                    observation_history[plan.subteam_id][robot_id].append(
                        bool(physical_packet.visible)
                    )
                measurements_by_subteam[plan.subteam_id] = measurements

                graph = current_task_graphs[plan.subteam_id]
                previous_active = set(tracker.active_robot_ids)
                observed_active = {
                    robot_id
                    for robot_id, history in observation_history[plan.subteam_id].items()
                    if any(history)
                }
                # Use the prior participants until the observation window contains a measurement.
                history_mature = all(
                    len(history) == history.maxlen
                    for history in observation_history[plan.subteam_id].values()
                )
                candidate_active = observed_active or previous_active
                tracking_active = any(
                    joint.vehicles[robot_id].task_active for robot_id in plan.home_positions
                )
                if history_mature and not observed_active and tracking_active:
                    # There is no node with observations relevant to the target left to carry the
                    # information from the observation window.  Keep the prior participant
                    # set only to keep the diagnostic simulation running and
                    # record failure of the certified method during tracking.
                    # Losing sight of the target after the scored task has
                    # ended (while returning home) is not a tracking failure.
                    handoff_failed[plan.subteam_id] = True
                leaving = previous_active.difference(candidate_active)
                recipients: dict[str, str] = {}
                blocked_leaving: set[str] = set()
                for leaving_id in sorted(leaving):
                    continuing = sorted(set(graph[leaving_id]).intersection(candidate_active))
                    if continuing:
                        recipients[leaving_id] = continuing[0]
                    else:
                        # Algorithm 1 can remove a node only when a continuing
                        # neighbor exists.  Retain this node and mark the
                        # source connectivity assumption violated if none does.
                        blocked_leaving.add(leaving_id)
                        if tracking_active:
                            handoff_failed[plan.subteam_id] = True
                active_ids = tuple(sorted(candidate_active.union(blocked_leaving)))
                if recipients:
                    tracker.handoff(
                        tuple(recipients),
                        recipients,
                        neighbors=graph,
                    )
                active_ids_by_subteam[plan.subteam_id] = active_ids
                for robot_id in plan.home_positions:
                    current_drwt_active[robot_id] = robot_id in active_ids

            for plan in plans:
                tracker = trackers[plan.subteam_id]
                target = targets[plan.subteam_id]
                for robot_id in plan.home_positions:
                    task_active = joint.vehicles[robot_id].task_active
                    task_active_timesteps[plan.subteam_id][robot_id] += int(task_active)
                    expected_scored_opportunities[plan.subteam_id][robot_id] += int(
                        task_active and current_drwt_active[robot_id]
                    )
                estimates = trackers[plan.subteam_id].step(
                    measurements_by_subteam[plan.subteam_id],
                    neighbors=current_task_graphs[plan.subteam_id],
                    active_robot_ids=active_ids_by_subteam[plan.subteam_id],
                )
                valid_estimates: set[str] = set()
                for robot_id, result in estimates.items():
                    # Score current estimates from active robots.
                    if robot_id not in plan.home_positions or not current_drwt_active[robot_id]:
                        continue
                    try:
                        estimate, covariance = result
                        estimate_value = np.asarray(estimate, dtype=float)
                        if estimate_value.shape != (4,) or not np.all(np.isfinite(estimate_value)):
                            continue
                        covariance_value = np.asarray(covariance, dtype=float)
                        native_covariance = self._native_position_covariance(
                            covariance_value,
                            robot_id,
                        )
                    except (TypeError, ValueError):
                        continue
                    valid_estimates.add(robot_id)
                    frame_estimates[robot_id] = estimate_value.copy()
                    frame_covariances[robot_id] = covariance_value.copy()
                    covariance_trace[plan.subteam_id].append(
                        float(np.trace(native_covariance))
                    )
                    if joint.vehicles[robot_id].task_active:
                        position_error = estimate_value[:2] - target.state[:2]
                        error = float(np.linalg.norm(position_error))
                        tracking_errors[plan.subteam_id].append(error)
                        tracking_score_samples[plan.subteam_id] += 1
                        platform = platforms[robot_id]
                        sample = uncertainty_observation(
                            position_error,
                            native_covariance,
                            alpha=self.config.classes[platform].alpha,
                            coverage_radius=robot_coverage_radii[robot_id],
                            name=f"{robot_id} target position",
                        )
                        uncertainty_summaries[plan.subteam_id]["classes"][
                            platform.value
                        ].add(sample)
                        uncertainty_summaries[plan.subteam_id]["robots"][robot_id].add(sample)
                        raw_score = float(sample["mahalanobis_score"])
                        previous_raw_score = raw_scores[plan.subteam_id][platform][robot_id]
                        raw_scores[plan.subteam_id][platform][robot_id] = (
                            raw_score
                            if previous_raw_score is None
                            else max(previous_raw_score, raw_score)
                        )
                        scores[plan.subteam_id][platform][robot_id] = max(
                            scores[plan.subteam_id][platform][robot_id],
                            conformal_excess_score(raw_score, self.config.classes[platform].alpha),
                        )
                        index = trace_index[robot_id]
                        frame_mahalanobis_scores[index] = raw_score
                        frame_scored_mask[index] = True
                        frame_reported_covariances[index] = np.asarray(
                            sample["reported_covariance"], dtype=float
                        )
                        frame_covariance_scale[index] = float(sample["covariance_scale"])
                if not tracker.last_graph_connected:
                    task_graph_disconnected[plan.subteam_id] = True
                    task_graph_disconnected_steps[plan.subteam_id] += 1

                # Keep the last estimate of each inactive robot for plotting.
                for robot_id in plan.home_positions:
                    if robot_id not in valid_estimates:
                        frame_estimates[robot_id] = tracker.x_hat[robot_id][-4:].copy()
                        frame_covariances[robot_id] = tracker.P_hat[robot_id][-4:, -4:].copy()

            for plan in plans:
                hasher = realization_hashes[plan.subteam_id]
                hash_array(hasher, targets[plan.subteam_id].state)
                for robot_id in sorted(plan.home_positions):
                    vehicle = joint.vehicles[robot_id]
                    hash_array(hasher, vehicle.position)
                    hash_array(hasher, vehicle.velocity)
                    hash_array(hasher, [vehicle.heading, float(frame_visible[robot_id])])
                    # Hash the physical measurements. Camera covariance depends on the local prior.
                    hash_array(hasher, [frame_lidar_range[robot_id]])
                    hash_array(hasher, [frame_lidar_bearing[robot_id]])
                    hash_array(hasher, [frame_camera_bearing[robot_id]])
                    hash_array(hasher, [frame_measurement_dimension[robot_id]])

            if trace_recorder is not None:
                record_trace_frame(
                    steps_executed * self.dt,
                    np.asarray(
                        [targets[index].state.copy() for index in range(self.config.num_subteams)]
                    ),
                    np.asarray([frame_estimates[rid] for rid in trace_robot_ids]),
                    np.asarray([frame_covariances[rid] for rid in trace_robot_ids]),
                    np.asarray([frame_visible[rid] for rid in trace_robot_ids], dtype=bool),
                    np.asarray([frame_lidar_range[rid] for rid in trace_robot_ids]),
                    np.asarray([frame_lidar_bearing[rid] for rid in trace_robot_ids]),
                    np.asarray([frame_camera_bearing[rid] for rid in trace_robot_ids]),
                    np.asarray(
                        [frame_measurement_dimension[rid] for rid in trace_robot_ids],
                        dtype=np.int8,
                    ),
                    np.asarray(
                        [frame_ugv_measurement_values[rid] for rid in trace_robot_ids]
                    ),
                    np.asarray(
                        [frame_uav_measurement_values[rid] for rid in trace_robot_ids]
                    ),
                    np.asarray(
                        [frame_ugv_measurement_H[rid] for rid in trace_robot_ids]
                    ),
                    np.asarray(
                        [frame_uav_measurement_H[rid] for rid in trace_robot_ids]
                    ),
                    np.asarray(
                        [frame_ugv_measurement_covariances[rid] for rid in trace_robot_ids]
                    ),
                    np.asarray(
                        [frame_uav_measurement_covariances[rid] for rid in trace_robot_ids]
                    ),
                    frame_mahalanobis_scores,
                    frame_scored_mask,
                    frame_reported_covariances,
                    frame_covariance_scale,
                )
                trace_recorder.record_task_graph(
                    steps_executed * self.dt,
                    {
                        subteam_id: {
                            robot_id: set(neighbors)
                            for robot_id, neighbors in graph.items()
                        }
                        for subteam_id, graph in current_task_graphs.items()
                    },
                    invalid_delivery_count=sum(
                        tracker.last_invalid_delivery_count for tracker in trackers.values()
                    ),
                )

            if all(vehicle.at_home for vehicle in joint.vehicles.values()):
                dwell += 1
                if dwell >= self.config.home_dwell_steps:
                    termination_reason = "returned_home"
                    break
            else:
                dwell = 0

        terminal_task_graphs = joint.current_task_graphs(self.config)
        connected, update_graph = joint.update_graph_connected(self.config)
        if trace_recorder is not None:
            trace_recorder.set_update_graph(update_graph)
        results: list[MissionResult] = []
        for plan in plans:
            mission_coverage_radii = {}
            for p in Platform:
                radii = {robot_coverage_radii[robot] for robot in plan.robot_ids[p]}
                mission_coverage_radii[p] = next(iter(radii)) if len(radii) == 1 else None
            # Track missing current returns only while on task and in
            # DRWT. The number of steps on task is the availability denominator.
            uncertainty_summary = uncertainty_summaries[plan.subteam_id]
            for robot_id, summary in uncertainty_summary["robots"].items():
                summary.set_expected_samples(
                    expected_scored_opportunities[plan.subteam_id][robot_id]
                )
            for platform in Platform:
                uncertainty_summary["classes"][platform.value].set_expected_samples(
                    sum(
                        expected_scored_opportunities[plan.subteam_id][robot_id]
                        for robot_id in plan.robot_ids[platform]
                    )
                )
            returned = all(joint.vehicles[robot_id].at_home for robot_id in plan.home_positions)
            failure_reason = None
            if not returned:
                failure_reason = "return_timeout"
            if task_graph_disconnected[plan.subteam_id]:
                failure_reason = (
                    f"{failure_reason};task_graph_disconnected"
                    if failure_reason
                    else "task_graph_disconnected"
                )
            if handoff_failed[plan.subteam_id]:
                failure_reason = (
                    f"{failure_reason};active_handoff_unavailable"
                    if failure_reason
                    else "active_handoff_unavailable"
                )
            if not connected:
                failure_reason = (
                    f"{failure_reason};update_graph_disconnected"
                    if failure_reason
                    else "update_graph_disconnected"
                )
            # An empty robot score domain is a violation for that robot. It
            # does not replace other robots' scores or change the rule requiring K of N robots.
            for platform in Platform:
                for robot_id in plan.robot_ids[platform]:
                    if uncertainty_summary["robots"][robot_id].samples == 0:
                        scores[plan.subteam_id][platform][robot_id] = float("inf")
            if failure_reason is not None:
                for platform in Platform:
                    for robot_id in scores[plan.subteam_id][platform]:
                        scores[plan.subteam_id][platform][robot_id] = float("inf")
            # Store each robot's final score at its own node.
            if local_calibration is not None:
                for platform in Platform:
                    for robot_id, score in scores[plan.subteam_id][platform].items():
                        local_calibration.record_robot_score(
                            plan.mission_id,
                            platform,
                            robot_id,
                            score,
                        )
            traces = covariance_trace[plan.subteam_id]
            errors = tracking_errors[plan.subteam_id]
            uncertainty_audit = {
                "dimension": 2,
                "policy": reporting_policy,
                "coverage_margin_mode": coverage_margin_mode,
                "coverage_radii_by_robot": {
                    robot: robot_coverage_radii[robot] for robot in plan.home_positions
                },
                "coverage_radii": {
                    platform.value: mission_coverage_radii[platform]
                    for platform in Platform
                },
                "classes": {
                    class_name: self._class_uncertainty_payload(
                        summary,
                        {robot: robot_coverage_radii[robot] for robot in plan.robot_ids[Platform(class_name)]},
                    )
                    for class_name, summary in uncertainty_summary["classes"].items()
                },
                "robots": {
                    robot_id: {
                        **summary.as_dict(),
                        "task_active_timesteps": task_active_timesteps[plan.subteam_id][robot_id],
                        "valid_drwt_score_timesteps": int(summary.samples),
                        "availability_fraction": (
                            summary.samples / task_active_timesteps[plan.subteam_id][robot_id]
                            if task_active_timesteps[plan.subteam_id][robot_id] else 0.0
                        ),
                    }
                    for robot_id, summary in uncertainty_summary["robots"].items()
                },
            }
            # Application outcomes remain separate from numerical score
            # coverage.  In particular, a robot can have a finite observed
            # maximum and complete numerical samples while the mission later
            # fails a return or graph protocol check.
            robot_failure_reasons: dict[str, list[str]] = {
                robot_id: [] for robot_id in plan.home_positions
            }
            for robot_id in plan.home_positions:
                vehicle = joint.vehicles[robot_id]
                if not vehicle.at_home:
                    robot_failure_reasons[robot_id].append("return_timeout")
                if task_graph_disconnected[plan.subteam_id]:
                    robot_failure_reasons[robot_id].append("task_graph_disconnected")
                if handoff_failed[plan.subteam_id]:
                    robot_failure_reasons[robot_id].append("active_handoff_unavailable")
                if not connected:
                    robot_failure_reasons[robot_id].append("update_graph_disconnected")
                if uncertainty_summary["robots"][robot_id].samples == 0:
                    robot_failure_reasons[robot_id].append("no_scored_estimates")
            robot_audit: dict[str, dict[str, Any]] = {}
            for platform in Platform:
                for robot_id in plan.robot_ids[platform]:
                    summary = uncertainty_summary["robots"][robot_id]
                    summary_payload = summary.as_dict()
                    samples = int(summary_payload["samples"])
                    task_steps = task_active_timesteps[plan.subteam_id][robot_id]
                    raw_score = raw_scores[plan.subteam_id][platform][robot_id]
                    available_interval_complete = True
                    intended_interval_finished = bool(
                        not joint.vehicles[robot_id].task_active
                    )
                    # Use positive infinity when a robot has no valid tracking estimate.
                    score_complete = True
                    reasons = list(robot_failure_reasons[robot_id])
                    summary_payload.update(
                        {
                            "platform": platform.value,
                            "score_semantics": TRACKING_SCORE_SEMANTICS,
                            "nominal_radius": nominal_radius(self.config.classes[platform].alpha),
                            "score_available": True,
                            "task_active_timesteps": int(task_steps),
                            "valid_drwt_score_timesteps": samples,
                            "availability_fraction": samples / task_steps if task_steps else 0.0,
                            "zero_valid_drwt_estimates": samples == 0,
                            "production_score_is_infinite": bool(
                                np.isposinf(scores[plan.subteam_id][platform][robot_id])
                            ),
                            "raw_score": None if raw_score is None else float(raw_score),
                            "production_score": (
                                None
                                if not np.isfinite(scores[plan.subteam_id][platform][robot_id])
                                else float(scores[plan.subteam_id][platform][robot_id])
                            ),
                            "available_interval_complete": available_interval_complete,
                            "final_task_active": bool(
                                joint.vehicles[robot_id].task_active
                            ),
                            "intended_interval_finished": intended_interval_finished,
                            "score_complete": score_complete,
                            "application_failed": bool(reasons),
                            "failure_reasons": reasons,
                            "graph_failure": bool(
                                task_graph_disconnected[plan.subteam_id]
                                or not connected
                            ),
                            "physical_graph_failure": bool(
                                physical_task_graph_disconnected[plan.subteam_id]
                            ),
                            "handoff_failure": bool(handoff_failed[plan.subteam_id]),
                            "return_failure": bool(not joint.vehicles[robot_id].at_home),
                            "no_estimate_failure": bool(summary.samples == 0),
                            "trajectory_containment": {
                                "native_chi_squared": summary_payload[
                                    "native_trajectory_contained"
                                ],
                                "reported_radius": summary_payload[
                                    "reported_trajectory_contained"
                                ],
                            },
                        }
                    )
                    robot_audit[robot_id] = summary_payload

            class_audit: dict[str, dict[str, Any]] = {}
            class_raw_scores: list[float] = []
            for platform in Platform:
                summary = uncertainty_summary["classes"][platform.value]
                summary_payload = summary.as_dict()
                class_robot_scores = [
                    raw_scores[plan.subteam_id][platform][robot_id]
                    for robot_id in plan.robot_ids[platform]
                ]
                finite_class_scores = [
                    value for value in class_robot_scores if value is not None
                ]
                if finite_class_scores:
                    class_raw_scores.extend(float(value) for value in finite_class_scores)
                class_robot_payloads = [
                    robot_audit[robot_id] for robot_id in plan.robot_ids[platform]
                ]
                available_complete = bool(
                    class_robot_payloads
                    and all(
                        payload["available_interval_complete"]
                        for payload in class_robot_payloads
                    )
                )
                intended_finished = bool(
                    class_robot_payloads
                    and all(
                        payload["intended_interval_finished"]
                        for payload in class_robot_payloads
                    )
                )
                complete = bool(
                    class_robot_payloads
                    and all(payload["score_complete"] for payload in class_robot_payloads)
                )
                class_task_steps = sum(
                    payload["task_active_timesteps"] for payload in class_robot_payloads
                )
                class_audit[platform.value] = {
                    **summary_payload,
                    "score_semantics": TRACKING_SCORE_SEMANTICS,
                    "task_active_timesteps": int(class_task_steps),
                    "valid_drwt_score_timesteps": int(summary.samples),
                    "availability_fraction": (
                        summary.samples / class_task_steps if class_task_steps else 0.0
                    ),
                    "raw_score": (
                        None
                        if not finite_class_scores
                        else float(max(finite_class_scores))
                    ),
                    "available_interval_complete": available_complete,
                    "intended_interval_finished": intended_finished,
                    "score_complete": complete,
                    "robot_ids": list(plan.robot_ids[platform]),
                    "trajectory_containment": {
                        "native_chi_squared": summary_payload[
                            "native_trajectory_contained"
                        ],
                        "reported_radius": summary_payload[
                            "reported_trajectory_contained"
                        ],
                    },
                }
            all_expected = sum(
                int(payload["expected_scored_opportunities"])
                for payload in robot_audit.values()
            )
            all_samples = sum(int(payload["samples"]) for payload in robot_audit.values())
            all_task_steps = sum(payload["task_active_timesteps"] for payload in robot_audit.values())
            all_missing = max(all_expected - all_samples, 0)
            available_interval_complete = bool(
                robot_audit
                and all(
                    payload["available_interval_complete"]
                    for payload in robot_audit.values()
                )
            )
            intended_interval_finished = bool(
                robot_audit
                and all(
                    payload["intended_interval_finished"]
                    for payload in robot_audit.values()
                )
            )
            fixed_failure_reasons = [] if failure_reason is None else failure_reason.split(";")
            fixed_internal_margins = (
                {p.value: float(class_margins[p]) for p in Platform}
                if class_margins is not None else
                {robot: robot_margins[robot] for robot in plan.home_positions}
            )
            fixed_margin_audit = {
                "score_semantics": TRACKING_SCORE_SEMANTICS,
                "mission_seed": int(plan.seed),
                "exogenous_realization_digest": realization_hashes[plan.subteam_id].hexdigest(),
                "exogenous_pairing_fields": [
                    "mission_identity", "initial_target", "sensor_severity",
                    "persistent_sensor_biases",
                    "homes", "routes", "target_truth", "patrol_positions",
                    "patrol_velocities", "patrol_headings", "visibility",
                    "lidar_range", "lidar_bearing", "camera_bearing",
                    "measurement_dimension",
                ],
                "sensor_configuration": {
                    "lambda_ugv": float(sensor_severity[plan.subteam_id]["lambda_ugv"]),
                    "lambda_uav": float(sensor_severity[plan.subteam_id]["lambda_uav"]),
                    "persistent_biases": {
                        robot_id: {
                            key: float(value)
                            for key, value in sensor_biases[plan.subteam_id][robot_id].items()
                        }
                        for robot_id in sorted(sensor_biases[plan.subteam_id])
                    },
                },
                # Keep the mission results directly addressable in the
                # result manifest as well as in the grouped configuration.
                "lambda_ugv": float(sensor_severity[plan.subteam_id]["lambda_ugv"]),
                "lambda_uav": float(sensor_severity[plan.subteam_id]["lambda_uav"]),
                "persistent_sensor_biases": {
                    robot_id: {
                        key: float(value)
                        for key, value in sensor_biases[plan.subteam_id][robot_id].items()
                    }
                    for robot_id in sorted(sensor_biases[plan.subteam_id])
                },
                "sensing_model": self.experiment_metadata["sensing_model"],
                "sensor_parameters": self.experiment_metadata["sensors"],
                "internal_margins": fixed_internal_margins,
                "margin_mode": margin_mode,
                "robot_internal_margins": {robot: robot_margins[robot] for robot in plan.home_positions},
                "coverage_margin_mode": coverage_margin_mode,
                "coverage_radii_by_robot": {robot: robot_coverage_radii[robot] for robot in plan.home_positions},
                "coverage_radii": {
                    platform.value: mission_coverage_radii[platform]
                    for platform in Platform
                },
                "reporting_mode": (
                    "native_chi_squared" if coverage_radii is None else "nominal_plus_robustness"
                ),
                "native_reporting": coverage_radii is None,
                "zero_internal_margins": bool(
                    all(robot_margins[robot] == 0.0 for robot in plan.home_positions)
                ),
                "zero_reporting_radii": bool(
                    all(
                        robot_coverage_radii[robot] == 0.0
                        for robot in plan.home_positions
                    )
                ),
                "native_nominal_reference": bool(
                    all(robot_margins[robot] == 0.0 for robot in plan.home_positions)
                    and (coverage_radii is None or all(reporting_margins[robot] == 0.0 for robot in plan.home_positions))
                ),
                "requested_steps": int(self.config.profile.simulation_steps),
                "executed_steps": int(steps_executed),
                "termination_reason": termination_reason,
                "elapsed_time": float(steps_executed * self.dt),
                "completion_time": (
                    float(steps_executed * self.dt) if returned else None
                ),
                "task_completed": bool(returned),
                "returned_home": bool(returned),
                "score_complete": bool(
                    all(payload["score_complete"] for payload in robot_audit.values())
                    and bool(robot_audit)
                ),
                "available_interval_complete": available_interval_complete,
                "intended_interval_finished": intended_interval_finished,
                "raw_score": None if not class_raw_scores else float(max(class_raw_scores)),
                "scored_samples": int(all_samples),
                "task_active_timesteps": int(all_task_steps),
                "valid_drwt_score_timesteps": int(all_samples),
                "availability_fraction": (
                    all_samples / all_task_steps if all_task_steps else 0.0
                ),
                "expected_scored_opportunities": int(all_expected),
                "missing_estimates": int(all_missing),
                "failure": {
                    "failed": bool(failure_reason is not None),
                    "reasons": fixed_failure_reasons,
                },
                "graph": {
                    "task_disconnected": bool(task_graph_disconnected[plan.subteam_id]),
                    "task_reason": (
                        "task_graph_disconnected"
                        if task_graph_disconnected[plan.subteam_id]
                        else None
                    ),
                    "physical_task_disconnected": bool(
                        physical_task_graph_disconnected[plan.subteam_id]
                    ),
                    "physical_task_reason": (
                        "physical_task_graph_disconnected"
                        if physical_task_graph_disconnected[plan.subteam_id]
                        else None
                    ),
                    "update_disconnected": bool(not connected),
                    "update_reason": (
                        "update_graph_disconnected" if not connected else None
                    ),
                    "task_disconnected_steps": int(
                        task_graph_disconnected_steps[plan.subteam_id]
                    ),
                    "physical_task_disconnected_steps": int(
                        physical_task_graph_disconnected_steps[plan.subteam_id]
                    ),
                },
                "handoff": {
                    "failed": bool(handoff_failed[plan.subteam_id]),
                    "reason": (
                        "active_handoff_unavailable"
                        if handoff_failed[plan.subteam_id]
                        else None
                    ),
                },
                "return": {
                    "failed": bool(not returned),
                    "reason": "return_timeout" if not returned else None,
                },
                "no_estimates": {
                    "failed": bool(
                        any(payload["samples"] == 0 for payload in robot_audit.values())
                    ),
                    "reason": (
                        "no_scored_estimates"
                        if any(payload["samples"] == 0 for payload in robot_audit.values())
                        else None
                    ),
                    "robot_ids": [
                        robot_id
                        for robot_id, payload in robot_audit.items()
                        if payload["samples"] == 0
                    ],
                },
                "metrics": {
                    "fov_measurement_count": int(fov_counts[plan.subteam_id]),
                    "tracking_score_samples": int(tracking_score_samples[plan.subteam_id]),
                    "maximum_tracking_error": (
                        None if not errors else float(max(errors))
                    ),
                    "rms_tracking_error": (
                        None
                        if not errors
                        else float(np.sqrt(np.mean(np.square(errors))))
                    ),
                    "mean_position_covariance_trace": (
                        None if not traces else float(np.mean(traces))
                    ),
                    "task_graph_updates": int(task_graph_updates[plan.subteam_id]),
                    "task_graph_min_edges": int(
                        min(task_graph_edges[plan.subteam_id], default=0)
                    ),
                    "task_graph_max_edges": int(
                        max(task_graph_edges[plan.subteam_id], default=0)
                    ),
                    "mean_task_graph_edges": (
                        None
                        if not task_graph_edges[plan.subteam_id]
                        else float(np.mean(task_graph_edges[plan.subteam_id]))
                    ),
                },
                "classes": class_audit,
                "robots": robot_audit,
            }
            results.append(
                MissionResult(
                    mission_id=plan.mission_id,
                    subteam_id=plan.subteam_id,
                    seed=plan.seed,
                    robot_scores=scores[plan.subteam_id],
                    mission_graph={node: tuple(sorted(peers)) for node, peers in terminal_task_graphs[plan.subteam_id].items()},
                    returned_home=returned,
                    update_graph_connected=connected,
                    failed=failure_reason is not None,
                    failure_reason=failure_reason,
                    audit={
                        "uncertainty": uncertainty_audit,
                        "fixed_margin": fixed_margin_audit,
                    },
                    metrics={
                        "mean_position_covariance_trace": float(np.mean(traces)) if traces else float("nan"),
                        "fov_measurement_count": fov_counts[plan.subteam_id],
                        "admm_iterations": self.config.profile.tracking_admm_steps,
                        "tracking_score_samples": tracking_score_samples[plan.subteam_id],
                        "maximum_tracking_error": max(errors, default=float("nan")),
                        "rms_tracking_error": (
                            float(np.sqrt(np.mean(np.square(errors)))) if errors else float("nan")
                        ),
                        "task_graph_updates": task_graph_updates[plan.subteam_id],
                        "task_graph_disconnected": task_graph_disconnected[plan.subteam_id],
                        "task_graph_disconnected_steps": task_graph_disconnected_steps[
                            plan.subteam_id
                        ],
                        "physical_task_graph_disconnected": physical_task_graph_disconnected[
                            plan.subteam_id
                        ],
                        "physical_task_graph_disconnected_steps": physical_task_graph_disconnected_steps[
                            plan.subteam_id
                        ],
                        "task_graph_min_edges": min(task_graph_edges[plan.subteam_id], default=0),
                        "task_graph_max_edges": max(task_graph_edges[plan.subteam_id], default=0),
                        "mean_task_graph_edges": (
                            float(np.mean(task_graph_edges[plan.subteam_id]))
                            if task_graph_edges[plan.subteam_id]
                            else float("nan")
                        ),
                    },
                )
            )
        return results, update_graph, platforms

    def collect_missions(
        self,
        margins: Mapping[Platform | str, float],
        count: int,
        seed: int,
        trace_recorder: WaveTraceRecorder | None = None,
        progress_callback: Callable[[int], None] | None = None,
        local_calibration: LocalCalibrationStores | None = None,
        coverage_radii: Mapping[Platform | str, float] | None = None,
    ) -> tuple[list[MissionResult], dict[str, set[str]], dict[str, Platform]]:
        if count <= 0 or count % self.config.num_subteams:
            raise ValueError("Mission count must be positive and divisible by the four subteams")
        wave_count = count // self.config.num_subteams
        configured_workers = self.config.mission_workers or min(4, os.cpu_count() or 1, wave_count)

        if configured_workers <= 1 or wave_count <= 1:
            results: list[MissionResult] = []
            final_graph: dict[str, set[str]] | None = None
            platforms: dict[str, Platform] | None = None
            for wave in range(wave_count):
                wave_results, final_graph, platforms = self.run_wave(
                    margins,
                    first_mission_id=wave * self.config.num_subteams,
                    wave_seed=seed + 1_000_003 * wave,
                    trace_recorder=trace_recorder if wave == 0 else None,
                    local_calibration=local_calibration,
                    coverage_radii=coverage_radii,
                )
                results.extend(wave_results)
                if progress_callback is not None:
                    progress_callback(len(wave_results))
            assert final_graph is not None and platforms is not None
            return results, final_graph, platforms

        collect_local = local_calibration is not None
        results = []
        final_graph = None
        platforms = None
        pending_waves = list(range(wave_count))

        # A trace is defined for the first wave only.  Running that wave in
        # the caller also keeps the recorder out of arguments passed to worker processes.
        if trace_recorder is not None:
            wave_results, final_graph, platforms = self.run_wave(
                margins,
                first_mission_id=0,
                wave_seed=seed,
                trace_recorder=trace_recorder,
                local_calibration=local_calibration,
                coverage_radii=coverage_radii,
            )
            results.extend(wave_results)
            if progress_callback is not None:
                progress_callback(len(wave_results))
            pending_waves = pending_waves[1:]

        worker_coverage = None if coverage_radii is None else dict(coverage_radii)
        if pending_waves:
            with ProcessPoolExecutor(
                max_workers=min(configured_workers, len(pending_waves)),
                initializer=_initialize_tracking_worker,
                initargs=(
                    type(self),
                    self.config,
                ),
            ) as executor:
                futures = [
                    executor.submit(
                        _run_tracking_wave_worker,
                        dict(margins),
                        wave * self.config.num_subteams,
                        seed + 1_000_003 * wave,
                        collect_local,
                        worker_coverage,
                    )
                    for wave in pending_waves
                ]
                for future in futures:
                    (
                        wave_results,
                        graph_payload,
                        graph_is_snapshot,
                        wave_platforms,
                        wave_local,
                    ) = future.result()
                    final_graph = (
                        GraphSnapshot.from_adjacency(graph_payload)
                        if graph_is_snapshot
                        else {node: set(neighbors) for node, neighbors in graph_payload.items()}
                    )
                    platforms = wave_platforms
                    results.extend(wave_results)
                    if local_calibration is not None:
                        assert wave_local is not None
                        local_calibration.merge_from(wave_local)
                    if progress_callback is not None:
                        progress_callback(len(wave_results))
        assert final_graph is not None and platforms is not None
        return results, final_graph, platforms

    def analytic_kappa(self, platforms: dict[str, Platform]) -> dict[str, float]:
        # A uniform positive lower bound on the smallest eigenvalue of each
        # information matrix over each observation window is unavailable for this case.
        return {robot_id: float("nan") for robot_id in platforms}


def run(output, resume=False):
    from maicp.configuration import load_config
    from maicp.experiment import run_case_study

    experiment = TrackingExperiment(load_config("tracking"))
    methods = ('distributed', 'centralized', 'unshifted', 'nominal', 'fixed', 'single_agent')
    run_case_study(experiment, "target_tracking", methods, output, resume)
