"""Record one synchronized wave of four subteams for replay."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np

from maicp.configuration import ExperimentConfig, Platform
from .types import MissionPlan


@dataclass
class WaveTraceRecorder:
    """Collect visualization data without participating in simulation logic."""

    context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    arrays: dict[str, list[np.ndarray]] = field(default_factory=dict)
    _keys: tuple[str, ...] = ()

    def begin_wave(
        self,
        application: str,
        config: ExperimentConfig,
        plans: tuple[MissionPlan, ...],
        task_graphs: dict[int, dict[str, set[str]]],
        wave_seed: int,
        dt: float,
        obstacles: list[dict[str, Any]] | None = None,
    ) -> None:
        if self.metadata:
            raise RuntimeError("A trace recorder can record only one wave")
        robots = []
        for plan in plans:
            for platform, robot_ids in plan.robot_ids.items():
                for robot_id in robot_ids:
                    sensor_type = None
                    if application == "target_tracking":
                        sensor_type = (
                            "ugv_lidar"
                            if platform is Platform.UGV
                            else "uav_camera_bearing"
                        )
                    robots.append(
                        {
                            "robot_id": robot_id,
                            "subteam_id": plan.subteam_id,
                            "platform": platform.value,
                            **({"sensor_type": sensor_type} if sensor_type else {}),
                            "home": list(plan.home_positions[robot_id]),
                            "waypoints": [list(point) for point in plan.waypoints[robot_id]],
                        }
                    )
        robots.sort(key=lambda item: item["robot_id"])
        task_edges = []
        for subteam, graph in task_graphs.items():
            for robot_id, neighbors in graph.items():
                for neighbor in neighbors:
                    if robot_id < neighbor:
                        task_edges.append([robot_id, neighbor, subteam])
        self.metadata = {
            "application": application,
            "profile": config.profile.name,
            "uav_dynamics": config.uav_dynamics.value,
            "sensitivity_mode": config.sensitivity_mode.value,
            "experiment_seed": int(config.seed),
            "wave_seed": int(wave_seed),
            "dt": float(dt),
            "profile_parameters": {
                "outer_replicates": config.profile.outer_replicates,
                "rounds": config.profile.rounds,
                "calibration_missions": config.profile.calibration_missions,
                "deployment_missions": config.profile.deployment_missions,
                "simulation_steps": config.profile.simulation_steps,
                "tracking_admm_steps": config.profile.tracking_admm_steps,
                "mission_pinball_steps": config.profile.mission_pinball_steps,
                "threshold_pinball_steps": config.profile.threshold_pinball_steps,
            },
            "class_parameters": {
                platform.value: {
                    "robots_per_mission": values.robots_per_mission,
                    "alpha": values.alpha,
                    "delta_cal": values.delta_cal,
                    "violations": values.violations,
                    "scale": values.scale,
                    "weight": values.weight,
                    "initial_margin": values.initial_margin,
                    "margin_min": values.margin_min,
                    "margin_max": values.margin_max,
                    "tracking_beta": values.tracking_beta,
                    "configured_kappa": values.configured_kappa,
                }
                for platform, values in config.classes.items()
            },
            "communication_ranges": {
                "task": config.task_communication_range,
                "update": config.update_communication_range,
            },
            "map_half_extent": float(config.map_half_extent),
            "quadrant_anchor_distance": float(config.quadrant_anchor_distance),
            "depot_pad_distance": float(config.depot_pad_distance),
            "home_tolerance": float(config.home_tolerance),
            "context": self.context,
            "robots": robots,
            "task_edges": sorted(task_edges),
            "task_graph_frames": [],
            "obstacles": obstacles or [],
            "update_edges": [],
        }
        if application == "target_tracking":
            self.metadata["trace_schema_version"] = 2
            # Store scalar UAV bearings separately from planar UGV measurements.
            self.metadata["measurement_schema"] = {
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
        self.record_task_graph(0.0, task_graphs)

    def record_task_graph(
        self,
        time: float,
        task_graphs: dict[int, dict[str, set[str]]],
        invalid_delivery_count: int = 0,
    ) -> None:
        """Save the physical edge snapshot used by an application at one step."""

        if not self.metadata:
            raise RuntimeError("begin_wave must be called before recording task graphs")
        if invalid_delivery_count < 0:
            raise ValueError("Invalid-delivery count cannot be negative")
        frames = self.metadata["task_graph_frames"]
        if frames and time <= frames[-1]["time"]:
            raise ValueError("Task graph frame times must increase")
        edges: list[list[object]] = []
        connected: dict[str, bool] = {}
        edge_counts: dict[str, int] = {}
        for subteam, graph in sorted(task_graphs.items()):
            nodes = set(graph)
            visited: set[str] = set()
            if nodes:
                pending = [min(nodes)]
                while pending:
                    node = pending.pop()
                    if node in visited:
                        continue
                    visited.add(node)
                    pending.extend(graph[node] - visited)
            connected[str(subteam)] = visited == nodes
            count = 0
            for node, neighbors in graph.items():
                for neighbor in neighbors:
                    if node < neighbor:
                        edges.append([node, neighbor, subteam])
                        count += 1
            edge_counts[str(subteam)] = count
        frames.append(
            {
                "time": float(time),
                "edges": sorted(edges),
                "edge_count": edge_counts,
                "connected": connected,
                "invalid_delivery_count": int(invalid_delivery_count),
            }
        )

    def set_update_graph(self, graph: dict[str, set[str]]) -> None:
        edges = sorted(
            [robot_id, neighbor]
            for robot_id, neighbors in graph.items()
            for neighbor in neighbors
            if robot_id < neighbor
        )
        if self.metadata["update_edges"] and self.metadata["update_edges"] != edges:
            raise ValueError("The update graph must remain fixed for a recorded round")
        self.metadata["update_edges"] = edges

    def record_frame(self, time: float, **values: Any) -> None:
        if not self.metadata:
            raise RuntimeError("begin_wave must be called before recording frames")
        expected = tuple(sorted(values))
        if not self._keys:
            self._keys = expected
            self.arrays = {"time": []} | {key: [] for key in expected}
        elif expected != self._keys:
            raise ValueError("Every frame in a trace must provide the same arrays")
        self.arrays["time"].append(np.asarray(time, dtype=float))
        for key in self._keys:
            self.arrays[key].append(np.asarray(values[key]))

    def save(self, path: str | Path) -> Path:
        if not self.arrays.get("time"):
            raise ValueError("Cannot save a trace without frames")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {key: np.stack(frames) for key, frames in self.arrays.items()}
        payload["metadata_json"] = np.asarray(json.dumps(self.metadata, sort_keys=True))
        np.savez_compressed(destination, **payload)
        return destination


def load_wave_trace(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load a compressed trace without enabling pickle deserialization."""

    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files if key != "metadata_json"}
        metadata = json.loads(str(archive["metadata_json"].item()))
    return arrays, metadata
