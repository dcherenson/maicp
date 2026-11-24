"""Parallel collection of formation missions."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import os
from typing import Any

import numpy as np

from maicp.configuration import ExperimentConfig, Platform
from experiments.communication import GraphSnapshot
from maicp.score_stores import LocalCalibrationStores
from maicp.trace import WaveTraceRecorder
from maicp.types import MissionResult


_MISSION_WORKER_EXPERIMENT: Any | None = None


def _initialize_mission_worker(
    experiment_type: type[Any], config: ExperimentConfig
) -> None:
    global _MISSION_WORKER_EXPERIMENT
    # Load the frozen models inside each worker process.
    _MISSION_WORKER_EXPERIMENT = experiment_type(config)


def _run_mission_wave_worker(
    margins: dict[Platform, float],
    wave: int,
    first_mission_id: int,
    wave_seed: int,
    collect_local_calibration: bool,
) -> tuple[
    int,
    list[MissionResult],
    dict[str, tuple[str, ...]],
    bool,
    dict[str, Platform],
    LocalCalibrationStores | None,
]:
    """Simulate one wave and return the scores held by each node."""

    if _MISSION_WORKER_EXPERIMENT is None:  # pragma: no cover - process setup guard
        raise RuntimeError("Mission worker was not initialized")
    local = LocalCalibrationStores() if collect_local_calibration else None
    results, graph, platforms = _MISSION_WORKER_EXPERIMENT.run_wave(
        margins,
        first_mission_id=first_mission_id,
        wave_seed=wave_seed,
        local_calibration=local,
    )
    graph_payload = {
        node: tuple(sorted(neighbors)) for node, neighbors in graph.items()
    }
    return (
        wave,
        results,
        graph_payload,
        isinstance(graph, GraphSnapshot),
        platforms,
        local,
    )


@dataclass(frozen=True)
class CircularObstacle:
    center: np.ndarray
    radius: float
    platform: Platform | None = None
    obstacle_id: str | None = None


class CollisionAvoidanceExperiment:
    dt = 0.2
    alpha = 1.5
    activation_distance = 5.0
    pair_activation_distance = 2.0
    agent_radius = 0.12
    controller_paper_fidelity = False

    def __init__(self, config: ExperimentConfig):
        self.config = config
        if self.pair_activation_distance > config.task_communication_range:
            raise ValueError(
                "pair_activation_distance must not exceed task_communication_range"
            )

    def collect_missions(
        self,
        margins: dict[Platform, float],
        count: int,
        seed: int,
        trace_recorder: WaveTraceRecorder | None = None,
        progress_callback: Callable[[int], None] | None = None,
        local_calibration: LocalCalibrationStores | None = None,
    ) -> tuple[list[MissionResult], GraphSnapshot[str], dict[str, Platform]]:
        if count <= 0 or count % self.config.num_subteams:
            raise ValueError(
                "Mission count must be positive and divisible by the four subteams"
            )
        wave_count = count // self.config.num_subteams
        configured_workers = int(getattr(self.config, "mission_workers", 1))
        workers = configured_workers or min(8, max(1, os.cpu_count() or 1), wave_count)
        collect_local = local_calibration is not None
        completed: dict[
            int,
            tuple[
                list[MissionResult],
                GraphSnapshot[str],
                dict[str, Platform],
                LocalCalibrationStores | None,
            ],
        ] = {}

        pending_waves = list(range(wave_count))
        if trace_recorder is not None:
            traced_local = LocalCalibrationStores() if collect_local else None
            traced = self.run_wave(
                margins,
                first_mission_id=0,
                wave_seed=seed,
                trace_recorder=trace_recorder,
                local_calibration=traced_local,
            )
            completed[0] = (*traced, traced_local)
            pending_waves = pending_waves[1:]

        if workers == 1 or len(pending_waves) <= 1:
            for wave in pending_waves:
                wave_local = LocalCalibrationStores() if collect_local else None
                wave_result = self.run_wave(
                    margins,
                    first_mission_id=wave * self.config.num_subteams,
                    wave_seed=seed + 1_000_003 * wave,
                    local_calibration=wave_local,
                )
                completed[wave] = (*wave_result, wave_local)
        elif pending_waves:
            with ProcessPoolExecutor(
                max_workers=min(workers, len(pending_waves)),
                initializer=_initialize_mission_worker,
                initargs=(type(self), self.config),
            ) as executor:
                futures = [
                    executor.submit(
                        _run_mission_wave_worker,
                        dict(margins),
                        wave,
                        wave * self.config.num_subteams,
                        seed + 1_000_003 * wave,
                        collect_local,
                    )
                    for wave in pending_waves
                ]
                for future in futures:
                    (
                        wave,
                        wave_results,
                        graph_payload,
                        graph_is_snapshot,
                        wave_platforms,
                        wave_local,
                    ) = future.result()
                    graph = (
                        GraphSnapshot.from_adjacency(graph_payload)
                        if graph_is_snapshot
                        else {
                            node: set(neighbors)
                            for node, neighbors in graph_payload.items()
                        }
                    )
                    completed[wave] = (
                        wave_results,
                        graph,
                        wave_platforms,
                        wave_local,
                    )

        results: list[MissionResult] = []
        final_graph: GraphSnapshot[str] | None = None
        platforms: dict[str, Platform] | None = None
        for wave in range(wave_count):
            wave_results, final_graph, platforms, wave_local = completed[wave]
            # Keep scores and summary statistics. Discard the rendering data for individual steps.
            for mission in wave_results:
                detail = mission.audit.get("fixed_margin", {})
                for key in ("trajectory", "trajectory_time_s", "trajectory_step"):
                    detail.pop(key, None)
            results.extend(wave_results)
            if local_calibration is not None:
                assert wave_local is not None
                local_calibration.merge_from(wave_local)
            if progress_callback is not None:
                progress_callback(len(wave_results))
        assert final_graph is not None and platforms is not None
        return results, final_graph, platforms
