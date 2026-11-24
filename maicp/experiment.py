"""Interactive mission episodes shared by the MA-ICP applications."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from hashlib import sha256
from dataclasses import asdict, dataclass, field
from json import dump, dumps, load, loads
import gzip
from math import isfinite, log, sqrt
from pathlib import Path
import os
import sys
import tempfile
from time import perf_counter
from typing import Any, Protocol

import numpy as np
from scipy.stats import beta

from maicp.configuration import (
    CollisionController,
    ExperimentConfig,
    Platform,
    SensitivityMode,
)

from .algorithm import MAICP, _diagnostic_mission_score
from maicp.score_stores import LocalCalibrationStores
from .outcomes import summarize_robot_outcomes
from .single_agent import own_score_history, single_agent_conformal_quantile, single_agent_icp_update
from .trace import WaveTraceRecorder
from .types import MAICPRoundResult, MissionResult
from .update import normalized_margin_distance


@dataclass(frozen=True, eq=False)
class AnalyticSensitivityContext(Mapping[str, Platform]):
    """Robot classes, communication graph, margins, and controller parameters for a sensitivity calculation."""

    config: ExperimentConfig
    margins: Mapping[Platform, float]
    update_graph: Mapping[str, Sequence[str] | set[str] | frozenset[str]]
    platforms: Mapping[str, Platform]
    controller_context: object | None = None
    horizon: float | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Copy the mappings before the simulator advances.
        object.__setattr__(self, "margins", dict(self.margins))
        object.__setattr__(
            self,
            "update_graph",
            {
                str(node): tuple(sorted(neighbors, key=repr))
                for node, neighbors in self.update_graph.items()
            },
        )
        object.__setattr__(self, "platforms", dict(self.platforms))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def robot_ids(self) -> tuple[str, ...]:
        """Robot ordering used to assemble matrices for the full network."""

        return tuple(sorted(self.platforms, key=repr))

    def __getitem__(self, robot_id: str) -> Platform:
        return self.platforms[robot_id]

    def __iter__(self):
        return iter(self.platforms)

    def __len__(self) -> int:
        return len(self.platforms)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, AnalyticSensitivityContext):
            return (
                self.config == other.config
                and dict(self.margins) == dict(other.margins)
                and dict(self.update_graph) == dict(other.update_graph)
                and dict(self.platforms) == dict(other.platforms)
                and self.controller_context == other.controller_context
                and self.horizon == other.horizon
                and dict(self.metadata) == dict(other.metadata)
            )
        # Support comparisons with mappings from robots to platforms.
        if isinstance(other, Mapping):
            return dict(self.platforms) == dict(other)
        return NotImplemented


# Short alias for application code that uses the kappa terminology.
AnalyticKappaContext = AnalyticSensitivityContext


def build_analytic_sensitivity_context(
    experiment: object,
    margins: Mapping[Platform, float],
    update_graph: Mapping[str, Sequence[str] | set[str] | frozenset[str]],
    platforms: Mapping[str, Platform],
) -> AnalyticSensitivityContext:
    """Collect the parameters needed to evaluate an application's analytical sensitivity bound."""

    custom_builder = getattr(experiment, "build_analytic_sensitivity_context", None)
    if callable(custom_builder):
        candidate = custom_builder(
            margins=dict(margins),
            update_graph=update_graph,
            platforms=dict(platforms),
        )
        if not isinstance(candidate, AnalyticSensitivityContext):
            raise TypeError(
                "build_analytic_sensitivity_context must return "
                "AnalyticSensitivityContext"
            )
        return candidate

    controller_name = getattr(experiment, "controller_name", None)
    metadata: dict[str, object] = {
        "application": type(experiment).__name__,
        "controller": str(controller_name) if controller_name is not None else None,
    }
    configured_controller = getattr(experiment.config, "collision_controller", None)
    if configured_controller is not None:
        metadata["collision_controller"] = (
            configured_controller.value
            if isinstance(configured_controller, CollisionController)
            else str(configured_controller)
        )
    experiment_metadata = getattr(experiment, "experiment_metadata", None)
    if isinstance(experiment_metadata, Mapping):
        metadata["experiment_metadata"] = dict(experiment_metadata)
    return AnalyticSensitivityContext(
        config=experiment.config,
        margins=dict(margins),
        update_graph=update_graph,
        platforms=dict(platforms),
        controller_context=getattr(experiment, "analytic_controller_context", None),
        horizon=getattr(experiment, "sensitivity_horizon", None),
        metadata=metadata,
    )


def normalize_analytic_kappa(
    context: AnalyticSensitivityContext,
    result: object,
) -> tuple[dict[str, float], dict[str, object]]:
    """Validate the supplied sensitivity coefficients and flag missing robot bounds."""

    raw_mapping: object
    if isinstance(result, Mapping):
        raw_mapping = result
    else:
        raw_mapping = getattr(result, "kappas", None)
        if raw_mapping is None:
            raw_mapping = getattr(result, "robot_kappa", None)
    if not isinstance(raw_mapping, Mapping):
        raw_mapping = {}

    expected = tuple(context.robot_ids)
    missing = sorted(set(expected).difference(raw_mapping), key=repr)
    unknown = sorted(set(raw_mapping).difference(expected), key=repr)
    invalid: list[str] = []
    values: dict[str, float] = {}
    for robot_id in expected:
        if robot_id not in raw_mapping:
            values[robot_id] = float("nan")
            continue
        try:
            value = float(raw_mapping[robot_id])
        except (TypeError, ValueError):
            value = float("nan")
        if not isfinite(value) or value < 0.0:
            invalid.append(robot_id)
            values[robot_id] = float("nan")
        else:
            values[robot_id] = value

    explicit_available = getattr(result, "available", None)
    explicit_certified = getattr(result, "certified", None)
    available = (
        bool(explicit_available)
        if explicit_available is not None
        else not missing and not invalid and bool(expected)
    )
    certified = (
        bool(explicit_certified)
        if explicit_certified is not None
        else available
    )
    blocker = getattr(result, "blocker", None)
    if blocker is None:
        blocker = getattr(result, "reason", None)
    if not missing and not invalid and available and certified:
        status = str(getattr(result, "status", "available"))
    else:
        # Explicitly invalidate all values when a result advertises an
        # unavailable or uncertified derivation, even if it carries diagnostic
        # finite values alongside the blocker.
        if not available or not certified:
            for robot_id in expected:
                values[robot_id] = float("nan")
        if blocker is None:
            reasons: list[str] = []
            if missing:
                reasons.append("missing robot coefficient")
            if invalid:
                reasons.append("nonfinite or negative robot coefficient")
            if not available:
                reasons.append("analytic sensitivity unavailable")
            if not certified:
                reasons.append("analytic sensitivity is not certified")
            blocker = "; ".join(reasons) or "analytic sensitivity unavailable"
        status = str(getattr(result, "status", "unavailable"))

    metadata: dict[str, object] = {
        "status": status,
        "available": bool(available and not missing and not invalid),
        "certified": bool(certified and not missing and not invalid),
        "blocker": str(blocker) if blocker is not None else None,
        "missing_robots": missing,
        "invalid_robots": sorted(invalid, key=repr),
        "unknown_robots": unknown,
    }
    extra = getattr(result, "metadata", None)
    if isinstance(extra, Mapping):
        metadata["details"] = dict(extra)
    return values, metadata


class ApplicationExperiment(Protocol):
    config: ExperimentConfig

    def collect_missions(
        self,
        margins: Mapping[Platform | str, float],
        count: int,
        seed: int,
        trace_recorder: WaveTraceRecorder | None = None,
        progress_callback: Callable[[int], None] | None = None,
        local_calibration: LocalCalibrationStores | None = None,
    ) -> tuple[list[MissionResult], dict[str, set[str]], dict[str, Platform]]: ...

    def run_wave(
        self,
        margins: Mapping[Platform | str, float],
        first_mission_id: int,
        wave_seed: int,
        trace_recorder: WaveTraceRecorder | None = None,
        local_calibration: LocalCalibrationStores | None = None,
    ) -> tuple[list[MissionResult], dict[str, set[str]], dict[str, Platform]]: ...

    def analytic_kappa(
        self, context: AnalyticSensitivityContext
    ) -> Mapping[str, float] | object: ...


@dataclass(frozen=True)
class _EpisodeBatch:
    """Mission observations used for evaluation and the next calibration update."""

    missions: tuple[MissionResult, ...]
    update_graph: Mapping[str, Sequence[str] | set[str] | frozenset[str]]
    robot_platforms: Mapping[str, Platform]
    local_calibration: LocalCalibrationStores
    margins: Mapping[Platform | str, float]
    collection_seed: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "missions", tuple(self.missions))
        object.__setattr__(
            self,
            "update_graph",
            {
                node: frozenset(neighbors)
                for node, neighbors in self.update_graph.items()
            },
        )
        object.__setattr__(self, "robot_platforms", dict(self.robot_platforms))
        object.__setattr__(self, "margins", dict(self.margins))


def _batch_audit(
    missions: Sequence[MissionResult],
    margins: Mapping[Platform | str, float],
    seed: int,
    *,
    source: str,
    used_for_next_margin: bool,
) -> dict[str, object]:
    return {
        "global_mission_failures": sum(mission.failed for mission in missions),
        "missions_with_robot_failures": sum(
            any(mission.failed_robots.get(platform, ()) for platform in Platform)
            for mission in missions
        ),
        "missions_with_failure_budget_exceeded": sum(
            any(
                mission.metrics.get(f"{platform.value}_failure_budget_exceeded", 0)
                for platform in Platform
            )
            for mission in missions
        ),
        "source": source,
        "collection_seed": int(seed),
        "mission_ids": [mission.mission_id for mission in missions],
        "mission_seeds": [mission.seed for mission in missions],
        "margins": _jsonable(dict(margins)),
        "exogenous_realization_digests": _realization_digests(missions),
        "used_for_next_margin": bool(used_for_next_margin),
        "mission_outcomes": [
            {
                "mission_id": mission.mission_id,
                "robot_scores": {p.value: dict(mission.robot_scores[p]) for p in Platform},
                "global_failure": bool(mission.failed),
                "global_failure_reason": mission.failure_reason,
                "returned_home": bool(mission.returned_home),
                "update_graph_connected": bool(mission.update_graph_connected),
                "failed_robots": {
                    platform.value: list(mission.failed_robots.get(platform, ()))
                    for platform in Platform
                },
                "robot_failure_reasons": {
                    robot_id: list(reasons)
                    for robot_id, reasons in mission.robot_failure_reasons.items()
                },
                "failure_budget_exceeded": {
                    platform.value: bool(
                        mission.metrics.get(
                            f"{platform.value}_failure_budget_exceeded", 0.0
                        )
                    )
                    for platform in Platform
                },
                "maximum_acceleration_model_error": mission.metrics.get(
                    "maximum_acceleration_model_error", float("nan")
                ),
                "failed_robot_count": {
                    platform.value: len(mission.failed_robots.get(platform, ()))
                    for platform in Platform
                },
                "local_qp_failures": int(
                    mission.metrics.get("local_qp_failures", 0)
                ),
                "obstacle_collisions": int(
                    mission.metrics.get("obstacle_collision_count", 0)
                ),
                "pair_collisions": int(
                    mission.metrics.get("pair_collision_count", 0)
                ),
                "wall_collisions": int(
                    mission.metrics.get("wall_collision_count", 0)
                ),
            }
            for mission in missions
        ],
    }


def _one_sided_binomial_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    if trials <= 0:
        return float("nan"), float("nan")
    lower = 0.0 if successes == 0 else float(beta.ppf(1.0 - confidence, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(beta.ppf(confidence, successes + 1, trials - successes))
    return lower, upper


def _format_terminal_value(value: float) -> str:
    if not isfinite(value):
        return "+inf" if value > 0 else ("-inf" if value < 0 else "nan")
    return f"{value:.4g}"


def _print_round_summary(
    application_name: str,
    method: str,
    replicate: int,
    round_index: int,
    config: ExperimentConfig,
    result: MAICPRoundResult,
) -> None:
    quantile_label = "q_exact" if method == "centralized" else "q_dist"
    quantiles = ", ".join(
        f"{platform.value}={_format_terminal_value(result.certified_thresholds[platform])}"
        for platform in Platform
    )
    margins = ", ".join(
        f"{platform.value}={_format_terminal_value(result.margin_update.margins[platform])}"
        for platform in Platform
    )
    kappas = ", ".join(
        f"{platform.value}={_format_terminal_value(0.0 if method == 'unshifted' else result.sensitivity.class_kappa[platform])}"
        for platform in Platform
    )
    print(
        f"[MA-ICP {application_name}] method={method} "
        f"replicate={replicate + 1}/{config.profile.outer_replicates} "
        f"episode={round_index}/{config.profile.rounds} "
        f"{quantile_label}=({quantiles}) "
        f"kappa=({kappas}) r_next=({margins}) "
        f"status={result.margin_update.status}",
        flush=True,
    )


class _MissionBatchProgress:
    """Small stderr progress bar advanced once each parallel mission wave finishes."""

    def __init__(self, label: str, total: int, width: int = 24):
        self.label = label
        self.total = total
        self.width = width
        self.completed = 0
        self._render()

    def advance(self, completed: int) -> None:
        self.completed = min(self.total, self.completed + completed)
        self._render()

    def _render(self) -> None:
        fraction = self.completed / self.total if self.total else 1.0
        filled = round(self.width * fraction)
        bar = "=" * filled + ">" * (filled < self.width) + " " * max(0, self.width - filled - 1)
        sys.stderr.write(
            f"\r{self.label} [{bar}] {self.completed}/{self.total} missions"
        )
        sys.stderr.flush()

    def finish(self) -> None:
        sys.stderr.write("\n")
        sys.stderr.flush()


def class_mission_scores(
    missions: Sequence[MissionResult], config: ExperimentConfig
) -> dict[Platform, list[float]]:
    # Record an unavailable mission score as positive infinity in the coverage summary.
    result: dict[Platform, list[float]] = {platform: [] for platform in Platform}
    for platform in Platform:
        for mission in missions:
            value = _diagnostic_mission_score(
                tuple(mission.robot_scores.get(platform, {}).values()),
                config.classes[platform].mission_rank,
            )
            result[platform].append(value if isfinite(value) else float("inf"))
    return result


def deployment_metrics(
    missions: Sequence[MissionResult],
    margins: Mapping[Platform | str, float],
    config: ExperimentConfig,
    certified: bool,
    *,
    coverage_radii: Mapping[Platform | str, float] | None = None,
) -> dict[str, Any]:
    thresholds = margins if coverage_radii is None else coverage_radii
    robot_specific = not all(platform in thresholds for platform in Platform)
    tracking_uncertainty = bool(missions) and all("uncertainty" in mission.audit for mission in missions)
    tracking_excess = tracking_uncertainty and all(
        mission.audit.get("fixed_margin", {}).get("score_semantics") == "drwt_active_valid_nominal_excess_v2"
        for mission in missions
    )
    if tracking_uncertainty and coverage_radii is None and any(
        mission.audit["uncertainty"].get("coverage_margin_mode") == "robot"
        for mission in missions
    ):
        # Use each robot's reported ellipse when aggregating areas.
        thresholds = {}
        for mission in missions:
            for p in Platform:
                q0 = sqrt(-2.0 * log(config.classes[p].alpha))
                for robot in mission.robot_scores[p]:
                    radius = mission.audit["uncertainty"]["robots"][robot]["reported_radius"]
                    value = max(0.0, radius - q0) if tracking_excess else radius
                    if robot in thresholds and thresholds[robot] != value:
                        raise ValueError("A deployment batch must use one reporting radius per robot")
                    thresholds[robot] = value
        robot_specific = True
    elif tracking_uncertainty and coverage_radii is None:
        # A directly collected tracking batch may report native ellipses even
        # with nonzero internal margins. Use its actual reporting radii.
        thresholds = {
            p: missions[0].audit["uncertainty"]["classes"][p.value]["reported_radius"]
            for p in Platform
        }
        if any(
            mission.audit["uncertainty"]["classes"][p.value]["reported_radius"] != thresholds[p]
            for mission in missions for p in Platform
        ):
            raise ValueError("A deployment batch must use one reporting radius per class")
        if tracking_excess:
            thresholds = {p: max(0.0, thresholds[p] - sqrt(-2.0 * log(config.classes[p].alpha)))
                          for p in Platform}
        robot_specific = False
    scores = class_mission_scores(missions, config)
    metrics: dict[str, Any] = {
        "certified_summary_included": certified,
        "missions": len(missions),
        "mission_failures": sum(mission.failed for mission in missions),
        "missions_with_robot_failures": sum(
            any(mission.failed_robots.get(platform, ()) for platform in Platform)
            for mission in missions
        ),
        "missions_with_failure_budget_exceeded": sum(
            any(
                mission.metrics.get(f"{platform.value}_failure_budget_exceeded", 0)
                for platform in Platform
            )
            for mission in missions
        ),
        "return_failures": sum(
            not mission.returned_home
            and mission.audit.get("fixed_margin", {}).get("return_required", True)
            for mission in missions
        ),
        "connectivity_failures": sum(not mission.update_graph_connected for mission in missions),
        "application_metrics": {},
    }
    audited = [
        {
            "mission_id": mission.mission_id,
            "subteam_id": mission.subteam_id,
            "seed": mission.seed,
            **mission.audit,
            "robot_scores": {p.value: dict(mission.robot_scores[p]) for p in Platform},
            "failed_robots": {p.value: list(mission.failed_robots.get(p, ())) for p in Platform},
            "robot_failure_reasons": {robot: list(reasons) for robot, reasons in mission.robot_failure_reasons.items()},
        }
        for mission in missions
    ]
    if audited:
        metrics["mission_audit"] = audited
    for platform in Platform:
        values = scores[platform]
        def covered(robot_id: str, score: float) -> bool:
            radius = thresholds[robot_id] if robot_specific else thresholds[platform]
            return isfinite(score) and score <= radius

        successes = (
            sum(
                sum(covered(robot_id, score) for robot_id, score in mission.robot_scores[platform].items())
                >= config.classes[platform].mission_rank
                for mission in missions
            )
            if robot_specific
            else sum(isfinite(score) and score <= thresholds[platform] for score in values)
        )
        robot_samples = [
            (robot_id, score)
            for mission in missions
            for robot_id, score in mission.robot_scores[platform].items()
        ]
        robot_score_covered = sum(covered(robot_id, score) for robot_id, score in robot_samples)
        robot_coverage = {}
        for robot_id in sorted({robot_id for robot_id, _ in robot_samples}):
            own = [score for owner, score in robot_samples if owner == robot_id]
            own_successes = sum(covered(robot_id, score) for score in own)
            robot_coverage[robot_id] = {
                "margin": thresholds[robot_id] if robot_specific else thresholds[platform],
                "successes": own_successes, "trials": len(own),
                "coverage": own_successes / len(own),
            }
        lower, upper = _one_sided_binomial_interval(successes, len(values))
        metrics[platform.value] = {
            "coverage": successes / len(values),
            "successes": successes,
            "trials": len(values),
            # Compatibility alias: these successes mean conformal score coverage.
            "robot_successes": robot_score_covered,
            "robot_score_covered": robot_score_covered,
            "robot_score_coverage": robot_score_covered / len(robot_samples),
            "robot_trials": len(robot_samples),
            "per_robot_score_coverage": robot_coverage,
            "mission_rank": config.classes[platform].mission_rank,
            "coverage_event": "at least K robots satisfy their own S_i <= r_i" if robot_specific else "K-th ordered robot score <= class margin",
            "binomial_interval_95": [lower, upper],
            "maximum_score": max(values),
            "rms_score": float(np.sqrt(np.mean(np.square(values)))),
        }
        physical = []
        for mission in missions:
            robots = mission.audit.get("fixed_margin", {}).get("robots", {})
            ids = tuple(mission.robot_scores[platform])
            if not all(robot in robots and "collision_flags" in robots[robot] for robot in ids):
                continue
            safe_count = sum(not robots[robot]["collision_flags"]["obstacle"] for robot in ids)
            physical.append(safe_count >= config.classes[platform].mission_rank)
        metrics[platform.value]["physical_obstacle_safety"] = {
            "K": config.classes[platform].mission_rank,
            "successes": sum(physical), "trials": len(physical),
            "coverage": sum(physical) / len(physical) if physical else None,
            "event": "at least K robots avoid every obstacle over the mission",
        }
        if tracking_uncertainty:
            native_radius = sqrt(-2.0 * log(config.classes[platform].alpha))
            records = [
                mission.audit["uncertainty"]["classes"][platform.value]
                for mission in missions
            ]
            samples = sum(item["samples"] for item in records)
            native_successes = sum(score <= (0.0 if tracking_excess else native_radius) for score in values)
            reported_radii = {
                robot: (native_radius + item["margin"] if tracking_excess else item["margin"])
                for robot, item in robot_coverage.items()
            }
            unique_radii = set(reported_radii.values())
            reported_radius = next(iter(unique_radii)) if len(unique_radii) == 1 else None
            native_area = sum(item["native_area_sum"] for item in records)
            reported_area = sum(item["reported_area_sum"] for item in records)
            metrics[platform.value]["uncertainty"] = {
                "coverage_radius": reported_radius,
                "native_coverage_radius": native_radius,
                "native_mission_coverage": native_successes / len(values),
                "reported_mission_coverage": successes / len(values),
                "samples": samples,
                "native_sample_coverage": (
                    sum(item["native_covered_samples"] for item in records) / samples
                    if samples else None
                ),
                "reported_sample_coverage": (
                    sum(item["reported_covered_samples"] for item in records) / samples
                    if samples else None
                ),
                "native_mean_ellipse_area": native_area / samples if samples else None,
                "reported_mean_ellipse_area": reported_area / samples if samples else None,
                "covariance_scale": None if reported_radius is None else reported_radius ** 2 / native_radius ** 2,
                "ellipse_axis_scale": None if reported_radius is None else reported_radius / native_radius,
                "sample_coverage_interpretation": "Descriptive correlated robot-time samples; not independent trials.",
                "mission_coverage_event": "At least n_c-v_c robots covered at every scored task sample; failed missions count as misses.",
            }
            if robot_specific:
                metrics[platform.value]["uncertainty"].update(
                    reporting_scope="physical_robot",
                    coverage_radii_by_robot=reported_radii,
                    coverage_radius_range=[min(unique_radii), max(unique_radii)],
                    covariance_scales_by_robot={robot: radius**2 / native_radius**2
                                               for robot, radius in reported_radii.items()},
                )
    outcomes = summarize_robot_outcomes(
        [mission.failed_robots for mission in missions],
        [mission.robot_failure_reasons for mission in missions],
        {p.value: metrics[p.value]["robot_trials"] for p in Platform},
    )
    for platform in Platform:
        metrics[platform.value].update(outcomes[platform.value])
    metrics["robot_failure_reason_counts"] = outcomes["robot_failure_reason_counts"]
    metric_names = sorted({name for mission in missions for name in mission.metrics})
    for name in metric_names:
        values = [
            float(mission.metrics[name]) for mission in missions
            if isinstance(mission.metrics.get(name), (int, float, np.number))
        ]
        if not values:
            continue
        finite = [value for value in values if isfinite(value)]
        metrics["application_metrics"][name] = {
            "mean": float(np.mean(finite)) if finite else float("nan"),
            "maximum": max(finite) if finite else float("nan"),
        }
    robot_audits = [
        (mission, robot_id, item)
        for mission in missions
        if "uncertainty" not in mission.audit
        for robot_id, item in mission.audit.get("fixed_margin", {}).get("robots", {}).items()
    ]
    metrics["qp_failures"] = sum(int(item["qp_failure_count"]) for _, _, item in robot_audits)
    metrics["fallbacks"] = sum(int(item["qp_fallback_count"]) for _, _, item in robot_audits)
    covered_collision_records = [
        item for mission, robot_id, item in robot_audits
        if isfinite(mission.robot_scores[Platform(item["platform"])][robot_id])
        and mission.robot_scores[Platform(item["platform"])][robot_id]
        <= (thresholds[robot_id] if robot_specific else thresholds[Platform(item["platform"])])
        and item["collision_flags"]["obstacle"]
    ]
    metrics["covered_obstacle_collisions"] = len(covered_collision_records)
    metrics["covered_obstacle_collision_events"] = sum(
        int(item["collision_event_counts"]["obstacle"]) for item in covered_collision_records
    )
    return metrics


def paired_sensitivity_ratios(
    experiment: ApplicationExperiment,
    current: dict[Platform, float],
    seed: int,
    step: float = 0.05,
    *,
    diagnostics: dict[str, object] | None = None,
) -> dict[str, float]:
    """Largest score change per unit margin distance in missions with the same random inputs.

    An optional diagnostics mapping retains the same paired runs'
    results/traces and individual score changes, without another simulation or
    changing any coefficient, random stream, or calibration/update decision.
    """

    candidate = {
        platform: min(
            experiment.config.classes[platform].margin_max,
            current[platform] + step * experiment.config.classes[platform].scale,
        )
        for platform in Platform
    }
    distance = normalized_margin_distance(candidate, current, experiment.config.classes)
    if distance == 0.0:
        return {}
    baseline_local = LocalCalibrationStores()
    perturbed_local = LocalCalibrationStores()
    traces = [WaveTraceRecorder(context={"margins": _jsonable(margins)})
              for margins in (current, candidate)] if diagnostics is not None else [None, None]
    baseline_results, _, _ = experiment.run_wave(
        current, 0, seed, local_calibration=baseline_local,
        **({"trace_recorder": traces[0]} if diagnostics is not None else {}),
    )
    perturbed_results, _, _ = experiment.run_wave(
        candidate, 0, seed, local_calibration=perturbed_local,
        **({"trace_recorder": traces[1]} if diagnostics is not None else {}),
    )
    observations = []
    ratios: dict[str, float] = {}
    if set(baseline_local.node_stores) != set(perturbed_local.node_stores):
        raise ValueError("Paired runs have different robot owners")
    for robot_id, node_a in baseline_local.node_stores.items():
        node_b = perturbed_local.node_stores[robot_id]
        for platform in Platform:
            if set(node_a.robot_scores[platform]) != set(node_b.robot_scores[platform]):
                raise ValueError("Paired runs have different mission owners")
            for mission_id, tagged_a in node_a.robot_scores[platform].items():
                tagged_b = node_b.robot_scores[platform][mission_id]
                if set(tagged_a) != set(tagged_b) or len(tagged_a) != 1:
                    raise ValueError("Paired runs have different score tags")
                tag = next(iter(tagged_a))
                score_a, score_b = tagged_a[tag], tagged_b[tag]
                ratio = (
                    abs(score_b - score_a) / distance
                    if isfinite(score_a) and isfinite(score_b)
                    else float("inf")
                )
                ratios[robot_id] = max(ratios.get(robot_id, 0.0), ratio)
                if diagnostics is not None:
                    observations.append(dict(robot_id=robot_id, platform=platform.value,
                        mission_id=mission_id, baseline_score=score_a, perturbed_score=score_b,
                        absolute_score_change=abs(score_b-score_a) if isfinite(score_a) and isfinite(score_b) else float("inf"),
                        sensitivity_ratio=ratio))
    if diagnostics is not None:
        diagnostics.update(baseline_results=baseline_results, perturbed_results=perturbed_results,
                           traces=traces, score_observations=observations,
                           current_margins=dict(current), perturbed_margins=candidate,
                           normalized_margin_distance=distance)
    return ratios


def _round_record(result: MAICPRoundResult) -> dict[str, Any]:
    return {
        "round": result.round_index,
        "mission_scores": result.mission_scores,
        "certified_mission_scores": result.certified_mission_scores,
        "first_mission_exact": {
            platform: scores[0] for platform, scores in result.mission_scores.items()
        },
        "first_mission_certified": {
            platform: scores[0] for platform, scores in result.certified_mission_scores.items()
        },
        "exact_thresholds": result.exact_thresholds,
        "certified_thresholds": result.certified_thresholds,
        "conformal_ranks": result.conformal_ranks,
        "kappa": result.sensitivity.class_kappa,
        "sensitivity": asdict(result.sensitivity),
        "margin_update": asdict(result.margin_update),
        "communication_rounds": result.communication_rounds,
        "total_communication_rounds": result.communication_rounds,
        "pinball_rounds": result.pinball_rounds,
        "flooding_rounds": result.flooding_rounds,
        "order_statistic_flooding_rounds": result.order_statistic_flooding_rounds,
        "kappa_flooding_rounds": result.kappa_flooding_rounds,
        "margin_consensus_residual": result.margin_update.margin_consensus_residual,
        "mission_certificate_owners": result.mission_certificate_owners,
        "order_statistic_histories": {
            platform: {name: values.tolist() for name, values in stages.items()}
            for platform, stages in result.order_statistic_histories.items()
        },
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Platform):
        return value.value
    if isinstance(value, dict):
        return {str(_jsonable(key)): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not isfinite(value):
        return "+inf" if value > 0 else ("-inf" if value < 0 else "nan")
    return value


def _sync_checkpoint_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: Any) -> None:
    """Replace a compact checkpoint only after its complete contents reach disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            dump(_jsonable(value), stream, indent=2, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_checkpoint_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_mission_audits(path: Path, audits: Sequence[Mapping[str, Any]]) -> None:
    """Write compressed mission records one at a time and atomically replace the output file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=3) as stream:
            for audit in audits:
                dump(_jsonable(audit), stream, separators=(",", ":"), allow_nan=False)
                stream.write("\n")
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_checkpoint_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def iter_mission_audits(deployment: Mapping[str, Any], result_root: str | Path):
    """Read old inline audits or new sidecars without loading an entire batch."""
    if "mission_audit" in deployment:
        yield from deployment["mission_audit"]
        return
    with gzip.open(Path(result_root) / deployment["mission_audit_file"], "rt",
                   encoding="utf-8") as stream:
        for line in stream:
            yield loads(line)


def _checkpoint_signature(experiment: ApplicationExperiment, application: str,
                          methods: Sequence[str]) -> str:
    root = Path(__file__).resolve().parent.parent
    cases = root / "experiments"
    prefix = "tracking" if application in {"tracking", "target_tracking"} else "formation"
    sources = list((root / "maicp").glob("*.py"))
    sources.extend((cases / prefix).glob("*.py"))
    sources.extend(cases / name for name in ("mission.py", "communication.py", "vehicles.py", "sensors.py"))
    if prefix == "formation":
        sources.append(cases / prefix / "models.json")
    hashes = {str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest()
              for path in sources if path.name != "visualization.py" and path.name != "plots.py"}
    payload = {"configuration": asdict(experiment.config), "application": application,
               "methods": list(methods), "sources": hashes,
               "metadata": getattr(experiment, "experiment_metadata", None)}
    return sha256(dumps(_jsonable(payload), sort_keys=True).encode()).hexdigest()


def _realization_digests(missions: Sequence[MissionResult]) -> dict[str, str | None]:
    return {
        f"{mission.subteam_id}:{mission.mission_id}": mission.audit.get(
            "exogenous_realization_digest", mission.metrics.get("exogenous_realization_digest")
        )
        for mission in missions
    }


def _episode_batch_digest(batch: _EpisodeBatch) -> str:
    payload = {
        "seed": batch.collection_seed,
        "realizations": _realization_digests(batch.missions),
        "missions": [
            {"mission_id": mission.mission_id, "subteam_id": mission.subteam_id,
             "seed": mission.seed, "scores": mission.robot_scores}
            for mission in batch.missions
        ],
    }
    return sha256(dumps(_jsonable(payload), sort_keys=True, allow_nan=False).encode()).hexdigest()


def _validate_episode_margins(
    margins: Mapping[Platform | str, float],
    platforms: Mapping[str, Platform],
    config: ExperimentConfig,
) -> None:
    class_keyed = all(platform in margins for platform in Platform)
    expected_keys = set(Platform) if class_keyed else set(platforms)
    if set(margins) != expected_keys:
        raise ValueError("Episode margins must contain exactly both classes or all robot IDs")
    for robot, platform in platforms.items():
        value = margins[platform] if class_keyed else margins[robot]
        cls = config.classes[platform]
        if not isfinite(value) or not cls.margin_min <= value <= cls.margin_max:
            raise ValueError(f"Episode margin for {robot} is nonfinite or outside its class bounds")


def _central_class_quantiles(
    missions: Sequence[MissionResult], config: ExperimentConfig,
) -> tuple[dict[Platform, float | None], dict[Platform, int | None]]:
    scores = class_mission_scores(missions, config)
    quantiles = {p: single_agent_conformal_quantile(scores[p], config.classes[p]) for p in Platform}
    return ({p: item["q"] for p, item in quantiles.items()},
            {p: item["rank"] for p, item in quantiles.items()})


def _collect_episode(
    experiment: ApplicationExperiment, margins: Mapping[Platform | str, float],
    seed: int, label: str, trace: WaveTraceRecorder | None, *, tracking: bool,
) -> _EpisodeBatch:
    local = LocalCalibrationStores()
    count = experiment.config.profile.calibration_missions
    progress = _MissionBatchProgress(label, count)
    try:
        missions, graph, platforms = experiment.collect_missions(
            dict(margins), count, seed, trace_recorder=trace,
            progress_callback=progress.advance, local_calibration=local,
            **({"coverage_radii": margins} if tracking else {}),
        )
    finally:
        progress.finish()
    if len(missions) != count:
        raise ValueError(f"Expected {count} episode missions, received {len(missions)}")
    _validate_episode_margins(margins, platforms, experiment.config)
    return _EpisodeBatch(tuple(missions), graph, platforms, local, dict(margins), seed)


def _episode_sensitivity(
    experiment: ApplicationExperiment, batch: _EpisodeBatch,
    local: LocalCalibrationStores, report: dict[str, Any],
    method: str, replicate: int, episode: int,
) -> dict[str, float]:
    config = experiment.config
    analytic_values: dict[str, float] = {}
    if config.sensitivity_mode is SensitivityMode.ANALYTIC:
        context = build_analytic_sensitivity_context(
            experiment, batch.margins, batch.update_graph, batch.robot_platforms,
        )
        analytic_values, metadata = normalize_analytic_kappa(context, experiment.analytic_kappa(context))
        report["analytic_sensitivity"]["calls"].append(
            {"method": method, "replicate": replicate, "episode": episode, **metadata}
        )
        report["analytic_sensitivity"].update(metadata)
    for robot, platform in batch.robot_platforms.items():
        local.record_configured_kappa(robot, config.classes[platform].configured_kappa)
        if analytic_values:
            local.record_analytic_kappa(robot, analytic_values[robot])
    # Empirical mode requires sensitivity estimates supplied by the application.
    return analytic_values


def run_nested_experiment(
    experiment: ApplicationExperiment,
    application_name: str,
    methods: Sequence[str] = ("distributed", "centralized", "unshifted", "fixed", "single_agent"),
    output_root: str | Path = "results",
    render_visuals: bool = True,
    resume: bool = False,
    save_paired_trace: bool = False,
) -> dict[str, Any]:
    """Run the initial batch under shared margins, then calibrate and evaluate each method on successive batches. Mission seeds depend on the replicate and episode."""
    allowed = {"distributed", "centralized", "unshifted", "fixed", "single_agent", "nominal"}
    if not methods or len(set(methods)) != len(methods) or set(methods) - allowed:
        raise ValueError("Comparison methods must be a nonempty unique sequence of supported names")
    config = experiment.config
    tracking = application_name in {"tracking", "target_tracking"}
    if "single_agent" in methods and not getattr(experiment, "supports_robot_margins", False):
        raise ValueError("single_agent requires an application that supports robot-specific margins")
    result_root = Path(output_root) / application_name
    collision_controller = None
    if application_name in {"collision", "collision_avoidance"}:
        collision_controller = CollisionController(config.collision_controller)
        result_root /= collision_controller.value
    result_root /= Path(config.profile.name) / config.uav_dynamics.value / config.sensitivity_mode.value
    result_root.mkdir(parents=True, exist_ok=True)
    destination = result_root / "metrics.json"
    manifest_path = result_root / "checkpoint_manifest.json"
    signature = _checkpoint_signature(experiment, application_name, methods)
    initial = {p: config.classes[p].initial_margin for p in Platform}
    report: dict[str, Any] = {
        "application": application_name, "profile": config.profile.name,
        "uav_dynamics": config.uav_dynamics.value, "sensitivity_mode": config.sensitivity_mode.value,
        "methods": list(methods), "class_alpha": {p.value: config.classes[p].alpha for p in Platform},
        "class_requirements": {p.value: {**asdict(config.classes[p]), "mission_rank": config.classes[p].mission_rank} for p in Platform},
        "profile_rounds": config.profile.rounds, "episode_count": config.profile.rounds + 1,
        "initial_margins": initial,
        "episode_semantics": {
            "version": 1, "indices": list(range(config.profile.rounds + 1)),
            "episode_0": "one common execution batch, reused exactly by all methods",
            "transition": "execute episode j -> calibrate on its same data -> margins for j+1",
            "terminal_episode": "execution and evaluation; no margin update",
            "exogenous_seed_key": "replicate and episode only; no method or margin",
            "separate_calibration_or_deployment_batches": False,
            "separate_fixed_pilot": False,
        },
        "iterative_batch_policy": {
            "configured_calibration_missions": config.profile.calibration_missions,
            "configured_deployment_missions": config.profile.deployment_missions,
            "episode_missions": config.profile.calibration_missions,
            "final_iterative_batch_is_evaluation_only": True,
        },
        "analytic_sensitivity": {"status": "not_requested", "available": False,
                                 "certified": False, "blocker": None, "calls": []},
        "sensitivity_diagnostics": "no extra mission batches in the episode protocol",
        "replicates": [],
    }
    if collision_controller is not None:
        report["collision_controller"] = collision_controller.value
    if getattr(experiment, "controller_name", None) is not None:
        report["controller"] = str(experiment.controller_name)
        report["controller_paper_fidelity"] = bool(getattr(experiment, "controller_paper_fidelity", False))
    metadata = getattr(experiment, "experiment_metadata", None)
    if isinstance(metadata, Mapping):
        report["experiment_metadata"] = dict(metadata)
    report["output_file"] = str(destination.resolve())
    report["run_status"] = "running"
    report["checkpoint_signature"] = signature
    report["storage"] = {
        "schema_version": 2,
        "mission_audits": "compressed JSONL sidecars; one physical batch per file",
        "round_checkpoints": "atomic compact JSON after each method/episode",
    }
    if resume:
        if not manifest_path.exists():
            raise ValueError("No round checkpoint manifest; this output cannot be resumed")
        with manifest_path.open(encoding="utf-8") as stream:
            manifest = load(stream)
        if manifest.get("signature") != signature:
            raise ValueError("Resume requires the same configuration, methods, and scientific source files")
        if destination.exists():
            with destination.open(encoding="utf-8") as stream:
                report = load(stream)
            if report.get("checkpoint_signature") != signature:
                raise ValueError("Metrics and checkpoint manifest do not match")
            for saved_replicate in report["replicates"]:
                for saved_records in saved_replicate["methods"].values():
                    for saved in saved_records:
                        audit_file = result_root / saved["deployment"]["mission_audit_file"]
                        if not audit_file.is_file():
                            raise ValueError(f"Missing saved mission audits: {audit_file}")
            report["output_file"] = str(destination.resolve())
            if report.get("run_status") == "completed":
                if render_visuals and "visualizations" not in report:
                    from .visualization import render_experiment_report
                    report["visualizations"] = render_experiment_report(report, result_root)
                    _atomic_json(destination, report)
                return report
    else:
        _atomic_json(manifest_path, {"schema_version": 2, "signature": signature})
    _atomic_json(destination, report)
    algorithm = MAICP(config)
    for replicate in range(config.profile.outer_replicates):
        if replicate < len(report["replicates"]):
            replicate_report = report["replicates"][replicate]
        else:
            replicate_report = {"replicate": replicate,
                                "methods": {method: [] for method in methods},
                                "protocol_validation": {"pairing": []}}
            report["replicates"].append(replicate_report)
        records = replicate_report["methods"]
        states: dict[str, dict[Platform | str, float]] = {method: dict(initial) for method in methods}
        state_certified = {method: False for method in methods}
        shared: _EpisodeBatch | None = None
        shared_audit_file: str | None = None
        fixed_q: dict[Platform, float] | None = None
        fixed_ranks: dict[Platform, int | None] = {}
        fixed_digest: str | None = None
        pairing = replicate_report["protocol_validation"]["pairing"]
        for episode in range(config.profile.rounds + 1):
            terminal = episode == config.profile.rounds
            seed = config.seed + replicate * 10_000_000 + episode * 100_000 + 50_000
            reference_digests = None
            for method in methods:
                if episode < len(records[method]):
                    saved = records[method][episode]
                    if saved["episode"] != episode:
                        raise ValueError("Round checkpoints must form a contiguous episode prefix")
                    digests = saved["exogenous_realization_digests"]
                    if reference_digests is None:
                        reference_digests = digests
                    elif digests != reference_digests:
                        raise ValueError("Saved episode realization pairing is inconsistent")
                    if saved["next_margins"] is not None:
                        states[method] = {
                            Platform(key) if key in {p.value for p in Platform} else key: value
                            for key, value in saved["next_margins"].items()
                        }
                    state_certified[method] = saved["next_state_certified"]
                    if episode == 0:
                        shared_audit_file = saved["deployment"]["mission_audit_file"]
                        if method == "fixed":
                            fixed_q = dict(states[method])
                            fixed_ranks = saved["conformal_ranks"]
                            fixed_digest = saved["episode_batch_digest"]
                    continue
                started = perf_counter()
                record_pair = save_paired_trace and episode == 1 and method in {"distributed", "nominal"}
                trace = WaveTraceRecorder(context={
                    "method": "shared_episode_0" if episode == 0 else method,
                    "replicate": replicate, "episode": episode, "margins": _jsonable(states[method]),
                    "episode_seed": seed,
                }) if (render_visuals or record_pair) and replicate == 0 and (episode != 0 or shared is None) else None
                collected = episode != 0 or shared is None
                if collected:
                    batch = _collect_episode(
                        experiment, states[method], seed,
                        f"Episode {episode}/{config.profile.rounds} | {application_name} | "
                        f"{'common' if episode == 0 else method} | replicate {replicate + 1}",
                        trace, tracking=tracking,
                    )
                    execution_seconds = perf_counter() - started
                    if episode == 0:
                        shared = batch
                        shared_execution_seconds = execution_seconds
                        if "single_agent" in methods and not records["single_agent"]:
                            states["single_agent"] = {
                                robot: initial[platform] for robot, platform in batch.robot_platforms.items()
                            }
                else:
                    assert shared is not None
                    batch = shared
                    execution_seconds = shared_execution_seconds
                margins = dict(states[method])
                execution_certified = state_certified[method]
                _validate_episode_margins(margins, batch.robot_platforms, config)
                if episode == 0:
                    if any(margins.get(robot, margins.get(platform)) != initial[platform]
                           for robot, platform in batch.robot_platforms.items()):
                        raise RuntimeError("Episode-0 initial margins differ across methods")
                elif margins != records[method][-1]["next_margins"]:
                    raise RuntimeError("An episode did not deploy the preceding episode's update")
                digests = _realization_digests(batch.missions)
                if reference_digests is None:
                    reference_digests = digests
                elif digests != reference_digests:
                    raise RuntimeError(f"Exogenous realization pairing failed at episode {episode}: {method}")
                if collision_controller in {CollisionController.HOCBF, CollisionController.WANG} and any(
                    not value for value in digests.values()
                ):
                    raise RuntimeError("Formation episode is missing realization digests")
                digest = _episode_batch_digest(batch)
                update_started = perf_counter()
                local = deepcopy(batch.local_calibration) if episode == 0 else batch.local_calibration
                next_margins = None
                q: dict[Any, Any] = {}
                counts: dict[Any, int] = {}
                ranks: dict[Any, Any] = {}
                kappa = {p: (0.0 if method in {"unshifted", "fixed", "nominal"}
                             else config.classes[p].configured_kappa) for p in Platform}
                cp_rounds = 0
                maicp_record = None
                result = None
                robot_updates = None
                certified = False
                calibration_source = "terminal_evaluation_only" if terminal else "executed_episode"
                calibration_episode = episode
                calibration_digest = digest
                if method == "single_agent":
                    robot_updates = {}
                    for robot, platform in sorted(batch.robot_platforms.items()):
                        own_store = local.node_stores[robot]
                        if terminal:
                            item = single_agent_conformal_quantile(own_score_history(own_store, platform), config.classes[platform])
                            item.update(kappa=config.classes[platform].configured_kappa,
                                        r_next=None, status="terminal_evaluation_only")
                        else:
                            item = single_agent_icp_update(config, platform, margins[robot], node_store=own_store)
                        item.update(r_used=margins[robot], platform=platform.value,
                                    kappa_used=None if terminal else item["kappa"],
                                    update_performed=not terminal,
                                    calibration_mission_ids=sorted(m for m, values in own_store.robot_scores[platform].items() if values))
                        robot_updates[robot] = item
                    q = {robot: item["q"] for robot, item in robot_updates.items()}
                    counts = {robot: item["count"] for robot, item in robot_updates.items()}
                    ranks = {robot: item["rank"] for robot, item in robot_updates.items()}
                    kappa = {robot: item["kappa"] for robot, item in robot_updates.items()}
                    if not terminal:
                        next_margins = {robot: item["r_next"] for robot, item in robot_updates.items()}
                    status = "terminal_evaluation_only" if terminal else (
                        "updated" if all(item["status"] == "updated" for item in robot_updates.values())
                        else "robot_update_statuses_recorded")
                elif method == "fixed":
                    if episode == 0:
                        quantiles, fixed_ranks = _central_class_quantiles(batch.missions, config)
                        if any(value is None for value in quantiles.values()):
                            raise ValueError("Fixed episode-0 conformal thresholds are unavailable")
                        fixed_q = {p: float(quantiles[p]) for p in Platform}
                        _validate_episode_margins(fixed_q, batch.robot_platforms, config)
                        fixed_digest = digest
                    assert fixed_q is not None
                    q, ranks = dict(fixed_q), dict(fixed_ranks)
                    counts = {p: config.profile.calibration_missions for p in Platform}
                    calibration_source, calibration_episode, calibration_digest = "frozen_episode_0", 0, fixed_digest
                    next_margins = None if terminal else dict(fixed_q)
                    status = "terminal_fixed" if terminal else "fixed_from_episode_0" if episode == 0 else "fixed_unchanged"
                elif method == "nominal":
                    next_margins = None if terminal else {p: 0.0 for p in Platform}
                    calibration_source = "none_debug_baseline"
                    status = "terminal_nominal" if terminal else "nominal_debug"
                elif terminal:
                    q, ranks = _central_class_quantiles(batch.missions, config)
                    counts = {p: len(batch.missions) for p in Platform}
                    status = "terminal_evaluation_only"
                else:
                    analytic = _episode_sensitivity(experiment, batch, local, report, method, replicate, episode)
                    result = algorithm.run_round(
                        round_index=episode, current_margins=margins, missions=batch.missions,
                        update_graph=batch.update_graph, robot_platforms=batch.robot_platforms,
                        analytic_kappa=analytic, local_calibration=local,
                        centralized=method == "centralized", unshifted=method == "unshifted",
                        capture_diagnostics=replicate == 0,
                    )
                    next_margins = dict(result.margin_update.margins)
                    q, ranks = dict(result.certified_thresholds), dict(result.conformal_ranks)
                    kappa = {p: 0.0 for p in Platform} if method == "unshifted" else dict(result.sensitivity.class_kappa)
                    counts = {p: len(batch.missions) for p in Platform}
                    cp_rounds = result.communication_rounds
                    maicp_record = _round_record(result)
                    maicp_record["kappa_used"] = kappa
                    status, certified = result.margin_update.status, result.margin_update.certified
                    _print_round_summary(application_name, method, replicate, episode, config, result)
                if next_margins is not None:
                    _validate_episode_margins(next_margins, batch.robot_platforms, config)
                    states[method] = dict(next_margins)
                    state_certified[method] = certified
                update_seconds = perf_counter() - update_started
                deployment = deployment_metrics(
                    batch.missions, margins, config, execution_certified,
                    **({"coverage_radii": margins} if tracking else {}),
                )
                audit_file = (
                    f"audits/replicate_{replicate:03d}/shared_episode_{episode:03d}.jsonl.gz"
                    if episode == 0 else
                    f"audits/replicate_{replicate:03d}/{method}_episode_{episode:03d}.jsonl.gz"
                )
                if collected:
                    _atomic_mission_audits(result_root / audit_file, deployment.pop("mission_audit"))
                    if episode == 0:
                        shared_audit_file = audit_file
                else:
                    deployment.pop("mission_audit")
                    audit_file = shared_audit_file
                deployment["mission_audit_file"] = audit_file
                deployment["mission_audit_count"] = len(batch.missions)
                record = {
                    "episode": episode, "margins": margins, "next_margins": next_margins,
                    "robot_platforms": dict(batch.robot_platforms),
                    "q": q, "conformal_ranks": ranks,
                    "kappa_used": {key: None for key in kappa} if terminal else kappa,
                    "margin_update_performed": not terminal and method != "nominal" and (method != "fixed" or episode == 0),
                    "q_source": "evaluation_only_exact" if terminal and method != "fixed" else calibration_source,
                    "calibration": {"source": calibration_source, "episode": calibration_episode,
                                    "batch_digest": calibration_digest, "sample_count": counts},
                    "deployment": deployment,
                    "episode_batch": _batch_audit(batch.missions, margins, seed,
                        source="shared_episode_0" if episode == 0 else "executed_episode",
                        used_for_next_margin=not terminal and method != "nominal" and (method != "fixed" or episode == 0)),
                    "episode_batch_digest": digest,
                    "exogenous_realization_digests": digests,
                    "shared_episode_0_reused": episode == 0,
                    "execution_batch_collected": collected,
                    "cp_communication_rounds": cp_rounds,
                    "runtime_seconds": perf_counter() - started,
                    "execution_runtime_seconds": execution_seconds,
                    "calibration_update_runtime_seconds": update_seconds,
                    "execution_runtime_shared": episode == 0,
                    "status": status,
                    "next_state_certified": certified,
                }
                if robot_updates is not None:
                    record["single_agent"] = {"robots": robot_updates, "cp_communication_rounds": 0,
                        "calibration_scope": "one physical robot's current episode history",
                        "scalar_updates_certified": not terminal and all(item.get("certified", False) for item in robot_updates.values()),
                        # The baseline for a single agent accounts only for a change in that robot's margin.
                        "joint_policy_shift_certified": False,
                        "peer_score_or_margin_inputs": False}
                if maicp_record is not None:
                    record["maicp"] = maicp_record
                if trace is not None:
                    name = f"replicate_{replicate}_shared_episode_0.npz" if episode == 0 else f"replicate_{replicate}_{method}_episode_{episode}.npz"
                    trace.save(result_root / "traces" / name)
                    record["trace_file"] = f"traces/{name}"
                    if episode == 0:
                        shared_trace_file = record["trace_file"]
                elif episode == 0 and render_visuals and replicate == 0:
                    record["trace_file"] = shared_trace_file
                records[method].append(record)
                checkpoint_file = f"rounds/replicate_{replicate:03d}/{method}/episode_{episode:03d}.json"
                record["round_checkpoint_file"] = checkpoint_file
                _atomic_json(result_root / checkpoint_file, record)
                report["last_completed_round"] = {
                    "replicate": replicate, "method": method, "episode": episode,
                }
                _atomic_json(destination, report)
                # Later methods/episodes need only margins and scalar summaries.
                # Release the current batch before collecting the next one.
                del batch, local, deployment, result, trace
            if episode >= len(pairing):
                pairing.append({"episode": episode, "exact_match": True,
                                "realization_digests": reference_digests})
            if episode == 0:
                shared = None
        common_digests = {records[method][0]["episode_batch_digest"] for method in methods}
        if len(common_digests) != 1:
            raise RuntimeError("Episode-0 mission batch was not reused exactly")
        replicate_report["protocol_validation"].update(
            common_initial_margins=True, exact_episode_0_batch_reuse=True,
            episode_transition_margins_match=True, all_margins_finite_and_in_bounds=True,
        )
        _atomic_json(destination, report)
    report["run_status"] = "completed"
    _atomic_json(destination, report)
    if render_visuals:
        from .visualization import render_experiment_report
        report["visualizations"] = render_experiment_report(report, result_root)
        _atomic_json(destination, report)
    return report


def run_case_study(experiment, application, methods, output, resume):
    from .visualization import paper_figures

    report = run_nested_experiment(
        experiment, application, methods, output_root=output,
        render_visuals=False, resume=resume, save_paired_trace=True,
    )
    case = "tracking" if application == "target_tracking" else "formation"
    paper_figures(case, Path(output), report)
