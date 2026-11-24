"""Select sensitivity certificates and aggregate bounds by class."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from scipy.linalg import expm

from maicp.configuration import ExperimentConfig, Platform, SensitivityMode

from maicp.score_stores import LocalCalibrationStores
from .distributed import _neighbor_exchange, graph_diameter
from .types import SensitivityResult


def continuous_time_robot_kappa(
    comparison_matrix: np.ndarray,
    input_gain: np.ndarray,
    score_gain: np.ndarray,
    direct_score_gain: float,
    horizon: float,
) -> float:
    """Evaluate the sensitivity certificate from the comparison system in continuous time."""

    matrix = np.asarray(comparison_matrix, dtype=float)
    b = np.asarray(input_gain, dtype=float)
    score = np.asarray(score_gain, dtype=float)
    size = len(b)
    if matrix.shape != (size, size) or score.shape != (size,):
        raise ValueError("Incompatible continuous-time certificate dimensions")
    if np.any(matrix - np.diag(np.diag(matrix)) < -1e-12):
        raise ValueError("The comparison matrix must be Metzler")
    if np.any(b < 0.0) or np.any(score < 0.0) or direct_score_gain < 0.0:
        raise ValueError("Sensitivity gains must be nonnegative")
    augmented = np.zeros((2 * size, 2 * size))
    augmented[:size, :size] = matrix
    augmented[:size, size:] = np.eye(size)
    integral = expm(augmented * horizon)[:size, size:]
    return float(direct_score_gain + score @ integral @ b)


def discrete_time_robot_kappa(
    transition_bounds: Sequence[np.ndarray],
    input_bounds: Sequence[np.ndarray],
    score_gain: np.ndarray,
    direct_score_gain: float,
) -> float:
    """Propagate sensitivity bounds through the sequence of discrete events."""

    if len(transition_bounds) != len(input_bounds):
        raise ValueError("Each event requires both A and b bounds")
    if not input_bounds:
        if direct_score_gain < 0.0:
            raise ValueError("Invalid direct score gain")
        return float(direct_score_gain)
    gain = np.zeros_like(np.asarray(input_bounds[0], dtype=float))
    maximum = gain.copy()
    for matrix, vector in zip(transition_bounds, input_bounds, strict=True):
        matrix = np.asarray(matrix, dtype=float)
        vector = np.asarray(vector, dtype=float)
        if np.any(matrix < 0.0) or np.any(vector < 0.0):
            raise ValueError("Discrete comparison bounds must be nonnegative")
        gain = matrix @ gain + vector
        maximum = np.maximum(maximum, gain)
    score = np.asarray(score_gain, dtype=float)
    if score.shape != maximum.shape or np.any(score < 0.0) or direct_score_gain < 0.0:
        raise ValueError("Invalid discrete score gain")
    return float(direct_score_gain + score @ maximum)


def build_sensitivity_result(
    config: ExperimentConfig,
    update_graph: Mapping[str, Sequence[str] | set[str]],
    robot_platforms: Mapping[str, Platform],
    analytic_kappa: Mapping[str, float] | None = None,
    empirical_ratios: Mapping[str, float] | None = None,
    local_calibration: LocalCalibrationStores | None = None,
) -> SensitivityResult:
    """Compute each robot’s coefficient and broadcast the local maxima for each class.

    Analytic values are supplied by the bound for the application under feedback.
    In the distributed path those values must come from the persistent node
    stores.  ``analytic_kappa`` remains as a compatibility/evaluation input for
    callers that do not use the distributed path.  Configured values are
    explicitly assumed to have been certified externally.  Paired finite
    differences are diagnostics only and never invalidate a certified
    coefficient.
    """

    analytic_kappa = analytic_kappa or {}
    empirical_ratios = empirical_ratios or {}
    mode = config.sensitivity_mode
    notes: list[str] = []
    robot_kappa: dict[Platform, dict[str, float]] = {platform: {} for platform in Platform}
    validation_ratio: dict[Platform, float] = {platform: 0.0 for platform in Platform}

    for robot_id, platform in robot_platforms.items():
        if mode is SensitivityMode.ANALYTIC:
            if local_calibration is None:
                value = float(analytic_kappa.get(robot_id, float("nan")))
            else:
                # Each robot supplies its own sensitivity coefficient.
                local = local_calibration.node_stores.get(robot_id)
                value = float(local.analytic_kappa) if local and local.analytic_kappa is not None else float("nan")
        elif mode is SensitivityMode.CONFIGURED:
            if local_calibration is None:
                value = float(config.classes[platform].configured_kappa)
            else:
                local = local_calibration.node_stores.get(robot_id)
                configured = getattr(local, "configured_kappa", None)
                value = float(configured) if configured is not None else float("nan")
        else:
            if local_calibration is None:
                value = float(empirical_ratios.get(robot_id, float("nan")))
            else:
                local = local_calibration.node_stores.get(robot_id)
                value = (
                    float(local.empirical_kappa)
                    if local is not None and local.empirical_kappa is not None
                    else float("nan")
                )
        robot_kappa[platform][robot_id] = value
        if robot_id in empirical_ratios:
            ratio_value = float(empirical_ratios[robot_id])
            if not np.isfinite(ratio_value):
                validation_ratio[platform] = float("inf")
            elif value > 0.0 and np.isfinite(value):
                validation_ratio[platform] = max(
                    validation_ratio[platform], ratio_value / value
                )
            elif ratio_value > 0.0:
                validation_ratio[platform] = float("inf")

    # Exchange class maxima with neighbors for one graph diameter.
    local_summary: dict[str, tuple[tuple[float, bool], ...]] = {}
    platforms = tuple(Platform)
    for node in sorted(update_graph):
        local_values: list[tuple[float, bool]] = []
        for platform in platforms:
            values = [
                robot_kappa[platform][robot_id]
                for robot_id, robot_platform in robot_platforms.items()
                if robot_id == node and robot_platform == platform
            ]
            invalid = any(not np.isfinite(value) or value < 0.0 for value in values)
            finite_values = [value for value in values if np.isfinite(value) and value >= 0.0]
            local_values.append((max(finite_values) if finite_values else float("nan"), invalid))
        local_summary[node] = tuple(local_values)

    def combine_max(
        left: tuple[tuple[float, bool], ...], right: tuple[tuple[float, bool], ...]
    ) -> tuple[tuple[float, bool], ...]:
        combined: list[tuple[float, bool]] = []
        for (left_value, left_invalid), (right_value, right_invalid) in zip(
            left, right, strict=True
        ):
            invalid = bool(left_invalid or right_invalid)
            if np.isnan(left_value):
                value = float(right_value)
            elif np.isnan(right_value):
                value = float(left_value)
            else:
                value = max(float(left_value), float(right_value))
            combined.append((value, invalid))
        return tuple(combined)

    # The tuple carries every class maximum in one packet, so this is one
    # maximum over the graph diameter for the complete sensitivity summary.
    kappa_flooding_rounds = graph_diameter(update_graph)
    for _ in range(kappa_flooding_rounds):
        inbox = _neighbor_exchange(update_graph, local_summary)
        next_summary: dict[str, tuple[float, ...]] = {}
        for node, own_summary in local_summary.items():
            aggregate = own_summary
            for neighbor_summary in inbox[node].values():
                aggregate = combine_max(aggregate, tuple(neighbor_summary))
            next_summary[node] = aggregate
        local_summary = next_summary
    aggregate = next(iter(local_summary.values()))
    class_kappa = {
        platform: (
            float("nan")
            if aggregate[index][1]
            else float(aggregate[index][0])
        )
        for index, platform in enumerate(platforms)
    }

    values_valid = all(np.isfinite(value) and value >= 0.0 for value in class_kappa.values())
    paired_valid = all(ratio <= 1.0 + 1e-8 for ratio in validation_ratio.values())
    if not values_valid:
        notes.append("missing or invalid sensitivity coefficient")
    if not paired_valid:
        notes.append("paired validation exceeded the claimed sensitivity bound (diagnostic)")
    certified = mode is not SensitivityMode.EMPIRICAL and values_valid
    if mode is SensitivityMode.EMPIRICAL:
        notes.append("empirical finite differences are diagnostic, not a certified kappa")
    return SensitivityResult(
        robot_kappa=robot_kappa,
        class_kappa=class_kappa,
        mode=mode.value,
        certified=certified,
        # Paired score differences are an empirical check on the supplied sensitivity bound.
        valid=values_valid,
        max_validation_ratio=validation_ratio,
        notes=tuple(notes),
        paired_validation_valid=paired_valid,
        kappa_flooding_rounds=kappa_flooding_rounds,
    )
