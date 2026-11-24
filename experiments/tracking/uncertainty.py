"""Mahalanobis scores and reported position covariance ellipses."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

TRACKING_SCORE_SEMANTICS = "drwt_active_valid_nominal_excess_v2"


def nominal_radius(alpha: float) -> float:
    """Confidence radius from the chi squared distribution with two degrees of freedom."""
    return math.sqrt(chi2_threshold(alpha))


def conformal_excess_score(raw_score: float, alpha: float) -> float:
    """Return the positive part of the excess Mahalanobis radius. An empty score domain has value +∞."""
    return max(0.0, float(raw_score) - nominal_radius(alpha))


def chi2_threshold(alpha: float) -> float:
    """Return ``chi2_2(1 - alpha)`` for a valid class error level."""

    value = float(alpha)
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError("alpha must be finite and strictly between zero and one")
    # With two degrees of freedom the chi squared survival function is exp(-q / 2).
    threshold = -2.0 * math.log(value)
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("chi-square threshold must be finite and positive")
    return threshold


def _validated_error(error: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    value = np.asarray(error, dtype=float)
    if value.shape != (2,):
        raise ValueError("position error must have shape (2,)")
    if not np.all(np.isfinite(value)):
        raise ValueError("position error must be finite")
    return value.copy()


def validate_xy_covariance(
    covariance: np.ndarray | list[list[float]] | tuple[tuple[float, ...], ...],
    *,
    name: str = "position covariance",
) -> np.ndarray:
    """Validate and return a finite, symmetric, positive definite matrix of shape (2, 2)."""

    return _validated_factor(covariance, name=name)[0]


def _validated_factor(
    covariance: np.ndarray | list[list[float]] | tuple[tuple[float, ...], ...],
    *,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate one covariance and retain its Cholesky factor for reuse."""

    value = np.asarray(covariance, dtype=float)
    if value.shape != (2, 2):
        raise ValueError(f"{name} must have shape (2, 2)")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be finite")
    if not np.allclose(value, value.T, rtol=1e-10, atol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    value = 0.5 * (value + value.T)
    try:
        factor = np.linalg.cholesky(value)
    except np.linalg.LinAlgError as error:
        raise ValueError(f"{name} must be positive definite") from error
    if not np.all(np.isfinite(factor)):
        raise ValueError(f"{name} factor must be finite")
    return value, factor


def mahalanobis_squared(
    error: np.ndarray | list[float] | tuple[float, ...],
    covariance: np.ndarray | list[list[float]] | tuple[tuple[float, ...], ...],
    *,
    name: str = "position uncertainty",
) -> float:
    """Compute ``error.T @ covariance^-1 @ error`` with SPD validation."""

    residual = _validated_error(error)
    _, factor = _validated_factor(covariance, name=f"{name} covariance")
    try:
        solved = np.linalg.solve(factor, residual)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} covariance factor is invalid") from exc
    result = float(np.dot(solved, solved))
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} Mahalanobis score is not finite")
    return result


def mahalanobis_radius(
    error: np.ndarray | list[float] | tuple[float, ...],
    covariance: np.ndarray | list[list[float]] | tuple[tuple[float, ...], ...],
    *,
    name: str = "position uncertainty",
) -> float:
    """Compute the Mahalanobis radius in two dimensions."""

    squared = mahalanobis_squared(error, covariance, name=name)
    radius = math.sqrt(squared)
    if not math.isfinite(radius):
        raise ValueError(f"{name} Mahalanobis radius is not finite")
    return radius


def uncertainty_observation(
    error: np.ndarray | list[float] | tuple[float, ...],
    covariance: np.ndarray | list[list[float]] | tuple[tuple[float, ...], ...],
    *,
    alpha: float,
    coverage_radius: float | None = None,
    name: str = "position uncertainty",
) -> dict[str, Any]:
    """Return one score sample and its native/reported ellipse diagnostics."""

    residual = _validated_error(error)
    raw, factor = _validated_factor(covariance, name=f"{name} covariance")
    try:
        solved = np.linalg.solve(factor, residual)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} covariance factor is invalid") from exc
    raw_squared = float(np.dot(solved, solved))
    if not math.isfinite(raw_squared) or raw_squared < 0.0:
        raise ValueError(f"{name} Mahalanobis score is not finite")
    raw_score = math.sqrt(raw_squared)
    threshold = chi2_threshold(alpha)
    scale = 1.0
    if coverage_radius is not None:
        radius = float(coverage_radius)
        if not math.isfinite(radius) or radius < 0.0:
            raise ValueError("coverage_radius must be finite and nonnegative")
        scale = radius * radius / threshold
    if not math.isfinite(scale) or scale < 0.0:
        raise ValueError(f"{name} covariance scale is not finite and nonnegative")
    reported = scale * raw
    if not np.all(np.isfinite(reported)):
        raise ValueError(f"{name} reported covariance is not finite")
    native_radius = math.sqrt(threshold)
    reported_radius = native_radius if coverage_radius is None else float(coverage_radius)
    native_contains = raw_score <= native_radius
    reported_contains = raw_score <= reported_radius
    sqrt_determinant = float(np.prod(np.diag(factor)))
    raw_area = float(math.pi * threshold * sqrt_determinant)
    reported_area = float(raw_area * scale)
    if not all(
        math.isfinite(value)
        for value in (raw_score, raw_squared, raw_area, reported_area, scale)
    ):
        raise ValueError(f"{name} uncertainty output is not finite")
    return {
        "error": residual.tolist(),
        "raw_covariance": raw.tolist(),
        "reported_covariance": reported.tolist(),
        "mahalanobis_score": float(raw_score),
        "mahalanobis_squared": float(raw_squared),
        "native_contains_truth": bool(native_contains),
        "reported_contains_truth": bool(reported_contains),
        "native_ellipse_area": raw_area,
        "reported_ellipse_area": reported_area,
        "native_covariance_scale": 1.0,
        "reported_covariance_scale": float(scale),
        "native_ellipse_axis_scale": 1.0,
        "reported_ellipse_axis_scale": float(math.sqrt(scale)),
        "ellipse_axis_scale": float(math.sqrt(scale)),
        "covariance_scale": float(scale),
        "chi2_threshold": float(threshold),
        "alpha": float(alpha),
        "coverage_radius": (
            None if coverage_radius is None else float(coverage_radius)
        ),
    }


@dataclass
class UncertaintySummary:
    """Record containment and ellipse areas for the native and reported estimates."""

    alpha: float
    coverage_radius: float | None = None
    # Count updates during the task separately from valid estimates used for scoring.
    expected_samples: int = 0

    def __post_init__(self) -> None:
        self.alpha = float(self.alpha)
        self.chi2_threshold = chi2_threshold(self.alpha)
        if self.coverage_radius is not None:
            radius = float(self.coverage_radius)
            if not math.isfinite(radius) or radius < 0.0:
                raise ValueError("coverage_radius must be finite and nonnegative")
            self.coverage_radius = radius
        try:
            expected_samples = int(self.expected_samples)
        except (TypeError, ValueError) as error:
            raise ValueError("expected_samples must be a nonnegative integer") from error
        if expected_samples != self.expected_samples or expected_samples < 0:
            raise ValueError("expected_samples must be a nonnegative integer")
        self.expected_samples = expected_samples
        self.covariance_scale = (
            1.0
            if self.coverage_radius is None
            else self.coverage_radius * self.coverage_radius / self.chi2_threshold
        )
        self.samples = 0
        self.native_covered_samples = 0
        self.reported_covered_samples = 0
        self.native_area_sum = 0.0
        self.reported_area_sum = 0.0
        self.euclidean_squared_error_sum = 0.0
        self.maximum_euclidean_error = 0.0
        self.native_covariance_trace_sum = 0.0
        self.native_covariance_trace_min = math.inf
        self.native_covariance_trace_max = -math.inf

    def set_expected_samples(self, count: int) -> None:
        """Set the intended scored interval after simulation bookkeeping."""

        try:
            value = int(count)
        except (TypeError, ValueError) as error:
            raise ValueError("expected_samples must be a nonnegative integer") from error
        if value != count or value < 0:
            raise ValueError("expected_samples must be a nonnegative integer")
        self.expected_samples = value

    def add(self, observation: dict[str, Any]) -> None:
        """Add a validated observation returned by :func:`uncertainty_observation`."""

        self.samples += 1
        self.native_covered_samples += int(observation["native_contains_truth"])
        self.reported_covered_samples += int(observation["reported_contains_truth"])
        self.native_area_sum += float(observation["native_ellipse_area"])
        self.reported_area_sum += float(observation["reported_ellipse_area"])
        error = np.asarray(observation["error"], dtype=float)
        squared_error = float(np.dot(error, error))
        if not math.isfinite(squared_error) or squared_error < 0.0:
            raise ValueError("observation Euclidean error is not finite")
        self.euclidean_squared_error_sum += squared_error
        self.maximum_euclidean_error = max(
            self.maximum_euclidean_error,
            math.sqrt(squared_error),
        )
        native_trace = float(np.trace(np.asarray(observation["raw_covariance"], dtype=float)))
        self.native_covariance_trace_sum += native_trace
        self.native_covariance_trace_min = min(self.native_covariance_trace_min, native_trace)
        self.native_covariance_trace_max = max(self.native_covariance_trace_max, native_trace)

    def as_dict(self) -> dict[str, Any]:
        """Return scalar mission diagnostics that can be serialized as JSON."""

        sample_count = int(self.samples)
        native_mean = self.native_area_sum / sample_count if sample_count else None
        reported_mean = self.reported_area_sum / sample_count if sample_count else None
        missing = max(int(self.expected_samples) - sample_count, 0)
        native_available = (
            self.native_covered_samples / sample_count if sample_count else None
        )
        reported_available = (
            self.reported_covered_samples / sample_count if sample_count else None
        )
        rms_error = (
            math.sqrt(self.euclidean_squared_error_sum / sample_count)
            if sample_count
            else None
        )
        return {
            "samples": sample_count,
            "expected_scored_opportunities": int(self.expected_samples),
            "missing_estimates": int(missing),
            "native_covered_samples": int(self.native_covered_samples),
            "reported_covered_samples": int(self.reported_covered_samples),
            # The mission evaluator applies the class rank to the robot scores.
            "native_available_coverage": (
                None if native_available is None else float(native_available)
            ),
            "reported_available_coverage": (
                None if reported_available is None else float(reported_available)
            ),
            "native_trajectory_contained": (
                None if not sample_count else self.native_covered_samples == sample_count
            ),
            "reported_trajectory_contained": (
                None if not sample_count else self.reported_covered_samples == sample_count
            ),
            "native_area_sum": float(self.native_area_sum),
            "reported_area_sum": float(self.reported_area_sum),
            "native_area_mean": None if native_mean is None else float(native_mean),
            "reported_area_mean": None if reported_mean is None else float(reported_mean),
            "euclidean_squared_error_sum": float(self.euclidean_squared_error_sum),
            "euclidean_error_count": sample_count,
            "maximum_euclidean_error": (
                None if not sample_count else float(self.maximum_euclidean_error)
            ),
            "rms_euclidean_error": None if rms_error is None else float(rms_error),
            "native_covariance_trace_sum": float(self.native_covariance_trace_sum),
            "native_covariance_trace_count": sample_count,
            "native_covariance_trace_mean": (
                self.native_covariance_trace_sum / sample_count if sample_count else None
            ),
            "native_covariance_trace_min": self.native_covariance_trace_min if sample_count else None,
            "native_covariance_trace_max": self.native_covariance_trace_max if sample_count else None,
            "native_covariance_scale": 1.0,
            "reported_covariance_scale": float(self.covariance_scale),
            "native_ellipse_axis_scale": 1.0,
            "reported_ellipse_axis_scale": float(math.sqrt(self.covariance_scale)),
            "ellipse_axis_scale": float(math.sqrt(self.covariance_scale)),
            "covariance_scale": float(self.covariance_scale),
            "alpha": float(self.alpha),
            "chi2_threshold": float(self.chi2_threshold),
            "native_radius": float(math.sqrt(self.chi2_threshold)),
            "reported_radius": (
                float(math.sqrt(self.chi2_threshold))
                if self.coverage_radius is None
                else float(self.coverage_radius)
            ),
            "coverage_radius": (
                None
                if self.coverage_radius is None
                else float(self.coverage_radius)
            ),
        }


__all__ = [
    "UncertaintySummary",
    "chi2_threshold",
    "mahalanobis_radius",
    "mahalanobis_squared",
    "uncertainty_observation",
    "validate_xy_covariance",
]
