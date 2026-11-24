"""Solve the coupled margin update subject to the sensitivity bounds."""

from __future__ import annotations

from collections.abc import Sequence

import cvxpy as cp
import numpy as np

from maicp.configuration import ClassConfig, Platform

from .types import MarginUpdateResult


def normalized_margin_distance(
    candidate: dict[Platform, float],
    current: dict[Platform, float],
    classes: dict[Platform, ClassConfig],
) -> float:
    return float(
        np.linalg.norm(
            [
                (candidate[platform] - current[platform]) / classes[platform].scale
                for platform in Platform
            ]
        )
    )


def _held_result(
    current: dict[Platform, float],
    status: str,
    sensitivity_certified: bool,
    *,
    residual: float = 0.0,
    replica_statuses: dict[str, str] | None = None,
    replica_failures: Sequence[str] = (),
) -> MarginUpdateResult:
    return MarginUpdateResult(
        margins=dict(current),
        previous_margins=dict(current),
        status=status,
        certified=False,
        objective=None,
        normalized_change=0.0,
        constraint_slacks={platform: float("nan") for platform in Platform},
        margin_consensus_residual=float(residual),
        replica_statuses=dict(replica_statuses or {}),
        replica_failures=tuple(replica_failures),
    )


def solve_margin_update_replica(
    current: dict[Platform, float],
    thresholds: dict[Platform, float],
    kappa: dict[Platform, float],
    classes: dict[Platform, ClassConfig],
    proximal_weight: float,
    sensitivity_certified: bool,
) -> MarginUpdateResult:
    """Solve one pure local copy of the coupled margin problem.

    The function has no graph or shared mutable state.  The MA-ICP
    orchestrator calls it independently at every update node, then compares
    the resulting candidates before accepting a certified update.
    """

    platforms = tuple(Platform)
    if any(platform not in current for platform in platforms):
        return _held_result(current, "no_certified_update:missing_current", sensitivity_certified)
    if any(not np.isfinite(current[platform]) for platform in platforms):
        return _held_result(current, "no_certified_update:nonfinite_current", sensitivity_certified)
    if any(not np.isfinite(thresholds[platform]) for platform in platforms):
        return _held_result(current, "no_certified_update:nonfinite_threshold", sensitivity_certified)
    if any(not np.isfinite(kappa[platform]) or kappa[platform] < 0.0 for platform in platforms):
        return _held_result(current, "no_certified_update:nonfinite_sensitivity", sensitivity_certified)

    previous = np.asarray([current[platform] for platform in platforms], dtype=float)
    scales = np.asarray([classes[platform].scale for platform in platforms], dtype=float)
    lower = np.asarray([classes[platform].margin_min for platform in platforms], dtype=float)
    upper = np.asarray([classes[platform].margin_max for platform in platforms], dtype=float)
    q = np.asarray([thresholds[platform] for platform in platforms], dtype=float)
    sensitivity = np.asarray([kappa[platform] for platform in platforms], dtype=float)
    weights = np.asarray([classes[platform].weight for platform in platforms], dtype=float)
    if (
        np.any(~np.isfinite(scales))
        or np.any(scales <= 0.0)
        or np.any(~np.isfinite(lower))
        or np.any(~np.isfinite(upper))
        or np.any(lower > upper)
        or not np.isfinite(proximal_weight)
        or proximal_weight <= 0.0
    ):
        return _held_result(current, "no_certified_update:invalid_problem_data", sensitivity_certified)

    margin = cp.Variable(len(platforms))
    delta = cp.norm(cp.multiply(1.0 / scales, margin - previous), 2)
    constraints = [margin >= lower, margin <= upper, q + sensitivity * delta <= margin]
    objective = cp.Minimize(
        weights @ cp.multiply(1.0 / scales, margin)
        + 0.5 * proximal_weight * cp.square(delta)
    )
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver="CLARABEL", verbose=False)
    except (cp.error.SolverError, ValueError, TypeError):
        return _held_result(current, "no_certified_update:solver_error", sensitivity_certified)
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or margin.value is None:
        return _held_result(current, f"no_certified_update:{problem.status}", sensitivity_certified)

    candidate_array = np.asarray(margin.value, dtype=float)
    if candidate_array.shape != (len(platforms),) or np.any(~np.isfinite(candidate_array)):
        return _held_result(current, "no_certified_update:nonfinite_solution", sensitivity_certified)
    candidate = {platform: float(candidate_array[i]) for i, platform in enumerate(platforms)}
    change = normalized_margin_distance(candidate, current, classes)
    slacks = {
        platform: candidate[platform] - thresholds[platform] - kappa[platform] * change
        for platform in platforms
    }
    if (
        np.any(candidate_array < lower - 1e-7)
        or np.any(candidate_array > upper + 1e-7)
        or min(slacks.values()) < -1e-6
        or not np.isfinite(change)
    ):
        return _held_result(current, "no_certified_update:residual_check", sensitivity_certified)
    return MarginUpdateResult(
        margins=candidate,
        previous_margins=dict(current),
        status="updated" if sensitivity_certified else "updated_noncertified",
        certified=sensitivity_certified,
        objective=float(problem.value) if problem.value is not None else None,
        normalized_change=change,
        constraint_slacks=slacks,
    )


def solve_margin_update(
    current: dict[Platform, float],
    thresholds: dict[Platform, float],
    kappa: dict[Platform, float],
    classes: dict[Platform, ClassConfig],
    proximal_weight: float,
    sensitivity_certified: bool,
    update_nodes: Sequence[str] | None = None,
    consensus_tolerance: float = 1e-7,
) -> MarginUpdateResult:
    """Solve the coupled margin problem at each update node. Hold the current margins if the solutions disagree."""

    if consensus_tolerance < 0.0 or not np.isfinite(consensus_tolerance):
        raise ValueError("Consensus tolerance must be finite and nonnegative")
    nodes = tuple(sorted(set(update_nodes))) if update_nodes is not None else ("local",)
    if not nodes:
        return _held_result(
            current,
            "no_certified_update:no_update_nodes",
            sensitivity_certified,
            residual=float("inf"),
        )

    replicas = {
        node: solve_margin_update_replica(
            current=current,
            thresholds=thresholds,
            kappa=kappa,
            classes=classes,
            proximal_weight=proximal_weight,
            sensitivity_certified=sensitivity_certified,
        )
        for node in nodes
    }
    statuses = {node: result.status for node, result in replicas.items()}
    failures = tuple(
        node for node, result in replicas.items() if result.status.startswith("no_certified_update:")
    )
    if failures:
        residual = float("inf")
        return _held_result(
            current,
            "no_certified_update:replica_failure",
            sensitivity_certified,
            residual=residual,
            replica_statuses=statuses,
            replica_failures=failures,
        )

    reference = replicas[nodes[0]]
    residual = 0.0
    disagreement = False
    for node in nodes[1:]:
        candidate = replicas[node]
        distance = normalized_margin_distance(candidate.margins, reference.margins, classes)
        residual = max(residual, distance)
        disagreement = disagreement or distance > consensus_tolerance
    if disagreement:
        return _held_result(
            current,
            "no_certified_update:replica_disagreement",
            sensitivity_certified,
            residual=residual,
            replica_statuses=statuses,
        )

    return MarginUpdateResult(
        margins=dict(reference.margins),
        previous_margins=dict(current),
        status=reference.status,
        certified=reference.certified,
        objective=reference.objective,
        normalized_change=reference.normalized_change,
        constraint_slacks=dict(reference.constraint_slacks),
        margin_consensus_residual=residual,
        replica_statuses=statuses,
        replica_failures=(),
    )


def solve_scalar_margin_update(
    current: float,
    threshold: float,
    kappa: float,
    class_config: ClassConfig,
    proximal_weight: float,
    sensitivity_certified: bool = True,
) -> dict[str, float | bool | str | None]:
    """Solve one robot's scalar robust margin update.

    This is the scalar counterpart of :func:`solve_margin_update`.
    It minimizes the same weighted linear objective plus proximal penalty,
    subject to the same class box and the scalar robust constraint

    ``q + kappa * abs(r_next - r_current) / scale <= r_next``.

    The feasible set is the union of at most two intervals split at the
    current margin.  Solving each interval by clipping the unconstrained
    minimizer avoids a separate optimizer while retaining the exact convex
    objective used by the coupled update.  The returned mapping contains only
    scalars that can be serialized as JSON so it can be embedded in a report for each robot.
    """

    def held(status: str, *, objective: float | None = None) -> dict[str, float | bool | str | None]:
        try:
            held_margin = float(current)
            if not np.isfinite(held_margin):
                held_margin = float("nan")
        except (TypeError, ValueError):
            held_margin = float("nan")
        return {
            "r_next": held_margin,
            "status": status,
            "certified": False,
            "objective": objective,
            "normalized_change": 0.0,
            "constraint_slack": float("nan"),
        }

    try:
        current_value = float(current)
        q_value = float(threshold)
        kappa_value = float(kappa)
        scale = float(class_config.scale)
        weight = float(class_config.weight)
        lower = float(class_config.margin_min)
        upper = float(class_config.margin_max)
        proximal = float(proximal_weight)
    except (TypeError, ValueError, AttributeError):
        return held("no_certified_update:invalid_problem_data")

    if (
        not np.isfinite(current_value)
        or not np.isfinite(q_value)
        or not np.isfinite(kappa_value)
        or kappa_value < 0.0
        or not np.isfinite(scale)
        or scale <= 0.0
        or not np.isfinite(weight)
        or weight <= 0.0
        or not np.isfinite(lower)
        or not np.isfinite(upper)
        or lower > upper
        or not np.isfinite(proximal)
        or proximal <= 0.0
    ):
        return held("no_certified_update:invalid_problem_data")

    # Work in d = r_next - r_current.  The two linear inequalities are
    # linear in d and can therefore be intersected directly with the box.
    box_lower = lower - current_value
    box_upper = upper - current_value
    slope = kappa_value / scale
    q_delta = q_value - current_value
    intervals: list[tuple[float, float]] = []

    # d <= 0: q - current <= (1 + kappa/scale) d.
    negative_lower = max(box_lower, q_delta / (1.0 + slope))
    negative_upper = min(box_upper, 0.0)
    if negative_lower <= negative_upper:
        intervals.append((negative_lower, negative_upper))

    # d >= 0: q - current <= (1 - kappa/scale) d.
    positive_lower = max(box_lower, 0.0)
    positive_upper = box_upper
    coefficient = 1.0 - slope
    if coefficient > 0.0:
        positive_lower = max(positive_lower, q_delta / coefficient)
    elif coefficient < 0.0:
        positive_upper = min(positive_upper, q_delta / coefficient)
    elif q_delta > 0.0:
        positive_lower = 1.0
        positive_upper = 0.0
    if positive_lower <= positive_upper:
        intervals.append((positive_lower, positive_upper))

    if not intervals:
        return held("no_certified_update:infeasible")

    unconstrained = current_value - weight * scale / proximal

    def objective(delta: float) -> float:
        return float(
            weight * (current_value + delta) / scale
            + 0.5 * proximal * (delta / scale) ** 2
        )

    candidates: list[tuple[float, float]] = []
    for interval_lower, interval_upper in intervals:
        delta = min(max(unconstrained - current_value, interval_lower), interval_upper)
        candidates.append((objective(delta), delta))
    _, selected_delta = min(candidates, key=lambda item: (item[0], item[1]))
    next_value = current_value + selected_delta
    normalized_change = abs(selected_delta) / scale
    slack = next_value - q_value - kappa_value * normalized_change
    if not np.isfinite(next_value) or not np.isfinite(slack) or slack < -1e-8:
        return held("no_certified_update:residual_check")

    status = "updated" if sensitivity_certified else "updated_noncertified"
    return {
        "r_next": float(next_value),
        "status": status,
        "certified": bool(sensitivity_certified),
        "objective": objective(selected_delta),
        "normalized_change": float(normalized_change),
        "constraint_slack": float(slack),
    }


def hold_margin_update(
    current: dict[Platform, float], status: str = "no_certified_update:invalid_sensitivity"
) -> MarginUpdateResult:
    """Record a skipped update and retain the current margins."""

    return _held_result(current, status, False)
