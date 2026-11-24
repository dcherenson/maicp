"""Independent scalar interactive conformal prediction for individual robots."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from maicp.ranks import binomial_calibration_rank
from typing import Any

import numpy as np

from maicp.configuration import ClassConfig, ExperimentConfig, Platform

from maicp.score_stores import NodeCalibrationStore
from .update import solve_scalar_margin_update


def own_score_history(
    node_store: NodeCalibrationStore,
    platform: Platform,
) -> tuple[float, ...]:
    """Read one robot's scores in mission order."""

    if not isinstance(node_store, NodeCalibrationStore):
        raise TypeError("node_store must be a NodeCalibrationStore")
    try:
        mission_values = node_store.robot_scores[platform]
    except (KeyError, TypeError):
        raise ValueError(f"No {platform.value} robot-score store is available") from None

    history: list[float] = []
    for mission_id in sorted(mission_values):
        raw_value = mission_values[mission_id]
        if isinstance(raw_value, Mapping):
            # Distributed calibration views may create an empty placeholder
            # for a node that did not participate in a mission.  It is not a
            # local sample and therefore does not enter this robot's history.
            if not raw_value:
                continue
            if len(raw_value) != 1:
                raise ValueError(
                    f"Mission {mission_id} must contain exactly one locally owned score"
                )
            raw_value = next(iter(raw_value.values()))
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            raise ValueError(f"Mission {mission_id} has a malformed local score") from None
        history.append(value)
    return tuple(history)


def single_agent_conformal_quantile(
    score_history: Sequence[float],
    class_config: ClassConfig,
) -> dict[str, float | int | None]:
    """Return the binomial calibration rank and exact scalar quantile for one robot.

    The rank is the same correction for a finite sample used by MA-ICP:

    ``rank = 1 + Binom(n, 1-alpha).ppf(1-delta_cal)``.

    The function is pure and has no requirement for a current margin, making
    it suitable for terminal evaluation records where no update is applied.
    """

    try:
        values = tuple(float(value) for value in score_history)
    except (TypeError, ValueError):
        return {"q": None, "rank": None, "count": 0}
    count = len(values)
    if count == 0:
        return {"q": None, "rank": None, "count": 0}
    if any(not np.isfinite(value) for value in values):
        return {"q": None, "rank": None, "count": count}

    try:
        rank = binomial_calibration_rank(count, float(class_config.alpha), float(class_config.delta_cal))
    except (TypeError, ValueError, OverflowError):
        return {"q": None, "rank": None, "count": count}
    return {
        "q": float(sorted(values)[rank - 1]),
        "rank": int(rank),
        "count": int(count),
    }


def _invalid_result(
    current_margin: object,
    kappa: object,
    *,
    q: float | None = None,
    rank: int | None = None,
    count: int = 0,
    status: str,
) -> dict[str, Any]:
    try:
        current = float(current_margin)
        if not np.isfinite(current):
            current = float("nan")
    except (TypeError, ValueError):
        current = float("nan")
    try:
        kappa_value = float(kappa)
        if not np.isfinite(kappa_value):
            kappa_value = float("nan")
    except (TypeError, ValueError):
        kappa_value = float("nan")
    return {
        "q": q,
        "rank": rank,
        "count": int(count),
        "kappa": kappa_value,
        "r_next": current,
        "status": status,
        "certified": False,
    }


def single_agent_icp_update(
    config: ExperimentConfig,
    platform: Platform,
    current_margin: float,
    node_store: NodeCalibrationStore | None = None,
    *,
    score_history: Sequence[float] | None = None,
    sensitivity_certified: bool = True,
) -> dict[str, Any]:
    """Compute a scalar margin update from one robot's score history. An explicit history takes precedence over the node store."""

    if not isinstance(config, ExperimentConfig):
        raise TypeError("config must be an ExperimentConfig")
    try:
        class_config = config.classes[platform]
    except (KeyError, TypeError):
        raise ValueError(f"No class configuration is available for {platform!r}") from None
    kappa = class_config.configured_kappa

    if score_history is None:
        if node_store is None:
            return _invalid_result(
                current_margin,
                kappa,
                status="no_certified_update:no_scores",
            )
        try:
            score_history = own_score_history(node_store, platform)
        except (TypeError, ValueError):
            return _invalid_result(
                current_margin,
                kappa,
                status="no_certified_update:invalid_score_store",
            )

    quantile = single_agent_conformal_quantile(score_history, class_config)
    q = quantile["q"]
    rank = quantile["rank"]
    count = int(quantile["count"] or 0)
    if q is None or rank is None:
        status = (
            "no_certified_update:no_scores"
            if count == 0
            else "no_certified_update:invalid_score"
        )
        return _invalid_result(
            current_margin,
            kappa,
            q=None,
            rank=None,
            count=count,
            status=status,
        )

    update = solve_scalar_margin_update(
        current=current_margin,
        threshold=float(q),
        kappa=float(kappa),
        class_config=class_config,
        proximal_weight=float(config.proximal_weight),
        sensitivity_certified=sensitivity_certified,
    )
    return {
        "q": float(q),
        "rank": int(rank),
        "count": count,
        "kappa": float(kappa),
        "r_next": float(update["r_next"]),
        "status": str(update["status"]),
        "certified": bool(update["certified"]),
        "objective": update["objective"],
        "normalized_change": update["normalized_change"],
        "constraint_slack": update["constraint_slack"],
    }


__all__ = [
    "own_score_history",
    "single_agent_conformal_quantile",
    "single_agent_icp_update",
]
