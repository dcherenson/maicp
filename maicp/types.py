"""Data exchanged between simulators and the shared MA-ICP implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from maicp.configuration import Platform


@dataclass(frozen=True)
class MissionPlan:
    mission_id: int
    subteam_id: int
    seed: int
    robot_ids: dict[Platform, tuple[str, ...]]
    home_positions: dict[str, tuple[float, float]]
    waypoints: dict[str, tuple[tuple[float, float], ...]]


@dataclass
class MissionResult:
    mission_id: int
    subteam_id: int
    seed: int
    robot_scores: dict[Platform, dict[str, float]]
    returned_home: bool
    update_graph_connected: bool
    failed: bool = False
    failure_reason: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    # Mission metadata. The trace stores time series.
    audit: dict[str, Any] = field(default_factory=dict)
    # ``failed`` is reserved for a global mission failure.  Local
    # application outcomes are independent of the conformal robot scores.
    # Nonfinite scores are reserved for genuine score/data failures.
    # These fields are appended to preserve positional compatibility.
    failed_robots: dict[Platform, tuple[str, ...]] = field(default_factory=dict)
    robot_failure_reasons: dict[str, tuple[str, ...]] = field(default_factory=dict)
    # Fixed subteam topology used after this mission to aggregate its robot scores.
    mission_graph: dict[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class OrderStatisticResult:
    value: float
    exact_value: float | None
    rank: int
    pinball_history: np.ndarray
    flooding_rounds: int
    recovery_steps: int
    owners: dict[str, tuple[str, ...]]

    pinball_rounds: int = 0
    total_communication_rounds: int | None = None
    certified: bool = True

    @property
    def communication_rounds(self) -> int:
        """Total communication rounds for pinball iterations and certified recovery."""

        if self.total_communication_rounds is not None:
            return int(self.total_communication_rounds)
        return int(self.pinball_rounds) + int(self.flooding_rounds)

    @property
    def total_rounds(self) -> int:
        """Alias used by reports and tests for the full communication count."""

        return self.communication_rounds

    @property
    def order_statistic_flooding_rounds(self) -> int:
        """Flooding rounds used to recover the order statistic."""

        return int(self.flooding_rounds)


@dataclass(frozen=True)
class SensitivityResult:
    robot_kappa: dict[Platform, dict[str, float]]
    class_kappa: dict[Platform, float]
    mode: str
    certified: bool
    valid: bool
    max_validation_ratio: dict[Platform, float]
    notes: tuple[str, ...] = ()
    paired_validation_valid: bool = True
    kappa_flooding_rounds: int = 0


@dataclass(frozen=True)
class MarginUpdateResult:
    margins: dict[Platform, float]
    previous_margins: dict[Platform, float]
    status: str
    certified: bool
    objective: float | None
    normalized_change: float
    constraint_slacks: dict[Platform, float]
    margin_consensus_residual: float = 0.0
    replica_statuses: dict[str, str] = field(default_factory=dict)
    replica_failures: tuple[str, ...] = ()


@dataclass(frozen=True)
class MAICPRoundResult:
    round_index: int
    mission_scores: dict[Platform, tuple[float, ...]]
    certified_mission_scores: dict[Platform, tuple[float, ...]]
    exact_thresholds: dict[Platform, float]
    certified_thresholds: dict[Platform, float]
    conformal_ranks: dict[Platform, int]
    sensitivity: SensitivityResult
    margin_update: MarginUpdateResult
    communication_rounds: int
    order_statistic_histories: dict[Platform, dict[str, np.ndarray]] = field(default_factory=dict)
    mission_certificate_owners: dict[Platform, dict[int, str]] = field(default_factory=dict)
    pinball_rounds: int = 0
    flooding_rounds: int = 0
    kappa_flooding_rounds: int = 0

    @property
    def order_statistic_flooding_rounds(self) -> int:
        """Flooding rounds across all distributed order statistics."""

        return int(self.flooding_rounds)
