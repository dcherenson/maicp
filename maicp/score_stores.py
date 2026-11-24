"""Calibration scores held by each robot and bounds on mission order statistics."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Sequence
from copy import deepcopy

from maicp.configuration import Platform


@dataclass
class NodeCalibrationStore:
    robot_scores: dict[Platform, dict[int, dict[str, float]]] = field(
        default_factory=lambda: {platform: {} for platform in Platform}
    )
    mission_certificates: dict[Platform, dict[str, float]] = field(
        default_factory=lambda: {platform: {} for platform in Platform}
    )
    analytic_kappa: float | None = None
    configured_kappa: float | None = None
    empirical_kappa: float | None = None


@dataclass
class LocalCalibrationStores:
    """Each key is a physical update node and each value is its persistent state."""

    node_stores: dict[str, NodeCalibrationStore] = field(default_factory=dict)

    def at(self, robot_id: str) -> NodeCalibrationStore:
        return self.node_stores.setdefault(robot_id, NodeCalibrationStore())

    def record_robot_score(
        self, mission_id: int, platform: Platform, robot_id: str, score: float
    ) -> None:
        local = self.at(robot_id).robot_scores[platform]
        if mission_id in local:
            raise ValueError(f"Duplicate local score for {robot_id} mission {mission_id}")
        local[mission_id] = {
            f"robot:{mission_id}:{platform.value}:{robot_id}": float(score)
        }

    def record_analytic_kappa(self, robot_id: str, value: float) -> None:
        self.at(robot_id).analytic_kappa = float(value)

    def record_empirical_kappa(self, robot_id: str, value: float) -> None:
        self.at(robot_id).empirical_kappa = float(value)

    def record_configured_kappa(self, robot_id: str, value: float) -> None:
        self.at(robot_id).configured_kappa = float(value)

    def record_mission_certificate(
        self, mission_id: int, platform: Platform, owner: str, score: float
    ) -> None:
        tag = f"mission:{mission_id}:{platform.value}"
        for node, store in self.node_stores.items():
            if tag in store.mission_certificates[platform]:
                raise ValueError(
                    f"Mission {mission_id} {platform.value} already owned by {node}"
                )
        self.at(owner).mission_certificates[platform][tag] = float(score)

    def merge_from(self, other: "LocalCalibrationStores") -> None:
        """Merge a disjoint worker contribution in deterministic key order."""

        for robot_id in sorted(other.node_stores):
            source = other.node_stores[robot_id]
            target = self.at(robot_id)
            for platform in Platform:
                for mission_id in sorted(source.robot_scores[platform]):
                    if mission_id in target.robot_scores[platform]:
                        raise ValueError(
                            f"Duplicate local score for {robot_id} mission {mission_id}"
                        )
                    target.robot_scores[platform][mission_id] = deepcopy(
                        source.robot_scores[platform][mission_id]
                    )
                for tag in sorted(source.mission_certificates[platform]):
                    if tag in target.mission_certificates[platform]:
                        raise ValueError(
                            f"Duplicate mission certificate {tag!r} for {robot_id}"
                        )
                    target.mission_certificates[platform][tag] = float(
                        source.mission_certificates[platform][tag]
                    )
            for name in ("analytic_kappa", "configured_kappa", "empirical_kappa"):
                value = getattr(source, name)
                if value is None:
                    continue
                existing = getattr(target, name)
                if existing is not None and existing != value:
                    raise ValueError(f"Conflicting {name} for {robot_id}")
                setattr(target, name, float(value))

    def robot_score_stores(
        self, mission_id: int, platform: Platform, nodes: Sequence[str]
    ) -> dict[str, dict[str, float]]:
        """Return references to the scores stored at each node for this mission."""

        result: dict[str, dict[str, float]] = {}
        for node in nodes:
            local = self.at(node).robot_scores[platform]
            result[node] = local.setdefault(mission_id, {})
        return result

    def mission_score_stores(
        self, platform: Platform, nodes: Sequence[str]
    ) -> dict[str, dict[str, float]]:
        return {
            node: self.at(node).mission_certificates[platform]
            for node in nodes
        }
