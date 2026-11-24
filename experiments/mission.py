"""Mission plans for the four quadrants and graphs for task execution and margin updates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from maicp.configuration import ExperimentConfig, Platform
from maicp.types import MissionPlan

from experiments.communication import GraphSnapshot
from experiments.vehicles import VehicleState


QUADRANT_SIGNS = ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))


@dataclass(frozen=True)
class CanonicalMissionFrame:
    """Orthogonal reflection mapping one quadrant mission to its template."""

    subteam_id: int
    signs: tuple[float, float]

    @property
    def matrix(self) -> np.ndarray:
        return np.diag(np.asarray(self.signs, dtype=float))

    def point_to_local(self, point: Sequence[float] | np.ndarray) -> np.ndarray:
        return self.matrix @ np.asarray(point, dtype=float)

    def point_to_global(self, point: Sequence[float] | np.ndarray) -> np.ndarray:
        return self.matrix.T @ np.asarray(point, dtype=float)

    def vector_to_local(self, vector: Sequence[float] | np.ndarray) -> np.ndarray:
        return self.matrix @ np.asarray(vector, dtype=float)

    def vector_to_global(self, vector: Sequence[float] | np.ndarray) -> np.ndarray:
        return self.matrix.T @ np.asarray(vector, dtype=float)


def canonical_mission_frame(subteam_id: int) -> CanonicalMissionFrame:
    """Return the known global/canonical reflection for one mission slot."""

    if not 0 <= subteam_id < len(QUADRANT_SIGNS):
        raise ValueError("Unknown quadrant subteam")
    return CanonicalMissionFrame(subteam_id, QUADRANT_SIGNS[subteam_id])


def _formation_offsets(count: int, spacing: float, radial_shift: float) -> list[np.ndarray]:
    columns = max(1, int(np.ceil(np.sqrt(count))))
    rows = int(np.ceil(count / columns))
    offsets: list[np.ndarray] = []
    for index in range(count):
        row, column = divmod(index, columns)
        offsets.append(
            np.array(
                [
                    radial_shift + spacing * (column - (columns - 1) / 2.0),
                    spacing * (row - (rows - 1) / 2.0),
                ]
            )
        )
    return offsets


def build_mission_plans(config: ExperimentConfig, first_mission_id: int, wave_seed: int) -> tuple[MissionPlan, ...]:
    plans: list[MissionPlan] = []
    for subteam_id, sign_tuple in enumerate(QUADRANT_SIGNS):
        sign = np.asarray(sign_tuple)
        pad_center = sign * config.depot_pad_distance
        anchor = sign * config.quadrant_anchor_distance
        robot_ids: dict[Platform, tuple[str, ...]] = {}
        home_positions: dict[str, tuple[float, float]] = {}
        waypoints: dict[str, tuple[tuple[float, float], ...]] = {}
        for platform, radial_shift in ((Platform.UGV, -0.8), (Platform.UAV, 0.8)):
            count = config.classes[platform].robots_per_mission
            ids = tuple(f"s{subteam_id}-{platform.value}-{index}" for index in range(count))
            robot_ids[platform] = ids
            offsets = _formation_offsets(count, spacing=6.0, radial_shift=radial_shift)
            for robot_id, offset in zip(ids, offsets, strict=True):
                transformed_offset = offset * sign
                home = pad_center + transformed_offset
                corridor = sign * 13.0 + transformed_offset
                loop_offsets = (
                    np.array([-4.5, -4.5]),
                    np.array([4.5, -4.5]),
                    np.array([4.5, 4.5]),
                    np.array([-4.5, 4.5]),
                    np.array([-4.5, -4.5]),
                )
                route = [home, corridor]
                route.extend(anchor + loop_offset * sign + transformed_offset for loop_offset in loop_offsets)
                route.extend([corridor, home])
                home_positions[robot_id] = tuple(float(value) for value in home)
                waypoints[robot_id] = tuple(tuple(float(value) for value in point) for point in route)
        plans.append(
            MissionPlan(
                mission_id=first_mission_id + subteam_id,
                subteam_id=subteam_id,
                seed=wave_seed + 104729 * subteam_id,
                robot_ids=robot_ids,
                home_positions=home_positions,
                waypoints=waypoints,
            )
        )
    return tuple(plans)


def robot_platforms(plans: tuple[MissionPlan, ...]) -> dict[str, Platform]:
    return {
        robot_id: platform
        for plan in plans
        for platform, robot_ids in plan.robot_ids.items()
        for robot_id in robot_ids
    }


def build_distance_graph(
    positions: Mapping[str, Sequence[float] | np.ndarray],
    communication_range: float,
    *,
    nodes: Sequence[str] | None = None,
    require_connected: bool = False,
) -> GraphSnapshot[str]:
    """Connect pairs within communication range. Set require_connected to reject disconnected graphs."""

    if communication_range <= 0.0:
        raise ValueError("Communication range must be positive")
    ids = tuple(sorted(nodes if nodes is not None else positions, key=str))
    if not ids:
        raise ValueError("At least one robot is required to build a graph")
    if any(robot_id not in positions for robot_id in ids):
        raise ValueError("Every graph node must have a position")

    adjacency = {robot_id: set() for robot_id in ids}
    for index, robot_id in enumerate(ids):
        position = np.asarray(positions[robot_id], dtype=float)
        if position.ndim != 1 or position.size == 0 or not np.all(np.isfinite(position)):
            raise ValueError(f"Invalid position for robot {robot_id!r}")
        for neighbor in ids[index + 1 :]:
            other_position = np.asarray(positions[neighbor], dtype=float)
            if other_position.shape != position.shape or not np.all(np.isfinite(other_position)):
                raise ValueError(f"Invalid position for robot {neighbor!r}")
            if float(np.linalg.norm(position - other_position)) <= communication_range:
                adjacency[robot_id].add(neighbor)
                adjacency[neighbor].add(robot_id)
    return GraphSnapshot.from_adjacency(adjacency, require_connected=require_connected)


def build_task_graph(
    plan: MissionPlan,
    communication_range: float,
    *,
    positions: Mapping[str, Sequence[float] | np.ndarray] | None = None,
) -> GraphSnapshot[str]:
    """Construct the subteam communication graph from the current positions, or from home positions at initialization."""

    current_positions = plan.home_positions if positions is None else positions
    ids = tuple(robot_id for values in plan.robot_ids.values() for robot_id in values)
    return build_distance_graph(
        current_positions,
        communication_range,
        nodes=ids,
        require_connected=False,
    )


def build_robust_nominal_task_graph(
    plan: MissionPlan,
    communication_range: float,
    position_uncertainty_radius: float,
    *,
    require_connected: bool = False,
) -> GraphSnapshot[str]:
    """Select fixed edges guaranteed feasible for every allowed initial jitter.

    An edge is selected only when its nominal separation plus the two endpoint
    uncertainty radii is within range.  Realized positions never add or remove
    an edge from this graph.
    """

    if position_uncertainty_radius < 0.0 or not np.isfinite(position_uncertainty_radius):
        raise ValueError("Position uncertainty radius must be finite and nonnegative")
    if communication_range <= 0.0 or not np.isfinite(communication_range):
        raise ValueError("Communication range must be finite and positive")
    ids = tuple(robot_id for values in plan.robot_ids.values() for robot_id in values)
    adjacency = {robot_id: set() for robot_id in ids}
    allowance = 2.0 * float(position_uncertainty_radius)
    for index, robot_id in enumerate(ids):
        position = np.asarray(plan.home_positions[robot_id], dtype=float)
        for neighbor in ids[index + 1 :]:
            other = np.asarray(plan.home_positions[neighbor], dtype=float)
            if float(np.linalg.norm(position - other)) + allowance <= communication_range:
                adjacency[robot_id].add(neighbor)
                adjacency[neighbor].add(robot_id)
    return GraphSnapshot.from_adjacency(adjacency, require_connected=require_connected)


def verify_graph_within_range(
    graph: GraphSnapshot[str],
    positions: Mapping[str, Sequence[float] | np.ndarray],
    communication_range: float,
) -> bool:
    """Return whether every frozen edge is feasible at the supplied positions."""

    return all(
        float(np.linalg.norm(
            np.asarray(positions[left], dtype=float)
            - np.asarray(positions[right], dtype=float)
        )) <= communication_range + 1e-12
        for left, right in graph.edges
    )


def build_task_graph_from_positions(
    plan: MissionPlan,
    positions: Mapping[str, Sequence[float] | np.ndarray],
    communication_range: float,
) -> GraphSnapshot[str]:
    """Explicit alias for constructing a task graph during a physical step."""

    return build_task_graph(plan, communication_range, positions=positions)


def build_current_task_graph(
    plan: MissionPlan,
    positions: Mapping[str, Sequence[float] | np.ndarray],
    communication_range: float,
) -> GraphSnapshot[str]:
    """Readable alias used by application simulators for online graph updates."""

    return build_task_graph_from_positions(plan, positions, communication_range)


def build_update_graph(
    vehicles: Mapping[str, VehicleState],
    communication_range: float,
    *,
    require_connected: bool = False,
) -> GraphSnapshot[str]:
    """Construct the update graph from the robot positions after the mission."""

    positions = {robot_id: vehicle.position for robot_id, vehicle in vehicles.items()}
    return build_distance_graph(
        positions,
        communication_range,
        require_connected=require_connected,
    )


@dataclass
class JointMissionState:
    plans: tuple[MissionPlan, ...]
    vehicles: dict[str, VehicleState]
    task_graphs: dict[int, GraphSnapshot[str]]
    task_graph_history: list[dict[int, GraphSnapshot[str]]] = field(default_factory=list)
    update_graph: GraphSnapshot[str] | None = None
    task_communication_range: float = 0.0
    update_communication_range: float = 0.0

    @classmethod
    def create(cls, config: ExperimentConfig, plans: tuple[MissionPlan, ...]) -> "JointMissionState":
        platforms = robot_platforms(plans)
        vehicles = {
            robot_id: VehicleState.from_plan(
                robot_id,
                platforms[robot_id],
                plan.home_positions[robot_id],
                plan.waypoints[robot_id],
                config.uav_dynamics,
            )
            for plan in plans
            for robot_id in plan.home_positions
        }
        state = cls(
            plans=plans,
            vehicles=vehicles,
            task_graphs={},
            task_communication_range=config.task_communication_range,
            update_communication_range=config.update_communication_range,
        )
        state.refresh_task_graphs()
        return state

    def refresh_task_graphs(
        self,
        config: ExperimentConfig | None = None,
    ) -> dict[int, GraphSnapshot[str]]:
        """Refresh each subteam graph from the current physical positions.

        A fresh snapshot is appended to ``task_graph_history`` on every call,
        which gives simulators a lightweight source for edge counts and
        connectivity diagnostics.  The returned mapping is a shallow copy, so
        callers cannot replace the state's graph entries accidentally.
        """

        communication_range = (
            self.task_communication_range
            if config is None
            else config.task_communication_range
        )
        if communication_range <= 0.0:
            raise ValueError("Task communication range must be positive")
        current: dict[int, GraphSnapshot[str]] = {}
        for plan in self.plans:
            positions = {
                robot_id: self.vehicles[robot_id].position
                for robot_id in plan.home_positions
            }
            current[plan.subteam_id] = build_task_graph(
                plan,
                communication_range,
                positions=positions,
            )
        self.task_graphs = current
        self.task_graph_history.append(dict(current))
        return dict(current)

    def current_task_graphs(
        self,
        config: ExperimentConfig | None = None,
    ) -> dict[int, GraphSnapshot[str]]:
        """Refresh and return task graphs for the current vehicle positions.

        Applications should call this once per physical simulation update and
        pass the returned snapshots to their local algorithms.  The explicit
        method makes it difficult to accidentally reuse the initial
        graph after the robots have moved.
        """

        return self.refresh_task_graphs(config)

    def freeze_update_graph(
        self,
        config: ExperimentConfig | None = None,
        *,
        require_connected: bool = True,
    ) -> GraphSnapshot[str]:
        """Keep a fixed graph for all communications in the calibration round."""

        if self.update_graph is None:
            communication_range = (
                self.update_communication_range
                if config is None
                else config.update_communication_range
            )
            self.update_graph = build_update_graph(
                self.vehicles,
                communication_range,
                require_connected=require_connected,
            )
        elif require_connected and not self.update_graph.is_connected:
            raise ValueError("The frozen update graph is disconnected")
        return self.update_graph

    def update_graph_connected(
        self,
        config: ExperimentConfig,
    ) -> tuple[bool, GraphSnapshot[str]]:
        """Record connectivity when freezing the update graph."""

        graph = self.freeze_update_graph(config, require_connected=False)
        return graph.is_connected, graph
