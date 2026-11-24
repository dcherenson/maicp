"""Dynamic average consensus on the formation communication ring."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Hashable

import numpy as np

from experiments.communication import GraphSnapshot, SynchronousMessageBus


Node = Hashable
Vector = np.ndarray


def _node_tuple(node_ids: Sequence[Node] | int = 6) -> tuple[Node, ...]:
    if isinstance(node_ids, bool):
        raise ValueError("node count must be an integer")
    if isinstance(node_ids, (int, np.integer)):
        count = int(node_ids)
        if count < 4 or count % 2:
            raise ValueError("the alternating ring requires an even node count >= 4")
        return tuple(range(count))
    nodes = tuple(node_ids)
    if len(nodes) < 4 or len(nodes) % 2:
        raise ValueError("the alternating ring requires an even node count >= 4")
    if len(set(nodes)) != len(nodes):
        raise ValueError("ring node IDs must be unique")
    return nodes


def alternating_ring_neighbors(
    node_ids: Sequence[Node] | int = 6,
) -> dict[Node, tuple[Node, Node]]:
    """Return the alternating ring with two neighbors per robot.

    The input order is the protocol order and should alternate UGV/UAV IDs,
    for example ``(ugv0, uav0, ugv1, uav1, ugv2, uav2)``. The function does
    not inspect ID strings, so callers can use any stable robot identifiers.
    """

    nodes = _node_tuple(node_ids)
    return {
        node: (nodes[(index - 1) % len(nodes)], nodes[(index + 1) % len(nodes)])
        for index, node in enumerate(nodes)
    }


def ring_mixing_matrix(node_ids: Sequence[Node] | int = 6) -> np.ndarray:
    """Return the symmetric, doubly stochastic mixing matrix for the ring."""

    nodes = _node_tuple(node_ids)
    matrix = np.zeros((len(nodes), len(nodes)), dtype=float)
    for index in range(len(nodes)):
        matrix[index, index] = 1.0 / 3.0
        matrix[index, (index - 1) % len(nodes)] = 1.0 / 3.0
        matrix[index, (index + 1) % len(nodes)] = 1.0 / 3.0
    return matrix


def alternating_ring_mixing_matrix(node_ids: Sequence[Node] | int = 6) -> np.ndarray:
    """Descriptive alias for :func:`ring_mixing_matrix`."""

    return ring_mixing_matrix(node_ids)


def center_average(observations: Mapping[Node, Sequence[float] | Vector]) -> Vector:
    """Return the instantaneous diagnostic average of local observations."""

    if not observations:
        raise ValueError("at least one center observation is required")
    values = np.asarray([_vector(observation, name="center observation") for observation in observations.values()])
    return np.mean(values, axis=0)


def _vector(value: Sequence[float] | Vector, *, name: str) -> Vector:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite nonempty vector")
    return array.copy()


def _snapshot(values: Mapping[Node, Sequence[float] | Vector], nodes: Sequence[Node], *, name: str) -> dict[Node, Vector]:
    missing = set(nodes).difference(values)
    extra = set(values).difference(nodes)
    if missing or extra:
        raise ValueError(f"{name} keys do not match the consensus population")
    result = {node: _vector(values[node], name=f"{name}[{node!r}]") for node in nodes}
    shapes = {value.shape for value in result.values()}
    if len(shapes) != 1:
        raise ValueError(f"{name} vectors must have the same shape")
    return result


@dataclass
class SynchronousNeighborBus:
    """A synchronous message bus for a fixed undirected graph.

    ``exchange`` takes one value per sender and returns only messages received
    from the configured neighbours. Values are copied at the exchange
    boundary, so a receiver cannot observe a sender's later changes to the packet.
    """

    neighbors: Mapping[Node, Sequence[Node]]
    exchange_count: int = 0
    last_received: dict[Node, dict[Node, Vector]] = field(default_factory=dict, init=False)
    last_deliveries: tuple[tuple[Node, Node], ...] = field(default=(), init=False)
    _nodes: tuple[Node, ...] = field(default=(), init=False, repr=False)
    _bus: SynchronousMessageBus[Node, Vector] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        nodes = tuple(self.neighbors)
        if not nodes or len(set(nodes)) != len(nodes):
            raise ValueError("a message bus requires unique nonempty node IDs")
        node_set = set(nodes)
        normalized: dict[Node, tuple[Node, ...]] = {}
        for node in nodes:
            adjacent = tuple(self.neighbors[node])
            if len(set(adjacent)) != len(adjacent):
                raise ValueError("neighbour lists must not contain duplicates")
            if node in adjacent or any(neighbor not in node_set for neighbor in adjacent):
                raise ValueError("neighbour lists must contain other known nodes only")
            normalized[node] = adjacent
        for node, adjacent in normalized.items():
            for neighbor in adjacent:
                if node not in normalized[neighbor]:
                    raise ValueError("message-bus graph must be undirected")
        self.neighbors = normalized
        self._nodes = nodes
        graph = GraphSnapshot.from_adjacency(normalized, require_connected=False)
        self._bus = SynchronousMessageBus(graph)

    @property
    def nodes(self) -> tuple[Node, ...]:
        return self._nodes

    def exchange(
        self,
        messages: Mapping[Node, Sequence[float] | Vector],
    ) -> dict[Node, dict[Node, Vector]]:
        snapshot = _snapshot(messages, self.nodes, name="messages")
        if self._bus is None:  # pragma: no cover - initialized in __post_init__
            raise RuntimeError("message bus is not initialized")
        inbox = self._bus.broadcast(snapshot)
        received = {
            receiver: {
                sender: _vector(packet, name=f"message[{sender!r}]")
                for sender, packet in inbox[receiver].items()
            }
            for receiver in self.nodes
        }
        self.exchange_count += 1
        self.last_received = {
            receiver: {sender: value.copy() for sender, value in incoming.items()}
            for receiver, incoming in received.items()
        }
        self.last_deliveries = self._bus.last_deliveries
        return received


@dataclass
class DynamicAverageConsensus:
    """One local consensus state per robot in a fixed alternating ring."""

    node_ids: tuple[Node, ...] = (0, 1, 2, 3, 4, 5)
    rounds: int = 1
    bus: SynchronousNeighborBus | None = None
    estimates: dict[Node, Vector] | None = field(default=None, init=False)
    observations: dict[Node, Vector] | None = field(default=None, init=False)
    step_index: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.node_ids = _node_tuple(self.node_ids)
        if isinstance(self.rounds, bool) or not isinstance(self.rounds, (int, np.integer)):
            raise ValueError("consensus rounds must be a positive integer")
        self.rounds = int(self.rounds)
        if self.rounds < 1:
            raise ValueError("consensus rounds must be a positive integer")
        if self.bus is None:
            self.bus = SynchronousNeighborBus(alternating_ring_neighbors(self.node_ids))
        elif self.bus.nodes != self.node_ids:
            raise ValueError("consensus bus node order must match node_ids")

    @property
    def initialized(self) -> bool:
        return self.estimates is not None and self.observations is not None

    def initialize(
        self,
        observations: Mapping[Node, Sequence[float] | Vector],
    ) -> dict[Node, Vector]:
        """Set ``z_i = y_i`` locally and return copied initial estimates."""

        values = _snapshot(observations, self.node_ids, name="observations")
        self.observations = {node: value.copy() for node, value in values.items()}
        self.estimates = {node: value.copy() for node, value in values.items()}
        self.step_index = 0
        return {node: value.copy() for node, value in self.estimates.items()}

    def _mix_once(self, values: Mapping[Node, Vector]) -> dict[Node, Vector]:
        assert self.bus is not None
        # Each exchange reads the previous synchronous state.
        incoming = self.bus.exchange(values)
        return {
            node: (
                values[node]
                + incoming[node][self.bus.neighbors[node][0]]
                + incoming[node][self.bus.neighbors[node][1]]
            )
            / 3.0
            for node in self.node_ids
        }

    def step(
        self,
        next_observations: Mapping[Node, Sequence[float] | Vector],
    ) -> dict[Node, Vector]:
        """Advance one physical step using only the new local observations."""

        if not self.initialized:
            raise RuntimeError("initialize local center observations before step")
        assert self.estimates is not None and self.observations is not None
        next_values = _snapshot(next_observations, self.node_ids, name="observations")
        previous_estimates = {node: value.copy() for node, value in self.estimates.items()}
        mixed = previous_estimates
        for _ in range(self.rounds):
            mixed = self._mix_once(mixed)
        self.estimates = {
            node: mixed[node] + next_values[node] - self.observations[node]
            for node in self.node_ids
        }
        self.observations = {node: value.copy() for node, value in next_values.items()}
        self.step_index += 1
        return {node: value.copy() for node, value in self.estimates.items()}

    update = step

    def local_estimate(self, node: Node) -> Vector:
        """Read one local estimate without exposing mutable internal state."""

        if not self.initialized or self.estimates is None:
            raise RuntimeError("consensus has not been initialized")
        if node not in self.estimates:
            raise KeyError(node)
        return self.estimates[node].copy()


def dynamic_average_consensus_step(
    estimates: Mapping[Node, Sequence[float] | Vector],
    previous_observations: Mapping[Node, Sequence[float] | Vector],
    next_observations: Mapping[Node, Sequence[float] | Vector],
    *,
    node_ids: Sequence[Node] | None = None,
    rounds: int = 1,
    bus: SynchronousNeighborBus | None = None,
) -> dict[Node, Vector]:
    """Perform one causal update from explicit local state mappings.

    This helper is useful when a simulator owns its state containers. It uses
    exactly the same synchronous neighbour bus as :class:`DynamicAverageConsensus`
    and never computes a centralized matrix product.
    """

    ids = _node_tuple(tuple(estimates) if node_ids is None else node_ids)
    previous = _snapshot(previous_observations, ids, name="previous_observations")
    current = _snapshot(estimates, ids, name="estimates")
    next_values = _snapshot(next_observations, ids, name="next_observations")
    worker = DynamicAverageConsensus(ids, rounds=rounds, bus=bus)
    worker.estimates = current
    worker.observations = previous
    return worker.step(next_values)


def nominal_formation_control(
    position: Sequence[float] | Vector,
    velocity: Sequence[float] | Vector,
    offset: Sequence[float] | Vector,
    center_estimate: Sequence[float] | Vector,
    center_reference: Sequence[float] | Vector,
    reference_velocity: Sequence[float] | Vector,
    reference_acceleration: Sequence[float] | Vector,
    *,
    formation_gain: float,
    goal_gain: float,
    velocity_gain: float,
) -> Vector:
    """Compute a_ref + k_form(z + offset − p) + k_goal(c_ref − z) − k_vel(v − v_ref) using the local center estimate z."""

    p = _vector(position, name="position")
    v = _vector(velocity, name="velocity")
    delta = _vector(offset, name="offset")
    z = _vector(center_estimate, name="center_estimate")
    c_ref = _vector(center_reference, name="center_reference")
    v_ref = _vector(reference_velocity, name="reference_velocity")
    a_ref = _vector(reference_acceleration, name="reference_acceleration")
    if any(value <= 0.0 or not np.isfinite(value) for value in (formation_gain, goal_gain, velocity_gain)):
        raise ValueError("nominal gains must be finite and positive")
    if np.isclose(float(formation_gain), float(goal_gain), rtol=0.0, atol=1e-15):
        raise ValueError("formation and goal gains must be distinct")
    return (
        a_ref
        + float(formation_gain) * (z + delta - p)
        + float(goal_gain) * (c_ref - z)
        - float(velocity_gain) * (v - v_ref)
    )


formation_nominal_acceleration = nominal_formation_control


__all__ = [
    "SynchronousNeighborBus",
    "DynamicAverageConsensus",
    "alternating_ring_neighbors",
    "ring_mixing_matrix",
    "alternating_ring_mixing_matrix",
    "center_average",
    "dynamic_average_consensus_step",
    "nominal_formation_control",
    "formation_nominal_acceleration",
]
