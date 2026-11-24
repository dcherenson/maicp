"""Synchronous communication between neighbors in the robot graph.

The simulator runs all robots in one Python process, but distributed
algorithms should still observe the same information boundary as the physical
robots.  :class:`GraphSnapshot` freezes an undirected communication graph for
one synchronous exchange.  :class:`SynchronousMessageBus` then delivers only
messages addressed across an edge of that snapshot.

A new graph snapshot (and bus) should be created when a task
graph changes.  The update graph used by MA-ICP can instead be frozen once at
the depot and reused for the whole update computation.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Generic, TypeVar


Node = TypeVar("Node", bound=Hashable)
Message = TypeVar("Message")


class CommunicationError(ValueError):
    """Raised when a message violates a graph snapshot boundary."""


@dataclass(frozen=True)
class GraphSnapshot(Generic[Node], Mapping[Node, frozenset[Node]]):
    """Immutable adjacency snapshot for one synchronous communication phase.

    The constructor normalizes every neighbor collection to ``frozenset`` and
    wraps the adjacency map in ``MappingProxyType``.  Mutating either the
    caller's original dictionary or a returned neighbor collection therefore
    cannot change the graph used by a bus.
    """

    _adjacency: Mapping[Node, frozenset[Node]]
    _edges: frozenset[tuple[Node, Node]]

    def __init__(self, adjacency: Mapping[Node, Mapping[Node, object] | set[Node] | frozenset[Node] | tuple[Node, ...] | list[Node]], *, require_connected: bool = False) -> None:
        nodes = tuple(sorted(adjacency, key=repr))
        if not nodes:
            raise ValueError("Communication graph must contain at least one node")

        normalized: dict[Node, frozenset[Node]] = {}
        for node in nodes:
            neighbors = frozenset(adjacency[node])
            if node in neighbors:
                raise ValueError("Communication graph must not contain self edges")
            unknown = neighbors.difference(adjacency)
            if unknown:
                raise ValueError(f"Communication graph references unknown nodes: {sorted(unknown, key=repr)!r}")
            for neighbor in neighbors:
                if node not in adjacency[neighbor]:
                    raise ValueError("Communication graph must be undirected")
            normalized[node] = neighbors

        edges = frozenset(
            (left, right)
            for left in nodes
            for right in normalized[left]
            if repr(left) < repr(right)
        )
        object.__setattr__(self, "_adjacency", MappingProxyType(normalized))
        object.__setattr__(self, "_edges", edges)

        if require_connected and not self.is_connected:
            raise ValueError("Communication graph must be connected")

    @classmethod
    def from_adjacency(
        cls,
        adjacency: Mapping[Node, Mapping[Node, object] | set[Node] | frozenset[Node] | tuple[Node, ...] | list[Node]],
        *,
        require_connected: bool = False,
    ) -> "GraphSnapshot[Node]":
        """Build a validated immutable snapshot from an adjacency mapping."""

        return cls(adjacency, require_connected=require_connected)

    def __getitem__(self, node: Node) -> frozenset[Node]:
        return self._adjacency[node]

    def __iter__(self) -> Iterator[Node]:
        return iter(self._adjacency)

    def __len__(self) -> int:
        return len(self._adjacency)

    @property
    def nodes(self) -> tuple[Node, ...]:
        """The deterministically ordered nodes in this snapshot."""

        return tuple(self._adjacency)

    @property
    def edges(self) -> frozenset[tuple[Node, Node]]:
        """Undirected edges represented once using deterministic endpoint order."""

        return self._edges

    @property
    def is_connected(self) -> bool:
        if not self._adjacency:
            return False
        start = next(iter(self._adjacency))
        seen = {start}
        frontier = [start]
        while frontier:
            node = frontier.pop()
            for neighbor in self._adjacency[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    frontier.append(neighbor)
        return len(seen) == len(self._adjacency)

    def has_edge(self, sender: Node, receiver: Node) -> bool:
        """Return whether a directed send is permitted by this undirected graph."""

        return sender in self._adjacency and receiver in self._adjacency.get(sender, frozenset())

    def neighbors(self, node: Node) -> frozenset[Node]:
        """Return the immutable neighbor set for ``node``."""

        return self._adjacency[node]


Inbox = dict[Node, dict[Node, Message]]


class SynchronousMessageBus(Generic[Node, Message]):
    """Deliver one synchronous round of messages between neighbors.

    ``broadcast`` implements the common ``outbound[node] -> payload`` form:
    one payload is delivered to every current neighbor.  ``exchange`` accepts
    the following form ``outbound[sender][receiver] -> payload``.
    Both methods take a snapshot of the outbound mapping before delivery, so a
    receiver cannot observe messages produced by another receiver in the same
    round.  ``send`` is a method for sending to one recipient and
    rejects recipients outside the neighbor set.
    """

    def __init__(self, graph: GraphSnapshot[Node] | Mapping[Node, Mapping[Node, object] | set[Node] | frozenset[Node] | tuple[Node, ...] | list[Node]]) -> None:
        self._graph = graph if isinstance(graph, GraphSnapshot) else GraphSnapshot.from_adjacency(graph)
        self._last_deliveries: tuple[tuple[Node, Node], ...] = ()
        self._invalid_delivery_count = 0

    @property
    def graph(self) -> GraphSnapshot[Node]:
        """The immutable graph snapshot used for every exchange on this bus."""

        return self._graph

    @property
    def last_deliveries(self) -> tuple[tuple[Node, Node], ...]:
        """Deterministic edge list delivered by the most recent round."""

        return self._last_deliveries

    @property
    def invalid_delivery_count(self) -> int:
        """Number of rejected sender, recipient, or nonedge attempts."""

        return self._invalid_delivery_count

    def _empty_inbox(self) -> Inbox[Node, Message]:
        return {node: {} for node in self._graph.nodes}

    def send(self, sender: Node, receiver: Node, message: Message) -> Inbox[Node, Message]:
        """Deliver a message to a neighbor of the sender."""

        self._validate_sender(sender)
        self._validate_receiver(sender, receiver)
        inbox = self._empty_inbox()
        # A delivery is a value boundary.  In the simulator sender and
        # receiver live in the same process, so retaining ``message`` here
        # would let changes by the receiver alter the sender's packet (and,
        # for a broadcast, every other recipient's packet).
        inbox[receiver][sender] = deepcopy(message)
        self._last_deliveries = ((sender, receiver),)
        return inbox

    def broadcast(self, outbound: Mapping[Node, Message]) -> Inbox[Node, Message]:
        """Broadcast each sender's payload to its current neighbors."""

        snapshot = dict(outbound)
        for sender in snapshot:
            self._validate_sender(sender)
        inbox = self._empty_inbox()
        deliveries: list[tuple[Node, Node]] = []
        for sender in self._graph.nodes:
            if sender not in snapshot:
                continue
            for receiver in sorted(self._graph.neighbors(sender), key=repr):
                inbox[receiver][sender] = deepcopy(snapshot[sender])
                deliveries.append((sender, receiver))
        self._last_deliveries = tuple(deliveries)
        return inbox

    def exchange(self, outbound: Mapping[Node, Mapping[Node, Message]]) -> Inbox[Node, Message]:
        """Deliver explicitly addressed messages in one synchronous round."""

        snapshot = {sender: dict(messages) for sender, messages in outbound.items()}
        for sender, messages in snapshot.items():
            self._validate_sender(sender)
            for receiver in messages:
                self._validate_receiver(sender, receiver)

        inbox = self._empty_inbox()
        deliveries: list[tuple[Node, Node]] = []
        for sender in self._graph.nodes:
            for receiver in sorted(snapshot.get(sender, {}), key=repr):
                inbox[receiver][sender] = deepcopy(snapshot[sender][receiver])
                deliveries.append((sender, receiver))
        self._last_deliveries = tuple(deliveries)
        return inbox

    def step(self, outbound: Mapping[Node, Message]) -> Inbox[Node, Message]:
        """Alias for the broadcast form used by synchronous algorithms."""

        return self.broadcast(outbound)

    def deliver(self, outbound: Mapping[Node, Message]) -> Inbox[Node, Message]:
        """Compatibility name for one synchronous broadcast exchange."""

        return self.broadcast(outbound)

    def _validate_sender(self, sender: Node) -> None:
        if sender not in self._graph:
            self._invalid_delivery_count += 1
            raise CommunicationError(f"Unknown sender {sender!r}")

    def _validate_receiver(self, sender: Node, receiver: Node) -> None:
        if receiver not in self._graph:
            self._invalid_delivery_count += 1
            raise CommunicationError(f"Unknown receiver {receiver!r}")
        if not self._graph.has_edge(sender, receiver):
            self._invalid_delivery_count += 1
            raise CommunicationError(
                f"Message from {sender!r} to non-neighbor {receiver!r} is not permitted"
            )


# Short aliases make the primitive easy to discover without duplicating its
# implementation or changing its semantics.
CommunicationBus = SynchronousMessageBus
MessageBus = SynchronousMessageBus
SynchronousExchange = SynchronousMessageBus


__all__ = [
    "CommunicationBus",
    "CommunicationError",
    "GraphSnapshot",
    "Inbox",
    "MessageBus",
    "SynchronousExchange",
    "SynchronousMessageBus",
]
