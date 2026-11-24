"""Synchronous communication used by the distributed algorithms."""

from __future__ import annotations

from collections import deque
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import fields
from math import isfinite
from typing import Callable

import numpy as np

from .types import OrderStatisticResult


def _ordered_nodes(graph: Mapping[str, set[str] | tuple[str, ...] | list[str]]) -> tuple[str, ...]:
    nodes = tuple(sorted(graph))
    if not nodes:
        raise ValueError("Graph must contain at least one node")
    for node, neighbors in graph.items():
        if node in neighbors:
            raise ValueError("Graph adjacency must not contain self edges")
        for neighbor in neighbors:
            if neighbor not in graph or node not in graph[neighbor]:
                raise ValueError("Graph must be undirected")
    if len(reachable_nodes(graph, nodes[0])) != len(nodes):
        raise ValueError("Distributed computation requires a connected graph")
    return nodes


def reachable_nodes(graph: Mapping[str, Sequence[str] | set[str]], start: str) -> set[str]:
    seen = {start}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return seen


def graph_diameter(graph: Mapping[str, Sequence[str] | set[str]]) -> int:
    nodes = _ordered_nodes(graph)
    diameter = 0
    for start in nodes:
        distances = {start: 0}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in graph[node]:
                if neighbor not in distances:
                    distances[neighbor] = distances[node] + 1
                    queue.append(neighbor)
        diameter = max(diameter, max(distances.values(), default=0))
    return diameter


def metropolis_matrix(graph: Mapping[str, Sequence[str] | set[str]]) -> tuple[tuple[str, ...], np.ndarray]:
    nodes = _ordered_nodes(graph)
    index = {node: i for i, node in enumerate(nodes)}
    degrees = {node: len(graph[node]) for node in nodes}
    matrix = np.zeros((len(nodes), len(nodes)), dtype=float)
    for node in nodes:
        i = index[node]
        for neighbor in graph[node]:
            j = index[neighbor]
            matrix[i, j] = 1.0 / (1.0 + max(degrees[node], degrees[neighbor]))
        matrix[i, i] = 1.0 - np.sum(matrix[i])
    if not np.allclose(matrix, matrix.T) or not np.allclose(matrix.sum(axis=0), 1.0):
        raise AssertionError("Metropolis matrix is not symmetric and doubly stochastic")
    return nodes, matrix


def flood_tagged(
    graph: Mapping[str, Sequence[str] | set[str]],
    local_values: Mapping[str, Mapping[Hashable, object]],
) -> tuple[dict[str, dict[Hashable, object]], int]:
    """Flood uniquely tagged values for exactly the graph diameter."""

    nodes = _ordered_nodes(graph)
    stores = {node: dict(local_values.get(node, {})) for node in nodes}
    diameter = graph_diameter(graph)
    for _ in range(diameter):
        previous = {node: dict(store) for node, store in stores.items()}
        for node in nodes:
            for neighbor in graph[node]:
                for tag, value in previous[neighbor].items():
                    if tag in stores[node]:
                        existing = stores[node][tag]
                        both_nan = (
                            isinstance(existing, (float, np.floating))
                            and isinstance(value, (float, np.floating))
                            and np.isnan(existing)
                            and np.isnan(value)
                        )
                        if not both_nan and existing != value:
                            raise ValueError(f"Conflicting values for tag {tag!r}")
                    stores[node][tag] = value
    expected = set().union(*(set(local_values.get(node, {})) for node in nodes))
    if any(set(store) != expected for store in stores.values()):
        raise AssertionError("Flooding did not deliver every unique tag")
    return stores, diameter


def _neighbor_exchange(
    graph: Mapping[str, Sequence[str] | set[str]],
    outbound: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    """Deliver one synchronous broadcast packet over a graph snapshot."""

    nodes = _ordered_nodes(graph)
    if set(outbound) != set(nodes):
        raise ValueError("Every graph node must provide exactly one outbound packet")
    from experiments.communication import SynchronousMessageBus

    inbox = SynchronousMessageBus(graph).deliver(outbound)
    if set(inbox) != set(nodes):
        raise AssertionError("Communication bus omitted a graph node")
    for node in nodes:
        expected = set(graph[node])
        received = set(inbox[node])
        if received != expected:
            raise AssertionError(
                f"Communication bus delivered the wrong neighbors to {node!r}: "
                f"expected {sorted(expected)!r}, received {sorted(received)!r}"
            )
    return {node: dict(inbox[node]) for node in nodes}


def _flood_node_summaries(
    graph: Mapping[str, Sequence[str] | set[str]],
    local_summaries: Mapping[str, object],
    combine: Callable[[object, object], object],
) -> tuple[object, int]:
    """Flood one summary per node for exactly the known graph diameter.

    This is the Algorithm 3 primitive used by the certified order statistic.
    The packets contain one value per *node*, never a node's locally owned raw
    sample set.  ``combine`` is applied only after all tagged summaries have
    reached every node.
    """

    nodes = _ordered_nodes(graph)
    if set(local_summaries) != set(nodes):
        raise ValueError("A summary is required for every graph node")
    diameter = graph_diameter(graph)
    stores: dict[str, dict[str, object]] = {
        node: {node: local_summaries[node]} for node in nodes
    }
    for _ in range(diameter):
        inbox = _neighbor_exchange(graph, stores)
        previous = stores
        stores = {node: dict(previous[node]) for node in nodes}
        for node in nodes:
            for neighbor, packet in inbox[node].items():
                if not isinstance(packet, Mapping):
                    raise TypeError("Flood packets must be tagged node-summary mappings")
                for owner, summary in packet.items():
                    if owner in stores[node] and stores[node][owner] != summary:
                        raise ValueError(f"Conflicting summaries for owner {owner!r}")
                    stores[node][owner] = summary
    expected = set(nodes)
    if any(set(store) != expected for store in stores.values()):
        raise AssertionError("Flooding did not deliver every node summary")
    aggregate: object | None = None
    for owner in nodes:
        summary = stores[nodes[0]][owner]
        aggregate = summary if aggregate is None else combine(aggregate, summary)
    assert aggregate is not None  # _ordered_nodes rejects an empty graph.
    return aggregate, diameter


def round_robin_ownership(
    tagged_values: Mapping[str, float], nodes: Sequence[str]
) -> dict[str, dict[str, float]]:
    """Assign observed tagged scalars deterministically, preserving empty stores."""

    ordered_nodes = tuple(sorted(nodes))
    if not ordered_nodes:
        raise ValueError("At least one owner is required")
    ownership = {node: {} for node in ordered_nodes}
    for index, (tag, value) in enumerate(tagged_values.items()):
        ownership[ordered_nodes[index % len(ordered_nodes)]][tag] = float(value)
    return ownership


def centralized_order_statistic(scores: Sequence[float], rank: int) -> float:
    """Return the requested order statistic of the pooled scores."""

    samples = tuple(float(score) for score in scores)
    if not samples:
        raise ValueError("At least one value is required")
    if not 1 <= rank <= len(samples):
        raise ValueError("Rank is outside the centralized value set")
    if any(np.isnan(score) or np.isneginf(score) for score in samples):
        raise ValueError("NaN and -inf are not valid order-statistic values")
    return float(sorted(samples)[rank - 1])


# A descriptive alias for callers that want to make the baseline/distributed
# distinction obvious at the call site.
centralized_exact_order_statistic = centralized_order_statistic


def _result_with_optional_fields(
    *,
    value: float,
    rank: int,
    pinball_history: np.ndarray,
    flooding_rounds: int,
    recovery_steps: int,
    owners: dict[str, tuple[str, ...]],
    pinball_rounds: int,
    total_communication_rounds: int,
    certified: bool,
) -> OrderStatisticResult:
    """Return the order statistic and available communication diagnostics."""

    available = {field.name for field in fields(OrderStatisticResult)}
    kwargs: dict[str, object] = {
        "value": value,
        # No exact statistic is available from the distributed computation.
        # ``None`` is accepted by the transitional dataclass and is the
        # sentinel consumed by the updated result type.
        "exact_value": None,
        "rank": rank,
        "pinball_history": pinball_history,
        "flooding_rounds": flooding_rounds,
        "recovery_steps": recovery_steps,
        "owners": owners,
    }
    optional = {
        "pinball_rounds": pinball_rounds,
        "total_communication_rounds": total_communication_rounds,
        "certified": certified,
    }
    kwargs.update({name: item for name, item in optional.items() if name in available})
    return OrderStatisticResult(**kwargs)  # type: ignore[arg-type]


def certified_order_statistic(
    graph: Mapping[str, Sequence[str] | set[str]],
    local_values: Mapping[str, Mapping[str, float]],
    rank: int,
    iterations: int,
    q_init: float,
    gamma_0: float = 0.05,
    exponent: float = 0.75,
    *,
    allow_nonfinite: bool = False,
) -> OrderStatisticResult:
    """Compute a distributed upper bound on the order statistic using pinball iterations and finite recovery. Each node holds its own scores. Certification requires finite scores on a connected undirected graph."""

    nodes = _ordered_nodes(graph)
    if set(local_values) - set(nodes):
        raise ValueError("Local stores contain an owner outside the update graph")
    if iterations < 0 or not 0.5 < exponent <= 1.0 or gamma_0 <= 0.0:
        raise ValueError("Invalid finite pinball schedule")
    if not isfinite(float(q_init)):
        raise ValueError("q_init must be finite")

    # Validate ownership and score admissibility without constructing a global
    # list of raw scores.  Duplicate tags indicate that one scalar has two owners.
    stores: dict[str, Mapping[str, float]] = {
        node: local_values.get(node, {}) for node in nodes
    }
    seen_tags: set[str] = set()
    sample_count = 0
    has_nonfinite = False
    has_invalid_nonfinite = False
    for node in nodes:
        store = stores[node]
        for tag, raw_score in store.items():
            if tag in seen_tags:
                raise ValueError(f"Every scalar value must have exactly one owner: {tag!r}")
            seen_tags.add(tag)
            score = float(raw_score)
            has_nonfinite = has_nonfinite or not np.isfinite(score)
            has_invalid_nonfinite = has_invalid_nonfinite or np.isnan(score) or score == float("-inf")
            sample_count += 1
    if sample_count == 0:
        raise ValueError("At least one value is required")
    if not 1 <= rank <= sample_count:
        raise ValueError("Rank is outside the owned value set")
    if has_invalid_nonfinite:
        raise ValueError("NaN and -inf are invalid order-statistic scores")
    if has_nonfinite and not allow_nonfinite:
        raise ValueError(
            "Certified order statistic requires finite scores; nonfinite score "
            "encountered in a local store"
        )

    # Metropolis weights are public graph coefficients.  Only the q packet
    # from each current neighbor is used in a node's update.
    _, mixing = metropolis_matrix(graph)
    index = {node: i for i, node in enumerate(nodes)}
    weights: dict[str, dict[str, float]] = {
        node: {
            neighbor: float(mixing[index[node], index[neighbor]])
            for neighbor in graph[node]
        }
        | {node: float(mixing[index[node], index[node]])}
        for node in nodes
    }

    tau = (rank - 0.5) / sample_count
    q = {node: float(q_init) for node in nodes}
    history = [np.asarray([q[node] for node in nodes], dtype=float)]
    for iteration in range(iterations):
        inbox = _neighbor_exchange(graph, q)
        next_q: dict[str, float] = {}
        step_size = gamma_0 / ((iteration + 1.0) ** exponent)
        for node in nodes:
            local_gradient = 0.0
            for raw_score in stores[node].values():
                score = float(raw_score)
                local_gradient += float(score < q[node]) - tau
            mixed = weights[node][node] * q[node]
            for neighbor in graph[node]:
                mixed += weights[node][neighbor] * float(inbox[node][neighbor])
            next_q[node] = mixed - step_size * local_gradient
        q = next_q
        history.append(np.asarray([q[node] for node in nodes], dtype=float))

    local_maximum = {
        node: max((float(score) for score in stores[node].values()), default=float("-inf"))
        for node in nodes
    }
    (maximum, q_max), diameter = _flood_node_summaries(
        graph,
        {node: (local_maximum[node], q[node]) for node in nodes},
        lambda left, right: (
            max(float(left[0]), float(right[0])),
            max(float(left[1]), float(right[1])),
        ),
    )
    maximum = float(maximum)
    q_max = float(q_max)
    candidate = min(maximum, q_max)
    recovery_steps = 0
    while True:
        local_count = {
            node: sum(float(score) <= candidate for score in stores[node].values())
            for node in nodes
        }
        local_next = {
            node: min(
                (float(score) for score in stores[node].values() if float(score) > candidate),
                default=float("inf"),
            )
            for node in nodes
        }
        (count, next_candidate), _ = _flood_node_summaries(
            graph,
            {node: (local_count[node], local_next[node]) for node in nodes},
            lambda left, right: (
                int(left[0]) + int(right[0]),
                min(float(left[1]), float(right[1])),
            ),
        )
        count = int(count)
        next_candidate = float(next_candidate)
        if count >= rank:
            break
        if not np.isfinite(next_candidate) and next_candidate != float("inf"):
            raise AssertionError("Recovery produced a NaN candidate")
        if next_candidate <= candidate:
            raise AssertionError("Finite recovery did not advance to a larger candidate")
        candidate = next_candidate
        recovery_steps += 1

    if not has_nonfinite:
        if not isfinite(candidate) or candidate > maximum:
            raise AssertionError("Certified recovery exceeded the local maximum")
    owners = {node: tuple(sorted(stores[node])) for node in nodes}
    flooding_rounds = diameter * (2 + recovery_steps)
    return _result_with_optional_fields(
        value=float(candidate),
        rank=rank,
        pinball_history=np.asarray(history),
        flooding_rounds=flooding_rounds,
        recovery_steps=recovery_steps,
        owners=owners,
        pinball_rounds=iterations,
        total_communication_rounds=iterations + flooding_rounds,
        certified=not has_nonfinite,
    )
