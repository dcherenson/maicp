"""Distributed rolling window tracking with the alternating direction method of multipliers (ADMM)."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.linalg import block_diag

from maicp.configuration import ClassConfig, Platform
from experiments.tracking.margins import resolve_robot_margins


Graph = Mapping[str, Sequence[str] | set[str] | frozenset[str]]
MessageExchange = Callable[
    [Mapping[str, set[str]], Mapping[str, object]],
    Mapping[str, Mapping[str, object]],
]


@dataclass(frozen=True)
class MeasurementPacket:
    """Observation, measurement matrix, and covariance for one distributed tracking update."""

    visible: bool
    value: np.ndarray
    H: np.ndarray
    covariance: np.ndarray
    sensor_type: str


class DRWTTracker:
    """Distributed rolling window tracker. Each physical update uses a fixed communication graph during the ADMM iterations."""

    def __init__(
        self,
        robot_ids: Sequence[str],
        robot_platforms: Mapping[str, Platform],
        neighbors: Graph | None,
        classes: Mapping[Platform, ClassConfig],
        margins: Mapping[Platform | str, float],
        window_size: int = 4,
        rho: float = 0.8,
        admm_iterations: int = 5,
        dt: float = 0.5,
        message_exchange: MessageExchange | None = None,
    ):
        self.robot_ids = tuple(robot_ids)
        if not self.robot_ids or len(set(self.robot_ids)) != len(self.robot_ids):
            raise ValueError("robot_ids must be a non-empty sequence of unique IDs")
        self.platforms = dict(robot_platforms)
        missing_platforms = set(self.robot_ids).difference(self.platforms)
        if missing_platforms:
            raise ValueError(f"Missing platform assignments: {sorted(missing_platforms)!r}")
        self.classes = dict(classes)
        self.set_margins(margins)
        self.window_size = int(window_size)
        self.rho = float(rho)
        self.admm_iterations = int(admm_iterations)
        self.dt = float(dt)
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if self.rho <= 0.0:
            raise ValueError("rho must be positive")
        if self.admm_iterations < 0:
            raise ValueError("admm_iterations cannot be negative")
        self.n = 4
        self.window_dim = self.n * (self.window_size + 1)
        self.prior_dim = self.n * self.window_size
        self.A = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        # Nominal 0.04 m/s covariance of the velocity increment at dt=0.5 s,
        # including 1e-4 I for positive definiteness.
        self.Q = np.array([
            [0.0005, 0.0, 0.0008, 0.0],
            [0.0, 0.0005, 0.0, 0.0008],
            [0.0008, 0.0, 0.0017, 0.0],
            [0.0, 0.0008, 0.0, 0.0017],
        ])

        # Store the graph used at the last physical update.
        self.neighbors = self._normalize_graph(neighbors or {})
        self._message_exchange = message_exchange
        self._initialized = False
        self._active_robot_ids = self.robot_ids

        self.x_bar: dict[str, np.ndarray] = {}
        self.P_bar: dict[str, np.ndarray] = {}
        self.x_hat: dict[str, np.ndarray] = {}
        self.P_hat: dict[str, np.ndarray] = {}

        self.graph_history: list[dict[str, set[str]]] = []
        self.graph_connectivity_history: list[bool] = []
        self.communication_log: list[dict[str, Any]] = []
        self.last_exchange_inbox: dict[str, dict[str, object]] = {}
        self.last_invalid_delivery_count = 0
        self.last_graph_connected = False
        self.last_process_information: np.ndarray | None = None
        self.last_local_information: dict[str, np.ndarray] = {}
        self.last_local_omega: dict[str, np.ndarray] = {}
        self.last_effective_covariances: dict[str, np.ndarray] = {}
        self.last_measurement_dimensions: dict[str, int] = {}
        self.last_dual: dict[str, np.ndarray] = {}
        self.last_primal: dict[str, np.ndarray] = {}
        self.dual_reset_count = 0

    def _normalize_graph(self, graph: Graph) -> dict[str, set[str]]:
        """Return a validated undirected graph over all tracker nodes."""

        known = set(self.robot_ids)
        unknown = set(graph).difference(known)
        if unknown:
            raise ValueError(f"Graph references unknown robot IDs: {sorted(unknown)!r}")
        normalized = {robot_id: set(graph.get(robot_id, ())) for robot_id in self.robot_ids}
        for robot_id, neighbors in normalized.items():
            if robot_id in neighbors:
                raise ValueError("A task graph cannot contain self edges")
            unknown_neighbors = neighbors.difference(known)
            if unknown_neighbors:
                raise ValueError(
                    f"Graph references unknown neighbors: {sorted(unknown_neighbors)!r}"
                )
            for neighbor in neighbors:
                if robot_id not in normalized[neighbor]:
                    raise ValueError("The task graph must be undirected")
        return normalized

    @staticmethod
    def _is_connected(graph: Mapping[str, set[str]]) -> bool:
        if not graph:
            return False
        start = next(iter(graph))
        visited = {start}
        pending = [start]
        while pending:
            node = pending.pop()
            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    pending.append(neighbor)
        return visited == set(graph)

    @staticmethod
    def _as_covariance(value: float | np.ndarray, dimension: int) -> np.ndarray:
        covariance = np.asarray(value, dtype=float)
        if covariance.ndim == 0:
            if float(covariance) <= 0.0:
                raise ValueError("Covariance must be positive")
            result = np.eye(dimension) * float(covariance)
        elif covariance.shape == (dimension, dimension):
            result = covariance.copy()
        else:
            raise ValueError(f"Expected a scalar or {dimension}x{dimension} covariance")
        if not np.all(np.isfinite(result)) or not np.allclose(result, result.T):
            raise ValueError("Covariance must be finite and symmetric")
        try:
            np.linalg.cholesky(result)
        except np.linalg.LinAlgError as error:
            raise ValueError("Covariance must be positive definite") from error
        return result

    def _validate_measurement_packet(
        self,
        packet: MeasurementPacket,
        robot_id: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Validate and normalize one visible packet's native dimensions."""

        if not isinstance(packet, MeasurementPacket):
            raise TypeError(
                f"Measurement for {robot_id!r} must be a MeasurementPacket"
            )
        if not isinstance(packet.visible, (bool, np.bool_)):
            raise ValueError(f"Measurement visibility for {robot_id!r} must be boolean")
        try:
            value = np.asarray(packet.value, dtype=float)
            design = np.asarray(packet.H, dtype=float)
            covariance = np.asarray(packet.covariance, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Measurement packet for {robot_id!r} is not numeric") from error
        if value.ndim != 1 or value.shape[0] <= 0:
            raise ValueError(
                f"Measurement value for {robot_id!r} must have a positive one-dimensional shape"
            )
        dimension = value.shape[0]
        if design.shape != (dimension, self.n):
            raise ValueError(
                f"Measurement H for {robot_id!r} must have shape {(dimension, self.n)}"
            )
        if covariance.shape != (dimension, dimension):
            raise ValueError(
                f"Measurement covariance for {robot_id!r} must have shape "
                f"{(dimension, dimension)}"
            )
        if not (
            np.all(np.isfinite(value))
            and np.all(np.isfinite(design))
            and np.all(np.isfinite(covariance))
        ):
            raise ValueError(f"Measurement packet for {robot_id!r} must be finite")
        if not np.allclose(covariance, covariance.T):
            raise ValueError(f"Measurement covariance for {robot_id!r} must be symmetric")
        try:
            np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as error:
            raise ValueError(
                f"Measurement covariance for {robot_id!r} must be positive definite"
            ) from error
        return value, design, covariance

    def initialize(
        self,
        prior_state: np.ndarray,
        covariance: float | np.ndarray = 1.0,
        *,
        prior_covariance: np.ndarray | None = None,
    ) -> None:
        """Initialize each robot from the common prior mean and N times the prior covariance, where N is the number of robots."""

        state = np.asarray(prior_state, dtype=float)
        if state.shape != (self.n,) or not np.all(np.isfinite(state)):
            raise ValueError(f"prior_state must have shape {(self.n,)}")
        central_covariance = self._as_covariance(
            covariance if prior_covariance is None else prior_covariance,
            self.n,
        )
        central_prior = block_diag(*([central_covariance] * self.window_size))
        local_prior_covariance = len(self.robot_ids) * central_prior
        prior_mean = np.tile(state, self.window_size)
        estimate_mean = np.tile(state, self.window_size + 1)
        estimate_covariance = block_diag(central_prior, central_covariance)
        self.x_bar = {robot_id: prior_mean.copy() for robot_id in self.robot_ids}
        self.P_bar = {
            robot_id: local_prior_covariance.copy() for robot_id in self.robot_ids
        }
        self.x_hat = {robot_id: estimate_mean.copy() for robot_id in self.robot_ids}
        self.P_hat = {
            robot_id: estimate_covariance.copy() for robot_id in self.robot_ids
        }
        self._active_robot_ids = self.robot_ids
        self._initialized = True

    def set_margins(self, margins: Mapping[Platform | str, float]) -> None:
        self.robot_margins, _ = resolve_robot_margins(
            margins, {robot: self.platforms[robot] for robot in self.robot_ids},
            classes=self.classes,
        )
        self.margins = dict(margins)

    @property
    def active_robot_ids(self) -> tuple[str, ...]:
        """Nodes participating in the most recently frozen physical update."""

        return self._active_robot_ids

    def _fuse_information(
        self,
        receiver: str,
        source: str,
        estimate: dict[str, np.ndarray],
        covariance: dict[str, np.ndarray],
    ) -> None:
        """Fuse a leaving node's Gaussian information into a continuing node."""

        receiver_information = np.linalg.inv(covariance[receiver])
        source_information = np.linalg.inv(covariance[source])
        fused_covariance = np.linalg.inv(receiver_information + source_information)
        fused_mean = fused_covariance @ (
            receiver_information @ estimate[receiver]
            + source_information @ estimate[source]
        )
        covariance[receiver] = fused_covariance
        estimate[receiver] = fused_mean

    def handoff(
        self,
        leaving_robot_ids: Sequence[str],
        continuing_neighbors: Mapping[str, str],
        *,
        neighbors: Graph | None = None,
    ) -> None:
        """Transfer Gaussian information from each departing robot to a continuing neighbor in the active graph."""

        graph = self._normalize_graph(self.neighbors if neighbors is None else neighbors)
        for leaving in leaving_robot_ids:
            if leaving not in self.robot_ids:
                raise ValueError(f"Unknown leaving robot {leaving!r}")
            receiver = continuing_neighbors.get(leaving)
            if receiver not in self.robot_ids or receiver not in graph[leaving]:
                raise ValueError(
                    f"Handoff for {leaving!r} must name a continuing graph neighbor"
                )
            self._fuse_information(receiver, leaving, self.x_hat, self.P_hat)
            self._fuse_information(receiver, leaving, self.x_bar, self.P_bar)

    def _exchange_packets(
        self,
        graph: Mapping[str, set[str]],
        outbound: Mapping[str, object],
    ) -> dict[str, dict[str, object]]:
        """Deliver one synchronous primal packet over the frozen graph."""

        nodes = tuple(self._active_robot_ids)
        if set(outbound) != set(nodes):
            raise ValueError("Every active node must provide one outbound packet")
        result: Mapping[str, Mapping[str, object]] | None = None
        exchange = self._message_exchange
        if exchange is None:
            try:
                from experiments.communication import synchronous_exchange
            except (ImportError, AttributeError):
                synchronous_exchange = None
            if synchronous_exchange is not None:
                result = synchronous_exchange(graph, outbound)
        else:
            if callable(exchange):
                result = exchange(graph, outbound)
            elif hasattr(exchange, "broadcast"):
                result = exchange.broadcast(outbound)
            else:
                raise TypeError("message_exchange must be callable or expose broadcast")

        if result is None:
            # Deliver one synchronous broadcast to each node's neighbors.
            from experiments.communication import GraphSnapshot, SynchronousMessageBus

            bus = SynchronousMessageBus(GraphSnapshot.from_adjacency(graph))
            result = bus.broadcast(outbound)

        if set(result) != set(nodes):
            raise ValueError("Communication exchange omitted an active node")
        inbox: dict[str, dict[str, object]] = {}
        for receiver in nodes:
            received = dict(result[receiver])
            expected = set(graph[receiver])
            if set(received) != expected:
                self.last_invalid_delivery_count += int(set(received) != expected)
                raise ValueError(
                    f"Communication delivered packets outside the frozen graph for {receiver!r}"
                )
            inbox[receiver] = received
        return inbox

    @staticmethod
    def _primal_packet(packet: object) -> np.ndarray:
        """Decode a packet containing the primal variable, including packets labelled by test adapters."""

        if isinstance(packet, Mapping) and "primal" in packet:
            packet = packet["primal"]
        primal = np.asarray(packet, dtype=float)
        if primal.ndim != 1:
            raise ValueError("A DRWT packet must contain one primal trajectory vector")
        return primal

    def step(
        self,
        measurements: Mapping[str, MeasurementPacket],
        neighbors: Graph | None = None,
        *,
        active_robot_ids: Sequence[str] | None = None,
        message_exchange: MessageExchange | None = None,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Run one physical update and its finite synchronous ADMM iterations."""

        if not self._initialized:
            raise RuntimeError("initialize must be called before step")
        graph = self._normalize_graph(self.neighbors if neighbors is None else neighbors)
        if active_robot_ids is None:
            active = self.robot_ids
        else:
            active = tuple(active_robot_ids)
            if not active or len(set(active)) != len(active):
                raise ValueError("active_robot_ids must be a non-empty unique sequence")
            if not set(active).issubset(self.robot_ids):
                raise ValueError("active_robot_ids contains an unknown robot")
        active_set = set(active)
        active_graph = {
            robot_id: graph[robot_id].intersection(active_set) for robot_id in active
        }
        self.neighbors = {robot_id: set(graph[robot_id]) for robot_id in self.robot_ids}
        self._active_robot_ids = active
        self.graph_history.append(
            {node: set(neighbors) for node, neighbors in active_graph.items()}
        )
        self.last_graph_connected = self._is_connected(active_graph)
        self.graph_connectivity_history.append(self.last_graph_connected)
        self.last_invalid_delivery_count = 0

        transition = np.zeros((self.n, self.window_dim))
        transition[:, self.n * (self.window_size - 1) : self.n * self.window_size] = -self.A
        transition[:, self.n * self.window_size :] = np.eye(self.n)
        prior_selector = np.zeros((self.prior_dim, self.window_dim))
        prior_selector[:, : self.prior_dim] = np.eye(self.prior_dim)

        # Eq. (11) of the source weights the process information by 1/N.
        # This is equivalent to using local process covariance N*Q.
        process_information = np.linalg.inv(self.Q) / float(len(active))
        self.last_process_information = process_information.copy()
        omega: dict[str, np.ndarray] = {}
        information: dict[str, np.ndarray] = {}
        inverse_update: dict[str, np.ndarray] = {}
        effective_covariances: dict[str, np.ndarray] = {}
        measurement_dimensions: dict[str, int] = {}
        for robot_id in active:
            matrices = [transition, prior_selector]
            observations = [np.zeros(self.n), self.x_bar[robot_id]]
            inverse_covariances = [
                process_information,
                np.linalg.inv(self.P_bar[robot_id]),
            ]
            packet = measurements.get(robot_id)
            if robot_id in measurements and not isinstance(packet, MeasurementPacket):
                raise TypeError(
                    f"Measurement for {robot_id!r} must be a MeasurementPacket"
                )
            if packet is not None and packet.visible:
                measurement, design, covariance = self._validate_measurement_packet(
                    packet, robot_id
                )
                platform = self.platforms[robot_id]
                beta = float(self.classes[platform].tracking_beta)
                margin = self.robot_margins[robot_id]
                if not np.isfinite(beta) or beta < 0.0:
                    raise ValueError(f"Invalid tracking beta for {platform!r}")
                if not np.isfinite(margin):
                    raise ValueError(f"Invalid robustness margin for {platform!r}")
                effective_covariance = covariance + beta * margin * np.eye(
                    measurement.shape[0]
                )
                # Validate the inflated covariance before forming its
                # information matrix, including arithmetic overflow checks.
                self._as_covariance(effective_covariance, measurement.shape[0])
                measurement_selector = np.zeros(
                    (measurement.shape[0], self.window_dim)
                )
                measurement_selector[:, -self.n :] = design
                matrices.insert(1, measurement_selector)
                observations.insert(1, measurement)
                inverse_covariances.insert(1, np.linalg.inv(effective_covariance))
                effective_covariances[robot_id] = effective_covariance.copy()
                measurement_dimensions[robot_id] = measurement.shape[0]
            H = np.vstack(matrices)
            W_inv = block_diag(*inverse_covariances)
            z = np.concatenate(observations)
            omega_i = H.T @ W_inv @ H
            information_i = H.T @ W_inv @ z
            degree = len(active_graph[robot_id])
            omega[robot_id] = omega_i
            information[robot_id] = information_i
            inverse_update[robot_id] = np.linalg.inv(
                omega_i + 2.0 * self.rho * degree * np.eye(self.window_dim)
            )

        primal = {
            robot_id: np.linalg.solve(omega[robot_id], information[robot_id])
            for robot_id in active
        }
        # Algorithm 1 resets p_i to zero at every physical update.
        dual = {robot_id: np.zeros(self.window_dim) for robot_id in active}
        self.dual_reset_count += 1
        self.communication_log = []
        exchange_fn = message_exchange or self._message_exchange
        for iteration in range(self.admm_iterations):
            outbound = {
                # The payload is exactly x_i^k.  A labelled {"primal": ...}
                # adapter is accepted on receipt for test transports, but is
                # not sent by the production tracker.
                robot_id: primal[robot_id].copy()
                for robot_id in active
            }
            previous_primal = {robot_id: primal[robot_id].copy() for robot_id in active}
            previous_dual = {robot_id: dual[robot_id].copy() for robot_id in active}
            # Use the same callback for this physical step, but do not read a
            # neighbour's local array directly.  All packets are from the
            # previous primal snapshot and are delivered synchronously.
            old_exchange = self._message_exchange
            self._message_exchange = exchange_fn
            try:
                inbox = self._exchange_packets(active_graph, outbound)
            finally:
                self._message_exchange = old_exchange
            self.last_exchange_inbox = inbox
            dual_next: dict[str, np.ndarray] = {}
            primal_next: dict[str, np.ndarray] = {}
            # Stable reduction order keeps serial and spawned mission results identical.
            for robot_id in active:
                disagreement = sum(
                    (
                        previous_primal[robot_id]
                        - self._primal_packet(inbox[robot_id][neighbor])
                        for neighbor in sorted(active_graph[robot_id])
                    ),
                    start=np.zeros(self.window_dim),
                )
                dual_next[robot_id] = previous_dual[robot_id] + self.rho * disagreement
            for robot_id in active:
                neighbor_sum = sum(
                    (
                        previous_primal[robot_id]
                        + self._primal_packet(inbox[robot_id][neighbor])
                        for neighbor in sorted(active_graph[robot_id])
                    ),
                    start=np.zeros(self.window_dim),
                )
                rhs = information[robot_id] - dual_next[robot_id] + self.rho * neighbor_sum
                primal_next[robot_id] = inverse_update[robot_id] @ rhs
            self.communication_log.append(
                {
                    "iteration": iteration,
                    "edges": tuple(
                        sorted(
                            (sender, receiver)
                            for sender in active
                            for receiver in active_graph[sender]
                            if sender < receiver
                        )
                    ),
                    "packets": {
                        sender: np.asarray(packet, dtype=float).copy()
                        for sender, packet in outbound.items()
                    },
                }
            )
            primal, dual = primal_next, dual_next

        self.last_local_information = {
            node: value.copy() for node, value in information.items()
        }
        self.last_local_omega = {node: value.copy() for node, value in omega.items()}
        self.last_effective_covariances = effective_covariances
        self.last_measurement_dimensions = measurement_dimensions
        self.last_primal = {node: value.copy() for node, value in primal.items()}
        self.last_dual = {node: value.copy() for node, value in dual.items()}

        result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for robot_id in active:
            self.x_hat[robot_id] = primal[robot_id]
            self.P_hat[robot_id] = np.linalg.inv(omega[robot_id])
            self.x_bar[robot_id] = primal[robot_id][self.n :].copy()
            self.P_bar[robot_id] = self.P_hat[robot_id][self.n :, self.n :].copy()
            state = primal[robot_id][-self.n :].copy()
            covariance = self.P_hat[robot_id][-self.n :, -self.n :].copy()
            result[robot_id] = state, covariance
        return result
