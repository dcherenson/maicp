"""Class mission scores, distributed calibration quantiles, and the coupled payload margin update."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal, localcontext
from typing import Any

import numpy as np
from scipy.optimize import minimize, nnls

from experiments.payload.config import ExperimentConfig, ModelConfig
from experiments.payload.model import MissionBatch
from maicp.ranks import binomial_calibration_rank


@dataclass
class Calibration:
    quantiles_by_node: tuple[np.ndarray, ...]
    mission_scores: np.ndarray  # Saved for auditing AFTER the distributed computation.
    reports: list[dict]
    owner_counts: list[list[int]]


def calibrate(
    batch: MissionBatch, model: ModelConfig, exp: ExperimentConfig, bus: Bus
) -> Calibration:
    count = len(batch.node_scores[0])
    if count != exp.calibration_missions or any(
        len(x) != count for x in batch.node_scores
    ):
        raise ValueError(
            "Calibration batch has the wrong number of independent missions"
        )
    output, reports, owner_counts = [], [], []
    for c in range(2):
        nodes = [i for i, cc in enumerate(model.classes) if cc == c]
        # Each robot stores its own score for each mission.
        local = [
            batch.node_scores[i][:, None] if i in nodes else np.empty((count, 0))
            for i in range(6)
        ]
        mission = order_statistic(local, rank=3, bus=bus, iterations=0)
        # Unique ownership of the resulting mission certificates (beta_is=1).
        stores = []
        owners = np.asarray([nodes[m % len(nodes)] for m in range(count)])
        for i in range(6):
            stores.append(mission.value[owners == i][None, :].copy())
        owner_counts.append([int(x.shape[1]) for x in stores])
        rank = binomial_calibration_rank(count, exp.alpha[c], exp.delta_cal[c])
        quantile = order_statistic(
            stores,
            rank,
            bus,
            iterations=exp.pinball_iterations,
            scale=exp.pinball_scale,
        )
        output.append((mission.value.copy(), float(quantile.value[0])))
        reports.append(
            {
                "class": c + 1,
                "independent_missions": count,
                "rank": rank,
                "mission_computation": mission.summary(),
                "quantile_computation": quantile.summary(),
                "quantile": float(quantile.value[0]),
            }
        )
    # Every node recovers the same upper bound through flooding.
    q = np.asarray([item[1] for item in output])
    return Calibration(
        tuple(q.copy() for _ in range(6)),
        np.column_stack([item[0] for item in output]),
        reports,
        owner_counts,
    )


def decimal_slacks(
    r: np.ndarray, old: np.ndarray, q: np.ndarray, kappa: np.ndarray
) -> list[Decimal]:
    """Bound the constraint slack by rounding the supplied margins outward in decimal arithmetic."""
    D = lambda x: Decimal.from_float(float(x))
    with localcontext() as ctx:
        ctx.prec = 100
        abs_d_upper = []
        for a, b in zip(r, old):
            ctx.rounding = ROUND_FLOOR
            lo = D(a) - D(b)
            ctx.rounding = ROUND_CEILING
            hi = D(a) - D(b)
            abs_d_upper.append(max(abs(lo), abs(hi)))
        ctx.rounding = ROUND_CEILING
        norm2_upper = sum(x * x for x in abs_d_upper)
        norm_upper = norm2_upper.sqrt().next_plus()
        correction_upper = [D(k) * norm_upper for k in kappa]
        ctx.rounding = ROUND_FLOOR
        return [D(a) - D(b) - c for a, b, c in zip(r, q, correction_upper)]


def objective_gap_bound(
    r: np.ndarray,
    old: np.ndarray,
    q: np.ndarray,
    kappa: np.ndarray,
    cap: np.ndarray,
    weights: np.ndarray,
    lam: float,
) -> float:
    """Bound the objective gap using strong convexity of the Lagrangian."""
    d = r - old
    norm = np.linalg.norm(d)
    direction = d / norm if norm > 1e-15 else np.zeros_like(d)
    # Convex constraints g<=0, including lower and upper domain bounds.
    g = np.r_[q + kappa * norm - r, -r, r - cap]
    jac = np.vstack(
        [kappa[:, None] * direction[None, :] - np.eye(2), -np.eye(2), np.eye(2)]
    )
    active = g >= -1e-5
    grad = weights + lam * d
    multipliers = np.zeros(6)
    if np.any(active):
        multipliers[active] = nnls(jac[active].T, -grad)[0]
    residual = grad + jac.T @ multipliers
    return max(0.0, float(-multipliers @ g + residual @ residual / (2 * lam)))


def margin_update(
    old: np.ndarray,
    quantiles: np.ndarray,
    kappa: np.ndarray,
    model: ModelConfig,
    exp: ExperimentConfig,
) -> tuple[np.ndarray, dict]:
    """Minimize wᵀr + λt²/2 subject to ‖r − r_old‖₂ ≤ t and q + κt ≤ r in the margin box. Check numerical feasibility and the objective gap before accepting a solution."""
    old, q, k = (np.asarray(x, dtype=float) for x in (old, quantiles, kappa))
    cap, weights = np.asarray(model.margin_cap), np.asarray(exp.objective_weights)
    if any(x.shape != (2,) or not np.all(np.isfinite(x)) for x in (old, q, k)):
        raise ValueError("Invalid margin problem")
    if np.any(old < 0) or np.any(old > cap) or np.any(q < 0) or np.any(k < 0):
        raise ValueError("Invalid margin domain or negative data")
    # If the upper corner is infeasible and ‖κ‖₂ < 1, r = q + κt is feasible for t ≥ ‖q − r_old‖₂ / (1 − ‖κ‖₂), subject to the margin box.
    start = cap.copy()
    t0 = np.linalg.norm(start - old)
    start_method = "upper_corner"
    if np.any(start - q - k * t0 < 0):
        norm_k = np.linalg.norm(k)
        if norm_k >= 1:
            raise ValueError("No certified start; no uncertified deployment")
        t0 = np.linalg.norm(q - old) / (1 - norm_k) + 1e-9
        start = q + k * t0
        start_method = "norm_contraction_construction"
        if (
            np.any(start > cap)
            or np.any(start < 0)
            or np.any(start - q - k * np.linalg.norm(start - old) < 0)
        ):
            raise ValueError(
                "Constructive feasible point outside domain; stop rather than clip"
            )
    lam = exp.proximal_weight

    def fun(x):
        return float(weights @ x[:2] + 0.5 * lam * x[2] ** 2)

    def jac(x):
        return np.r_[weights, lam * x[2]]

    def cone(x):
        return x[2] - np.linalg.norm(x[:2] - old)

    def cone_jac(x):
        d = x[:2] - old
        size = np.linalg.norm(d)
        return np.r_[-d / size if size > 0 else np.zeros(2), 1.0]

    result = minimize(
        fun,
        np.r_[start, t0],
        jac=jac,
        method="SLSQP",
        bounds=[
            (0, float(cap[0])),
            (0, float(cap[1])),
            (0, float(np.linalg.norm(cap))),
        ],
        constraints=[
            {
                "type": "ineq",
                "fun": lambda x: x[:2] - q - k * x[2],
                "jac": lambda x: np.column_stack((np.eye(2), -k)),
            },
            {"type": "ineq", "fun": cone, "jac": cone_jac},
        ],
        options={"ftol": 1e-12, "maxiter": 300},
    )
    # Check feasibility and the objective gap independently of the SLSQP exit status.
    if not np.all(np.isfinite(result.x)):
        raise RuntimeError(f"Nonfinite margin candidate: {result.message}")
    candidate = result.x[:2].copy()
    original = candidate.copy()
    guard = Decimal("1e-10")
    coefficient = 1 - np.sqrt(2) * float(np.max(k))
    if coefficient <= 0:
        raise ValueError("The explicit numerical guard needs sqrt(2)*max(kappa)<1")
    for _ in range(6):
        slacks = decimal_slacks(candidate, old, q, k)
        if min(slacks) >= guard:
            break
        increase = float(guard - min(slacks)) / coefficient + 1e-12
        candidate = np.nextafter(candidate + increase, np.inf)
        if np.any(candidate > cap):
            raise RuntimeError(
                "Feasibility repair would leave the certified margin domain"
            )
    else:
        raise RuntimeError("Outward margin feasibility check failed")
    slacks = decimal_slacks(candidate, old, q, k)
    if min(slacks) < 0 or np.any(candidate < 0) or np.any(candidate > cap):
        raise AssertionError("Invalid certified margin")
    gap = objective_gap_bound(candidate, old, q, k, cap, weights, lam)
    if gap > 1e-6:
        raise RuntimeError(
            f"Margin optimizer's independent objective gap bound is {gap}"
        )
    return candidate, {
        "solver": "SciPy SLSQP, analytic objective/constraint derivatives",
        "feasible_start": start_method,
        "solver_iterations": int(result.nit),
        "solver_success": bool(result.success),
        "solver_message": str(result.message),
        "original_candidate": original.tolist(),
        "deployed_margin": candidate.tolist(),
        "upward_numerical_guard": (candidate - original).tolist(),
        "constraint_slack_lower_decimal": [str(x) for x in slacks],
        "objective_gap_bound_numerical": gap,
        "shift_correction": (k * np.linalg.norm(candidate - old)).tolist(),
    }



@dataclass
class Traffic:
    rounds: int = 0
    directed_packets: int = 0
    scalar_payloads: int = 0

    def dictionary(self) -> dict:
        return asdict(self)


class Bus:
    def __init__(self, adjacency: Sequence[Sequence[int]] | None = None):
        self.adjacency = tuple(
            tuple(sorted(x))
            for x in (
                adjacency
                if adjacency is not None
                else tuple(((i - 1) % 6, (i + 1) % 6) for i in range(6))
            )
        )
        self.n = len(self.adjacency)
        if self.n < 1:
            raise ValueError("Empty graph")
        for i, neighbors in enumerate(self.adjacency):
            if len(set(neighbors)) != len(neighbors) or i in neighbors:
                raise ValueError("Self edges or duplicate edges")
            for j in neighbors:
                if not 0 <= j < self.n or i not in self.adjacency[j]:
                    raise ValueError("Graph must be undirected")
        diam = 0
        for root in range(self.n):
            distance = {root: 0}
            queue = [root]
            for i in queue:
                for j in self.adjacency[i]:
                    if j not in distance:
                        distance[j] = distance[i] + 1
                        queue.append(j)
            if len(distance) != self.n:
                raise ValueError("Disconnected communication graph")
            diam = max(diam, max(distance.values()))
        self.diameter = diam
        self.traffic = Traffic()
        self.W = np.zeros((self.n, self.n))
        for i, neighbors in enumerate(self.adjacency):
            for j in neighbors:
                self.W[i, j] = 1 / (1 + max(len(neighbors), len(self.adjacency[j])))
            self.W[i, i] = 1 - self.W[i].sum()
        if not (np.allclose(self.W, self.W.T) and np.allclose(self.W.sum(0), 1)):
            raise AssertionError("Mixing weights failed validation")

    @staticmethod
    def _size(x: Any) -> int:
        if isinstance(x, dict):
            return sum(Bus._size(v) for v in x.values())
        if isinstance(x, (tuple, list)):
            return sum(Bus._size(v) for v in x)
        return int(np.size(x))

    def exchange(self, outbound: Sequence[Any]) -> tuple[dict[int, Any], ...]:
        if len(outbound) != self.n:
            raise ValueError("Exactly one outbound packet per node is required")
        self.traffic.rounds += 1
        inbox = []
        for i, neighbors in enumerate(self.adjacency):
            inbox.append({j: outbound[j] for j in neighbors})
            self.traffic.directed_packets += len(neighbors)
            self.traffic.scalar_payloads += sum(
                self._size(outbound[j]) for j in neighbors
            )
        return tuple(inbox)

    def flood(
        self, local: Sequence[Any], aggregate: Callable[[list[Any]], Any]
    ) -> tuple[Any, ...]:
        """Flood immutable tagged node summaries for the known graph diameter."""
        if len(local) != self.n:
            raise ValueError("One local summary per node is required")
        stores = [{i: local[i]} for i in range(self.n)]
        for _ in range(self.diameter):
            incoming = self.exchange(stores)
            updated = [dict(s) for s in stores]
            for i in range(self.n):
                for packet in incoming[i].values():
                    # Unique node tags prevent double counting around cycles.
                    updated[i].update(packet)
            stores = updated
        expected = set(range(self.n))
        if any(set(s) != expected for s in stores):
            raise AssertionError("Incomplete flooding")
        return tuple(aggregate([s[j] for j in range(self.n)]) for s in stores)

    def mix(self, values: np.ndarray) -> np.ndarray:
        if values.shape[0] != self.n:
            raise ValueError("Leading dimension must index nodes")
        incoming = self.exchange(tuple(values[i].copy() for i in range(self.n)))
        result = np.zeros_like(values)
        for i in range(self.n):
            result[i] = self.W[i, i] * values[i]
            for j, value in incoming[i].items():
                result[i] += self.W[i, j] * value
        return result


@dataclass
class OrderStatistic:
    value: np.ndarray  # One value per independent mission.
    rank: int
    count: int
    pinball_iterations: int
    recovery_steps: int
    communication_rounds: int
    # Exactly equal common output reconstructed at every node.
    node_agreement: bool

    def summary(self) -> dict:
        return {k: v for k, v in asdict(self).items() if k != "value"}


def order_statistic(
    local: Sequence[np.ndarray],
    rank: int,
    bus: Bus,
    iterations: int = 100,
    scale: float = 2.0,
    initial: float = 0.0,
) -> OrderStatistic:
    """Distributed order statistic with one owner per observation.

    Each local array has shape (independent batches, locally owned observations).
    Each node sorts its own observations to answer rank queries.
    """
    if len(local) != bus.n:
        raise ValueError("Each node must have a local store, possibly empty")
    values = tuple(np.asarray(x, dtype=float) for x in local)
    if any(x.ndim != 2 for x in values):
        raise ValueError("Local stores must be matrices")
    batches = values[0].shape[0]
    if batches < 1 or any(x.shape[0] != batches for x in values):
        raise ValueError("Inconsistent or empty batches")
    if any(not np.all(np.isfinite(x)) for x in values):
        raise ValueError("Certified order statistics require finite scores")
    nobs = sum(x.shape[1] for x in values)
    if not 1 <= rank <= nobs:
        raise ValueError("Invalid order statistic rank")
    if (
        iterations < 0
        or not isinstance(iterations, int)
        or not math.isfinite(scale)
        or scale <= 0
    ):
        raise ValueError("Invalid pinball schedule")
    if not math.isfinite(initial):
        raise ValueError("Initial iterate must be finite")
    start_rounds = bus.traffic.rounds
    # Unique ownership is predetermined: no overlap/weight discovery required.
    local_sorted = tuple(np.sort(x, axis=1) for x in values)
    tau = (rank - 0.5) / nobs
    q = np.full((bus.n, batches), initial)
    for ell in range(iterations):
        grads = np.empty_like(q)
        for i, a in enumerate(local_sorted):
            if batches == 1:
                lt = np.array([np.searchsorted(a[0], q[i, 0], side="left")])
            else:
                lt = np.sum(a < q[i, :, None], axis=1)
            grads[i] = lt - tau * a.shape[1]
        step = scale / (nobs * (ell + 1) ** 0.75)
        q = bus.mix(q) - step * grads
    summary = []
    for i, a in enumerate(local_sorted):
        amax = np.max(a, axis=1) if a.shape[1] else np.full(batches, -np.inf)
        summary.append((amax, q[i]))
    extrema = bus.flood(
        summary,
        lambda xs: (
            np.maximum.reduce([x[0] for x in xs]),
            np.maximum.reduce([x[1] for x in xs]),
        ),
    )
    y = np.minimum(extrema[0][0], extrema[0][1])
    recovery = 0
    while True:
        summaries = []
        for a in local_sorted:
            if batches == 1:
                nle = int(np.searchsorted(a[0], y[0], side="right"))
                successor = float(a[0, nle]) if nle < a.shape[1] else np.inf
                summaries.append(
                    (np.array([nle], dtype=np.int64), np.array([successor]))
                )
            else:
                nle = np.sum(a <= y[:, None], axis=1, dtype=np.int64)
                successor = (
                    np.min(np.where(a > y[:, None], a, np.inf), axis=1)
                    if a.shape[1]
                    else np.full(batches, np.inf)
                )
                summaries.append((nle, successor))
        ranks = bus.flood(
            summaries,
            lambda xs: (
                np.add.reduce([x[0] for x in xs]),
                np.minimum.reduce([x[1] for x in xs]),
            ),
        )
        active = ranks[0][0] < rank
        if not np.any(active):
            break
        successor = ranks[0][1]
        if np.any(~np.isfinite(successor[active])) or np.any(
            successor[active] <= y[active]
        ):
            raise AssertionError("Rank recovery failed to progress")
        y = np.where(active, successor, y)
        recovery += 1
        if recovery > rank:
            raise AssertionError("Exceeded the paper's finite recovery bound")
    agreement = all(
        np.array_equal(x[0], ranks[0][0]) and np.array_equal(x[1], ranks[0][1])
        for x in ranks
    )
    if not agreement:
        raise AssertionError("Nodes disagree")
    return OrderStatistic(
        y.copy(),
        rank,
        nobs,
        iterations,
        recovery,
        bus.traffic.rounds - start_rounds,
        agreement,
    )
