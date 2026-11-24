"""Interfaces for analytical bounds on the sensitivity of the formation score. The case study uses configured coefficients."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeAlias

import numpy as np


Array: TypeAlias = np.ndarray
ActiveSet: TypeAlias = tuple[int, ...]


@dataclass(frozen=True)
class AnalyticSensitivityResult(Mapping[str, float]):
    """Sensitivity bounds for each robot and their certification status. NaN denotes a missing bound."""

    kappas: Mapping[str, float]
    available: bool
    certified: bool
    status: str
    blocker: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "kappas",
            {str(robot_id): float(value) for robot_id, value in self.kappas.items()},
        )
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.available and self.certified:
            invalid = [
                robot_id
                for robot_id, value in self.kappas.items()
                if not np.isfinite(value) or value < 0.0
            ]
            if invalid:
                raise ValueError(
                    "A certified analytic result must have finite nonnegative "
                    f"kappas: {sorted(invalid)!r}"
                )

    @property
    def robot_kappa(self) -> Mapping[str, float]:
        """Compatibility alias used by shared MA-ICP diagnostics."""

        return self.kappas

    @property
    def reason(self) -> str | None:
        """Compatibility alias for consumers that call the blocker a reason."""

        return self.blocker

    def __getitem__(self, robot_id: str) -> float:
        return self.kappas[robot_id]

    def __iter__(self):
        return iter(self.kappas)

    def __len__(self) -> int:
        return len(self.kappas)


def _context_platforms(context: object) -> Mapping[str, object]:
    platforms = getattr(context, "platforms", context)
    if not isinstance(platforms, Mapping):
        raise TypeError("Analytic sensitivity context must expose a platform mapping")
    return platforms


def unavailable_analytic_sensitivity(
    context: object,
    blocker: str = (
        "certified comparison bounds A_k,b_k and uniform QP bounds are "
        "unavailable"
    ),
    *,
    metadata: Mapping[str, object] | None = None,
) -> AnalyticSensitivityResult:
    """Return an unavailable analytical bound for each robot in the supplied context."""

    kappas = {
        str(robot_id): float("nan")
        for robot_id in _context_platforms(context)
    }
    return AnalyticSensitivityResult(
        kappas=kappas,
        available=False,
        certified=False,
        status="unavailable",
        blocker=blocker,
        metadata=dict(metadata or {}),
    )


# Compatibility name for callers that use the kappa terminology.
unavailable_analytic_kappa = unavailable_analytic_sensitivity


@dataclass(frozen=True)
class UniformQPBounds:
    """Explicit certificate marker for bounds uniform over the controller domain.

    Values are supplied by the application derivation.  This class does not
    choose, infer, or estimate them.  ``certified`` must be true before a
    result can be marked analytic and certified.
    """

    values: Mapping[str, float]
    certified: bool

    def __post_init__(self) -> None:
        normalized: dict[str, float] = {}
        for name, value in self.values.items():
            numeric = float(value)
            if not np.isfinite(numeric) or numeric < 0.0:
                raise ValueError(f"Uniform QP bound {name!r} must be finite and nonnegative")
            normalized[str(name)] = numeric
        object.__setattr__(self, "values", normalized)


@dataclass(frozen=True)
class BlockComparisonBounds:
    """Comparison matrices and input bounds for each discrete event."""

    transition_bounds: Sequence[Array]
    input_bounds: Sequence[Array]
    intersample_transition_bounds: Sequence[Array] | None = None
    intersample_input_bounds: Sequence[Array] | None = None


@dataclass(frozen=True)
class BlockComparisonRecursion:
    """Validated gains produced by :func:`block_comparison_recursion`."""

    gains: tuple[Array, ...]
    interval_gains: tuple[Array, ...]
    maximum_gain: Array


def _as_nonnegative_matrix(value: Array, name: str, dimension: int | None = None) -> Array:
    matrix = np.asarray(value, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    if dimension is not None and matrix.shape != (dimension, dimension):
        raise ValueError(f"{name} has incompatible dimension {matrix.shape}")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError(f"{name} must be finite and nonnegative")
    return matrix


def _as_nonnegative_vector(value: Array, name: str, dimension: int | None = None) -> Array:
    vector = np.asarray(value, dtype=float)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be a vector")
    if dimension is not None and vector.shape != (dimension,):
        raise ValueError(f"{name} has incompatible dimension {vector.shape}")
    if not np.all(np.isfinite(vector)) or np.any(vector < 0.0):
        raise ValueError(f"{name} must be finite and nonnegative")
    return vector


def block_comparison_recursion(
    transition_bounds: Sequence[Array],
    input_bounds: Sequence[Array],
    *,
    intersample_transition_bounds: Sequence[Array] | None = None,
    intersample_input_bounds: Sequence[Array] | None = None,
    dimension: int | None = None,
) -> BlockComparisonRecursion:
    """Propagate the block comparison bounds over the sequence of events.

    For a normalized margin perturbation ``Delta``, the caller supplies bounds
    satisfying ``d[k+1] <= A[k] d[k] + b[k] Delta``.  This helper computes
    ``G[0] = 0`` and ``G[k+1] = A[k] @ G[k] + b[k]``.  Optional intersample
    bounds compute ``Gamma[k] = C[k] @ G[k] + c[k]`` and are included in the
    componentwise maximum.  All arrays are copied and validated. No bound is
    inferred from graph size, QP dimension, or observed trajectories.
    """

    transitions = tuple(transition_bounds)
    inputs = tuple(input_bounds)
    if len(transitions) != len(inputs):
        raise ValueError("Each comparison event requires both A_k and b_k bounds")

    if transitions:
        first_matrix = _as_nonnegative_matrix(transitions[0], "A_0", dimension)
        inferred_dimension = first_matrix.shape[0]
    elif dimension is not None:
        if dimension < 1:
            raise ValueError("dimension must be positive")
        inferred_dimension = int(dimension)
    else:
        # An empty sequence carries no information from which a network size
        # could be safely inferred.
        raise ValueError("dimension is required when no comparison events are supplied")

    matrices: list[Array] = []
    vectors: list[Array] = []
    for index, (matrix, vector) in enumerate(zip(transitions, inputs, strict=True)):
        matrices.append(
            _as_nonnegative_matrix(matrix, f"A_{index}", inferred_dimension).copy()
        )
        vectors.append(
            _as_nonnegative_vector(vector, f"b_{index}", inferred_dimension).copy()
        )

    if (intersample_transition_bounds is None) != (intersample_input_bounds is None):
        raise ValueError(
            "Intersample C_k and c_k bounds must be supplied together"
        )
    interval_matrices: tuple[Array, ...] = ()
    interval_vectors: tuple[Array, ...] = ()
    if intersample_transition_bounds is not None:
        interval_matrices = tuple(intersample_transition_bounds)
        interval_vectors = tuple(
            () if intersample_input_bounds is None else intersample_input_bounds
        )
        if len(interval_matrices) != len(matrices) or len(interval_vectors) != len(matrices):
            raise ValueError(
                "Intersample bounds must have one C_k,c_k pair per event"
            )
        interval_matrices = tuple(
            _as_nonnegative_matrix(matrix, f"C_{index}", inferred_dimension).copy()
            for index, matrix in enumerate(interval_matrices)
        )
        interval_vectors = tuple(
            _as_nonnegative_vector(vector, f"c_{index}", inferred_dimension).copy()
            for index, vector in enumerate(interval_vectors)
        )

    gains: list[Array] = [np.zeros(inferred_dimension, dtype=float)]
    for matrix, vector in zip(matrices, vectors, strict=True):
        gains.append(matrix @ gains[-1] + vector)

    interval_gains = tuple(
        matrix @ gains[index] + vector
        for index, (matrix, vector) in enumerate(
            zip(interval_matrices, interval_vectors, strict=True)
        )
    )
    candidates = [*gains, *interval_gains]
    maximum_gain = np.maximum.reduce(candidates)
    return BlockComparisonRecursion(
        gains=tuple(gains),
        interval_gains=interval_gains,
        maximum_gain=maximum_gain,
    )


class ActiveSetSelector(Protocol):
    """Hook that reports the active QP constraints for one controller state."""

    def __call__(self, context: object) -> Sequence[int]: ...


class KKTMatrixBuilder(Protocol):
    """Hook that builds a KKT/Jacobian matrix for a selected active set."""

    def __call__(self, context: object, active_set: Sequence[int]) -> Array: ...


class KKTMarginRHSBuilder(Protocol):
    """Hook that builds the margin derivative on a selected active set."""

    def __call__(self, context: object, active_set: Sequence[int]) -> Array: ...


@dataclass(frozen=True)
class ActiveSetSensitivityHooks:
    """Application callbacks needed for a future KKT sensitivity derivation.

    These callbacks are interfaces only.  The shared module never guesses an
    active set, differentiates through a switch, or inverts a KKT matrix.
    """

    active_set: ActiveSetSelector | None = None
    kkt_matrix: KKTMatrixBuilder | None = None
    margin_rhs: KKTMarginRHSBuilder | None = None

    @property
    def complete(self) -> bool:
        return (
            self.active_set is not None
            and self.kkt_matrix is not None
            and self.margin_rhs is not None
        )


# A descriptive alias for callers that spell out the KKT part of the API.
KKTActiveSetHooks = ActiveSetSensitivityHooks


def prepare_analytic_sensitivity(
    context: object,
    *,
    comparison_bounds: BlockComparisonBounds | None = None,
    uniform_qp_bounds: UniformQPBounds | None = None,
    active_set_hooks: ActiveSetSensitivityHooks | None = None,
    robot_kappas: Mapping[str, float] | None = None,
) -> AnalyticSensitivityResult:
    """Prepare a bounded analytic result, or report why it is unavailable.

    A certified result requires all of the following supplied
    evidence: finite nonnegative ``A_k,b_k`` (and intersample bounds when the
    score is continuous), a certified uniform QP bound, complete active set
    and KKT derivative hooks, and one finite nonnegative robot coefficient for
    every robot in the context.  This function validates that evidence but
    does not derive any of it.
    """

    platforms = _context_platforms(context)
    robot_ids = tuple(str(robot_id) for robot_id in platforms)
    blockers: list[str] = []
    recursion: BlockComparisonRecursion | None = None
    if comparison_bounds is None:
        blockers.append("certified comparison bounds A_k,b_k are not supplied")
    else:
        try:
            recursion = block_comparison_recursion(
                comparison_bounds.transition_bounds,
                comparison_bounds.input_bounds,
                intersample_transition_bounds=(
                    comparison_bounds.intersample_transition_bounds
                ),
                intersample_input_bounds=comparison_bounds.intersample_input_bounds,
            )
        except (TypeError, ValueError) as error:
            blockers.append(f"invalid comparison bounds: {error}")

    if uniform_qp_bounds is None:
        blockers.append("uniform QP bounds are not supplied")
    elif not uniform_qp_bounds.certified:
        blockers.append("uniform QP bounds are not certified")
    elif not uniform_qp_bounds.values:
        blockers.append("uniform QP bounds are empty")

    if active_set_hooks is None or not active_set_hooks.complete:
        blockers.append("KKT/active-set sensitivity hooks are not complete")

    normalized_kappas: dict[str, float] = {}
    if robot_kappas is None:
        blockers.append("robot-level analytic kappas are not supplied")
    else:
        missing = sorted(set(robot_ids).difference(robot_kappas), key=repr)
        unknown = sorted(set(robot_kappas).difference(robot_ids), key=repr)
        if missing:
            blockers.append(f"missing robot-level kappas: {missing!r}")
        if unknown:
            blockers.append(f"unknown robot-level kappas: {unknown!r}")
        for robot_id in robot_ids:
            if robot_id not in robot_kappas:
                normalized_kappas[robot_id] = float("nan")
                continue
            try:
                value = float(robot_kappas[robot_id])
            except (TypeError, ValueError):
                value = float("nan")
            if not np.isfinite(value) or value < 0.0:
                blockers.append(f"invalid robot-level kappa for {robot_id!r}")
                normalized_kappas[robot_id] = float("nan")
            else:
                normalized_kappas[robot_id] = value

    if blockers:
        return AnalyticSensitivityResult(
            kappas={robot_id: float("nan") for robot_id in robot_ids},
            available=False,
            certified=False,
            status="unavailable",
            blocker="; ".join(dict.fromkeys(blockers)),
            metadata={
                "required_comparison_bounds": True,
                "required_uniform_qp_bounds": True,
                "required_active_set_hooks": True,
            },
        )

    assert recursion is not None
    return AnalyticSensitivityResult(
        kappas=normalized_kappas,
        available=True,
        certified=True,
        status="available",
        metadata={
            "comparison_events": len(recursion.gains) - 1,
            "maximum_block_gain": recursion.maximum_gain.tolist(),
            "uniform_qp_bound_names": sorted(uniform_qp_bounds.values),
        },
    )


__all__ = [
    "ActiveSetSensitivityHooks",
    "AnalyticSensitivityResult",
    "BlockComparisonBounds",
    "BlockComparisonRecursion",
    "KKTActiveSetHooks",
    "UniformQPBounds",
    "block_comparison_recursion",
    "prepare_analytic_sensitivity",
    "unavailable_analytic_kappa",
    "unavailable_analytic_sensitivity",
]
