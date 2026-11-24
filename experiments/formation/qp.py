"""Quadratic programs and barrier constraints for planar acceleration control."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite, sqrt
from typing import Any, Literal

import cvxpy as cp
import numpy as np

try:  # Keep this module useful in small standalone tests as well.
    from maicp.configuration import Platform
except ImportError:  # pragma: no cover - only relevant outside the repository
    Platform = Any  # type: ignore[misc,assignment]


Array = np.ndarray
Role = Literal["i", "j"]


def _vector(value: Any, name: str) -> Array:
    result = np.asarray(value, dtype=float)
    if result.shape != (2,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite 2D vector")
    return result.copy()


def _scalar(value: Any, name: str, *, nonnegative: bool = False, positive: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite scalar") from exc
    if not isfinite(result) or (positive and result <= 0.0) or (nonnegative and result < 0.0):
        qualifier = "positive" if positive else "nonnegative" if nonnegative else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _platform(value: Any) -> Any:
    """Normalize a platform identifier without importing the configuration module."""

    if Platform is Any:  # pragma: no cover - fallback path only
        return value
    try:
        return Platform(value)
    except (TypeError, ValueError):
        return value


@dataclass(frozen=True)
class WangState:
    """Planar position and velocity used by the barrier controller."""

    robot_id: str
    platform: Any
    position: Array
    velocity: Array
    acceleration_limit: float = 1.0
    radius: float = 0.12
    margin: float = 0.0
    max_speed: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "platform", _platform(self.platform))
        object.__setattr__(self, "position", _vector(self.position, "position"))
        object.__setattr__(self, "velocity", _vector(self.velocity, "velocity"))
        object.__setattr__(
            self,
            "acceleration_limit",
            _scalar(self.acceleration_limit, "acceleration_limit", positive=True),
        )
        object.__setattr__(self, "radius", _scalar(self.radius, "radius", nonnegative=True))
        object.__setattr__(self, "margin", _scalar(self.margin, "margin", nonnegative=True))
        if self.max_speed is not None:
            object.__setattr__(self, "max_speed", _scalar(self.max_speed, "max_speed", positive=True))

    @property
    def alpha(self) -> float:
        """Wang's acceleration limit in the infinity norm (Eqs. (5) and (12))."""

        return self.acceleration_limit

    def neighbor_packet(self) -> "WangNeighborPacket":
        return WangNeighborPacket(
            robot_id=self.robot_id,
            platform=self.platform,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            acceleration_limit=self.acceleration_limit,
            radius=self.radius,
            margin=self.margin,
        )


@dataclass(frozen=True)
class WangNeighborPacket:
    """State and class data used to build a Strategy B constraint."""

    robot_id: str
    platform: Any
    position: Array
    velocity: Array
    acceleration_limit: float
    radius: float = 0.12
    margin: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", _vector(self.position, "position"))
        object.__setattr__(self, "velocity", _vector(self.velocity, "velocity"))
        object.__setattr__(
            self,
            "acceleration_limit",
            _scalar(self.acceleration_limit, "acceleration_limit", positive=True),
        )
        object.__setattr__(self, "radius", _scalar(self.radius, "radius", nonnegative=True))
        object.__setattr__(self, "margin", _scalar(self.margin, "margin", nonnegative=True))


@dataclass(frozen=True)
class WangObstacle:
    """Circular obstacle represented as a stationary agent in the Wang controller."""

    center: Array
    radius: float
    obstacle_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", _vector(self.center, "center"))
        object.__setattr__(self, "radius", _scalar(self.radius, "radius", nonnegative=True))


@dataclass(frozen=True)
class WangWall:
    """Static halfspace wall with admissible region ``normal @ p <= offset``."""

    normal: Array
    offset: float
    wall_id: str | None = None

    def __post_init__(self) -> None:
        normal = _vector(self.normal, "wall normal")
        norm = float(np.linalg.norm(normal))
        if not np.isclose(norm, 1.0, rtol=0.0, atol=1e-10):
            raise ValueError("wall normal must have unit Euclidean norm")
        object.__setattr__(self, "normal", normal)
        object.__setattr__(self, "offset", _scalar(self.offset, "wall offset"))


@dataclass(frozen=True)
class WangParameters:
    """Design values shared by all classes."""

    acceleration_limit: float
    gamma: float
    radius: float = 0.12
    margin: float = 0.0
    max_speed: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "acceleration_limit",
            _scalar(self.acceleration_limit, "acceleration_limit", positive=True),
        )
        object.__setattr__(self, "gamma", _scalar(self.gamma, "gamma", nonnegative=True))
        object.__setattr__(self, "radius", _scalar(self.radius, "radius", nonnegative=True))
        object.__setattr__(self, "margin", _scalar(self.margin, "margin", nonnegative=True))
        if self.max_speed is not None:
            object.__setattr__(self, "max_speed", _scalar(self.max_speed, "max_speed", positive=True))


@dataclass(frozen=True)
class WangConstraint:
    """One local affine CBF inequality.

    ``bound`` is the right side at ``omega=1``.  When optimal decay is
    enabled the actual inequality is

    ``normal @ u <= bound + decay_coefficient * (omega - 1)``.

    Thus fixing ``omega=1`` recovers the ordinary controller with linear decay
    exactly, without maintaining a second constraint representation.
    """

    key: str
    normal: Array
    bound: float
    kind: str = "barrier"
    decay_coefficient: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "normal", _vector(self.normal, "constraint normal"))
        object.__setattr__(self, "bound", _scalar(self.bound, "constraint bound"))
        object.__setattr__(
            self,
            "decay_coefficient",
            _scalar(self.decay_coefficient, "constraint decay coefficient"),
        )


@dataclass(frozen=True)
class WangQPDiagnostics:
    """Diagnostics from the planar quadratic program for one robot."""

    feasible: bool
    status: str
    active_constraint_keys: tuple[str, ...] = ()
    active_normal_matrix: Array = field(default_factory=lambda: np.empty((0, 2)))
    min_active_singular_value: float = float("nan")
    active_condition_number: float = float("nan")
    control_norm: float = float("nan")
    acceleration_limit_slack: float = float("nan")
    objective: float = float("nan")
    optimized_decay_multipliers: Mapping[str, float] = field(default_factory=dict)
    optimized_decay_coefficients: Mapping[str, float] = field(default_factory=dict)
    minimum_decay_multiplier: float = float("nan")
    maximum_decay_multiplier: float = float("nan")
    fraction_decay_multipliers_changed: float = 0.0
    decay_penalty: float = 0.0

    def __post_init__(self) -> None:
        matrix = np.asarray(self.active_normal_matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != 2 or not np.all(np.isfinite(matrix)):
            raise ValueError("active_normal_matrix must be finite with two columns")
        object.__setattr__(self, "active_normal_matrix", matrix.copy())
        omegas = {str(key): float(value) for key, value in self.optimized_decay_multipliers.items()}
        gammas = {str(key): float(value) for key, value in self.optimized_decay_coefficients.items()}
        if any(not isfinite(value) or value < 0.0 for value in (*omegas.values(), *gammas.values())):
            raise ValueError("optimized Wang decay values must be finite and nonnegative")
        object.__setattr__(self, "optimized_decay_multipliers", omegas)
        object.__setattr__(self, "optimized_decay_coefficients", gammas)

@dataclass(frozen=True)
class WangControlStep:
    """Solution of the Wang quadratic program for one robot."""

    control: Array
    qp_diagnostics: WangQPDiagnostics
    constraints: tuple[WangConstraint, ...] = ()
    failed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "control", _vector(self.control, "control"))

def _pair_inputs(
    p_i: Any,
    p_j: Any,
    v_i: Any,
    v_j: Any,
    alpha_i: Any,
    alpha_j: Any,
    D_ij: Any,
    gamma: Any,
) -> tuple[Array, Array, Array, Array, float, float, float, float, Array, Array, float, float, float]:
    """Validate and derive common pair quantities.

    Returns ``(p_i,p_j,v_i,v_j,alpha_i,alpha_j,D,gamma,q,w,r,s)`` with
    ``s = sqrt(2*(alpha_i+alpha_j)*(r-D))``.  Keeping this calculation in one
    place prevents equation (8) and equation (11) from acquiring subtly
    different conventions near the singularity.
    """

    p_i = _vector(p_i, "p_i")
    p_j = _vector(p_j, "p_j")
    v_i = _vector(v_i, "v_i")
    v_j = _vector(v_j, "v_j")
    alpha_i = _scalar(alpha_i, "alpha_i", positive=True)
    alpha_j = _scalar(alpha_j, "alpha_j", nonnegative=True)
    D_ij = _scalar(D_ij, "D_ij", nonnegative=True)
    gamma = _scalar(gamma, "gamma", nonnegative=True)
    q = p_i - p_j
    w = v_i - v_j
    r = float(np.linalg.norm(q))
    if not isfinite(r) or r <= 0.0:
        raise ValueError("Wang pair requires a nonzero relative position")
    gap = r - D_ij
    if gap <= 0.0:
        # The barrier derivative is singular at zero clearance and undefined inside the boundary.
        raise ValueError("Wang pair requires ||p_i-p_j|| > D_ij")
    s = sqrt(2.0 * (alpha_i + alpha_j) * gap)
    return p_i, p_j, v_i, v_j, alpha_i, alpha_j, D_ij, gamma, q, w, r, s, gap


def _barrier_from_quantities(q: Array, w: Array, r: float, s: float) -> float:
    # Wang et al. ACC 2016, Eq. (8).
    return float(s + q @ w / r)


def _rhs_from_quantities(q: Array, w: Array, r: float, s: float, alpha_sum: float, gamma: float) -> float:
    # Bound computed from the state after replacing the source's cubic alpha with the
    # standard linear extended class K choice alpha(h)=gamma*h.
    dot = float(q @ w)
    h = _barrier_from_quantities(q, w, r, s)
    return float(
        gamma * h * r
        - dot**2 / r**2
        + float(w @ w)
        + alpha_sum * dot / s
    )


def h_ij(
    p_i: Any,
    p_j: Any,
    v_i: Any,
    v_j: Any,
    alpha_i: float,
    alpha_j: float,
    D_ij: float,
    gamma: float | None = None,
) -> float:
    """Evaluate Wang's barrier ``h_ij`` for the pair (Eq. (8)).

    The controller interface accepts ``gamma``. Equation (8) depends only on the state and acceleration bounds.
    """

    del gamma
    *_, q, w, r, s, _ = _pair_inputs(p_i, p_j, v_i, v_j, alpha_i, alpha_j, D_ij, 0.0)
    return _barrier_from_quantities(q, w, r, s)


def b_ij(
    p_i: Any,
    p_j: Any,
    v_i: Any,
    v_j: Any,
    alpha_i: float,
    alpha_j: float,
    D_ij: float,
    gamma: float,
) -> float:
    """Evaluate the bound from the state using the linear function ``alpha(h)=gamma*h``."""

    *_, q, w, r, s, _ = _pair_inputs(p_i, p_j, v_i, v_j, alpha_i, alpha_j, D_ij, gamma)
    return _rhs_from_quantities(q, w, r, s, float(alpha_i) + float(alpha_j), float(gamma))


def pair_barrier_from_states(
    state_i: WangState,
    state_j: WangState,
    D_ij: float | None = None,
) -> float:
    if D_ij is None:
        D_ij = state_i.radius + state_j.radius
    return h_ij(
        state_i.position,
        state_j.position,
        state_i.velocity,
        state_j.velocity,
        state_i.acceleration_limit,
        state_j.acceleration_limit,
        D_ij,
    )


def pair_rhs_from_states(
    state_i: WangState,
    state_j: WangState,
    gamma: float,
    D_ij: float | None = None,
) -> float:
    if D_ij is None:
        D_ij = state_i.radius + state_j.radius
    return b_ij(
        state_i.position,
        state_j.position,
        state_i.velocity,
        state_j.velocity,
        state_i.acceleration_limit,
        state_j.acceleration_limit,
        D_ij,
        gamma,
    )


def pair_b_hat(
    p_i: Any,
    p_j: Any,
    v_i: Any,
    v_j: Any,
    alpha_i: float,
    alpha_j: float,
    D_ij: float,
    gamma: float,
    d_hat_i: Any,
    d_hat_j: Any,
) -> float:
    """Implemented learned RHS ``b_hat_ij`` (paper request, learned fields)."""

    q = _vector(p_i, "p_i") - _vector(p_j, "p_j")
    d_hat_i = _vector(d_hat_i, "d_hat_i")
    d_hat_j = _vector(d_hat_j, "d_hat_j")
    return float(b_ij(p_i, p_j, v_i, v_j, alpha_i, alpha_j, D_ij, gamma) + q @ (d_hat_i - d_hat_j))


def pair_B_hat(
    p_i: Any,
    p_j: Any,
    v_i: Any,
    v_j: Any,
    u_i: Any,
    u_j: Any,
    alpha_i: float,
    alpha_j: float,
    D_ij: float,
    gamma: float,
    d_hat_i: Any,
    d_hat_j: Any,
) -> float:
    """Return ``Bhat_ij = bhat_ij + q.T @ (u_i-u_j)``."""

    q = _vector(p_i, "p_i") - _vector(p_j, "p_j")
    u_i = _vector(u_i, "u_i")
    u_j = _vector(u_j, "u_j")
    return float(pair_b_hat(p_i, p_j, v_i, v_j, alpha_i, alpha_j, D_ij, gamma, d_hat_i, d_hat_j) + q @ (u_i - u_j))


def pair_b_true(
    p_i: Any,
    p_j: Any,
    v_i: Any,
    v_j: Any,
    alpha_i: float,
    alpha_j: float,
    D_ij: float,
    gamma: float,
    d_i: Any,
    d_j: Any,
    w_i: Any | None = None,
    w_j: Any | None = None,
) -> float:
    """True RHS including true fields and unobserved acceleration noise."""

    if w_i is None:
        w_i = np.zeros(2)
    if w_j is None:
        w_j = np.zeros(2)
    q = _vector(p_i, "p_i") - _vector(p_j, "p_j")
    d_i = _vector(d_i, "d_i")
    d_j = _vector(d_j, "d_j")
    w_i = _vector(w_i, "w_i")
    w_j = _vector(w_j, "w_j")
    return float(b_ij(p_i, p_j, v_i, v_j, alpha_i, alpha_j, D_ij, gamma) + q @ (d_i + w_i - d_j - w_j))


def pair_B_true(
    p_i: Any,
    p_j: Any,
    v_i: Any,
    v_j: Any,
    u_i: Any,
    u_j: Any,
    alpha_i: float,
    alpha_j: float,
    D_ij: float,
    gamma: float,
    d_i: Any,
    d_j: Any,
    w_i: Any | None = None,
    w_j: Any | None = None,
) -> float:
    """Return ``B_ij = btrue_ij + q.T @ (u_i-u_j)``."""

    q = _vector(p_i, "p_i") - _vector(p_j, "p_j")
    u_i = _vector(u_i, "u_i")
    u_j = _vector(u_j, "u_j")
    return float(pair_b_true(p_i, p_j, v_i, v_j, alpha_i, alpha_j, D_ij, gamma, d_i, d_j, w_i, w_j) + q @ (u_i - u_j))


def _obstacle_geometry(
    p_i: Any,
    v_i: Any,
    alpha_i: float,
    robot_radius: float,
    obstacle_center: Any,
    obstacle_radius: float,
    gamma: float,
) -> tuple[Array, Array, float, float, float, float, float]:
    p_i = _vector(p_i, "p_i")
    v_i = _vector(v_i, "v_i")
    obstacle_center = _vector(obstacle_center, "obstacle_center")
    alpha_i = _scalar(alpha_i, "alpha_i", positive=True)
    robot_radius = _scalar(robot_radius, "robot_radius", nonnegative=True)
    obstacle_radius = _scalar(obstacle_radius, "obstacle_radius", nonnegative=True)
    gamma = _scalar(gamma, "gamma", nonnegative=True)
    q = p_i - obstacle_center
    r = float(np.linalg.norm(q))
    D_io = robot_radius + obstacle_radius
    gap = r - D_io
    if r <= 0.0 or gap <= 0.0:
        raise ValueError("Wang obstacle requires ||p_i-p_obstacle|| > D_io")
    s = sqrt(2.0 * alpha_i * gap)  # alpha_obstacle=0 stationary agent
    return q, v_i, alpha_i, D_io, r, s, gamma


def obstacle_h(
    p_i: Any,
    v_i: Any,
    alpha_i: float,
    robot_radius: float,
    obstacle_center: Any,
    obstacle_radius: float,
    gamma: float | None = None,
) -> float:
    """Evaluate Eq. (8) for a stationary obstacle with ``v_j=u_j=alpha_j=0``."""

    del gamma
    q, v_i, _, _, r, s, _ = _obstacle_geometry(
        p_i, v_i, alpha_i, robot_radius, obstacle_center, obstacle_radius, 0.0
    )
    return _barrier_from_quantities(q, v_i, r, s)


def obstacle_b(
    p_i: Any,
    v_i: Any,
    alpha_i: float,
    robot_radius: float,
    obstacle_center: Any,
    obstacle_radius: float,
    gamma: float,
) -> float:
    """Evaluate Eq. (11) for a stationary obstacle with alpha_obstacle=0."""

    q, v_i, alpha_i, _, r, s, gamma = _obstacle_geometry(
        p_i, v_i, alpha_i, robot_radius, obstacle_center, obstacle_radius, gamma
    )
    return _rhs_from_quantities(q, v_i, r, s, alpha_i, gamma)


def obstacle_b_hat(
    p_i: Any,
    v_i: Any,
    alpha_i: float,
    robot_radius: float,
    obstacle_center: Any,
    obstacle_radius: float,
    gamma: float,
    d_hat_i: Any,
) -> float:
    q = _vector(p_i, "p_i") - _vector(obstacle_center, "obstacle_center")
    d_hat_i = _vector(d_hat_i, "d_hat_i")
    return float(obstacle_b(p_i, v_i, alpha_i, robot_radius, obstacle_center, obstacle_radius, gamma) + q @ d_hat_i)


def obstacle_B_hat(
    p_i: Any,
    v_i: Any,
    u_i: Any,
    alpha_i: float,
    robot_radius: float,
    obstacle_center: Any,
    obstacle_radius: float,
    gamma: float,
    d_hat_i: Any,
) -> float:
    q = _vector(p_i, "p_i") - _vector(obstacle_center, "obstacle_center")
    u_i = _vector(u_i, "u_i")
    return float(obstacle_b_hat(p_i, v_i, alpha_i, robot_radius, obstacle_center, obstacle_radius, gamma, d_hat_i) + q @ u_i)


def obstacle_b_true(
    p_i: Any,
    v_i: Any,
    alpha_i: float,
    robot_radius: float,
    obstacle_center: Any,
    obstacle_radius: float,
    gamma: float,
    d_i: Any,
    w_i: Any | None = None,
) -> float:
    if w_i is None:
        w_i = np.zeros(2)
    q = _vector(p_i, "p_i") - _vector(obstacle_center, "obstacle_center")
    d_i = _vector(d_i, "d_i")
    w_i = _vector(w_i, "w_i")
    return float(obstacle_b(p_i, v_i, alpha_i, robot_radius, obstacle_center, obstacle_radius, gamma) + q @ (d_i + w_i))


def obstacle_B_true(
    p_i: Any,
    v_i: Any,
    u_i: Any,
    alpha_i: float,
    robot_radius: float,
    obstacle_center: Any,
    obstacle_radius: float,
    gamma: float,
    d_i: Any,
    w_i: Any | None = None,
) -> float:
    q = _vector(p_i, "p_i") - _vector(obstacle_center, "obstacle_center")
    u_i = _vector(u_i, "u_i")
    return float(obstacle_b_true(p_i, v_i, alpha_i, robot_radius, obstacle_center, obstacle_radius, gamma, d_i, w_i) + q @ u_i)


def wall_clearance(
    p_i: Any,
    robot_radius: float,
    wall: WangWall,
) -> float:
    """Return the distance from the robot surface to the wall."""

    if not isinstance(wall, WangWall):
        wall = WangWall(wall.normal, wall.offset, getattr(wall, "wall_id", None))
    p_i = _vector(p_i, "p_i")
    radius = _scalar(robot_radius, "robot_radius", nonnegative=True)
    return float(wall.offset - wall.normal @ p_i - radius)


def wall_h(
    p_i: Any,
    v_i: Any,
    alpha_i: float,
    robot_radius: float,
    wall: WangWall,
) -> float:
    """Evaluate ``sqrt(2*a_i*g_iw) - n_w.T v_i`` for a wall."""

    if not isinstance(wall, WangWall):
        wall = WangWall(wall.normal, wall.offset, getattr(wall, "wall_id", None))
    velocity = _vector(v_i, "v_i")
    alpha = _scalar(alpha_i, "alpha_i", positive=True)
    clearance = wall_clearance(p_i, robot_radius, wall)
    if clearance <= 0.0:
        raise ValueError("Wang wall barrier requires positive physical clearance")
    return float(sqrt(2.0 * alpha * clearance) - wall.normal @ velocity)


def wall_B_hat(
    p_i: Any,
    v_i: Any,
    u_i: Any,
    alpha_i: float,
    robot_radius: float,
    wall: WangWall,
    gamma: float,
    d_hat_i: Any,
) -> float:
    """Evaluate the wall residual from the learned model before robust tightening."""

    velocity = _vector(v_i, "v_i")
    control = _vector(u_i, "u_i")
    learned = _vector(d_hat_i, "d_hat_i")
    alpha = _scalar(alpha_i, "alpha_i", positive=True)
    gamma = _scalar(gamma, "gamma", nonnegative=True)
    clearance = wall_clearance(p_i, robot_radius, wall)
    if clearance <= 0.0:
        raise ValueError("Wang wall residual requires positive physical clearance")
    barrier = wall_h(p_i, velocity, alpha, robot_radius, wall)
    root = sqrt(2.0 * alpha * clearance)
    return float(
        -wall.normal @ (control + learned)
        - alpha * float(wall.normal @ velocity) / root
        + gamma * barrier
    )


def wall_B_true(
    p_i: Any,
    v_i: Any,
    u_i: Any,
    alpha_i: float,
    robot_radius: float,
    wall: WangWall,
    gamma: float,
    d_i: Any,
    w_i: Any | None = None,
) -> float:
    """True wall residual using realized disturbance and process noise."""

    noise = np.zeros(2) if w_i is None else _vector(w_i, "w_i")
    return wall_B_hat(
        p_i,
        v_i,
        u_i,
        alpha_i,
        robot_radius,
        wall,
        gamma,
        _vector(d_i, "d_i") + noise,
    )


def wall_constraint(
    p_i: Any,
    v_i: Any,
    alpha_i: float,
    robot_radius: float,
    wall: WangWall,
    gamma: float,
    d_hat_i: Any,
    margin_i: float,
    *,
    key: str = "wall",
) -> WangConstraint:
    """Return the robust local wall inequality in WangConstraint form."""

    if not isinstance(wall, WangWall):
        wall = WangWall(wall.normal, wall.offset, getattr(wall, "wall_id", None))
    velocity = _vector(v_i, "v_i")
    learned = _vector(d_hat_i, "d_hat_i")
    alpha = _scalar(alpha_i, "alpha_i", positive=True)
    gamma = _scalar(gamma, "gamma", nonnegative=True)
    margin = _scalar(margin_i, "margin_i", nonnegative=True)
    clearance = wall_clearance(p_i, robot_radius, wall)
    if clearance <= 0.0:
        raise ValueError("Wang wall constraint requires positive physical clearance")
    barrier = wall_h(p_i, velocity, alpha, robot_radius, wall)
    root = sqrt(2.0 * alpha * clearance)
    bound = (
        gamma * barrier
        - float(wall.normal @ learned)
        - alpha * float(wall.normal @ velocity) / root
        - margin
    )
    return WangConstraint(
        key,
        wall.normal,
        bound,
        "wall",
        gamma * barrier,
    )


def pair_eta(alpha_i: float, alpha_j: float) -> tuple[float, float]:
    """Strategy B weights ``(eta_ij, eta_ji)``."""

    alpha_i = _scalar(alpha_i, "alpha_i", nonnegative=True)
    alpha_j = _scalar(alpha_j, "alpha_j", nonnegative=True)
    total = alpha_i + alpha_j
    if total <= 0.0:
        raise ValueError("Strategy-B acceleration limits must have a positive sum")
    return alpha_i / total, alpha_j / total


def eta_ij(alpha_i: float, alpha_j: float) -> float:
    return pair_eta(alpha_i, alpha_j)[0]


def eta_ji(alpha_i: float, alpha_j: float) -> float:
    return pair_eta(alpha_i, alpha_j)[1]


def pair_robust_tightening(
    p_i: Any,
    p_j: Any,
    margin_i: float,
    margin_j: float,
) -> float:
    """Return ``||p_i-p_j||_2 * (r_i+r_j)`` for bounds on acceleration error."""

    q = _vector(p_i, "p_i") - _vector(p_j, "p_j")
    r_i = _scalar(margin_i, "margin_i", nonnegative=True)
    r_j = _scalar(margin_j, "margin_j", nonnegative=True)
    return float(np.linalg.norm(q) * (r_i + r_j))


def obstacle_robust_tightening(
    p_i: Any,
    obstacle_center: Any,
    margin_i: float,
) -> float:
    """Return ``||p_i-p_o||_2 * r_i`` for a bound on acceleration error."""

    q = _vector(p_i, "p_i") - _vector(obstacle_center, "obstacle_center")
    r_i = _scalar(margin_i, "margin_i", nonnegative=True)
    return float(np.linalg.norm(q) * r_i)


def strategy_b_pair_constraints(
    p_i: Any,
    p_j: Any,
    v_i: Any,
    v_j: Any,
    alpha_i: float,
    alpha_j: float,
    D_ij: float,
    gamma: float,
    d_hat_i: Any,
    d_hat_j: Any,
    margin_i: float,
    margin_j: float,
    *,
    key: str = "pair",
    gamma_j: float | None = None,
) -> tuple[WangConstraint, WangConstraint]:
    """Return both allocated Strategy B pair inequalities.

    The first constraint is robot ``i``'s ``-q.T u_i`` side and the second is
    robot ``j``'s ``+q.T u_j`` side.  The full robust RHS is allocated once.
    """

    p_i = _vector(p_i, "p_i")
    p_j = _vector(p_j, "p_j")
    q = p_i - p_j
    eta_i, eta_j = pair_eta(alpha_i, alpha_j)
    gamma_i = _scalar(gamma, "gamma_i", nonnegative=True)
    gamma_j = gamma_i if gamma_j is None else _scalar(gamma_j, "gamma_j", nonnegative=True)
    tightening = pair_robust_tightening(p_i, p_j, margin_i, margin_j)
    rhs_i = pair_b_hat(
        p_i, p_j, v_i, v_j, alpha_i, alpha_j, D_ij, gamma_i, d_hat_i, d_hat_j
    ) - tightening
    rhs_j = pair_b_hat(
        p_i, p_j, v_i, v_j, alpha_i, alpha_j, D_ij, gamma_j, d_hat_i, d_hat_j
    ) - tightening
    h = h_ij(p_i, p_j, v_i, v_j, alpha_i, alpha_j, D_ij)
    distance = float(np.linalg.norm(q))
    return (
        WangConstraint(
            f"{key}:i", -q, eta_i * rhs_i, "pair", eta_i * gamma_i * h * distance
        ),
        WangConstraint(
            f"{key}:j", q, eta_j * rhs_j, "pair", eta_j * gamma_j * h * distance
        ),
    )


def effective_pair_decay(
    eta_i: float,
    gamma_i: float,
    eta_j: float,
    gamma_j: float,
) -> float:
    """Weighted aggregate Strategy B coefficient ``eta_i*gamma_i+eta_j*gamma_j``."""

    eta_i = _scalar(eta_i, "eta_i", nonnegative=True)
    eta_j = _scalar(eta_j, "eta_j", nonnegative=True)
    gamma_i = _scalar(gamma_i, "gamma_i", nonnegative=True)
    gamma_j = _scalar(gamma_j, "gamma_j", nonnegative=True)
    if not np.isclose(eta_i + eta_j, 1.0, rtol=0.0, atol=1e-10):
        raise ValueError("Strategy-B weights must sum to one")
    return float(eta_i * gamma_i + eta_j * gamma_j)


def strategy_b_obstacle_constraint(
    p_i: Any,
    v_i: Any,
    alpha_i: float,
    robot_radius: float,
    obstacle: WangObstacle | Any,
    gamma: float,
    d_hat_i: Any,
    margin_i: float,
    *,
    key: str = "obstacle",
) -> WangConstraint:
    """Return the constraint bound on robot ``i`` for a stationary obstacle (eta=1)."""

    gamma = _scalar(gamma, "gamma", nonnegative=True)
    if not isinstance(obstacle, WangObstacle):
        obstacle = WangObstacle(obstacle.center, obstacle.radius, getattr(obstacle, "obstacle_id", None))
    q = _vector(p_i, "p_i") - obstacle.center
    rhs = obstacle_b_hat(
        p_i,
        v_i,
        alpha_i,
        robot_radius,
        obstacle.center,
        obstacle.radius,
        gamma,
        d_hat_i,
    ) - obstacle_robust_tightening(p_i, obstacle.center, margin_i)
    h = obstacle_h(
        p_i,
        v_i,
        alpha_i,
        robot_radius,
        obstacle.center,
        obstacle.radius,
    )
    return WangConstraint(key, -q, rhs, "obstacle", gamma * h * float(np.linalg.norm(q)))


def pair_constraint_residual(
    p_i: Any,
    p_j: Any,
    u_i: Any,
    u_j: Any,
    b_hat: float,
    tightening: float = 0.0,
) -> float:
    """Return ``Bhat-rho`` for an already evaluated pair RHS."""

    q = _vector(p_i, "p_i") - _vector(p_j, "p_j")
    return float(b_hat + q @ (_vector(u_i, "u_i") - _vector(u_j, "u_j")) - tightening)


def _base_jacobians(
    q: Array,
    w: Array,
    r: float,
    s: float,
    alpha_sum: float,
    gamma: float,
) -> tuple[Array, Array, float, Array, Array]:
    """Derivatives of Eq. (8)/(11) with respect to q and w.

    Returns ``(dh_dq, dh_dw, h, db_dq, db_dw)``.  These are analytic source
    derivatives. No approximation by finite differences is used by the
    controller/certificate code.
    """

    n = q / r
    dot = float(q @ w)
    c = dot / r
    dc_dq = (w - c * n) / r
    dh_dq = alpha_sum / s * n + dc_dq
    dh_dw = n.copy()
    h = _barrier_from_quantities(q, w, r, s)
    db_dq = (
        gamma * (r * dh_dq + h * n)
        - 2.0 * c * dc_dq
        + alpha_sum * w / s
        - (alpha_sum**2 * dot / s**3) * n
    )
    db_dw = gamma * r * n + 2.0 * w - 2.0 * c * n + alpha_sum * q / s
    return dh_dq, dh_dw, h, db_dq, db_dw


def pair_barrier_jacobian(
    p_i: Any,
    p_j: Any,
    v_i: Any,
    v_j: Any,
    alpha_i: float,
    alpha_j: float,
    D_ij: float,
) -> tuple[Array, Array]:
    """Return the row derivatives of the pairwise barrier with respect to relative position and velocity."""

    *_, q, w, r, s, _ = _pair_inputs(p_i, p_j, v_i, v_j, alpha_i, alpha_j, D_ij, 0.0)
    dh_dq, dh_dw, _, _, _ = _base_jacobians(q, w, r, s, float(alpha_i) + float(alpha_j), 0.0)
    return dh_dq, dh_dw


def pair_rhs_jacobian(
    p_i: Any,
    p_j: Any,
    v_i: Any,
    v_j: Any,
    alpha_i: float,
    alpha_j: float,
    D_ij: float,
    gamma: float,
) -> tuple[Array, Array]:
    """Return analytic ``(∂b/∂q,∂b/∂w)`` for Wang Eq. (11)."""

    *_, q, w, r, s, _ = _pair_inputs(p_i, p_j, v_i, v_j, alpha_i, alpha_j, D_ij, gamma)
    _, _, _, db_dq, db_dw = _base_jacobians(q, w, r, s, float(alpha_i) + float(alpha_j), float(gamma))
    return db_dq, db_dw


def pair_b_hat_jacobian(
    p_i: Any,
    p_j: Any,
    v_i: Any,
    v_j: Any,
    alpha_i: float,
    alpha_j: float,
    D_ij: float,
    gamma: float,
    d_hat_i: Any,
    d_hat_j: Any,
) -> tuple[Array, Array, Array]:
    """Return ``(∂bhat/∂q,∂bhat/∂w,∂bhat/∂e)`` for ``e=dhat_i-dhat_j``."""

    db_dq, db_dw = pair_rhs_jacobian(p_i, p_j, v_i, v_j, alpha_i, alpha_j, D_ij, gamma)
    e = _vector(d_hat_i, "d_hat_i") - _vector(d_hat_j, "d_hat_j")
    return db_dq + e, db_dw, _vector(p_i, "p_i") - _vector(p_j, "p_j")


def pair_constraint_jacobians(
    p_i: Any,
    p_j: Any,
    v_i: Any,
    v_j: Any,
    alpha_i: float,
    alpha_j: float,
    D_ij: float,
    gamma: float,
    d_hat_i: Any | None = None,
    d_hat_j: Any | None = None,
    *,
    role: Role = "i",
    allocation: float = 1.0,
    robust_margin: float = 0.0,
) -> dict[str, Array | float]:
    """Analytic state Jacobians of a local Strategy B normal/RHS.

    The state variable order in ``rhs_jacobian`` is ``(p_i,p_j,v_i,v_j)``.
    Learned fields are held fixed. Callers with models that depend on the state can add
    their model Jacobian to the returned ``∂bhat/∂e`` term from
    :func:`pair_b_hat_jacobian`.
    """

    if role not in ("i", "j"):
        raise ValueError("role must be 'i' or 'j'")
    allocation = _scalar(allocation, "allocation", nonnegative=True)
    robust_margin = _scalar(robust_margin, "robust_margin", nonnegative=True)
    p_i = _vector(p_i, "p_i")
    p_j = _vector(p_j, "p_j")
    if d_hat_i is None:
        d_hat_i = np.zeros(2)
    if d_hat_j is None:
        d_hat_j = np.zeros(2)
    db_dq, db_dw, _ = pair_b_hat_jacobian(
        p_i, p_j, v_i, v_j, alpha_i, alpha_j, D_ij, gamma, d_hat_i, d_hat_j
    )
    q = p_i - p_j
    normal = -q if role == "i" else q
    normal_q = -np.eye(2) if role == "i" else np.eye(2)
    rhs_q = allocation * db_dq
    rhs_w = allocation * db_dw
    # Since q=p_i-p_j and w=v_i-v_j, concatenate in state order.
    rhs_jacobian = np.concatenate((rhs_q, -rhs_q, rhs_w, -rhs_w))
    normal_jacobian = np.concatenate((normal_q, -normal_q), axis=1)
    rhs = allocation * (pair_b_hat(p_i, p_j, v_i, v_j, alpha_i, alpha_j, D_ij, gamma, d_hat_i, d_hat_j) - robust_margin)
    return {
        "normal": normal,
        "normal_jacobian": normal_jacobian,
        "rhs": float(rhs),
        "rhs_jacobian": rhs_jacobian,
        "db_dq": db_dq,
        "db_dw": db_dw,
    }


def obstacle_barrier_jacobian(
    p_i: Any,
    v_i: Any,
    alpha_i: float,
    robot_radius: float,
    obstacle_center: Any,
    obstacle_radius: float,
) -> tuple[Array, Array]:
    """Return analytic ``(∂h/∂q,∂h/∂v_i)`` for the obstacle specialization."""

    q, v_i, alpha_i, _, r, s, _ = _obstacle_geometry(
        p_i, v_i, alpha_i, robot_radius, obstacle_center, obstacle_radius, 0.0
    )
    dh_dq, dh_dw, _, _, _ = _base_jacobians(q, v_i, r, s, alpha_i, 0.0)
    return dh_dq, dh_dw


def obstacle_rhs_jacobian(
    p_i: Any,
    v_i: Any,
    alpha_i: float,
    robot_radius: float,
    obstacle_center: Any,
    obstacle_radius: float,
    gamma: float,
) -> tuple[Array, Array]:
    """Return analytic ``(∂b/∂q,∂b/∂v_i)`` for the obstacle specialization."""

    q, v_i, alpha_i, _, r, s, gamma = _obstacle_geometry(
        p_i, v_i, alpha_i, robot_radius, obstacle_center, obstacle_radius, gamma
    )
    _, _, _, db_dq, db_dw = _base_jacobians(q, v_i, r, s, alpha_i, gamma)
    return db_dq, db_dw


def obstacle_constraint_jacobians(
    p_i: Any,
    v_i: Any,
    alpha_i: float,
    robot_radius: float,
    obstacle_center: Any,
    obstacle_radius: float,
    gamma: float,
    d_hat_i: Any | None = None,
    *,
    robust_margin: float = 0.0,
) -> dict[str, Array | float]:
    """Analytic state Jacobians of the local obstacle normal and RHS."""

    if d_hat_i is None:
        d_hat_i = np.zeros(2)
    p_i = _vector(p_i, "p_i")
    obstacle_center = _vector(obstacle_center, "obstacle_center")
    d_hat_i = _vector(d_hat_i, "d_hat_i")
    robust_margin = _scalar(robust_margin, "robust_margin", nonnegative=True)
    db_dq, db_dw = obstacle_rhs_jacobian(
        p_i, v_i, alpha_i, robot_radius, obstacle_center, obstacle_radius, gamma
    )
    q = p_i - obstacle_center
    normal = -q
    normal_jacobian = -np.eye(2)
    rhs_q = db_dq + d_hat_i
    rhs_jacobian = np.concatenate((rhs_q, db_dw))
    rhs = obstacle_b_hat(
        p_i, v_i, alpha_i, robot_radius, obstacle_center, obstacle_radius, gamma, d_hat_i
    ) - robust_margin
    return {
        "normal": normal,
        "normal_jacobian": normal_jacobian,
        "rhs": float(rhs),
        "rhs_jacobian": rhs_jacobian,
        "db_dq": db_dq,
        "db_dw": db_dw,
    }


def _invalid_diagnostics(status: str, *, alpha: float = float("nan")) -> WangQPDiagnostics:
    del alpha
    return WangQPDiagnostics(
        feasible=False,
        status=status,
        active_constraint_keys=(),
        active_normal_matrix=np.empty((0, 2)),
        min_active_singular_value=float("nan"),
        active_condition_number=float("inf"),
        control_norm=float("nan"),
        acceleration_limit_slack=float("nan"),
        objective=float("nan"),
        optimized_decay_multipliers={},
        optimized_decay_coefficients={},
        minimum_decay_multiplier=float("nan"),
        maximum_decay_multiplier=float("nan"),
        fraction_decay_multipliers_changed=float("nan"),
        decay_penalty=float("nan"),
    )


def build_optimal_decay_qp(
    nominal: Any,
    constraints: Sequence[WangConstraint],
    acceleration_limit: float,
    *,
    decay_weight: float,
    decay_multiplier_min: float,
    decay_multiplier_max: float,
) -> tuple[cp.Problem, cp.Variable, cp.Variable, tuple[WangConstraint, ...]]:
    """Build the convex local QP in ``(u, omega)`` without solving it."""

    nominal_v = _vector(nominal, "nominal")
    alpha = _scalar(acceleration_limit, "acceleration_limit", positive=True)
    weight = _scalar(decay_weight, "decay_weight", positive=True)
    omega_min = _scalar(decay_multiplier_min, "decay_multiplier_min", nonnegative=True)
    omega_max = _scalar(decay_multiplier_max, "decay_multiplier_max", nonnegative=True)
    if omega_max < omega_min:
        raise ValueError("decay multiplier bounds must be ordered")
    barriers = tuple(constraint for constraint in constraints if constraint.kind != "input")
    u = cp.Variable(2, name="u")
    omega = cp.Variable(len(barriers), name="omega")
    qp_constraints: list[cp.Constraint] = [u <= alpha, u >= -alpha]
    barrier_index = {id(constraint): index for index, constraint in enumerate(barriers)}
    for constraint in constraints:
        if constraint.kind == "input":
            qp_constraints.append(constraint.normal @ u <= constraint.bound)
            continue
        index = barrier_index[id(constraint)]
        qp_constraints.append(
            constraint.normal @ u
            <= constraint.bound + constraint.decay_coefficient * (omega[index] - 1.0)
        )
    qp_constraints.extend((omega >= omega_min, omega <= omega_max))
    objective = cp.Minimize(
        0.5 * cp.sum_squares(u - nominal_v)
        + weight * cp.sum_squares(omega - 1.0)
    )
    problem = cp.Problem(objective, qp_constraints)
    return problem, u, omega, barriers


class PersistentWangOptimalDecayQP:
    """Reusable CVXPY problem for one robot with a fixed constraint layout."""

    def __init__(
        self,
        constraint_structure: Sequence[tuple[str, str]],
        acceleration_limit: float,
        *,
        decay_weight: float,
        decay_multiplier_min: float,
        decay_multiplier_max: float,
    ) -> None:
        self.constraint_structure = tuple(
            (str(key), str(kind)) for key, kind in constraint_structure
        )
        self.acceleration_limit = _scalar(
            acceleration_limit, "acceleration_limit", positive=True
        )
        self.decay_weight = _scalar(decay_weight, "decay_weight", positive=True)
        self.decay_multiplier_min = _scalar(
            decay_multiplier_min, "decay_multiplier_min", nonnegative=True
        )
        self.decay_multiplier_max = _scalar(
            decay_multiplier_max, "decay_multiplier_max", nonnegative=True
        )
        if self.decay_multiplier_max < self.decay_multiplier_min:
            raise ValueError("decay multiplier bounds must be ordered")

        self.nominal_parameter = cp.Parameter(2, name="u_nom")
        self.normal_parameters = tuple(
            cp.Parameter(2, name=f"cbf_normal_{index}")
            for index in range(len(self.constraint_structure))
        )
        self.rhs_parameters = tuple(
            cp.Parameter(name=f"cbf_rhs_{index}")
            for index in range(len(self.constraint_structure))
        )
        barrier_positions = tuple(
            index
            for index, (_, kind) in enumerate(self.constraint_structure)
            if kind != "input"
        )
        self.barrier_positions = barrier_positions
        self.decay_parameters = tuple(
            cp.Parameter(name=f"decay_coefficient_{index}")
            for index in range(len(barrier_positions))
        )
        self.u_variable = cp.Variable(2, name="u")
        self.omega_variable = cp.Variable(len(barrier_positions), name="omega")
        qp_constraints: list[cp.Constraint] = [
            self.u_variable <= self.acceleration_limit,
            self.u_variable >= -self.acceleration_limit,
        ]
        barrier_index = {position: index for index, position in enumerate(barrier_positions)}
        for position, (_, kind) in enumerate(self.constraint_structure):
            left = self.normal_parameters[position] @ self.u_variable
            if kind == "input":
                qp_constraints.append(left <= self.rhs_parameters[position])
            else:
                index = barrier_index[position]
                qp_constraints.append(
                    left
                    <= self.rhs_parameters[position]
                    + self.decay_parameters[index]
                    * (self.omega_variable[index] - 1.0)
                )
        qp_constraints.extend(
            (
                self.omega_variable >= self.decay_multiplier_min,
                self.omega_variable <= self.decay_multiplier_max,
            )
        )
        objective = cp.Minimize(
            0.5 * cp.sum_squares(self.u_variable - self.nominal_parameter)
            + self.decay_weight * cp.sum_squares(self.omega_variable - 1.0)
        )
        self.problem = cp.Problem(objective, qp_constraints)
        if (
            not self.problem.is_dcp()
            or not self.problem.is_qp()
            or not self.problem.is_dpp()
        ):
            raise ValueError(
                "Persistent Wang optimal-decay problem is not a parameterized convex QP"
            )
        self.solve_count = 0

    def solve(
        self,
        nominal: Array,
        constraints: Sequence[WangConstraint],
        acceleration_limit: float,
        tolerance: float,
    ) -> tuple[cp.Problem, cp.Variable, cp.Variable, tuple[WangConstraint, ...]]:
        normalized = tuple(constraints)
        structure = tuple((constraint.key, constraint.kind) for constraint in normalized)
        if structure != self.constraint_structure:
            raise ValueError("optimal-decay constraint structure changed")
        if acceleration_limit != self.acceleration_limit:
            raise ValueError("optimal-decay acceleration limit changed")

        self.nominal_parameter.value = nominal
        barrier_values: list[float] = []
        for index, constraint in enumerate(normalized):
            self.normal_parameters[index].value = constraint.normal
            self.rhs_parameters[index].value = constraint.bound
            if constraint.kind != "input":
                barrier_values.append(constraint.decay_coefficient)
        for parameter, value in zip(
            self.decay_parameters, barrier_values, strict=True
        ):
            parameter.value = value

        # CVXPY retains the previous primal solution on the variables, which
        # OSQP consumes as the next warm start.  Seed only the first solve.
        if self.solve_count == 0:
            self.u_variable.value = nominal.copy()
            initial_omega = float(
                np.clip(
                    1.0,
                    self.decay_multiplier_min,
                    self.decay_multiplier_max,
                )
            )
            self.omega_variable.value = np.full(len(barrier_values), initial_omega)
        try:
            self.problem.solve(
                solver=cp.OSQP,
                eps_abs=max(tolerance, 1e-9),
                eps_rel=max(tolerance, 1e-9),
                max_iter=20_000,
                polishing=True,
                warm_start=True,
                verbose=False,
            )
        except cp.error.SolverError:
            self.problem.solve(solver=cp.CLARABEL, warm_start=True, verbose=False)
        self.solve_count += 1
        barriers = tuple(
            normalized[position] for position in self.barrier_positions
        )
        return self.problem, self.u_variable, self.omega_variable, barriers


def solve_wang_qp(
    nominal: Any,
    constraints: Sequence[WangConstraint],
    acceleration_limit: float,
    *,
    tolerance: float = 1e-8,
    optimal_decay: bool = False,
    nominal_gamma: float = 1.0,
    decay_weight: float = 10.0,
    decay_multiplier_min: float = 0.0,
    decay_multiplier_max: float = 10.0,
    material_difference_tolerance: float = 1e-3,
    persistent_optimal_decay_qp: PersistentWangOptimalDecayQP | None = None,
) -> tuple[Array, WangQPDiagnostics]:
    """Solve one decentralized Wang local QP.

    Fixed mode uses exact enumeration of active sets in two dimensions.  Optimal
    mode solves the strictly convex quadratic objective in ``(u, omega)``
    subject only to affine inequalities and finite nonnegative omega bounds.
    """

    try:
        nominal = _vector(nominal, "nominal")
        alpha = _scalar(acceleration_limit, "acceleration_limit", positive=True)
        tolerance = _scalar(tolerance, "tolerance", nonnegative=True)
        nominal_gamma = _scalar(nominal_gamma, "nominal_gamma", nonnegative=True)
        material_difference_tolerance = _scalar(
            material_difference_tolerance,
            "material_difference_tolerance",
            nonnegative=True,
        )
    except ValueError as exc:
        return np.zeros(2), _invalid_diagnostics(f"invalid_input:{exc}")
    try:
        normalized = tuple(constraints)
        if any(not isinstance(c, WangConstraint) for c in normalized):
            raise ValueError("constraints must contain WangConstraint values")
    except (TypeError, ValueError) as exc:
        return np.zeros(2), _invalid_diagnostics(f"invalid_constraint:{exc}")

    barriers = tuple(constraint for constraint in normalized if constraint.kind != "input")
    if optimal_decay and not barriers:
        optimal_decay = False
    if optimal_decay:
        try:
            if persistent_optimal_decay_qp is None:
                problem, u_variable, omega_variable, barriers = build_optimal_decay_qp(
                    nominal,
                    normalized,
                    alpha,
                    decay_weight=decay_weight,
                    decay_multiplier_min=decay_multiplier_min,
                    decay_multiplier_max=decay_multiplier_max,
                )
                if not problem.is_dcp() or not problem.is_qp():
                    return np.zeros(2), _invalid_diagnostics("invalid_nonconvex_qp")
                try:
                    problem.solve(
                        solver=cp.OSQP,
                        eps_abs=max(tolerance, 1e-9),
                        eps_rel=max(tolerance, 1e-9),
                        max_iter=20_000,
                        polishing=True,
                        warm_start=False,
                        verbose=False,
                    )
                except cp.error.SolverError:
                    problem.solve(solver=cp.CLARABEL, verbose=False)
            else:
                if (
                    persistent_optimal_decay_qp.decay_weight != float(decay_weight)
                    or persistent_optimal_decay_qp.decay_multiplier_min
                    != float(decay_multiplier_min)
                    or persistent_optimal_decay_qp.decay_multiplier_max
                    != float(decay_multiplier_max)
                ):
                    raise ValueError("persistent optimal-decay QP settings changed")
                problem, u_variable, omega_variable, barriers = (
                    persistent_optimal_decay_qp.solve(
                        nominal, normalized, alpha, tolerance
                    )
                )
        except (TypeError, ValueError, cp.error.SolverError) as exc:
            return np.zeros(2), _invalid_diagnostics(f"invalid_optimal_decay_qp:{exc}")
        if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            return np.zeros(2), _invalid_diagnostics(
                "infeasible" if problem.status in {cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE} else str(problem.status)
            )
        if u_variable.value is None or omega_variable.value is None:
            return np.zeros(2), _invalid_diagnostics("solver_returned_no_solution")
        control = np.asarray(u_variable.value, dtype=float).reshape(2)
        omega_values = np.asarray(omega_variable.value, dtype=float).reshape(len(barriers))
        if not np.all(np.isfinite(control)) or not np.all(np.isfinite(omega_values)):
            return np.zeros(2), _invalid_diagnostics("solver_returned_nonfinite_solution")
        omega_min = float(decay_multiplier_min)
        omega_max = float(decay_multiplier_max)
        omega_values = np.clip(omega_values, omega_min, omega_max)
    else:
        omega_values = np.ones(len(barriers), dtype=float)

    # Eq. (12): ||u||_infinity <= alpha.  Keep box keys distinct so that
    # diagnostics reveal which acceleration face was active.
    all_constraints = list(normalized) + [
        WangConstraint("accel:+x", np.array([1.0, 0.0]), alpha, "input"),
        WangConstraint("accel:-x", np.array([-1.0, 0.0]), alpha, "input"),
        WangConstraint("accel:+y", np.array([0.0, 1.0]), alpha, "input"),
        WangConstraint("accel:-y", np.array([0.0, -1.0]), alpha, "input"),
    ]
    normals = np.stack([constraint.normal for constraint in all_constraints])
    bounds = np.asarray([constraint.bound for constraint in all_constraints], dtype=float)
    if not np.all(np.isfinite(normals)) or not np.all(np.isfinite(bounds)):
        return np.zeros(2), _invalid_diagnostics("invalid_constraint")

    if optimal_decay:
        barrier_omega = {id(constraint): omega_values[index] for index, constraint in enumerate(barriers)}
        bounds = np.asarray(
            [
                constraint.bound
                + constraint.decay_coefficient * (barrier_omega.get(id(constraint), 1.0) - 1.0)
                for constraint in all_constraints
            ],
            dtype=float,
        )
        if np.any(normals @ control > bounds + max(1e-6, tolerance * 100.0)):
            return np.zeros(2), _invalid_diagnostics("solver_constraint_violation")
    else:
        candidates: list[Array] = [nominal.copy()]
        for normal, bound in zip(normals, bounds, strict=True):
            squared = float(normal @ normal)
            if squared > 1e-14:
                candidates.append(nominal + ((bound - normal @ nominal) / squared) * normal)
        for index in range(len(all_constraints)):
            a = normals[index]
            for other in range(index + 1, len(all_constraints)):
                b = normals[other]
                determinant = float(a[0] * b[1] - a[1] * b[0])
                if abs(determinant) <= 1e-12:
                    continue
                candidates.append(
                    np.array(
                        [
                            (bounds[index] * b[1] - a[1] * bounds[other]) / determinant,
                            (a[0] * bounds[other] - bounds[index] * b[0]) / determinant,
                        ]
                    )
                )
        feasible = [
            candidate
            for candidate in candidates
            if np.all(np.isfinite(candidate)) and np.all(normals @ candidate <= bounds + tolerance)
        ]
        if not feasible:
            return np.zeros(2), _invalid_diagnostics("infeasible")
        control = min(feasible, key=lambda point: float(np.sum((point - nominal) ** 2))).copy()
    gaps = bounds - normals @ control
    active_indices = tuple(index for index, gap in enumerate(gaps) if abs(gap) <= max(1e-7, tolerance * 10.0))
    active_keys = tuple(all_constraints[index].key for index in active_indices)
    active_matrix = normals[list(active_indices)] if active_indices else np.empty((0, 2))
    if not active_indices:
        sigma_min = float("nan")
        condition = float("nan")
    elif len(active_indices) > 2:
        # A nonredundant active set in R^2 has at most two rows.  Reporting
        # zero/inf for larger sets is conservative and makes degeneracy clear.
        singular_values = np.linalg.svd(active_matrix, compute_uv=False)
        sigma_min = 0.0 if len(active_indices) > 2 else float(singular_values[-1])
        condition = float("inf")
    else:
        singular_values = np.linalg.svd(active_matrix, compute_uv=False)
        sigma_min = float(singular_values[-1])
        condition = float("inf") if sigma_min <= 1e-12 else float(singular_values[0] / sigma_min)
    input_gaps = [gaps[index] for index, c in enumerate(all_constraints) if c.kind == "input"]
    slack = float(min(input_gaps))
    omega_by_key = {
        constraint.key: float(omega_values[index]) for index, constraint in enumerate(barriers)
    }
    gamma_by_key = {key: nominal_gamma * omega for key, omega in omega_by_key.items()}
    decay_penalty = float(decay_weight * np.sum((omega_values - 1.0) ** 2)) if optimal_decay else 0.0
    omega_minimum = float(np.min(omega_values)) if len(omega_values) else float("nan")
    omega_maximum = float(np.max(omega_values)) if len(omega_values) else float("nan")
    changed_fraction = (
        float(np.mean(np.abs(omega_values - 1.0) > material_difference_tolerance))
        if len(omega_values)
        else 0.0
    )
    control_objective = float(0.5 * np.sum((control - nominal) ** 2))
    diagnostics = WangQPDiagnostics(
        feasible=True,
        status="optimal",
        active_constraint_keys=active_keys,
        active_normal_matrix=active_matrix,
        min_active_singular_value=sigma_min,
        active_condition_number=condition,
        control_norm=float(np.linalg.norm(control)),
        acceleration_limit_slack=slack,
        objective=control_objective + decay_penalty,
        optimized_decay_multipliers=omega_by_key,
        optimized_decay_coefficients=gamma_by_key,
        minimum_decay_multiplier=omega_minimum,
        maximum_decay_multiplier=omega_maximum,
        fraction_decay_multipliers_changed=changed_fraction,
        decay_penalty=decay_penalty,
    )
    return control, diagnostics


def _lookup(mapping: Mapping[Any, Any] | Any, key: Any, default: Any = None) -> Any:
    if isinstance(mapping, Mapping):
        if key in mapping:
            return mapping[key]
        value = getattr(key, "value", None)
        if value in mapping:
            return mapping[value]
        if default is not None:
            return default
        raise KeyError(key)
    if mapping is None:
        return default
    return mapping


class WangController:
    """Solve each robot's acceleration QP using the fixed pair constraints."""

    def __init__(
        self,
        pair_edges: Sequence[tuple[str, str]] | None = None,
        robot_platforms: Mapping[str, Any] | None = None,
        acceleration_limits: Mapping[Any, float] | float | None = None,
        gammas: Mapping[Any, float] | float = 1.0,
        margins: Mapping[Any, float] | float = 0.0,
        safety_radii: Mapping[Any, float] | float | None = None,
        learned_model: Callable[[WangState], Array] | Mapping[str, Array] | None = None,
        obstacles: Mapping[str, Sequence[WangObstacle]] | None = None,
        walls: Mapping[str, Sequence[WangWall]] | None = None,
        optimal_decay: bool = True,
        decay_weight: float = 10.0,
        decay_multiplier_min: float = 0.0,
        decay_multiplier_max: float = 10.0,
        reuse_optimal_decay_qps: bool = True,
    ) -> None:
        self.pair_edges = None if pair_edges is None else tuple(tuple(edge) for edge in pair_edges)
        if self.pair_edges is not None:
            if any(len(edge) != 2 or edge[0] == edge[1] for edge in self.pair_edges):
                raise ValueError("pair_edges must contain distinct robot pairs")
            if len(set(tuple(sorted(edge)) for edge in self.pair_edges)) != len(self.pair_edges):
                raise ValueError("pair_edges must be unique")
        self.acceleration_limits = acceleration_limits
        self.robot_platforms = dict(robot_platforms or {})
        self.gammas = gammas
        self.margins = margins
        self.safety_radii = safety_radii
        self.learned_model = learned_model
        self.obstacles = {key: tuple(value) for key, value in (obstacles or {}).items()}
        self.walls = {key: tuple(value) for key, value in (walls or {}).items()}
        if not isinstance(optimal_decay, bool):
            raise ValueError("optimal_decay must be boolean")
        self.optimal_decay = optimal_decay
        self.decay_weight = _scalar(decay_weight, "decay_weight", positive=True)
        self.decay_multiplier_min = _scalar(
            decay_multiplier_min, "decay_multiplier_min", nonnegative=True
        )
        self.decay_multiplier_max = _scalar(
            decay_multiplier_max, "decay_multiplier_max", nonnegative=True
        )
        if self.decay_multiplier_max < self.decay_multiplier_min:
            raise ValueError("decay multiplier bounds must be ordered")
        if not isinstance(reuse_optimal_decay_qps, bool):
            raise ValueError("reuse_optimal_decay_qps must be boolean")
        self.reuse_optimal_decay_qps = reuse_optimal_decay_qps
        self._optimal_decay_qps: dict[str, PersistentWangOptimalDecayQP] = {}
        if self.optimal_decay and self.reuse_optimal_decay_qps and self.robot_platforms:
            edges = self.pair_edges
            if edges is None:
                ids = tuple(self.robot_platforms)
                edges = tuple(
                    (ids[left], ids[right])
                    for left in range(len(ids))
                    for right in range(left + 1, len(ids))
                )
            for robot_id, platform in self.robot_platforms.items():
                selected_alpha = _lookup(self.acceleration_limits, platform, None)
                if selected_alpha is None:
                    continue
                structure = [
                    (f"pair:{edge[0]}:{edge[1]}", "pair")
                    for edge in edges
                    if robot_id in edge
                ]
                structure.extend(
                    (
                        f"obstacle:{obstacle.obstacle_id or index}",
                        "obstacle",
                    )
                    for index, obstacle in enumerate(self.obstacles.get(robot_id, ()))
                )
                structure.extend(
                    (f"wall:{wall.wall_id or index}", "wall")
                    for index, wall in enumerate(self.walls.get(robot_id, ()))
                )
                if structure:
                    self._optimal_decay_qps[robot_id] = PersistentWangOptimalDecayQP(
                        structure,
                        float(selected_alpha),
                        decay_weight=self.decay_weight,
                        decay_multiplier_min=self.decay_multiplier_min,
                        decay_multiplier_max=self.decay_multiplier_max,
                    )

    def _alpha(self, state: WangState) -> float:
        selected = _lookup(self.acceleration_limits, state.platform, None)
        if selected is None:
            selected = state.acceleration_limit
        return _scalar(selected, f"alpha[{state.robot_id}]", positive=True)

    def _gamma(self, state: WangState) -> float:
        return _scalar(_lookup(self.gammas, state.platform, 1.0), f"gamma[{state.robot_id}]", nonnegative=True)

    def _margin(self, state: WangState) -> float:
        selected = _lookup(self.margins, state.platform, None)
        if selected is None:
            selected = state.margin
        return _scalar(selected, f"margin[{state.robot_id}]", nonnegative=True)

    def _radius(self, state: WangState) -> float:
        selected = _lookup(self.safety_radii, state.platform, None)
        if selected is None:
            selected = state.radius
        return _scalar(selected, f"radius[{state.robot_id}]", nonnegative=True)

    def _learned(self, state: WangState) -> Array:
        if self.learned_model is None:
            return np.zeros(2)
        if isinstance(self.learned_model, Mapping):
            if state.robot_id in self.learned_model:
                value = self.learned_model[state.robot_id]
            else:
                value = self.learned_model[state.platform]
            return _vector(value, f"d_hat[{state.robot_id}]")
        try:
            value = self.learned_model(state)
        except TypeError:
            # Existing bundles of learned models in this repository use the
            # ``(platform, position)`` callback shape.  Passing only WangState
            # remains the preferred interface because it cannot expose truth
            # or process noise fields accidentally.
            value = self.learned_model(state.platform, state.position)  # type: ignore[misc,call-arg]
        return _vector(value, f"d_hat[{state.robot_id}]")

    def _edges_for(self, states: Mapping[str, WangState]) -> tuple[tuple[str, str], ...]:
        if self.pair_edges is None:
            ids = tuple(states)
            return tuple((ids[i], ids[j]) for i in range(len(ids)) for j in range(i + 1, len(ids)))
        return self.pair_edges

    def local_constraints(
        self,
        robot_id: str,
        states: Mapping[str, WangState],
        *,
        pair_edges: Sequence[tuple[str, str]] | None = None,
        obstacles: Sequence[WangObstacle] | None = None,
        walls: Sequence[WangWall] | None = None,
    ) -> list[WangConstraint]:
        """Build constraints for the fixed neighbors of ``robot_id``."""

        if robot_id not in states:
            raise KeyError(robot_id)
        state_i = states[robot_id]
        alpha_i = self._alpha(state_i)
        gamma_i = self._gamma(state_i)
        d_hat_i = self._learned(state_i)
        edges = self._edges_for(states) if pair_edges is None else tuple(pair_edges)
        constraints: list[WangConstraint] = []
        for edge in edges:
            if robot_id not in edge:
                continue
            other_id = edge[1] if edge[0] == robot_id else edge[0]
            if other_id not in states:
                raise ValueError(f"missing fixed-graph neighbor {other_id!r}")
            state_j = states[other_id]
            alpha_j = self._alpha(state_j)
            gamma = gamma_i  # source Strategy B uses the local design gamma
            D_ij = self._radius(state_i) + self._radius(state_j)
            d_hat_j = self._learned(state_j)
            rhs = pair_b_hat(
                state_i.position,
                state_j.position,
                state_i.velocity,
                state_j.velocity,
                alpha_i,
                alpha_j,
                D_ij,
                gamma,
                d_hat_i,
                d_hat_j,
            )
            rhs -= pair_robust_tightening(
                state_i.position,
                state_j.position,
                self._margin(state_i),
                self._margin(state_j),
            )
            # The local robot always receives its own Strategy B share.  The
            # q orientation below changes with the role, but the share does
            # not: eta_local = alpha(state_i)/(alpha_i+alpha_j).
            eta_local = alpha_i / (alpha_i + alpha_j)
            q = state_i.position - state_j.position
            h = h_ij(
                state_i.position,
                state_j.position,
                state_i.velocity,
                state_j.velocity,
                alpha_i,
                alpha_j,
                D_ij,
            )
            # Orient the relative position from the current robot to its neighbor before allocating the pair constraint.
            side = -q
            constraints.append(
                WangConstraint(
                    f"pair:{edge[0]}:{edge[1]}",
                    side,
                    eta_local * rhs,
                    "pair",
                    eta_local * gamma * h * float(np.linalg.norm(q)),
                )
            )
        selected_obstacles = self.obstacles.get(robot_id, ()) if obstacles is None else tuple(obstacles)
        for index, obstacle in enumerate(selected_obstacles):
            if not isinstance(obstacle, WangObstacle):
                obstacle = WangObstacle(obstacle.center, obstacle.radius, getattr(obstacle, "obstacle_id", None))
            constraints.append(
                strategy_b_obstacle_constraint(
                    state_i.position,
                    state_i.velocity,
                    alpha_i,
                    self._radius(state_i),
                    obstacle,
                    gamma_i,
                    d_hat_i,
                    self._margin(state_i),
                    key=f"obstacle:{obstacle.obstacle_id or index}",
                )
            )
        selected_walls = self.walls.get(robot_id, ()) if walls is None else tuple(walls)
        for index, wall in enumerate(selected_walls):
            if not isinstance(wall, WangWall):
                wall = WangWall(wall.normal, wall.offset, getattr(wall, "wall_id", None))
            constraints.append(
                wall_constraint(
                    state_i.position,
                    state_i.velocity,
                    alpha_i,
                    self._radius(state_i),
                    wall,
                    gamma_i,
                    d_hat_i,
                    self._margin(state_i),
                    key=f"wall:{wall.wall_id or index}",
                )
            )
        return constraints

    def step(
        self,
        states: Mapping[str, WangState],
        nominal_controls: Mapping[str, Array],
        *,
        obstacles: Mapping[str, Sequence[WangObstacle]] | None = None,
        walls: Mapping[str, Sequence[WangWall]] | None = None,
    ) -> dict[str, WangControlStep]:
        """Solve independent local QPs for all robots, failing closed per robot."""

        if set(states) != set(nominal_controls):
            raise ValueError("states and nominal_controls must have identical robot IDs")
        if obstacles is not None:
            selected_obstacles = {key: tuple(value) for key, value in obstacles.items()}
        else:
            selected_obstacles = self.obstacles
        if walls is not None:
            selected_walls = {key: tuple(value) for key, value in walls.items()}
        else:
            selected_walls = self.walls
        result: dict[str, WangControlStep] = {}
        for robot_id, state in states.items():
            try:
                constraints = self.local_constraints(
                    robot_id,
                    states,
                    obstacles=selected_obstacles.get(robot_id, ()),
                    walls=selected_walls.get(robot_id, ()),
                )
                persistent_qp = self._optimal_decay_qps.get(robot_id)
                if (
                    self.optimal_decay
                    and self.reuse_optimal_decay_qps
                    and persistent_qp is None
                    and any(constraint.kind != "input" for constraint in constraints)
                ):
                    persistent_qp = PersistentWangOptimalDecayQP(
                        tuple((constraint.key, constraint.kind) for constraint in constraints),
                        self._alpha(state),
                        decay_weight=self.decay_weight,
                        decay_multiplier_min=self.decay_multiplier_min,
                        decay_multiplier_max=self.decay_multiplier_max,
                    )
                    self._optimal_decay_qps[robot_id] = persistent_qp
                control, diagnostics = solve_wang_qp(
                    nominal_controls[robot_id],
                    constraints,
                    self._alpha(state),
                    optimal_decay=self.optimal_decay,
                    nominal_gamma=self._gamma(state),
                    decay_weight=self.decay_weight,
                    decay_multiplier_min=self.decay_multiplier_min,
                    decay_multiplier_max=self.decay_multiplier_max,
                    persistent_optimal_decay_qp=persistent_qp,
                )
            except (KeyError, TypeError, ValueError, FloatingPointError) as exc:
                control = np.zeros(2)
                constraints = ()
                diagnostics = _invalid_diagnostics(f"invalid_state:{exc}")
            result[robot_id] = WangControlStep(
                control=control,
                qp_diagnostics=diagnostics,
                constraints=tuple(constraints),
                failed=not diagnostics.feasible,
            )
        return result


__all__ = [
    "WangConstraint",
    "WangControlStep",
    "WangController",
    "WangNeighborPacket",
    "WangObstacle",
    "PersistentWangOptimalDecayQP",
    "WangParameters",
    "WangQPDiagnostics",
    "WangState",
    "WangWall",
    "b_ij",
    "build_optimal_decay_qp",
    "effective_pair_decay",
    "eta_ij",
    "eta_ji",
    "h_ij",
    "obstacle_B_true",
    "obstacle_b",
    "obstacle_b_hat",
    "obstacle_b_true",
    "obstacle_constraint_jacobians",
    "obstacle_h",
    "obstacle_rhs_jacobian",
    "pair_B_hat",
    "pair_B_true",
    "pair_barrier_from_states",
    "pair_barrier_jacobian",
    "pair_b_hat",
    "pair_b_hat_jacobian",
    "pair_b_true",
    "pair_constraint_jacobians",
    "pair_constraint_residual",
    "pair_eta",
    "pair_robust_tightening",
    "pair_rhs_from_states",
    "pair_rhs_jacobian",
    "obstacle_robust_tightening",
    "solve_wang_qp",
    "strategy_b_obstacle_constraint",
    "strategy_b_pair_constraints",
    "wall_B_hat",
    "wall_B_true",
    "wall_clearance",
    "wall_constraint",
    "wall_h",
]
