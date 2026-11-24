"""Each robot solves a quadratic program with high order control barrier function constraints."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np

from experiments.formation.qp import (
    WangConstraint,
    WangControlStep,
    WangObstacle,
    WangQPDiagnostics,
    WangState,
    PersistentWangOptimalDecayQP,
    solve_wang_qp,
)


Array = np.ndarray


def _vector(value: Any, name: str) -> Array:
    result = np.asarray(value, dtype=float)
    if result.shape != (2,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite 2D vector")
    return result.copy()


def _scalar(
    value: Any,
    name: str,
    *,
    nonnegative: bool = False,
    positive: bool = False,
) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite scalar") from exc
    if not isfinite(result) or (positive and result <= 0.0) or (
        nonnegative and result < 0.0
    ):
        qualifier = "positive" if positive else "nonnegative" if nonnegative else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def robust_hocbf_tightening(
    q_or_position: Any = None,
    margin_or_center: Any = None,
    margin: Any | None = None,
    *,
    q: Any | None = None,
    position: Any | None = None,
    obstacle_center: Any | None = None,
) -> float:
    """Return the robust HOCBF tightening ``2*||q||*margin``.

    The canonical call is ``robust_hocbf_tightening(q, margin)``.  For consistency
    with the obstacle helpers elsewhere in the repository, the positional form
    ``robust_hocbf_tightening(position, obstacle_center, margin)`` and the
    keyword form
    ``robust_hocbf_tightening(position=..., obstacle_center=..., margin=...)``
    are accepted as well.
    """

    if q is not None:
        if q_or_position is not None:
            raise TypeError("specify q either positionally or by keyword")
        q_value = _vector(q, "q")
        if margin is None:
            margin = margin_or_center
    elif position is not None or obstacle_center is not None:
        if q_or_position is not None or margin_or_center is not None:
            raise TypeError("do not mix positional and keyword geometry")
        if position is None or obstacle_center is None:
            raise TypeError("position and obstacle_center must be supplied together")
        q_value = _vector(position, "position") - _vector(
            obstacle_center, "obstacle_center"
        )
    elif q_or_position is not None and margin_or_center is None:
        # A vector supplied positionally with the margin named.
        q_value = _vector(q_or_position, "q")
    elif margin is None:
        # Two positional arguments: (q, margin).
        q_value = _vector(q_or_position, "q")
        margin = margin_or_center
    else:
        # Three positional arguments: (position, obstacle_center, margin).
        q_value = _vector(q_or_position, "position") - _vector(
            margin_or_center, "obstacle_center"
        )

    if margin is None:
        raise TypeError("margin is required")
    margin_value = _scalar(margin, "margin", nonnegative=True)
    return float(2.0 * np.linalg.norm(q_value) * margin_value)


def robust_hocbf_slack(
    position: Any,
    obstacle_center: Any,
    velocity: Any,
    control: Any,
    robot_radius: float,
    obstacle_radius: float,
    lambda1: float,
    lambda2: float,
    d_hat: Any,
    margin: float,
    omega: float = 1.0,
) -> float:
    """Evaluate the signed residual of the robust HOCBF constraint.

    The returned value is the signed residual of the documented ``>= 0``
    inequality, including the control term, learned disturbance, optimized
    decay multiplier, and robust tightening.  In the Wang affine convention it
    is exactly ``terms.rhs - terms.normal @ control``.
    """

    terms = hocbf_terms(
        position,
        obstacle_center,
        velocity,
        robot_radius,
        obstacle_radius,
        lambda1,
        lambda2,
        d_hat,
        margin,
        omega,
    )
    return float(terms.rhs - terms.normal @ _vector(control, "control"))


@dataclass(frozen=True)
class HOCBFTerms:
    """Terms of the robust HOCBF inequality for one stationary obstacle.

    ``normal`` and ``rhs`` use the existing Wang convention
    ``normal @ u <= rhs``.  Thus ``normal`` is ``-2*q`` and ``rhs`` is the
    part that depends on the state of the documented ``>= 0`` expression at ``omega``.
    ``robust_slack`` stores the positive robust tightening.  The object supports
    both attribute access and ``terms["h"]`` style access for lightweight audit
    code.  Iteration yields ``(h, psi, normal, rhs)``.
    """

    q: Array
    D: float
    h: float
    psi: float
    normal: Array
    rhs: float
    robust_slack: float
    d_hat: Array
    base_rhs: float
    decay_coefficient: float
    omega: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "q", _vector(self.q, "q"))
        object.__setattr__(self, "D", _scalar(self.D, "D", nonnegative=True))
        object.__setattr__(self, "h", _scalar(self.h, "h"))
        object.__setattr__(self, "psi", _scalar(self.psi, "psi"))
        object.__setattr__(self, "normal", _vector(self.normal, "normal"))
        object.__setattr__(self, "rhs", _scalar(self.rhs, "rhs"))
        object.__setattr__(
            self,
            "robust_slack",
            _scalar(self.robust_slack, "robust_slack", nonnegative=True),
        )
        object.__setattr__(self, "d_hat", _vector(self.d_hat, "d_hat"))
        object.__setattr__(self, "base_rhs", _scalar(self.base_rhs, "base_rhs"))
        object.__setattr__(
            self,
            "decay_coefficient",
            _scalar(self.decay_coefficient, "decay_coefficient"),
        )
        object.__setattr__(self, "omega", _scalar(self.omega, "omega", nonnegative=True))

    @property
    def bound(self) -> float:
        """Return the bound used in the affine inequality."""

        return self.rhs

    @property
    def affine_constant(self) -> float:
        """Alias for ``rhs`` in ``normal @ u <= rhs`` form."""

        return self.rhs

    def __getitem__(self, key: str) -> Any:
        if key in {"slack", "margin_slack"}:
            return self.robust_slack
        return getattr(self, key)

    def __iter__(self):
        yield self.h
        yield self.psi
        yield self.normal.copy()
        yield self.rhs


def hocbf_terms(
    position: Any = None,
    obstacle_center: Any = None,
    velocity: Any = None,
    robot_radius: Any = 0.0,
    obstacle_radius: Any = 0.0,
    lambda1: Any = 1.0,
    lambda2: Any = 1.0,
    d_hat: Any | None = None,
    margin: Any = 0.0,
    omega: Any = 1.0,
    **aliases: Any,
) -> HOCBFTerms:
    """Evaluate the robust HOCBF constraint for one robot.

    Parameters are ``position``, ``obstacle_center``, ``velocity``, the two
    radii, ``lambda1``, ``lambda2``, the learned total disturbance ``d_hat``,
    and its norm margin ``margin``.  ``omega`` is the decay for this obstacle
    multiplier. It defaults to one, and is constrained by the controller to
    ``omega_min <= omega <= omega_max``.  The returned object gives ``h``,
    ``psi``, and the affine ``normal``/``rhs`` pair.  The robust barrier constraint is
    ``normal @ u <= rhs``.

    Short aliases ``p``, ``p_o``/``po``, ``v``, ``dhat``, ``lambda_1``,
    ``lambda_2``, ``robust_margin``, and ``D`` are accepted to keep direct
    equation checks concise.  Supplying ``D`` means ``robot_radius=D`` and
    ``obstacle_radius=0`` unless radii were also supplied explicitly.
    """

    if position is None:
        position = aliases.pop("p", None)
    if obstacle_center is None:
        obstacle_center = aliases.pop("p_o", aliases.pop("po", aliases.pop("center", None)))
    if velocity is None:
        velocity = aliases.pop("v", None)
    if d_hat is None:
        d_hat = aliases.pop("dhat", aliases.pop("disturbance", None))
    if "lambda_1" in aliases:
        lambda1 = aliases.pop("lambda_1")
    if "lambda_2" in aliases:
        lambda2 = aliases.pop("lambda_2")
    if "robust_margin" in aliases:
        margin = aliases.pop("robust_margin")
    if "decay_multiplier" in aliases:
        omega = aliases.pop("decay_multiplier")
    if "omega" in aliases:
        omega = aliases.pop("omega")
    if "D" in aliases:
        if robot_radius != 0.0 or obstacle_radius != 0.0:
            raise TypeError("D cannot be combined with explicit nonzero radii")
        robot_radius = aliases.pop("D")
    if aliases:
        unknown = ", ".join(sorted(aliases))
        raise TypeError(f"unexpected hocbf_terms arguments: {unknown}")
    if position is None or obstacle_center is None or velocity is None:
        raise TypeError("position, obstacle_center, and velocity are required")

    p = _vector(position, "position")
    p_o = _vector(obstacle_center, "obstacle_center")
    v = _vector(velocity, "velocity")
    robot_r = _scalar(robot_radius, "robot_radius", nonnegative=True)
    obstacle_r = _scalar(obstacle_radius, "obstacle_radius", nonnegative=True)
    gain1 = _scalar(lambda1, "lambda1", nonnegative=True)
    gain2 = _scalar(lambda2, "lambda2", nonnegative=True)
    learned = np.zeros(2) if d_hat is None else _vector(d_hat, "d_hat")
    margin_value = _scalar(margin, "margin", nonnegative=True)
    omega_value = _scalar(omega, "omega", nonnegative=True)

    q_value = p - p_o
    clearance = robot_r + obstacle_r
    h_value = float(q_value @ q_value - clearance * clearance)
    psi_value = float(2.0 * q_value @ v + gain1 * h_value)
    robust_slack = robust_hocbf_tightening(q_value, margin_value)
    # Move every term independent of u to the right side of the Wang
    # <= representation: (-2q)@u <= rhs.
    base_rhs = float(
        2.0 * q_value @ learned
        + 2.0 * (v @ v)
        + 2.0 * gain1 * (q_value @ v)
        - robust_slack
    )
    decay_coefficient = float(gain2 * psi_value)
    rhs = float(base_rhs + omega_value * decay_coefficient)
    return HOCBFTerms(
        q=q_value,
        D=clearance,
        h=h_value,
        psi=psi_value,
        normal=-2.0 * q_value,
        rhs=rhs,
        robust_slack=robust_slack,
        d_hat=learned,
        base_rhs=base_rhs,
        decay_coefficient=decay_coefficient,
        omega=omega_value,
    )


def hocbf_constraint(
    position: Any,
    obstacle_center: Any,
    velocity: Any,
    robot_radius: float,
    obstacle_radius: float,
    lambda1: float,
    lambda2: float,
    d_hat: Any = None,
    margin: float = 0.0,
    omega: float = 1.0,
    *,
    key: str = "obstacle",
) -> WangConstraint:
    """Build a :class:`WangConstraint` from :func:`hocbf_terms`."""

    terms = hocbf_terms(
        position,
        obstacle_center,
        velocity,
        robot_radius,
        obstacle_radius,
        lambda1,
        lambda2,
        d_hat,
        margin,
        1.0,
    )
    return WangConstraint(
        key,
        terms.normal,
        float(terms.base_rhs + float(omega) * terms.decay_coefficient),
        "obstacle",
        terms.decay_coefficient,
    )


def _failed_diagnostics(status: str) -> WangQPDiagnostics:
    """Construct a truthful failed diagnostic with compatible fields."""

    return WangQPDiagnostics(
        feasible=False,
        status=str(status),
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


def _as_obstacle(value: Any) -> WangObstacle:
    if isinstance(value, WangObstacle):
        return value
    try:
        return WangObstacle(
            value.center,
            value.radius,
            getattr(value, "obstacle_id", None),
        )
    except AttributeError as exc:
        raise ValueError("obstacles must contain WangObstacle-like values") from exc


def _normalize_obstacles(
    obstacles: Mapping[Any, WangObstacle | Sequence[WangObstacle]] | None,
) -> dict[Any, tuple[WangObstacle, ...]]:
    if obstacles is None:
        return {}
    if not isinstance(obstacles, Mapping):
        raise ValueError("obstacles must be a mapping from robot IDs to obstacles")
    normalized: dict[Any, tuple[WangObstacle, ...]] = {}
    for robot_key, values in obstacles.items():
        if isinstance(values, WangObstacle) or (
            hasattr(values, "center") and hasattr(values, "radius")
        ):
            values = (values,)  # type: ignore[assignment]
        try:
            normalized[robot_key] = tuple(_as_obstacle(value) for value in values)
        except TypeError as exc:
            raise ValueError(
                f"obstacles[{robot_key!r}] must be an obstacle or sequence"
            ) from exc
    return normalized


def _mapping_lookup(mapping: Mapping[Any, Any], state: WangState) -> Any:
    """Look up robot ID, platform object, or platform value in that order."""

    for key in (state.robot_id, state.platform, getattr(state.platform, "value", None)):
        if key is not None:
            try:
                if key in mapping:
                    return mapping[key]
            except TypeError:
                continue
    raise KeyError(state.robot_id)


class RobotLocalHOCBFController:
    """Solve the HOCBF quadratic program for each robot using its obstacles, acceleration limits, and class margin."""

    def __init__(
        self,
        obstacles: Mapping[Any, WangObstacle | Sequence[WangObstacle]] | None = None,
        acceleration_limits: Mapping[Any, float] | float | None = None,
        margins: Mapping[Any, float] | float | None = None,
        lambda1: Mapping[Any, float] | float = 1.0,
        lambda2: Mapping[Any, float] | float = 1.0,
        learned_model: Mapping[Any, Any] | Any | None = None,
        *,
        decay_weight: float = 10.0,
        omega_min: float = 0.0,
        omega_max: float = 10.0,
        optimal_decay: bool = True,
        reuse_optimal_decay_qps: bool = True,
        decay_multiplier_min: float | None = None,
        decay_multiplier_max: float | None = None,
    ) -> None:
        if decay_multiplier_min is not None:
            omega_min = decay_multiplier_min
        if decay_multiplier_max is not None:
            omega_max = decay_multiplier_max
        self.obstacles = _normalize_obstacles(obstacles)
        self.acceleration_limits = acceleration_limits
        self.margins = margins
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.learned_model = learned_model
        self.decay_weight = _scalar(decay_weight, "decay_weight", positive=True)
        self.omega_min = _scalar(omega_min, "omega_min", nonnegative=True)
        self.omega_max = _scalar(omega_max, "omega_max", nonnegative=True)
        if self.omega_max < self.omega_min:
            raise ValueError("omega bounds must be ordered")
        if not isinstance(optimal_decay, bool):
            raise ValueError("optimal_decay must be boolean")
        if not isinstance(reuse_optimal_decay_qps, bool):
            raise ValueError("reuse_optimal_decay_qps must be boolean")
        self.optimal_decay = optimal_decay
        self.reuse_optimal_decay_qps = reuse_optimal_decay_qps
        self._optimal_decay_qps: dict[str, PersistentWangOptimalDecayQP] = {}

    def _setting(
        self,
        setting: Mapping[Any, Any] | Any | None,
        state: WangState,
        *,
        default: float,
        name: str,
        nonnegative: bool = True,
        positive: bool = False,
    ) -> float:
        if setting is None:
            value = default
        elif isinstance(setting, Mapping):
            try:
                value = _mapping_lookup(setting, state)
            except KeyError:
                value = default
        else:
            value = setting
        return _scalar(value, f"{name}[{state.robot_id}]", nonnegative=nonnegative, positive=positive)

    def _alpha(self, state: WangState) -> float:
        return self._setting(
            self.acceleration_limits,
            state,
            default=state.acceleration_limit,
            name="acceleration_limit",
            nonnegative=False,
            positive=True,
        )

    def _margin(self, state: WangState) -> float:
        return self._setting(
            self.margins,
            state,
            default=state.margin,
            name="margin",
        )

    def _lambda1(self, state: WangState) -> float:
        return self._setting(self.lambda1, state, default=1.0, name="lambda1")

    def _lambda2(self, state: WangState) -> float:
        return self._setting(self.lambda2, state, default=1.0, name="lambda2")

    def _learned(self, state: WangState) -> Array:
        if self.learned_model is None:
            return np.zeros(2)
        if isinstance(self.learned_model, Mapping):
            try:
                value = _mapping_lookup(self.learned_model, state)
            except KeyError as exc:
                raise KeyError(f"missing learned disturbance for {state.robot_id!r}") from exc
            return _vector(value, f"d_hat[{state.robot_id}]")
        try:
            value = self.learned_model(state)
        except TypeError:
            value = self.learned_model(state.platform, state.position)
        return _vector(value, f"d_hat[{state.robot_id}]")

    def _obstacles_for(
        self,
        state: WangState,
        selected: Mapping[Any, WangObstacle | Sequence[WangObstacle]] | None,
    ) -> tuple[WangObstacle, ...]:
        source = self.obstacles if selected is None else _normalize_obstacles(selected)
        if state.robot_id in source:
            return source[state.robot_id]
        for key in (state.platform, getattr(state.platform, "value", None)):
            if key in source:
                return source[key]
        return ()

    def _persistent_qp(
        self,
        robot_id: str,
        state: WangState,
        constraints: Sequence[WangConstraint],
    ) -> PersistentWangOptimalDecayQP | None:
        if not self.optimal_decay or not self.reuse_optimal_decay_qps:
            return None
        barrier_constraints = tuple(
            constraint for constraint in constraints if constraint.kind != "input"
        )
        if not barrier_constraints:
            return None
        structure = tuple(
            (constraint.key, constraint.kind) for constraint in barrier_constraints
        )
        alpha = self._alpha(state)
        cached = self._optimal_decay_qps.get(robot_id)
        if (
            cached is None
            or cached.constraint_structure != structure
            or cached.acceleration_limit != alpha
            or cached.decay_weight != self.decay_weight
            or cached.decay_multiplier_min != self.omega_min
            or cached.decay_multiplier_max != self.omega_max
        ):
            cached = PersistentWangOptimalDecayQP(
                structure,
                alpha,
                decay_weight=self.decay_weight,
                decay_multiplier_min=self.omega_min,
                decay_multiplier_max=self.omega_max,
            )
            self._optimal_decay_qps[robot_id] = cached
        return cached

    def local_constraints(
        self,
        robot_id: str,
        states: Mapping[str, WangState],
        *,
        obstacles: Mapping[Any, WangObstacle | Sequence[WangObstacle]] | None = None,
    ) -> list[WangConstraint]:
        """Build the constraints for stationary obstacles seen by this robot."""

        if robot_id not in states:
            raise KeyError(robot_id)
        state = states[robot_id]
        alpha1 = self._lambda1(state)
        alpha2 = self._lambda2(state)
        margin = self._margin(state)
        learned = self._learned(state)
        constraints: list[WangConstraint] = []
        for index, obstacle in enumerate(self._obstacles_for(state, obstacles)):
            obstacle_id = obstacle.obstacle_id if obstacle.obstacle_id is not None else index
            key = f"obstacle:{robot_id}:{obstacle_id}"
            constraints.append(
                hocbf_constraint(
                    state.position,
                    obstacle.center,
                    state.velocity,
                    state.radius,
                    obstacle.radius,
                    alpha1,
                    alpha2,
                    learned,
                    margin,
                    key=key,
                )
            )
        return constraints

    def step(
        self,
        states: Mapping[str, WangState],
        nominal_controls: Mapping[str, Array],
        *,
        obstacles: Mapping[Any, WangObstacle | Sequence[WangObstacle]] | None = None,
    ) -> dict[str, WangControlStep]:
        """Solve one independent hard QP per robot, failing closed."""

        if set(states) != set(nominal_controls):
            raise ValueError("states and nominal_controls must have identical robot IDs")
        result: dict[str, WangControlStep] = {}
        for robot_id, state in states.items():
            constraints: tuple[WangConstraint, ...] = ()
            try:
                constraints = tuple(
                    self.local_constraints(robot_id, states, obstacles=obstacles)
                )
                persistent_qp = self._persistent_qp(robot_id, state, constraints)
                control, diagnostics = solve_wang_qp(
                    nominal_controls[robot_id],
                    constraints,
                    self._alpha(state),
                    optimal_decay=self.optimal_decay,
                    nominal_gamma=self._lambda2(state),
                    decay_weight=self.decay_weight,
                    decay_multiplier_min=self.omega_min,
                    decay_multiplier_max=self.omega_max,
                    persistent_optimal_decay_qp=persistent_qp,
                )
            except (KeyError, TypeError, ValueError, FloatingPointError) as exc:
                control = np.zeros(2)
                diagnostics = _failed_diagnostics(f"invalid_state:{exc}")
            # Apply zero acceleration when the QP is infeasible.
            if not diagnostics.feasible:
                control = np.zeros(2)
            result[robot_id] = WangControlStep(
                control=control,
                qp_diagnostics=diagnostics,
                constraints=constraints,
                failed=not diagnostics.feasible,
            )
        return result


__all__ = [
    "HOCBFTerms",
    "RobotLocalHOCBFController",
    "hocbf_constraint",
    "hocbf_terms",
    "robust_hocbf_tightening",
    "robust_hocbf_slack",
]
