"""Formation dynamics with a double integrator and barrier residuals."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from maicp.configuration import Platform

Array = np.ndarray

def _as_vector(value: Sequence[float] | Array, *, name: str) -> Array:
    result = np.asarray(value, dtype=float)
    if result.shape != (2,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite planar vector")
    return result.copy()


def wang_barrier(
    p_i: Sequence[float] | Array,
    p_j: Sequence[float] | Array,
    v_i: Sequence[float] | Array,
    v_j: Sequence[float] | Array,
    alpha_i: float,
    alpha_j: float,
    clearance: float,
) -> float:
    """Evaluate Wang et al. Eq. (8) for one pair.

    ``h = sqrt(2(alpha_i+alpha_j)(||Delta p||-D)) +
    Delta p.T Delta v / ||Delta p||``.  The source domain requires a strict
    positive distance gap. No numerical regularization is applied here.
    """

    q = _as_vector(p_i, name="p_i") - _as_vector(p_j, name="p_j")
    dv = _as_vector(v_i, name="v_i") - _as_vector(v_j, name="v_j")
    ai = float(alpha_i)
    aj = float(alpha_j)
    distance = float(np.linalg.norm(q))
    gap = distance - float(clearance)
    total = ai + aj
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Wang acceleration limits must have a positive sum")
    if not np.isfinite(clearance) or clearance < 0.0:
        raise ValueError("Wang clearance must be finite and nonnegative")
    if distance <= 0.0 or gap <= 0.0:
        raise ValueError("Wang barrier requires a strictly positive distance gap")
    return float(np.sqrt(2.0 * total * gap) + q @ dv / distance)


def wang_barrier_rhs(
    p_i: Sequence[float] | Array,
    p_j: Sequence[float] | Array,
    v_i: Sequence[float] | Array,
    v_j: Sequence[float] | Array,
    alpha_i: float,
    alpha_j: float,
    clearance: float,
    gamma: float,
) -> float:
    """Evaluate the bound ``b_ij`` from the state in Wang Eq. (11).

    With ``q = Delta p``, ``dv = Delta v``, ``r = ||q||`` and
    ``A = alpha_i+alpha_j``, Eq. (11) is

    ``b = gamma*h*r - (q.T dv)**2/r**2 + ||dv||**2 +
    A*q.T dv/sqrt(2*A*(r-D))``.

    This is the equivalent form of ``-q.T(u_i-u_j) <= b`` obtained from
    ``dot(h)+gamma*h >= 0``. In particular the final term is not omitted.
    """

    q = _as_vector(p_i, name="p_i") - _as_vector(p_j, name="p_j")
    dv = _as_vector(v_i, name="v_i") - _as_vector(v_j, name="v_j")
    ai = float(alpha_i)
    aj = float(alpha_j)
    total = ai + aj
    distance = float(np.linalg.norm(q))
    gap = distance - float(clearance)
    if not np.isfinite(gamma) or gamma < 0.0:
        raise ValueError("Wang gamma must be finite and nonnegative")
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Wang acceleration limits must have a positive sum")
    if distance <= 0.0 or gap <= 0.0:
        raise ValueError("Wang barrier requires a strictly positive distance gap")
    h = float(np.sqrt(2.0 * total * gap) + q @ dv / distance)
    projection = float(q @ dv)
    return float(
        gamma * h * distance
        - projection**2 / distance**2
        + dv @ dv
        + total * projection / np.sqrt(2.0 * total * gap)
    )


wang_h = wang_barrier
wang_b = wang_barrier_rhs


def wang_allocation_weights(alpha_i: float, alpha_j: float) -> tuple[float, float]:
    """Return Strategy B weights ``eta_ij, eta_ji`` from Wang Eq. (15)."""

    ai = float(alpha_i)
    aj = float(alpha_j)
    if not np.isfinite(ai) or not np.isfinite(aj) or ai <= 0.0 or aj <= 0.0:
        raise ValueError("Strategy-B pair weights require positive robot limits")
    total = ai + aj
    return ai / total, aj / total


def wang_acceleration_model_error(
    true_field: Sequence[float] | Array,
    motion_noise: Sequence[float] | Array,
    learned_field: Sequence[float] | Array,
) -> float:
    """Return the score ``||d_i+w_i-d_hat_i||_2`` for one robot."""

    error = (
        _as_vector(true_field, name="true_field")
        + _as_vector(motion_noise, name="motion_noise")
        - _as_vector(learned_field, name="learned_field")
    )
    return float(np.linalg.norm(error))


def wang_pair_residuals(
    p_i: Sequence[float] | Array,
    p_j: Sequence[float] | Array,
    v_i: Sequence[float] | Array,
    v_j: Sequence[float] | Array,
    u_i: Sequence[float] | Array,
    u_j: Sequence[float] | Array,
    learned_i: Sequence[float] | Array,
    learned_j: Sequence[float] | Array,
    true_i: Sequence[float] | Array,
    true_j: Sequence[float] | Array,
    alpha_i: float,
    alpha_j: float,
    clearance: float,
    gamma: float,
    *,
    motion_noise_i: Sequence[float] | Array | None = None,
    motion_noise_j: Sequence[float] | Array | None = None,
) -> tuple[float, float]:
    """Evaluate the predicted and true barrier residuals under the same applied controls."""

    p_i_v = _as_vector(p_i, name="p_i")
    p_j_v = _as_vector(p_j, name="p_j")
    q = p_i_v - p_j_v
    u_i_v = _as_vector(u_i, name="u_i")
    u_j_v = _as_vector(u_j, name="u_j")
    learned_i_v = _as_vector(learned_i, name="learned_i")
    learned_j_v = _as_vector(learned_j, name="learned_j")
    true_i_v = _as_vector(true_i, name="true_i")
    true_j_v = _as_vector(true_j, name="true_j")
    noise_i = np.zeros(2) if motion_noise_i is None else _as_vector(motion_noise_i, name="motion_noise_i")
    noise_j = np.zeros(2) if motion_noise_j is None else _as_vector(motion_noise_j, name="motion_noise_j")
    b = wang_barrier_rhs(p_i_v, p_j_v, v_i, v_j, alpha_i, alpha_j, clearance, gamma)
    applied = float(q @ (u_i_v - u_j_v))
    b_hat = b + float(q @ (learned_i_v - learned_j_v))
    b_true = b + float(q @ ((true_i_v + noise_i) - (true_j_v + noise_j)))
    return float(b_hat + applied), float(b_true + applied)


def wang_obstacle_residuals(
    p: Sequence[float] | Array,
    v: Sequence[float] | Array,
    u: Sequence[float] | Array,
    obstacle_center: Sequence[float] | Array,
    learned: Sequence[float] | Array,
    true: Sequence[float] | Array,
    alpha: float,
    clearance: float,
    gamma: float,
    *,
    motion_noise: Sequence[float] | Array | None = None,
) -> tuple[float, float]:
    """Evaluate :func:`wang_pair_residuals` for a stationary obstacle.

    The obstacle is a fixed agent with ``v_j=u_j=0`` and ``alpha_j=0``. Therefore the robot carries the entire local constraint.  Its robust
    tightening is ``||p_i-p_o||_2 * r_C(i)``.
    """

    zero = np.zeros(2)
    return wang_pair_residuals(
        p,
        obstacle_center,
        v,
        zero,
        u,
        zero,
        learned,
        zero,
        true,
        zero,
        alpha,
        0.0,
        clearance,
        gamma,
        motion_noise_i=motion_noise,
        motion_noise_j=zero,
    )


@dataclass
class WangVehicleState:
    """Position and velocity for a fixed slot in the formation."""

    robot_id: str
    platform: Platform
    position: Array
    home: Array
    waypoints: tuple[Array, ...]
    velocity: Array
    waypoint_index: int = 1
    control_effort: float = 0.0
    formation_offset: Array | None = None

    @classmethod
    def from_plan(
        cls,
        robot_id: str,
        platform: Platform,
        home: Sequence[float],
        waypoints: Sequence[Sequence[float]],
    ) -> "WangVehicleState":
        points = tuple(_as_vector(point, name="waypoint") for point in waypoints)
        if len(points) < 2:
            raise ValueError("A Wang vehicle needs at least a home and one waypoint")
        return cls(
            robot_id=robot_id,
            platform=platform,
            position=_as_vector(home, name="home"),
            home=_as_vector(home, name="home"),
            waypoints=points,
            velocity=np.zeros(2, dtype=float),
        )

    @property
    def task_active(self) -> bool:
        return 2 <= self.waypoint_index <= len(self.waypoints) - 3

    def at_home(self, tolerance: float = 0.8) -> bool:
        """Check whether the waypoint index has advanced past the last waypoint."""
        return (
            self.waypoint_index >= len(self.waypoints)
            and float(np.linalg.norm(self.position - self.home)) <= float(tolerance)
        )

    def target(self) -> Array:
        return self.waypoints[min(self.waypoint_index, len(self.waypoints) - 1)]

    def update_waypoint(self, tolerance: float = 0.75) -> None:
        """Advance the waypoint when the state meets the transition condition."""

        if self.waypoint_index < len(self.waypoints):
            if float(np.linalg.norm(self.target() - self.position)) <= float(tolerance):
                self.waypoint_index += 1

    def nominal_acceleration(
        self,
        kp: float,
        kd: float,
        maximum: float,
        *,
        waypoint_tolerance: float = 0.75,
        advance_waypoint: bool = True,
    ) -> Array:
        if advance_waypoint:
            self.update_waypoint(waypoint_tolerance)
        command = float(kp) * (self.target() - self.position) - float(kd) * self.velocity
        return np.clip(command, -float(maximum), float(maximum))


def propagate_true_wang_dynamics(
    vehicle: WangVehicleState,
    planar_control: Sequence[float] | Array,
    true_disturbance: Sequence[float] | Array,
    dt: float,
    motion_noise: Sequence[float] | Array | None = None,
    *,
    home_tolerance: float = 0.8,
) -> Array:
    """Advance the double integrator under constant acceleration a = u + d + w over dt. Update position by dt*v + dt²*a/2 and velocity by dt*a."""

    control = _as_vector(planar_control, name="planar_control")
    disturbance = _as_vector(true_disturbance, name="true_disturbance")
    noise = np.zeros(2) if motion_noise is None else _as_vector(motion_noise, name="motion_noise")
    if not np.isfinite(dt) or float(dt) <= 0.0:
        raise ValueError("dt must be finite and positive")
    old_velocity = vehicle.velocity.copy()
    acceleration = control + disturbance + noise
    vehicle.position = (
        vehicle.position
        + float(dt) * old_velocity
        + 0.5 * float(dt) ** 2 * acceleration
    )
    vehicle.velocity = old_velocity + float(dt) * acceleration
    vehicle.control_effort += float(dt) * float(control @ control)
    return acceleration


