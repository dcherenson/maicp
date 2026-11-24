"""Formation offsets, center trajectories that minimize jerk, obstacle layouts, and interval clearance."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from math import cos, isfinite, pi, sin
from typing import Any

import numpy as np

from experiments.formation.workers import CircularObstacle
from experiments.mission import CanonicalMissionFrame, canonical_mission_frame
from maicp.configuration import (
    FORMATION_CENTER_DURATION_S,
    FORMATION_CENTER_GOAL,
    FORMATION_CENTER_START,
    FORMATION_GATE_JITTER_M,
    FORMATION_GATE_PENETRATION,
    FORMATION_GATE_RADIUS,
    FORMATION_OBSTACLE_COUNT,
    FORMATION_OBSTACLE_LAYOUT,
    FORMATION_RADIUS,
    ExperimentConfig,
    Platform,
)
from maicp.types import MissionPlan

FORMATION_CENTER = tuple(float(value) for value in FORMATION_CENTER_START)
FORMATION_GOAL = tuple(float(value) for value in FORMATION_CENTER_GOAL)
FORMATION_DURATION = float(FORMATION_CENTER_DURATION_S)
FORMATION_RADIUS_M = float(FORMATION_RADIUS)
# Public geometry spelling used by manifests and lightweight callers.
FORMATION_RADIUS = FORMATION_RADIUS_M

# The pillars are shared by UGVs and UAVs (``platform=None``). Their IDs are
# stable across reflected subteams, so a collision report refers to the same
# catalog entry regardless of the quadrant in which it occurred.
FORMATION_PILLAR_CATALOG = (
    ("pillar_0", (15.0, 12.0), 1.1),
    ("pillar_1", (20.0, 19.0), 1.0),
    ("pillar_2", (25.0, 15.5), 1.2),
)

_SLOT_COUNT = 6
_UGV_SLOTS = (0, 2, 4)
_UAV_SLOTS = (1, 3, 5)


def sample_formation_phase(plan: MissionPlan) -> float:
    """Return the keyed canonical formation phase for one mission."""

    # Import lazily to keep this geometry module independent of the simulator
    # import graph while sharing its stream of random mission inputs.
    from experiments.formation.disturbances import _keyed_rng, collision_mission_seed

    rng = _keyed_rng(collision_mission_seed(plan), "formation-phase")
    return float(rng.uniform(0.0, 2.0 * pi))


def _validate_formation_phase(value: float) -> float:
    phase = float(value)
    if not isfinite(phase) or not 0.0 <= phase < 2.0 * pi:
        raise ValueError("formation phase must be finite and lie in [0, 2*pi)")
    return phase


def rotate_formation_offsets(
    canonical_offsets: Mapping[str, Sequence[float] | np.ndarray],
    formation_phase_rad: float,
) -> dict[str, np.ndarray]:
    """Rotate canonical slot vectors rigidly in the canonical mission frame."""

    phase = _validate_formation_phase(formation_phase_rad)
    rotation = np.asarray(
        [[cos(phase), -sin(phase)], [sin(phase), cos(phase)]], dtype=float
    )
    return {
        str(robot_id): rotation @ _as_planar(offset, name=f"formation offset[{robot_id}]")
        for robot_id, offset in canonical_offsets.items()
    }


def _as_planar(value: Sequence[float] | np.ndarray, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (2,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite planar vector")
    return result.copy()


def _config_value(config: ExperimentConfig | None, name: str, default: Any) -> Any:
    return default if config is None else getattr(config, name, default)


def _frame(value: CanonicalMissionFrame | int | None) -> CanonicalMissionFrame | None:
    if value is None:
        return None
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return canonical_mission_frame(int(value))
    if not all(hasattr(value, name) for name in ("point_to_global", "vector_to_global")):
        raise TypeError("frame must be a canonical mission frame or subteam index")
    return value  # type: ignore[return-value]


def _canonical_slots(radius: float) -> tuple[np.ndarray, ...]:
    if not isfinite(radius) or radius <= 0.0:
        raise ValueError("formation radius must be finite and positive")
    return tuple(
        radius
        * np.asarray(
            [cos(2.0 * pi * index / _SLOT_COUNT), sin(2.0 * pi * index / _SLOT_COUNT)],
            dtype=float,
        )
        for index in range(_SLOT_COUNT)
    )


def canonical_formation_offsets(
    count: int = _SLOT_COUNT,
    radius: float = FORMATION_RADIUS_M,
) -> tuple[np.ndarray, ...]:
    """Return six polygon slots in alternating UGV and UAV order."""

    if isinstance(count, bool) or not isinstance(count, (int, np.integer)):
        raise ValueError("formation_obstacles_v1 has exactly six slots")
    if int(count) != _SLOT_COUNT:
        raise ValueError("formation_obstacles_v1 has exactly six slots")
    return _canonical_slots(float(radius))


def canonical_formation_points(
    count: int = _SLOT_COUNT,
    *,
    center: Sequence[float] = FORMATION_CENTER,
    radius: float = FORMATION_RADIUS_M,
) -> tuple[np.ndarray, ...]:
    """Return the six canonical slot points around ``center``."""

    origin = _as_planar(center, name="formation center")
    return tuple(origin + offset for offset in canonical_formation_offsets(count, radius))


def offsets(
    plan: MissionPlan,
    config: ExperimentConfig | None = None,
    *,
    formation_phase_rad: float | None = None,
) -> dict[str, np.ndarray]:
    """Assign UGVs to even polygon slots and UAVs to odd slots in mission coordinates."""

    radius = FORMATION_RADIUS_M
    if config is not None:
        radius = float(getattr(config, "formation_radius", radius))
    slots = canonical_formation_offsets(radius=radius)
    canonical_result: dict[str, np.ndarray] = {}
    for platform, slot_indices in (
        (Platform.UGV, _UGV_SLOTS),
        (Platform.UAV, _UAV_SLOTS),
    ):
        ids = tuple(plan.robot_ids.get(platform, ()))
        if len(ids) != len(slot_indices):
            raise ValueError(
                "formation_obstacles_v1 requires exactly three UGVs and three UAVs"
        )
        for robot_id, slot_index in zip(ids, slot_indices, strict=True):
            canonical_result[str(robot_id)] = slots[slot_index].copy()
    if len(canonical_result) != _SLOT_COUNT:
        raise ValueError("formation_obstacles_v1 requires exactly six robots per subteam")
    total = np.sum(np.asarray(tuple(canonical_result.values())), axis=0)
    if not np.allclose(total, 0.0, atol=1e-12, rtol=0.0):
        raise ValueError("formation offsets must have zero sum")
    phase = (
        sample_formation_phase(plan)
        if formation_phase_rad is None
        else _validate_formation_phase(formation_phase_rad)
    )
    return rotate_formation_offsets(canonical_result, phase)


def _minimum_jerk(
    time_s: float,
    start: np.ndarray,
    goal: np.ndarray,
    duration: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not isfinite(float(time_s)):
        raise ValueError("reference time must be finite")
    if not isfinite(duration) or duration <= 0.0:
        raise ValueError("reference duration must be finite and positive")
    tau = float(np.clip(float(time_s) / duration, 0.0, 1.0))
    displacement = goal - start
    smooth = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
    smooth_dot = (30.0 * tau**2 - 60.0 * tau**3 + 30.0 * tau**4) / duration
    smooth_ddot = (60.0 * tau - 180.0 * tau**2 + 120.0 * tau**3) / duration**2
    # Derivatives are exactly zero after the hold begins. The polynomial
    # happens to evaluate to zero at tau=1, but making the hold explicit keeps
    # the endpoint semantics clear and avoids accumulation of rounding errors.
    if float(time_s) >= duration:
        smooth_dot = 0.0
        smooth_ddot = 0.0
    return (
        start + smooth * displacement,
        smooth_dot * displacement,
        smooth_ddot * displacement,
    )


def reference(
    time_s: float,
    frame: CanonicalMissionFrame | int | ExperimentConfig | None = None,
    config: ExperimentConfig | None = None,
    *,
    return_start_s: float | None = None,
    return_origin: Sequence[float] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the center position, velocity, and acceleration along the trajectory that minimizes jerk."""

    if isinstance(frame, ExperimentConfig):
        if config is not None:
            raise TypeError("config was supplied twice")
        config = frame
        frame = None
    selected_frame = _frame(frame)
    start = _as_planar(
        _config_value(config, "formation_center_start", FORMATION_CENTER),
        name="formation_center_start",
    )
    goal = _as_planar(
        _config_value(config, "formation_center_goal", FORMATION_GOAL),
        name="formation_center_goal",
    )
    duration = float(
        _config_value(config, "formation_center_duration_s", FORMATION_DURATION)
    )
    if return_start_s is not None:
        if not np.isfinite(return_start_s) or return_start_s < duration:
            raise ValueError("return_start_s must be finite and follow the outbound leg")
    if return_start_s is not None and float(time_s) >= return_start_s:
        center, velocity, acceleration = _minimum_jerk(
            float(time_s) - return_start_s,
            goal if return_origin is None else _as_planar(return_origin, name="return_origin"),
            start, duration
        )
    else:
        center, velocity, acceleration = _minimum_jerk(float(time_s), start, goal, duration)
    if selected_frame is None:
        return center, velocity, acceleration
    return (
        selected_frame.point_to_global(center),
        selected_frame.vector_to_global(velocity),
        selected_frame.vector_to_global(acceleration),
    )


def pillar_catalog(config: ExperimentConfig | None = None) -> tuple:
    """Return the pillar centers and radii for the configured layout."""

    count = _config_value(config, "formation_obstacle_count", FORMATION_OBSTACLE_COUNT)
    if (
        isinstance(count, bool)
        or not isinstance(count, (int, np.integer))
        or int(count) < len(FORMATION_PILLAR_CATALOG)
    ):
        raise ValueError("formation_obstacle_count must be an integer >= 3")
    count = int(count)
    positions = _config_value(config, "formation_pillar_positions", None)
    if positions is None:
        base_catalog = tuple(
            (str(obstacle_id), tuple(center), float(radius))
            for obstacle_id, center, radius in FORMATION_PILLAR_CATALOG
        )
    else:
        if len(positions) != len(FORMATION_PILLAR_CATALOG):
            raise ValueError("exactly three pillar positions are required")
        overridden = tuple(
            tuple(_as_planar(position, name=f"pillar_{index}"))
            for index, position in enumerate(positions)
        )
        base_catalog = tuple(
            (str(obstacle_id), overridden[index], float(radius))
            for index, (obstacle_id, _, radius) in enumerate(FORMATION_PILLAR_CATALOG)
        )
    return tuple(
        (
            f"pillar_{index}",
            base_catalog[index % len(base_catalog)][1],
            base_catalog[index % len(base_catalog)][2],
        )
        for index in range(count)
    )


def _slot_gate_obstacles(
    plan: MissionPlan,
    config: ExperimentConfig | None,
    frame: CanonicalMissionFrame | int | None,
    *,
    formation_phase_rad: float | None,
    start: np.ndarray,
    goal: np.ndarray,
    parallel: np.ndarray,
    perpendicular: np.ndarray,
    robot_radius: float,
    initial_buffer: float,
) -> tuple[CircularObstacle, ...]:
    """Place six obstacles relative to the rotated formation slots."""

    from experiments.formation.disturbances import _keyed_rng, collision_mission_seed

    count = _config_value(config, "formation_obstacle_count", FORMATION_OBSTACLE_COUNT)
    if isinstance(count, bool) or int(count) != _SLOT_COUNT:
        raise ValueError("slot_gates requires exactly six formation obstacles")
    radius = float(_config_value(config, "formation_gate_radius", FORMATION_GATE_RADIUS))
    penetration = float(
        _config_value(config, "formation_gate_penetration", FORMATION_GATE_PENETRATION)
    )
    jitter_m = float(
        _config_value(config, "formation_gate_jitter_m", FORMATION_GATE_JITTER_M)
    )
    if not isfinite(radius) or not 1.0 <= radius <= 1.6:
        raise ValueError("formation_gate_radius must lie in [1.0, 1.6]")
    if not isfinite(penetration) or not 0.15 <= penetration <= 0.6:
        raise ValueError("formation_gate_penetration must lie in [0.15, 0.6]")
    if not isfinite(jitter_m) or jitter_m < 0.0:
        raise ValueError("formation_gate_jitter_m must be finite and nonnegative")
    clearance = radius + robot_radius - penetration
    if not isfinite(clearance) or clearance <= 0.0:
        raise ValueError("slot gate footprint clearance must be finite and positive")

    rotated_offsets = offsets(
        plan,
        config,
        formation_phase_rad=formation_phase_rad,
    )
    ring_ids = tuple(
        robot_id
        for index in range(3)
        for platform in (Platform.UGV, Platform.UAV)
        for robot_id in plan.robot_ids[platform][index:index + 1]
    )
    if len(ring_ids) != _SLOT_COUNT or set(ring_ids) != set(rotated_offsets):
        raise ValueError("slot_gates requires exactly six alternating formation robots")
    slot_offsets = tuple(rotated_offsets[robot_id] for robot_id in ring_ids)

    # The first three slots share the first interior station and the final
    # three share the second.  This clusters encounters between classes while
    # still leaving a free side of the route around each group of three obstacles.
    # Move the gates slightly toward the route midpoint.  At the default
    # radius and footprint this leaves enough endpoint separation for every
    # formation phase while retaining two distinct interior stations.
    station_fractions = np.asarray((0.46, 0.46, 0.46, 0.54, 0.54, 0.54), dtype=float)
    # Place the three obstacles on the same side of the route to preserve slot spacing.
    side_signs = np.asarray((1.0, 1.0, 1.0, -1.0, -1.0, -1.0), dtype=float)
    mission_seed = collision_mission_seed(plan)
    rng = _keyed_rng(mission_seed, "formation-slot-gates")
    endpoint_points = (
        start + np.asarray(slot_offsets),
        goal + np.asarray(slot_offsets),
    )
    for _ in range(1000):
        route_jitter = rng.uniform(-jitter_m, jitter_m, size=_SLOT_COUNT)
        target_points = np.asarray(
            [
                start
                + (float(fraction) * (goal - start) + float(delta) * parallel)
                + offset
                for fraction, delta, offset in zip(
                    station_fractions, route_jitter, slot_offsets, strict=True
                )
            ],
            dtype=float,
        )
        centers = target_points + side_signs[:, None] * clearance * perpendicular
        if all(
            np.all(
                np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
                > radius + robot_radius + buffer
            )
            for points, buffer in zip(endpoint_points, (initial_buffer, 0.0), strict=True)
        ):
            break
    else:
        raise RuntimeError("No valid slot-gate layout after 1000 attempts")

    selected_frame = _frame(
        canonical_mission_frame(plan.subteam_id) if frame is None else frame
    )
    if selected_frame is None:  # pragma: no cover - plan always has a subteam
        raise ValueError("an obstacle catalog requires a mission frame")
    return tuple(
        CircularObstacle(
            center=selected_frame.point_to_global(np.asarray(center, dtype=float)),
            radius=radius,
            platform=None,
            obstacle_id=f"pillar_{index}",
        )
        for index, center in enumerate(centers)
    )


def obstacles(
    plan: MissionPlan,
    config: ExperimentConfig | None = None,
    frame: CanonicalMissionFrame | int | None = None,
    *,
    formation_phase_rad: float | None = None,
) -> tuple[CircularObstacle, ...]:
    """Sample one canonical layout at mission setup, then reflect shared pillars."""

    from experiments.formation.disturbances import _keyed_rng, collision_mission_seed

    start = _as_planar(
        _config_value(config, "formation_center_start", FORMATION_CENTER),
        name="formation_center_start",
    )
    goal = _as_planar(
        _config_value(config, "formation_center_goal", FORMATION_GOAL),
        name="formation_center_goal",
    )
    route = goal - start
    length = float(np.linalg.norm(route))
    if length == 0.0:
        raise ValueError("Obstacle sampling requires a nonzero formation-center route")
    parallel = route / length
    perpendicular = np.asarray([-parallel[1], parallel[0]])
    sigma = float(_config_value(config, "obstacle_position_sigma_m", 1.0))
    if not isfinite(sigma) or sigma < 0.0:
        raise ValueError("obstacle_position_sigma_m must be finite and nonnegative")
    robot_radius = float(_config_value(config, "collision_formation_agent_radius", 0.45))
    initial_buffer = float(_config_value(config, "collision_initial_position_radius", 0.0))
    slots = np.asarray(
        tuple(
            offsets(
                plan,
                config,
                formation_phase_rad=formation_phase_rad,
            ).values()
        )
    )
    endpoints = (start + slots, goal + slots)
    layout = _config_value(config, "formation_obstacle_layout", FORMATION_OBSTACLE_LAYOUT)
    if layout == "slot_gates":
        return _slot_gate_obstacles(
            plan,
            config,
            frame,
            formation_phase_rad=formation_phase_rad,
            start=start,
            goal=goal,
            parallel=parallel,
            perpendicular=perpendicular,
            robot_radius=robot_radius,
            initial_buffer=initial_buffer,
        )
    if layout != "random":
        raise ValueError("formation_obstacle_layout must be 'random' or 'slot_gates'")
    catalog = pillar_catalog(config)
    base_catalog = catalog[: len(FORMATION_PILLAR_CATALOG)]
    radii = np.asarray([radius for _, _, radius in base_catalog])
    nominal = np.asarray([center for _, center, _ in base_catalog])
    # Initial positions and disturbances use separate random streams.
    rng = _keyed_rng(collision_mission_seed(plan), "formation-obstacle-positions")
    for _ in range(1000):
        along = rng.uniform(-sigma, sigma, size=len(FORMATION_PILLAR_CATALOG))
        across = rng.uniform(-sigma, sigma, size=len(FORMATION_PILLAR_CATALOG))
        centers = nominal + along[:, None] * parallel + across[:, None] * perpendicular
        if all(
            np.all(np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
                   > radii[None, :] + robot_radius + buffer)
            for points, buffer in zip(endpoints, (initial_buffer, 0.0), strict=True)
        ):
            break
    else:
        raise RuntimeError("No valid formation obstacle layout after 1000 attempts")

    # Density candidates are independently keyed so adding a pillar cannot
    # consume or shift any draw used by the original layout with three pillars.
    centers = [np.asarray(center, dtype=float) for center in centers]
    mission_seed = collision_mission_seed(plan)
    for obstacle_id, nominal_center, radius in catalog[len(FORMATION_PILLAR_CATALOG) :]:
        extra_rng = _keyed_rng(
            mission_seed, "formation-obstacle-positions", str(obstacle_id)
        )
        for _ in range(1000):
            along = float(extra_rng.uniform(-sigma, sigma))
            across = float(extra_rng.uniform(-sigma, sigma))
            candidate = (
                np.asarray(nominal_center, dtype=float)
                + along * parallel
                + across * perpendicular
            )
            if (
                np.all(
                    np.linalg.norm(start + slots - candidate, axis=1)
                    > float(radius) + robot_radius + initial_buffer
                )
                and np.all(
                    np.linalg.norm(goal + slots - candidate, axis=1)
                    > float(radius) + robot_radius
                )
            ):
                centers.append(candidate)
                break
        else:
            raise RuntimeError(
                f"No valid formation obstacle layout for {obstacle_id} after 1000 attempts"
            )

    selected_frame = _frame(
        canonical_mission_frame(plan.subteam_id) if frame is None else frame
    )
    if selected_frame is None:  # pragma: no cover - plan always has a subteam
        raise ValueError("an obstacle catalog requires a mission frame")
    return tuple(
        CircularObstacle(
            center=selected_frame.point_to_global(np.asarray(center, dtype=float)),
            radius=float(radius),
            platform=None,
            obstacle_id=str(obstacle_id),
        )
        for (obstacle_id, _, radius), center in zip(catalog, centers, strict=True)
    )


def _validate_initial_geometry(
    plan: MissionPlan,
    config: ExperimentConfig,
    agent_radius: float,
    formation_phase_rad: float | None = None,
) -> None:
    """Check the initial separation between formation slots."""

    if not isfinite(agent_radius) or agent_radius < 0.0:
        raise ValueError("agent_radius must be finite and nonnegative")
    initial_buffer = float(getattr(config, "collision_initial_position_radius", 0.0))
    if not isfinite(initial_buffer) or initial_buffer < 0.0:
        raise ValueError(
            "collision_initial_position_radius must be finite and nonnegative"
        )
    local_offsets = offsets(
        plan,
        config,
        formation_phase_rad=formation_phase_rad,
    )
    required_spacing = 2.0 * (agent_radius + initial_buffer)
    for platform in Platform:
        ids = tuple(plan.robot_ids[platform])
        for left_index, left in enumerate(ids):
            for right in ids[left_index + 1 :]:
                distance = float(np.linalg.norm(local_offsets[left] - local_offsets[right]))
                if distance <= required_spacing:
                    raise ValueError("formation slots are not robustly separated")


def build_formation_plan(
    plan: MissionPlan,
    config: ExperimentConfig,
    *,
    agent_radius: float | None = None,
    formation_phase_rad: float | None = None,
) -> MissionPlan:
    """Attach the formation route for this phase to the mission plan.

    Waypoints contain the start and destination slots. The continuous
    reference governs the outbound leg and holds the destination afterward. Each new mission initializes fresh states at its start slots.
    """

    radius = (
        float(getattr(config, "collision_formation_agent_radius", 0.45))
        if agent_radius is None
        else float(agent_radius)
    )
    phase = (
        sample_formation_phase(plan)
        if formation_phase_rad is None
        else _validate_formation_phase(formation_phase_rad)
    )
    _validate_initial_geometry(plan, config, radius, phase)
    frame = canonical_mission_frame(plan.subteam_id)
    local_offsets = offsets(plan, config, formation_phase_rad=phase)
    center_start, _, _ = reference(0.0, config=config)
    center_goal, _, _ = reference(
        float(getattr(config, "formation_center_duration_s", FORMATION_DURATION)),
        config=config,
    )
    home_positions: dict[str, tuple[float, float]] = {}
    waypoints: dict[str, tuple[tuple[float, float], ...]] = {}
    for robot_id, local_offset in local_offsets.items():
        global_start = frame.point_to_global(center_start + local_offset)
        global_goal = frame.point_to_global(center_goal + local_offset)
        home_positions[robot_id] = tuple(float(value) for value in global_start)
        waypoints[robot_id] = (
            tuple(float(value) for value in global_start),
            tuple(float(value) for value in global_goal),
        )
    return replace(plan, home_positions=home_positions, waypoints=waypoints)


__all__ = [
    "FORMATION_CENTER",
    "FORMATION_GOAL",
    "FORMATION_DURATION",
    "FORMATION_RADIUS",
    "FORMATION_RADIUS_M",
    "FORMATION_PILLAR_CATALOG",
    "canonical_formation_offsets",
    "sample_formation_phase",
    "rotate_formation_offsets",
    "canonical_formation_points",
    "offsets",
    "reference",
    "obstacles",
    "pillar_catalog",
    "build_formation_plan",
]




def _vector(value: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (2,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite planar vector")
    return result.copy()


def quadratic_segment_circle_clearance(
    position: Sequence[float] | np.ndarray,
    velocity: Sequence[float] | np.ndarray,
    acceleration: Sequence[float] | np.ndarray,
    obstacle_center: Sequence[float] | np.ndarray,
    footprint_radius: float,
    dt: float,
) -> tuple[float, float]:
    """Return ``(minimum_clearance, tau)`` for a step with constant acceleration.

    ``tau`` lies in ``[0, 1]`` and identifies the point attaining the minimum
    (the first one in a numerical tie).  The clearance subtracts the obstacle
    and robot footprint radii.  The computation checks all roots of the cubic
    derivative of the squared distance and ignores complex or inadmissible
    roots.
    """

    p0 = _vector(position, "position")
    v0 = _vector(velocity, "velocity")
    accel = _vector(acceleration, "acceleration")
    center = _vector(obstacle_center, "obstacle_center")
    radius = float(footprint_radius)
    step = float(dt)
    if not np.isfinite(radius) or radius < 0.0:
        raise ValueError("footprint_radius must be finite and nonnegative")
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("dt must be finite and positive")

    # q(t) = c0 + c1*tau + c2*tau^2, with tau in [0,1].
    c0 = p0 - center
    c1 = step * v0
    c2 = 0.5 * step**2 * accel
    # d/dtau ||q||^2 / 2 = b0 + b1*tau + b2*tau^2 + b3*tau^3.
    derivative = np.asarray(
        [
            float(c0 @ c1),
            float(c1 @ c1 + 2.0 * c0 @ c2),
            float(3.0 * c1 @ c2),
            float(2.0 * c2 @ c2),
        ],
        dtype=float,
    )
    candidates = [0.0, 1.0]
    degree = 3
    while degree > 0 and abs(float(derivative[degree])) <= 1e-14:
        degree -= 1
    if degree:
        roots = np.roots(derivative[: degree + 1][::-1])
        candidates.extend(
            float(root.real)
            for root in roots
            if abs(float(root.imag)) <= 1e-10
            and -1e-12 <= float(root.real) <= 1.0 + 1e-12
        )
    candidates = sorted(min(1.0, max(0.0, value)) for value in candidates)

    def clearance(tau: float) -> float:
        point = c0 + c1 * tau + c2 * tau**2
        return float(np.linalg.norm(point) - radius)

    values = [(clearance(tau), tau) for tau in candidates]
    return min(values, key=lambda value: (value[0], value[1]))


def quadratic_segment_circle_first_violation(
    position: Sequence[float] | np.ndarray,
    velocity: Sequence[float] | np.ndarray,
    acceleration: Sequence[float] | np.ndarray,
    obstacle_center: Sequence[float] | np.ndarray,
    footprint_radius: float,
    dt: float,
) -> float | None:
    """Return the first ``tau`` at which the held segment enters the disk.

    The polynomial for squared distance is quartic.  Real roots in the closed
    interval are checked in order, with a midpoint test after each root so a
    tangent contact is retained while roots that leave the disk are skipped.
    ``None`` means the complete interval stays outside the footprint.
    """

    p0 = _vector(position, "position")
    v0 = _vector(velocity, "velocity")
    accel = _vector(acceleration, "acceleration")
    center = _vector(obstacle_center, "obstacle_center")
    radius = float(footprint_radius)
    step = float(dt)
    if not np.isfinite(radius) or radius < 0.0:
        raise ValueError("footprint_radius must be finite and nonnegative")
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("dt must be finite and positive")
    c0 = p0 - center
    c1 = step * v0
    c2 = 0.5 * step**2 * accel
    distance_squared = np.polynomial.polynomial.polymul(
        np.asarray([c0 @ c0, 2.0 * (c0 @ c1), c1 @ c1 + 2.0 * (c0 @ c2),
                    2.0 * (c1 @ c2), c2 @ c2], dtype=float),
        np.asarray([1.0], dtype=float),
    )
    distance_squared[0] -= radius**2
    degree = len(distance_squared) - 1
    while degree > 0 and abs(float(distance_squared[degree])) <= 1e-14:
        degree -= 1
    polynomial = distance_squared[: degree + 1]

    def value(tau: float) -> float:
        return float(np.polynomial.polynomial.polyval(tau, polynomial))

    tolerance = 1e-10
    if value(0.0) <= tolerance:
        return 0.0
    if degree == 0:
        return None
    roots = np.roots(polynomial[::-1])
    real_roots = sorted(
        {
            round(float(root.real), 14)
            for root in roots
            if abs(float(root.imag)) <= 1e-9
            and tolerance * -10.0 <= float(root.real) <= 1.0 + tolerance * 10.0
        }
    )
    for root in real_roots:
        tau = min(1.0, max(0.0, root))
        if abs(value(tau)) <= tolerance:
            return tau
    if value(1.0) <= tolerance:
        # A numerically unresolved root can only occur at the endpoint here.
        return 1.0
    return None


def exact_segment_circle_clearance(*args, **kwargs) -> tuple[float, float]:
    """Readable alias for :func:`quadratic_segment_circle_clearance`."""

    return quadratic_segment_circle_clearance(*args, **kwargs)
