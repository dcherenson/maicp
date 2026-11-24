"""Shared dynamics for ground robots and aerial robots flying at a fixed altitude."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from maicp.configuration import Platform, UAVDynamics


def _clip_norm(vector: np.ndarray, maximum: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector if norm <= maximum or norm == 0.0 else vector * (maximum / norm)


@dataclass
class VehicleState:
    robot_id: str
    platform: Platform
    position: np.ndarray
    home: np.ndarray
    waypoints: tuple[np.ndarray, ...]
    uav_dynamics: UAVDynamics
    velocity: np.ndarray
    heading: float
    altitude: float
    waypoint_index: int = 1
    control_effort: float = 0.0

    ugv_cbf_offset = 0.45

    @classmethod
    def from_plan(
        cls,
        robot_id: str,
        platform: Platform,
        home: tuple[float, float],
        waypoints: tuple[tuple[float, float], ...],
        uav_dynamics: UAVDynamics,
    ) -> "VehicleState":
        points = tuple(np.asarray(point, dtype=float) for point in waypoints)
        return cls(
            robot_id=robot_id,
            platform=platform,
            position=np.asarray(home, dtype=float),
            home=np.asarray(home, dtype=float),
            waypoints=points,
            uav_dynamics=uav_dynamics,
            velocity=np.zeros(2, dtype=float),
            heading=float(np.arctan2(points[1][1] - home[1], points[1][0] - home[0])),
            altitude=0.0 if platform is Platform.UGV else 12.0,
        )

    @property
    def at_home(self) -> bool:
        return self.waypoint_index >= len(self.waypoints) and np.linalg.norm(self.position - self.home) <= 0.8

    @property
    def cbf_position(self) -> np.ndarray:
        if self.platform is Platform.UAV:
            return self.position.copy()
        return self.position + self.ugv_cbf_offset * np.array(
            [np.cos(self.heading), np.sin(self.heading)]
        )

    @property
    def task_active(self) -> bool:
        return 2 <= self.waypoint_index <= len(self.waypoints) - 3

    def desired_planar_input(self, dt: float) -> np.ndarray:
        if self.waypoint_index >= len(self.waypoints):
            # Hold position at the final waypoint to compensate for wind.
            target = self.home
        else:
            target = self.waypoints[self.waypoint_index]
        displacement = target - self.position
        if self.waypoint_index < len(self.waypoints) and np.linalg.norm(displacement) <= 0.75:
            self.waypoint_index += 1
            if self.waypoint_index >= len(self.waypoints):
                target = self.home
            else:
                target = self.waypoints[self.waypoint_index]
            displacement = target - self.position
        if self.platform is Platform.UGV:
            return _clip_norm(displacement / max(dt, 1e-9), 5.0)
        if self.uav_dynamics is UAVDynamics.VELOCITY_CONTROLLED:
            return _clip_norm(1.6 * displacement, 6.0)
        desired_velocity = _clip_norm(1.4 * displacement, 7.0)
        return _clip_norm(2.0 * (desired_velocity - self.velocity), 4.0)

    def step(self, command: np.ndarray, dt: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Advance and return `(nominal_velocity, reference_velocity)`."""

        if self.at_home:
            self.position = self.home.copy()
            self.velocity = np.zeros(2)
            return self.velocity.copy(), self.velocity.copy()
        command = np.asarray(command, dtype=float)
        if self.platform is Platform.UGV:
            forward = np.array([np.cos(self.heading), np.sin(self.heading)])
            lateral = np.array([-np.sin(self.heading), np.cos(self.heading)])
            speed = float(np.clip(forward @ command, -8.0, 8.0))
            angular_rate = float(
                np.clip((lateral @ command) / self.ugv_cbf_offset, -2.5, 2.5)
            )
            nominal_center_velocity = speed * forward
            slip = 0.045 * speed * angular_rate * lateral
            drag = -0.018 * abs(speed) * nominal_center_velocity
            disturbance = slip + drag + rng.normal(0.0, 0.015, size=2)
            reference_center_velocity = nominal_center_velocity + disturbance
            nominal_velocity = nominal_center_velocity + self.ugv_cbf_offset * angular_rate * lateral
            reference_velocity = reference_center_velocity + self.ugv_cbf_offset * angular_rate * lateral
            self.position = self.position + reference_center_velocity * dt
            self.heading += angular_rate * dt
            self.velocity = reference_center_velocity
        elif self.uav_dynamics is UAVDynamics.VELOCITY_CONTROLLED:
            nominal_velocity = _clip_norm(command, 12.0)
            # Reflect the same local disturbance law into each quadrant.
            quadrant_sign = np.where(self.home >= 0.0, 1.0, -1.0)
            wind = quadrant_sign * np.array([0.16, -0.09]) + rng.normal(0.0, 0.02, size=2)
            reference_velocity = nominal_velocity + wind - 0.025 * nominal_velocity
            self.velocity = reference_velocity
        else:
            nominal_acceleration = _clip_norm(command, 8.0)
            quadrant_sign = np.where(self.home >= 0.0, 1.0, -1.0)
            wind_acceleration = quadrant_sign * np.array([0.11, -0.07]) - 0.035 * self.velocity
            reference_acceleration = nominal_acceleration + wind_acceleration + rng.normal(0.0, 0.015, size=2)
            nominal_velocity = self.velocity + nominal_acceleration * dt
            self.velocity = _clip_norm(self.velocity + reference_acceleration * dt, 12.0)
            reference_velocity = self.velocity.copy()
        if self.platform is Platform.UAV:
            self.position = self.position + reference_velocity * dt
        self.control_effort += float(np.dot(command, command) * dt)
        return nominal_velocity, reference_velocity
