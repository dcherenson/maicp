"""UGV LiDAR and UAV camera measurement models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


# Fixed physical sensor parameters for this case study.  Keep these values in
# one place so traces, tests, and the simulator describe the same model.
LIDAR_MAX_RANGE = 25.0
LIDAR_FOV_RADIANS = 1.5 * np.pi
LIDAR_SIGMA_RANGE = 0.20
LIDAR_SIGMA_BEARING = 0.020
CAMERA_MAX_RANGE = 50.0
CAMERA_FOV_RADIANS = 2.0 * np.pi / 3.0
CAMERA_SIGMA_BEARING = 0.025
UAV_CROSS_TRACK_FLOOR = 0.05



def wrap_angle(angle: float) -> float:
    """Return ``angle`` wrapped to ``[-pi, pi]``."""

    return float(np.arctan2(np.sin(float(angle)), np.cos(float(angle))))


def _as_position(value: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (2,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite planar position")
    return result


def _as_heading(value: float, name: str = "heading") -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _as_positive(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _as_finite(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass
class SensorPacket:
    """One heterogeneous physical measurement before estimator processing.

    ``value``, ``H``, and ``covariance`` are the exact arrays supplied to the
    tracker.  Polar fields are retained only as traceable physical sensor
    diagnostics. ``prior_range`` is populated for a UAV packet and documents
    how its angular covariance was scaled. It is not an additional measurement.

    Estimator code should use the named fields, including ``H`` and
    ``sensor_type``.
    """

    visible: bool
    value: np.ndarray
    H: np.ndarray
    covariance: np.ndarray
    sensor_type: str
    lidar_range: float | None = None
    lidar_bearing: float | None = None
    camera_bearing: float | None = None
    prior_range: float | None = None
    measurement_dimension: int | None = None

    def __post_init__(self) -> None:
        self.visible = bool(self.visible)
        self.value = np.asarray(self.value, dtype=float)
        self.H = np.asarray(self.H, dtype=float)
        self.covariance = np.asarray(self.covariance, dtype=float)
        if self.value.ndim != 1:
            raise ValueError("Sensor packet value must be one-dimensional")
        if self.H.ndim != 2 or self.H.shape[0] != self.value.size or self.H.shape[1] != 4:
            raise ValueError("Sensor packet H must have shape (measurement_dimension, 4)")
        if self.covariance.shape != (self.value.size, self.value.size):
            raise ValueError("Sensor packet covariance dimension must match value")
        if not np.all(np.isfinite(self.value)) or not np.all(np.isfinite(self.H)):
            raise ValueError("Sensor packet value and H must be finite")
        if not np.all(np.isfinite(self.covariance)):
            raise ValueError("Sensor packet covariance must be finite")
        self.measurement_dimension = int(self.value.size)

UGV_H = np.asarray(
    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
    dtype=float,
)


def _empty_ugv_packet() -> SensorPacket:
    return SensorPacket(
        visible=False,
        value=np.zeros(2),
        H=UGV_H.copy(),
        covariance=np.eye(2),
        sensor_type="ugv_lidar",
    )


def _empty_uav_packet() -> SensorPacket:
    return SensorPacket(
        visible=False,
        value=np.zeros(1),
        H=np.zeros((1, 4)),
        covariance=np.eye(1),
        sensor_type="uav_camera_bearing",
    )


def ugv_lidar_measurement(
    observer: Sequence[float] | np.ndarray,
    heading: float,
    target: Sequence[float] | np.ndarray,
    rng: np.random.Generator,
    *,
    severity: float = 1.0,
    range_bias: float = 0.0,
    bearing_bias: float = 0.0,
    max_range: float = LIDAR_MAX_RANGE,
    fov: float = LIDAR_FOV_RADIANS,
) -> SensorPacket:
    """Generate one UGV planar LiDAR range+bearing packet.

    Visibility is decided from the true geometry.  Once visible, the noisy
    polar measurement is converted to world Cartesian coordinates.  Its
    covariance contains only the contribution from nominal measurement noise. The
    persistent calibration biases stay unmodelled as required by the case
    study.
    """

    observer_value = _as_position(observer, "observer")
    target_value = _as_position(target, "target")
    heading_value = _as_heading(heading)
    severity_value = _as_positive(severity, "severity")
    range_bias_value = _as_finite(range_bias, "range_bias")
    bearing_bias_value = _as_finite(bearing_bias, "bearing_bias")
    max_range_value = _as_positive(max_range, "max_range")
    fov_value = _as_positive(fov, "fov")
    if fov_value > 2.0 * np.pi:
        raise ValueError("fov cannot exceed 2*pi")

    displacement = target_value - observer_value
    rho = float(np.linalg.norm(displacement))
    absolute_bearing = float(np.arctan2(displacement[1], displacement[0]))
    true_bearing = wrap_angle(absolute_bearing - heading_value)
    if rho > max_range_value or abs(true_bearing) > 0.5 * fov_value:
        return _empty_ugv_packet()

    sigma_range = severity_value * LIDAR_SIGMA_RANGE
    sigma_bearing = severity_value * LIDAR_SIGMA_BEARING
    measured_range = rho + range_bias_value + float(rng.normal(0.0, sigma_range))
    measured_bearing = true_bearing + bearing_bias_value + float(
        rng.normal(0.0, sigma_bearing)
    )
    measured_world_bearing = heading_value + measured_bearing
    measurement = observer_value + measured_range * np.asarray(
        [np.cos(measured_world_bearing), np.sin(measured_world_bearing)]
    )

    # Evaluate the covariance Jacobian at the measured range and bearing.
    jacobian = np.asarray(
        [
            [np.cos(measured_world_bearing), -measured_range * np.sin(measured_world_bearing)],
            [np.sin(measured_world_bearing), measured_range * np.cos(measured_world_bearing)],
        ],
        dtype=float,
    )
    covariance = jacobian @ np.diag([sigma_range**2, sigma_bearing**2]) @ jacobian.T
    return SensorPacket(
        visible=True,
        value=measurement,
        H=UGV_H.copy(),
        covariance=covariance,
        sensor_type="ugv_lidar",
        lidar_range=float(measured_range),
        lidar_bearing=float(measured_bearing),
    )


def uav_camera_bearing_measurement(
    observer: Sequence[float] | np.ndarray,
    heading: float,
    target: Sequence[float] | np.ndarray,
    rng: np.random.Generator,
    *,
    prior_state: Sequence[float] | np.ndarray,
    camera_bias: float = 0.0,
    max_range: float = CAMERA_MAX_RANGE,
    fov: float = CAMERA_FOV_RADIANS,
) -> SensorPacket:
    """Generate one UAV line constraint from a monocular bearing measurement.

    ``prior_state`` must be the local tracker state immediately before the
    current update.  It is used only to scale angular uncertainty into a
    covariance across the line of sight, and no true target range is used in the packet.
    Visibility uses a field of view spanning 120 degrees about the heading and 50 m range. Angular
    noise is constant at 0.025 rad. The prior range is not clipped.
    """

    observer_value = _as_position(observer, "observer")
    target_value = _as_position(target, "target")
    heading_value = _as_heading(heading)
    prior_value = np.asarray(prior_state, dtype=float)
    if prior_value.shape not in ((2,), (4,)) or not np.all(np.isfinite(prior_value)):
        raise ValueError("prior_state must be a finite 2D position or 4D target state")
    prior_position = prior_value[:2]
    camera_bias_value = _as_finite(camera_bias, "camera_bias")
    max_range_value = _as_positive(max_range, "max_range")
    fov_value = _as_positive(fov, "fov")
    if fov_value > 2.0 * np.pi:
        raise ValueError("fov cannot exceed 2*pi")

    displacement = target_value - observer_value
    rho = float(np.linalg.norm(displacement))
    absolute_bearing = float(np.arctan2(displacement[1], displacement[0]))
    true_bearing = wrap_angle(absolute_bearing - heading_value)
    if rho > max_range_value or abs(true_bearing) > 0.5 * fov_value:
        return _empty_uav_packet()

    measured_bearing = true_bearing + camera_bias_value + float(
        rng.normal(0.0, CAMERA_SIGMA_BEARING)
    )
    measured_world_bearing = heading_value + measured_bearing
    normal = np.asarray(
        [-np.sin(measured_world_bearing), np.cos(measured_world_bearing)],
        dtype=float,
    )
    value = np.asarray([float(normal @ observer_value)], dtype=float)
    H = np.asarray([[normal[0], normal[1], 0.0, 0.0]], dtype=float)

    prior_range = float(np.linalg.norm(prior_position - observer_value))
    sigma_perpendicular = np.hypot(
        prior_range * CAMERA_SIGMA_BEARING,
        UAV_CROSS_TRACK_FLOOR,
    )
    covariance = np.asarray([[sigma_perpendicular**2]], dtype=float)
    return SensorPacket(
        visible=True,
        value=value,
        H=H,
        covariance=covariance,
        sensor_type="uav_camera_bearing",
        camera_bearing=float(measured_bearing),
        prior_range=prior_range,
    )


__all__ = [
    "CAMERA_FOV_RADIANS",
    "CAMERA_MAX_RANGE",
    "CAMERA_SIGMA_BEARING",
    "LIDAR_FOV_RADIANS",
    "LIDAR_MAX_RANGE",
    "LIDAR_SIGMA_BEARING",
    "LIDAR_SIGMA_RANGE",
    "SensorPacket",
    "UGV_H",
    "UAV_CROSS_TRACK_FLOOR",
    "ugv_lidar_measurement",
    "uav_camera_bearing_measurement",
    "wrap_angle",
]
