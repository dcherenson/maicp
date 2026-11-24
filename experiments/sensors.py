"""Sensors for each robot class with a common Cartesian output."""

from __future__ import annotations

import numpy as np


def _angle_difference(angle: float, reference: float) -> float:
    return float(np.arctan2(np.sin(angle - reference), np.cos(angle - reference)))


def range_bearing_measurement(
    observer: np.ndarray,
    heading: float,
    target: np.ndarray,
    rng: np.random.Generator,
    max_range: float = 25.0,
    fov: float = 1.5 * np.pi,
    sigma_range: float = 0.20,
    sigma_bearing: float = 0.025,
) -> tuple[bool, np.ndarray, np.ndarray]:
    displacement = np.asarray(target) - np.asarray(observer)
    distance = float(np.linalg.norm(displacement))
    bearing = float(np.arctan2(displacement[1], displacement[0]))
    if distance > max_range or abs(_angle_difference(bearing, heading)) > 0.5 * fov:
        return False, np.zeros(2), np.eye(2)
    measured_range = max(0.0, distance + rng.normal(0.0, sigma_range))
    measured_bearing = bearing + rng.normal(0.0, sigma_bearing)
    measurement = np.asarray(observer) + measured_range * np.array(
        [np.cos(measured_bearing), np.sin(measured_bearing)]
    )
    jacobian = np.array(
        [
            [np.cos(bearing), -distance * np.sin(bearing)],
            [np.sin(bearing), distance * np.cos(bearing)],
        ]
    )
    covariance = jacobian @ np.diag([sigma_range**2, sigma_bearing**2]) @ jacobian.T
    covariance += 1e-4 * np.eye(2)
    return True, measurement, covariance


def cartesian_measurement(
    observer: np.ndarray,
    target: np.ndarray,
    rng: np.random.Generator,
    max_range: float = 38.0,
    sigma: float = 0.12,
) -> tuple[bool, np.ndarray, np.ndarray]:
    if np.linalg.norm(np.asarray(target) - np.asarray(observer)) > max_range:
        return False, np.zeros(2), np.eye(2)
    covariance = np.eye(2) * sigma**2
    return True, np.asarray(target) + rng.normal(0.0, sigma, size=2), covariance
