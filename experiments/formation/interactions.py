"""True and learned acceleration fields for nearby robots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from maicp.configuration import Platform


PAIR_RADIUS = 6.0
TRUE_INTERACTION_AMPLITUDE = 0.25
DEFAULT_OUTPUT_SCALE = 0.32


def _platform(platform: Platform | str) -> Platform:
    if isinstance(platform, Platform):
        return platform
    try:
        return Platform(platform)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unknown platform {platform!r}") from exc


def _positions(value: np.ndarray | Sequence[float]) -> tuple[np.ndarray, bool, tuple[int, ...]]:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 1:
        if array.shape != (2,):
            raise ValueError(f"a relative position must have shape (2,), got {array.shape}")
        batch, single = array[None, :], True
    elif array.ndim >= 2 and array.shape[-1] == 2:
        batch, single = array.reshape(-1, 2), False
    else:
        raise ValueError("relative positions must have shape (2,) or (..., 2)")
    if not np.all(np.isfinite(batch)):
        raise ValueError("relative positions must be finite")
    return batch, single, array.shape


def _radius(value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("radius must be finite and positive")
    return value


def _gain(value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("gain must be finite and nonnegative")
    return value


def interaction_kernel(
    relative_positions: np.ndarray | Sequence[float], *, radius: float = PAIR_RADIUS
) -> np.ndarray:
    """Return ``max(0, 1 - ||relative_position/radius||^2)^2`` exactly."""

    radius = _radius(radius)
    batch, single, _ = _positions(relative_positions)
    normalized = batch / radius
    squared_norm = np.einsum("ij,ij->i", normalized, normalized)
    values = np.zeros(len(batch), dtype=np.float64)
    inside = squared_norm < 1.0
    values[inside] = (1.0 - squared_norm[inside]) ** 2
    return values[0] if single else values


def _true_pair(
    platform: Platform, relative_positions: np.ndarray, radius: float
) -> np.ndarray:
    """Evaluate the ungained smooth synthetic pair field for a batch."""

    q = relative_positions / radius
    x, y = q[:, 0], q[:, 1]
    squared_norm = x * x + y * y
    gate = np.zeros(len(q), dtype=np.float64)
    inside = squared_norm < 1.0
    gate[inside] = (1.0 - squared_norm[inside]) ** 2

    # Even terms in the disturbance field add for a symmetric pair of neighbors.
    if platform is Platform.UGV:
        longitudinal, lateral, shear_sign = 1.00, 1.00, 1.00
    else:
        longitudinal, lateral, shear_sign = 0.86, 1.08, -0.82
    x_component = longitudinal * (0.58 * x + 0.20 * (1.0 - y * y) + 0.09 * x * y)
    y_component = lateral * (
        0.40 * y + 0.26 * shear_sign * (1.0 - 2.0 * y * y) + 0.14 * x
    )
    return TRUE_INTERACTION_AMPLITUDE * gate[:, None] * np.column_stack(
        (x_component, y_component)
    )


def true_pair_field(
    platform: Platform | str,
    relative_positions: np.ndarray | Sequence[float],
    *,
    radius: float = PAIR_RADIUS,
    gain: float = 1.0,
) -> np.ndarray:
    """Evaluate one peer's true acceleration contribution."""

    class_id = _platform(platform)
    radius = _radius(radius)
    gain = _gain(gain)
    batch, single, shape = _positions(relative_positions)
    values = gain * _true_pair(class_id, batch, radius)
    if single:
        return values[0].copy()
    return values.reshape(*shape[:-1], 2)


def _frozen(value: np.ndarray) -> np.ndarray:
    result = np.frombuffer(np.asarray(value, dtype=np.float64).tobytes(), dtype=np.float64)
    result = result.reshape(np.asarray(value).shape)
    result.setflags(write=False)
    return result


def _raw_forward(
    positions: np.ndarray,
    weights: tuple[np.ndarray, ...],
    biases: tuple[np.ndarray, ...],
    output_scale: float,
    radius: float,
) -> tuple[np.ndarray, tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    activations = positions / radius
    inputs = [activations]
    preactivations = []
    for index, (weight, bias) in enumerate(zip(weights, biases, strict=True)):
        preactivation = activations @ weight.T + bias
        preactivations.append(preactivation)
        activations = output_scale * np.tanh(preactivation) if index else np.tanh(preactivation)
        if index == 0:
            inputs.append(activations)
    return activations, tuple(preactivations), tuple(inputs)


@dataclass(frozen=True)
class NeighborInteractionModel:
    """Frozen tanh network with two inputs, 16 hidden units, and two outputs for each platform."""

    platform: Platform
    weights: tuple[np.ndarray, ...]
    biases: tuple[np.ndarray, ...]
    report: dict[str, object]

    def __post_init__(self) -> None:
        class_id = _platform(self.platform)
        if tuple(self.weights[0].shape) != (16, 2) or tuple(self.weights[1].shape) != (2, 16):
            raise ValueError("neighbor predictor architecture must be 2 -> 16 -> 2")
        object.__setattr__(self, "platform", class_id)
        object.__setattr__(self, "weights", tuple(_frozen(value) for value in self.weights))
        object.__setattr__(self, "biases", tuple(_frozen(value) for value in self.biases))
        object.__setattr__(self, "report", dict(self.report))

    @property
    def frozen(self) -> bool:
        return True

    @property
    def train_mse(self) -> float:
        return float(self.report["train_mse"])

    @property
    def validation_mse(self) -> float:
        return float(self.report["validation_mse"])

    def parameters(self) -> tuple[np.ndarray, ...]:
        return self.weights + self.biases

    def predict(
        self,
        relative_positions: np.ndarray | Sequence[float],
        *,
        radius: float = PAIR_RADIUS,
    ) -> np.ndarray:
        """Predict a gated pair contribution in m/s^2."""

        radius = _radius(radius)
        batch, single, shape = _positions(relative_positions)
        raw, _, _ = _raw_forward(batch, self.weights, self.biases, DEFAULT_OUTPUT_SCALE, radius)
        values = np.asarray(interaction_kernel(batch, radius=radius))[:, None] * raw
        if single:
            return values[0].copy()
        return values.reshape(*shape[:-1], 2)

    __call__ = predict

    def true_field(
        self,
        relative_positions: np.ndarray | Sequence[float],
        *,
        radius: float = PAIR_RADIUS,
        gain: float = 1.0,
    ) -> np.ndarray:
        return true_pair_field(
            self.platform, relative_positions, radius=radius, gain=gain
        )

    def residual(
        self,
        relative_positions: np.ndarray | Sequence[float],
        *,
        radius: float = PAIR_RADIUS,
        gain: float = 1.0,
    ) -> np.ndarray:
        return self.true_field(relative_positions, radius=radius, gain=gain) - self.predict(
            relative_positions, radius=radius
        )
