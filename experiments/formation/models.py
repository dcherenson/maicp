"""Frozen models of acceleration as a function of position."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from maicp.configuration import COLLISION_ERROR_FIELD_FREQUENCY_SCALE, Platform

# Normalize network inputs using the size of the mission region.
POSITION_SCALE = 48.0
POSITION_DIMENSION = 2
DISTURBANCE_DIMENSION = 2
FIELD_GAIN = 0.3


def _as_platform(platform: Platform | str) -> Platform:
    """Normalize a platform enum or its serialized value."""

    if isinstance(platform, Platform):
        return platform
    try:
        return Platform(platform)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in Platform)
        raise ValueError(f"unknown platform {platform!r}; expected one of {allowed}") from exc


def _positions(position: np.ndarray | Sequence[float]) -> tuple[np.ndarray, bool]:
    """Return a floating point position batch and whether the input was one point."""

    values = np.asarray(position, dtype=np.float64)
    if values.ndim == 1:
        if values.shape != (POSITION_DIMENSION,):
            raise ValueError(
                f"a single position must have shape ({POSITION_DIMENSION},), got {values.shape}"
            )
        return values[None, :], True
    if values.ndim >= 2 and values.shape[-1] == POSITION_DIMENSION:
        return values.reshape(-1, POSITION_DIMENSION), False
    raise ValueError(
        "positions must have shape (2,) or (..., 2); "
        f"got {values.shape}"
    )


@dataclass(frozen=True)
class ErrorBounds:
    """A conservative magnitude/Lipschitz envelope for a vector field.

    ``M`` bounds the Euclidean norm of the field and ``L`` bounds its global
    Euclidean Lipschitz constant.  The bounds can also be unpacked as ``M, L``.
    """

    M: float
    L: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.M) or not np.isfinite(self.L):
            raise ValueError("error-field bounds must be finite")
        if self.M < 0.0 or self.L < 0.0:
            raise ValueError("error-field bounds must be nonnegative")

    @property
    def magnitude(self) -> float:
        return self.M

    @property
    def lipschitz(self) -> float:
        return self.L

    @property
    def magnitude_bound(self) -> float:
        return self.M

    @property
    def lipschitz_bound(self) -> float:
        return self.L

    def __iter__(self):
        yield self.M
        yield self.L


@dataclass(frozen=True)
class ModelReport:
    """Reproducibility and fit report for one frozen class model."""

    platform: Platform
    train_seed: int
    validation_seed: int
    initialization_seed: int
    train_samples: int
    validation_samples: int
    hidden_layers: tuple[int, ...]
    train_mse: float
    validation_mse: float
    spectral_norms: tuple[float, ...]
    prediction_lipschitz: float
    true_field_bounds: ErrorBounds
    error_bounds: ErrorBounds
    output_bound: float
    frequency_scale: float = COLLISION_ERROR_FIELD_FREQUENCY_SCALE
    certificate: bool = False

    @property
    def val_mse(self) -> float:
        """Short alias used by experiment summaries."""

        return self.validation_mse

    @property
    def lipschitz_bound(self) -> float:
        return self.prediction_lipschitz

    @property
    def spectral_lipschitz_bound(self) -> float:
        return self.prediction_lipschitz

    @property
    def M(self) -> float:
        return self.error_bounds.M

    @property
    def L(self) -> float:
        return self.error_bounds.L

    @property
    def error_M(self) -> float:
        return self.error_bounds.M

    @property
    def error_L(self) -> float:
        return self.error_bounds.L

    @property
    def error_lipschitz_bound(self) -> float:
        return self.error_bounds.L


@dataclass(frozen=True)
class _FieldSpec:
    """Coefficients used both to evaluate a field and to bound its derivative."""

    # For each output component, terms are (coefficient, frequency along x,
    # frequency along y, phase_kind), where phase_kind is ``sin`` or ``cos`` and
    # frequencies may be zero for an term along one axis.
    terms: tuple[tuple[tuple[float, float, float, str], ...], ...]

    def scaled(self, frequency_scale: float) -> _FieldSpec:
        """Scale every phase frequency, never a coefficient or FIELD_GAIN."""
        if not np.isfinite(frequency_scale) or frequency_scale <= 0.0:
            raise ValueError("frequency_scale must be finite and positive")
        if frequency_scale == 1.0:
            return self
        return _FieldSpec(
            tuple(
                tuple((a, frequency_scale * fx, frequency_scale * fy, kind)
                      for a, fx, fy, kind in component)
                for component in self.terms
            )
        )

    def value(self, positions: np.ndarray) -> np.ndarray:
        outputs = np.zeros((len(positions), DISTURBANCE_DIMENSION), dtype=np.float64)
        x = positions[:, 0]
        y = positions[:, 1]
        for output_index, component_terms in enumerate(self.terms):
            for coefficient, x_frequency, y_frequency, phase_kind in component_terms:
                phase = x_frequency * x + y_frequency * y
                if phase_kind == "sin":
                    outputs[:, output_index] += coefficient * np.sin(phase)
                else:
                    outputs[:, output_index] += coefficient * np.cos(phase)
        return FIELD_GAIN * outputs

    def bounds(self) -> ErrorBounds:
        component_bounds = []
        derivative_bounds = np.zeros((DISTURBANCE_DIMENSION, POSITION_DIMENSION))
        for output_index, component_terms in enumerate(self.terms):
            component_bound = 0.0
            for coefficient, x_frequency, y_frequency, _ in component_terms:
                component_bound += abs(coefficient)
                derivative_bounds[output_index, 0] += abs(coefficient * x_frequency)
                derivative_bounds[output_index, 1] += abs(coefficient * y_frequency)
            component_bounds.append(component_bound)

        # Frobenius norm is a simple rigorous upper bound on the operator norm
        # of every Jacobian whose entries are bounded by derivative_bounds.
        return ErrorBounds(
            M=float(FIELD_GAIN * np.linalg.norm(np.asarray(component_bounds), ord=2)),
            L=float(FIELD_GAIN * np.linalg.norm(derivative_bounds, ord="fro")),
        )


_FIELD_SPECS: Mapping[Platform, _FieldSpec] = MappingProxyType(
    {
        Platform.UGV: _FieldSpec(
            terms=(
                (
                    (0.18, 0.09, 0.0, "sin"),
                    (0.07, 0.0, 0.13, "cos"),
                    (0.035, 0.04, 0.04, "sin"),
                ),
                (
                    (-0.16, 0.11, 0.0, "cos"),
                    (0.08, 0.0, 0.07, "sin"),
                    (0.025, 0.05, -0.05, "cos"),
                ),
            )
        ),
        Platform.UAV: _FieldSpec(
            terms=(
                (
                    (0.11, 0.06, 0.0, "cos"),
                    (-0.14, 0.0, 0.10, "sin"),
                    (0.04, 0.03, -0.03, "cos"),
                ),
                (
                    (0.19, 0.05, 0.0, "sin"),
                    (0.055, 0.0, 0.12, "cos"),
                    (-0.03, 0.045, 0.045, "sin"),
                ),
            )
        ),
    }
)


def true_error(
    platform: Platform | str, position: np.ndarray | Sequence[float],
    *, frequency_scale: float = COLLISION_ERROR_FIELD_FREQUENCY_SCALE,
) -> np.ndarray:
    """Evaluate the true disturbance field shared by all robots in the class.

    The output has two components (the planar disturbance).  A point input of
    shape ``(2,)`` returns ``(2,)``. A batch or arbitrary leading shape
    ``(..., 2)`` returns the corresponding ``(..., 2)`` output.
    """

    class_id = _as_platform(platform)
    batch, single = _positions(position)
    values = _FIELD_SPECS[class_id].scaled(frequency_scale).value(batch)
    if single:
        return values[0].copy()
    original_shape = np.asarray(position).shape
    return values.reshape(*original_shape[:-1], DISTURBANCE_DIMENSION)


true_disturbance = true_error
true_field = true_error


def true_error_bounds(
    platform: Platform | str, *, frequency_scale: float = COLLISION_ERROR_FIELD_FREQUENCY_SCALE,
) -> ErrorBounds:
    """Return an analytic global envelope for the synthetic true field."""

    return _FIELD_SPECS[_as_platform(platform)].scaled(frequency_scale).bounds()


def _frozen_array(value: np.ndarray) -> np.ndarray:
    """Store an array on an immutable bytes buffer.

    ``setflags(write=False)`` alone can be reversed by a caller when an array
    owns its data.  A view of an immutable byte buffer keeps the weights immutable even
    when a caller attempts to enable writing.
    """

    array = np.asarray(value, dtype=np.float64)
    frozen = np.frombuffer(array.tobytes(order="C"), dtype=np.float64).reshape(array.shape)
    frozen.setflags(write=False)
    return frozen


def _forward(
    positions: np.ndarray,
    weights: tuple[np.ndarray, ...],
    biases: tuple[np.ndarray, ...],
    output_scale: float,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Return the output and the inputs to each activation function."""

    activations = positions / POSITION_SCALE
    preactivations: list[np.ndarray] = []
    for index, (weight, bias) in enumerate(zip(weights, biases, strict=True)):
        preactivation = activations @ weight.T + bias
        preactivations.append(preactivation)
        if index == len(weights) - 1:
            activations = output_scale * np.tanh(preactivation)
        else:
            activations = np.tanh(preactivation)
    return activations, tuple(preactivations)


@dataclass(frozen=True)
class LearnedDynamicsModel:
    """Immutable tanh model for disturbances in one class."""

    platform: Platform
    weights: tuple[np.ndarray, ...]
    biases: tuple[np.ndarray, ...]
    output_scale: float
    report: ModelReport

    def __post_init__(self) -> None:
        if len(self.weights) != len(self.biases) or not self.weights:
            raise ValueError("a model needs matching nonempty weight and bias layers")
        if self.output_scale <= 0.0 or not np.isfinite(self.output_scale):
            raise ValueError("output_scale must be finite and positive")
        frozen_weights = []
        frozen_biases = []
        for weight, bias in zip(self.weights, self.biases, strict=True):
            frozen_weights.append(_frozen_array(weight))
            frozen_biases.append(_frozen_array(bias))
        object.__setattr__(self, "weights", tuple(frozen_weights))
        object.__setattr__(self, "biases", tuple(frozen_biases))

    @property
    def frozen(self) -> bool:
        return True

    @property
    def train_mse(self) -> float:
        return self.report.train_mse

    @property
    def validation_mse(self) -> float:
        return self.report.validation_mse

    @property
    def spectral_lipschitz_bound(self) -> float:
        return self.report.prediction_lipschitz

    @property
    def lipschitz_bound(self) -> float:
        return self.spectral_lipschitz_bound

    @property
    def spectral_norm_lipschitz(self) -> float:
        return self.spectral_lipschitz_bound

    @property
    def error_bounds(self) -> ErrorBounds:
        return self.report.error_bounds

    def predict(self, position: np.ndarray | Sequence[float]) -> np.ndarray:
        """Predict the learned disturbance at one point or a position batch."""

        batch, single = _positions(position)
        prediction, _ = _forward(batch, self.weights, self.biases, self.output_scale)
        if single:
            return prediction[0].copy()
        original_shape = np.asarray(position).shape
        return prediction.reshape(*original_shape[:-1], DISTURBANCE_DIMENSION)

    __call__ = predict

    def true_error(self, position: np.ndarray | Sequence[float]) -> np.ndarray:
        return true_error(self.platform, position, frequency_scale=self.report.frequency_scale)

    def residual(self, position: np.ndarray | Sequence[float]) -> np.ndarray:
        """Return the synthetic model error, true field minus prediction."""

        return self.true_error(position) - self.predict(position)

    def bounds(self) -> ErrorBounds:
        return self.report.error_bounds

    @property
    def M(self) -> float:
        return self.report.error_bounds.M

    @property
    def L(self) -> float:
        return self.report.error_bounds.L

    def parameters(self) -> tuple[np.ndarray, ...]:
        """Return the weights and biases in layer order as immutable arrays."""

        return self.weights + self.biases

    def state_dict(self) -> Mapping[str, np.ndarray]:
        """Return an immutable snapshot with named parameters."""

        values: dict[str, np.ndarray] = {}
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            values[f"layers.{index}.weight"] = weight
            values[f"layers.{index}.bias"] = bias
        return MappingProxyType(values)


@dataclass(frozen=True)
class LearnedDynamicsBundle:
    """The two frozen class models and their common integration surface."""

    models: Mapping[Platform, LearnedDynamicsModel]
    seed: int

    def __post_init__(self) -> None:
        normalized = {_as_platform(key): value for key, value in self.models.items()}
        if set(normalized) != set(Platform):
            raise ValueError("a learned dynamics bundle must contain one model per Platform class")
        if any(model.platform is not key for key, model in normalized.items()):
            raise ValueError("model keys and model platform values must agree")
        object.__setattr__(self, "models", MappingProxyType(normalized))

    def model(self, platform: Platform | str) -> LearnedDynamicsModel:
        return self.models[_as_platform(platform)]

    def true_error(
        self, platform: Platform | str, position: np.ndarray | Sequence[float]
    ) -> np.ndarray:
        return self.model(platform).true_error(position)

    def predict(
        self, platform: Platform | str, position: np.ndarray | Sequence[float]
    ) -> np.ndarray:
        return self.model(platform).predict(position)

    predict_disturbance = predict

    def residual(
        self, platform: Platform | str, position: np.ndarray | Sequence[float]
    ) -> np.ndarray:
        return self.model(platform).residual(position)

    error = residual

    def bounds(self, platform: Platform | str) -> ErrorBounds:
        return self.model(platform).bounds()

    error_bounds = bounds

    def report(self, platform: Platform | str) -> ModelReport:
        return self.model(platform).report

    @property
    def reports(self) -> Mapping[Platform, ModelReport]:
        return MappingProxyType({platform: model.report for platform, model in self.models.items()})



def load_models(path: Path):
    """Load a serialized frozen Wang model payload without fitting models."""

    from experiments.formation.interactions import NeighborInteractionModel

    document = json.loads(path.read_text())
    if document.get("schema_version") != 1:
        raise ValueError(f"Unsupported collision model artifact schema: {path}")
    positions = {}
    for entry in document.get("position_models", []):
        platform = Platform(entry["platform"])
        report_data = dict(entry["report"])
        report_data["platform"] = platform
        for key in ("true_field_bounds", "error_bounds"):
            report_data[key] = ErrorBounds(**report_data[key])
        report = ModelReport(**report_data)
        model = LearnedDynamicsModel(
            platform=platform,
            weights=tuple(np.asarray(value, dtype=float) for value in entry["weights"]),
            biases=tuple(np.asarray(value, dtype=float) for value in entry["biases"]),
            output_scale=float(entry["output_scale"]),
            report=report,
        )
        positions[platform] = model
    neighbors = {}
    for entry in document.get("neighbor_models", []):
        platform = Platform(entry["platform"])
        model = NeighborInteractionModel(
            platform,
            tuple(np.asarray(value, dtype=float) for value in entry["weights"]),
            tuple(np.asarray(value, dtype=float) for value in entry["biases"]),
            dict(entry["report"]),
        )
        neighbors[platform] = model
    return LearnedDynamicsBundle(positions, seed=int(document["seed"])), neighbors
