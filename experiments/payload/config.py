from __future__ import annotations
from dataclasses import asdict, dataclass
import numpy as np


@dataclass(frozen=True)
class ModelConfig:
    classes: tuple[int, ...] = (0, 1, 0, 1, 0, 1)
    capability: tuple[float, ...] = (1.08, 2.34, 1.20, 2.60, 1.32, 2.86)
    mu: tuple[float, ...] = (0.10, 0.18, 0.10, 0.18, 0.10, 0.18)
    edges: tuple[tuple[int, int], ...] = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 0),
    )
    gamma: float = 0.10
    rounds: int = 5
    margin_cap: tuple[float, float] = (2.0, 2.0)
    force_center: float = 21.0
    force_amplitude: float = 1.2
    dt: float = 0.1
    steps: int = 100
    dynamic: bool = False
    metric: str = "lyapunov"
    # Position dependence of the local cost weights: exp(sigma*tanh(e/s)).
    sigma: tuple[float, ...] = (0.08, 0.08, 0.08, 0.08, 0.08, 0.08)
    position_scale: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)  # metres
    mass: tuple[float, ...] = (1.0, 1.8, 1.1, 2.0, 1.2, 2.2)
    kp: tuple[float, ...] = (8.0, 12.0, 9.0, 13.0, 10.0, 14.0)  # N/m
    kd: tuple[float, ...] = (5.0, 8.0, 5.5, 8.5, 6.0, 9.0)  # Ns/m
    theta_bound: tuple[float, float] = (0.08, 0.25)
    phi_bound: tuple[float, float] = (0.03, 0.08)
    # Amplitude of the force residual as a function of position.
    position_residual_bound: float = 0.08  # N
    eta_shared: float = 0.018
    eta_robot: float = 0.010
    eta_step: float = 0.006
    command_cap: tuple[float, float] = (4.5, 8.0)
    actual_force_cap: tuple[float, float] = (5.5, 10.0)
    initial_position_support: float = 0.05  # m, both signs
    initial_velocity_support: float = 0.02  # m/s, both signs

    def __post_init__(self):
        n = len(self.classes)
        if (
            n != 6
            or tuple(self.classes).count(0) != 3
            or tuple(self.classes).count(1) != 3
        ):
            raise ValueError(
                "This calibration experiment has six nodes and three per class"
            )
        for name in ("capability", "mu", "position_scale", "mass", "kp", "kd"):
            v = np.asarray(getattr(self, name))
            if v.shape != (n,) or not np.all(np.isfinite(v)) or np.any(v <= 0):
                raise ValueError(f"Invalid {name}")
        if len(self.sigma) != n or min(self.sigma) < 0:
            raise ValueError("Invalid sigma")
        if self.rounds < 1 or self.steps < 1 or self.gamma <= 0 or self.dt <= 0:
            raise ValueError("Invalid time/rounds")
        if self.force_center <= self.force_amplitude or self.force_amplitude < 0:
            raise ValueError("Invalid force demand")
        for name in ("eta_shared", "eta_robot", "eta_step", "position_residual_bound"):
            if getattr(self, name) < 0:
                raise ValueError(f"Invalid {name}")

    @property
    def n(self):
        return len(self.classes)

    @property
    def force_max(self):
        return self.force_center + self.force_amplitude

    @property
    def force_min(self):
        return self.force_center - self.force_amplitude

    @property
    def eta_bound(self):
        return self.eta_shared + self.eta_robot + self.eta_step

    @property
    def horizon(self):
        return self.steps * self.dt

    def dictionary(self):
        return asdict(self)


@dataclass(frozen=True)
class ExperimentConfig:
    replicates: int = 8
    updates: int = 5
    calibration_missions: int = 500
    evaluation_missions: int = 1500
    alpha: tuple[float, float] = (0.1, 0.1)
    delta_cal: tuple[float, float] = (0.05, 0.05)
    initial_margin: tuple[float, float] = (1.5, 1.5)
    objective_weights: tuple[float, float] = (1.0, 1.0)
    proximal_weight: float = 1.0
    pinball_iterations: int = 100
    pinball_scale: float = 2.0
    seed: int = 730142

    def dictionary(self):
        return asdict(self)
