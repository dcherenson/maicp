"""Sample initial conditions, process noise, and disturbances near obstacles."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations
from types import MappingProxyType

import numpy as np

from experiments.mission import (
    CanonicalMissionFrame,
    canonical_mission_frame,
)
from maicp.configuration import ExperimentConfig, Platform
from maicp.types import MissionPlan


def _keyed_rng(seed: int, *keys: object) -> np.random.Generator:
    """Create a random stream from the supplied key."""

    payload = "\x1f".join(str(value) for value in (seed, *keys)).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=16).digest()
    return np.random.default_rng(int.from_bytes(digest, "little") % (2**63 - 1))


def collision_mission_seed(plan: MissionPlan) -> int:
    """Derive the collision realization seed from its external mission identity."""

    payload = (
        f"collision-mission\x1f{plan.seed}\x1f{plan.mission_id}\x1f{plan.subteam_id}"
    ).encode("utf-8")
    return int.from_bytes(
        hashlib.blake2b(payload, digest_size=8).digest(), "little"
    ) % (2**63 - 1)


def _frozen_vector(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    frozen = np.frombuffer(array.tobytes(order="C"), dtype=np.float64).reshape(
        array.shape
    )
    frozen.setflags(write=False)
    return frozen


def _uniform_disk_sample(seed: int, *keys: object, radius: float) -> np.ndarray:
    rng = _keyed_rng(seed, *keys)
    radial = float(radius) * np.sqrt(float(rng.random()))
    angle = 2.0 * np.pi * float(rng.random())
    return radial * np.array([np.cos(angle), np.sin(angle)])


def _truncated_gaussian_sample(
    seed: int,
    *keys: object,
    sigma: float,
    maximum: float,
) -> np.ndarray:
    """Draw exactly from an isotropic Gaussian conditioned on a norm ball."""

    if sigma == 0.0:
        return np.zeros(2)
    # Give each rejection attempt its own random seed.
    for attempt in range(10_000):
        candidate = _keyed_rng(seed, *keys, "rejection", attempt).normal(
            0.0, sigma, size=2
        )
        if float(np.linalg.norm(candidate)) <= maximum:
            return np.asarray(candidate, dtype=float)
    raise RuntimeError("Truncated-Gaussian rejection sampler did not accept a draw")


def _class_noise_parameters(
    config: ExperimentConfig, platform: Platform
) -> tuple[float, float]:
    if platform is Platform.UGV:
        return (
            config.collision_motion_noise_sigma_ugv,
            config.collision_motion_noise_max_ugv,
        )
    return (
        config.collision_motion_noise_sigma_uav,
        config.collision_motion_noise_max_uav,
    )


@dataclass(frozen=True)
class CollisionMissionRealization:
    """Random inputs shared across margin settings for one collision mission."""

    mission_seed: int
    frame: CanonicalMissionFrame
    formation_phase_rad: float
    formation_offsets_canonical: Mapping[str, np.ndarray]
    initial_perturbations: Mapping[str, np.ndarray]
    local_initial_perturbations: Mapping[str, np.ndarray]
    motion_noise: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        phase = float(self.formation_phase_rad)
        if not np.isfinite(phase) or not 0.0 <= phase < 2.0 * np.pi:
            raise ValueError("formation_phase_rad must be finite and lie in [0, 2*pi)")
        object.__setattr__(self, "formation_phase_rad", phase)
        object.__setattr__(
            self,
            "formation_offsets_canonical",
            MappingProxyType(
                {
                    robot_id: _frozen_vector(value)
                    for robot_id, value in self.formation_offsets_canonical.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "initial_perturbations",
            MappingProxyType(
                {
                    robot_id: _frozen_vector(value)
                    for robot_id, value in self.initial_perturbations.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "local_initial_perturbations",
            MappingProxyType(
                {
                    robot_id: _frozen_vector(value)
                    for robot_id, value in self.local_initial_perturbations.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "motion_noise",
            MappingProxyType(
                {
                    robot_id: _frozen_vector(value)
                    for robot_id, value in self.motion_noise.items()
                }
            ),
        )

    @property
    def digest(self) -> str:
        """Hash the sampled inputs so paired missions can be compared."""

        value = hashlib.blake2b(digest_size=16)
        value.update(str(self.mission_seed).encode("ascii"))
        if self.formation_offsets_canonical:
            value.update(
                np.asarray([self.formation_phase_rad], dtype=np.float64).tobytes()
            )
            for robot_id in sorted(self.formation_offsets_canonical):
                value.update(robot_id.encode("utf-8"))
                value.update(
                    np.asarray(
                        self.formation_offsets_canonical[robot_id], dtype=np.float64
                    ).tobytes()
                )
        for collection in (self.initial_perturbations, self.motion_noise):
            for robot_id in sorted(collection):
                value.update(robot_id.encode("utf-8"))
                value.update(
                    np.asarray(collection[robot_id], dtype=np.float64).tobytes()
                )
        return value.hexdigest()


def generate_collision_mission_realization(
    config: ExperimentConfig,
    plan: MissionPlan,
    *,
    physical_agent_radius: float = 0.12,
    formation_phase_rad: float | None = None,
) -> CollisionMissionRealization:
    """Sample the complete disturbance realization from the mission seed and local slots."""

    frame = canonical_mission_frame(plan.subteam_id)
    mission_seed = collision_mission_seed(plan)
    controller = getattr(config, "collision_controller", None)
    controller_name = getattr(controller, "value", controller)
    is_formation = (
        getattr(config, "collision_scenario", None) == "formation_obstacles_v1"
        and controller_name in {"wang", "hocbf"}
        and all(len(plan.robot_ids[platform]) == 3 for platform in Platform)
    )
    if is_formation:
        from experiments.formation.geometry import offsets, sample_formation_phase

        phase = (
            sample_formation_phase(plan)
            if formation_phase_rad is None
            else float(formation_phase_rad)
        )
        canonical_offsets = offsets(
            plan,
            config,
            formation_phase_rad=phase,
        )
    else:
        # The realization container is shared with the generic collision
        # baselines, which have no formation slots to rotate.
        phase = 0.0
        canonical_offsets = {}
    slot_keys = {
        robot_id: (platform.value, slot)
        for platform in Platform
        for slot, robot_id in enumerate(plan.robot_ids[platform])
    }
    radius = float(config.collision_initial_position_radius)
    # Check initial clearance after sampling all position perturbations.
    local_perturbations = {
        robot_id: _uniform_disk_sample(
            mission_seed,
            "collision-initial-position",
            platform,
            slot,
            radius=radius,
        )
        for robot_id, (platform, slot) in slot_keys.items()
    }
    global_perturbations = {
        robot_id: frame.vector_to_global(value)
        for robot_id, value in local_perturbations.items()
    }
    positions = {
        robot_id: np.asarray(plan.home_positions[robot_id], dtype=float)
        + global_perturbations[robot_id]
        for robot_id in plan.home_positions
    }
    if any(
        float(np.linalg.norm(positions[left] - positions[right]))
        <= 2.0 * physical_agent_radius
        for left, right in same_layer_safety_edges(plan)
    ):
        raise RuntimeError("Sampled an initially colliding physical configuration")

    noise: dict[str, np.ndarray] = {}
    for robot_id, (platform_name, slot) in slot_keys.items():
        platform = Platform(platform_name)
        sigma, maximum = _class_noise_parameters(config, platform)
        local_sequence = np.asarray(
            [
                _truncated_gaussian_sample(
                    mission_seed,
                    "collision-motion-noise",
                    platform.value,
                    slot,
                    step_index,
                    sigma=sigma,
                    maximum=maximum,
                )
                for step_index in range(config.profile.simulation_steps)
            ]
        )
        noise[robot_id] = np.asarray(
            [frame.vector_to_global(sample) for sample in local_sequence]
        )
    return CollisionMissionRealization(
        mission_seed=mission_seed,
        frame=frame,
        formation_phase_rad=phase,
        formation_offsets_canonical=canonical_offsets,
        initial_perturbations=global_perturbations,
        local_initial_perturbations=local_perturbations,
        motion_noise=noise,
    )


def evaluate_canonical_class_field(
    models,
    platform: Platform,
    global_position: np.ndarray,
    frame: CanonicalMissionFrame,
    *,
    learned: bool,
) -> np.ndarray:
    """Evaluate a class field locally and transform its vector back globally."""

    local_position = frame.point_to_local(global_position)
    local_value = (
        models.predict(platform, local_position)
        if learned
        else models.true_error(platform, local_position)
    )
    return frame.vector_to_global(local_value)


def same_layer_safety_edges(plan: MissionPlan) -> tuple[tuple[str, str], ...]:
    """Every pair of robots in the same class shares a fixed safety edge."""

    return tuple(
        sorted(
            tuple(sorted(edge))
            for platform in Platform
            for edge in combinations(plan.robot_ids[platform], 2)
        )
    )



def obstacle_encounter_acceleration(position, peer_positions, obstacle_centers,
                                    footprint_radii, *, amplitude, band,
                                    peer_radius, strength, mode="density"):
    """Return the bounded inward vector and the peer statistic for the selected mode.

    ``mode="density"`` uses the normalized density of the other class.  In ``peer_encounter`` mode, each peer gets a smooth
    union of its gates for nearby obstacles and the largest peer exposure drives
    the bounded modulation ``0.1 + 0.9 * exposure``.
    """
    for name, value in (("amplitude", amplitude), ("band", band),
                        ("peer_radius", peer_radius), ("strength", strength)):
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if amplitude < 0 or band <= 0 or peer_radius <= 0 or not 0 <= strength <= 1:
        raise ValueError("Invalid obstacle encounter parameters")
    if mode not in {"density", "peer_encounter"}:
        raise ValueError("mode must be 'density' or 'peer_encounter'")
    p = np.asarray(position, dtype=float)
    centers = np.asarray(obstacle_centers, dtype=float).reshape(-1, 2)
    radii = np.asarray(footprint_radii, dtype=float).reshape(-1)
    peers = np.asarray(peer_positions, dtype=float).reshape(-1, 2)
    if (p.shape != (2,) or len(centers) != len(radii) or np.any(radii < 0)
            or not all(np.all(np.isfinite(x)) for x in (p, centers, radii, peers))):
        raise ValueError("Invalid obstacle encounter geometry")
    density = float(np.mean(np.maximum(0., 1. - np.sum((peers-p)**2, axis=1)
                                      / peer_radius**2)**2)) if len(peers) else 0.
    if mode == "peer_encounter":
        if len(peers) and len(centers):
            peer_exposures = []
            for peer in peers:
                peer_toward = centers - peer
                peer_distance = np.linalg.norm(peer_toward, axis=1)
                peer_x = np.clip(1. - (peer_distance-radii) / band, 0., 1.)
                peer_gate = peer_x**3 * (10. - 15.*peer_x + 6.*peer_x*peer_x)
                # Smooth union over the shared obstacle catalog.  The union
                # and max operations are both bounded in [0, 1].
                peer_exposures.append(
                    float(np.clip(1. - float(np.prod(1. - peer_gate)), 0.0, 1.0))
                )
            exposure = float(max(peer_exposures))
        else:
            exposure = 0.
    else:
        exposure = density
    if amplitude == 0 or not len(centers):
        return np.zeros(2), density if mode == "density" else exposure
    toward = centers - p
    distance = np.linalg.norm(toward, axis=1)
    x = np.clip(1. - (distance-radii)/band, 0., 1.)
    gate = x**3 * (10. - 15.*x + 6.*x*x)
    # Smooth regularization at obstacle centers and vanishing total gate.
    directions = toward / np.sqrt(distance[:, None]**2 + 1e-12)
    union_gate = 1. - float(np.prod(1. - gate))
    direction = np.sum(gate[:, None]*directions, axis=0) / (1e-3 + np.sum(gate))
    if mode == "density":
        field = amplitude * strength * (0.5 + 0.5*density) * union_gate * direction
        return field, density
    field = amplitude * strength * (0.1 + 0.9*exposure) * union_gate * direction
    return field, exposure
