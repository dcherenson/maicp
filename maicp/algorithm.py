"""One round of MA-ICP calibration and margin updates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from maicp.ranks import binomial_calibration_rank

import numpy as np

from maicp.configuration import ExperimentConfig, Platform, SensitivityMode

from maicp.score_stores import LocalCalibrationStores
from .distributed import certified_order_statistic, reachable_nodes
from .sensitivity import build_sensitivity_result
from .types import MAICPRoundResult, MissionResult, SensitivityResult
from .update import hold_margin_update, solve_margin_update


def centralized_mission_score(scores: Sequence[float], rank: int) -> float:
    if not 1 <= rank <= len(scores):
        raise ValueError("Invalid mission rank")
    values = tuple(float(score) for score in scores)
    if any(not np.isfinite(score) for score in values):
        raise ValueError("Mission scores must be finite")
    return float(sorted(values)[rank - 1])


def _diagnostic_mission_score(scores: Sequence[float], rank: int) -> float:
    """Keep invalid data visible in reports without aborting the held round."""

    try:
        return centralized_mission_score(scores, rank)
    except (TypeError, ValueError, OverflowError):
        return float("nan")


def eta_owner(mission_index: int, nodes: Sequence[str]) -> str:
    """Return the owner ``eta_j`` of the mission certificate."""

    ordered = tuple(sorted(nodes))
    if not ordered:
        raise ValueError("At least one update node is required")
    if mission_index < 0:
        raise ValueError("Mission index must be nonnegative")
    return ordered[mission_index % len(ordered)]


def _validate_local_values(
    local: Mapping[str, Mapping[str, float]],
    expected_count: int,
    mission_id: int,
    platform: Platform,
) -> tuple[dict[str, dict[str, float]], bool]:
    """Validate ownership and require finite robot scores for certification."""

    tags: set[str] = set()
    count = 0
    invalid = False
    finite_local = {node: {} for node in local}
    for node, values in local.items():
        if not isinstance(values, Mapping):
            raise TypeError(f"Node {node!r} has a malformed {platform.value} score store")
        for tag, raw_value in values.items():
            if tag in tags:
                raise ValueError(
                    f"Mission {mission_id} {platform.value} score {tag!r} has multiple owners"
                )
            tags.add(tag)
            count += 1
            value = float(raw_value)
            if np.isfinite(value):
                finite_local[node][tag] = value
            else:
                invalid = True
    if count != expected_count:
        raise ValueError(
            f"Mission {mission_id} has {count} owned {platform.value} scores; "
            f"expected {expected_count}"
        )
    return finite_local, invalid


def _check_mission_certificate_store(
    stores: Mapping[str, Mapping[str, float]],
    mission_ids: Sequence[int],
    platform: Platform,
) -> None:
    expected = {f"mission:{mission_id}:{platform.value}" for mission_id in mission_ids}
    seen: dict[str, str] = {}
    for node, values in stores.items():
        for tag in values:
            if tag not in expected:
                raise ValueError(f"Unexpected {platform.value} mission certificate tag {tag!r}")
            if tag in seen:
                raise ValueError(
                    f"Mission certificate {tag!r} has multiple owners ({seen[tag]!r}, {node!r})"
                )
            seen[tag] = node
    missing = expected - set(seen)
    if missing:
        raise ValueError(f"Missing mission certificates: {sorted(missing)}")


def _centralized_sensitivity_result(
    config: ExperimentConfig,
    robot_platforms: Mapping[str, Platform],
    analytic_kappa: Mapping[str, float] | None = None,
    empirical_ratios: Mapping[str, float] | None = None,
) -> SensitivityResult:
    """Compute the class maximum of the robot sensitivity coefficients for the centralized baseline."""

    analytic_kappa = analytic_kappa or {}
    empirical_ratios = empirical_ratios or {}
    mode = config.sensitivity_mode
    robot_kappa: dict[Platform, dict[str, float]] = {platform: {} for platform in Platform}
    validation_ratio: dict[Platform, float] = {platform: 0.0 for platform in Platform}

    for robot_id, platform in robot_platforms.items():
        if mode is SensitivityMode.CONFIGURED:
            value = float(config.classes[platform].configured_kappa)
        elif mode is SensitivityMode.ANALYTIC:
            try:
                value = float(analytic_kappa.get(robot_id, float("nan")))
            except (TypeError, ValueError):
                value = float("nan")
        else:
            try:
                value = float(empirical_ratios.get(robot_id, float("nan")))
            except (TypeError, ValueError):
                value = float("nan")
        robot_kappa[platform][robot_id] = value
        if robot_id in empirical_ratios:
            try:
                ratio_value = float(empirical_ratios[robot_id])
            except (TypeError, ValueError):
                ratio_value = float("nan")
            if not np.isfinite(ratio_value):
                validation_ratio[platform] = float("inf")
            elif np.isfinite(value) and value > 0.0:
                validation_ratio[platform] = max(
                    validation_ratio[platform], ratio_value / value
                )
            elif ratio_value > 0.0:
                validation_ratio[platform] = float("inf")

    class_kappa: dict[Platform, float] = {}
    for platform in Platform:
        values = tuple(robot_kappa[platform].values())
        if not values or any(not np.isfinite(value) or value < 0.0 for value in values):
            class_kappa[platform] = float("nan")
        else:
            class_kappa[platform] = float(max(values))

    values_valid = all(np.isfinite(value) and value >= 0.0 for value in class_kappa.values())
    paired_valid = all(ratio <= 1.0 + 1e-8 for ratio in validation_ratio.values())
    notes: list[str] = []
    if not values_valid:
        notes.append("missing or invalid sensitivity coefficient")
    if not paired_valid:
        notes.append("paired validation exceeded the claimed sensitivity bound (diagnostic)")
    if mode is SensitivityMode.EMPIRICAL:
        notes.append("empirical finite differences are diagnostic, not a certified kappa")

    return SensitivityResult(
        robot_kappa=robot_kappa,
        class_kappa=class_kappa,
        mode=mode.value,
        certified=mode is not SensitivityMode.EMPIRICAL and values_valid,
        valid=values_valid,
        max_validation_ratio=validation_ratio,
        notes=tuple(notes),
        paired_validation_valid=paired_valid,
        kappa_flooding_rounds=0,
    )


class MAICP:
    def __init__(self, config: ExperimentConfig):
        config.validate()
        self.config = config

    def run_round(
        self,
        round_index: int,
        current_margins: dict[Platform, float],
        missions: Sequence[MissionResult],
        update_graph: Mapping[str, Sequence[str] | set[str]],
        robot_platforms: Mapping[str, Platform],
        analytic_kappa: Mapping[str, float],
        local_calibration: LocalCalibrationStores | None = None,
        empirical_ratios: Mapping[str, float] | None = None,
        centralized: bool = False,
        unshifted: bool = False,
        capture_diagnostics: bool = False,
    ) -> MAICPRoundResult:
        """Compute mission scores, calibration quantiles, and the next margins. Each robot supplies its own scores for distributed calibration."""

        expected = self.config.profile.calibration_missions
        missions = tuple(missions)
        if len(missions) != expected and not centralized:
            raise ValueError(f"Expected {expected} missions, received {len(missions)}")
        evaluator_invalid = False
        if not centralized and local_calibration is None:
            raise ValueError("Distributed MA-ICP requires persistent local calibration stores")

        graph_nodes = tuple(sorted(update_graph))
        if not graph_nodes:
            raise ValueError("Distributed MA-ICP requires a nonempty update graph")
        mission_ids = tuple(mission.mission_id for mission in missions)
        if len(set(mission_ids)) != len(mission_ids):
            raise ValueError("Mission IDs must be unique within a calibration round")

        exact_mission: dict[Platform, list[float]] = {platform: [] for platform in Platform}
        for mission in missions:
            for platform in Platform:
                class_config = self.config.classes[platform]
                try:
                    scores = mission.robot_scores[platform]
                    value = _diagnostic_mission_score(
                        tuple(scores.values()), class_config.mission_rank
                    )
                    valid = len(scores) == class_config.robots_per_mission and np.isfinite(value)
                except (KeyError, TypeError, AttributeError):
                    value, valid = float("nan"), False
                exact_mission[platform].append(value)
                evaluator_invalid = evaluator_invalid or not valid

        # Centralized calibration reads the evaluator scores. Distributed calibration reads the scores stored by each robot.
        if centralized and evaluator_invalid:
            return self._held_failure_round(
                round_index,
                current_margins,
                exact_mission,
                update_graph,
                robot_platforms,
                "no_certified_update:invalid_score",
            )

        if not centralized and len(reachable_nodes(update_graph, graph_nodes[0])) != len(graph_nodes):
            return self._held_failure_round(
                round_index,
                current_margins,
                exact_mission,
                update_graph,
                robot_platforms,
                "no_certified_update:update_graph_disconnected",
            )

        order_statistic_histories: dict[Platform, dict[str, np.ndarray]] = {
            platform: {} for platform in Platform
        }
        certified_mission: dict[Platform, list[float]] = {platform: [] for platform in Platform}
        pinball_rounds = 0
        flooding_rounds = 0
        mission_certificate_owners: dict[Platform, dict[int, str]] = {
            platform: {} for platform in Platform
        }

        # The centralized baseline sorts the pooled scores.
        if centralized:
            certified_mission = {
                platform: list(values) for platform, values in exact_mission.items()
            }
        else:
            assert local_calibration is not None
            local_robot_values: dict[tuple[Platform, int], dict[str, dict[str, float]]] = {}
            invalid_score = False
            for mission in missions:
                graph = mission.mission_graph
                if (not graph or not graph.keys() <= update_graph.keys()
                        or len(reachable_nodes(graph, next(iter(graph)))) != len(graph)):
                    return self._held_failure_round(
                        round_index, current_margins, exact_mission, update_graph,
                        robot_platforms, "no_certified_update:invalid_mission_graph",
                        analytic_kappa=analytic_kappa, local_calibration=local_calibration,
                        empirical_ratios=empirical_ratios,
                    )
                for platform in Platform:
                    try:
                        local = local_calibration.robot_score_stores(
                            mission.mission_id, platform, tuple(graph)
                        )
                        class_config = self.config.classes[platform]
                        finite_local, invalid = _validate_local_values(
                            local,
                            class_config.robots_per_mission,
                            mission.mission_id,
                            platform,
                        )
                        local_robot_values[(platform, mission.mission_id)] = finite_local
                        invalid_score = invalid_score or invalid
                    except (KeyError, TypeError, ValueError, AttributeError, OverflowError):
                        invalid_score = True

            if invalid_score:
                return self._held_failure_round(
                    round_index,
                    current_margins,
                    exact_mission,
                    update_graph,
                    robot_platforms,
                    "no_certified_update:invalid_score",
                    analytic_kappa=analytic_kappa,
                    local_calibration=local_calibration,
                    empirical_ratios=empirical_ratios,
                )
            for mission_index, mission in enumerate(missions):
                owner = eta_owner(mission_index, tuple(mission.mission_graph))
                for platform in Platform:
                    class_config = self.config.classes[platform]
                    result = certified_order_statistic(
                        mission.mission_graph,
                        local_robot_values[(platform, mission.mission_id)],
                        # K_c=N_c-ell_c applies to all finite robot scores.
                        rank=class_config.mission_rank,
                        iterations=self.config.profile.mission_pinball_steps,
                        q_init=current_margins[platform],
                        gamma_0=self.config.pinball_gamma_0,
                        exponent=self.config.pinball_exponent,
                    )
                    certified_mission[platform].append(float(result.value))
                    pinball_rounds += result.pinball_rounds
                    flooding_rounds += result.flooding_rounds
                    if capture_diagnostics and mission_index == 0:
                        order_statistic_histories[platform]["mission"] = result.pinball_history.copy()
                    mission_certificate_owners[platform][mission.mission_id] = owner
                    self._record_mission_certificate(
                        local_calibration,
                        mission.mission_id,
                        platform,
                        owner,
                        float(result.value),
                    )

        exact_thresholds: dict[Platform, float] = {}
        certified_thresholds: dict[Platform, float] = {}
        conformal_ranks: dict[Platform, int] = {}
        mission_count = len(missions)
        for platform in Platform:
            class_config = self.config.classes[platform]
            rank = binomial_calibration_rank(
                mission_count, class_config.alpha, class_config.delta_cal
            )
            conformal_ranks[platform] = rank
            if any(not np.isfinite(value) for value in exact_mission[platform]):
                exact_thresholds[platform] = float("nan")
            else:
                exact_thresholds[platform] = float(
                    sorted(exact_mission[platform])[rank - 1]
                )

            if centralized:
                certified_thresholds[platform] = exact_thresholds[platform]
                continue

            assert local_calibration is not None
            mission_stores = local_calibration.mission_score_stores(platform, graph_nodes)
            _check_mission_certificate_store(mission_stores, mission_ids, platform)
            result = certified_order_statistic(
                update_graph,
                mission_stores,
                rank=rank,
                iterations=self.config.profile.threshold_pinball_steps,
                q_init=current_margins[platform],
                gamma_0=self.config.pinball_gamma_0,
                exponent=self.config.pinball_exponent,
            )
            certified_thresholds[platform] = float(result.value)
            pinball_rounds += result.pinball_rounds
            flooding_rounds += result.flooding_rounds
            if capture_diagnostics:
                order_statistic_histories[platform]["threshold"] = result.pinball_history.copy()

        if centralized:
            # Use the class maximum directly for the centralized baseline.
            sensitivity = _centralized_sensitivity_result(
                self.config,
                robot_platforms,
                analytic_kappa,
                empirical_ratios,
            )
        else:
            sensitivity = build_sensitivity_result(
                self.config,
                update_graph,
                robot_platforms,
                analytic_kappa,
                empirical_ratios,
                local_calibration,
            )
        # Count exchanges for pinball iterations, order statistic recovery, and class maxima.
        kappa_flooding_rounds = sensitivity.kappa_flooding_rounds
        communication_rounds = (
            pinball_rounds + flooding_rounds + kappa_flooding_rounds
        )
        kappa = (
            {platform: 0.0 for platform in Platform}
            if unshifted
            else sensitivity.class_kappa
        )
        if (
            self.config.sensitivity_mode is not SensitivityMode.EMPIRICAL
            and not sensitivity.certified
            and not unshifted
        ):
            update = hold_margin_update(current_margins)
        else:
            update = solve_margin_update(
                current=current_margins,
                thresholds=certified_thresholds,
                kappa=kappa,
                classes=self.config.classes,
                proximal_weight=self.config.proximal_weight,
                sensitivity_certified=sensitivity.certified and not unshifted,
                update_nodes=graph_nodes if not centralized else None,
                consensus_tolerance=float(
                    getattr(self.config, "margin_consensus_tolerance", 1e-7)
                ),
            )
        return MAICPRoundResult(
            round_index=round_index,
            mission_scores={platform: tuple(values) for platform, values in exact_mission.items()},
            certified_mission_scores={
                platform: tuple(values) for platform, values in certified_mission.items()
            },
            exact_thresholds=exact_thresholds,
            certified_thresholds=certified_thresholds,
            conformal_ranks=conformal_ranks,
            sensitivity=sensitivity,
            margin_update=update,
            communication_rounds=communication_rounds,
            order_statistic_histories=order_statistic_histories,
            mission_certificate_owners=mission_certificate_owners,
            pinball_rounds=pinball_rounds,
            flooding_rounds=flooding_rounds,
            kappa_flooding_rounds=kappa_flooding_rounds,
        )

    @staticmethod
    def _record_mission_certificate(
        stores: LocalCalibrationStores,
        mission_id: int,
        platform: Platform,
        owner: str,
        score: float,
    ) -> None:
        tag = f"mission:{mission_id}:{platform.value}"
        nodes = tuple(sorted(stores.node_stores))
        existing = [
            (node, float(values[tag]))
            for node, values in stores.mission_score_stores(platform, nodes).items()
            if tag in values
        ]
        if existing:
            if len(existing) != 1 or existing[0][0] != owner:
                raise ValueError(f"Mission certificate {tag!r} is not uniquely owned by {owner!r}")
            if not np.isclose(existing[0][1], score, rtol=0.0, atol=1e-12):
                raise ValueError(f"Mission certificate {tag!r} changed after ownership")
            return
        stores.record_mission_certificate(mission_id, platform, owner, score)

    def _held_failure_round(
        self,
        round_index: int,
        current_margins: dict[Platform, float],
        exact_mission: dict[Platform, list[float]],
        update_graph: Mapping[str, Sequence[str] | set[str]],
        robot_platforms: Mapping[str, Platform],
        reason: str,
        analytic_kappa: Mapping[str, float] | None = None,
        local_calibration: LocalCalibrationStores | None = None,
        empirical_ratios: Mapping[str, float] | None = None,
    ) -> MAICPRoundResult:
        mission_count = len(next(iter(exact_mission.values())))
        exact_thresholds: dict[Platform, float] = {}
        conformal_ranks: dict[Platform, int] = {}
        for platform in Platform:
            class_config = self.config.classes[platform]
            rank = binomial_calibration_rank(
                mission_count, class_config.alpha, class_config.delta_cal
            )
            conformal_ranks[platform] = rank
            if any(not np.isfinite(value) for value in exact_mission[platform]):
                exact_thresholds[platform] = float("nan")
            else:
                exact_thresholds[platform] = float(
                    sorted(exact_mission[platform])[rank - 1]
                )
        sensitivity = SensitivityResult(
            robot_kappa={platform: {} for platform in Platform},
            class_kappa={platform: float("nan") for platform in Platform},
            mode=self.config.sensitivity_mode.value,
            certified=False,
            valid=False,
            max_validation_ratio={platform: float("nan") for platform in Platform},
            notes=(reason,),
            paired_validation_valid=False,
        )
        update = hold_margin_update(current_margins, reason)
        return MAICPRoundResult(
            round_index=round_index,
            mission_scores={platform: tuple(values) for platform, values in exact_mission.items()},
            certified_mission_scores={
                platform: tuple(float("inf") for _ in values)
                for platform, values in exact_mission.items()
            },
            exact_thresholds=exact_thresholds,
            certified_thresholds={platform: float("inf") for platform in Platform},
            conformal_ranks=conformal_ranks,
            sensitivity=sensitivity,
            margin_update=update,
            communication_rounds=0,
            order_statistic_histories={},
            mission_certificate_owners={platform: {} for platform in Platform},
            pinball_rounds=0,
            flooding_rounds=0,
            kappa_flooding_rounds=0,
        )
