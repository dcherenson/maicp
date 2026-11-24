"""Application outcomes independent of conformal score coverage."""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any
import warnings

from maicp.configuration import Platform


def summarize_robot_outcomes(
    failed_robots: Sequence[Mapping[str, Sequence[str]]],
    failure_reasons: Sequence[Mapping[str, Sequence[str]]],
    robot_trials: Mapping[str, int],
) -> dict[str, Any]:
    """Count robot failures by mission and tally their recorded causes."""
    result: dict[str, Any] = {}
    for platform in Platform:
        trials = robot_trials[platform.value]
        failures = sum(len(set(mission.get(platform.value, ()))) for mission in failed_robots)
        if not 0 <= failures <= trials or trials <= 0:
            raise ValueError(f"Invalid {platform.value} robot outcome counts: {failures}/{trials}")
        fraction = failures / trials
        result[platform.value] = {
            "application_robot_failures": failures,
            "application_robot_successes": trials - failures,
            "application_robot_failure_fraction": fraction,
            "application_robot_success_fraction": 1.0 - fraction,
        }
    counts: Counter[str] = Counter()
    for mission in failure_reasons:
        for reasons in mission.values():
            for reason in reasons:
                prefix = reason.partition(":")[0]
                # Remove neighbor IDs, but retain details such as exception types.
                key = prefix if prefix in {"agent_collision", "unsafe_initial_pair_state"} else reason
                counts[key] += 1
    result["robot_failure_reason_counts"] = dict(sorted(counts.items()))
    return result


def enrich_saved_robot_outcomes(report: dict[str, Any]) -> None:
    """Add derived fields to saved reports without rerunning any simulation.

    Prefer the batch outcome audit copied directly from MissionResult. Missing
    audits remain unavailable, never inferred from scores or collision counts.
    """
    for replicate in report["replicates"]:
        for method, rounds in replicate["methods"].items():
            for index, record in enumerate(rounds, 1):
                deployment = record["deployment"]
                for platform in Platform:
                    entry = deployment[platform.value]
                    entry.setdefault("robot_score_covered", entry["robot_successes"])
                    entry.setdefault("robot_score_coverage", entry["robot_score_covered"] / entry["robot_trials"])
                if all("application_robot_failures" in deployment[p.value] for p in Platform):
                    continue
                audits = record.get("episode_batch", record.get("deployment_batch", {})).get("mission_outcomes")
                if audits is None:
                    audits = deployment.get("mission_audit", [])
                if (len(audits) != deployment["missions"]
                    or len({a["mission_id"] for a in audits}) != deployment["missions"]
                    or any("failed_robots" not in a or "robot_failure_reasons" not in a for a in audits)):
                    warnings.warn(
                        f"{method} round {index}: robot application outcomes unavailable; "
                        "complete robot failure audits are required", stacklevel=2,
                    )
                    continue
                summary = summarize_robot_outcomes(
                    [a["failed_robots"] for a in audits],
                    [a["robot_failure_reasons"] for a in audits],
                    {p.value: deployment[p.value]["robot_trials"] for p in Platform},
                )
                for platform in Platform:
                    deployment[platform.value].update(summary[platform.value])
                deployment["robot_failure_reason_counts"] = summary["robot_failure_reason_counts"]
