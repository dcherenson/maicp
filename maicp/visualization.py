"""Headless plots for MA-ICP reports and replayable heterogeneous mission traces."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator, MultipleLocator
import numpy as np
from scipy.stats import beta

from maicp.configuration import Platform, build_config
from .trace import load_wave_trace
from .outcomes import enrich_saved_robot_outcomes


METHOD_ORDER = (
    "distributed",
    "centralized",
    "unshifted",
    "fixed",
    "single_agent",
    "nominal",
)

METHOD_LABELS = {
    "distributed": "MA-ICP (ours)",
    "centralized": "centralized",
    "unshifted": "unshifted",
    "fixed": "fixed margin",
    "single_agent": "single-agent",
    "nominal": "non-robust",
}

METHOD_COLORS = {
    "distributed": "#1f77b4",
    "centralized": "#ff7f0e",
    "unshifted": "#2ca02c",
    "fixed": "#9467bd",
    "single_agent": "#d62728",
    "nominal": "#7f7f7f",
}


METHOD_STYLES = {
    # Draw ours above other curves without changing insertion/legend order.
    "distributed": {"marker": "o", "linestyle": "-", "zorder": 3},
    "centralized": {"marker": "s", "linestyle": "--"},
    "unshifted": {"marker": "^", "linestyle": "-."},
    "fixed": {"marker": "P", "linestyle": (0, (5, 2, 1, 2, 1, 2))},
    "single_agent": {"marker": "x", "linestyle": "--", "zorder": 2},
    "nominal": {"marker": "D", "linestyle": ":"},
}


def save_static_figure(figure: plt.Figure, base_path: str | Path) -> list[Path]:
    base = Path(base_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    outputs = [base.with_suffix(".svg"), base.with_suffix(".png")]
    if base.name.startswith("figure_"):
        outputs.append(base.with_suffix(".pdf"))
    # Keep summary canvases identical across applications and legend contents.
    bbox = None if base.name.startswith("maicp_summary_") else "tight"
    for output in outputs:
        figure.savefig(output, dpi=180 if output.suffix == ".png" else None, bbox_inches=bbox)
    plt.close(figure)
    return outputs


def _number(value: Any) -> float:
    try:
        if isinstance(value, str) and value in {"+inf", "-inf", "nan"}:
            return {"+inf": float("inf"), "-inf": float("-inf"), "nan": float("nan")}[
                value
            ]
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _class_value(mapping: dict[Any, Any], platform: Platform) -> Any:
    return mapping.get(platform, mapping.get(platform.value))

def _episode_semantics_version(report: dict[str, Any]) -> Any:
    return report.get("episode_semantics", {}).get("version")


def _robot_platform(value: Any) -> Any:
    return value.value if isinstance(value, Platform) else value

def _mean_finite(values: list[float]) -> float:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    return float(finite.mean()) if finite.size else float("nan")


def _pooled_coverage(records: list[dict[str, Any]], platform: Platform) -> tuple[float, float, float]:
    successes = sum(int(record["deployment"][platform.value]["successes"]) for record in records)
    trials = sum(int(record["deployment"][platform.value]["trials"]) for record in records)
    if not trials:
        return float("nan"), float("nan"), float("nan")
    lower = 0.0 if successes == 0 else float(beta.ppf(0.025, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(beta.ppf(0.975, successes + 1, trials - successes))
    return successes / trials, lower, upper


def _records_by_method(report: dict[str, Any]) -> dict[str, dict[int, list[dict[str, Any]]]]:
    grouped: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    zero_based = _episode_semantics_version(report) is not None
    for replicate in report["replicates"]:
        for method, rounds in replicate["methods"].items():
            for round_index, record in enumerate(rounds):
                episode = int(record.get("episode", round_index)) if zero_based else round_index
                grouped[method][episode].append(record)
    return grouped


def _add_coverage_zero_axis(ax: plt.Axes) -> plt.Axes:
    """Keep the zoomed coverage range and show zero across an explicit axis break."""
    grid = ax.get_subplotspec().subgridspec(2, 1, height_ratios=(10, 1), hspace=0.08)
    ax.set_subplotspec(grid[0])
    zero_ax = ax.figure.add_subplot(grid[1], sharex=ax)
    zero_ax.set(ylim=(-0.01, 0.04), yticks=[0.0])
    ax.spines["bottom"].set_visible(False)
    zero_ax.spines["top"].set_visible(False)
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    # Marker dimensions are in points, so both sides of the break match even
    # though the two axes have different heights.
    for panel, y in ((ax, 0), (zero_ax, 1)):
        panel.scatter(
            [0, 1], [y, y], transform=panel.transAxes,
            marker=[(-1, -0.5), (1, 0.5)], s=100,
            color="black", linewidths=1,
            clip_on=False,
        )
    return zero_ax


def _add_data_gap_axis(ax: plt.Axes) -> plt.Axes | None:
    """Show separated low and high series without clipping observations."""
    values = np.unique(np.concatenate([
        np.asarray(line.get_ydata(), dtype=float).ravel() for line in ax.lines
    ]))
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return None
    gaps = np.diff(values)
    cut = int(np.argmax(gaps))
    gap = float(gaps[cut])
    if gap < 0.25 * float(values[-1] - values[0]):
        return None
    low, high = values[:cut + 1], values[cut + 1:]
    lower_ax = _add_coverage_zero_axis(ax)
    low_pad = max(float(np.ptp(low)) * 0.15, gap * 0.025)
    high_pad = max(float(np.ptp(high)) * 0.08, gap * 0.025)
    lower_ax.set_ylim(float(low[0]) - low_pad, float(low[-1]) + low_pad)
    lower_ax.set_yticks([float(np.mean(low))])
    ax.set_ylim(float(high[0]) - high_pad, float(high[-1]) + high_pad)
    for line in list(ax.lines):
        lower_ax.plot(line.get_xdata(), line.get_ydata(), color=line.get_color(),
                      linestyle=line.get_linestyle(), marker=line.get_marker(),
                      markersize=line.get_markersize(), linewidth=line.get_linewidth(),
                      alpha=line.get_alpha(), label="_nolegend_")
    return lower_ax


def _plot_summary(report: dict[str, Any], directory: Path, paper_directory: Path | None = None) -> list[Path]:
    enrich_saved_robot_outcomes(report)
    grouped = _records_by_method(report)
    methods = [method for method in METHOD_ORDER if method in grouped]
    rounds = max((index for method in methods for index in grouped[method]), default=-1) + 1
    if not methods or not rounds:
        return []
    tracking = report.get("application") == "target_tracking"
    zero_based = _episode_semantics_version(report) is not None
    # Version 1 shares the initial batch across methods and starts with nonzero margins.
    # Keep those records, but do not present them as nominal performance.
    omit_nominal_initial = tracking and _episode_semantics_version(report) == 1
    x = np.arange(rounds) if zero_based else np.arange(1, rounds + 1)
    x_label = "MA-ICP Iteration"
    artifacts: list[Path] = []

    def draw_platform(figure, platform):
        axes = figure.subplots(1, 3)
        ax_margin, ax_coverage, ax_outcome = axes
        ax_coverage_zero = _add_coverage_zero_axis(ax_coverage) if tracking else None
        coverage_axes = (ax_coverage, ax_coverage_zero) if tracking else (ax_coverage,)
        coverage_values: list[float] = []
        outcome_values: list[float] = []
        for method in methods:
            color = METHOD_COLORS.get(method)
            style = METHOD_STYLES[method]
            margins, coverage, outcomes = ([] for _ in range(3))
            for round_index in range(rounds):
                records = grouped[method].get(round_index, [])
                if omit_nominal_initial and method == "nominal" and round_index == 0:
                    records = []
                if method == "single_agent":
                    margins.append(_mean_finite([
                        _number(record["margins"][robot_id])
                        for record in records
                        for robot_id, owner in record["robot_platforms"].items()
                        if _robot_platform(owner) == platform.value
                        and robot_id in record["margins"]
                    ]) if tracking else float("nan"))
                else:
                    margins.append(
                        _mean_finite([_number(_class_value(record["margins"], platform)) for record in records])
                    )
                cov, _, _ = _pooled_coverage(records, platform) if records else (np.nan, np.nan, np.nan)
                coverage.append(cov)
                robot_trials = sum(record["deployment"][platform.value]["robot_trials"] for record in records)
                if tracking:
                    uncertainty = [record["deployment"][platform.value]["uncertainty"] for record in records]
                    samples = sum(item["samples"] for item in uncertainty)
                    # Pool scored estimates over robots, time steps, and replicates.
                    outcomes.append(
                        sum(item["reported_mean_ellipse_area"] * item["samples"]
                            for item in uncertainty if item["samples"] > 0) / samples
                        if samples else float("nan")
                    )
                else:
                    # Count each robot with an obstacle collision once per mission.
                    failure_counts = []
                    for record in records:
                        audits = record.get("episode_batch", record.get("deployment_batch", {})).get(
                            "mission_outcomes", record["deployment"].get("mission_audit", [])
                        )
                        if len(audits) != record["deployment"]["missions"] or any(
                            "robot_failure_reasons" not in audit for audit in audits
                        ):
                            failure_counts.append(None)
                            continue
                        failure_counts.append(sum(
                            _robot_platform(record["robot_platforms"].get(robot_id)) == platform.value
                            and any(reason.partition(":")[0] in {
                                "obstacle_collision", "intersample_obstacle_collision"
                            } for reason in reasons)
                            for audit in audits
                            for robot_id, reasons in audit["robot_failure_reasons"].items()
                        ))
                    outcomes.append(
                        1.0 - sum(failure_counts) / robot_trials
                        if robot_trials and all(count is not None for count in failure_counts)
                        else float("nan")
                    )

            if method == "single_agent" and not tracking:
                robot_ids = sorted({
                    robot_id
                    for by_episode in grouped[method].values()
                    for record in by_episode
                    for robot_id, owner in record["robot_platforms"].items()
                    if _robot_platform(owner) == platform.value
                })
                for robot_index, robot_id in enumerate(robot_ids):
                    robot_margins = []
                    for episode in range(rounds):
                        robot_margins.append(_mean_finite([
                            _number(record["margins"][robot_id])
                            for record in grouped[method].get(episode, [])
                            if robot_id in record["margins"]
                            and _robot_platform(record["robot_platforms"].get(robot_id)) == platform.value
                        ]))
                    ax_margin.plot(
                        x,
                        robot_margins,
                        color=color,
                        linestyle="-",
                        marker=".",
                        alpha=0.5,
                        label="single-agent (individual robots)" if robot_index == 0 else "_nolegend_",
                    )
            else:
                ax_margin.plot(
                    x, margins, color=color, **style,
                    label="single-agent (mean)" if method == "single_agent" else METHOD_LABELS.get(method, method),
                )
            for ax in coverage_axes:
                ax.plot(
                    x,
                    coverage,
                    color=color,
                    **style,
                    label=METHOD_LABELS.get(method, method)
                )
            ax_outcome.plot(x, outcomes, color=color, **style, label=METHOD_LABELS.get(method, method))
            coverage_values.extend(coverage)
            outcome_values.extend(outcomes)

        alpha = _number(report.get("class_alpha", {}).get(platform.value, 0.1))
        for ax in coverage_axes:
            ax.axhline(1.0 - alpha, color="black", linestyle=":", label="Coverage target")

        ax_margin.set(title="(i) Robustness Margin Evolution", ylabel="Margin")
        ax_coverage.set(title="(ii) Mission-Level Score Coverage", ylabel="Coverage", ylim=(0.55, 1.05))
        if tracking:
            ax_outcome.set(title="(iii) Mean Covariance Ellipse Area", ylabel="Area (m²)")
            ax_outcome.set_yscale("linear")
            ax_outcome.set_ylim(bottom=0.0)
        else:
            ax_outcome.set(title="(iii) Safe Fraction", ylabel="Safe fraction")
            ax_outcome.set_ylim(0.0, 1.04)
        visible_coverage = [v for v in coverage_values if np.isfinite(v) and (not tracking or v > 0.04)]
        if visible_coverage:
            top_values = visible_coverage + [1.0 - alpha]
            span = max(top_values) - min(top_values)
            padding = max(0.015, 0.08 * span)
            ax_coverage.set_ylim(max(0.04 if tracking else -0.02, np.floor((min(top_values) - padding) * 10) / 10),
                                 min(1.02, np.ceil((max(top_values) + padding) * 10) / 10))
            ax_coverage.yaxis.set_major_locator(MultipleLocator(0.1))

        split_upper_axes = [ax_coverage] if tracking else []
        for ax in ((ax_margin, ax_outcome) if tracking else ()):
            if _add_data_gap_axis(ax) is not None:
                split_upper_axes.append(ax)
        if not tracking:
            ax_outcome.yaxis.set_major_locator(MultipleLocator(0.1))
        for ax in figure.axes:
            ax.title.set_fontsize(18)
            ax.xaxis.label.set_fontsize(17)
            ax.yaxis.label.set_fontsize(17)
            ax.tick_params(axis="both", labelsize=15)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}".rstrip("0").rstrip(".")))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}".rstrip("0").rstrip(".")))
            if ax not in split_upper_axes:
                ax.set_xlabel(x_label)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
            ax.set_xlim(-0.5 if zero_based else 0.5,
                        rounds - 0.5 if zero_based else rounds + 0.5)
            ax.grid(True, linestyle=":", alpha=0.5)
        legend_entries = {}
        for ax in figure.axes:
            for line in ax.lines:
                line.set_linewidth(3.2)
            handles, labels = ax.get_legend_handles_labels()
            for handle, label in zip(handles, labels):
                if label in ("single-agent (mean)", "single-agent"):
                    label = "single-agent (mean margin)" if tracking else "single-agent (individual robots)"
                legend_entries.setdefault(label, handle)
        return legend_entries

    for platform in Platform:
        fig = plt.figure(figsize=(18, 4.8))
        legend_entries = draw_platform(fig, platform)
        fig.legend(legend_entries.values(), legend_entries.keys(), loc="lower center",
                   bbox_to_anchor=(0.5, 0.01), ncols=len(legend_entries), fontsize=17,
                   columnspacing=1.0, handlelength=1.8, handletextpad=0.5,
                   frameon=False)
        fig.tight_layout(rect=(0, 0.15, 1, 1))
        artifacts.extend(save_static_figure(fig, directory / f"maicp_summary_{platform.value}"))

    paper = plt.figure(figsize=(15, 7.6))
    paper.subplots_adjust(left=0.08, right=0.98, bottom=0.18, top=0.83, wspace=0.37)
    rows = paper.subfigures(2, 1)
    paper_entries = {}
    for row, platform, label in zip(rows, (Platform.UAV, Platform.UGV), ("(a) UAV", "(b) UGV")):
        row.subplots_adjust(left=0.08, right=0.985, top=0.83, bottom=0.28 if platform == Platform.UGV else 0.15, wspace=0.37)
        paper_entries.update(draw_platform(row, platform))
        row.text(0.014, 0.95, label, weight="bold", fontsize=17,
                 bbox=dict(facecolor="#fff1b4", edgecolor="black", pad=2))
    paper.legend(paper_entries.values(), paper_entries.keys(), loc="lower center",
                 ncols=len(paper_entries), frameon=False, fontsize=12)
    name = "figure_8_tracking_results" if tracking else "figure_6_formation_results"
    artifacts.extend(save_static_figure(paper, (paper_directory or directory) / name))

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    for method in methods:
        runtime, communication, global_failure = [], [], []
        for round_index in range(rounds):
            records = grouped[method].get(round_index, [])
            if omit_nominal_initial and method == "nominal" and round_index == 0:
                records = []
            mission_count = sum(record["deployment"]["missions"] for record in records)
            global_failure.append(
                sum(record["deployment"]["mission_failures"] for record in records) / mission_count
                if mission_count else float("nan")
            )
            runtime.append(_mean_finite([_number(record["runtime_seconds"]) for record in records]))
            communication.append(
                _mean_finite([
                    _number(record.get("cp_communication_rounds", record.get("maicp", {}).get("communication_rounds", float("nan"))))
                    for record in records
                ])
            )
        axes[0].plot(x, runtime, **METHOD_STYLES[method], color=METHOD_COLORS.get(method), label=METHOD_LABELS.get(method, method))
        axes[1].plot(x, communication, **METHOD_STYLES[method], color=METHOD_COLORS.get(method), label=METHOD_LABELS.get(method, method))
        axes[2].plot(x, global_failure, **METHOD_STYLES[method], color=METHOD_COLORS.get(method), label=METHOD_LABELS.get(method, method))
    axes[0].set(title="Runtime", xlabel=x_label, ylabel="Seconds")
    axes[1].set(title="Communication", xlabel=x_label, ylabel="Flooding rounds")
    axes[2].set(title="Global mission failures", xlabel=x_label, ylabel="Failure fraction", ylim=(-0.05, 1.05))
    for ax in axes:
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.title.set_fontsize(18)
        ax.xaxis.label.set_fontsize(17)
        ax.yaxis.label.set_fontsize(17)
        ax.tick_params(axis="both", labelsize=15)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}".rstrip("0").rstrip(".")))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}".rstrip("0").rstrip(".")))
        for line in ax.lines:
            line.set_linewidth(3.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01),
               ncols=len(labels), fontsize=17, frameon=False)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    artifacts.extend(save_static_figure(fig, directory / "execution_diagnostics"))
    return artifacts


def paper_figures(case, results, report=None):
    """Draw the quantitative figures and include the submitted mission illustration."""
    import json
    from shutil import copyfile

    figures = results / "figures"
    if case == "payload":
        from experiments.payload.plots import plot_mission, plot_sensitivity

        plot_sensitivity(results, figures)
        if (results / "registration.json").exists():
            plot_mission(results, figures)
        number = 3
    else:
        if report is None:
            candidates = sorted(results.rglob("metrics.json"))
            if len(candidates) != 1:
                raise ValueError(f"Expected one metrics.json under {results}, found {len(candidates)}.")
            metrics_path = candidates[0]
            report = json.loads(metrics_path.read_text())
        else:
            metrics_path = Path(report["output_file"])
        expected = "target_tracking" if case == "tracking" else "collision_avoidance"
        if report["application"] != expected:
            raise ValueError(f"Expected {expected} metrics, found {report['application']}.")
        _plot_summary(report, metrics_path.parent / "figures", figures)
        pair = [metrics_path.parent / "traces" / f"replicate_0_{method}_episode_1.npz"
                for method in ("distributed", "nominal")]
        if all(path.is_file() for path in pair):
            if case == "tracking":
                from experiments.tracking.plots import plot_mission_comparison
            else:
                from experiments.formation.plots import plot_mission_comparison
            robust, nominal = (load_wave_trace(path) for path in pair)
            if robust[1]["context"]["episode_seed"] != nominal[1]["context"]["episode_seed"]:
                raise ValueError("Mission traces must share the episode seed.")
            if robust[1]["robots"] != nominal[1]["robots"]:
                raise ValueError("Mission traces must describe the same robots.")
            length = min(len(robust[0]["time"]), len(nominal[0]["time"]))
            np.testing.assert_array_equal(robust[0]["time"][:length], nominal[0]["time"][:length])
            robust = ({key: value[:length] for key, value in robust[0].items()}, robust[1])
            nominal = ({key: value[:length] for key, value in nominal[0].items()}, nominal[1])
            plot_mission_comparison(robust, nominal, figures)
        number = 7 if case == "tracking" else 5
    artwork = Path(__file__).resolve().parents[1] / "experiments" / case / "mission.pdf"
    if artwork.is_file():
        copyfile(artwork, figures / f"figure_{number}_{case}_mission.pdf")


def _plot_order_statistic_histories(report: dict[str, Any], directory: Path) -> list[Path]:
    artifacts: list[Path] = []
    first_replicate = report["replicates"][0] if report.get("replicates") else {}
    zero_based = _episode_semantics_version(report) is not None
    for method, rounds in first_replicate.get("methods", {}).items():
        for round_index, record in enumerate(rounds):
            if "maicp" not in record:
                continue
            maicp = record["maicp"]
            histories = maicp.get("order_statistic_histories", {})
            episode = int(record.get("episode", round_index)) if zero_based else round_index + 1
            if not histories:
                continue
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            for row, platform in enumerate((Platform.UGV, Platform.UAV)):
                stages = histories.get(platform.value, {})
                for col, stage in enumerate(("mission", "threshold")):
                    ax = axes[row, col]
                    history = np.asarray(stages.get(stage, []), dtype=float)
                    if history.ndim == 2 and history.size:
                        for node in range(history.shape[1]):
                            ax.plot(np.arange(history.shape[0]), history[:, node], alpha=0.30, linewidth=0.8)
                    exact_key = "first_mission_exact" if stage == "mission" else "exact_thresholds"
                    certified_key = "first_mission_certified" if stage == "mission" else "certified_thresholds"
                    exact = _number(maicp.get(exact_key, {}).get(platform.value))
                    conservative = _number(maicp.get(certified_key, {}).get(platform.value))
                    if np.isfinite(exact):
                        ax.axhline(exact, color="black", linestyle=":", label="Exact order statistic")
                    if np.isfinite(conservative):
                        ax.axhline(conservative, color="#d62728", linestyle="--", label="Conservative result")
                    ax.set(title=f"{platform.value.upper()} {stage} pinball", xlabel="Iteration", ylabel="Agent estimate")
                    ax.grid(True, linestyle=":", alpha=0.5)
                    if ax.lines:
                        ax.legend(fontsize=8)
            fig.tight_layout()
            suffix = f"episode_{episode}" if zero_based else f"round_{episode}"
            artifacts.extend(save_static_figure(fig, directory / f"pinball_{method}_{suffix}"))
    return artifacts


def render_experiment_report(report: dict[str, Any], result_root: str | Path) -> list[str]:
    """Render the summary plots and representative waves for each application."""

    root = Path(result_root)
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    paths = _plot_summary(report, figures)
    paths.extend(_plot_order_statistic_histories(report, figures))
    zero_based = _episode_semantics_version(report) is not None
    for replicate in report.get("replicates", [])[:1]:
        for method, rounds in replicate.get("methods", {}).items():
            for round_index, record in enumerate(rounds):
                trace_name = record.get("trace_file")
                if not trace_name:
                    continue
                episode = int(record.get("episode", round_index)) if zero_based else round_index + 1
                arrays, metadata = load_wave_trace(root / trace_name)
                if report["application"] == "collision_avoidance":
                    from experiments.formation.plots import render_collision_trace

                    paths.extend(
                        render_collision_trace(arrays, metadata, figures / "missions", method, episode)
                    )
                elif report["application"] == "target_tracking":
                    from experiments.tracking.plots import render_tracking_trace

                    paths.extend(
                        render_tracking_trace(arrays, metadata, figures / "missions", method, episode)
                    )
    return [path.relative_to(root).as_posix() for path in paths]
