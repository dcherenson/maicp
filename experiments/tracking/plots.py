"""Tracking maps, uncertainty/error plots, and animations from saved wave traces."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Rectangle, Wedge
import numpy as np
from scipy.stats import chi2

from maicp.visualization import save_static_figure


def _task_edges_for_frame(metadata: dict[str, Any], frame: int) -> list[list[Any]]:
    frames = metadata.get("task_graph_frames", [])
    if frames:
        return list(frames[min(frame, len(frames) - 1)].get("edges", []))
    return list(metadata.get("task_edges", []))


def _task_edge_candidates(metadata: dict[str, Any]) -> list[list[Any]]:
    frames = metadata.get("task_graph_frames", [])
    if not frames:
        return list(metadata.get("task_edges", []))
    return [list(edge) for edge in sorted({tuple(edge) for frame in frames for edge in frame.get("edges", [])})]


def _draw_background(ax: plt.Axes, metadata: dict[str, Any]) -> None:
    extent = float(metadata["map_half_extent"])
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", alpha=0.45)
    ax.axhline(0.0, color="#888888", linewidth=0.7, alpha=0.5)
    ax.axvline(0.0, color="#888888", linewidth=0.7, alpha=0.5)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    for robot in metadata["robots"]:
        route = np.asarray(robot["waypoints"])
        color = plt.get_cmap("tab10")(int(robot["subteam_id"]))
        style = "-" if robot["platform"] == "ugv" else "--"
        ax.plot(route[:, 0], route[:, 1], color=color, linestyle=style, alpha=0.17, linewidth=0.8)
        ax.scatter(*robot["home"], marker="+", color=color, s=18, alpha=0.45)
    for target in metadata.get("targets", []):
        center = target["center"]
        half_width = target["half_width"]
        color = plt.get_cmap("tab10")(int(target["subteam_id"]))
        ax.add_patch(
            Rectangle(
                (center[0] - half_width, center[1] - half_width),
                2.0 * half_width,
                2.0 * half_width,
                facecolor=color,
                edgecolor=color,
                alpha=0.08,
                linestyle=":",
            )
        )


def _sensor_geometry(metadata: dict[str, Any]) -> dict[str, float]:
    """Read the canonical physical footprints from a heterogeneous trace."""

    try:
        sensors = metadata["sensors"]
        ugv = sensors["ugv"]
        uav = sensors["uav"]
        geometry = {
            "lidar_max_range": float(ugv["max_range_m"]),
            "lidar_fov_radians": float(ugv["fov_radians"]),
            "camera_max_range": float(uav["max_range_m"]),
            "camera_fov_radians": float(uav["fov_radians"]),
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "Tracking trace metadata must contain sensors.ugv/uav max_range_m "
            "and fov_radians"
        ) from error
    if any(not np.isfinite(value) or value <= 0.0 for value in geometry.values()):
        raise ValueError("Tracking sensor footprint metadata must be finite and positive")
    for platform, key in (
        ("LiDAR", "lidar_fov_radians"),
        ("camera", "camera_fov_radians"),
    ):
        if geometry[key] > 2.0 * np.pi:
            raise ValueError(f"{platform} field of view cannot exceed 2*pi radians")
    return geometry


def _sensor_spec_label(geometry: dict[str, float], platform: str) -> str:
    """Read the sensor footprint label from the metadata."""

    if platform == "ugv":
        sensor_name = "UGV LiDAR"
        max_range = geometry["lidar_max_range"]
        fov_radians = geometry["lidar_fov_radians"]
    elif platform == "uav":
        sensor_name = "UAV camera"
        max_range = geometry["camera_max_range"]
        fov_radians = geometry["camera_fov_radians"]
    else:
        raise ValueError(f"Unknown sensor platform {platform!r}")
    return f"{sensor_name} ({np.degrees(fov_radians):g}° FOV, {max_range:g} m range)"


def _sensor_legend_handles(metadata: dict[str, Any]) -> list[Line2D]:
    sensor_geometry = _sensor_geometry(metadata)
    return [
        Line2D(
            [0],
            [0],
            color="#2171b5",
            linewidth=1.2,
            label=f"{_sensor_spec_label(sensor_geometry, 'ugv')} sector",
        ),
        Line2D(
            [0],
            [0],
            color="#e6550d",
            linewidth=1.2,
            label=f"{_sensor_spec_label(sensor_geometry, 'uav')} sector",
        ),
    ]


def _camera_bearing(
    arrays: dict[str, np.ndarray], frame: int, robot_index: int
) -> float:
    """Get the measured relative camera bearing for one trace frame.

    Camera packets are scalar constraints across the line of sight, so a UAV
    observation cannot be plotted as a Cartesian point.  The simulator stores
    the scalar camera bearing explicitly.  It is the only trace field used for
    the rendered UAV line.
    """

    value = float(np.asarray(arrays["camera_bearing"])[frame, robot_index])
    return (
        float(np.arctan2(np.sin(value), np.cos(value)))
        if np.isfinite(value)
        else float("nan")
    )


def _ugv_observation(
    arrays: dict[str, np.ndarray], frame: int, robot_index: int
) -> np.ndarray | None:
    """Return the planar point measured by LiDAR when available in the trace."""

    dimension = int(np.asarray(arrays["measurement_dimension"])[frame, robot_index])
    if dimension != 2:
        return None
    value = np.asarray(arrays["ugv_measurement_values"], dtype=float)[frame, robot_index]
    if value.shape != (2,):
        raise ValueError("ugv_measurement_values must have shape (time, robot, 2)")
    return value if np.all(np.isfinite(value)) else None


def _draw_bearing_line(
    ax: plt.Axes,
    position: np.ndarray,
    heading: float,
    relative_bearing: float,
    length: float,
    *,
    color: Any,
    alpha: float = 0.7,
    linewidth: float = 0.9,
    linestyle: str = "-",
) -> Line2D | None:
    """Draw a camera line of bearing without inventing a target depth."""

    if not np.isfinite(relative_bearing) or length <= 0.0:
        return None
    theta = heading + relative_bearing
    endpoint = np.asarray(position, dtype=float) + length * np.asarray(
        [np.cos(theta), np.sin(theta)]
    )
    (line,) = ax.plot(
        [position[0], endpoint[0]],
        [position[1], endpoint[1]],
        color=color,
        alpha=alpha,
        linewidth=linewidth,
        linestyle=linestyle,
    )
    return line


def _draw_sensor(
    ax: plt.Axes,
    metadata: dict[str, Any],
    robot: dict[str, Any],
    position: np.ndarray,
    heading: float,
) -> None:
    sensors = _sensor_geometry(metadata)
    if robot["platform"] == "ugv":
        radius = sensors["lidar_max_range"]
        center_deg = np.degrees(heading)
        half_fov_deg = np.degrees(sensors["lidar_fov_radians"] / 2.0)
        ax.add_patch(
            Wedge(
                position,
                radius,
                center_deg - half_fov_deg,
                center_deg + half_fov_deg,
                facecolor="none",
                edgecolor="#2171b5",
                alpha=0.35,
                linewidth=0.55,
            )
        )
    else:
        center_deg = np.degrees(heading)
        half_fov_deg = np.degrees(sensors["camera_fov_radians"] / 2.0)
        ax.add_patch(
            Wedge(
                position,
                sensors["camera_max_range"],
                center_deg - half_fov_deg,
                center_deg + half_fov_deg,
                facecolor="none",
                edgecolor="#e6550d",
                alpha=0.28,
                linewidth=0.5,
            )
        )


def _class_alpha(metadata: dict[str, Any], platform: str) -> float:
    """Return the nominal class alpha used to report a position ellipse."""

    return float(metadata["class_parameters"][platform]["alpha"])


def _nominal_confidence(metadata: dict[str, Any], platform: str) -> float:
    return 1.0 - _class_alpha(metadata, platform)


def _mahalanobis_threshold(metadata: dict[str, Any], platform: str) -> float:
    """Return the radius used by the reported covariance for a class."""

    radius = metadata["uncertainty"]["coverage_radii"][platform]
    if radius is None:
        return float(np.sqrt(chi2.ppf(1.0 - _class_alpha(metadata, platform), df=2)))
    return float(radius)


def _reported_covariance(
    arrays: dict[str, np.ndarray], frame: int, robot_index: int
) -> np.ndarray:
    """Select the output covariance used for reporting, in position space."""

    covariance = np.asarray(arrays["reported_covariances"][frame, robot_index], dtype=float)
    if covariance.shape != (2, 2):
        raise ValueError(f"Reported position covariance must have shape (2, 2), got {covariance.shape}")
    return 0.5 * (covariance + covariance.T)


def _ellipse_geometry(
    covariance: np.ndarray, alpha: float
) -> tuple[float, float, float] | None:
    """Return the width, height, and angle of the nominal confidence ellipse."""

    if not np.all(np.isfinite(covariance)):
        return None
    quantile = float(chi2.ppf(1.0 - alpha, df=2))
    if not np.isfinite(quantile) or quantile <= 0.0:
        return None
    values, vectors = np.linalg.eigh(covariance)
    if not np.all(np.isfinite(values)):
        return None
    if np.any(values < -1e-10):
        return None
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    values = np.maximum(values, 0.0)
    angle = float(np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0])))
    width, height = 2.0 * np.sqrt(quantile * values)
    return float(width), float(height), angle


def _ellipse_legend_handles(metadata: dict[str, Any]) -> list[Line2D]:
    handles = []
    for platform, linestyle in (("ugv", "--"), ("uav", "-")):
        confidence = 100.0 * _nominal_confidence(metadata, platform)
        handles.append(
            Line2D(
                [0],
                [0],
                color="#444444",
                linestyle=linestyle,
                linewidth=1.5,
                label=f"Reported P ({platform.upper()}, nominal {confidence:.1f}% confidence)",
            )
        )
    return handles


def _draw_snapshot(
    ax: plt.Axes,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    frame: int,
) -> None:
    _draw_background(ax, metadata)
    robots = metadata["robots"]
    positions = arrays["positions"][frame]
    targets = arrays["target_truth"][frame]
    sensor_geometry = _sensor_geometry(metadata)
    id_to_index = {robot["robot_id"]: index for index, robot in enumerate(robots)}
    representative_sensors = set()
    for team in range(4):
        team_indices = [i for i, robot in enumerate(robots) if int(robot["subteam_id"]) == team]
        for platform in ("ugv", "uav"):
            eligible = [i for i in team_indices if robots[i]["platform"] == platform]
            representative_sensors.add(
                min(eligible, key=lambda i: np.linalg.norm(positions[i] - targets[team, :2]))
            )
    for edge in _task_edges_for_frame(metadata, frame):
        u, v = (id_to_index[edge[0]], id_to_index[edge[1]])
        ax.plot(
            positions[[u, v], 0],
            positions[[u, v], 1],
            color=plt.get_cmap("tab10")(int(edge[2])),
            linestyle=":",
            alpha=0.28,
            linewidth=0.7,
        )
    for index, robot in enumerate(robots):
        pos = positions[index]
        color = plt.get_cmap("tab10")(int(robot["subteam_id"]))
        if index in representative_sensors:
            _draw_sensor(ax, metadata, robot, pos, arrays["headings"][frame, index])
        ax.plot(
            pos[0], pos[1], marker="o" if robot["platform"] == "ugv" else "^",
            markerfacecolor=color, markeredgecolor="black", linestyle="None", markersize=5,
        )
        if arrays["phase"][frame, index] == 1:
            estimate = arrays["estimates"][frame, index, :2]
            ax.plot(estimate[0], estimate[1], marker="x", color=color, alpha=0.65, markersize=5)
            if arrays["visible"][frame, index]:
                if robot["platform"] == "ugv":
                    observation = _ugv_observation(arrays, frame, index)
                    if observation is not None:
                        ax.scatter(
                            *observation,
                            color=color,
                            marker=".",
                            s=22,
                            alpha=0.7,
                        )
                else:
                    bearing = _camera_bearing(
                        arrays,
                        frame,
                        index,
                    )
                    _draw_bearing_line(
                        ax,
                        pos,
                        float(arrays["headings"][frame, index]),
                        bearing,
                        sensor_geometry["camera_max_range"],
                        color=color,
                        alpha=0.62,
                        linewidth=1.0,
                        linestyle="--",
                    )
    for target_index, truth in enumerate(targets):
        color = plt.get_cmap("tab10")(target_index)
        ax.plot(truth[0], truth[1], marker="s", color=color, markeredgecolor="black", markersize=8)
        team_robots = [i for i, robot in enumerate(robots) if int(robot["subteam_id"]) == target_index]
        for platform in ("ugv", "uav"):
            eligible = [i for i in team_robots if robots[i]["platform"] == platform]
            closest = min(eligible, key=lambda i: np.linalg.norm(positions[i] - truth[:2]))
            estimate = arrays["estimates"][frame, closest, :2]
            ellipse_geometry = _ellipse_geometry(
                _reported_covariance(arrays, frame, closest),
                _class_alpha(metadata, platform),
            )
            if ellipse_geometry is not None:
                width, height, angle = ellipse_geometry
                ellipse = Ellipse(
                    estimate,
                    width,
                    height,
                    angle=angle,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.13,
                    linestyle="--" if platform == "ugv" else "-",
                )
                ax.add_patch(ellipse)
    context = metadata.get("context", {})
    method = context.get("method", metadata.get("method", "trace")) if isinstance(context, dict) else "trace"
    sensor_summary = (
        f"{_sensor_spec_label(sensor_geometry, 'ugv')}; "
        f"{_sensor_spec_label(sensor_geometry, 'uav')}"
    )
    ax.set_title(
        f"All four target missions — t={arrays['time'][frame]:.1f}s — {method}\n"
        f"{sensor_summary}\n"
        "Reported P ellipses use nominal 1-alpha confidence",
        fontsize=10,
    )
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="black", linestyle="None", label="UGV"),
            Line2D([0], [0], marker="^", color="black", linestyle="None", label="UAV"),
            Line2D([0], [0], marker="s", color="black", linestyle="None", label="Target truth"),
            Line2D([0], [0], marker="x", color="black", linestyle="None", label="Agent estimate"),
            Line2D([0], [0], marker=".", color="#2171b5", linestyle="None", label="UGV LiDAR Cartesian observation"),
            Line2D([0], [0], color="#e6550d", linestyle="--", linewidth=1.0, label="UAV camera line of bearing"),
        ] + _sensor_legend_handles(metadata) + _ellipse_legend_handles(metadata),
        loc="upper left",
        fontsize=8,
    )


def _plot_tracking_errors(
    arrays: dict[str, np.ndarray], metadata: dict[str, Any], base: Path
) -> list[Path]:
    robots = metadata["robots"]
    errors = np.full(arrays["phase"].shape, np.nan, dtype=float)
    sigma = np.full_like(errors, np.nan)
    for index, robot in enumerate(robots):
        team = int(robot["subteam_id"])
        errors[:, index] = np.linalg.norm(
            arrays["estimates"][:, index, :2] - arrays["target_truth"][:, team, :2], axis=1
        )
        sigma[:, index] = np.sqrt(
            np.maximum(
                0.0,
                np.trace(arrays["covariances"][:, index, :2, :2], axis1=1, axis2=2),
            )
        )
        errors[arrays["phase"][:, index] != 1, index] = np.nan
        sigma[arrays["phase"][:, index] != 1, index] = np.nan

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
    for team, ax in enumerate(axes.flat):
        indices = [i for i, robot in enumerate(robots) if int(robot["subteam_id"]) == team]
        active_any = np.any(arrays["phase"][:, indices] == 1, axis=1)
        if np.any(active_any):
            first, last = np.flatnonzero(active_any)[[0, -1]]
            ax.axvspan(arrays["time"][first], arrays["time"][last], color="#9ecae1", alpha=0.16)
        for index in indices:
            robot = robots[index]
            color = "#2171b5" if robot["platform"] == "ugv" else "#e6550d"
            ax.plot(arrays["time"], errors[:, index], color=color, alpha=0.52, linewidth=0.9)
        for platform, color in (("ugv", "#2171b5"), ("uav", "#e6550d")):
            group = [i for i in indices if robots[i]["platform"] == platform]
            if group:
                block_error = errors[:, group]
                block_sigma = sigma[:, group]
                median_error = np.asarray([
                    np.median(row[np.isfinite(row)]) if np.isfinite(row).any() else np.nan
                    for row in block_error
                ])
                median_sigma = np.asarray([
                    np.median(row[np.isfinite(row)]) if np.isfinite(row).any() else np.nan
                    for row in block_sigma
                ])
                ax.plot(arrays["time"], median_error, color=color, linewidth=2.0, label=f"{platform.upper()} median error")
                ax.plot(
                    arrays["time"],
                    median_sigma,
                    color=color,
                    linestyle="--",
                    linewidth=1.2,
                    label=f"{platform.upper()} median sqrt(trace(P_native)) diagnostic",
                )
        ax.set_title(f"Quadrant subteam {team + 1}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Euclidean position error (m)")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(fontsize=7)
    fig.suptitle(
        "Secondary diagnostic: Euclidean position error and sqrt(trace(P_native))"
    )
    fig.tight_layout()
    return save_static_figure(fig, base)


def _platform_indices(robots: list[dict[str, Any]], platform: str) -> list[int]:
    return [index for index, robot in enumerate(robots) if robot["platform"] == platform]


def _finite_median(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [np.nanmedian(row) if np.isfinite(row).any() else np.nan for row in values],
        dtype=float,
    )


def _plot_tracking_scores(
    arrays: dict[str, np.ndarray], metadata: dict[str, Any], base: Path
) -> list[Path]:
    """Plot conformal Mahalanobis scores and reported covariance inflation."""

    required = {"mahalanobis_scores", "scored_mask", "covariance_scale"}
    if not required.issubset(arrays):
        return []
    robots = metadata["robots"]
    time = np.asarray(arrays["time"], dtype=float)
    scores = np.asarray(arrays["mahalanobis_scores"], dtype=float)
    scored = np.asarray(arrays["scored_mask"], dtype=bool)
    scales = np.asarray(arrays["covariance_scale"], dtype=float)
    if scores.shape != scored.shape or scales.shape != scores.shape:
        raise ValueError("Tracking score diagnostics must all have shape (time, robot)")

    fig, (score_ax, scale_ax) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    colors = {"ugv": "#2171b5", "uav": "#e6550d"}
    for platform in ("ugv", "uav"):
        indices = _platform_indices(robots, platform)
        if not indices:
            continue
        color = colors[platform]
        platform_scores = np.where(
            scored[:, indices] & np.isfinite(scores[:, indices]), scores[:, indices], np.nan
        )
        for column in range(platform_scores.shape[1]):
            score_ax.plot(time, platform_scores[:, column], color=color, alpha=0.18, linewidth=0.7)
        score_ax.plot(
            time,
            _finite_median(platform_scores),
            color=color,
            linewidth=2.0,
            label=f"{platform.upper()} median Mahalanobis score",
        )
        robot_radii = metadata["uncertainty"].get("coverage_radii_by_robot", {})
        thresholds = {
            robot_radii[robots[index]["robot_id"]]
            for index in indices
            if robots[index]["robot_id"] in robot_radii
            and robot_radii[robots[index]["robot_id"]] is not None
        }
        threshold = next(iter(thresholds)) if len(thresholds) == 1 else _mahalanobis_threshold(metadata, platform)
        if len(thresholds) > 1:
            for radius in sorted(thresholds):
                score_ax.axhline(radius, color=color, linestyle="--", linewidth=0.5, alpha=0.3)
            score_ax.axhspan(min(thresholds), max(thresholds), color=color, alpha=0.08,
                            label=f"{platform.upper()} robot thresholds q0+r_i")
            threshold = np.nan
        if np.isfinite(threshold):
            nominal = metadata["uncertainty"]["coverage_radii"][platform] is None
            threshold_label = (
                f"{platform.upper()} nominal sqrt(chi2_2(1-alpha)) = {threshold:.3g}"
                if nominal
                else f"{platform.upper()} threshold q0+r = {threshold:.3g}"
            )
            score_ax.axhline(
                threshold,
                color=color,
                linestyle="--",
                linewidth=1.2,
                label=threshold_label,
            )

        platform_scales = np.where(
            np.isfinite(scales[:, indices]), scales[:, indices], np.nan
        )
        for column in range(platform_scales.shape[1]):
            scale_ax.plot(time, platform_scales[:, column], color=color, alpha=0.18, linewidth=0.7)
        scale_ax.plot(
            time,
            _finite_median(platform_scales),
            color=color,
            linewidth=2.0,
            label=f"{platform.upper()} median reported/native scale",
        )
        alpha = _class_alpha(metadata, platform)
        quantile = float(chi2.ppf(1.0 - alpha, df=2))
        if np.isfinite(threshold) and np.isfinite(quantile) and quantile > 0.0:
            scale_ax.axhline(
                threshold**2 / quantile,
                color=color,
                linestyle="--",
                linewidth=1.2,
                label=f"{platform.upper()} target scale = radius^2 / chi2_2(1-alpha)",
            )

    context = metadata.get("context", {})
    method = context.get("method", metadata.get("method", "trace")) if isinstance(context, dict) else "trace"
    score_ax.set_title(f"Mahalanobis score and conformal threshold — {method}")
    score_ax.set_ylabel("Mahalanobis score")
    score_ax.grid(True, linestyle=":", alpha=0.5)
    score_ax.legend(fontsize=8, ncol=2)
    scale_ax.axhline(1.0, color="#555555", linestyle=":", linewidth=1.0, label="Native P scale = 1")
    scale_ax.set_title("Reported covariance scaling (reporting output)")
    scale_ax.set_xlabel("Time (s)")
    scale_ax.set_ylabel("Reported P / native P scale")
    scale_ax.grid(True, linestyle=":", alpha=0.5)
    scale_ax.legend(fontsize=8, ncol=2)
    fig.suptitle("Tracking uncertainty diagnostics: scores, thresholds, and covariance inflation")
    fig.tight_layout()
    return save_static_figure(fig, base)


def render_tracking_trace(
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    directory: str | Path,
    method: str,
    round_number: int,
) -> list[Path]:
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    stem = f"{method}_round_{round_number}"
    snapshot_candidates = np.flatnonzero(np.any(arrays["phase"] == 1, axis=1))
    snapshot_frame = int(snapshot_candidates[len(snapshot_candidates) // 2]) if len(snapshot_candidates) else len(arrays["time"]) - 1
    figure, ax = plt.subplots(figsize=(10, 8))
    _draw_snapshot(ax, arrays, metadata, snapshot_frame)
    paths = save_static_figure(figure, output / f"tracking_{stem}_snapshot")
    paths.extend(_plot_tracking_scores(arrays, metadata, output / f"tracking_{stem}_scores"))
    paths.extend(_plot_tracking_errors(arrays, metadata, output / f"tracking_{stem}_errors"))

    robots = metadata["robots"]
    positions = arrays["positions"]
    sampled_frames = np.unique(
        np.linspace(0, len(arrays["time"]) - 1, min(80, len(arrays["time"])), dtype=int)
    )
    figure, ax = plt.subplots(figsize=(10, 8))
    _draw_background(ax, metadata)
    id_to_index = {robot["robot_id"]: index for index, robot in enumerate(robots)}
    sensor_geometry = _sensor_geometry(metadata)
    trails, markers, estimate_markers, observation_markers, bearing_lines = [], [], [], [], []
    sensor_patches = []
    for robot in robots:
        color = plt.get_cmap("tab10")(int(robot["subteam_id"]))
        trail, = ax.plot([], [], color=color, alpha=0.42, linewidth=1.0)
        marker, = ax.plot([], [], marker="o" if robot["platform"] == "ugv" else "^", color=color, markeredgecolor="black", linestyle="None", markersize=5)
        estimate, = ax.plot([], [], marker="x", color=color, alpha=0.65, linestyle="None", markersize=4)
        observation = ax.scatter([], [], marker=".", color=color, s=18, alpha=0.7)
        (bearing_line,) = ax.plot([], [], color=color, alpha=0.7, linewidth=0.9, linestyle="--")
        trails.append(trail)
        markers.append(marker)
        estimate_markers.append(estimate)
        observation_markers.append(observation)
        bearing_lines.append(bearing_line)
        if robot["platform"] == "ugv":
            half_fov = np.degrees(sensor_geometry["lidar_fov_radians"] / 2.0)
            patch = Wedge(
                (0, 0),
                sensor_geometry["lidar_max_range"],
                -half_fov,
                half_fov,
                facecolor="none",
                edgecolor="#2171b5",
                alpha=0.28,
                linewidth=0.45,
            )
        else:
            half_fov = np.degrees(sensor_geometry["camera_fov_radians"] / 2.0)
            patch = Wedge(
                (0, 0),
                sensor_geometry["camera_max_range"],
                -half_fov,
                half_fov,
                facecolor="none",
                edgecolor="#e6550d",
                alpha=0.18,
                linewidth=0.4,
            )
        ax.add_patch(patch)
        sensor_patches.append(patch)
    target_lines, target_markers, covariance_patches = [], [], []
    for target_index in range(4):
        color = plt.get_cmap("tab10")(target_index)
        line, = ax.plot([], [], color=color, linewidth=1.4, alpha=0.55)
        marker, = ax.plot([], [], marker="s", color=color, markeredgecolor="black", linestyle="None", markersize=7)
        target_lines.append(line)
        target_markers.append(marker)
        covariance_patches.append([])
        for platform in ("ugv", "uav"):
            ellipse = Ellipse(
                (0, 0),
                1,
                1,
                fill=True,
                facecolor=color,
                edgecolor=color,
                alpha=0.12,
                linestyle="--" if platform == "ugv" else "-",
                visible=False,
            )
            ax.add_patch(ellipse)
            covariance_patches[-1].append(ellipse)
    task_edge_candidates = _task_edge_candidates(metadata)
    task_links = []
    for _ in task_edge_candidates:
        line, = ax.plot([], [], color="#555555", linestyle=":", linewidth=0.65, alpha=0.3)
        task_links.append(line)
    update_links = []
    for _ in metadata.get("update_edges", []):
        line, = ax.plot([], [], color="black", linewidth=0.5, alpha=0.13)
        update_links.append(line)
    context = metadata.get("context", {})
    method = context.get("method", metadata.get("method", "trace")) if isinstance(context, dict) else "trace"
    sensor_summary = (
        f"{_sensor_spec_label(sensor_geometry, 'ugv')}; "
        f"{_sensor_spec_label(sensor_geometry, 'uav')}"
    )
    title = ax.set_title(
        f"Four isolated target tracking subteams — {method}\n"
        f"{sensor_summary}\n"
        "Reported P ellipses use nominal 1-alpha confidence",
        fontsize=10,
    )
    artists = trails + markers + estimate_markers + observation_markers + bearing_lines + sensor_patches + target_lines + target_markers + task_links + update_links
    artists.extend(patch for pair in covariance_patches for patch in pair)
    artists.append(title)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="black", linestyle="None", label="UGV"),
            Line2D([0], [0], marker="^", color="black", linestyle="None", label="UAV"),
            Line2D([0], [0], marker="s", color="black", linestyle="None", label="Target truth"),
            Line2D([0], [0], marker="x", color="black", linestyle="None", label="Agent estimate"),
            Line2D([0], [0], marker=".", color="#2171b5", linestyle="None", label="UGV LiDAR Cartesian observation"),
            Line2D([0], [0], color="#e6550d", linestyle="--", linewidth=1.0, label="UAV camera line of bearing"),
        ] + _sensor_legend_handles(metadata) + _ellipse_legend_handles(metadata),
        loc="upper left",
        fontsize=8,
    )

    def update(sample_index: int):
        frame = int(sampled_frames[sample_index])
        representative_sensors: set[int] = set()
        for team in range(4):
            team_indices = [i for i, robot in enumerate(robots) if int(robot["subteam_id"]) == team]
            target_position = arrays["target_truth"][frame, team, :2]
            for platform in ("ugv", "uav"):
                eligible = [i for i in team_indices if robots[i]["platform"] == platform]
                representative_sensors.add(
                    min(eligible, key=lambda i: np.linalg.norm(positions[frame, i] - target_position))
                )
        for index, robot in enumerate(robots):
            pos = positions[frame, index]
            trails[index].set_data(positions[: frame + 1, index, 0], positions[: frame + 1, index, 1])
            markers[index].set_data([pos[0]], [pos[1]])
            if arrays["phase"][frame, index] == 1:
                estimate_pos = arrays["estimates"][frame, index, :2]
                estimate_markers[index].set_data([estimate_pos[0]], [estimate_pos[1]])
            else:
                estimate_markers[index].set_data([], [])
            visible = bool(arrays["visible"][frame, index])
            if robot["platform"] == "ugv":
                observation = _ugv_observation(arrays, frame, index)
                observation_markers[index].set_offsets(
                    observation[None, :]
                    if visible and observation is not None
                    else np.empty((0, 2))
                )
                bearing_lines[index].set_data([], [])
            else:
                observation_markers[index].set_offsets(np.empty((0, 2)))
                bearing = _camera_bearing(
                    arrays,
                    frame,
                    index,
                )
                if visible and arrays["phase"][frame, index] == 1:
                    theta = float(arrays["headings"][frame, index]) + bearing
                    endpoint = pos + sensor_geometry["camera_max_range"] * np.asarray(
                        [np.cos(theta), np.sin(theta)]
                    )
                    bearing_lines[index].set_data(
                        [pos[0], endpoint[0]], [pos[1], endpoint[1]]
                    )
                else:
                    bearing_lines[index].set_data([], [])
            patch = sensor_patches[index]
            patch.set_visible(index in representative_sensors)
            patch.set_center(pos)
            angle = np.degrees(arrays["headings"][frame, index])
            fov_key = (
                "lidar_fov_radians"
                if robot["platform"] == "ugv"
                else "camera_fov_radians"
            )
            half_fov = np.degrees(sensor_geometry[fov_key] / 2.0)
            patch.set_theta1(angle - half_fov)
            patch.set_theta2(angle + half_fov)
        for target_index in range(4):
            truth_history = arrays["target_truth"][: frame + 1, target_index]
            target_lines[target_index].set_data(truth_history[:, 0], truth_history[:, 1])
            truth = arrays["target_truth"][frame, target_index]
            target_markers[target_index].set_data([truth[0]], [truth[1]])
            team_robots = [i for i, robot in enumerate(robots) if int(robot["subteam_id"]) == target_index]
            for patch_index, platform in enumerate(("ugv", "uav")):
                eligible = [i for i in team_robots if robots[i]["platform"] == platform]
                closest = min(eligible, key=lambda i: np.linalg.norm(positions[frame, i] - truth[:2]))
                mean = arrays["estimates"][frame, closest, :2]
                ellipse = covariance_patches[target_index][patch_index]
                geometry = _ellipse_geometry(
                    _reported_covariance(arrays, frame, closest),
                    _class_alpha(metadata, platform),
                )
                if geometry is None:
                    ellipse.set_visible(False)
                else:
                    width, height, angle = geometry
                    ellipse.set_center(mean)
                    ellipse.width, ellipse.height = width, height
                    ellipse.angle = angle
                    ellipse.set_visible(True)
        active_edges = {tuple(edge) for edge in _task_edges_for_frame(metadata, frame)}
        for line, edge in zip(task_links, task_edge_candidates, strict=True):
            if tuple(edge) in active_edges:
                u, v = id_to_index[edge[0]], id_to_index[edge[1]]
                line.set_data(positions[frame, [u, v], 0], positions[frame, [u, v], 1])
            else:
                line.set_data([], [])
        if frame == len(arrays["time"]) - 1:
            for line, edge in zip(update_links, metadata.get("update_edges", []), strict=True):
                u, v = id_to_index[edge[0]], id_to_index[edge[1]]
                line.set_data(positions[frame, [u, v], 0], positions[frame, [u, v], 1])
        title.set_text(
            f"All four target missions — t={arrays['time'][frame]:.1f}s — {method}\n"
            f"{sensor_summary}\n"
            "Reported P ellipses use nominal 1-alpha confidence"
        )
        return artists

    movie = animation.FuncAnimation(figure, update, frames=len(sampled_frames), interval=100, blit=False)
    gif_path = output / f"tracking_{stem}.gif"
    movie.save(gif_path, writer=animation.PillowWriter(fps=10))
    plt.close(figure)
    paths.append(gif_path)
    return paths


def plot_mission_comparison(robust, nominal, figures):
    """Compare reported ellipses for the same robot and time in both missions."""
    arrays, metadata = robust
    baseline, nominal_metadata = nominal
    truth = arrays["target_truth"][:, 0, :2]
    np.testing.assert_allclose(truth, baseline["target_truth"][:, 0, :2])
    team = [i for i, r in enumerate(metadata["robots"]) if r["subteam_id"] == 0]
    valid = arrays["scored_mask"][:, team] & baseline["scored_mask"][:, team]
    times = np.flatnonzero(valid.any(axis=1))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(*truth.T, color="0.2", linewidth=1.5, label="Target trajectory")
    for n, frame in enumerate(times[np.linspace(0, len(times)-1, min(7, len(times)), dtype=int)]):
        available = np.flatnonzero(valid[frame])
        i = team[available[n % len(available)]]
        platform = metadata["robots"][i]["platform"]
        for (trace, meta), color, label in [
            ((baseline, nominal_metadata), "#ee902c", "Nominal ellipse"),
            ((arrays, metadata), "#2475aa", "Calibrated ellipse"),
        ]:
            geometry = _ellipse_geometry(_reported_covariance(trace, frame, i), _class_alpha(meta, platform))
            if geometry is None:
                raise ValueError("Invalid reported covariance at a scored update.")
            w, h, angle = geometry
            ax.add_patch(Ellipse(trace["estimates"][frame, i, :2], w, h, angle=angle,
                                 fill=False, edgecolor=color, linewidth=1.3,
                                 label=label if n == 0 else None))
        ax.scatter(*truth[frame], s=12, color="black")
        ax.annotate(f"{arrays['time'][frame]:g} s", truth[frame], xytext=(4, 4),
                    textcoords="offset points", fontsize=8)
    ax.autoscale_view()
    ax.set(aspect="equal", xlabel="East (m)", ylabel="North (m)")
    ax.legend()
    return save_static_figure(fig, figures / "tracking_mission_from_trace")
