"""Collision map, route animation, and residual diagnostics for saved wave traces."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from maicp.configuration import Platform
from maicp.visualization import save_static_figure


PLATFORM_MARKERS = {"ugv": "o", "uav": "^"}


def _obstacle_color(obstacle: dict[str, Any]) -> str:
    """Choose the footprint color from the classes affected by the obstacle."""

    platform = obstacle.get("platform")
    if platform in (None, "shared", "both"):
        return "#636363"
    return "#8c2d04" if platform == Platform.UGV.value else "#54278f"


def _frame_index(arrays: dict[str, np.ndarray]) -> int:
    phases = arrays["phase"]
    active_count = np.sum(phases == 1, axis=1)
    if np.any(active_count):
        return int(np.argmax(active_count))
    # Select the midpoint of the formation trace.
    return max(0, (len(phases) - 1) // 2)


def _task_edges_for_frame(
    arrays: dict[str, np.ndarray], metadata: dict[str, Any], frame: int
) -> list[list[Any]]:
    """Return the edge snapshot recorded at the displayed physical time."""

    frames = metadata.get("task_graph_frames", [])
    if frames:
        index = min(frame, len(frames) - 1)
        return list(frames[index].get("edges", []))
    return list(metadata.get("task_edges", []))


def _task_edge_candidates(metadata: dict[str, Any]) -> list[list[Any]]:
    candidates = metadata.get("task_edge_candidates")
    if candidates:
        return [list(edge) for edge in candidates]
    frames = metadata.get("task_graph_frames", [])
    return sorted(
        {
            tuple(edge)
            for frame in frames
            for edge in frame.get("edges", [])
        }
        or {tuple(edge) for edge in metadata.get("task_edges", [])}
    )


def _controller_label(metadata: dict[str, Any]) -> str:
    return str(metadata.get("controller", "simplified_projection_baseline")).replace(
        "_", " "
    ).title()


def _draw_static(ax: plt.Axes, metadata: dict[str, Any]) -> None:
    extent = float(metadata["map_half_extent"])
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", alpha=0.45)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.axhline(0.0, color="#888888", linewidth=0.7, alpha=0.5)
    ax.axvline(0.0, color="#888888", linewidth=0.7, alpha=0.5)
    robots = metadata["robots"]
    for robot in robots:
        route = np.asarray(robot["waypoints"])
        color = plt.get_cmap("tab10")(int(robot["subteam_id"]))
        style = "-" if robot["platform"] == "ugv" else "--"
        ax.plot(route[:, 0], route[:, 1], color=color, linestyle=style, alpha=0.19, linewidth=0.9)
        ax.scatter(*robot["home"], color=color, marker="+", s=20, alpha=0.55)
    # The formation case has one obstacle catalog per subteam.  A
    # footprint is shared by both operating layers and is therefore drawn once
    # without a filter by class.
    for obstacle in metadata.get("obstacles", []):
        center = obstacle["center"]
        color = _obstacle_color(obstacle)
        ax.add_patch(
            plt.Circle(center, obstacle["radius"], facecolor=color, edgecolor=color, alpha=0.25)
        )
    ax.scatter([0.0], [0.0], marker="s", color="black", s=34, label="Map center")


def _draw_snapshot(
    ax: plt.Axes,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    frame: int,
) -> None:
    _draw_static(ax, metadata)
    robots = metadata["robots"]
    positions = arrays["positions"]
    phases = arrays["phase"]
    events = arrays["collision_events"]
    id_to_index = {robot["robot_id"]: index for index, robot in enumerate(robots)}
    for edge in _task_edges_for_frame(arrays, metadata, frame):
        u, v = edge[:2]
        p1, p2 = positions[frame, id_to_index[u]], positions[frame, id_to_index[v]]
        color = plt.get_cmap("tab10")(int(edge[2]))
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linestyle=":", alpha=0.45, linewidth=0.8)
    for index, robot in enumerate(robots):
        color = plt.get_cmap("tab10")(int(robot["subteam_id"]))
        marker = PLATFORM_MARKERS[robot["platform"]]
        pos = positions[frame, index]
        ax.plot(pos[0], pos[1], marker=marker, color=color, markeredgecolor="black", markersize=6)
        history = positions[: frame + 1, index]
        ax.plot(history[:, 0], history[:, 1], color=color, alpha=0.36, linewidth=1.2)
        if events[frame, index]:
            ax.scatter(*pos, marker="X", color="red", s=85, zorder=8)
    if frame == len(arrays["time"]) - 1:
        for u, v in metadata.get("update_edges", []):
            p1, p2 = positions[frame, id_to_index[u]], positions[frame, id_to_index[v]]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="black", linewidth=0.65, alpha=0.2)
    ax.set_title(f"{_controller_label(metadata)} — t={arrays['time'][frame]:.1f}s")
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="black", linestyle="None", label="UGV"),
            Line2D([0], [0], marker="^", color="black", linestyle="None", label="UAV"),
            Line2D([0], [0], marker="X", color="red", linestyle="None", label="Collision event"),
            Line2D([0], [0], color="#555555", linestyle=":", label="Current task graph"),
        ],
        loc="upper left",
        fontsize=8,
    )


def _plot_residual_diagnostics(
    arrays: dict[str, np.ndarray], metadata: dict[str, Any], base: Path
) -> list[Path]:
    robots = metadata["robots"]
    times = arrays["time"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    if metadata.get("controller") == "hocbf":
        # The HOCBF trace uses the same acceleration score recorded by the
        # Wang experiment, while ``aggregate_cbf_residual`` is the minimum
        # robust HOCBF slack over the local obstacle constraints.
        series = (
            (arrays["dynamics_mismatch"], "Acceleration-model error (m/s²)"),
            (arrays["aggregate_cbf_residual"], "Minimum robust HOCBF slack"),
        )
    elif metadata.get("controller") == "wang":
        series = (
            (arrays["dynamics_mismatch"], "Acceleration-model error (m/s²)"),
            (arrays["aggregate_cbf_residual"], "Minimum implemented Wang residual"),
        )
    else:
        residual_label = (
            "Maximum positive CBF optimism"
            if metadata.get("controller") == "sampled_distributed_mestres_eq19"
            else "Maximum absolute CBF residual gap"
        )
        series = (
            (arrays["dynamics_mismatch"], "Nominal/reference velocity difference (m/s)"),
            (arrays["residual_gap"], residual_label),
        )
    for ax, (values, ylabel) in zip(axes, series, strict=True):
        for platform in (Platform.UGV.value, Platform.UAV.value):
            indices = [index for index, robot in enumerate(robots) if robot["platform"] == platform]
            if not indices:
                continue
            block = values[:, indices]
            median = np.asarray([
                np.median(row[np.isfinite(row)]) if np.isfinite(row).any() else np.nan
                for row in block
            ])
            upper = np.asarray([
                np.quantile(row[np.isfinite(row)], 0.9) if np.isfinite(row).any() else np.nan
                for row in block
            ])
            ax.plot(times, median, label=f"{platform.upper()} median")
            ax.plot(times, upper, linestyle="--", alpha=0.7, label=f"{platform.upper()} 90th percentile")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(ncols=2, fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"{_controller_label(metadata)} diagnostics")
    fig.tight_layout()
    return save_static_figure(fig, base)


def render_collision_trace(
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    directory: str | Path,
    method: str,
    round_number: int,
) -> list[Path]:
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    stem = f"{method}_round_{round_number}"
    snapshot_frame = _frame_index(arrays)
    figure, ax = plt.subplots(figsize=(10, 8))
    _draw_snapshot(ax, arrays, metadata, snapshot_frame)
    paths = save_static_figure(figure, output / f"collision_{stem}_snapshot")
    paths.extend(_plot_residual_diagnostics(arrays, metadata, output / f"collision_{stem}_residuals"))

    robots = metadata["robots"]
    positions = arrays["positions"]
    sampled_frames = np.unique(
        np.linspace(0, len(arrays["time"]) - 1, min(80, len(arrays["time"])), dtype=int)
    )
    figure, ax = plt.subplots(figsize=(10, 8))
    _draw_static(ax, metadata)
    id_to_index = {robot["robot_id"]: index for index, robot in enumerate(robots)}
    trails, markers = [], []
    for robot in robots:
        color = plt.get_cmap("tab10")(int(robot["subteam_id"]))
        trail, = ax.plot([], [], color=color, alpha=0.55, linewidth=1.2)
        marker, = ax.plot(
            [], [], marker=PLATFORM_MARKERS[robot["platform"]], color=color,
            markeredgecolor="black", linestyle="None", markersize=6,
        )
        trails.append(trail)
        markers.append(marker)
    task_edge_candidates = _task_edge_candidates(metadata)
    task_links = []
    for _ in task_edge_candidates:
        line, = ax.plot([], [], color="#555555", linestyle=":", linewidth=0.7, alpha=0.4)
        task_links.append(line)
    update_links = []
    for _ in metadata.get("update_edges", []):
        line, = ax.plot([], [], color="black", linewidth=0.6, alpha=0.16)
        update_links.append(line)
    collision_scatter = ax.scatter([], [], marker="X", color="red", s=80)
    title = ax.set_title(_controller_label(metadata))
    artists = trails + markers + task_links + update_links + [collision_scatter, title]

    def update(sample_index: int):
        frame = int(sampled_frames[sample_index])
        for index, (trail, marker) in enumerate(zip(trails, markers, strict=True)):
            trail.set_data(positions[: frame + 1, index, 0], positions[: frame + 1, index, 1])
            marker.set_data([positions[frame, index, 0]], [positions[frame, index, 1]])
        active_edges = {tuple(edge[:2]): edge for edge in _task_edges_for_frame(arrays, metadata, frame)}
        for line, edge in zip(task_links, task_edge_candidates, strict=True):
            key = tuple(edge[:2])
            if key in active_edges:
                u, v = id_to_index[key[0]], id_to_index[key[1]]
                line.set_data(positions[frame, [u, v], 0], positions[frame, [u, v], 1])
            else:
                line.set_data([], [])
        if frame == len(arrays["time"]) - 1:
            for line, edge in zip(update_links, metadata.get("update_edges", []), strict=True):
                u, v = id_to_index[edge[0]], id_to_index[edge[1]]
                line.set_data(positions[frame, [u, v], 0], positions[frame, [u, v], 1])
        active = np.flatnonzero(arrays["collision_events"][frame])
        collision_scatter.set_offsets(positions[frame, active] if len(active) else np.empty((0, 2)))
        title.set_text(f"{_controller_label(metadata)} — t={arrays['time'][frame]:.1f}s")
        return artists

    movie = animation.FuncAnimation(
        figure, update, frames=len(sampled_frames), interval=100, blit=False
    )
    gif_path = output / f"collision_{stem}.gif"
    movie.save(gif_path, writer=animation.PillowWriter(fps=10))
    plt.close(figure)
    paths.append(gif_path)
    return paths


def plot_mission_comparison(robust, nominal, figures):
    """Show the first subteam and its closest nominal UAV approach to a pillar."""
    arrays, metadata = robust
    baseline, _ = nominal
    team = [i for i, r in enumerate(metadata["robots"]) if r["subteam_id"] == 0]
    obstacles = [o for o in metadata["obstacles"] if o["subteam_id"] == 0]
    uavs = [i for i in team if metadata["robots"][i]["platform"] == "uav"]
    paths = arrays["positions"][:, :, :2]
    other = baseline["positions"][:, :, :2]
    fig, ax = plt.subplots(figsize=(9, 5))
    for i in team:
        kind = metadata["robots"][i]["platform"]
        color = "#316b91" if kind == "uav" else "#588361"
        ax.plot(*paths[:, i].T, color=color, alpha=0.8)
        ax.scatter(*paths[-1, i], color=color, marker=PLATFORM_MARKERS[kind])
    for obstacle in obstacles:
        ax.add_patch(plt.Circle(obstacle["center"], obstacle["radius"], color="0.65"))
    if obstacles and uavs:
        candidates = []
        for i in uavs:
            for j, obstacle in enumerate(obstacles):
                distances = np.linalg.norm(other[:, i] - obstacle["center"], axis=1)
                frame = int(np.argmin(distances))
                candidates.append((distances[frame] - obstacle["radius"], i, j, frame))
        _, i, j, frame = min(candidates)
        obstacle = obstacles[j]
        center = np.array(obstacle["center"])
        inset = ax.inset_axes([0.64, 0.57, 0.34, 0.41], facecolor="white")
        inset.add_patch(plt.Circle(center, obstacle["radius"], color="0.7"))
        radius = 1.5 * max(obstacle["radius"], np.linalg.norm(paths[frame, i] - center),
                           np.linalg.norm(other[frame, i] - center))
        for path, color, label in [(paths, "#1f77b4", "Calibrated"), (other, "#e08829", "Nominal")]:
            inset.plot(*path[:frame+1, i].T, color=color, label=label)
            inset.scatter(*path[frame, i], color=color, marker="^")
        inset.set(xlim=center[0]+np.array([-radius,radius]), ylim=center[1]+np.array([-radius,radius]),
                  aspect="equal", xticks=[], yticks=[], title=f"UAV at {arrays['time'][frame]:g} s")
        inset.legend(fontsize=7)
    ax.set(aspect="equal", xlabel="East (m)", ylabel="North (m)")
    return save_static_figure(fig, figures / "formation_mission_from_trace")
