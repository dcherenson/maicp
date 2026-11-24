"""Score sensitivity for the payload experiment."""

import json

import numpy as np
from matplotlib.lines import Line2D

from maicp.visualization import plt, save_static_figure


def plot_sensitivity(results, figures):
    directory = results / "sensitivity"
    checks = json.loads((directory / "checks.json").read_text())
    with np.load(directory / "score_shifts.npz") as data:
        shifts = data["shifts"]
    distances = np.array([record["distance"] for record in checks["records"]])
    if shifts.shape != (len(distances), checks["missions_per_law"], 2):
        raise ValueError("The saved score changes and margin pairs do not match.")
    if not np.all(np.isfinite(distances) & (distances > 0)):
        raise ValueError("Each margin pair must have a positive finite distance.")
    ratios = shifts / distances[:, None, None]
    np.testing.assert_allclose(ratios.max(axis=(0, 1)), checks["maximum_mission_ratio_by_class"])
    x = np.repeat(distances, shifts.shape[1])
    bins = np.floor(x / 0.3).astype(int)
    occupied = np.unique(bins)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3), sharex=True)
    for c, (ax, title, color, marker) in enumerate(zip(
        axes, ("(a) Light lift UAVs", "(b) Heavy lift UAVs"), ("#1f77b4", "#c2188b"), ("o", "s")
    )):
        y = ratios[:, :, c].ravel()
        bound = checks["kappa"][c]
        ax.scatter(x, y, s=16, alpha=0.12, color=color, marker=marker,
                   edgecolors="none", rasterized=True)
        ax.plot((occupied + 0.5) * 0.3, [y[bins == b].max() for b in occupied],
                color=color, marker="o", linewidth=2.2)
        ax.axhline(bound, color=color, linestyle="--", linewidth=2.3)
        ax.annotate(fr"$\kappa_{c+1}\approx {bound:.4f}$", (0.04, bound),
                    xycoords=("axes fraction", "data"), xytext=(0, -4),
                    textcoords="offset points", va="top", color=color)
        ax.set(title=title, xlabel=r"$\|\Delta\mathbf{r}\|_2$ (N)", xlim=(0, x.max()*1.05),
               ylim=(0, 1.1*max(bound, y.max())))
        ax.grid(linestyle=":", alpha=0.5)
    fig.supylabel(r"Empirical sensitivity $\hat{\kappa}_c$")
    handles = [
        Line2D([], [], color="0.5", marker="o", linestyle="none", label="Empirical sensitivity (all samples)"),
        Line2D([], [], color="0.3", marker="o", label="Max. in bin (empirical)"),
        Line2D([], [], color="0.3", linestyle="--", label=r"Analytical bound $\kappa_c$"),
    ]
    fig.legend(handles=handles, loc="lower center", ncols=3, frameon=False)
    fig.tight_layout(rect=(0.015, 0.11, 1, 1))
    return save_static_figure(fig, figures / "figure_4_payload_sensitivity")


def plot_mission(results, figures):
    """Plot the final calibrated mission saved for the first replicate."""
    registration = json.loads((results / "registration.json").read_text())
    episode = registration["experiment"]["updates"]
    classes = registration["config"]["classes"]
    with np.load(results / f"trace_maicp_{episode}.npz") as trace:
        commands = trace["commands"]
        delivered = commands + trace["residual"]
        time = trace["time"]
        position = trace["states"][:, :, 0]
    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    for i, c in enumerate(classes):
        color = ("#1f77b4", "#c2188b")[c]
        label = ("Light lift", "Heavy lift")[c] if classes.index(c) == i else None
        axes[0].plot(time[:len(commands)], delivered[:, i], color=color, label=label)
        axes[1].plot(time, position[:, i], color=color)
    axes[0].set_ylabel("Delivered force (N)")
    axes[1].set(xlabel="Time (s)", ylabel="Vertical position error (m)")
    axes[0].legend()
    fig.tight_layout()
    return save_static_figure(fig, figures / "payload_mission_from_trace")
