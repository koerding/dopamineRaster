#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[1]
SPIKES = ROOT / "data" / "dopamine_raster_spike_times.csv"
FIGS = ROOT / "figures"
FIGS.mkdir(exist_ok=True, parents=True)

spikes = pd.read_csv(SPIKES)

panel_order = [
    "no_prediction_reward_occurs",
    "reward_predicted_reward_occurs",
    "reward_predicted_no_reward_occurs",
]

panel_titles = {
    "no_prediction_reward_occurs": "No prediction, reward occurs",
    "reward_predicted_reward_occurs": "Reward predicted, reward occurs",
    "reward_predicted_no_reward_occurs": "Reward predicted, reward omitted",
}

panel_colors = {
    "no_prediction_reward_occurs": "#1f9e89",
    "reward_predicted_reward_occurs": "#3b82f6",
    "reward_predicted_no_reward_occurs": "#a855f7",
}

panel_specs = {
    "no_prediction_reward_occurs": {
        "xlim": (-0.5, 2.0),
        "xticks": [-0.5, 0, 0.5, 1.0, 1.5, 2.0],
        "xlabel": "Time (s)",
        "events": [("Reward", 1.0, "#f59e0b", "-")],
    },
    "reward_predicted_reward_occurs": {
        "xlim": (-1.0, 2.0),
        "xticks": [-1.0, 0, 1.0, 2.0],
        "xlabel": "Time (s)",
        "events": [("CS", 0.0, "#2563eb", "-"), ("Reward", 1.0, "#f59e0b", "-")],
    },
    "reward_predicted_no_reward_occurs": {
        "xlim": (-1.0, 2.0),
        "xticks": [-1.0, 0, 1.0, 2.0],
        "xlabel": "Time (s)",
        "events": [("CS", 0.0, "#2563eb", "-"), ("Expected reward", 1.0, "#ef4444", "--")],
    },
}

plot_rows = []
trial_counts = {}
for panel in panel_order:
    df = spikes[spikes["panel"] == panel].copy()
    unique_y = np.sort(df["y_px_in_panel"].unique())
    y_to_trial = {y: i + 1 for i, y in enumerate(unique_y[::-1])}
    df["trial"] = df["y_px_in_panel"].map(y_to_trial)
    plot_rows.append(df)
    trial_counts[panel] = len(unique_y)

plot_df = pd.concat(plot_rows, ignore_index=True)

plt.close("all")
fig = plt.figure(figsize=(11.8, 15.8), constrained_layout=False)
gs = GridSpec(
    nrows=8,
    ncols=1,
    height_ratios=[0.95, 2.2, 0.66, 0.95, 2.2, 0.66, 0.95, 2.2],
    hspace=0.08,
    figure=fig,
)

fig.patch.set_facecolor("#fbfbfd")
panel_row_map = [(0, 1), (3, 4), (6, 7)]
bin_width = 0.025

for i, panel in enumerate(panel_order):
    rate_row, raster_row = panel_row_map[i]
    color = panel_colors[panel]
    spec = panel_specs[panel]
    df = plot_df[plot_df["panel"] == panel].copy()
    n_trials = trial_counts[panel]

    ax_rate = fig.add_subplot(gs[rate_row, 0])
    ax_raster = fig.add_subplot(gs[raster_row, 0], sharex=ax_rate)

    ax_rate.set_facecolor("white")
    ax_raster.set_facecolor("white")

    bins = np.arange(spec["xlim"][0], spec["xlim"][1] + bin_width, bin_width)
    bin_centers = bins[:-1] + bin_width / 2
    counts, _ = np.histogram(df["time_s"], bins=bins)
    rate_hz = counts / max(n_trials, 1) / bin_width
    smooth_rate_hz = gaussian_filter1d(rate_hz, sigma=1.7)

    ax_rate.fill_between(bin_centers, smooth_rate_hz, 0, color=color, alpha=0.22, linewidth=0)
    ax_rate.plot(bin_centers, smooth_rate_hz, color=color, linewidth=2.8)

    ax_raster.scatter(
        df["time_s"],
        df["trial"],
        s=18,
        c=color,
        alpha=0.9,
        edgecolors="white",
        linewidths=0.25,
        zorder=3,
    )

    ymax = max(float(smooth_rate_hz.max()), 1.0)
    for label, x, event_color, linestyle in spec["events"]:
        for ax in (ax_rate, ax_raster):
            ax.axvspan(x - 0.045, x + 0.045, color=event_color, alpha=0.09, zorder=0)
            ax.axvline(x, color=event_color, linewidth=1.9, linestyle=linestyle, alpha=0.95, zorder=1)

        ax_rate.text(
            x,
            ymax * 1.04,
            label,
            ha="center",
            va="bottom",
            fontsize=10.2,
            color=event_color,
            fontweight="bold",
            clip_on=False,
        )

    ax_rate.set_xlim(*spec["xlim"])
    ax_rate.set_ylabel("Rate\n(Hz)", fontsize=10)
    ax_rate.spines["top"].set_visible(False)
    ax_rate.spines["right"].set_visible(False)
    ax_rate.spines["bottom"].set_visible(False)
    ax_rate.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_rate.tick_params(axis="y", labelsize=9)
    ax_rate.grid(axis="y", alpha=0.18, linewidth=0.8)
    ax_rate.margins(y=0.18)

    ax_raster.set_xlim(*spec["xlim"])
    ax_raster.set_ylim(0.5, n_trials + 0.8)
    ax_raster.set_ylabel("Trial", fontsize=10)
    ax_raster.set_xticks(spec["xticks"])
    ax_raster.set_xlabel(spec["xlabel"], fontsize=11)
    ax_raster.spines["top"].set_visible(False)
    ax_raster.spines["right"].set_visible(False)
    ax_raster.tick_params(axis="both", labelsize=9)
    ax_raster.grid(axis="x", alpha=0.12, linewidth=0.8)
    ax_raster.grid(axis="y", alpha=0.04, linewidth=0.6)

    ax_rate.text(
        0.0,
        1.14,
        panel_titles[panel],
        transform=ax_rate.transAxes,
        ha="left",
        va="bottom",
        fontsize=14.5,
        fontweight="bold",
        color="#111827",
    )

fig.suptitle(
    "Dopamine responses as reward prediction error",
    fontsize=20,
    fontweight="bold",
    x=0.5,
    y=0.985,
    color="#111827",
)

fig.text(
    0.5,
    0.962,
    "Replotted from extracted spike times digitized from the classic raster figure",
    ha="center",
    va="center",
    fontsize=11.5,
    color="#4b5563",
)

fig.text(
    0.012,
    0.5,
    "Single-unit raster and smoothed response",
    rotation=90,
    va="center",
    fontsize=11,
    color="#4b5563",
)

fig.subplots_adjust(top=0.91, bottom=0.055, left=0.1, right=0.98)

fig.savefig(FIGS / "dopamine_raster_modern_replot_axis_ranges_fixed.png", dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
fig.savefig(FIGS / "dopamine_raster_modern_replot_axis_ranges_fixed.pdf", dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
