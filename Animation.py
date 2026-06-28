"""Create a GIF animation of the micro-model swarm.

Inputs:
    A Config object, output GIF path, optional simulated seconds, optional rest
    time, playback duration, and frame rate.

Outputs:
    A GIF showing robot states, displayed food stars, energy, and time.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import tempfile

_MPL_CONFIG_DIR = Path(tempfile.gettempdir()) / "swarm_foraging_mpl"
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
_CACHE_DIR = Path(tempfile.gettempdir()) / "swarm_foraging_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
import numpy as np

from config import Config
from agents import MicroModel
from map import PaperMap


def make_animation(
    cfg: Config,
    out_path: str | Path,
    seconds: float | None = None,
    rest_time_s: float | None = None,
    playback_seconds: float | None = None,
    fps: int | None = None,
) -> Path:
    """Run the micro model and save a color-keyed swarm animation."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tau_r = float(rest_time_s if rest_time_s is not None else cfg.get("behaviour", "default_rest_time_s"))
    seconds = float(seconds if seconds is not None else cfg.get("run", "animation_seconds", default=4000))
    playback_seconds = float(
        playback_seconds
        if playback_seconds is not None
        else cfg.get("run", "animation_playback_seconds", default=30)
    )
    fps = int(fps if fps is not None else cfg.get("run", "animation_fps", default=15))
    if seconds <= 0 or playback_seconds <= 0 or fps <= 0:
        raise ValueError("animation seconds, playback seconds, and fps must be positive")
    if fps > 100:
        raise ValueError("GIF frame rate cannot exceed 100 fps")

    world = PaperMap.from_config(cfg)
    gif_frame_ms = max(10, (int(1000 / fps) // 10) * 10)
    target_frames = max(2, int(round(playback_seconds * 1000 / gif_frame_ms)))
    stride = max(1, int(math.ceil(world.steps(seconds) / target_frames)))
    seed = int(cfg.get("run", "random_seed", default=7))
    model = MicroModel(cfg, rest_time_s=tau_r, seed=seed)
    trace = model.run(seconds=seconds, stride=stride, keep_frames=True)

    max_food_stars = max(1, int(math.floor(float(trace["food_items"].max()))))
    food_rng = np.random.default_rng(seed + 1)
    radii = np.sqrt(food_rng.uniform(world.rinner**2, world.router**2, max_food_stars))
    angles = food_rng.uniform(0.0, 2.0 * math.pi, max_food_stars)
    food_positions = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))

    colors = {
        "searching": "tab:blue",
        "grabbing": "tab:orange",
        "deposit": "tab:green",
        "homing": "tab:red",
        "resting": "tab:purple",
        "avoidance": "tab:brown",
    }

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    fig.subplots_adjust(right=0.75)
    ax.set_aspect("equal")
    ax.set_xlim(-world.router - 0.3, world.router + 0.3)
    ax.set_ylim(-world.router - 0.3, world.router + 0.3)
    ax.set_title("Swarm foraging PFSM")
    outer = plt.Circle((0, 0), world.router, fill=False, linewidth=1.5)
    inner = plt.Circle((0, 0), world.rinner, fill=False, linestyle="--", linewidth=1.0)
    home = plt.Circle((0, 0), world.home_radius, fill=False, linewidth=1.5)
    ax.add_patch(outer)
    ax.add_patch(inner)
    ax.add_patch(home)
    food_scatter = ax.scatter(
        [],
        [],
        marker="*",
        s=95,
        color="gold",
        edgecolor="darkgoldenrod",
        linewidth=0.6,
        zorder=2,
    )
    scatter = ax.scatter([], [], s=80, zorder=3)
    label = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")
    state_labels = {
        "searching": "Searching",
        "grabbing": "Grabbing",
        "deposit": "Depositing",
        "homing": "Homing",
        "resting": "Resting",
        "avoidance": "Avoidance",
    }
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color,
            markeredgecolor=color,
            markersize=7,
            label=state_labels[state],
        )
        for state, color in colors.items()
    ]
    legend_handles.append(
        Line2D(
            [0],
            [0],
            marker="*",
            color="none",
            markerfacecolor="gold",
            markeredgecolor="darkgoldenrod",
            markersize=10,
            label="Food",
        )
    )
    ax.legend(
        handles=legend_handles,
        title="Color key",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        fancybox=False,
        edgecolor="black",
    )

    def update(frame_idx: int):
        """Draw one animation frame from the sampled micro-model trace.

        Input:
            frame_idx is the row index in the sampled trace.

        Output:
            Matplotlib artists updated in place for FuncAnimation.
        """
        row = trace.iloc[frame_idx]
        positions = row["positions"]
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        cs = [colors[p[2]] for p in positions]
        scatter.set_offsets(list(zip(xs, ys)))
        scatter.set_color(cs)
        food_count = min(len(food_positions), max(0, int(math.floor(row["food_items"]))))
        food_scatter.set_offsets(food_positions[:food_count] if food_count else np.empty((0, 2)))
        label.set_text(f"t={row['time_s']:.0f}s\nE={row['energy']/1e5:.2f} x10^5\nfood={food_count:d}")
        return food_scatter, scatter, label

    anim = FuncAnimation(fig, update, frames=len(trace), interval=1000 / fps, blit=True)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out_path
