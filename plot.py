from __future__ import annotations

from copy import deepcopy
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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

from agents import MicroModel
from config import Config
from macro import MacroModel


SWARM_SIZES = [8, 16, 32, 64, 128]
FOOD_MULTIPLIERS = [0.25, 0.5, 1.0, 2.0, 4.0]


def ensure_dir(path: str | Path) -> Path:
    """Create an output folder if needed and return it as a Path."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def output_folders(out_dir: str | Path) -> tuple[Path, Path, Path]:
    """Create and return presentation, supporting, and data output folders.

    Input:
        out_dir is the root output folder requested by the CLI.

    Output:
        Paths for presentation plots, supporting plots, and CSV data.
    """
    root = ensure_dir(out_dir)
    presentation = ensure_dir(root / "presentation_plots")
    supporting = ensure_dir(root / "supporting_plots")
    data = ensure_dir(root / "data")
    return presentation, supporting, data


def remove_known_outputs(folder: Path, names: list[str]) -> None:
    """Remove stale generated files from an output folder.

    Inputs:
        folder is an output subfolder and names are generated filenames that can
        be safely replaced.

    Output:
        Matching files are removed when present.
    """
    for name in names:
        path = folder / name
        if path.exists():
            path.unlink()


def write_macro_csv(cfg: Config, out_dir: str | Path, rest_time_s: float | None = None) -> Path:
    """Write one aggregate macro-model trace CSV.

    Inputs:
        cfg is the loaded model configuration, out_dir is the destination
        folder, and rest_time_s optionally overrides the rest time.

    Output:
        Path to the written macro trace CSV.
    """
    out = ensure_dir(Path(out_dir) / "data")
    tau_r = float(rest_time_s if rest_time_s is not None else cfg.get("behaviour", "default_rest_time_s"))
    trace = MacroModel(cfg, rest_time_s=tau_r).run().trace
    path = out / f"macro_trace_tau_r_{int(tau_r)}.csv"
    trace.to_csv(path, index=False)
    return path


def make_all_plots(
    cfg: Config,
    out_dir: str | Path,
    seconds: float | None = None,
    seeds: int = 5,
    sample_every: float = 50.0,
) -> list[Path]:
    """Write the split presentation/supporting plot set and matching CSV files.

    Inputs:
        cfg is the loaded configuration, out_dir receives outputs, seconds is
        simulated duration, seeds controls repeated runs, and sample_every
        controls time-series sampling.

    Output:
        Paths to written PNG and CSV files.
    """
    presentation, supporting, data = output_folders(out_dir)
    presentation_split = ensure_dir(presentation / "split_panels")
    supporting_split = ensure_dir(supporting / "split_panels")
    remove_known_outputs(
        presentation,
        [
            "fig_01_strategy_learning_entropy.png",
            "fig_02_swarm_size_congestion.png",
            "fig_03_throughput_scaling.png",
            "fig_07_food_spawn_rate_effects.png",
            "slide1_abm_mechanism_schematic.png",
            "slide1_strategy_learning_entropy.png",
            "slide2_swarm_size_congestion.png",
            "slide2_throughput_scaling.png",
            "slide2_food_spawn_rate_effects.png",
        ],
    )
    remove_known_outputs(
        supporting,
        [
            "fig_rest_time_energy_tradeoff.png",
            "fig_sector_score_evolution.png",
            "fig_food_spawn_rate_effects.png",
        ],
    )
    remove_known_outputs(
        data,
        [
            "fig_01_strategy_learning_entropy.csv",
            "fig_02_swarm_size_congestion.csv",
            "fig_03_throughput_scaling.csv",
            "fig_07_food_spawn_rate_effects.csv",
            "fig_rest_time_energy_tradeoff.csv",
            "fig_sector_score_evolution.csv",
            "strategy_learning_entropy.csv",
            "swarm_size_congestion.csv",
            "throughput_scaling.csv",
            "food_spawn_rate_effects.csv",
            "rest_time_energy_tradeoff.csv",
            "sector_score_evolution.csv",
        ],
    )
    seconds = float(seconds if seconds is not None else 1500.0)
    seed0 = int(cfg.get("run", "random_seed", default=7))
    paths: list[Path] = []

    paths.append(plot_abm_mechanism_schematic(presentation / "slide1_abm_mechanism_schematic.png"))

    rest = rest_time_energy_tradeoff(cfg, seconds, seeds, seed0)
    paths.extend(write_plot_data(rest, data / "rest_time_energy_tradeoff.csv"))
    paths.append(plot_rest_time_energy_tradeoff(rest, supporting / "fig_rest_time_energy_tradeoff.png"))
    paths.extend(plot_rest_time_energy_tradeoff_split(rest, supporting_split))

    strategy = strategy_learning_entropy(cfg, seconds, seeds, sample_every, seed0)
    paths.extend(write_plot_data(strategy, data / "strategy_learning_entropy.csv"))
    paths.append(plot_strategy_learning_entropy(strategy, presentation / "slide1_strategy_learning_entropy.png"))
    paths.extend(plot_strategy_learning_entropy_split(strategy, presentation_split))

    congestion = congestion_game_data(cfg, seconds, seeds, seed0)
    paths.extend(write_plot_data(congestion, data / "swarm_size_congestion.csv"))
    paths.append(plot_congestion_game(congestion, presentation / "slide2_swarm_size_congestion.png"))
    paths.extend(plot_congestion_game_split(congestion, presentation_split))

    scaling = congestion.copy()
    paths.extend(write_plot_data(scaling, data / "throughput_scaling.csv"))
    paths.append(plot_emergent_scaling(scaling, presentation / "slide2_throughput_scaling.png"))
    paths.extend(plot_emergent_scaling_split(scaling, presentation_split))

    food = food_spawn_check_data(cfg, seconds, seeds, seed0)
    paths.extend(write_plot_data(food, data / "food_spawn_rate_effects.csv"))
    paths.append(plot_food_spawn_check(food, supporting / "fig_food_spawn_rate_effects.png"))
    paths.extend(plot_food_spawn_check_split(food, supporting_split))

    sector = sector_scores_over_time(cfg, seconds, seeds, max(sample_every, seconds / 8.0), seed0)
    paths.extend(write_plot_data(sector, data / "sector_score_evolution.csv"))
    paths.append(plot_sector_heatmap(sector, supporting / "fig_sector_score_evolution.png"))

    return paths


def write_plot_data(data: pd.DataFrame, path: Path) -> list[Path]:
    """Write a plot source CSV and return its path in a list."""
    data.to_csv(path, index=False)
    return [path]


def plot_abm_mechanism_schematic(path: str | Path) -> Path:
    """Create Slide 1A: ABM mechanism schematic.

    Input:
        path is the PNG destination.

    Output:
        Path to the saved schematic figure.
    """
    path = Path(path)
    fig, ax = plt.subplots(figsize=(14.0, 7.2), constrained_layout=True)
    ax.set_axis_off()
    fig.suptitle("Agent-Based Structure of the Swarm Foraging Model", fontsize=18, y=0.98)

    def box(x: float, y: float, w: float, h: float, title: str, body: str, color: str) -> None:
        """Draw one rounded schematic box on the axis."""
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.018,rounding_size=0.025",
            linewidth=1.2,
            edgecolor="#2b2b2b",
            facecolor=color,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h - 0.055, title, ha="center", va="top", fontsize=12.5, fontweight="bold")
        ax.text(x + 0.025, y + h - 0.12, body, ha="left", va="top", fontsize=10.2, linespacing=1.22)

    def arrow(start: tuple[float, float], end: tuple[float, float], rad: float = 0.0) -> None:
        """Draw one arrow between schematic elements."""
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=16,
                linewidth=1.4,
                color="#3a3a3a",
                connectionstyle=f"arc3,rad={rad}",
            )
        )

    box(
        0.04,
        0.62,
        0.25,
        0.25,
        "Individual Robot Agents",
        "Discrete robots with identity\nOwn position, state, timer, energy\nNo central controller",
        "#e8f1fb",
    )
    box(
        0.04,
        0.28,
        0.25,
        0.27,
        "Internal PFSM States",
        "Searching\nGrabbing\nDepositing\nHoming\nResting\nAvoidance",
        "#f3f6fa",
    )
    box(
        0.375,
        0.60,
        0.25,
        0.27,
        "Local Environment Interaction",
        "Food discovery probability\nTarget-loss probability\nEnergy cost/reward\nSpatial sector scores",
        "#edf7ee",
    )
    box(
        0.375,
        0.27,
        0.25,
        0.25,
        "Robot-Robot Interaction",
        "Collision probability\nCongestion from active robots\nEach robot changes others'\npayoff environment",
        "#fff4df",
    )
    box(
        0.70,
        0.58,
        0.26,
        0.29,
        "Strategy Update After Trip",
        "Trip payoff\nRisk/loss-sensitive utility\nPropensity update\nNext search/rest strategy",
        "#f4ecf7",
    )
    box(
        0.70,
        0.24,
        0.26,
        0.25,
        "Aggregate Swarm-Level Behaviour",
        "Energy\nThroughput\nCollision rate\nStrategy entropy",
        "#f0f0f0",
    )

    arrow((0.165, 0.62), (0.165, 0.55))
    arrow((0.29, 0.74), (0.375, 0.74))
    arrow((0.29, 0.42), (0.375, 0.42))
    arrow((0.625, 0.73), (0.70, 0.73))
    arrow((0.625, 0.40), (0.70, 0.40))
    arrow((0.83, 0.58), (0.83, 0.49))
    arrow((0.70, 0.62), (0.29, 0.34), rad=0.18)

    ax.text(
        0.50,
        0.08,
        "Local state transitions and local events are measured at robot level; system-level patterns emerge from their repeated interaction.",
        ha="center",
        va="center",
        fontsize=11,
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def copy_config(
    cfg: Config,
    *,
    n_robots: int | None = None,
    food_multiplier: float | None = None,
) -> Config:
    """Return a copied config with observational sweep settings changed.

    Inputs:
        cfg is the base configuration. n_robots changes paper.n_robots when
        provided. food_multiplier scales food.growth_rate_s when provided.

    Output:
        A copied Config for a run.
    """
    raw = deepcopy(cfg.raw)
    if n_robots is not None:
        raw["paper"]["n_robots"] = int(n_robots)
    if food_multiplier is not None:
        raw["food"]["growth_rate_s"] = float(cfg.get("food", "growth_rate_s")) * float(food_multiplier)
    return Config(raw=raw, path=cfg.path)


def run_observed_model(
    cfg: Config,
    seconds: float,
    seed: int,
    rest_time_s: float | None = None,
) -> dict[str, float]:
    """Run the micro model and return read-only summary metrics.

    Inputs:
        cfg is the model configuration, seconds is the simulated duration, seed
        initializes the run, and rest_time_s optionally changes the configured
        rest-time argument used by the existing model.

    Output:
        Energy, throughput, collision, activity, strategy, and deposit metrics.
    """
    model = MicroModel(cfg, rest_time_s=rest_time_s, seed=seed)
    steps = model.world.steps(seconds)
    final_window_start = int(0.75 * steps)
    gamma_r_values: list[float] = []
    active_values: list[float] = []
    ratio_values: list[float] = []

    for step in range(steps):
        stats = model.step()
        gamma_r_values.append(float(stats["gamma_r"]))
        if step >= final_window_start:
            active_values.append(float(stats["active_fraction"]))
            ratio_values.append(float(stats["mean_weighted_search_rest_ratio"]))

    throughput = model.total_completed_deposit / max(seconds, model.world.dt)
    n_robots = max(1, model.world.n_robots)
    return {
        "final_net_energy": float(model.energy),
        "final_net_energy_per_robot": float(model.energy / n_robots),
        "throughput": float(throughput),
        "throughput_per_robot": float(throughput / n_robots),
        "mean_collision_probability": float(np.mean(gamma_r_values)) if gamma_r_values else 0.0,
        "mean_active_fraction": float(np.mean(active_values)) if active_values else 0.0,
        "mean_weighted_search_rest_ratio": float(np.mean(ratio_values)) if ratio_values else 0.0,
        "total_completed_deposit": float(model.total_completed_deposit),
        "total_collisions": float(model.total_collisions),
    }


def rest_time_energy_tradeoff(cfg: Config, seconds: float, seeds: int, seed0: int) -> pd.DataFrame:
    """Measure energy, throughput, and activity across rest-time settings.

    Inputs:
        cfg is the model configuration, seconds is run length, seeds is the
        number of repeated seeds, and seed0 is the first seed.

    Output:
        Summary DataFrame for Figure 1.
    """
    rows: list[dict[str, float]] = []
    for rest_time_s in [float(value) for value in cfg.get("behaviour", "rest_times_s")]:
        metrics = [run_observed_model(cfg, seconds, seed0 + offset, rest_time_s) for offset in range(seeds)]
        rows.append(summary_row({"rest_time_s": rest_time_s}, metrics))
    return pd.DataFrame(rows)


def summary_row(prefix: dict[str, float], metrics: list[dict[str, float]]) -> dict[str, float]:
    """Summarize repeated metrics with mean and standard deviation columns."""
    row = dict(prefix)
    for key in metrics[0]:
        values = [metric[key] for metric in metrics]
        row[f"mean_{key}"] = float(np.mean(values))
        row[f"std_{key}"] = float(np.std(values, ddof=0))
    row["seeds"] = float(len(metrics))
    return row


def plot_rest_time_energy_tradeoff(data: pd.DataFrame, path: str | Path) -> Path:
    """Create the supporting rest-time effects figure."""
    path = Path(path)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    fig.suptitle("Effect of Rest-Time Setting on Swarm Energy and Activity")
    axes[0].set_title("A. Final net energy per robot")
    axes[0].errorbar(
        data["rest_time_s"],
        data["mean_final_net_energy_per_robot"],
        yerr=data["std_final_net_energy_per_robot"],
        marker="o",
        capsize=3,
        color="tab:blue",
        label="Mean across seeds",
    )
    axes[0].set_xlabel("Rest-time setting (s)")
    axes[0].set_ylabel("Final net energy per robot")
    axes[0].legend(frameon=True, edgecolor="black")

    axes[1].set_title("B. Throughput per robot and active fraction")
    line1 = axes[1].errorbar(
        data["rest_time_s"],
        data["mean_throughput_per_robot"],
        yerr=data["std_throughput_per_robot"],
        marker="o",
        capsize=3,
        color="tab:green",
        label="Throughput per robot",
    )
    axes[1].set_xlabel("Rest-time setting (s)")
    axes[1].set_ylabel("Throughput per robot (items s$^{-1}$ robot$^{-1}$)")
    twin = axes[1].twinx()
    line2 = twin.errorbar(
        data["rest_time_s"],
        data["mean_mean_active_fraction"],
        yerr=data["std_mean_active_fraction"],
        marker="s",
        capsize=3,
        color="tab:orange",
        label="Mean active fraction",
    )
    twin.set_ylabel("Mean active fraction")
    axes[1].legend([line1, line2], ["Throughput per robot", "Mean active fraction"], frameon=True, edgecolor="black")
    for axis in axes:
        axis.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_rest_time_energy_tradeoff_split(data: pd.DataFrame, out_dir: str | Path) -> list[Path]:
    """Create separate supporting panels for the rest-time tradeoff figure.

    Input:
        data is the rest-time summary DataFrame and out_dir receives PNG files.

    Output:
        Paths to the written single-panel figures.
    """
    out = ensure_dir(out_dir)
    paths = [
        plot_errorbar_panel(
            data,
            "rest_time_s",
            "mean_final_net_energy_per_robot",
            "std_final_net_energy_per_robot",
            out / "rest_time_final_net_energy_per_robot.png",
            "Effect of Rest-Time Setting on Final Net Energy per Robot",
            "Rest-time setting (s)",
            "Final net energy per robot",
        ),
        plot_errorbar_panel(
            data,
            "rest_time_s",
            "mean_throughput_per_robot",
            "std_throughput_per_robot",
            out / "rest_time_throughput_per_robot.png",
            "Effect of Rest-Time Setting on Throughput per Robot",
            "Rest-time setting (s)",
            "Throughput per robot (items s$^{-1}$ robot$^{-1}$)",
            color="tab:green",
        ),
        plot_errorbar_panel(
            data,
            "rest_time_s",
            "mean_mean_active_fraction",
            "std_mean_active_fraction",
            out / "rest_time_active_fraction.png",
            "Effect of Rest-Time Setting on Mean Active Fraction",
            "Rest-time setting (s)",
            "Mean active fraction",
            color="tab:orange",
        ),
    ]
    return paths


def strategy_learning_entropy(
    cfg: Config,
    seconds: float,
    seeds: int,
    sample_every: float,
    seed0: int,
) -> pd.DataFrame:
    """Sample propensity-weighted strategy ratio and entropy over time.

    Inputs:
        cfg is the model configuration, seconds is run length, seeds is repeated
        runs, sample_every is the sampling interval, and seed0 is the first seed.

    Output:
        Seed-averaged time series for Figure 2.
    """
    traces: list[pd.DataFrame] = []
    for offset in range(seeds):
        model = MicroModel(cfg, seed=seed0 + offset)
        steps = model.world.steps(seconds)
        stride = max(1, model.world.steps(sample_every))
        rows: list[dict[str, float]] = []
        for step in range(steps):
            stats = model.step()
            if step % stride != 0 and step != steps - 1:
                continue
            rows.append(
                {
                    "time_s": (step + 1) * model.world.dt,
                    "weighted_ratio": float(stats["mean_weighted_search_rest_ratio"]),
                    "strategy_entropy": mean_strategy_entropy(model),
                    "seed": float(seed0 + offset),
                }
            )
        traces.append(pd.DataFrame(rows))
    raw = pd.concat(traces, ignore_index=True)
    return raw.groupby("time_s", as_index=False).agg(
        mean_weighted_ratio=("weighted_ratio", "mean"),
        std_weighted_ratio=("weighted_ratio", lambda values: float(values.std(ddof=0))),
        mean_strategy_entropy=("strategy_entropy", "mean"),
        std_strategy_entropy=("strategy_entropy", lambda values: float(values.std(ddof=0))),
    )


def mean_strategy_entropy(model: MicroModel) -> float:
    """Return mean strategy entropy across agents in nats."""
    entropies: list[float] = []
    for agent in model.agents:
        total = sum(agent.propensities)
        if total <= 0.0:
            entropies.append(0.0)
            continue
        probs = np.asarray([propensity / total for propensity in agent.propensities], dtype=float)
        probs = probs[probs > 0.0]
        entropies.append(float(-(probs * np.log(probs)).sum()))
    return float(np.mean(entropies)) if entropies else 0.0


def plot_strategy_learning_entropy(data: pd.DataFrame, path: str | Path) -> Path:
    """Create presentation Figure 1: strategy ratio and entropy over time."""
    path = Path(path)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    fig.suptitle("Temporal Evolution of Search/Rest Strategy Propensities")
    axes[0].set_title("A. Mean propensity-weighted search/rest ratio")
    axes[0].plot(data["time_s"], data["mean_weighted_ratio"], color="tab:blue", label="Mean across seeds")
    axes[0].fill_between(
        data["time_s"],
        data["mean_weighted_ratio"] - data["std_weighted_ratio"],
        data["mean_weighted_ratio"] + data["std_weighted_ratio"],
        color="tab:blue",
        alpha=0.18,
        label="Mean +/- standard deviation",
    )
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Search/rest ratio")

    axes[1].set_title("B. Mean strategy entropy")
    axes[1].plot(data["time_s"], data["mean_strategy_entropy"], color="tab:purple", label="Mean across seeds")
    axes[1].fill_between(
        data["time_s"],
        data["mean_strategy_entropy"] - data["std_strategy_entropy"],
        data["mean_strategy_entropy"] + data["std_strategy_entropy"],
        color="tab:purple",
        alpha=0.18,
        label="Mean +/- standard deviation",
    )
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Strategy entropy (nats)")
    for axis in axes:
        axis.grid(True, linestyle="--", alpha=0.35)
        axis.legend(frameon=True, edgecolor="black")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_strategy_learning_entropy_split(data: pd.DataFrame, out_dir: str | Path) -> list[Path]:
    """Create separate presentation panels for strategy ratio and entropy.

    Input:
        data is the strategy-learning time series and out_dir receives PNG files.

    Output:
        Paths to the written single-panel figures.
    """
    out = ensure_dir(out_dir)
    specs = [
        (
            "mean_weighted_ratio",
            "std_weighted_ratio",
            "slide1_strategy_weighted_search_rest_ratio.png",
            "Temporal Evolution of Propensity-Weighted Search/Rest Ratio",
            "Search/rest ratio",
            "tab:blue",
        ),
        (
            "mean_strategy_entropy",
            "std_strategy_entropy",
            "slide1_strategy_entropy.png",
            "Temporal Evolution of Strategy Entropy",
            "Strategy entropy (nats)",
            "tab:purple",
        ),
    ]
    paths: list[Path] = []
    for y_col, err_col, filename, title, ylabel, color in specs:
        path = out / filename
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        ax.set_title(title)
        ax.plot(data["time_s"], data[y_col], color=color, label="Mean across seeds")
        ax.fill_between(
            data["time_s"],
            data[y_col] - data[err_col],
            data[y_col] + data[err_col],
            color=color,
            alpha=0.18,
            label="Mean +/- standard deviation",
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(frameon=True, edgecolor="black")
        fig.tight_layout()
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def congestion_game_data(cfg: Config, seconds: float, seeds: int, seed0: int) -> pd.DataFrame:
    """Measure congestion-game metrics across swarm sizes."""
    rows: list[dict[str, float]] = []
    for n_robots in SWARM_SIZES:
        run_cfg = copy_config(cfg, n_robots=n_robots)
        metrics = [run_observed_model(run_cfg, seconds, seed0 + offset) for offset in range(seeds)]
        rows.append(summary_row({"n_robots": float(n_robots)}, metrics))
    return pd.DataFrame(rows)


def plot_congestion_game(data: pd.DataFrame, path: str | Path) -> Path:
    """Create presentation Figure 2: swarm-size effects on payoff and congestion."""
    path = Path(path)
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0))
    fig.suptitle("Effect of Swarm Size on Energy, Collision Probability, and Throughput")
    panels = [
        ("A. Final net energy per robot", "mean_final_net_energy_per_robot", "std_final_net_energy_per_robot", "Final net energy per robot"),
        ("B. Mean collision probability", "mean_mean_collision_probability", "std_mean_collision_probability", "Mean collision probability"),
        ("C. Throughput per robot", "mean_throughput_per_robot", "std_throughput_per_robot", "Throughput per robot (items s$^{-1}$ robot$^{-1}$)"),
    ]
    for axis, (title, y_col, err_col, ylabel) in zip(axes, panels):
        axis.set_title(title)
        axis.errorbar(data["n_robots"], data[y_col], yerr=data[err_col], marker="o", capsize=3, color="tab:blue")
        axis.set_xscale("log", base=2)
        axis.set_xticks(data["n_robots"])
        axis.set_xticklabels([str(int(value)) for value in data["n_robots"]])
        axis.set_xlabel("Number of robots")
        axis.set_ylabel(ylabel)
        axis.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_congestion_game_split(data: pd.DataFrame, out_dir: str | Path) -> list[Path]:
    """Create separate presentation panels for the congestion figure.

    Input:
        data is the swarm-size summary DataFrame and out_dir receives PNG files.

    Output:
        Paths to the written single-panel figures.
    """
    out = ensure_dir(out_dir)
    return [
        plot_errorbar_panel(
            data,
            "n_robots",
            "mean_final_net_energy_per_robot",
            "std_final_net_energy_per_robot",
            out / "slide2_swarm_size_energy_per_robot.png",
            "Effect of Swarm Size on Final Net Energy per Robot",
            "Number of robots",
            "Final net energy per robot",
            xscale="log2",
        ),
        plot_errorbar_panel(
            data,
            "n_robots",
            "mean_mean_collision_probability",
            "std_mean_collision_probability",
            out / "slide2_swarm_size_collision_probability.png",
            "Effect of Swarm Size on Mean Collision Probability",
            "Number of robots",
            "Mean collision probability",
            xscale="log2",
        ),
        plot_errorbar_panel(
            data,
            "n_robots",
            "mean_throughput_per_robot",
            "std_throughput_per_robot",
            out / "slide2_swarm_size_throughput_per_robot.png",
            "Effect of Swarm Size on Throughput per Robot",
            "Number of robots",
            "Throughput per robot (items s$^{-1}$ robot$^{-1}$)",
            xscale="log2",
        ),
    ]


def plot_emergent_scaling(data: pd.DataFrame, path: str | Path) -> Path:
    """Create presentation Figure 3: throughput scaling with population size."""
    path = Path(path)
    x = data["n_robots"].to_numpy(dtype=float)
    y = data["mean_throughput"].to_numpy(dtype=float)
    beta, intercept = fit_scaling_exponent(x, y)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    fig.suptitle("Scaling of Swarm Throughput with Population Size")
    axes[0].set_title("A. Total throughput as a function of swarm size")
    axes[0].errorbar(x, y, yerr=data["std_throughput"], fmt="o", capsize=3, color="tab:green", label="Simulation mean")
    if np.isfinite(beta):
        axes[0].plot(x, np.exp(intercept) * x**beta, "--", color="black", label="Power-law fit")
        axes[0].text(0.05, 0.92, f"β = {beta:.2f}", transform=axes[0].transAxes)
    if y[0] > 0.0:
        axes[0].plot(x, y[0] * (x / x[0]), ":", color="0.4", label="Linear scaling reference")
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Number of robots")
    axes[0].set_ylabel("Total throughput (items s$^{-1}$)")
    axes[0].legend(frameon=True, edgecolor="black")

    axes[1].set_title("B. Per-robot throughput as a function of swarm size")
    axes[1].errorbar(
        x,
        data["mean_throughput_per_robot"],
        yerr=data["std_throughput_per_robot"],
        fmt="o-",
        capsize=3,
        color="tab:blue",
        label="Simulation mean",
    )
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("Number of robots")
    axes[1].set_ylabel("Throughput per robot (items s$^{-1}$ robot$^{-1}$)")
    axes[1].legend(frameon=True, edgecolor="black")
    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels([str(int(value)) for value in x])
        axis.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_emergent_scaling_split(data: pd.DataFrame, out_dir: str | Path) -> list[Path]:
    """Create separate presentation panels for the throughput scaling figure.

    Input:
        data is the swarm-size summary DataFrame and out_dir receives PNG files.

    Output:
        Paths to the written single-panel figures.
    """
    out = ensure_dir(out_dir)
    x = data["n_robots"].to_numpy(dtype=float)
    y = data["mean_throughput"].to_numpy(dtype=float)
    beta, intercept = fit_scaling_exponent(x, y)

    total_path = out / "slide2_total_throughput_scaling.png"
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.set_title("Scaling of Total Swarm Throughput with Population Size")
    ax.errorbar(x, y, yerr=data["std_throughput"], fmt="o", capsize=3, color="tab:green", label="Simulation mean")
    if np.isfinite(beta):
        ax.plot(x, np.exp(intercept) * x**beta, "--", color="black", label="Power-law fit")
        ax.text(0.05, 0.92, f"β = {beta:.2f}", transform=ax.transAxes)
    if y[0] > 0.0:
        ax.plot(x, y[0] * (x / x[0]), ":", color="0.4", label="Linear scaling reference")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(value)) for value in x])
    ax.set_xlabel("Number of robots")
    ax.set_ylabel("Total throughput (items s$^{-1}$)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=True, edgecolor="black")
    fig.tight_layout()
    fig.savefig(total_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    per_robot_path = out / "slide2_per_robot_throughput_scaling.png"
    plot_errorbar_panel(
        data,
        "n_robots",
        "mean_throughput_per_robot",
        "std_throughput_per_robot",
        per_robot_path,
        "Scaling of Per-Robot Throughput with Population Size",
        "Number of robots",
        "Throughput per robot (items s$^{-1}$ robot$^{-1}$)",
        xscale="log2",
    )
    return [total_path, per_robot_path]


def fit_scaling_exponent(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit log(y) = intercept + beta log(x) for positive values."""
    mask = (x > 0.0) & (y > 0.0)
    if mask.sum() < 2:
        return float("nan"), float("nan")
    beta, intercept = np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)
    return float(beta), float(intercept)


def food_spawn_check_data(cfg: Config, seconds: float, seeds: int, seed0: int) -> pd.DataFrame:
    """Measure food-spawn sensitivity with copied configurations."""
    rows: list[dict[str, float]] = []
    for multiplier in FOOD_MULTIPLIERS:
        run_cfg = copy_config(cfg, food_multiplier=multiplier)
        metrics = [run_observed_model(run_cfg, seconds, seed0 + offset) for offset in range(seeds)]
        rows.append(summary_row({"food_spawn_multiplier": float(multiplier)}, metrics))
    return pd.DataFrame(rows)


def plot_food_spawn_check(data: pd.DataFrame, path: str | Path) -> Path:
    """Create presentation Figure 7: food-spawn effects on energy and throughput."""
    path = Path(path)
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.9))
    fig.suptitle("Effect of Food Spawn Rate on Swarm Energy and Throughput")
    panels = [
        ("A. Final net energy per robot", "mean_final_net_energy_per_robot", "std_final_net_energy_per_robot", "Final net energy per robot"),
        ("B. Throughput per robot", "mean_throughput_per_robot", "std_throughput_per_robot", "Throughput per robot (items s$^{-1}$ robot$^{-1}$)"),
        ("C. Mean collision probability", "mean_mean_collision_probability", "std_mean_collision_probability", "Mean collision probability"),
    ]
    for axis, (title, y_col, err_col, ylabel) in zip(axes, panels):
        axis.set_title(title)
        axis.errorbar(data["food_spawn_multiplier"], data[y_col], yerr=data[err_col], marker="o", capsize=3)
        axis.set_xscale("log", base=2)
        axis.set_xticks(data["food_spawn_multiplier"])
        axis.set_xticklabels([f"{value:g}x" for value in data["food_spawn_multiplier"]])
        axis.set_xlabel("Food spawn-rate multiplier")
        axis.set_ylabel(ylabel)
        axis.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_food_spawn_check_split(data: pd.DataFrame, out_dir: str | Path) -> list[Path]:
    """Create separate supporting panels for food-spawn effects.

    Input:
        data is the food-spawn summary DataFrame and out_dir receives PNG files.

    Output:
        Paths to the written single-panel figures.
    """
    out = ensure_dir(out_dir)
    return [
        plot_errorbar_panel(
            data,
            "food_spawn_multiplier",
            "mean_final_net_energy_per_robot",
            "std_final_net_energy_per_robot",
            out / "food_spawn_energy_per_robot.png",
            "Effect of Food Spawn Rate on Final Net Energy per Robot",
            "Food spawn-rate multiplier",
            "Final net energy per robot",
            xscale="log2",
            xtick_suffix="x",
        ),
        plot_errorbar_panel(
            data,
            "food_spawn_multiplier",
            "mean_throughput_per_robot",
            "std_throughput_per_robot",
            out / "food_spawn_throughput_per_robot.png",
            "Effect of Food Spawn Rate on Throughput per Robot",
            "Food spawn-rate multiplier",
            "Throughput per robot (items s$^{-1}$ robot$^{-1}$)",
            xscale="log2",
            xtick_suffix="x",
            color="tab:green",
        ),
        plot_errorbar_panel(
            data,
            "food_spawn_multiplier",
            "mean_mean_collision_probability",
            "std_mean_collision_probability",
            out / "food_spawn_collision_probability.png",
            "Effect of Food Spawn Rate on Mean Collision Probability",
            "Food spawn-rate multiplier",
            "Mean collision probability",
            xscale="log2",
            xtick_suffix="x",
            color="tab:orange",
        ),
    ]


def plot_errorbar_panel(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    err_col: str,
    path: str | Path,
    title: str,
    xlabel: str,
    ylabel: str,
    *,
    xscale: str | None = None,
    xtick_suffix: str = "",
    color: str = "tab:blue",
) -> Path:
    """Create one reusable error-bar panel from a summary DataFrame.

    Inputs:
        data is a summary DataFrame, x_col/y_col/err_col name the plotted
        columns, path is the PNG destination, and labels describe the axes.

    Output:
        Path to the written single-panel figure.
    """
    path = Path(path)
    x = data[x_col].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.set_title(title)
    ax.errorbar(x, data[y_col], yerr=data[err_col], marker="o", capsize=3, color=color, label="Mean across seeds")
    if xscale == "log2":
        ax.set_xscale("log", base=2)
    ax.set_xticks(x)
    if xtick_suffix:
        ax.set_xticklabels([f"{value:g}{xtick_suffix}" for value in x])
    else:
        ax.set_xticklabels([str(int(value)) if float(value).is_integer() else f"{value:g}" for value in x])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=True, edgecolor="black")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def sector_scores_over_time(cfg: Config, seconds: float, seeds: int, sample_every: float, seed0: int) -> pd.DataFrame:
    """Sample mean sector scores across agents and seeds over time."""
    traces: list[pd.DataFrame] = []
    for offset in range(seeds):
        model = MicroModel(cfg, seed=seed0 + offset)
        steps = model.world.steps(seconds)
        stride = max(1, model.world.steps(sample_every))
        rows: list[dict[str, float]] = []
        for step in range(steps):
            model.step()
            if step % stride != 0 and step != steps - 1:
                continue
            scores = np.asarray([agent.sector_scores for agent in model.agents], dtype=float).mean(axis=0)
            rows.append(
                {
                    "time_s": (step + 1) * model.world.dt,
                    "sector_0": float(scores[0]),
                    "sector_1": float(scores[1]),
                    "sector_2": float(scores[2]),
                    "sector_3": float(scores[3]),
                }
            )
        traces.append(pd.DataFrame(rows))
    raw = pd.concat(traces, ignore_index=True)
    return raw.groupby("time_s", as_index=False).mean()


def plot_sector_heatmap(data: pd.DataFrame, path: str | Path) -> Path:
    """Create the supporting mean sector-score heatmap."""
    path = Path(path)
    values = data[["sector_0", "sector_1", "sector_2", "sector_3"]].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    image = ax.imshow(values, aspect="auto", cmap="viridis")
    ax.set_title("Mean sector score by arena sector")
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["Sector 0", "Sector 1", "Sector 2", "Sector 3"])
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels([f"{value:.0f}" for value in data["time_s"]])
    ax.set_xlabel("Arena sector")
    ax.set_ylabel("Time (s)")
    fig.suptitle("Temporal Evolution of Mean Sector Scores")
    fig.colorbar(image, ax=ax, label="Mean sector score")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path
