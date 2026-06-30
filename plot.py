"""Create the static plots and CSV traces used by the project.

Inputs:
    A loaded Config object, an output folder, and optionally a mean rest time.

Outputs:
    PNG figures and CSV files for the paper reproduction plots and the added
    learning/cognitive-behaviour plots.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

_MPL_CONFIG_DIR = Path(tempfile.gettempdir()) / "swarm_foraging_mpl"
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
_CACHE_DIR = Path(tempfile.gettempdir()) / "swarm_foraging_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from config import Config
from macro import MacroModel
from agents import MicroModel


def ensure_dir(path: str | Path) -> Path:
    """Create an output folder if needed and return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _paper_style() -> None:
    """Apply the compact plotting style used by the paper-style figures."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 9,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "axes.grid": False,
        }
    )


def _stride_steps(cfg: Config) -> int:
    """Convert the configured CSV stride from seconds to model steps."""
    stride_s = float(cfg.get("run", "csv_stride_s", default=5.0))
    return max(1, int(round(stride_s / cfg.dt)))


def _macro_trace(cfg: Config, rest_time_s: float, stride: int | None = None) -> pd.DataFrame:
    """Run the macro model and return its sampled trace."""
    return MacroModel(cfg, rest_time_s=rest_time_s).run(stride=stride or _stride_steps(cfg)).trace


def _final_stride(cfg: Config) -> int:
    """Return a stride that keeps only the first and final sampled rows."""
    return max(1, int(round(cfg.duration_s / cfg.dt)) + 1)


def _padded_limits(values: np.ndarray) -> tuple[float, float]:
    """Return readable y-axis limits for data that may be positive or negative."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return -1.0, 1.0
    ymin = float(values.min())
    ymax = float(values.max())
    if abs(ymax - ymin) < 1e-12:
        pad = max(1.0, abs(ymax) * 0.1)
    else:
        pad = 0.12 * (ymax - ymin)
    return ymin - pad, ymax + pad


def write_macro_csv(cfg: Config, out_dir: str | Path, rest_time_s: float | None = None) -> Path:
    """Write one macro trace CSV for the requested rest time."""
    out = ensure_dir(out_dir)
    tau_r = float(rest_time_s if rest_time_s is not None else cfg.get("behaviour", "default_rest_time_s"))
    trace = _macro_trace(cfg, tau_r)
    path = out / f"macro_trace_tau_r_{int(tau_r)}.csv"
    trace.to_csv(path, index=False)
    return path


def plot_fig8(cfg: Config, out_dir: str | Path) -> tuple[Path, Path]:
    """Write Fig. 8 from actual macro and repeated micro simulation runs."""
    _paper_style()
    out = ensure_dir(out_dir)
    rest_times = np.array([float(x) for x in cfg.get("behaviour", "rest_times_s")], dtype=float)
    final_stride = _final_stride(cfg)
    runs = int(cfg.get("run", "micro_runs", default=10))
    seed0 = int(cfg.get("run", "random_seed", default=7))
    rows: list[dict[str, float]] = []

    for tau_r in rest_times:
        macro_trace = MacroModel(cfg, rest_time_s=tau_r).run(stride=final_stride).trace
        micro_energy = []
        for seed in range(seed0, seed0 + runs):
            trace = MicroModel(cfg, rest_time_s=tau_r, seed=seed).run(stride=final_stride)
            micro_energy.append(float(trace["energy"].iloc[-1]))
        rows.append(
            {
                "rest_time_s": tau_r,
                "macro_energy": float(macro_trace["energy"].iloc[-1]),
                "micro_energy_mean": float(np.mean(micro_energy)),
                "micro_energy_std": float(np.std(micro_energy, ddof=0)),
                "micro_runs": float(runs),
            }
        )

    data = pd.DataFrame(rows)
    data = data.assign(
        macro_energy_1e5=data["macro_energy"] / 1e5,
        micro_energy_mean_1e5=data["micro_energy_mean"] / 1e5,
        micro_energy_std_1e5=data["micro_energy_std"] / 1e5,
    )
    csv_path = out / "fig8_macro_summary.csv"
    data.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(6.1, 4.1))
    micro_handle = ax.errorbar(
        data["rest_time_s"],
        data["micro_energy_mean_1e5"],
        yerr=data["micro_energy_std_1e5"],
        fmt="o--",
        color="blue",
        ecolor="black",
        elinewidth=0.7,
        capsize=3,
        markersize=5,
        linewidth=0.7,
        markerfacecolor="white",
        markeredgewidth=0.8,
        label="micro simulation",
    )
    macro_handle, = ax.plot(
        data["rest_time_s"],
        data["macro_energy_1e5"],
        color="black",
        linewidth=0.8,
        label="macro model",
    )
    peak_i = int(np.argmax(data["macro_energy_1e5"].to_numpy()))
    peak_tau = float(data["rest_time_s"].iloc[peak_i])
    peak_y = float(data["macro_energy_1e5"].iloc[peak_i])
    ax.axvline(peak_tau, color="black", linestyle="--", linewidth=0.55, dashes=(7, 7))
    ax.axhline(peak_y, color="black", linestyle="--", linewidth=0.55, dashes=(7, 7))
    all_y = np.concatenate(
        [
            data["macro_energy_1e5"].to_numpy(),
            (data["micro_energy_mean_1e5"] - data["micro_energy_std_1e5"]).to_numpy(),
            (data["micro_energy_mean_1e5"] + data["micro_energy_std_1e5"]).to_numpy(),
        ]
    )
    ymin, ymax = _padded_limits(all_y)
    ax.set_xlim(-20, 220)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks([0, 40, 80, 120, 160, 200])
    ax.set_xlabel(r"$\tau_r$  (seconds)")
    ax.set_ylabel(r"energy of swarm  ($10^5$ units)")
    ax.legend([micro_handle, macro_handle], ["micro simulation", "macro model"], loc="best", frameon=True, fancybox=False, edgecolor="black")
    fig.tight_layout(pad=0.8)
    png_path = out / "fig8_macro_energy.png"
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    return png_path, csv_path


def _micro_runs(cfg: Config, rest_time_s: float, stride: int) -> list[pd.DataFrame]:
    """Run the micro model over the configured seed range."""
    runs = int(cfg.get("run", "micro_runs", default=10))
    seed0 = int(cfg.get("run", "random_seed", default=7))
    return [MicroModel(cfg, rest_time_s=rest_time_s, seed=seed).run(stride=stride) for seed in range(seed0, seed0 + runs)]


def _micro_mean_std(frames: list[pd.DataFrame], cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average the same columns across repeated micro-model runs."""
    min_len = min(len(f) for f in frames)
    time = frames[0]["time_s"].to_numpy()[:min_len]
    mean = pd.DataFrame({"time_s": time})
    std = pd.DataFrame({"time_s": time})
    for col in cols:
        stack = np.vstack([f[col].to_numpy()[:min_len] for f in frames])
        mean[col] = stack.mean(axis=0)
        std[col] = stack.std(axis=0)
    return mean, std


def _save_fig9(cfg: Config, out: Path, tau_r: float, macro: pd.DataFrame, micro_mean: pd.DataFrame, micro_std: pd.DataFrame) -> Path:
    """Draw Figure 9 from already-computed macro and micro energy traces."""
    _paper_style()
    fig, ax = plt.subplots(figsize=(6.1, 4.0))
    sim_handle = ax.errorbar(
        micro_mean["time_s"],
        micro_mean["energy"] / 1e5,
        yerr=micro_std["energy"] / 1e5,
        color="0.65",
        linewidth=0.45,
        errorevery=max(1, len(micro_mean) // 35),
        capsize=1.5,
        label="simulation",
    )
    model_handle, = ax.plot(macro["time_s"], macro["energy"] / 1e5, color="red", linewidth=0.9, label="model")
    ax.set_xlim(0, cfg.duration_s)
    all_y = np.concatenate(
        [
            macro["energy"].to_numpy() / 1e5,
            (micro_mean["energy"] - micro_std["energy"]).to_numpy() / 1e5,
            (micro_mean["energy"] + micro_std["energy"]).to_numpy() / 1e5,
        ]
    )
    ax.set_ylim(*_padded_limits(all_y))
    ax.set_xticks(np.arange(0, cfg.duration_s + 1, 4000))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"energy of swarm  ($10^5$ units)")
    ax.legend([sim_handle, model_handle], ["simulation", "model"], loc="upper left", frameon=True, fancybox=False, edgecolor="black")
    fig.tight_layout(pad=0.8)
    path = out / f"fig9_energy_tau_r_{int(tau_r)}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_fig9(cfg: Config, out_dir: str | Path, rest_time_s: float | None = None) -> Path:
    """Write Figure 9 for energy over time at one rest time."""
    out = ensure_dir(out_dir)
    tau_r = float(rest_time_s if rest_time_s is not None else cfg.get("behaviour", "default_rest_time_s"))
    stride = _stride_steps(cfg)
    macro = _macro_trace(cfg, tau_r, stride)
    micro_mean, micro_std = _micro_mean_std(_micro_runs(cfg, tau_r, stride), ["energy"])
    return _save_fig9(cfg, out, tau_r, macro, micro_mean, micro_std)


def _save_fig10(cfg: Config, out: Path, tau_r: float, macro: pd.DataFrame, micro_mean: pd.DataFrame) -> Path:
    """Draw Figure 10 from already-computed macro and averaged micro state traces."""
    _paper_style()
    fig, ax = plt.subplots(figsize=(6.1, 4.05))
    colors = {"searching": "tab:red", "resting": "tab:green", "homing": "tab:blue"}
    for state, color in colors.items():
        ax.plot(micro_mean["time_s"], micro_mean[state], color=color, linewidth=0.35, alpha=0.75)
        ax.plot(macro["time_s"], macro[state], color=color, linestyle="--", linewidth=1.0)
    ax.set_xlim(0, cfg.duration_s)
    ax.set_ylim(0, 8)
    ax.set_xticks([0, 5000, 10000, 15000, 20000])
    ax.set_yticks(np.arange(0, 9, 1))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Robots")
    state_handles = [
        Line2D([0], [0], color=color, linewidth=1.2, label=state.title())
        for state, color in colors.items()
    ]
    series_handles = [
        Line2D([0], [0], color="0.3", linewidth=0.6, label="Micro average"),
        Line2D([0], [0], color="0.3", linestyle="--", linewidth=1.0, label="Macro average"),
    ]
    state_legend = ax.legend(
        handles=state_handles,
        title="State",
        loc="upper left",
        frameon=True,
        fancybox=False,
        edgecolor="black",
    )
    ax.add_artist(state_legend)
    ax.legend(
        handles=series_handles,
        title="Series",
        loc="upper right",
        frameon=True,
        fancybox=False,
        edgecolor="black",
    )
    fig.tight_layout(pad=0.8)
    path = out / f"fig10_states_tau_r_{int(tau_r)}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_fig10(cfg: Config, out_dir: str | Path, rest_time_s: float | None = None) -> Path:
    """Write Figure 10 for selected state populations over time."""
    out = ensure_dir(out_dir)
    tau_r = float(rest_time_s if rest_time_s is not None else cfg.get("behaviour", "default_rest_time_s"))
    stride = _stride_steps(cfg)
    macro = _macro_trace(cfg, tau_r, stride)
    micro_mean, _ = _micro_mean_std(_micro_runs(cfg, tau_r, stride), ["searching", "resting", "homing"])
    return _save_fig10(cfg, out, tau_r, macro, micro_mean)


def plot_fig11_strategy_evolution(cfg: Config, out_dir: str | Path) -> tuple[Path, Path]:
    """Write the learning-strategy propensity plot and its CSV trace.

    Input:
        A Config object and output folder.

    Output:
        fig11_strategy_evolution.png and fig11_strategy_evolution.csv.
    """
    _paper_style()
    out = ensure_dir(out_dir)
    stride = _stride_steps(cfg)
    seed = int(cfg.get("run", "random_seed", default=7))
    model = MicroModel(cfg, seed=seed)
    steps = model.world.steps(cfg.duration_s)
    rows: list[dict[str, float]] = []

    for step in range(steps):
        model.step()
        if step % stride == 0 or step == steps - 1:
            strategies = [int(round(s * model.world.dt)) for s in model.agents[0].strategies]
            propensities = np.array([agent.propensities for agent in model.agents], dtype=float)
            mean_props = propensities.mean(axis=0)
            row = {"time_s": (step + 1) * model.world.dt}
            for strategy_s, value in zip(strategies, mean_props):
                row[f"propensity_tau_r_{strategy_s}s"] = float(value)
            rows.append(row)

    data = pd.DataFrame(rows)
    csv_path = out / "fig11_strategy_evolution.csv"
    data.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(6.1, 4.0))
    colors = ["tab:red", "tab:orange", "tab:green", "tab:blue"]
    for column, color in zip(data.columns[1:], colors):
        label = column.replace("propensity_tau_r_", r"$\tau_r$ ").replace("s", " s")
        ax.plot(data["time_s"], data[column], label=label, color=color, linewidth=1.3)

    ax.set_xlim(0, cfg.duration_s)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mean strategy propensity")
    ax.set_title("Strategy propensity over time")
    ax.legend(loc="upper left", frameon=True, fancybox=False, edgecolor="black")
    fig.tight_layout(pad=0.8)
    png_path = out / "fig11_strategy_evolution.png"
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    return png_path, csv_path


def plot_fig12_energy_divergence(cfg: Config, out_dir: str | Path) -> tuple[Path, Path]:
    """Write the macro-vs-micro energy comparison and its CSV trace.

    Input:
        A Config object and output folder.

    Output:
        fig12_energy_divergence.png and fig12_energy_divergence.csv.
    """
    _paper_style()
    out = ensure_dir(out_dir)
    tau_r = float(cfg.get("behaviour", "default_rest_time_s"))
    stride = _stride_steps(cfg)
    macro = _macro_trace(cfg, tau_r, stride)
    micro_mean, micro_std = _micro_mean_std(_micro_runs(cfg, tau_r, stride), ["energy"])

    min_len = min(len(macro), len(micro_mean))
    data = pd.DataFrame(
        {
            "time_s": macro["time_s"].to_numpy()[:min_len],
            "macro_energy": macro["energy"].to_numpy()[:min_len],
            "micro_energy_mean": micro_mean["energy"].to_numpy()[:min_len],
            "micro_energy_std": micro_std["energy"].to_numpy()[:min_len],
        }
    )
    csv_path = out / "fig12_energy_divergence.csv"
    data.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(6.1, 4.0))
    ax.plot(data["time_s"], data["macro_energy"] / 1e5, label="Macro model", color="black", linestyle="--", linewidth=1.2)
    ax.plot(data["time_s"], data["micro_energy_mean"] / 1e5, label="Micro average", color="tab:purple", linewidth=1.4)
    ax.fill_between(
        data["time_s"],
        (data["micro_energy_mean"] - data["micro_energy_std"]) / 1e5,
        (data["micro_energy_mean"] + data["micro_energy_std"]) / 1e5,
        color="tab:purple",
        alpha=0.18,
        linewidth=0,
    )
    ax.set_xlim(0, cfg.duration_s)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"Net energy ($10^5$ units)")
    ax.set_title("Macro and micro energy over time")
    ax.legend(loc="upper left", frameon=True, fancybox=False, edgecolor="black")
    fig.tight_layout(pad=0.8)
    png_path = out / "fig12_energy_divergence.png"
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    return png_path, csv_path


def make_all_plots(cfg: Config, out_dir: str | Path) -> list[Path]:
    """Write all static plot outputs from the existing plotting workflow."""
    out = ensure_dir(out_dir)
    p8, csv = plot_fig8(cfg, out)
    tau_r = float(cfg.get("behaviour", "default_rest_time_s"))
    stride = _stride_steps(cfg)
    macro = _macro_trace(cfg, tau_r, stride)
    micro_runs = _micro_runs(cfg, tau_r, stride)
    micro_mean, micro_std = _micro_mean_std(
        micro_runs,
        ["energy", "searching", "resting", "homing"],
    )
    p9 = _save_fig9(cfg, out, tau_r, macro, micro_mean[["time_s", "energy"]], micro_std[["time_s", "energy"]])
    p10 = _save_fig10(cfg, out, tau_r, macro, micro_mean[["time_s", "searching", "resting", "homing"]])
    p11, c11 = plot_fig11_strategy_evolution(cfg, out)
    p12, c12 = plot_fig12_energy_divergence(cfg, out)
    macro_csv = out / f"macro_trace_tau_r_{int(tau_r)}.csv"
    macro.to_csv(macro_csv, index=False)
    clustering_plot_path = plot_clustering_comparison(cfg)
    
    return [p8, csv, p9, p10, p11, c11, p12, c12, macro_csv, clustering_plot_path]


def plot_clustering_comparison(cfg):
    # Run 1: without learning (Stap 7 turned off)
    model_off = MicroModel(cfg, use_spatial_learning=False)
    df_off = model_off.run(seconds=4000, stride=4) # kortere duur voor snelle test
    
    # Run 2: with learning (Stap 7 turned on)
    model_on = MicroModel(cfg, use_spatial_learning=True)
    df_on = model_on.run(seconds=4000, stride=4)
    
    # plot the clustering index over time for both runs
    plt.figure(figsize=(10, 5))
    plt.plot(df_off["time_s"], df_off["clustering_index"], label="Without Learning (Homogeneous)", color="gray", linestyle="--")
    plt.plot(df_on["time_s"], df_on["clustering_index"], label="With Learning (Spatial Memory)", color="tab:blue")
    
    plt.title("Swarm Clustering Over Time")
    plt.xlabel("Simulation time (seconds)")
    plt.ylabel("Clustering Index (Standard deviation of sectors)")
    plt.legend()
    plt.grid(True)
    plt.savefig("clustering_comparison.png")
    plt.close()

