"""Run Morris screening sensitivity analysis for the ABM.

Inputs:
    A Config object, output folder, Morris sample count, simulated seconds, and
    repeated seed count.

Outputs:
    morris_raw_runs.csv, morris_sensitivity_summary.csv, a combined presentation
    ranking plot, two supporting ranking plots, and one interaction diagnostic.
"""

from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import tempfile
from typing import Any

_MPL_CONFIG_DIR = Path(tempfile.gettempdir()) / "swarm_foraging_mpl"
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
_CACHE_DIR = Path(tempfile.gettempdir()) / "swarm_foraging_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agents import MicroModel
from config import Config


PARAMETER_BOUNDS = {
    "n_robots": (8, 128),
    "gamma_r_scale": (0.5, 1.5),
    "alpha": (0.5, 1.0),
    "lambda_loss": (1.0, 5.0),
    "congestion_tolerance": (0.01, 0.10),
    "p_new_multiplier": (0.25, 4.0),
    "sector_decay": (0.0, 0.02),
    "sector_reward": (0.25, 2.0),
    "sector_pull": (0.0, 0.40),
}

ROBOT_LEVELS = [8, 16, 32, 64, 128]
MORRIS_OUTPUTS = [
    "final_net_energy",
    "final_net_energy_per_robot",
    "throughput",
    "throughput_per_robot",
    "mean_collision_probability",
    "mean_weighted_search_rest_ratio",
    "mean_active_fraction",
    "total_completed_deposit",
    "total_collisions",
]

IMPORTANT_DIAGNOSTIC_PARAMETERS = {
    "p_new_multiplier",
    "n_robots",
    "gamma_r_scale",
    "lambda_loss",
    "alpha",
    "congestion_tolerance",
}


def output_folders(out_dir: str | Path) -> tuple[Path, Path, Path]:
    """Create and return presentation, supporting, and data folders.

    Input:
        out_dir is the root output folder requested by the CLI.

    Output:
        Paths for presentation plots, supporting plots, and CSV data.
    """
    root = Path(out_dir)
    presentation = root / "presentation_plots"
    supporting = root / "supporting_plots"
    data = root / "data"
    presentation.mkdir(parents=True, exist_ok=True)
    supporting.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    return presentation, supporting, data


def remove_known_outputs(folder: Path, names: list[str]) -> None:
    """Remove stale generated sensitivity files from an output folder.

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


def run_sensitivity_analysis(
    cfg: Config,
    out_dir: str | Path,
    samples: int = 8,
    seconds: float = 1500.0,
    seeds: int = 3,
    fast: bool = False,
    robust: bool = False,
    seed: int | None = None,
    method: str = "morris",
    **_: Any,
) -> list[Path]:
    """Run Morris screening and write summary CSVs and ranking plots.

    Inputs:
        cfg is the loaded configuration, out_dir receives files, samples is the
        Morris base-point count, seconds is the run length, seeds is the common
        random seed count, fast/robust apply presets, seed is the first seed,
        and method must be "morris".

    Output:
        Paths to written sensitivity files.
    """
    if method != "morris":
        raise ValueError("Only Morris screening is supported")
    samples, seconds, seeds = mode_settings(samples, seconds, seeds, fast, robust)
    presentation, supporting, data = output_folders(out_dir)
    presentation_split = presentation / "split_panels"
    supporting_split = supporting / "split_panels"
    presentation_split.mkdir(parents=True, exist_ok=True)
    supporting_split.mkdir(parents=True, exist_ok=True)
    remove_known_outputs(
        presentation,
        [
            "fig_04_morris_energy_sensitivity.png",
            "fig_05_morris_strategy_sensitivity.png",
            "fig_06_morris_interaction_diagnostics.png",
            "slide2_morris_sensitivity_summary.png",
            "slide2_morris_interaction_diagnostics.png",
        ],
    )
    remove_known_outputs(
        supporting,
        [
            "fig_morris_energy_sensitivity.png",
            "fig_morris_strategy_sensitivity.png",
        ],
    )
    remove_known_outputs(
        data,
        [
            "sensitivity_raw_runs.csv",
            "sensitivity_morris_summary.csv",
            "morris_raw_runs.csv",
            "morris_sensitivity_summary.csv",
        ],
    )
    seed0 = int(seed if seed is not None else cfg.get("run", "random_seed", default=7))
    raw, summary = run_morris_screening(cfg, samples, seconds, seeds, seed0)

    raw_path = data / "morris_raw_runs.csv"
    summary_path = data / "morris_sensitivity_summary.csv"
    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    summary_plot = plot_morris_sensitivity_summary(
        summary,
        presentation / "slide2_morris_sensitivity_summary.png",
    )
    split_energy_plot = plot_morris_ranking(
        summary,
        "final_net_energy_per_robot",
        presentation_split / "slide2_morris_energy_sensitivity.png",
        "Morris Sensitivity Analysis for Final Net Energy per Robot",
    )
    split_strategy_plot = plot_morris_ranking(
        summary,
        "mean_weighted_search_rest_ratio",
        presentation_split / "slide2_morris_strategy_sensitivity.png",
        "Morris Sensitivity Analysis for Learned Search/Rest Ratio",
    )
    energy_plot = plot_morris_ranking(
        summary,
        "final_net_energy_per_robot",
        supporting / "fig_morris_energy_sensitivity.png",
        "Morris Sensitivity Analysis for Final Net Energy per Robot",
    )
    strategy_plot = plot_morris_ranking(
        summary,
        "mean_weighted_search_rest_ratio",
        supporting / "fig_morris_strategy_sensitivity.png",
        "Morris Sensitivity Analysis for Learned Search/Rest Ratio",
    )
    diagnostics_plot = plot_morris_interaction_diagnostics(
        summary,
        presentation / "slide2_morris_interaction_diagnostics.png",
    )
    split_energy_diagnostic = plot_morris_interaction_panel(
        summary,
        "final_net_energy_per_robot",
        presentation_split / "slide2_morris_interaction_energy.png",
        "Morris Interaction Diagnostics for Final Net Energy per Robot",
    )
    split_strategy_diagnostic = plot_morris_interaction_panel(
        summary,
        "mean_weighted_search_rest_ratio",
        presentation_split / "slide2_morris_interaction_strategy.png",
        "Morris Interaction Diagnostics for Learned Search/Rest Ratio",
    )
    return [
        raw_path,
        summary_path,
        summary_plot,
        diagnostics_plot,
        split_energy_plot,
        split_strategy_plot,
        split_energy_diagnostic,
        split_strategy_diagnostic,
        energy_plot,
        strategy_plot,
    ]


def mode_settings(samples: int, seconds: float, seeds: int, fast: bool, robust: bool) -> tuple[int, float, int]:
    """Return effective Morris settings from mode flags."""
    if robust:
        return 12, 3000.0, 5
    if fast:
        return 4, 1000.0, 2
    return max(1, int(samples)), float(seconds), max(1, int(seeds))


def run_morris_screening(
    cfg: Config,
    samples: int,
    seconds: float,
    seeds: int,
    seed0: int,
    delta: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run fast Morris screening with common random seeds.

    Inputs:
        cfg is the base config, samples is the number of base points, seconds is
        simulated duration, seeds is repeated seed count, seed0 is the first
        seed, and delta is the normalized perturbation.

    Output:
        Raw run DataFrame and Morris summary DataFrame.
    """
    rng = np.random.default_rng(seed0)
    parameters = list(PARAMETER_BOUNDS.keys())
    raw_rows: list[dict[str, float | int | str]] = []
    effects: list[dict[str, float | str]] = []
    for sample_id in range(samples):
        base = rng.random(len(parameters))
        base_metrics = evaluate_vector(cfg, base, sample_id, seed0, seeds, seconds, "base", raw_rows)
        for index, parameter in enumerate(parameters):
            perturbed = base.copy()
            direction = delta if base[index] + delta <= 1.0 else -delta
            perturbed[index] = base[index] + direction
            perturbed_metrics = evaluate_vector(cfg, perturbed, sample_id, seed0, seeds, seconds, parameter, raw_rows)
            for output in MORRIS_OUTPUTS:
                effects.append(
                    {
                        "output": output,
                        "parameter": parameter,
                        "effect": (perturbed_metrics[output] - base_metrics[output]) / direction,
                    }
                )
    raw = pd.DataFrame(raw_rows)
    summary = summarize_effects(pd.DataFrame(effects), samples, seeds, seconds)
    return raw, summary


def evaluate_vector(
    cfg: Config,
    normalized: np.ndarray,
    sample_id: int,
    seed0: int,
    seeds: int,
    seconds: float,
    perturbation: str,
    raw_rows: list[dict[str, float | int | str]],
) -> dict[str, float]:
    """Evaluate one normalized parameter vector across common seeds."""
    params = denormalize_parameters(normalized)
    metrics: list[dict[str, float]] = []
    for offset in range(seeds):
        seed = seed0 + offset
        observed = run_model_metrics(cfg, params, seed, seconds)
        metrics.append(observed)
        row: dict[str, float | int | str] = {"method": "morris", "sample_id": sample_id, "seed": seed, "perturbation": perturbation}
        row.update(params)
        row.update(observed)
        raw_rows.append(row)
    return {output: float(np.mean([metric[output] for metric in metrics])) for output in MORRIS_OUTPUTS}


def denormalize_parameters(normalized: np.ndarray) -> dict[str, float | int]:
    """Convert normalized values in [0, 1] into model parameter values."""
    params: dict[str, float | int] = {}
    for value, (parameter, bounds) in zip(normalized, PARAMETER_BOUNDS.items()):
        lo, hi = bounds
        scaled = float(lo + value * (hi - lo))
        params[parameter] = nearest_robot_count(scaled) if parameter == "n_robots" else scaled
    return params


def nearest_robot_count(value: float) -> int:
    """Round a continuous robot count to the nearest tested level."""
    return min(ROBOT_LEVELS, key=lambda level: abs(level - value))


def copy_config(cfg: Config, n_robots: int, p_new_multiplier: float) -> Config:
    """Return a copied config with robot count and food-spawn multiplier changed."""
    raw = deepcopy(cfg.raw)
    raw["paper"]["n_robots"] = int(n_robots)
    raw["food"]["growth_rate_s"] = float(cfg.get("food", "growth_rate_s")) * float(p_new_multiplier)
    return Config(raw=raw, path=cfg.path)


def run_model_metrics(cfg: Config, params: dict[str, float | int], seed: int, seconds: float) -> dict[str, float]:
    """Run the existing micro model and compute observational metrics.

    Inputs:
        cfg is the base configuration, params are tested values, seed controls
        randomness, and seconds is simulated duration.

    Output:
        Required sensitivity metrics. They are read-only and do not affect model
        behaviour.
    """
    n_robots = int(params["n_robots"])
    run_cfg = copy_config(cfg, n_robots=n_robots, p_new_multiplier=float(params["p_new_multiplier"]))
    kwargs = {
        key: float(params[key])
        for key in [
            "gamma_r_scale",
            "alpha",
            "lambda_loss",
            "congestion_tolerance",
            "sector_decay",
            "sector_reward",
            "sector_pull",
        ]
    }
    model = MicroModel(run_cfg, seed=seed, **kwargs)
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
    return {
        "final_net_energy": float(model.energy),
        "final_net_energy_per_robot": float(model.energy / max(1, n_robots)),
        "throughput": float(throughput),
        "throughput_per_robot": float(throughput / max(1, n_robots)),
        "mean_collision_probability": float(np.mean(gamma_r_values)) if gamma_r_values else 0.0,
        "mean_weighted_search_rest_ratio": float(np.mean(ratio_values)) if ratio_values else 0.0,
        "mean_active_fraction": float(np.mean(active_values)) if active_values else 0.0,
        "total_completed_deposit": float(model.total_completed_deposit),
        "total_collisions": float(model.total_collisions),
    }


def summarize_effects(effects: pd.DataFrame, samples: int, seeds: int, seconds: float) -> pd.DataFrame:
    """Compute Morris mu_star, sigma, and rank by output."""
    grouped = effects.groupby(["output", "parameter"], as_index=False).agg(
        mu_star=("effect", lambda values: float(np.mean(np.abs(values)))),
        sigma=("effect", lambda values: float(np.std(values, ddof=0))),
    )
    rows: list[dict[str, float | int | str]] = []
    for output, group in grouped.groupby("output", sort=False):
        ranked = group.sort_values("mu_star", ascending=False).reset_index(drop=True)
        for rank, row in enumerate(ranked.itertuples(index=False), start=1):
            rows.append(
                {
                    "output": output,
                    "parameter": row.parameter,
                    "mu_star": row.mu_star,
                    "sigma": row.sigma,
                    "rank": rank,
                    "samples": samples,
                    "seeds": seeds,
                    "seconds": float(seconds),
                }
            )
    return pd.DataFrame(rows)


def plot_morris_ranking(summary: pd.DataFrame, output: str, path: str | Path, title: str) -> Path:
    """Create a horizontal Morris ranking plot for one output."""
    path = Path(path)
    data = summary[summary["output"] == output].sort_values("mu_star", ascending=True)
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    ax.barh([parameter_label(value) for value in data["parameter"]], data["mu_star"], xerr=data["sigma"], color="tab:blue")
    ax.set_title(title, pad=12)
    ax.set_xlabel("Morris sensitivity index μ*")
    ax.set_ylabel("Model parameter")
    ax.grid(True, axis="x", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_morris_sensitivity_summary(summary: pd.DataFrame, path: str | Path) -> Path:
    """Create the two-panel Morris ranking figure for Slide 2.

    Inputs:
        summary is the Morris summary DataFrame and path is the PNG destination.

    Output:
        Path to the written combined sensitivity figure.
    """
    path = Path(path)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.0), constrained_layout=True)
    fig.suptitle("Morris Sensitivity Analysis for Energy and Strategy Outputs", fontsize=16)
    panels = [
        ("final_net_energy_per_robot", "A. Final net energy per robot"),
        ("mean_weighted_search_rest_ratio", "B. Learned search/rest ratio"),
    ]
    for axis, (output, title) in zip(axes, panels):
        data = summary[summary["output"] == output].sort_values("mu_star", ascending=True)
        axis.barh(
            [parameter_label(value) for value in data["parameter"]],
            data["mu_star"],
            xerr=data["sigma"],
            color="tab:blue",
            alpha=0.9,
        )
        axis.set_title(title)
        axis.set_xlabel("Morris sensitivity index μ*")
        axis.set_ylabel("Model parameter")
        axis.grid(True, axis="x", linestyle="--", alpha=0.35)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_morris_interaction_diagnostics(summary: pd.DataFrame, path: str | Path) -> Path:
    """Create the Morris μ*/σ diagnostic scatter plot.

    Inputs:
        summary is the Morris summary DataFrame and path is the PNG destination.

    Output:
        Path to the written diagnostic figure.
    """
    path = Path(path)
    outputs = [
        ("final_net_energy_per_robot", "A. Final net energy per robot"),
        ("mean_weighted_search_rest_ratio", "B. Learned search/rest ratio"),
    ]
    label_offsets = {
        "final_net_energy_per_robot": {
            "p_new_multiplier": (-142, -8),
            "n_robots": (6, 8),
            "gamma_r_scale": (6, -14),
            "lambda_loss": (6, 12),
            "alpha": (6, -8),
            "congestion_tolerance": (6, 8),
        },
        "mean_weighted_search_rest_ratio": {
            "p_new_multiplier": (8, -12),
            "n_robots": (8, 8),
            "gamma_r_scale": (8, 10),
            "lambda_loss": (8, 6),
            "alpha": (8, -14),
            "congestion_tolerance": (-92, 6),
        },
    }
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9))
    fig.suptitle("Morris Interaction Diagnostics for Energy and Strategy Outputs")
    for axis, (output, title) in zip(axes, outputs):
        data = summary[summary["output"] == output].copy()
        axis.scatter(data["mu_star"], data["sigma"], s=44, color="tab:blue", alpha=0.85)
        limit = float(max(data["mu_star"].max(), data["sigma"].max(), 1e-9))
        axis.plot([0.0, limit], [0.0, limit], "--", color="0.35", linewidth=1.0, label="σ = μ*")
        for row in data.itertuples(index=False):
            if row.parameter not in IMPORTANT_DIAGNOSTIC_PARAMETERS:
                continue
            axis.annotate(
                parameter_label(row.parameter),
                (row.mu_star, row.sigma),
                xytext=label_offsets[output].get(row.parameter, (4, 4)),
                textcoords="offset points",
                fontsize=7.5,
            )
        axis.set_title(title)
        axis.set_xlabel("Morris sensitivity index μ*")
        axis.set_ylabel("Standard deviation of elementary effects σ")
        axis.set_xlim(left=0.0, right=limit * 1.35)
        axis.set_ylim(bottom=0.0, top=limit * 1.20)
        axis.grid(True, linestyle="--", alpha=0.35)
        axis.legend(frameon=True, edgecolor="black")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_morris_interaction_panel(summary: pd.DataFrame, output: str, path: str | Path, title: str) -> Path:
    """Create one single-panel Morris μ*/σ diagnostic scatter plot.

    Inputs:
        summary is the Morris summary DataFrame, output selects the model output,
        path is the PNG destination, and title labels the figure.

    Output:
        Path to the written single-panel diagnostic figure.
    """
    path = Path(path)
    data = summary[summary["output"] == output].copy()
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.set_title(title)
    ax.scatter(data["mu_star"], data["sigma"], s=50, color="tab:blue", alpha=0.85)
    limit = float(max(data["mu_star"].max(), data["sigma"].max(), 1e-9))
    ax.plot([0.0, limit], [0.0, limit], "--", color="0.35", linewidth=1.0, label="σ = μ*")
    for row in data.itertuples(index=False):
        if row.parameter not in IMPORTANT_DIAGNOSTIC_PARAMETERS:
            continue
        ax.annotate(
            parameter_label(row.parameter),
            (row.mu_star, row.sigma),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("Morris sensitivity index μ*")
    ax.set_ylabel("Standard deviation of elementary effects σ")
    ax.set_xlim(left=0.0, right=limit * 1.35)
    ax.set_ylim(bottom=0.0, top=limit * 1.20)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=True, edgecolor="black")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def parameter_label(parameter: str) -> str:
    """Return readable parameter labels for sensitivity plots."""
    labels = {
        "n_robots": "Number of robots",
        "gamma_r_scale": "Collision probability scale",
        "alpha": "Risk curvature α",
        "lambda_loss": "Loss-aversion coefficient λ",
        "congestion_tolerance": "Congestion tolerance",
        "p_new_multiplier": "Food spawn-rate multiplier",
        "sector_decay": "Sector-score decay",
        "sector_reward": "Sector-score reward",
        "sector_pull": "Sector steering strength",
    }
    return labels.get(parameter, parameter)
