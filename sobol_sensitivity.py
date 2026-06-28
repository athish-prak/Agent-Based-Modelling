"""Run parallel SALib Sobol sensitivity analysis for the ABM.

Inputs:
    A Config object, output folder, Saltelli/Sobol sample count, simulated
    seconds, repeated seed count, and worker count.

Outputs:
    Sobol sample CSVs, raw per-seed CSVs, seed-averaged CSVs, Sobol index
    summaries, warning notes, and presentation plots.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import csv
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
from SALib.analyze import sobol
from SALib.sample import saltelli

from agents import MicroModel
from config import Config


SOBOL_PARAMETER_BOUNDS = {
    "p_new_multiplier": (0.25, 4.0),
    "n_robots": (8, 128),
    "gamma_r_scale": (0.5, 1.5),
    "lambda_loss": (1.0, 5.0),
    "alpha": (0.5, 1.0),
    "congestion_tolerance": (0.01, 0.10),
}
VALID_ROBOT_COUNTS = [8, 16, 32, 64, 128]
SOBOL_OUTPUTS = [
    "final_net_energy_per_robot",
    "throughput_per_robot",
    "mean_collision_probability",
    "mean_weighted_search_rest_ratio",
    "final_strategy_entropy",
    "mean_strategy_entropy_final_window",
]
SALTELLI_CALC_SECOND_ORDER = True


def run_sobol_analysis(
    cfg: Config,
    out_dir: str | Path,
    samples: int | None = None,
    seconds: float | None = None,
    seeds: int | None = None,
    workers: str | int | None = "auto",
    fast: bool = False,
    robust: bool = False,
    extensive: bool = False,
) -> list[Path]:
    """Run parallel Sobol analysis and write CSV and plot outputs.

    Inputs:
        cfg is the loaded configuration, out_dir receives files, samples is the
        Saltelli base sample size, seconds is simulated duration, seeds is the
        repeated seed count, workers controls process count, and fast,
        extensive, or robust select presets.

    Output:
        Paths to written Sobol files.
    """
    samples, seconds, seeds = sobol_mode_settings(samples, seconds, seeds, fast, robust, extensive)
    root = Path(out_dir)
    data_dir = root / "data"
    plot_dir = root / "presentation_plots"
    data_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    problem = sobol_problem()
    param_values = saltelli.sample(problem, samples, calc_second_order=SALTELLI_CALC_SECOND_ORDER)
    parameter_rows = sample_rows(param_values)
    sample_path = data_dir / "sobol_parameter_samples.csv"
    raw_path = data_dir / "sobol_raw_runs.csv"
    settings_path = data_dir / "sobol_run_settings.csv"
    reset_incompatible_resume(data_dir, parameter_rows, samples, seeds, seconds)
    pd.DataFrame(parameter_rows).to_csv(sample_path, index=False)
    write_run_settings(settings_path, samples, seeds, seconds)

    done = completed_jobs(raw_path)
    seed0 = int(cfg.get("run", "random_seed", default=7))
    seed_values = [seed0 + offset for offset in range(seeds)]
    jobs = [
        {
            "sample_id": row["sample_id"],
            "seed": seed,
            "params": {name: row[name] for name in SOBOL_PARAMETER_BOUNDS},
            "config_raw": cfg.raw,
            "seconds": seconds,
        }
        for row in parameter_rows
        for seed in seed_values
        if (int(row["sample_id"]), int(seed)) not in done
    ]

    total_jobs = len(parameter_rows) * len(seed_values)
    completed_count = len(done)
    if jobs:
        run_jobs_parallel(jobs, raw_path, total_jobs, completed_count, workers)
    else:
        print(f"Completed {total_jobs} / {total_jobs} runs")

    raw = pd.read_csv(raw_path)
    averaged = seed_averaged_outputs(raw, parameter_rows, seed_values)
    averaged_path = data_dir / "sobol_seed_averaged_outputs.csv"
    averaged.to_csv(averaged_path, index=False)

    summary = analyze_outputs(problem, averaged, samples, seeds, seconds)
    summary_path = data_dir / "sobol_indices_summary.csv"
    summary.to_csv(summary_path, index=False)

    warnings_path = data_dir / "sobol_warnings.txt"
    write_warnings(summary, warnings_path)

    energy_plot = plot_sobol_indices(
        summary,
        "final_net_energy_per_robot",
        plot_dir / "sobol_energy_indices.png",
        "Sobol Sensitivity Indices for Final Net Energy per Robot",
    )
    strategy_plot = plot_sobol_strategy_indices(
        summary,
        plot_dir / "sobol_strategy_indices.png",
    )
    return [sample_path, raw_path, averaged_path, summary_path, warnings_path, energy_plot, strategy_plot]


def reset_incompatible_resume(
    data_dir: Path,
    parameter_rows: list[dict[str, float | int]],
    samples: int,
    seeds: int,
    seconds: float,
) -> None:
    """Remove stale Sobol resume files when sample design or settings changed.

    Inputs:
        data_dir contains previous Sobol CSVs, parameter_rows are the new sample
        design, and samples/seeds/seconds are the requested run settings.

    Output:
        Incompatible raw and derived files are removed.
    """
    sample_path = data_dir / "sobol_parameter_samples.csv"
    settings_path = data_dir / "sobol_run_settings.csv"
    raw_path = data_dir / "sobol_raw_runs.csv"
    if not raw_path.exists():
        return
    compatible = sample_path.exists() and settings_path.exists()
    if compatible:
        compatible = samples_match(pd.read_csv(sample_path), pd.DataFrame(parameter_rows))
    if compatible:
        settings = pd.read_csv(settings_path).iloc[0]
        compatible = (
            int(settings["samples"]) == int(samples)
            and int(settings["seeds"]) == int(seeds)
            and float(settings["seconds"]) == float(seconds)
        )
    if compatible:
        return
    for filename in [
        "sobol_raw_runs.csv",
        "sobol_seed_averaged_outputs.csv",
        "sobol_indices_summary.csv",
        "sobol_warnings.txt",
    ]:
        path = data_dir / filename
        if path.exists():
            path.unlink()


def samples_match(existing: pd.DataFrame, new: pd.DataFrame) -> bool:
    """Return whether two Sobol sample tables contain the same sample design."""
    columns = ["sample_id", *SOBOL_PARAMETER_BOUNDS.keys()]
    if list(existing.columns) != columns or list(new.columns) != columns:
        return False
    if len(existing) != len(new):
        return False
    for column in columns:
        left = existing[column].to_numpy()
        right = new[column].to_numpy()
        if column in {"sample_id", "n_robots"}:
            if not np.array_equal(left.astype(int), right.astype(int)):
                return False
        elif not np.allclose(left.astype(float), right.astype(float), rtol=1e-12, atol=1e-12):
            return False
    return True


def write_run_settings(path: Path, samples: int, seeds: int, seconds: float) -> None:
    """Write the Sobol run settings used for resume compatibility checks."""
    pd.DataFrame([{"samples": int(samples), "seeds": int(seeds), "seconds": float(seconds)}]).to_csv(path, index=False)


def sobol_mode_settings(
    samples: int | None,
    seconds: float | None,
    seeds: int | None,
    fast: bool,
    robust: bool,
    extensive: bool = False,
) -> tuple[int, float, int]:
    """Return effective Sobol settings from CLI values and mode flags."""
    preset_samples = 64
    preset_seconds = 1500.0
    preset_seeds = 3
    if robust:
        preset_samples, preset_seconds, preset_seeds = 512, 1500.0, 5
    elif extensive:
        preset_samples, preset_seconds, preset_seeds = 128, 1500.0, 5
    elif fast:
        preset_samples, preset_seconds, preset_seeds = 64, 1500.0, 3
    return (
        int(samples if samples is not None else preset_samples),
        float(seconds if seconds is not None else preset_seconds),
        int(seeds if seeds is not None else preset_seeds),
    )


def sobol_problem() -> dict[str, Any]:
    """Build the SALib problem definition for the six Sobol parameters."""
    return {
        "num_vars": len(SOBOL_PARAMETER_BOUNDS),
        "names": list(SOBOL_PARAMETER_BOUNDS.keys()),
        "bounds": list(SOBOL_PARAMETER_BOUNDS.values()),
    }


def sample_rows(param_values: np.ndarray) -> list[dict[str, float | int]]:
    """Convert SALib sample rows into CSV-ready parameter dictionaries."""
    names = list(SOBOL_PARAMETER_BOUNDS.keys())
    rows: list[dict[str, float | int]] = []
    for sample_id, values in enumerate(param_values):
        row: dict[str, float | int] = {"sample_id": sample_id}
        for name, value in zip(names, values):
            row[name] = nearest_robot_count(float(value)) if name == "n_robots" else float(value)
        rows.append(row)
    return rows


def nearest_robot_count(value: float) -> int:
    """Round a continuous Sobol robot-count sample to a valid swarm size."""
    return min(VALID_ROBOT_COUNTS, key=lambda level: abs(level - value))


def completed_jobs(raw_path: Path) -> set[tuple[int, int]]:
    """Read completed sample/seed pairs from an existing raw CSV."""
    if not raw_path.exists() or raw_path.stat().st_size == 0:
        return set()
    raw = pd.read_csv(raw_path, usecols=["sample_id", "seed"])
    return {(int(row.sample_id), int(row.seed)) for row in raw.itertuples(index=False)}


def run_jobs_parallel(
    jobs: list[dict[str, Any]],
    raw_path: Path,
    total_jobs: int,
    completed_count: int,
    workers: str | int | None,
) -> None:
    """Run Sobol jobs in a process pool and append rows as they finish.

    Inputs:
        jobs are independent simulation jobs, raw_path receives appended rows,
        total_jobs is the expected full run count, completed_count is the resume
        count, and workers controls process count.

    Output:
        Raw CSV rows are appended incrementally.
    """
    worker_count = resolve_workers(workers)
    fieldnames = ["sample_id", "seed", *SOBOL_PARAMETER_BOUNDS.keys(), *SOBOL_OUTPUTS]
    write_header = not raw_path.exists() or raw_path.stat().st_size == 0
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with open(raw_path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        try:
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                futures = [executor.submit(run_one_sobol_job, job) for job in jobs]
                for index, future in enumerate(as_completed(futures), start=1):
                    row = future.result()
                    writer.writerow({key: row[key] for key in fieldnames})
                    if index % 25 == 0:
                        handle.flush()
                    completed = completed_count + index
                    if completed % 250 == 0 or completed == total_jobs:
                        print(f"Completed {completed} / {total_jobs} runs")
        except PermissionError:
            print("Process pool unavailable; running Sobol jobs serially.")
            for index, job in enumerate(jobs, start=1):
                row = run_one_sobol_job(job)
                writer.writerow({key: row[key] for key in fieldnames})
                if index % 25 == 0:
                    handle.flush()
                completed = completed_count + index
                if completed % 250 == 0 or completed == total_jobs:
                    print(f"Completed {completed} / {total_jobs} runs")


def resolve_workers(workers: str | int | None) -> int:
    """Resolve CLI worker settings into a positive process count."""
    if workers is None or workers == "auto":
        return max(1, (os.cpu_count() or 2) - 1)
    return max(1, int(workers))


def run_one_sobol_job(job: dict[str, Any]) -> dict[str, float | int]:
    """Run one Sobol simulation job and return parameter values plus metrics.

    Input:
        job contains sample_id, seed, parameter values, raw config, and seconds.

    Output:
        A flat dictionary suitable for the raw Sobol CSV.
    """
    params = dict(job["params"])
    cfg = config_for_params(Config(raw=deepcopy(job["config_raw"])), params)
    model = MicroModel(
        cfg,
        seed=int(job["seed"]),
        gamma_r_scale=float(params["gamma_r_scale"]),
        alpha=float(params["alpha"]),
        lambda_loss=float(params["lambda_loss"]),
        congestion_tolerance=float(params["congestion_tolerance"]),
    )
    seconds = float(job["seconds"])
    steps = model.world.steps(seconds)
    final_window_start = int(0.75 * steps)
    gamma_r_values: list[float] = []
    ratio_values: list[float] = []
    entropy_values: list[float] = []
    final_entropy = 0.0

    for step in range(steps):
        stats = model.step()
        entropy = strategy_entropy(model)
        gamma_r_values.append(float(stats["gamma_r"]))
        final_entropy = entropy
        if step >= final_window_start:
            ratio_values.append(float(stats["mean_weighted_search_rest_ratio"]))
            entropy_values.append(entropy)

    n_robots = max(1, int(params["n_robots"]))
    throughput = model.total_completed_deposit / max(seconds, model.world.dt)
    row: dict[str, float | int] = {
        "sample_id": int(job["sample_id"]),
        "seed": int(job["seed"]),
        "p_new_multiplier": float(params["p_new_multiplier"]),
        "n_robots": n_robots,
        "gamma_r_scale": float(params["gamma_r_scale"]),
        "lambda_loss": float(params["lambda_loss"]),
        "alpha": float(params["alpha"]),
        "congestion_tolerance": float(params["congestion_tolerance"]),
        "final_net_energy_per_robot": float(model.energy / n_robots),
        "throughput_per_robot": float(throughput / n_robots),
        "mean_collision_probability": float(np.mean(gamma_r_values)) if gamma_r_values else 0.0,
        "mean_weighted_search_rest_ratio": float(np.mean(ratio_values)) if ratio_values else 0.0,
        "final_strategy_entropy": float(final_entropy),
        "mean_strategy_entropy_final_window": float(np.mean(entropy_values)) if entropy_values else 0.0,
    }
    return row


def config_for_params(cfg: Config, params: dict[str, float | int]) -> Config:
    """Return a copied config with Sobol robot count and food multiplier applied."""
    raw = deepcopy(cfg.raw)
    raw["paper"]["n_robots"] = int(params["n_robots"])
    raw["food"]["growth_rate_s"] = float(cfg.get("food", "growth_rate_s")) * float(params["p_new_multiplier"])
    return Config(raw=raw, path=cfg.path)


def strategy_entropy(model: MicroModel) -> float:
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


def seed_averaged_outputs(raw: pd.DataFrame, parameter_rows: list[dict[str, float | int]], seeds: list[int]) -> pd.DataFrame:
    """Average per-seed Sobol outputs by sample_id in SALib sample order.

    Inputs:
        raw contains per-seed rows, parameter_rows define the SALib sample order,
        and seeds are the required repeated seeds.

    Output:
        One row per sample_id, sorted to match the original sample matrix.
    """
    expected_seeds = set(seeds)
    rows: list[dict[str, float | int]] = []
    for sample in parameter_rows:
        sample_id = int(sample["sample_id"])
        group = raw[raw["sample_id"] == sample_id]
        have = {int(value) for value in group["seed"].unique()}
        if have != expected_seeds:
            missing = sorted(expected_seeds - have)
            raise RuntimeError(f"Sobol sample_id {sample_id} is missing seeds: {missing}")
        row = dict(sample)
        for output in SOBOL_OUTPUTS:
            row[output] = float(group[output].mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)


def analyze_outputs(problem: dict[str, Any], averaged: pd.DataFrame, samples: int, seeds: int, seconds: float) -> pd.DataFrame:
    """Compute Sobol S1 and ST indices for every requested output."""
    rows: list[dict[str, float | int | str]] = []
    for output in SOBOL_OUTPUTS:
        values = averaged.sort_values("sample_id")[output].to_numpy(dtype=float)
        indices = sobol.analyze(problem, values, calc_second_order=SALTELLI_CALC_SECOND_ORDER, print_to_console=False)
        for index, parameter in enumerate(problem["names"]):
            rows.append(
                {
                    "output": output,
                    "parameter": parameter,
                    "S1": float(indices["S1"][index]),
                    "S1_conf": float(indices["S1_conf"][index]),
                    "ST": float(indices["ST"][index]),
                    "ST_conf": float(indices["ST_conf"][index]),
                    "samples": int(samples),
                    "seeds": int(seeds),
                    "seconds": float(seconds),
                }
            )
    return pd.DataFrame(rows)


def write_warnings(summary: pd.DataFrame, path: Path) -> None:
    """Write short warnings for large out-of-range Sobol estimates."""
    warnings: list[str] = []
    for row in summary.itertuples(index=False):
        for column in ["S1", "ST"]:
            value = float(getattr(row, column))
            if value > 1.2 or value < -0.2:
                warnings.append(f"{row.output} / {row.parameter}: {column}={value:.3f}")
    if warnings:
        text = "Large out-of-range Sobol estimates detected:\n" + "\n".join(warnings) + "\n"
        print(text.strip())
    else:
        text = "No large out-of-range Sobol estimates detected.\n"
    path.write_text(text, encoding="utf-8")


def plot_sobol_indices(summary: pd.DataFrame, output: str, path: str | Path, title: str) -> Path:
    """Create one grouped S1/ST Sobol bar chart."""
    path = Path(path)
    data = summary[summary["output"] == output].copy()
    data["label"] = data["parameter"].map(parameter_label)
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    positions = np.arange(len(data))
    width = 0.36
    ax.bar(positions - width / 2, data["S1"], width, yerr=data["S1_conf"], label="S1", color="tab:blue", capsize=3)
    ax.bar(positions + width / 2, data["ST"], width, yerr=data["ST_conf"], label="ST", color="tab:orange", capsize=3)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axhline(1.0, color="0.4", linewidth=0.9, linestyle=":")
    ax.set_title(title)
    ax.set_ylabel("Sobol index")
    ax.set_xticks(positions)
    ax.set_xticklabels(data["label"], rotation=30, ha="right")
    ax.legend(frameon=True, edgecolor="black")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    set_sobol_ylim(ax, data)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_sobol_strategy_indices(summary: pd.DataFrame, path: str | Path) -> Path:
    """Create the two-panel Sobol strategy-output figure."""
    path = Path(path)
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.2), constrained_layout=True)
    fig.suptitle("Sobol Sensitivity Indices for Learned Strategy Outcomes", fontsize=15)
    panels = [
        ("mean_weighted_search_rest_ratio", "A. Mean search/rest ratio over final window"),
        ("mean_strategy_entropy_final_window", "B. Mean strategy entropy over final window"),
    ]
    for axis, (output, title) in zip(axes, panels):
        data = summary[summary["output"] == output].copy()
        labels = [parameter_label(value) for value in data["parameter"]]
        positions = np.arange(len(data))
        width = 0.36
        axis.bar(positions - width / 2, data["S1"], width, yerr=data["S1_conf"], label="S1", color="tab:blue", capsize=3)
        axis.bar(positions + width / 2, data["ST"], width, yerr=data["ST_conf"], label="ST", color="tab:orange", capsize=3)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.axhline(1.0, color="0.4", linewidth=0.9, linestyle=":")
        axis.set_title(title)
        axis.set_ylabel("Sobol index")
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=30, ha="right")
        axis.legend(frameon=True, edgecolor="black")
        axis.grid(True, axis="y", linestyle="--", alpha=0.35)
        set_sobol_ylim(axis, data)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def set_sobol_ylim(axis: plt.Axes, data: pd.DataFrame) -> None:
    """Set a readable y-axis range without clipping finite-sample estimates."""
    values = pd.concat(
        [
            data["S1"] - data["S1_conf"],
            data["S1"] + data["S1_conf"],
            data["ST"] - data["ST_conf"],
            data["ST"] + data["ST_conf"],
        ]
    )
    finite = values[np.isfinite(values)]
    if finite.empty:
        axis.set_ylim(-0.1, 1.1)
        return
    lower = min(-0.05, float(finite.min()) - 0.05)
    upper = max(1.05, float(finite.max()) + 0.05)
    axis.set_ylim(lower, upper)


def parameter_label(parameter: str) -> str:
    """Return readable parameter labels for Sobol plots."""
    labels = {
        "p_new_multiplier": "Food spawn-rate multiplier",
        "n_robots": "Number of robots",
        "gamma_r_scale": "Collision probability scale",
        "lambda_loss": "Loss-aversion coefficient λ",
        "alpha": "Risk curvature α",
        "congestion_tolerance": "Congestion tolerance",
    }
    return labels.get(parameter, parameter)
