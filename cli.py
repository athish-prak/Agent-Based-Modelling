"""Command-line runner for final ABM outputs.

Inputs:
    A command name plus optional configuration, output, simulation, sensitivity,
    and animation settings.

Outputs:
    Final report plots, Morris sensitivity files, macro CSVs, or animation GIFs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from animation import make_animation
from config import Config
from plot import make_all_plots, write_macro_csv
from sensitivity import run_sensitivity_analysis


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description="Swarm-foraging ABM outputs")
    parser.add_argument("command", choices=["plots", "sensitivity", "sobol", "animation", "all", "csv"], help="output workflow")
    parser.add_argument("--config", default="default_config.yaml", help="YAML parameter file")
    parser.add_argument("--out", default="outputs", help="output directory, or GIF path for animation")
    parser.add_argument("--seconds", type=float, default=None, help="simulated seconds")
    parser.add_argument("--seeds", type=int, default=None, help="random seeds for averaging")
    parser.add_argument("--samples", type=int, default=None, help="sensitivity base samples")
    parser.add_argument("--workers", default="auto", help="Sobol worker count, or auto")
    parser.add_argument("--fast", action="store_true", help="use fast sensitivity settings")
    parser.add_argument("--extensive", action="store_true", help="use extensive Sobol sensitivity settings")
    parser.add_argument("--robust", action="store_true", help="use robust sensitivity settings")
    parser.add_argument("--fps", type=int, default=None, help="animation frames per second")
    parser.add_argument("--playback-seconds", type=float, default=None, help="GIF playback duration")
    parser.add_argument("--rest", type=float, default=None, help="rest time for macro CSV or animation")
    return parser


def main() -> None:
    """Read arguments, run the requested workflow, and print output paths."""
    args = build_parser().parse_args()
    cfg = Config.load(args.config)
    plot_seeds = int(args.seeds if args.seeds is not None else 5)
    sensitivity_seeds = int(args.seeds if args.seeds is not None else 3)
    if args.command == "plots":
        paths = make_all_plots(
            cfg,
            args.out,
            seconds=float(args.seconds if args.seconds is not None else 1500.0),
            seeds=plot_seeds,
        )
    elif args.command == "sensitivity":
        paths = run_sensitivity_analysis(
            cfg,
            args.out,
            samples=int(args.samples if args.samples is not None else 8),
            seconds=float(args.seconds if args.seconds is not None else 1500.0),
            seeds=sensitivity_seeds,
            fast=args.fast,
            robust=args.robust,
        )
    elif args.command == "sobol":
        from sobol_sensitivity import run_sobol_analysis

        paths = run_sobol_analysis(
            cfg,
            args.out,
            samples=args.samples,
            seconds=args.seconds,
            seeds=args.seeds,
            workers=args.workers,
            fast=args.fast,
            extensive=args.extensive,
            robust=args.robust,
        )
    elif args.command == "animation":
        out = Path(args.out)
        gif_path = out if out.suffix.lower() == ".gif" else out / "swarm_animation.gif"
        paths = [
            make_animation(
                cfg,
                gif_path,
                seconds=args.seconds,
                rest_time_s=args.rest,
                playback_seconds=args.playback_seconds,
                fps=args.fps,
            )
        ]
    elif args.command == "all":
        out = Path(args.out)
        paths = make_all_plots(
            cfg,
            out,
            seconds=float(args.seconds if args.seconds is not None else 1500.0),
            seeds=plot_seeds,
        )
        paths.extend(
            run_sensitivity_analysis(
                cfg,
                out,
                samples=int(args.samples if args.samples is not None else 8),
                seconds=float(args.seconds if args.seconds is not None else 1500.0),
                seeds=sensitivity_seeds,
                fast=args.fast,
                robust=args.robust,
            )
        )
        paths.append(
            make_animation(
                cfg,
                out / "swarm_animation.gif",
                seconds=args.seconds,
                rest_time_s=args.rest,
                playback_seconds=args.playback_seconds,
                fps=args.fps,
            )
        )
    elif args.command == "csv":
        paths = [write_macro_csv(cfg, args.out, rest_time_s=args.rest)]
    else:
        raise SystemExit(f"unknown command: {args.command}")

    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
