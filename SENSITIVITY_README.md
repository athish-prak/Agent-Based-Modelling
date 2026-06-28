# Sensitivity Analysis

Sensitivity analysis tests which parameters most affect swarm performance and
learned strategy behaviour.

## Methods

Morris is used for fast global screening. It ranks parameters with:

- `mu_star`: mean absolute elementary effect, used as the importance score.
- `sigma`: variation in elementary effects, suggesting nonlinear or interaction
  effects when large.

Sobol is used for variance-based confirmation on a smaller parameter set. It
reports:

- `S1`: first-order effect of a parameter alone.
- `ST`: total-order effect, including interactions.

If `ST` is much larger than `S1`, the parameter matters mainly through
interactions or nonlinear behaviour.

## Parameters

Morris screens the broader model parameter set, including sector-learning terms.

Sobol focuses on six presentation-level parameters:

- Food spawn-rate multiplier
- Number of robots
- Collision probability scale
- Loss-aversion coefficient `lambda`
- Risk curvature `alpha`
- Congestion tolerance

## Outputs

Sobol measures:

- final net energy per robot
- throughput per robot
- mean collision probability
- mean weighted search/rest ratio
- final strategy entropy
- mean strategy entropy over the final window

Each parameter sample is run with repeated random seeds. Raw per-seed outputs
are saved first, then averaged before Sobol indices are computed. Sobol runs are
parallelized because each simulation is independent.

## Commands

```bash
python cli.py sensitivity --out outputs
python cli.py sensitivity --fast --out outputs
python cli.py sensitivity --robust --out outputs

python cli.py sobol --out outputs/sobol
python cli.py sobol --fast --out outputs/sobol_fast
python cli.py sobol --extensive --out outputs/sobol_extensive
python cli.py sobol --robust --out outputs/sobol_robust
```

Sobol presets:

| Mode | N samples | Seeds | Seconds | Approx. simulations |
|---|---:|---:|---:|---:|
| fast/default | 64 | 3 | 1500 | 2,688 |
| extensive | 128 | 5 | 1500 | 8,960 |
| robust | 512 | 5 | 1500 | 35,840 |

`N` is the Saltelli base sample size. With six parameters and second-order terms
enabled, the number of Sobol evaluations is `N * (2D + 2)`, repeated across
seeds.

## Files

Morris writes:

- `data/morris_raw_runs.csv`
- `data/morris_sensitivity_summary.csv`
- `presentation_plots/slide2_morris_sensitivity_summary.png`
- `presentation_plots/slide2_morris_interaction_diagnostics.png`

Sobol writes:

- `data/sobol_parameter_samples.csv`
- `data/sobol_raw_runs.csv`
- `data/sobol_seed_averaged_outputs.csv`
- `data/sobol_indices_summary.csv`
- `data/sobol_warnings.txt`
- `presentation_plots/sobol_energy_indices.png`
- `presentation_plots/sobol_strategy_indices.png`

## Interpretation

Small negative Sobol values or values slightly above 1 can occur from
finite-sample noise. Large violations mean the run is too small or too noisy for
strong interpretation.

Use Morris as the main screening result. Use Sobol to confirm whether the main
presentation parameters explain output variance directly or mainly through
interactions.
