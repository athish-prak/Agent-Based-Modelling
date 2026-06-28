# Swarm Foraging ABM

Agent-based model of a robot swarm that searches for food, avoids collisions,
returns to the nest, rests, and adapts its search/rest behaviour from experience.

The model is decentralized: each robot follows the same local rules, while
swarm-level outcomes emerge from food availability, congestion, collision risk,
energy costs, and learned strategy propensities.

## Model

Each robot moves through six PFSM states:

- `searching`
- `grabbing`
- `deposit`
- `homing`
- `resting`
- `avoidance`

At each simulation step the model:

1. Counts robots in each state.
2. Computes food-finding, collision, and food-loss probabilities.
3. Updates every robot's state, energy, position, trip memory, and learning.
4. Updates food availability and aggregate swarm energy.
5. Records outputs for plots, CSVs, or animation.

Robots learn search/rest ratio strategies. Higher ratios mean more search time
relative to rest time; lower ratios mean more rest and less congestion pressure.
After each trip, the strategy just used is rewarded or punished using the trip's
net energy payoff, risk curvature, loss aversion, and the robot's current energy
condition. Sector scores provide a simple spatial memory for movement: empty
searching reduces a sector's score, while successful grabbing increases it.

The central mechanism is a congestion trade-off. More active robots create more
food opportunities, but also increase collisions and wasted energy. The swarm's
performance depends on how individual learning settles within that shared
environment.

## Install

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

## Run

Create report plots:

```bash
python cli.py plots --out outputs
```

Run Morris sensitivity:

```bash
python cli.py sensitivity --out outputs
```

Run Sobol sensitivity:

```bash
python cli.py sobol --out outputs/sobol
```

Create the animation:

```bash
python cli.py animation --out outputs/swarm_animation.gif
```

Run the full output workflow:

```bash
python cli.py all --out outputs
```

Useful options:

- `--config FILE`: configuration file.
- `--out PATH`: output folder or GIF path.
- `--seconds N`: simulated seconds.
- `--seeds N`: repeated seeds for averaging.
- `--samples N`: sensitivity sample count.
- `--fast`, `--robust`: Morris or Sobol presets.
- `--extensive`: larger Sobol preset.
- `--workers N`: Sobol worker count, or `auto`.
- `--fps N`: animation frame rate.
- `--playback-seconds N`: GIF playback duration.

## Outputs

The main output folder contains:

- `presentation_plots/`: figures for the final presentation.
- `misc_plots/`: appendix and diagnostic figures.
- `presentation_plots/split_panels/`: single-panel presentation figures.
- `supporting_plots/split_panels/`: single-panel supporting figures.
- `data/`: CSV files behind the plots and sensitivity rankings.
- `swarm_animation.gif`: animation output when requested.

Key presentation figures:

- `slide1_abm_mechanism_schematic.png`
- `slide1_strategy_learning_entropy.png`
- `slide2_swarm_size_congestion.png`
- `slide2_throughput_scaling.png`
- `slide2_morris_sensitivity_summary.png`
- `slide2_morris_interaction_diagnostics.png`

## Files

- `agents.py`: individual robot model, PFSM states, learning, and diagnostics.
- `map.py`: arena geometry and transition probabilities.
- `macro.py`: aggregate model for optional macro traces.
- `plot.py`: report figures and CSV exports.
- `sensitivity.py`: Morris screening workflow.
- `sobol_sensitivity.py`: Sobol sensitivity workflow.
- `animation.py`: swarm GIF animation.
- `cli.py`: command-line runner.
- `default_config.yaml`: default model settings.

## Interpretation

- The plots measure the existing model; they do not change its rules.
- Morris ranks influential parameters but is still a screening method.
- Sobol estimates variance contributions but depends on sample size and noise.
- Scaling results apply to the tested swarm sizes, not a universal law.
- Strategy adaptation represents bounded rationality, not global optimization.

For sensitivity details, see `SENSITIVITY_README.md`. For the strategy-learning
and game-theory logic, see `Game Theory Strategy Evolution.md`.
