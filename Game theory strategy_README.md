# Strategy Evolution and Game-Theory Logic

## Core Model

Each robot is one agent with its own PFSM state, position, energy, trip memory,
strategy propensities, and sector scores.

States:

- `searching`: looking for food.
- `grabbing`: trying to secure food after finding it.
- `deposit`: returning food to the nest.
- `homing`: returning home after a failed search.
- `resting`: waiting before the next search.
- `avoidance`: temporary collision recovery.

One simulation step:

```text
count robot states
compute gamma_f, gamma_r, gamma_l
update each robot
update food availability
update swarm energy
record outputs
```

The three key probabilities are:

- `gamma_f`: probability of finding food.
- `gamma_r`: probability of collision, increasing with active robots.
- `gamma_l`: probability of losing a food target, increasing with competition.

`avoidance` is temporary. When it ends, the robot returns to the interrupted
state unless the timer has already expired.

## Energy

Robots pay an energy cost every step. Resting is cheaper than active movement.
Successful delivery adds the food reward.

Trip payoff is:

```text
trip_delta = food reward - active/resting costs - collision delays
```

A fast successful trip is positive. A long failed or collision-heavy trip can be
negative.

## Strategies

The learned strategies are search/rest ratios:

```text
0.5, 1.0, 1.5, 2.0
```

The model converts each ratio into paired search and rest timers over a fixed
cycle length.

Higher ratio:

- more search time
- less rest time
- more food opportunities
- more congestion pressure

Lower ratio:

- less search time
- more rest time
- fewer collisions
- fewer food opportunities

Robots do not choose a new strategy every step. A strategy is updated and
resampled only when a trip ends and the robot enters `resting`.

## Propensity Learning

Each robot stores one propensity per strategy:

```text
q = [q_0.5, q_1.0, q_1.5, q_2.0]
```

All propensities start equal. Strategy probabilities are proportional to their
propensities:

```text
P_i = q_i / sum(q)
```

After a trip, only the strategy just used is updated:

```text
q_i <- max(0.1, (1 - rho) * q_i + learning_reward)
```

with `rho = 0.05`. The lower bound keeps every strategy possible, so robots can
continue exploring.

## Risk and Loss Utility

Trip payoff is converted into subjective utility:

```text
U(x) = x^alpha                  if x >= 0
U(x) = -lambda * (-x)^alpha     if x < 0
```

Default values:

```text
alpha = 0.88
lambda = 2.25
```

`alpha < 1` gives diminishing sensitivity to gains. `lambda > 1` makes losses
hurt more than equal gains help.

The learning reward combines subjective trip utility and the robot's current
energy condition:

```text
learning_reward = 5 * utility_factor + 2 * energy_factor
```

Recent trip performance carries the larger weight.

## Snooze Check

When resting ends, a robot may wait longer before searching again. It compares
recent collision and food-finding memory:

```text
if recent collision risk is high
and expected reward is below expected cost
then snooze with probability 0.75
```

If it snoozes, the robot rests for half of its current rest timer. This is a
temporary congestion response, not a full strategy update.

## Sector Scores

Each robot keeps four sector scores for the arena quadrants.

- Empty searching slightly lowers the current sector score.
- Successful grabbing increases the current sector score.
- During search, the robot's heading is pulled toward its best-scored sector.

Sector scores guide movement, but food discovery is still governed by the
global food-finding probability `gamma_f`. The movement can look spatially
adaptive without making food detection fully local.

## Game-Theory Interpretation

The model behaves like a congestion game. A robot's strategy affects its own
search/rest timing, but it also changes the shared environment by altering how
many robots are active.

The causal chain is:

```text
higher search/rest ratio
more active robots
higher collision probability
more delays and energy loss
lower payoff
```

Short rest can be individually attractive because it creates more food
opportunities. If too many robots search aggressively, congestion rises and
everyone pays the cost. If too many robots rest for too long, the swarm collects
food slowly. Good swarm performance can therefore require a mixture of
strategies.

## Bounded Rationality

Robots do not solve for a global optimum. They adapt through:

- noisy trial-and-error learning
- recent collision and food memory
- risk and loss sensitivity
- simple congestion checks
- local sector scores

Strategy evolution here is not biological evolution. There is no reproduction
or death. Strategies evolve as probabilities inside each robot: good trip
outcomes raise a strategy's propensity, while bad outcomes reduce it.

## Timing

```text
RESTING
  timer ends
  maybe snooze

SEARCHING
  steer by sector score
  maybe collide
  maybe find food
  cool sector if nothing found

GRABBING
  maybe collide
  maybe lose target
  reward sector if grab completes

DEPOSIT or HOMING
  return to nest

RESTING
  evaluate trip_delta
  update used strategy
  sample next strategy
```

Main takeaway:

```text
Strategy learning happens after each trip.
Sector learning happens during search and grabbing.
Congestion links individual choices to collective performance.
```
