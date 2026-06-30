# Model Explanation: Individual Robot Swarm Model

This document explains the logic of `agents.py` in a simple and visual way.

The model is an **agent-based swarm simulation**. Each robot moves through a finite set of states, collects food, pays energy costs, avoids collisions, and learns which **rest-time strategy** works best.

---

## 1. One-sentence summary

Each robot repeatedly searches for food, grabs it, deposits it, rests, and then searches again. Over time, each robot learns whether it should rest for a short or long time before going out again.

The main learned strategy is:

$$T_r \in \{20, 60, 100, 160\}\text{ seconds}$$

where $T_r$ means **rest time**.

---

## 2. What is an agent?

Each robot is one `Agent`.

Each agent stores:

| Variable | Meaning |
|---|---|
| `state` | What the robot is currently doing |
| `timer` | How long it has left in the current state |
| `search_credit` | Remaining search time before giving up |
| `energy` | Robot's accumulated energy |
| `trip_delta` | Net energy gained or lost during the current trip |
| `trip_collisions` | Collisions during the current trip |
| `trip_food_encounters` | Food encounters during the current trip |
| `memory_gamma_r` | Recent personal collision rates |
| `memory_gamma_f` | Recent personal food-finding rates |
| `strategies` | Possible rest-time choices |
| `propensities` | Learned preference weights for each strategy |
| `sector_scores` | Learned preference for four spatial sectors |

---

## 3. Robot states

Each robot can be in one of six states:

| State | Meaning |
|---|---|
| `searching` | Looking for food |
| `grabbing` | Found food and is trying to grab it |
| `deposit` | Carrying food back to the nest |
| `homing` | Returning home after failing to find food |
| `resting` | Waiting before searching again |
| `avoidance` | Temporarily avoiding a collision |

---

## 4. Visual state machine

### Successful trip

```text
SEARCHING
   |
   | finds food
   v
GRABBING
   |
   | finishes grabbing
   v
DEPOSIT
   |
   | finishes deposit
   v
RESTING
   |
   | rest timer ends
   v
SEARCHING again
```

### Failed search

```text
SEARCHING
   |
   | search time runs out
   v
HOMING
   |
   | reaches home
   v
RESTING
   |
   | rest timer ends
   v
SEARCHING again
```

### Collision interruption

```text
SEARCHING / GRABBING / DEPOSIT / HOMING
   |
   | collision happens
   v
AVOIDANCE
   |
   | avoidance timer ends
   v
return to interrupted state
```

So `avoidance` is not a final state. It is a temporary interruption.

---

## 5. What happens in one simulation step?

Each simulation step follows this order:

```text
ONE SIMULATION STEP

1. Count how many robots are in each state

2. Compute global event probabilities:
      gamma_f = probability of finding food
      gamma_r = probability of robot collision
      gamma_l = probability of losing food target

3. Loop through every robot:
      a. update its display position
      b. subtract energy cost
      c. update its state logic
      d. maybe update sector score
      e. maybe enter rest and update strategy

4. Update food count

5. Update total swarm energy

6. Save output statistics
```

The important point is:

> The model computes global probabilities once per simulation step, then applies them to individual robots.

---

## 6. The three key probabilities

The model uses three important probabilities.

### 6.1 Food finding probability

$$\gamma_f = \text{probability of finding food}$$

If there is more food, robots are more likely to find food.

---

### 6.2 Robot collision probability

$$\gamma_r = \text{probability of collision}$$

This depends on how many robots are active.

$$N_{active} = N_{robots} - N_{resting}$$

If many robots are active:

$$N_{active} \uparrow \Rightarrow \gamma_r \uparrow$$

So when many robots choose short rest times, they create congestion.

---

### 6.3 Food loss probability

$$\gamma_l = \text{probability of losing the food target}$$

This depends on competition around food.

$$N_{competing} = N_{searching} + N_{grabbing} + N_{avoidance}$$

If many robots are competing:

$$N_{competing} \uparrow \Rightarrow \gamma_l \uparrow$$

So food is harder to secure when too many robots are active.

---

## 7. Searching logic

When a robot is in `searching`, the model draws a random number:

$$u \sim U(0,1)$$

Then the robot decides what happens:

```text
if u < gamma_r:
    collision -> avoidance

else if u < gamma_r + gamma_f:
    food found -> grabbing

else:
    continue searching
```

Mathematically:

$$P(\text{collision}) = \gamma_r$$

$$P(\text{find food}) = \gamma_f$$

$$P(\text{keep searching}) = 1 - \gamma_r - \gamma_f$$

If the robot searches for too long without finding food, its search credit reaches zero:

$$search\_credit \leq 0$$

Then it gives up and goes home:

```text
SEARCHING -> HOMING
```

---

## 8. Grabbing logic

When a robot is in `grabbing`, it has already found food, but the grab is not guaranteed.

Again the model draws:

$$u \sim U(0,1)$$

Then:

```text
if u < gamma_r:
    collision -> avoidance

else if u < gamma_r + gamma_l:
    lose target -> searching

else:
    continue grabbing
```

Mathematically:

$$P(\text{collision}) = \gamma_r$$

$$P(\text{lose food target}) = \gamma_l$$

$$P(\text{continue grabbing}) = 1 - \gamma_r - \gamma_l$$

If the grabbing timer finishes, the robot moves to `deposit`:

```text
GRABBING -> DEPOSIT
```

At that moment, the food item is removed from the environment.

---

## 9. Deposit logic

When a robot is in `deposit`, it is carrying food back to the nest.

It can still collide:

$$P(\text{collision}) = \gamma_r$$

If there is no collision, its deposit timer decreases.

When the timer reaches zero:

```text
DEPOSIT -> RESTING
```

A completed deposit gives energy reward:

$$+R_{food}$$

---

## 10. Homing logic

When a robot fails to find food before search time expires, it goes to `homing`.

```text
SEARCHING -> HOMING -> RESTING
```

Homing gives no food reward. It only gets the robot back to rest.

---

## 11. Avoidance logic

Avoidance happens after a collision.

The robot remembers:

```text
return_state
return_timer
```

This means it remembers what it was doing before the collision.

Example:

```text
GRABBING
   |
   | collision
   v
AVOIDANCE
   |
   | avoidance timer ends
   v
GRABBING again
```

Avoidance costs time and energy, so collisions reduce efficiency.

---

## 12. Energy logic

Each robot pays energy every simulation step.

If resting:

$$\text{cost} = c_{rest} \cdot dt$$

If active:

$$\text{cost} = c_{active} \cdot dt$$

The robot's trip payoff is updated as:

$$trip\_delta \leftarrow trip\_delta - cost$$

When food is successfully delivered:

$$trip\_delta \leftarrow trip\_delta + R_{food}$$

So the trip payoff is:

$$trip\_delta = \text{food reward} - \text{energy costs}$$

If the robot finds and deposits food efficiently, `trip_delta` is positive.

If it wastes time, collides, or fails to find food, `trip_delta` can be negative.

---

# 13. What are the strategies?

The model gives each robot four possible rest-time strategies:

```text
Strategy 0: rest for 20 seconds
Strategy 1: rest for 60 seconds
Strategy 2: rest for 100 seconds
Strategy 3: rest for 160 seconds
```

Mathematically:

$$T_r \in \{20,60,100,160\}$$

These are the only explicit strategies in the script.

They are not different search algorithms. They are different decisions about **how long to rest before searching again**.

---

## 14. Meaning of each strategy

| Strategy | Behavior | Risk |
|---|---|---|
| 20 s rest | Aggressive | Searches often but creates more congestion |
| 60 s rest | Moderately active | Balanced |
| 100 s rest | Cautious | Less congestion but fewer food attempts |
| 160 s rest | Very cautious | Avoids congestion but may collect little food |

Short rest means:

$$T_r \downarrow \Rightarrow N_{active} \uparrow$$

More active robots means:

$$N_{active} \uparrow \Rightarrow \gamma_r \uparrow$$

So short rest can be good individually, but bad collectively if too many robots do it.

---

# 15. When does a robot choose its strategy?

This is the most important timing point.

A robot does **not** choose a strategy every simulation step.

A robot chooses a new strategy only when it enters `resting`.

The timing is:

```text
Robot completes a trip
        |
        v
Robot enters RESTING
        |
        v
Evaluate the trip payoff
        |
        v
Update the propensity of the strategy just used
        |
        v
Convert propensities into probabilities
        |
        v
Randomly choose next rest-time strategy
        |
        v
Set rest timer to chosen strategy
```

So the strategy update happens at the transition:

```text
DEPOSIT -> RESTING
```

or:

```text
HOMING -> RESTING
```

or after avoidance returns into a completed state.

---

## 16. Visual strategy decision cycle

```text
START OF TRIP
Robot has selected rest strategy T_r

        |
        v
RESTING for T_r seconds

        |
        v
SEARCHING / GRABBING / DEPOSIT / HOMING / AVOIDANCE

        |
        v
Trip ends and robot enters RESTING

        |
        v
Compute trip_delta

        |
        v
Apply risk/loss utility

        |
        v
Update propensity of the strategy it just used

        |
        v
Sample the next strategy

        |
        v
New T_r is used for the next rest period
```

In simple words:

> The robot chooses its next rest strategy only after seeing how good or bad the previous trip was.

---

# 17. Propensities: how strategies are remembered

Each robot has four strategy weights called propensities:

$$q = [q_{20}, q_{60}, q_{100}, q_{160}]$$

At the beginning:

$$q = [10,10,10,10]$$

So all strategies are equally likely.

The probability of choosing strategy $i$ is:

$$P_i = \frac{q_i}{\sum_j q_j}$$

At the start:

$$P_i = \frac{10}{40} = 0.25$$

So each strategy has 25% probability.

---

## 18. How a strategy is updated

At the end of a trip, the robot evaluates the strategy it just used.

Let:

$$x = trip\_delta$$

where $x$ is the net payoff from the trip.

If the trip made energy:

$$x > 0$$

If the trip lost energy:

$$x < 0$$

The model converts $x$ into subjective utility.

---

# 19. Risk aversion and loss aversion

The script uses a prospect-theory-style utility function.

For gains:

$$U(x) = x^\alpha \quad \text{if } x \geq 0$$

For losses:

$$U(x) = -\lambda(-x)^\alpha \quad \text{if } x < 0$$

The parameters are:

$$\alpha = 0.88$$

$$\lambda = 2.25$$

---

## 20. What does risk aversion mean here?

Because:

$$0 < \alpha < 1$$

large gains have diminishing value.

Example:

$$100^{0.88} \approx 57.5$$

$$200^{0.88} \approx 105.9$$

Doubling the gain from 100 to 200 does not double the utility.

This makes the robot less obsessed with very large rewards.

That is risk aversion for gains.

---

## 21. What does loss aversion mean here?

Losses are multiplied by:

$$\lambda = 2.25$$

So losses hurt more than equal gains help.

Example:

$$U(100) = 100^{0.88} \approx 57.5$$

$$U(-100) = -2.25(100^{0.88}) \approx -129.5$$

So losing 100 energy hurts more than gaining 100 energy helps.

In plain English:

> A bad trip strongly punishes the strategy that caused it.

---

# 22. Learning reward

The subjective utility is normalized:

$$utility\_factor = \frac{U(x)}{R_{food}^{\alpha}}$$

The robot also considers its current energy level:

$$energy\_factor = \frac{energy - R_{food}}{R_{food}}$$

Then the final learning reward is:

$$learning\_reward = 5 \cdot utility\_factor + 2 \cdot energy\_factor$$

So the update uses:

```text
recent trip performance  +  overall energy condition
```

The recent trip is weighted more strongly because it has coefficient 5.

---

## 23. Propensity update equation

Only the strategy that was just used gets updated.

If strategy $i$ was used, then:

$$q_i(t+1) = (1 - \rho)q_i(t) + learning\_reward$$

where:

$$\rho = 0.05$$

So:

$$q_i(t+1) = 0.95q_i(t) + learning\_reward$$

The script also prevents propensities from falling below 0.1:

$$q_i(t+1) = \max(0.1, q_i(t+1))$$

This means bad strategies become unlikely, but never impossible.

---

## 24. Strategy selection example

Suppose a robot has:

$$q = [20,10,5,5]$$

Then:

$$\sum q = 40$$

The probabilities are:

$$P_{20} = \frac{20}{40} = 0.50$$

$$P_{60} = \frac{10}{40} = 0.25$$

$$P_{100} = \frac{5}{40} = 0.125$$

$$P_{160} = \frac{5}{40} = 0.125$$

So the robot is most likely to choose 20 seconds, but it can still explore other strategies.

---

# 25. Snooze decision while resting

There is one extra decision inside `resting`.

When the rest timer ends, the robot may choose to rest a bit longer if recent conditions look bad.

This is not a full strategy update. It is a temporary safety check.

The robot remembers recent personal collision and food rates:

$$\bar{\gamma}_r = \text{average recent collision rate}$$

$$\bar{\gamma}_f = \text{average recent food encounter rate}$$

It estimates the probability of finding food during a search:

$$P(\text{find food}) = 1 - (1 - \bar{\gamma}_f)^{T_s}$$

Expected reward:

$$E[R] = P(\text{find food})R_{food}$$

Expected base cost:

$$C_{base} = T_s c_{active}dt$$

Expected collision cost:

$$C_{collision} = \bar{\gamma}_r T_s T_a c_{active}dt$$

Total expected cost:

$$E[C] = C_{base} + C_{collision}$$

The robot may snooze if:

$$\bar{\gamma}_r > congestion\_tolerance$$

and:

$$E[R] < E[C]$$

If both are true, it snoozes with probability 0.75.

If it snoozes:

$$timer = 0.5T_r$$

Simple interpretation:

> If recent collisions are high and expected reward is not worth the expected cost, the robot waits longer before searching again.
---

# 26. Sector scores: what they are

Each robot also has four sector scores:

$$sector\_scores = [s_0, s_1, s_2, s_3]$$

The arena is split into four quadrants:

```text
          y+
          |
    S1    |    S0
          |
----------+---------- x+
          |
    S2    |    S3
          |
```

The exact sector is calculated from the robot's angle:

$$\theta = atan2(y,x)$$

Then the angle is converted into one of four sectors.

---

# 27. When are sector scores updated?

Sector scores are updated during the robot's trip, not at the end of the trip.

There are two updates:

## 27.1 During unsuccessful searching

If the robot is searching and does **not** find food, the score of its current sector decreases slightly:

$$s_k \leftarrow \max(0, s_k - sector\_decay)$$

This happens inside the searching step when nothing useful happens.

Visual:

```text
Robot searches in sector k
        |
        | no food found this step
        v
sector_scores[k] decreases slightly
```

Meaning:

> This area seems less useful, so trust it a little less.

---

## 27.2 After successful grabbing

If the robot successfully finishes grabbing food, the score of its current sector increases:

$$s_k \leftarrow s_k + sector\_reward$$

Visual:

```text
Robot grabs food successfully in sector k
        |
        v
sector_scores[k] increases
```

Meaning:

> This area produced food, so trust it more next time.

---

# 28. How sector scores affect movement

When a robot is searching, it looks at its sector scores:

$$best\_sector = \arg\max_k s_k$$

If all scores are zero, it just wanders randomly.

If one sector has the highest score, the robot steers toward the center of that sector.

Visual:

```text
sector_scores = [0.0, 3.0, 1.0, 0.0]

Best sector = S1

Robot heading is gently pulled toward S1
```

The steering is not instant. The model blends the current heading with the target sector direction:

$$heading \leftarrow heading + w \cdot angle\_gap$$

where:

$$w = sector\_pull$$

Default:

$$sector\_pull = 0.20$$

So the robot turns gradually, not sharply.

---

## 29. Important caveat about sector learning

Sector learning affects the display movement.

But the actual probability of finding food is still computed globally using:

$$\gamma_f$$

So the visual motion may look spatially intelligent, but the core food-finding event is not fully local.

In simple words:

> Sector scores guide where the robot appears to move, but food discovery is still mostly controlled by the global probability model.

---

# 30. Visual timeline of strategy and sector decisions

This is the clearest way to see the timing.

```text
ONE ROBOT OVER TIME

RESTING
  |
  | timer ends
  | maybe snooze if congestion looks bad
  v
SEARCHING
  |
  | every searching step:
  |   - move
  |   - steer toward best sector if one exists
  |   - maybe collide
  |   - maybe find food
  |   - if no food, reduce current sector score
  v
GRABBING
  |
  | if grab succeeds:
  |   - increase current sector score
  v
DEPOSIT
  |
  | if deposit completes:
  v
RESTING
  |
  | NOW strategy is updated
  | NOW next rest-time strategy is chosen
  v
RESTING for new chosen duration
```

Key difference:

```text
Sector score updates = during search/grab experience
Strategy updates     = only when the trip ends and robot enters resting
```

---

# 31. Game theory interpretation

The model behaves like a congestion game.

Each robot chooses a rest-time strategy:

$$T_r \in \{20,60,100,160\}$$

The payoff depends on its own choice and the choices of other robots.

If many robots choose short rest:

$$T_r \downarrow \Rightarrow N_{active} \uparrow$$

Then:

$$N_{active} \uparrow \Rightarrow \gamma_r \uparrow$$

Then:

$$\gamma_r \uparrow \Rightarrow \text{more collisions}$$

Then:

$$\text{more collisions} \Rightarrow \text{lower payoff}$$

So one robot's strategy changes the environment for everyone else.

That is the game-theoretic part.

---

# 32. Social dilemma

Short rest can be individually attractive because the robot searches more often:

$$T_r \downarrow \Rightarrow \text{more food opportunities}$$

But if too many robots do this:

$$\gamma_r \uparrow$$

and collisions increase.

So the swarm has a trade-off:

| Too many short rests | Too many long rests |
|---|---|
| High congestion | Low activity |
| Many collisions | Fewer food attempts |
| Wasted energy | Slow collection |

The best collective behavior may be a mixture of strategies.

---

# 33. Bounded rationality

The robots are not perfectly rational.

They do not solve the exact optimal strategy mathematically.

Instead, each robot uses:

- recent memory,
- noisy trial-and-error learning,
- subjective utility,
- loss aversion,
- a simple congestion check,
- local sector scores.

This is bounded rationality.

The robot learns from experience, but imperfectly.

---

# 34. Evolution of strategies

The strategy evolution is not biological evolution.

There is no reproduction and no death.

Instead, strategies evolve as probabilities inside each robot.

A strategy that produces good trips gets a higher propensity:

$$q_i \uparrow \Rightarrow P_i \uparrow$$

A strategy that produces bad trips gets a lower propensity:

$$q_i \downarrow \Rightarrow P_i \downarrow$$

Over time, the swarm can shift toward the rest times that work better under current congestion and food conditions.

---
# 35. Full model logic in one diagram

```text
GLOBAL MODEL STEP

Count states
   |
   v
Compute gamma_f, gamma_r, gamma_l
   |
   v
Update each robot
   |
   +--> If searching:
   |        collide / find food / keep searching
   |        update sector score if no food
   |
   +--> If grabbing:
   |        collide / lose target / finish grab
   |        reward sector if grab finishes
   |
   +--> If deposit:
   |        collide / finish delivery
   |
   +--> If homing:
   |        collide / reach home
   |
   +--> If avoidance:
   |        wait, then return to previous state
   |
   +--> If resting:
            timer decreases
            maybe snooze
            if trip just ended:
                update strategy propensities
                choose next strategy

Update food
   |
   v
Update energy
   |
   v
Record output row
```

---

# 36. The most important takeaway

There are two learning systems:

## Strategy learning

```text
What does the robot learn?
How long to rest.

When does it update?
When the robot enters resting after a trip.

What reward does it use?
Trip energy payoff transformed by risk/loss utility.
```

## Sector learning

```text
What does the robot learn?
Which arena sector seems promising.

When does it update?
During searching and successful grabbing.

What reward does it use?
Food success increases sector score; empty searching decreases it.
```

Final simple version:

```text
Rest strategy = learned after each trip.
Sector score  = learned during movement and food search.
```
