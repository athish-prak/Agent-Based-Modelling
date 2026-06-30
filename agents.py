from __future__ import annotations

from dataclasses import dataclass, field
import math
import random

import pandas as pd

from config import Config
from map import PaperMap


@dataclass
class Agent:
    """Hold one robot's PFSM state, position, memory, and learning weights."""

    state: str = "searching"
    timer: int = 0
    search_credit: int = 0
    return_state: str = ""
    return_timer: int = 0
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0
    target_x: float | None = None
    target_y: float | None = None
    
    energy: float = 2000.0
    trip_delta: float = 0.0
    
    trip_collisions: int = 0
    trip_food_encounters: int = 0
    trip_active_steps: int = 0
    memory_gamma_r: list[float] = field(default_factory=list)
    memory_gamma_f: list[float] = field(default_factory=list)
    
    strategy_ratios: list[float] = field(default_factory=list)
    search_strategy_steps: list[int] = field(default_factory=list)
    rest_strategy_steps: list[int] = field(default_factory=list)
    propensities: list[float] = field(default_factory=list)
    current_strategy_idx: int = 0
    sector_scores: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])


def get_sector(x: float, y: float) -> int:
    """Return the zero-based quadrant sector for an arena position.

    Inputs:
        x and y are the robot's display coordinates in metres.

    Output:
        An integer sector index from 0 to 3, moving counter-clockwise from the
        positive x-axis.
    """
    angle = math.atan2(y, x)
    if angle < 0.0:
        angle += 2.0 * math.pi
    return min(3, max(0, int(angle // (math.pi / 2.0))))


class MicroModel:
    """Track each robot's state, timers, energy contribution, and display position."""

    def __init__(self, cfg: Config, rest_time_s: float | None = None, seed: int | None = None, **kwargs):
        """Create a micro model from configuration and optional behaviour parameters."""
        self.cfg = cfg
        self.world = PaperMap.from_config(cfg)
        seed_value = seed if seed is not None else int(cfg.get("run", "random_seed", default=1))
        self.rng = random.Random(seed_value)
        self.food_rng = random.Random(seed_value + 1)
        self.rest_time_s = float(rest_time_s if rest_time_s is not None else cfg.get("behaviour", "default_rest_time_s"))
        self.base_search_time_s = float(cfg.get("behaviour", "search_time_s"))
        self.strategy_cycle_time_s = float(
            cfg.get(
                "behaviour",
                "strategy_cycle_time_s",
                default=self.base_search_time_s + self.rest_time_s,
            )
        )
        self.strategy_ratios = self._load_strategy_ratios()
        
        self.alpha = kwargs.get("alpha", 0.88)
        self.lambda_loss = kwargs.get("lambda_loss", 2.25)
        self.recency = kwargs.get("recency", 0.05)
        self.congestion_tolerance = kwargs.get("congestion_tolerance", 0.04)
        self.gamma_r_scale = kwargs.get("gamma_r_scale", 1.0)
        self.sector_decay = float(kwargs.get("sector_decay", 0.005))
        self.sector_reward = float(kwargs.get("sector_reward", 1.0))
        self.sector_pull = float(kwargs.get("sector_pull", 0.20))
        
        self.ts = self.world.steps(self.base_search_time_s)
        self.ta = self.world.steps(float(cfg.get("behaviour", "avoidance_time_s")))
        self.tg = self.world.steps(self.world.tau_grab)
        self.td = self.world.steps(self.world.tau_deposit)
        self.th = self.world.steps(self.world.tau_home)
        self.food = float(cfg.get("food", "initial_count"))
        self.energy = 0.0
        self.resting_cost = float(cfg.get("energy", "resting_cost"))
        self.active_cost = float(cfg.get("energy", "active_cost"))
        self.food_reward = float(cfg.get("energy", "food_reward"))
        self.agents = [self._new_agent() for _ in range(self.world.n_robots)]
        self.food_positions: list[tuple[float, float]] = []
        self.total_collisions = 0.0
        self.total_completed_deposit = 0.0
        self.total_entered_deposit = 0.0
        self._sync_food_positions()

    def _new_agent(self) -> Agent:
        """Create one robot with a random starting position and strategy weights."""
        r = self.rng.uniform(self.world.rinner, self.world.router)
        theta = self.rng.uniform(0.0, 2.0 * math.pi)
        search_steps, rest_steps = self._build_ratio_strategy_steps()
        num_strategies = len(self.strategy_ratios)
        
        start_idx = self.rng.randint(0, num_strategies - 1)
        starting_search_steps = search_steps[start_idx]
        
        return Agent(
            state="searching",
            timer=starting_search_steps,
            search_credit=starting_search_steps,
            return_state="",
            return_timer=0,
            x=r * math.cos(theta),
            y=r * math.sin(theta),
            heading=theta,
            energy=self.food_reward,  
            trip_delta=0.0,
            trip_collisions=0,
            trip_food_encounters=0,
            trip_active_steps=0,
            memory_gamma_r=[],
            memory_gamma_f=[],
            strategy_ratios=list(self.strategy_ratios),
            search_strategy_steps=search_steps,
            rest_strategy_steps=rest_steps,
            propensities=[10.0] * num_strategies,
            current_strategy_idx=start_idx
        )

    def _load_strategy_ratios(self) -> list[float]:
        """Read and validate search/rest ratio strategies from configuration.

        Input:
            The model configuration, optionally containing
            behaviour.strategy_search_rest_ratios.

        Output:
            A non-empty list of positive search_time/rest_time ratios.
        """
        raw = self.cfg.get("behaviour", "strategy_search_rest_ratios", default=[0.5, 1.0, 1.5, 2.0])
        ratios = [float(value) for value in raw]
        ratios = [value for value in ratios if value > 0.0 and math.isfinite(value)]
        if not ratios:
            raise ValueError("behaviour.strategy_search_rest_ratios must contain at least one positive value")
        return ratios

    def _build_ratio_strategy_steps(self) -> tuple[list[int], list[int]]:
        """Convert ratio strategies into paired search and rest timers.

        Input:
            The configured strategy cycle length and search/rest ratios.

        Output:
            Two lists of timer steps: one for search duration and one for rest
            duration, aligned by strategy index.
        """
        search_steps: list[int] = []
        rest_steps: list[int] = []
        for ratio in self.strategy_ratios:
            rest_time_s = self.strategy_cycle_time_s / (1.0 + ratio)
            search_time_s = self.strategy_cycle_time_s - rest_time_s
            search_steps.append(max(1, self.world.steps(search_time_s)))
            rest_steps.append(max(0, self.world.steps(rest_time_s)))
        return search_steps, rest_steps

    def run(self, seconds: float | None = None, stride: int = 1, keep_frames: bool = False) -> pd.DataFrame:
        """Run the model and return sampled state, energy, and optional frame data."""
        steps = self.world.steps(float(seconds if seconds is not None else self.cfg.duration_s))
        rows: list[dict[str, float]] = []
        stride = max(1, int(stride))
        for step in range(steps):
            stats = self.step()
            if step % stride == 0 or step == steps - 1:
                stats["time_s"] = (step + 1) * self.world.dt
                stats["rest_time_s"] = self.rest_time_s
                stats["energy"] = self.energy
                stats["food_items"] = self.food
                if keep_frames:
                    stats["positions"] = [(a.x, a.y, a.state) for a in self.agents]
                    stats["food_positions"] = list(self.food_positions)
                rows.append(stats)
        return pd.DataFrame(rows)

    def step(self) -> dict[str, float]:
        """Advance the swarm by one time step and return aggregate counters."""
        counts = self.counts()
        active = self.world.n_robots - counts["resting"]
        
        gamma_f = self.world.find_probability(self.food)
        gamma_r = max(0.0, min(1.0, self.world.collision_probability(active) * self.gamma_r_scale))
        competing = counts["searching"] + counts["grabbing"] + counts["avoidance"]
        gamma_l = self.world.loss_probability(self.food, competing, self.tg)

        if gamma_f + gamma_r > 0.98:
            scale = 0.98 / (gamma_f + gamma_r)
            gamma_f *= scale
            gamma_r *= scale
        if gamma_l + gamma_r > 0.98:
            scale = 0.98 / (gamma_l + gamma_r)
            gamma_l *= scale
            gamma_r *= scale

        entered_deposit = 0
        completed_deposit = 0

        for agent in self.agents:
            self._update_display_position(agent)

            if agent.state == "resting":
                cost = self.resting_cost * self.world.dt
            else:
                cost = self.active_cost * self.world.dt
                agent.trip_active_steps += 1
                
            agent.energy -= cost
            agent.trip_delta -= cost

            if agent.state == "searching":
                entered_deposit += self._step_searching(agent, gamma_f, gamma_r)

            elif agent.state == "grabbing":
                entered_deposit += self._step_grabbing(agent, gamma_l, gamma_r)

            elif agent.state == "deposit":
                done = self._step_deposit(agent, gamma_r)
                if done > 0:
                    agent.energy += self.food_reward
                    agent.trip_delta += self.food_reward
                completed_deposit += done

            elif agent.state == "homing":
                self._step_homing(agent, gamma_r)

            elif agent.state == "avoidance":
                done_deposit = self._step_avoidance(agent)
                if done_deposit > 0:
                    agent.energy += self.food_reward
                    agent.trip_delta += self.food_reward
                completed_deposit += done_deposit

            elif agent.state == "resting":
                self._step_resting(agent)

        self.food = max(0.0, self.food + self.world.p_new_per_step - entered_deposit)
        self.total_entered_deposit += float(entered_deposit)
        self.total_completed_deposit += float(completed_deposit)
        self._sync_food_positions()
        counts = self.counts()
        active = self.world.n_robots - counts["resting"]
        self.energy += self.food_reward * completed_deposit - self.world.dt * (
            self.resting_cost * counts["resting"] + self.active_cost * active
        )
        ratio_stats = self._ratio_stats()
        counts["gamma_f"] = gamma_f
        counts["gamma_r"] = gamma_r
        counts["gamma_l"] = gamma_l
        counts["entered_deposit"] = float(entered_deposit)
        counts["completed_deposit"] = float(completed_deposit)
        counts["active_fraction"] = active / max(1, self.world.n_robots)
        counts["total_collisions"] = self.total_collisions
        counts["total_entered_deposit"] = self.total_entered_deposit
        counts["total_completed_deposit"] = self.total_completed_deposit
        counts.update(ratio_stats)
        return counts

    def _step_searching(self, agent: Agent, gamma_f: float, gamma_r: float) -> int:
        """Update one searching robot and return newly grabbed food count."""
        if agent.search_credit <= 0:
            self._go_homing(agent)
            return 0
        u = self.rng.random()
        if u < gamma_r:
            self._go_avoidance(agent, "searching", agent.timer)
        elif u < gamma_r + gamma_f:
            agent.state = "grabbing"
            agent.timer = self.tg
            agent.trip_food_encounters += 1
            self._assign_food_target(agent)
        else:
            agent.search_credit -= 1
            agent.timer = agent.search_credit
            self._cool_current_sector(agent)
            if agent.search_credit <= 0:
                self._go_homing(agent)
        return 0

    def _step_grabbing(self, agent: Agent, gamma_l: float, gamma_r: float) -> int:
        """Update one grabbing robot and return food entering deposit."""
        if agent.search_credit <= 0:
            self._go_homing(agent)
            return 0
        u = self.rng.random()
        if u < gamma_r:
            self._go_avoidance(agent, "grabbing", agent.timer)
            return 0
        if u < gamma_r + gamma_l:
            self._clear_food_target(agent)
            agent.state = "searching"
            agent.timer = max(0, agent.search_credit)
            return 0
        agent.search_credit -= 1
        agent.timer -= 1
        if agent.search_credit <= 0:
            self._go_homing(agent)
        elif agent.timer <= 0:
            self._consume_food_target(agent)
            self._reward_current_sector(agent)
            agent.state = "deposit"
            agent.timer = self.td
            return 1
        return 0

    def _step_deposit(self, agent: Agent, gamma_r: float) -> int:
        """Update one depositing robot and return completed deliveries."""
        if self.rng.random() < gamma_r:
            self._go_avoidance(agent, "deposit", agent.timer)
            return 0
        agent.timer -= 1
        if agent.timer <= 0:
            self._go_resting(agent)
            return 1
        return 0

    def _step_homing(self, agent: Agent, gamma_r: float) -> None:
        """Update one homing robot."""
        if self.rng.random() < gamma_r:
            self._go_avoidance(agent, "homing", agent.timer)
            return
        agent.timer -= 1
        if agent.timer <= 0:
            self._go_resting(agent)

    def _step_avoidance(self, agent: Agent) -> int:
        """Update one avoiding robot and return any delivery completed during avoidance."""
        agent.timer -= 1
        if agent.return_state in {"searching", "grabbing"}:
            agent.search_credit = max(0, agent.search_credit - 1)
        if agent.return_state in {"grabbing", "deposit", "homing"}:
            agent.return_timer -= 1

        if agent.timer > 0:
            return 0

        previous = agent.return_state
        return_timer = agent.return_timer
        agent.return_state = ""
        agent.return_timer = 0

        if previous == "searching":
            if agent.search_credit <= 0:
                self._go_homing(agent)
            else:
                agent.state = "searching"
                agent.timer = agent.search_credit
            return 0

        if previous == "grabbing":
            if agent.search_credit <= 0:
                self._go_homing(agent)
                return 0
            if return_timer <= 0:
                self._consume_food_target(agent)
                self._reward_current_sector(agent)
                agent.state = "deposit"
                agent.timer = self.td
                return 1
            agent.state = "grabbing"
            agent.timer = return_timer
            return 0

        if previous == "deposit":
            if return_timer <= 0:
                self._go_resting(agent)
                return 1
            agent.state = "deposit"
            agent.timer = return_timer
            return 0

        if previous == "homing":
            if return_timer <= 0:
                self._go_resting(agent)
            else:
                agent.state = "homing"
                agent.timer = return_timer
            return 0

        agent.state = "searching"
        agent.timer = max(0, agent.search_credit)
        return 0

    def _step_resting(self, agent: Agent) -> None:
        """Update one resting robot and decide whether it wakes or waits longer."""
        agent.timer -= 1
        if agent.timer <= 0:
            snooze = False
            
            if len(agent.memory_gamma_r) > 0:
                avg_gamma_r = sum(agent.memory_gamma_r) / len(agent.memory_gamma_r)
                avg_gamma_f = sum(agent.memory_gamma_f) / len(agent.memory_gamma_f)
                search_steps = self._current_search_steps(agent)
                
                prob_find_in_trip = 1.0 - (1.0 - avg_gamma_f) ** search_steps
                expected_reward = prob_find_in_trip * self.food_reward
                
                base_cost = search_steps * (self.active_cost * self.world.dt)
                collision_cost = avg_gamma_r * search_steps * self.ta * (self.active_cost * self.world.dt)
                expected_cost = base_cost + collision_cost
                
                if avg_gamma_r > self.congestion_tolerance and expected_reward < expected_cost:
                    if self.rng.random() < 0.75: 
                        snooze = True
            
            if snooze:
                agent.timer = max(1, int(self._current_rest_steps(agent) * 0.5))
            else:
                agent.state = "searching"
                agent.search_credit = self._current_search_steps(agent)
                agent.timer = agent.search_credit

    def _go_avoidance(self, agent: Agent, previous: str, previous_timer: int) -> None:
        """Move a robot into avoidance while remembering the interrupted state."""
        agent.return_state = previous
        agent.return_timer = previous_timer
        agent.state = "avoidance"
        agent.timer = self.ta
        agent.trip_collisions += 1
        self.total_collisions += 1.0

    def _go_homing(self, agent: Agent) -> None:
        """Send a robot back toward the nest."""
        self._clear_food_target(agent)
        agent.state = "homing"
        agent.timer = self.th
        agent.return_state = ""
        agent.return_timer = 0

    def _go_resting(self, agent: Agent) -> None:
        """Move a robot to rest and update its trip memory and strategy weights."""
        self._clear_food_target(agent)
        agent.state = "resting"
        
        if agent.trip_active_steps > 0:
            subj_gamma_r = agent.trip_collisions / agent.trip_active_steps
            subj_gamma_f = agent.trip_food_encounters / agent.trip_active_steps
            
            agent.memory_gamma_r.append(subj_gamma_r)
            agent.memory_gamma_f.append(subj_gamma_f)
            
            if len(agent.memory_gamma_r) > 5:  
                agent.memory_gamma_r.pop(0)
                agent.memory_gamma_f.pop(0)
                
        agent.trip_collisions = 0
        agent.trip_food_encounters = 0
        agent.trip_active_steps = 0
        
        x = agent.trip_delta
        
        if x >= 0:
            subjective_utility = x ** self.alpha
        else:
            subjective_utility = -self.lambda_loss * ((-x) ** self.alpha)
            
        energy_factor = (agent.energy - self.food_reward) / max(self.food_reward, 1.0)
        max_expected_utility = self.food_reward ** self.alpha
        utility_factor = subjective_utility / max(max_expected_utility, 1.0) 

        learning_reward = (utility_factor * 5.0) + (energy_factor * 2.0)
        
        idx = agent.current_strategy_idx
        
        new_propensity = (1.0 - self.recency) * agent.propensities[idx] + learning_reward
        agent.propensities[idx] = max(0.1, new_propensity)
        
        total_propensity = sum(agent.propensities)
        probabilities = [p / total_propensity for p in agent.propensities]
        
        rand_val = self.rng.random()
        cumulative = 0.0
        for i, prob in enumerate(probabilities):
            cumulative += prob
            if rand_val <= cumulative:
                agent.current_strategy_idx = i
                break
                
        agent.timer = self._current_rest_steps(agent)
        
        agent.search_credit = 0
        agent.return_state = ""
        agent.return_timer = 0
        agent.trip_delta = 0.0

    def _current_search_steps(self, agent: Agent) -> int:
        """Return the search timer for an agent's selected ratio strategy.

        Input:
            One Agent with a current strategy index.

        Output:
            The search duration in model steps for that strategy.
        """
        return agent.search_strategy_steps[agent.current_strategy_idx]

    def _current_rest_steps(self, agent: Agent) -> int:
        """Return the rest timer for an agent's selected ratio strategy.

        Input:
            One Agent with a current strategy index.

        Output:
            The rest duration in model steps for that strategy.
        """
        return agent.rest_strategy_steps[agent.current_strategy_idx]

    def _current_ratio(self, agent: Agent) -> float:
        """Return the selected search/rest ratio for one agent.

        Input:
            One Agent with a current strategy index.

        Output:
            The selected search_time/rest_time ratio.
        """
        return agent.strategy_ratios[agent.current_strategy_idx]

    def _ratio_stats(self) -> dict[str, float]:
        """Summarize selected and propensity-weighted ratios across robots.

        Input:
            The model's current list of agents.

        Output:
            Mean selected ratio, mean weighted ratio, and mean selected search
            and rest durations in seconds.
        """
        selected_ratios = [self._current_ratio(agent) for agent in self.agents]
        selected_search = [self._current_search_steps(agent) * self.world.dt for agent in self.agents]
        selected_rest = [self._current_rest_steps(agent) * self.world.dt for agent in self.agents]
        weighted_ratios = []
        for agent in self.agents:
            total = sum(agent.propensities)
            if total <= 0.0:
                weighted_ratios.append(self._current_ratio(agent))
            else:
                weighted_ratios.append(
                    sum(ratio * propensity for ratio, propensity in zip(agent.strategy_ratios, agent.propensities)) / total
                )
        return {
            "mean_selected_search_rest_ratio": sum(selected_ratios) / len(selected_ratios),
            "mean_weighted_search_rest_ratio": sum(weighted_ratios) / len(weighted_ratios),
            "mean_selected_search_time_s": sum(selected_search) / len(selected_search),
            "mean_selected_rest_time_s": sum(selected_rest) / len(selected_rest),
        }

    def counts(self) -> dict[str, float]:
        """Count how many robots are in each PFSM state."""
        names = ["searching", "grabbing", "deposit", "homing", "resting", "avoidance"]
        out = {name: 0.0 for name in names}
        for agent in self.agents:
            out[agent.state] += 1.0
        return out

    def _update_display_position(self, agent: Agent) -> None:
        """Move one animation marker without changing PFSM transition rates."""
        if agent.state == "resting":
            agent.x *= 0.9
            agent.y *= 0.9
            return

        random_turn = self.rng.uniform(-0.25, 0.25)
        step = self.world.speed * self.world.dt

        if agent.state == "grabbing":
            self._move_toward_food(agent, step)
            return

        agent.heading += random_turn
        if agent.state == "searching":
            self._steer_toward_best_sector(agent)
        if agent.state in {"deposit", "homing"} or agent.return_state in {"deposit", "homing"}:
            target = math.atan2(-agent.y, -agent.x)
            agent.heading = 0.85 * agent.heading + 0.15 * target
        agent.x += step * math.cos(agent.heading)
        agent.y += step * math.sin(agent.heading)
        radius = math.hypot(agent.x, agent.y)
        if radius > self.world.router:
            agent.heading += math.pi
            scale = self.world.router / max(radius, 1e-9)
            agent.x *= scale
            agent.y *= scale

    def _cool_current_sector(self, agent: Agent) -> None:
        """Slightly reduce the current sector score during unrewarded search.

        Input:
            One searching Agent with sector scores.

        Output:
            The agent's score for its current sector is updated in place.
        """
        sector = get_sector(agent.x, agent.y)
        agent.sector_scores[sector] = max(0.0, agent.sector_scores[sector] - self.sector_decay)

    def _reward_current_sector(self, agent: Agent) -> None:
        """Increase the current sector score after a successful food grab.

        Input:
            One Agent that has just completed grabbing food.

        Output:
            The agent's score for its current sector is updated in place.
        """
        sector = get_sector(agent.x, agent.y)
        agent.sector_scores[sector] += self.sector_reward

    def _steer_toward_best_sector(self, agent: Agent) -> None:
        """Bias a searching robot's heading toward its best food sector.

        Input:
            One searching Agent with learned sector scores.

        Output:
            The agent's heading is updated in place. PFSM transition
            probabilities are unchanged.
        """
        max_score = max(agent.sector_scores)
        if max_score <= 0.0:
            return
        best_sector = agent.sector_scores.index(max_score)
        target_angle = best_sector * (math.pi / 2.0) + (math.pi / 4.0)
        agent.heading = self._blend_angles(agent.heading, target_angle, self.sector_pull)

    @staticmethod
    def _blend_angles(current: float, target: float, weight: float) -> float:
        """Interpolate two angles along the shortest circular direction.

        Inputs:
            current and target are headings in radians. Weight is the fraction
            of the angular gap to move toward the target.

        Output:
            A blended heading in radians.
        """
        gap = math.atan2(math.sin(target - current), math.cos(target - current))
        return current + max(0.0, min(1.0, weight)) * gap

    def _move_toward_food(self, agent: Agent, step: float) -> None:
        """Move one display marker directly toward its assigned food target."""
        if agent.target_x is None or agent.target_y is None:
            self._assign_food_target(agent)
        if agent.target_x is None or agent.target_y is None:
            return

        dx = agent.target_x - agent.x
        dy = agent.target_y - agent.y
        distance = math.hypot(dx, dy)
        if distance <= 1e-12:
            return

        agent.heading = math.atan2(dy, dx)
        travel = min(step, distance)
        agent.x += travel * math.cos(agent.heading)
        agent.y += travel * math.sin(agent.heading)

    def _assign_food_target(self, agent: Agent) -> None:
        """Assign the nearest displayed food marker to a grabbing robot."""
        self._sync_food_positions()
        if not self.food_positions:
            return
        target = min(self.food_positions, key=lambda point: math.hypot(point[0] - agent.x, point[1] - agent.y))
        agent.target_x, agent.target_y = target

    def _clear_food_target(self, agent: Agent) -> None:
        """Clear a robot's displayed food target."""
        agent.target_x = None
        agent.target_y = None

    def _consume_food_target(self, agent: Agent) -> None:
        """Remove a displayed food marker when a robot finishes grabbing it."""
        if agent.target_x is not None and agent.target_y is not None:
            target = (agent.target_x, agent.target_y)
            shared = any(
                other is not agent and (other.target_x, other.target_y) == target
                for other in self.agents
            )
            if not shared and target in self.food_positions:
                self.food_positions.remove(target)
        self._clear_food_target(agent)

    def _sync_food_positions(self) -> None:
        """Keep displayed food markers consistent with the scalar food count."""
        desired = int(math.ceil(max(0.0, self.food) - 1e-12))
        while len(self.food_positions) < desired:
            radius = math.sqrt(
                self.food_rng.uniform(self.world.rinner**2, self.world.router**2)
            )
            angle = self.food_rng.uniform(0.0, 2.0 * math.pi)
            self.food_positions.append((radius * math.cos(angle), radius * math.sin(angle)))

        targeted = {
            (agent.target_x, agent.target_y)
            for agent in self.agents
            if agent.target_x is not None and agent.target_y is not None
        }
        index = len(self.food_positions) - 1
        while len(self.food_positions) > desired and index >= 0:
            if self.food_positions[index] not in targeted:
                self.food_positions.pop(index)
            index -= 1
