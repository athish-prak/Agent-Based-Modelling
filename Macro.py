"""Run the aggregate timer-queue version of the foraging model.

Inputs:
    A Config object, optional rest time, optional collision-rate scale, duration,
    and output stride.

Outputs:
    A MacroResult containing a pandas trace of state counts, food, energy, and
    transition probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from config import Config
from map import PaperMap


@dataclass
class MacroResult:
    """Bundle a macro-model trace with the rest time used to produce it."""

    trace: pd.DataFrame
    rest_time_s: float


class MacroModel:
    """Fast macroscopic PFSM with transparent timer queues.

    This keeps the paper's important timed structure (Ts, Tg, Td, Th, Ta, Tr)
    without the huge nested sub-PFSM bookkeeping from the derivation.  The micro
    model is the exact stochastic counterpart used to check the state traces.
    """

    def __init__(self, cfg: Config, rest_time_s: float | None = None, gamma_r_scale: float = 1.0):
        """Create the macro model from configuration and optional rest setting."""
        self.cfg = cfg
        self.world = PaperMap.from_config(cfg)
        self.rest_time_s = float(rest_time_s if rest_time_s is not None else cfg.get("behaviour", "default_rest_time_s"))
        self.ts = self.world.steps(float(cfg.get("behaviour", "search_time_s")))
        self.ta = self.world.steps(float(cfg.get("behaviour", "avoidance_time_s")))
        self.tr = self.world.steps(self.rest_time_s)
        self.tg = self.world.steps(self.world.tau_grab)
        self.td = self.world.steps(self.world.tau_deposit)
        self.th = self.world.steps(self.world.tau_home)
        self.initial_food = float(cfg.get("food", "initial_count"))
        self.resting_cost = float(cfg.get("energy", "resting_cost"))
        self.active_cost = float(cfg.get("energy", "active_cost"))
        self.food_reward = float(cfg.get("energy", "food_reward"))
        self.gamma_r_scale = float(gamma_r_scale)

    def run(self, seconds: float | None = None, stride: int = 1) -> MacroResult:
        """Run the macro model and return a sampled trace."""
        seconds = float(seconds if seconds is not None else self.cfg.duration_s)
        steps = self.world.steps(seconds)
        stride = max(1, int(stride))

        search = np.zeros(self.ts + 1)
        search[self.ts] = self.world.n_robots
        avoid = np.zeros((self.ts + 1, self.ta + 1))
        grab = np.zeros(self.tg + 1)
        deposit = np.zeros(self.td + 1)
        homing = np.zeros(self.th + 1)
        rest = np.zeros(self.tr + 1 if self.tr > 0 else 1)

        food = self.initial_food
        energy = 0.0
        rows: list[dict[str, float]] = []

        for step in range(steps):
            resting_now = float(rest.sum())
            active_now = self.world.n_robots - resting_now
            searching_now = float(search.sum())
            grabbing_now = float(grab.sum())
            deposit_now = float(deposit.sum())
            homing_now = float(homing.sum())
            avoid_now = float(avoid.sum())

            gamma_f = self.world.find_probability(food)
            gamma_r = max(0.0, min(1.0, self.world.collision_probability(active_now) * self.gamma_r_scale))
            competing = searching_now + grabbing_now + avoid_now
            gamma_l = self.world.loss_probability(food, competing, self.tg)
            gamma_f, gamma_r = _renormalize_pair(gamma_f, gamma_r)
            gamma_l, gamma_r = _renormalize_pair(gamma_l, gamma_r)

            ns = np.zeros_like(search)
            na = np.zeros_like(avoid)
            ng = np.zeros_like(grab)
            nd = np.zeros_like(deposit)
            nh = np.zeros_like(homing)
            nr = np.zeros_like(rest)
            entered_deposit = 0.0
            completed_deposit = 0.0

            def add_rest(w: float) -> None:
                """Move completed robots into rest or directly back to search.

                Input:
                    w is the aggregate number of robots completing a task.

                Output:
                    The next-step rest or search queue is updated in place.
                """
                if w <= 0:
                    return
                if self.tr <= 0:
                    ns[self.ts] += w
                else:
                    nr[self.tr] += w

            # Resting is a fixed delay, not a memoryless 1/Tr drain.
            if self.tr > 0:
                ns[self.ts] += rest[1]
                if self.tr >= 2:
                    nr[1:self.tr] += rest[2:self.tr + 1]
            else:
                ns[self.ts] += rest[0]

            # Searching with remaining search credit.
            w = search[1:]
            to_avoid = gamma_r * w
            to_grab = gamma_f * w
            stay = w - to_avoid - to_grab
            nh[self.th] += stay[0] + to_avoid[0] + to_grab[0]
            if self.ts >= 2:
                ns[1:self.ts] += stay[1:]
                na[1:self.ts, self.ta] += to_avoid[1:]
                # Fast aggregate: grabbing is represented by its own Tg queue.
                # The microscopic model carries exact remaining search credit.
                ng[self.tg] += float(to_grab[1:].sum())

            # Avoidance from searching/grabbing. Search budget continues to run;
            # timeout has priority over completion of avoidance.
            for a in range(1, self.ta + 1):
                col = avoid[1:, a]
                nh[self.th] += col[0]
                if self.ts >= 2:
                    if a == 1:
                        ns[1:self.ts] += col[1:]
                    else:
                        na[1:self.ts, a - 1] += col[1:]

            # Grabbing: loss returns to searching; collision loses the target and
            # enters the same avoidance/search-credit queue.
            lost = gamma_l * grab
            bumped = gamma_r * grab
            keep = grab - lost - bumped
            ns[self.ts] += float(lost.sum())
            # Approximation: bumped grabbers avoid, then search again with a fresh
            # search episode. The micro model carries the exact remaining credit.
            na[self.ts, self.ta] += float(bumped.sum())
            entered_deposit = float(keep[1])
            nd[self.td] += entered_deposit
            if self.tg >= 2:
                ng[1:self.tg] += keep[2:self.tg + 1]

            # Deposit and homing use fixed timers; collisions are already included
            # in gamma_r/interference through the active count, but do not change
            # the task completion queue in this fast macro approximation.
            completed_deposit = float(deposit[1])
            add_rest(completed_deposit)
            if self.td >= 2:
                nd[1:self.td] += deposit[2:self.td + 1]

            add_rest(float(homing[1]))
            if self.th >= 2:
                nh[1:self.th] += homing[2:self.th + 1]

            search, avoid, grab, deposit, homing, rest = ns, na, ng, nd, nh, nr
            food = max(0.0, food + self.world.p_new_per_step - entered_deposit)
            energy += self.food_reward * completed_deposit - self.world.dt * (
                self.resting_cost * resting_now + self.active_cost * active_now
            )

            if step % stride == 0 or step == steps - 1:
                rows.append(
                    {
                        "time_s": (step + 1) * self.world.dt,
                        "rest_time_s": self.rest_time_s,
                        "energy": energy,
                        "food_items": food,
                        "searching": float(search.sum()),
                        "grabbing": float(grab.sum()),
                        "deposit": float(deposit.sum()),
                        "homing": float(homing.sum()),
                        "resting": float(rest.sum()),
                        "avoidance": float(avoid.sum()),
                        "gamma_f": gamma_f,
                        "gamma_r": gamma_r,
                        "gamma_l": gamma_l,
                        "entered_deposit": entered_deposit,
                        "completed_deposit": completed_deposit,
                    }
                )

        return MacroResult(pd.DataFrame(rows), self.rest_time_s)


def _renormalize_pair(a: float, b: float, limit: float = 0.98) -> tuple[float, float]:
    """Keep two competing probabilities inside a shared upper limit."""
    total = a + b
    if total <= limit:
        return a, b
    scale = limit / total
    return a * scale, b * scale
