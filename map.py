"""Convert paper parameters into arena geometry and transition probabilities.

Inputs:
    A Config object containing map, robot, food, and timing settings.

Outputs:
    A PaperMap object with distances, durations, per-step food growth, and the
    probability helpers used by the macro and micro models.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from config import Config


@dataclass(frozen=True)
class PaperMap:
    """Arena geometry and transition-probability estimates from Section 4."""

    n_robots: int
    dt: float
    speed: float
    router: float
    rinner: float
    home_radius: float
    camera_range: float
    camera_angle: float
    bumper_range: float
    bumper_angle: float
    robot_radius: float
    turn_food: float
    turn_home: float
    load_time: float
    p_new_per_second: float

    @classmethod
    def from_config(cls, cfg: Config) -> "PaperMap":
        """Build arena and robot geometry from the project configuration."""
        deg = math.pi / 180.0
        return cls(
            n_robots=cfg.n_robots,
            dt=cfg.dt,
            speed=float(cfg.get("robot", "speed_m_s")),
            router=float(cfg.get("map", "router_m")),
            rinner=float(cfg.get("map", "rinner_m")),
            home_radius=float(cfg.get("map", "home_radius_m")),
            camera_range=float(cfg.get("robot", "camera_range_m")),
            camera_angle=float(cfg.get("robot", "camera_angle_deg")) * deg,
            bumper_range=float(cfg.get("robot", "bumper_range_m")),
            bumper_angle=float(cfg.get("robot", "bumper_angle_deg")) * deg,
            robot_radius=float(cfg.get("robot", "robot_radius_m")),
            turn_food=float(cfg.get("robot", "turn_to_food_deg_s")) * deg,
            turn_home=float(cfg.get("robot", "turn_home_deg_s")) * deg,
            load_time=float(cfg.get("robot", "load_time_s")),
            p_new_per_second=float(cfg.get("food", "growth_rate_s")),
        )

    @property
    def ring_area_without_pi(self) -> float:
        """Return the annulus area term before multiplying by pi."""
        # The paper repeatedly writes denominators as pi * (Router^2 - Rinner^2).
        return self.router**2 - self.rinner**2

    @property
    def p_new_per_step(self) -> float:
        """Return the expected food growth per discrete model step."""
        # Table 1 gives p_new with the continuous-time parameters; convert to one discrete step.
        return self.p_new_per_second * self.dt

    @property
    def tau_grab(self) -> float:
        """Return the expected seconds needed to turn, reach, and load food."""
        # Eq. 61.
        return self.camera_angle / (2.0 * self.turn_food) + self.camera_range / self.speed + self.load_time

    @property
    def tau_deposit(self) -> float:
        """Return the expected seconds needed to return and deposit food."""
        # Eq. 62-64.
        distance_home = 2.0 * (self.router**3 - self.rinner**3) / (3.0 * (self.router**2 - self.rinner**2)) - self.home_radius
        turn_time = (math.pi / 2.0) / self.turn_home
        return distance_home / self.speed + turn_time

    @property
    def tau_home(self) -> float:
        """Return the expected seconds needed to go home without food."""
        # Paper: Th = Td.
        return self.tau_deposit

    def steps(self, seconds: float) -> int:
        """Convert seconds to discrete model steps."""
        return max(0, int(round(seconds / self.dt)))

    def find_probability(self, food_count: float) -> float:
        """Eq. 49-50: probability that a searching robot sees >= 1 item."""
        if food_count <= 0.0:
            return 0.0
        pf = self.camera_angle * self.camera_range * self.speed * self.dt / (math.pi * self.ring_area_without_pi)
        return _clip01(1.0 - (1.0 - pf) ** food_count)

    def collision_probability(self, active_robots: float) -> float:
        """Eq. 52-54 with Pa = 0.5 from the text after Eq. 54."""
        if active_robots <= 1.0:
            return 0.0
        pin = 2.0 * self.speed * self.dt * self.bumper_angle * (self.bumper_range + self.robot_radius) / (
            math.pi * self.ring_area_without_pi
        )
        pa = 0.5
        return _clip01(1.0 - (1.0 - pin * pa) ** (active_robots - 1.0))

    def loss_probability(self, food_count: float, competing_robots: float, tg_steps: int) -> float:
        """Eq. 55-59: probability that a grabbing robot loses its target."""
        if food_count <= 0.0 or competing_robots <= 1.0 or tg_steps <= 0:
            return 0.0
        n_fa = (self.camera_range**2 / self.ring_area_without_pi) * competing_robots
        m_fa = (self.camera_angle * self.camera_range**2 / (2.0 * math.pi * self.ring_area_without_pi)) * food_count
        if n_fa <= 1.0 or m_fa <= 1e-12:
            return 0.0
        p_g = self.camera_angle / (2.0 * math.pi)
        # Keep pg / Mfa inside [0, 1] so the probability expression remains physical
        # when the expected number of visible food items is very small.
        q = min(1.0, p_g / m_fa)
        value = 2.0 * (1.0 - 1.0 / n_fa) * (1.0 - (1.0 - q) ** (n_fa - 1.0)) / tg_steps
        return _clip01(value)


def _clip01(x: float) -> float:
    """Clamp a numeric value into the probability range [0, 1]."""
    return max(0.0, min(1.0, float(x)))
