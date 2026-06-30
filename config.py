"""Load and expose the YAML configuration.

Inputs:
    A YAML file path, normally default_config.yaml.

Outputs:
    A Config object with typed helpers for commonly used model values.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml


@dataclass(frozen=True)
class Config:
    """Store raw YAML values and expose small typed accessors."""

    raw: dict[str, Any]
    path: Path = Path("default_config.yaml")

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        """Read a YAML file and return a Config object."""
        p = Path(path)
        with open(p, "r", encoding="utf-8") as f:
            return cls(yaml.safe_load(f), p)

    def get(self, *keys: str, default: Any = None) -> Any:
        """Read a nested configuration value with an optional default."""
        node: Any = self.raw
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    @property
    def dt(self) -> float:
        """Return the configured time step in seconds."""
        return float(self.get("paper", "dt_s"))

    @property
    def duration_s(self) -> float:
        """Return the configured simulation duration in seconds."""
        return float(self.get("paper", "duration_s"))

    @property
    def n_robots(self) -> int:
        """Return the configured number of robots in the swarm."""
        return int(self.get("paper", "n_robots"))
