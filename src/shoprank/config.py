"""Load `config.yaml`; paths are resolved relative to the config file."""

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


class Config(dict):
    """The parsed YAML plus helpers for paths and dates."""

    def __init__(self, data: dict, root: Path):
        super().__init__(data)
        self.root = root

    def path(self, key: str) -> Path:
        """Absolute directory for `paths.<key>`."""
        return (self.root / self["paths"][key]).resolve()

    @property
    def t1(self) -> pd.Timestamp:
        return pd.Timestamp(self["split"]["t1"])

    @property
    def t2(self) -> pd.Timestamp:
        return pd.Timestamp(self["split"]["t2"])

    @property
    def seed(self) -> int:
        return int(self["random_state"])


def load_config(path: str | Path = "config.yaml") -> Config:
    path = Path(path).resolve()
    with open(path) as f:
        data: dict[str, Any] = yaml.safe_load(f)
    return Config(data, path.parent)
