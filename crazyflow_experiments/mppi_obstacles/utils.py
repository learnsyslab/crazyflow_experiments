"""Utils for the MPPI obstacle avoidance experiment."""

import logging
from pathlib import Path

import toml

logger = logging.getLogger(__name__)


def experiment_name(config: dict, controller: str, source: str = "") -> str:
    """Return the name identifying an experiment run."""
    experiment = config["experiment"]
    scaling = "-".join(str(x) for x in experiment["scaling"])
    name = (
        f"{controller}_{experiment['trajectory']}_{scaling}_{experiment['plane']}"
        f"_{experiment['time']}s_{config['drone']['model']}"
    )
    return f"{name}_{source}" if source else name


def load_config(folder: Path) -> dict:
    """Load the config of the flown experiment."""
    return toml.load(folder / "config.toml")
