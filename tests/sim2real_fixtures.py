"""Helpers for running the sim2real simulator tests."""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

import numpy as np
import toml

from crazyflow_experiments.control.config import TrajectoryConfig
from tests.conftest import PACKAGE_ROOT

if TYPE_CHECKING:
    from pathlib import Path
    from types import ModuleType

    import pytest

CONFIG_PATH = PACKAGE_ROOT / "sim2real" / "config.toml"


def single_lap_config(controller: str) -> dict:
    """Load the sim2real config, shortened to a single lap and without rendering."""
    config = toml.load(CONFIG_PATH)
    config["controller"]["type"] = controller
    config["experiment"]["n_runs"] = 1
    config["sim"]["n_rollouts"] = 1
    config["sim"]["render"] = False
    config["experiment"]["trajectory"] = "figure8"
    config["experiment"]["plane"] = "xz"
    config["experiment"]["scaling"] = [2.0, 0.0, 1.0]
    config["experiment"]["time"] = 10.0
    return config


def fly(
    config: dict, simulator: ModuleType, recordings_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> list[dict]:
    """Run a simulator and return the recorded observations."""
    monkeypatch.setattr(simulator, "RECORDINGS_DIR", recordings_dir)
    simulator.main(config)
    recordings = list(recordings_dir.glob("*/*.pkl"))  # One folder per drone model
    assert len(recordings) == 1, f"expected one recording, got {recordings}"
    with open(recordings[0], "rb") as f:
        return pickle.load(f)


def tracking_error(config: dict, data: list[dict]) -> np.ndarray:
    """Return the distance to the reference at every recorded step."""
    trajectory = TrajectoryConfig.from_config(config).build()
    t = np.array([obs["t"] for obs in data])  # Trajectory time, zero where the trajectory starts
    reference = np.asarray(trajectory.pos(t))
    return np.linalg.norm(np.array([obs["pos"] for obs in data]) - reference, axis=1)
