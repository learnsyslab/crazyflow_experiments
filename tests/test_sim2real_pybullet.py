"""Run the sim2real PyBullet simulator with every attitude loop variant it supports."""

from __future__ import annotations

from tests.conftest import module_requires_env

module_requires_env("pybullet")

from typing import TYPE_CHECKING  # noqa: E402

import pytest  # noqa: E402

from crazyflow_experiments.sim2real import sim_pybullet  # noqa: E402
from crazyflow_experiments.sim2real.sim_pybullet import VARIANTS  # noqa: E402
from tests.sim2real_fixtures import fly, single_lap_config, tracking_error  # noqa: E402

if TYPE_CHECKING:
    from pathlib import Path

MAX_TRACKING_ERROR = 0.15  # m


@pytest.mark.parametrize("physics", VARIANTS)
def test_sim_pybullet_tracks_the_trajectory(
    physics: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Every attitude loop variant flies the trajectory and follows the reference."""
    config = single_lap_config("mellinger")
    config["sim"]["physics"] = physics
    config["drone"]["model"] = "cf2x_L250"  # gym-pybullet-drones models one drone
    error = tracking_error(config, fly(config, sim_pybullet, tmp_path, monkeypatch))
    assert error.max() < MAX_TRACKING_ERROR, f"{physics} peaked at {error.max() * 100:.1f}cm"
