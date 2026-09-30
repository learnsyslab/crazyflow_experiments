"""Run the sim2real Crazyflow simulator with every controller it supports."""

from __future__ import annotations

from tests.conftest import module_requires_env

module_requires_env("sim2real")

from typing import TYPE_CHECKING  # noqa: E402

import pytest  # noqa: E402

from crazyflow_experiments.sim2real import sim_crazyflow  # noqa: E402
from tests.sim2real_fixtures import fly, single_lap_config, tracking_error  # noqa: E402

if TYPE_CHECKING:
    from pathlib import Path

# The policies and the MPC fly from their own experiments, see test_figure8 and test_learn2fly
MAX_TRACKING_ERROR = {"mellinger": 0.15, "state": 0.15}
CONTROLLERS = tuple(MAX_TRACKING_ERROR)


@pytest.mark.parametrize("controller", CONTROLLERS)
def test_sim_crazyflow_tracks_the_trajectory(
    controller: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Every controller flies the configured trajectory and follows the reference."""
    config = single_lap_config(controller)
    error = tracking_error(config, fly(config, sim_crazyflow, tmp_path, monkeypatch))
    assert error.max() < MAX_TRACKING_ERROR[controller], (
        f"{controller} peaked at {error.max() * 100:.1f}cm"
    )
