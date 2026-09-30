"""Run the sim2real l2f simulator."""

from __future__ import annotations

from tests.conftest import module_requires_env

module_requires_env("l2f")

from typing import TYPE_CHECKING  # noqa: E402

from crazyflow_experiments.sim2real import sim_l2f  # noqa: E402
from tests.sim2real_fixtures import fly, single_lap_config, tracking_error  # noqa: E402

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

MAX_TRACKING_ERROR = 0.15  # m


def test_sim_l2f_tracks_the_trajectory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """l2f flies the figure-8 and follows the reference."""
    config = single_lap_config("mellinger")
    config["drone"]["model"] = "cf2x_L250"
    config["experiment"].update(trajectory="figure8", plane="xz", scaling=[2.0, 0.0, 1.0], time=5.5)
    error = tracking_error(config, fly(config, sim_l2f, tmp_path, monkeypatch))
    assert error.max() < MAX_TRACKING_ERROR, f"l2f peaked at {error.max() * 100:.1f}cm"
