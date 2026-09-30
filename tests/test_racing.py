"""Test the racing controller."""

from __future__ import annotations

from tests.conftest import module_requires_env

module_requires_env("racing")

from crazyflow_experiments.racing.sim import simulate  # noqa: E402


def test_racing():
    """Test that the racing controller flies the drone through all gates."""
    ep_times = simulate(render=False)
    assert ep_times and all(t is not None for t in ep_times), f"Track not finished: {ep_times}"
