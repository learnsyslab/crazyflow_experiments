"""Check the shared position errors against hand-computed values."""

from __future__ import annotations

from tests.conftest import module_requires_env

module_requires_env("sim2real")

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from crazyflow_experiments.control.trajectories import Ellipse  # noqa: E402
from crazyflow_experiments.utils.errors import (  # noqa: E402
    run_mask,
    sim2real_rmse,
    tracking_error,
    tracking_rmse,
)

TRAJECTORY = Ellipse(jnp.array([2.0, 1.0, 0.0]), 5.0, 15.0, jnp.array([0.0, 0.0, 1.0]))
T = np.linspace(0.0, 15.0, 301)


def test_tracking_error_is_the_distance_to_the_reference():
    """A flight shifted off the reference by a constant vector has that vector's norm as error."""
    reference = np.asarray(TRAJECTORY.pos(T))
    offset = np.array([0.03, -0.04, 0.0])
    error = tracking_error(T, reference + offset, TRAJECTORY)
    assert error.shape == T.shape
    np.testing.assert_allclose(error, 0.05, atol=1e-6)
    assert tracking_rmse(T, reference + offset, TRAJECTORY) == pytest.approx(0.05, abs=1e-6)
    assert tracking_rmse(T, reference, TRAJECTORY) == pytest.approx(0.0, abs=1e-6)


def test_tracking_error_rejects_mismatched_shapes():
    """Positions must come one per time stamp."""
    with pytest.raises(ValueError, match="shape"):
        tracking_error(T, np.zeros((3, T.size)), TRAJECTORY)


def test_sim2real_rmse_samples_the_simulation_at_the_real_timestamps():
    """A simulated flight on another grid is compared at the samples of the real one."""
    t_sim = np.linspace(0.0, 10.0, 501)
    t_real = np.linspace(1.0, 9.0, 333)  # Different rate and offset samples
    pos_sim = np.stack([t_sim, np.zeros_like(t_sim), np.ones_like(t_sim)], axis=-1)
    pos_real = np.stack([t_real + 0.02, np.zeros_like(t_real), np.ones_like(t_real)], axis=-1)
    assert sim2real_rmse(t_sim, pos_sim, t_real, pos_real) == pytest.approx(0.02, abs=1e-9)
    with pytest.raises(ValueError, match="covers"):
        sim2real_rmse(t_sim, pos_sim, t_real + 2.0, pos_real)  # The real flight outlasts the sim


def test_run_mask_selects_all_or_one_run():
    """The mask spans the tracked runs, or one of them, and rejects runs outside the trajectory."""
    t = np.array([-1.0, 0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 16.0])
    np.testing.assert_array_equal(run_mask(t, 5.0, 3), [0, 1, 1, 1, 1, 1, 1, 1, 0])
    np.testing.assert_array_equal(run_mask(t, 5.0, 3, run=1), [0, 0, 0, 1, 1, 1, 0, 0, 0])
    with pytest.raises(ValueError, match="out of range"):
        run_mask(t, 5.0, 3, run=3)
