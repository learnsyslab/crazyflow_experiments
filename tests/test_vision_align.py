"""Check that a rotated splat keeps the colour field it had before."""

import os

os.environ["SCIPY_ARRAY_API"] = "1"  # Must be set before anything imports scipy

import numpy as np  # noqa: E402

from tests.conftest import module_requires_env  # noqa: E402

module_requires_env("vision")

import jax.numpy as jnp  # noqa: E402
import splax  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402

from crazyflow_experiments.vision.reconstruct.align import (  # noqa: E402
    COEFFICIENTS,
    rotate_harmonics,
)

ROTATION = R.from_euler("xyz", [4.2, -2.7, 31.0], degrees=True)


def colour(sh: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """The colour splax renders along each direction."""
    coeffs, views = jnp.asarray(sh, jnp.float32), jnp.asarray(directions, jnp.float32)
    return np.asarray(splax.spherical_harmonics(coeffs, views))


def test_rotation_carries_the_colour_field():
    """Every degree splax renders shows the same colour along the rotated direction."""
    rng = np.random.default_rng(0)
    views = rng.normal(size=(256, 3))
    views /= np.linalg.norm(views, axis=1, keepdims=True)
    for n_coeffs in COEFFICIENTS:
        sh = rng.normal(scale=0.3, size=(256, n_coeffs, 3))
        carried = colour(rotate_harmonics(sh, ROTATION), ROTATION.apply(views))
        assert np.abs(carried - colour(sh, views)).max() < 1e-5, f"{n_coeffs} coefficients"
    # Leaving the harmonics behind must disagree, or the check above would pass on a no-op
    assert np.abs(colour(sh, ROTATION.apply(views)) - colour(sh, views)).max() > 1e-2
