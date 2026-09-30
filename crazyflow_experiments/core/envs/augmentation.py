"""Image augmentations for robust training."""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jp
import numpy as np
from jax import Array
from jax.image import scale_and_translate

NOMINAL_DEPTH = 3.0  # Scene distance in m the translational blur term is evaluated at
BLUR_TAPS = 3  # Samples along the motion direction, approximating the exposure integral


class CameraNoise(NamedTuple):
    """Per-step sensor effects, all applied independently to every world."""

    sharpen: float = 1.4
    pixel_noise: float = 0.03
    ramp_std: float = 0.085
    readout_time: float = 1 / 60


def focal_length(height: int, fovy: float) -> float:
    """Pinhole focal length in pixels for an image height and vertical field of view in degrees."""
    return height / (2.0 * np.tan(np.deg2rad(fovy) / 2.0))


def build_zoom(
    render_resolution: tuple[int, int], resolution: tuple[int, int], render_fovy: float
) -> Callable[[Array, Array], Array]:
    """Build the resampling that cuts the view of a narrower camera out of a wider render."""
    width, height = resolution
    render_width, render_height = render_resolution
    assert width * render_height == height * render_width, "the render has a different aspect ratio"
    span = render_height / height / np.tan(np.deg2rad(render_fovy) / 2.0)
    centre = jp.array([height, width]) / 2.0
    extent = jp.array([render_height, render_width]) / 2.0

    def zoom(image: Array, fovy: Array) -> Array:
        """Resample one render to the frame a camera of this vertical field of view sees."""
        box = span * jp.tan(jp.deg2rad(fovy) / 2.0)
        scale, translation = jp.full(2, 1.0 / box), centre - extent / box
        return scale_and_translate(image, (height, width), (0, 1), scale, translation, "area")

    return jax.vmap(zoom)


def build_augment(
    resolution: tuple[int, int], fovy: float, noise: CameraNoise
) -> Callable[[Array, Array, Array, Array, Array], Array]:
    """Build the batched sensor effect pipeline for a fixed image size.

    Args:
        resolution: Image resolution as (width, height).
        fovy: Vertical field of view in degrees, which sets the focal length.
        noise: Per-step sensor effect magnitudes.
    """
    width, height = resolution
    focal = focal_length(height, fovy)
    rows, cols = jp.arange(height), jp.arange(width)
    ramp = rows[:, None] / (height - 1) - 0.5  # Signed distance from the center row
    # Squared radius from the image centre, normalized so the corners fall off fully
    vignette_falloff = (
        (rows[:, None] / (height - 1) - 0.5) ** 2 + (cols[None, :] / (width - 1) - 0.5) ** 2
    ) / 0.5

    def sharpen(image: Array, amount: Array) -> Array:
        """Lift the fine detail a denser reconstruction would have resolved."""
        blurred = image
        for axis in (0, 1):
            blurred = (jp.roll(blurred, 1, axis) + 2.0 * blurred + jp.roll(blurred, -1, axis)) / 4.0
        return image + amount * (image - blurred)

    def shear(image: Array, shift: Array) -> Array:
        """Offset each row horizontally, the shape a rolling shutter gives a yawing camera."""
        offsets = shift * (rows / (height - 1) - 0.5)
        index = jp.clip(jp.round(cols[None, :] + offsets[:, None]), 0, width - 1).astype(int)
        return jp.take_along_axis(image, index, axis=1)

    def smear(image: Array, ang_vel: Array, vel: Array, exposure: Array) -> Array:
        """Average samples along the direction the scene moves in during the exposure."""
        rate_x = focal * (-ang_vel[2] + vel[1] / NOMINAL_DEPTH)
        rate_y = focal * (ang_vel[1] + vel[2] / NOMINAL_DEPTH)
        shift = jp.stack([rate_x, rate_y]) * exposure
        taps = jp.linspace(-0.5, 0.5, BLUR_TAPS)
        row = jp.clip(jp.round(rows[:, None] + taps * shift[1]), 0, height - 1).astype(int)
        col = jp.clip(jp.round(cols[:, None] + taps * shift[0]), 0, width - 1).astype(int)
        return jp.mean(image[row[:, None, :], col[None, :, :]], axis=-1)

    def augment(image: Array, photometry: Array, ang_vel: Array, vel: Array, key: Array) -> Array:
        """Apply every sensor effect to one world's image."""
        noise_key, ramp_key, sharp_key = jax.random.split(key, 3)
        brightness, contrast, gamma, vignette, exposure = photometry
        image = sharpen(image, noise.sharpen * jax.random.uniform(sharp_key))
        image = shear(image, focal * ang_vel[2] * noise.readout_time)
        image = smear(image, ang_vel, vel, exposure)
        image = image * (1.0 - vignette * vignette_falloff)
        image = jp.clip((image - 0.5) * contrast + 0.5 + brightness, 1e-4, 1.0) ** gamma
        image = image + noise.pixel_noise * jax.random.normal(noise_key, image.shape)
        image = image + noise.ramp_std * jax.random.normal(ramp_key) * ramp
        return jp.clip(image, 0.0, 1.0)

    return jax.vmap(augment)
