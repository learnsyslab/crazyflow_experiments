"""Align the splats with the real scene.

Run with
    pixi run -e vision python crazyflow_experiments/vision/reconstruct/align.py refine \
        saves/scenes/lab/raw/100k_sh0.ply
    pixi run -e vision python crazyflow_experiments/vision/reconstruct/align.py apply \
        saves/scenes/lab/raw/20k.ply saves/scenes/lab/aligned/20k.ply
"""

from __future__ import annotations

import os

os.environ["SCIPY_ARRAY_API"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import functools
import json
import logging
import time
from pathlib import Path
from typing import Callable, NamedTuple

import fire
import jax.numpy as jnp
import numpy as np
import splax
import viser
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)

FRAME_FILE = "frame.json"
Splat = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
COEFFICIENTS = (1, 4, 9, 16)  # Allowed SH coefficient numbers
CONVENTION = "world = scale * rotation @ reconstruction + translation"
# splax's real spherical harmonics normalization
C0 = 0.5 / np.sqrt(np.pi)
C1 = 0.5 * np.sqrt(3.0 / np.pi)
C2 = (
    0.5 * np.sqrt(15.0 / np.pi),
    -0.5 * np.sqrt(15.0 / np.pi),
    0.25 * np.sqrt(5.0 / np.pi),
    -0.5 * np.sqrt(15.0 / np.pi),
    0.25 * np.sqrt(15.0 / np.pi),
)
C3 = (
    -0.25 * np.sqrt(35.0 / (2.0 * np.pi)),
    0.5 * np.sqrt(105.0 / np.pi),
    -0.25 * np.sqrt(21.0 / (2.0 * np.pi)),
    0.25 * np.sqrt(7.0 / np.pi),
    -0.25 * np.sqrt(21.0 / (2.0 * np.pi)),
    0.25 * np.sqrt(105.0 / np.pi),
    -0.25 * np.sqrt(35.0 / (2.0 * np.pi)),
)
# Sampling directions for reconstructing the SH basis
_step = np.arange(32)
_z = 1.0 - 2.0 * (_step + 0.5) / 32.0
_phi = _step * np.pi * (3.0 - np.sqrt(5.0))
DIRECTIONS = np.stack(
    [np.sqrt(1.0 - _z**2) * np.cos(_phi), np.sqrt(1.0 - _z**2) * np.sin(_phi), _z], axis=1
)
AXIS_NEAR = 1.0
AXIS_FAR = 10.0
AXIS_COLORS = ((255, 60, 60), (60, 220, 60), (80, 120, 255))  # x red, y green, z blue
LINE_SPAN = 14.0  # m the reference lines run along x, long enough to cross any room
RAMP_GROWTH = 1.15
RAMP_RESET = 0.25  # in s
SHIFT_STEP = (0.001, 0.02)
TURN_STEP = (0.0005, 0.01)  # rad per keypress
SCALE_STEP = (0.0002, 0.005)  # fraction per keypress
# viser flies its camera with w a s d q e and the arrows, so the shift keys sit clear of those
KEYS = "xcv", "jkl", "io"
START_EYE = (2.0, 0.0, 1.0)  # Camera pose every client opens at, looking back at the origin


class SimilarityTransform(NamedTuple):
    """Similarity carrying a reconstruction into the room's metric frame."""

    rotation: R
    scale: float
    translation: np.ndarray

    def as_dict(self) -> dict[str, object]:
        """The similarity as plain values, for writing beside the scene."""
        return {
            "quat": self.rotation.as_quat().tolist(),
            "scale": self.scale,
            "translation": self.translation.tolist(),
            "convention": CONVENTION,
        }


def basis(directions: np.ndarray, n_coeffs: int) -> np.ndarray:
    """Evaluate the real spherical harmonic basis splax renders with, one row per direction."""
    x, y, z = directions.T
    xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
    values = [np.full_like(x, C0), -C1 * y, C1 * z, -C1 * x]
    if n_coeffs > 4:
        values += [C2[0] * xy, C2[1] * yz, C2[2] * (2.0 * zz - xx - yy)]
        values += [C2[3] * xz, C2[4] * (xx - yy)]
    if n_coeffs > 9:
        values += [C3[0] * y * (3.0 * xx - yy), C3[1] * xy * z, C3[2] * y * (4.0 * zz - xx - yy)]
        values += [C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy), C3[4] * x * (4.0 * zz - xx - yy)]
        values += [C3[5] * z * (xx - yy), C3[6] * x * (xx - 3.0 * yy)]
    return np.stack(values[:n_coeffs], axis=1)


def rotate_harmonics(sh: np.ndarray, rotation: R) -> np.ndarray:
    """Rotate spherical harmonic coefficients along with the scene they colour."""
    assert sh.ndim == 3 and sh.shape[2] == 3, f"sh must be (N, K, 3), got {sh.shape}"
    n_coeffs = sh.shape[1]
    assert n_coeffs in COEFFICIENTS, f"coefficients must be one of {COEFFICIENTS}, got {n_coeffs}"
    before, after = basis(DIRECTIONS, n_coeffs), basis(rotation.apply(DIRECTIONS), n_coeffs)
    sh = np.array(sh)
    rotated = sh.copy()
    for band in range(1, COEFFICIENTS.index(n_coeffs) + 1):
        columns = slice(band**2, (band + 1) ** 2)
        matrix = np.linalg.lstsq(before[:, columns], after[:, columns], rcond=None)[0].T
        assert np.allclose(matrix @ matrix.T, np.eye(2 * band + 1), atol=1e-9)
        rotated[:, columns] = np.einsum("ij,njc->nic", matrix, sh[:, columns])
    return rotated


def transform(splat: Splat, similarity: SimilarityTransform) -> Splat:
    """Scale, rotate and shift a splat."""
    new_means = similarity.scale * similarity.rotation.apply(splat[0]) + similarity.translation
    new_log_scales = splat[1] + np.log(similarity.scale)
    new_quats = (similarity.rotation * R.from_quat(np.array(splat[2]), scalar_first=True)).as_quat(
        scalar_first=True
    )
    new_sh = rotate_harmonics(splat[3], similarity.rotation)
    return new_means, new_log_scales, new_quats, new_sh, splat[4]


class Ramp:
    """Step size that grows while a key repeats and falls back to its base after a pause."""

    def __init__(self, base: float, cap: float):
        """Hold the step between a resting size and the cap a held key reaches."""
        self.base, self.cap = base, cap
        self.step, self.last = base, 0.0

    def next(self) -> float:
        """Step for this keypress, larger the longer the key is held down."""
        now = time.monotonic()
        quiet = now - self.last > RAMP_RESET
        self.step = self.base if quiet else min(self.step * RAMP_GROWTH, self.cap)
        self.last = now
        return self.step


def readout_text(similarity: SimilarityTransform) -> str:
    """The similarity and the keys that change it, as markdown for the viewer panel."""
    angles = similarity.rotation.as_euler("xyz", degrees=True)
    return (
        f"**shift** `{similarity.translation[0]:+.3f} {similarity.translation[1]:+.3f} "
        f"{similarity.translation[2]:+.3f}` m\n\n"
        f"**turn** `{angles[0]:+.2f} {angles[1]:+.2f} {angles[2]:+.2f}` deg\n\n"
        f"**scale** `{similarity.scale:.5f}`\n\n"
        "---\n\n"
        "`x c v` shift along x y z\n\n"
        "`j k l` turn about x y z\n\n"
        "`i` `o` scale up and down\n\n"
        "hold **shift** to shift or turn the other way, hold a key to speed up"
    )


def draw_axes(scene: viser.SceneApi):
    """Draw the world axes at the origin, thick over the first metre and thin out to ten."""
    for axis, colour in enumerate(AXIS_COLORS):
        direction = np.eye(3)[axis]
        for reach, width, tag in ((AXIS_NEAR, 6.0, "near"), (AXIS_FAR, 1.5, "far")):
            scene.add_line_segments(
                f"/axis_{'xyz'[axis]}_{tag}",
                points=np.array([[np.zeros(3), reach * direction]]),
                colors=colour,
                line_width=width,
            )


def place_camera(client: viser.ClientHandle):
    """Open every client looking at the origin from a fixed pose, so the frame is always in view."""
    client.camera.position = np.asarray(START_EYE)
    client.camera.look_at = np.zeros(3)


def redraw(handle: viser.GaussianSplatHandle, readout: viser.GuiMarkdownHandle, state: dict):
    """Push the current similarity onto the splat and into the panel."""
    similarity = state["similarity"]
    handle.position = similarity.translation
    handle.wxyz = similarity.rotation.as_quat(scalar_first=True)  # viser orders quaternions wxyz
    handle.scale = similarity.scale
    readout.content = readout_text(similarity)


def nudge(_: object, *, state: dict, ramp: Ramp, kind: str, axis: int, sign: float, draw: Callable):
    """Move the similarity by one keypress worth of its ramped step."""
    step = sign * ramp.next()
    similarity = state["similarity"]
    if kind == "shift":
        shifted = similarity.translation + step * np.eye(3)[axis]
        state["similarity"] = similarity._replace(translation=shifted)
    elif kind == "turn":
        turn = R.from_rotvec(step * np.eye(3)[axis])
        state["similarity"] = similarity._replace(rotation=turn * similarity.rotation)
    else:
        state["similarity"] = similarity._replace(scale=similarity.scale * (1.0 + step))
    draw()


def save(_: object, *, state: dict, path: Path):
    """Write the similarity into the scene directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state["similarity"].as_dict(), indent=1))
    logger.info(f"wrote {path}: {state['similarity'].as_dict()}")


def bind(server: viser.ViserServer, state: dict, draw: Callable):
    """Register the hotkeys that move the similarity, one ramp per key and direction."""
    steps = {"shift": SHIFT_STEP, "turn": TURN_STEP, "scale": SCALE_STEP}
    for kind, keys in zip(("shift", "turn", "scale"), KEYS):
        for axis, key in enumerate(keys):
            # Scale has no axes, its two keys are the two directions instead
            signs = ((1.0, None), (-1.0, "shift")) if kind != "scale" else ((1.0 - 2 * axis, None),)
            for sign, modifier in signs:
                way = "up" if sign > 0 else "down"
                label = f"{kind} {way}" if kind == "scale" else f"{kind} {'xyz'[axis]} {way}"
                command = server.gui.add_command(
                    label, hotkey=key.upper(), modifier=modifier, description=CONVENTION
                )
                command.on_trigger(
                    functools.partial(
                        nudge,
                        state=state,
                        ramp=Ramp(*steps[kind]),
                        kind=kind,
                        axis=axis,
                        sign=sign,
                        draw=draw,
                    )
                )


# The landmark defaults were measured in the lab, so any other room has to pass its own
def refine(
    splat: Path,
    frame: Path | None = None,
    front: float = 3.15,
    back: float = -2.98,
    marker: tuple[float, float, float] = (-0.14, 3.15, 0.0),
    port: int = 8080,
):
    """Align a reconstruction against landmarks measured in the room."""
    splat_path = Path(splat)
    assert splat_path.exists(), f"no splat at {splat_path}"
    frame_path = Path(frame) if frame is not None else splat_path.parents[1] / FRAME_FILE
    # Pick up where the last session left off, or start from the identity
    similarity = SimilarityTransform(R.identity(), 1.0, np.zeros(3))
    if frame_path.exists():
        raw = json.loads(frame_path.read_text())
        similarity = SimilarityTransform(
            R.from_quat(raw["quat"]), float(raw["scale"]), np.array(raw["translation"], float)
        )
    state = {"similarity": similarity}
    # scipy's quaternion path cannot take device buffers
    means, log_scales, quats, sh, opacity = (np.array(x) for x in splax.io.load_ply(splat_path))

    server = viser.ViserServer(port=port)
    scales, opacities = splax.io.apply_activations(jnp.asarray(log_scales), jnp.asarray(opacity))
    orientation = R.from_quat(quats, scalar_first=True).as_matrix()
    covariances = np.einsum("nij,nj,nkj->nik", orientation, np.asarray(scales) ** 2, orientation)
    handle = server.scene.add_gaussian_splats(
        "/scene",
        centers=np.asarray(means, np.float32),
        covariances=covariances,
        rgbs=np.clip(np.asarray(splax.io.sh_to_rgb(jnp.asarray(sh[:, 0]))), 0.0, 1.0),
        opacities=np.asarray(opacities, np.float32)[:, None],
    )
    draw_axes(server.scene)
    for name, y in (("front", front), ("back", back)):
        server.scene.add_line_segments(
            f"/{name}",
            points=np.array([[[-LINE_SPAN / 2, y, 0.0], [LINE_SPAN / 2, y, 0.0]]]),
            colors=(255, 90, 90),
            line_width=3.0,
        )
    server.scene.add_icosphere("/marker", radius=0.05, color=(90, 200, 255), position=marker)

    server.on_client_connect(place_camera)
    readout = server.gui.add_markdown(readout_text(similarity))
    draw = functools.partial(redraw, handle, readout, state)
    bind(server, state, draw)
    server.gui.add_button("save").on_click(functools.partial(save, state=state, path=frame_path))
    draw()

    logger.info(f"viewer on http://localhost:{port}, similarity from {frame_path}")
    logger.info(f"lines at y={front} and y={back}, marker at {marker}")
    while True:
        time.sleep(1.0)


def main(splat: Path, out: Path, frame: Path | None = None):
    """Carry a splat into the room's frame with the similarity its scene was aligned with.

    Args:
        splat: Ply of the raw splat to place.
        out: Where the placed ply is written.
        frame: Similarity transform to apply. Defaults to the frame file of the scene holding the splat.
    """
    splat_path = Path(splat)
    assert splat_path.exists(), f"no splat at {splat_path}"
    frame_path = Path(frame) if frame is not None else splat_path.parents[1] / FRAME_FILE
    assert frame_path.exists(), f"no similarity transform at {frame_path}, align the scene first"
    out_path = Path(out)
    assert out_path.resolve() != splat_path.resolve(), "cannot overwrite the input splat"
    # Read the similarity transform and apply it to the splat
    raw = json.loads(frame_path.read_text())
    similarity = SimilarityTransform(
        R.from_quat(raw["quat"]), float(raw["scale"]), np.array(raw["translation"], float)
    )
    aligned_splat = transform(splax.io.prune(*splax.io.load_ply(splat_path)), similarity)
    # Write the transformed splat to the file
    out_path.parent.mkdir(parents=True, exist_ok=True)
    splax.io.write_ply(out_path, *(jnp.asarray(x, jnp.float32) for x in aligned_splat))
    means = aligned_splat[0]
    logger.info(f"wrote {out_path}: {len(means):,} gaussians, similarity from {frame_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire({"refine": refine, "apply": main})
