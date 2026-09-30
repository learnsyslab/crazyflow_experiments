"""Export a COLMAP reconstruction to the scene used by the splat trainer.

Scenes are arbitrarily scaled and have to be aligned to the real world before training.

Run with
    pixi run -e vision python crazyflow_experiments/vision/reconstruct/export.py \
        saves/scenes/lab/sparse/0
"""

from __future__ import annotations

import logging
from pathlib import Path

import fire
import numpy as np
import pycolmap

from crazyflow_experiments.utils.paths import SCENES_DIR

logger = logging.getLogger(__name__)


def export(model: Path, out: Path, max_error: float = 5.0) -> tuple[int, int]:
    """Write a COLMAP model out as the arrays the trainer consumes.

    Args:
        model: Directory holding the COLMAP model.
        out: Path of the scene archive to write.
        max_error: Reprojection error in px above which a triangulated point is dropped.

    Returns:
        The number of views and points written.
    """
    reconstruction = pycolmap.Reconstruction(str(model))
    cameras = list(reconstruction.cameras.values())
    assert len(cameras) == 1, f"expected one camera, found {len(cameras)}"
    camera = cameras[0]
    assert camera.model.name == "SIMPLE_RADIAL", f"expected SIMPLE_RADIAL, got {camera.model.name}"
    params = dict(zip(camera.params_info.split(", "), camera.params, strict=True))
    logger.info(f"camera {camera.model.name} {camera.width}x{camera.height} {params}")

    images = sorted(
        (image for image in reconstruction.images.values() if image.has_pose), key=lambda i: i.name
    )
    assert images, f"no registered images in {model}"
    viewmats = np.stack([image.cam_from_world().matrix() for image in images]).astype(np.float32)
    homogeneous = np.tile(np.array([[[0.0, 0.0, 0.0, 1.0]]], np.float32), (len(images), 1, 1))
    viewmats = np.concatenate([viewmats, homogeneous], axis=1)

    kept = [point for point in reconstruction.points3D.values() if point.error <= max_error]
    dropped = reconstruction.num_points3D() - len(kept)
    assert kept, f"every point in {model} exceeds {max_error} px of reprojection error"
    focal = params["f"]
    intrinsics = np.array([focal, focal, params["cx"], params["cy"]], np.float32)
    dist = np.array([params["k"], 0.0, 0.0, 0.0, 0.0], np.float32)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        width=camera.width,
        height=camera.height,
        intrinsics=intrinsics,
        dist=dist,
        names=np.array([image.name for image in images]),
        viewmats=viewmats,
        points_xyz=np.stack([point.xyz for point in kept]).astype(np.float32),
        points_rgb=np.stack([point.color for point in kept]).astype(np.uint8),
        points_track=np.array([point.track.length() for point in kept], np.int32),
    )
    logger.info(
        f"wrote {out}: {len(images)} views, {len(kept)} points, dropped {dropped} above "
        f"{max_error} px"
    )
    return len(images), len(kept)


def main(model: str, scene: str | None = None, out: str | None = None, max_error: float = 5.0):
    """Export a reconstruction for the splat trainer.

    Args:
        model: Directory holding the COLMAP model, such as saves/scenes/lab/sparse/0.
        scene: Name of the scene the archive is written to. Defaults to the name of the directory
            holding the model.
        out: Explicit path for the archive, overriding scene.
        max_error: Reprojection error in px above which a triangulated point is dropped.
    """
    source = Path(model).expanduser()
    assert source.exists(), f"no COLMAP model at {source}"
    if out is not None:
        target = Path(out).expanduser()
    else:
        # The model sits at <scene>/sparse/<n>, so its grandparent names the scene
        target = SCENES_DIR / (scene or source.parents[1].name) / "scene.npz"
    export(source, target, max_error)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(main)
