"""Reconstruct camera poses from a video without mocap using COLMAP with CASPAR.

Run with
    pixi run -e vision python crazyflow_experiments/vision/reconstruct/sfm.py obstacle
"""

from __future__ import annotations

import logging
import os
import sqlite3
import subprocess
from pathlib import Path

import fire
import pycolmap

from crazyflow_experiments.utils.paths import SCENES_DIR

logger = logging.getLogger(__name__)

COLMAP = Path(os.environ["CONDA_PREFIX"]) / "bin/colmap"
# COLMAP moved from flann to faiss indices in 2025, so the legacy trees no longer load
VOCAB_TREE = Path.home() / ".cache/colmap/vocab_tree_faiss_flickr100K_words256K.bin"
CAMERA_MODEL = "SIMPLE_RADIAL"
MAX_FEATURES = 16384
LOOP_PERIOD = 10  # frames between loop closure attempts
LOOP_CANDIDATES = 80  # vocabulary tree matches considered per attempt


def counts(database: Path) -> tuple[int, int]:
    """Count images with features and image pairs with verified matches."""
    if not database.exists():  # Nothing has run yet, so both stages are still to do
        return 0, 0
    with sqlite3.connect(database) as connection:
        images = connection.execute("SELECT count(*) FROM keypoints").fetchone()[0]
        pairs = connection.execute("SELECT count(*) FROM two_view_geometries").fetchone()[0]
    return images, pairs


def run(colmap: Path, command: str, flags: dict[str, str]):
    """Run one COLMAP command.

    Args:
        colmap: The COLMAP binary to invoke.
        command: The COLMAP subcommand, for instance "mapper".
        flags: Flag names mapped to their values, flattened onto the command line in order.
    """
    arguments = [str(colmap), command]
    for flag, value in flags.items():
        arguments += [flag, value]
    logger.info(f"colmap {command}")
    subprocess.run(arguments, check=True)


def extract_features(colmap: Path, database: Path, images: Path, max_image_size: int):
    """Detect SIFT features on the GPU."""
    run(
        colmap,
        "feature_extractor",
        {
            "--database_path": str(database),
            "--image_path": str(images),
            "--ImageReader.camera_model": CAMERA_MODEL,
            "--ImageReader.single_camera": "1",
            "--FeatureExtraction.type": "SIFT",
            "--FeatureExtraction.max_image_size": str(max_image_size),
            "--SiftExtraction.max_num_features": str(MAX_FEATURES),
            # COLMAP 4.x renamed this, it is not --SiftExtraction.use_gpu any more
            "--FeatureExtraction.use_gpu": "1",
        },
    )


def match_features(colmap: Path, database: Path, overlap: int):
    """Match consecutive frames, then close loops through the vocabulary tree.

    Args:
        colmap: The COLMAP binary.
        database: Database holding the features.
        overlap: Sequential matching window in frames.
    """
    assert VOCAB_TREE.exists(), f"no vocabulary tree at {VOCAB_TREE}."
    run(
        colmap,
        "sequential_matcher",
        {
            "--database_path": str(database),
            "--FeatureMatching.type": "SIFT_BRUTEFORCE",
            "--FeatureMatching.use_gpu": "1",
            "--SequentialMatching.overlap": str(overlap),
            "--SequentialMatching.quadratic_overlap": "1",
            "--SequentialMatching.loop_detection": "1",
            "--SequentialMatching.vocab_tree_path": str(VOCAB_TREE),
            "--SequentialMatching.loop_detection_period": str(LOOP_PERIOD),
            "--SequentialMatching.loop_detection_num_images": str(LOOP_CANDIDATES),
        },
    )


def reconstruct(colmap: Path, database: Path, images: Path, output: Path):
    """Run the incremental mapper with CASPAR bundle adjustment on the GPU."""
    output.mkdir(parents=True, exist_ok=True)
    run(
        colmap,
        "mapper",
        {
            "--database_path": str(database),
            "--image_path": str(images),
            "--output_path": str(output),
            "--Mapper.ba_local_backend": "CASPAR",
            "--Mapper.ba_global_backend": "CASPAR",
        },
    )


def survey(output: Path, images: int) -> list[dict]:
    """Report every model the mapper produced, largest first.

    Args:
        output: Directory the mapper wrote its models into.
        images: Number of images that were offered to the mapper.

    Returns:
        One record per model, carrying its registered image count, points, mean track length,
        mean observations per image and mean reprojection error.
    """
    models = []
    for path in sorted(output.iterdir()):
        if not (path / "cameras.bin").exists():
            continue
        model = pycolmap.Reconstruction(str(path))
        observed = [image.num_points3D for image in model.images.values()]
        models.append(
            {
                "path": path.name,
                "registered": model.num_reg_images(),
                "points": model.num_points3D(),
                "mean_track_length": float(model.compute_mean_track_length()),
                "mean_observations": float(sum(observed) / len(observed)),
                "reprojection_error": float(model.compute_mean_reprojection_error()),
            }
        )
    models.sort(key=lambda entry: -entry["registered"])
    for entry in models:
        logger.info(
            f"model {entry['path']}: {entry['registered']}/{images} images, {entry['points']} "
            f"points, track {entry['mean_track_length']:.2f}, reprojection "
            f"{entry['reprojection_error']:.3f} px"
        )
    return models


def main(
    scene: str,
    images: str | None = None,
    out: str | None = None,
    overlap: int = 15,
    max_image_size: int = 1600,
    min_registered: float = 0.8,
) -> list[dict]:
    """Reconstruct the recording and report whether the result is a single consistent model.

    Args:
        scene: Name of the scene holding the frames the frames stage extracted.
        images: Explicit directory of frames, overriding scene.
        out: Explicit directory for the database and the models, overriding scene.
        overlap: Sequential matching window in frames.
        max_image_size: Longest image edge fed to the feature detector.
        min_registered: Fraction of frames the largest model must hold to count as consistent.

    Returns:
        One record per model the mapper produced, largest first.
    """
    folder = SCENES_DIR / scene
    image_dir = Path(images).expanduser() if images else folder / "images"
    output = Path(out).expanduser() if out else folder
    count = len(list(image_dir.glob("*.png")))
    assert count, f"no frames in {image_dir}"
    database, models = output / "database.db", output / "sparse"
    assert COLMAP.exists(), (
        f"no colmap at {COLMAP}. Build it with `pixi run -e vision colmap-build`, which takes "
        f"roughly half an hour."
    )
    logger.info(f"reconstructing {count} frames from {image_dir} with {COLMAP}")

    featured, pairs = counts(database)
    if featured < count:
        extract_features(COLMAP, database, image_dir, max_image_size)
    if not pairs:
        match_features(COLMAP, database, overlap)
    logger.info(f"database holds {counts(database)[1]} verified pairs")
    reconstruct(COLMAP, database, image_dir, models)

    found = survey(models, count)
    assert found, "the mapper produced no reconstruction"
    fraction = found[0]["registered"] / count
    if len(found) > 1 or fraction < min_registered:
        logger.warning(
            f"reconstruction is fractured: {len(found)} models, the largest holding "
            f"{fraction:.1%} of frames. Loop closure likely failed, so training on this would fit "
            f"a fragment of the room."
        )
    else:
        logger.info(f"single consistent model with {fraction:.1%} of frames registered")
    return found


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    fire.Fire(main)
