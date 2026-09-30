"""Locate the gaussian splats a scene is rendered from, on disk or in the scene repository."""

from __future__ import annotations

from typing import TYPE_CHECKING

from splax.io import fetch

from crazyflow_experiments.utils.paths import SCENES_DIR

if TYPE_CHECKING:
    from pathlib import Path

SCENES_URL = "https://huggingface.co/datasets/amacati/splats/resolve/main"


def splat(scene: str, variant: str, stage: str = "aligned") -> Path:
    """Return a splat from the scene directory or from the remote dataset."""
    local = SCENES_DIR / scene / stage / f"{variant}.ply"
    # The remote paths only use the variant name
    return local if local.exists() else fetch(f"{SCENES_URL}/{variant}.ply")
