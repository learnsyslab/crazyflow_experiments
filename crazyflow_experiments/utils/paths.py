"""Filesystem locations shared across the experiments."""

from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]
RESULTS_DIR = REPO_ROOT / "results"  # wandb appends "wandb", so the final path is results/wandb
SAVES_DIR = REPO_ROOT / "saves"  # Checkpoints, one subdirectory per experiment
PLOTS_DIR = REPO_ROOT / "plots"  # Figures, one subdirectory per experiment
RECORDINGS_DIR = REPO_ROOT / "recordings"  # Raw captures a reconstruction is built from
SCENES_DIR = SAVES_DIR / "scenes"  # Reconstructed scenes, one subdirectory per scene


def new_recording_path(folder: Path, name: str, suffix: str = ".pkl") -> Path:
    """Reserve a path for a recording that is about to be written, creating the directory.

    Args:
        folder: Directory the file is written to.
        name: Stem the timestamp is appended to.
        suffix: File suffix including the leading dot.

    Returns:
        A path, stamped with the current date and time.
    """
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}{suffix}"
