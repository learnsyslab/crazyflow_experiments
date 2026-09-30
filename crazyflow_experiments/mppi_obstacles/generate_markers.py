"""Generate distinguishable mocap marker layouts for the obstacle pillars.

Run with
    pixi run -e mppi python crazyflow_experiments/mppi_obstacles/generate_markers.py
"""

import csv
import logging
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
from dppy.finite_dpps import FiniteDPP
from numpy.typing import NDArray
from scipy.spatial.distance import pdist, squareform

from crazyflow_experiments.mppi_obstacles import RECORDINGS_DIR

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

HOLE_SPACING = 10.0  # mm
HOLE_RADIUS = 45.0  # mm, the holes cover a circular area around the deck center
HOLE_HEIGHTS = (0.0, 10.0, 14.0)  # mm, the deck carries markers on three levels


def marker_positions() -> NDArray[np.floating]:
    """Marker hole positions of a pillar deck.

    Returns:
        Hole positions of shape (N, 3) in mm, ordered by height, then x, then y.
    """
    grid = np.arange(-(HOLE_RADIUS // HOLE_SPACING), HOLE_RADIUS // HOLE_SPACING + 1) * HOLE_SPACING
    xy = np.stack(np.meshgrid(grid, grid, indexing="ij"), axis=-1).reshape(-1, 2)
    xy = xy[np.linalg.norm(xy, axis=1) <= HOLE_RADIUS]
    return np.array([[x, y, z] for z in HOLE_HEIGHTS for x, y in xy])


def is_unique_deck(
    signature: NDArray[np.floating], accepted: list[NDArray[np.floating]], tolerance: float
) -> bool:
    """Check if a deck is physically distinguishable from the already accepted decks.

    Args:
        signature: Sorted pairwise marker distances of the new deck.
        accepted: Sorted pairwise marker distances of the accepted decks.
        tolerance: Distance in mm below which two decks count as identical.

    Returns:
        True if the new deck differs from all accepted decks.
    """
    return not any(np.allclose(signature, s, atol=tolerance) for s in accepted)


def generate_mocap_decks(
    holes: NDArray[np.floating],
    k_markers: int,
    n_decks: int,
    sigma: float,
    tolerance: float = 10.0,
    min_marker_distance: float = 15.0,
    max_attempts: int = 5000,
) -> list[NDArray[np.floating]]:
    """Sample distinguishable marker decks with a k-DPP over the available holes.

    Args:
        holes: Hole positions of shape (N, 3) in mm.
        k_markers: Number of markers per deck.
        n_decks: Number of decks to generate.
        sigma: Length scale in mm of the similarity kernel that spreads the markers out.
        tolerance: Distance in mm below which two decks count as identical.
        min_marker_distance: Minimum distance in mm between two markers of the same deck.
        max_attempts: Number of k-DPP samples to draw before giving up.

    Returns:
        A list of n_decks marker positions, each of shape (k_markers, 3) in mm.
    """
    similarity = np.exp(-(squareform(pdist(holes)) ** 2) / (2 * sigma**2))
    dpp = FiniteDPP("likelihood", L=similarity + 1e-6 * np.eye(len(holes)))

    decks, signatures = [], []
    logger.info(
        f"Generating {n_decks} decks with a minimum marker distance of {min_marker_distance}mm"
    )
    for attempt in range(max_attempts):
        if len(decks) == n_decks:
            break
        dpp.sample_exact_k_dpp(size=k_markers)
        markers = holes[dpp.list_of_samples[-1]]
        distances = pdist(markers)
        if np.min(distances) < min_marker_distance:
            continue
        signature = np.sort(distances)
        if is_unique_deck(signature, signatures, tolerance):
            decks.append(markers)
            signatures.append(signature)
            logger.info(f"Deck {len(decks)} found after {attempt + 1} attempts")

    assert len(decks) == n_decks, (
        f"Only found {len(decks)} of {n_decks} decks in {max_attempts} attempts"
    )
    return decks


def export_decks_to_csv(decks: list[NDArray[np.floating]], path: Path):
    """Write the marker positions of each deck to a CSV file.

    Args:
        decks: Marker positions of each deck, each of shape (k_markers, 3) in mm.
        path: Destination of the CSV file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Object_ID", "Marker_ID", "X_mm", "Y_mm", "Z_mm"])
        for i, deck in enumerate(decks):
            for j, marker in enumerate(deck):
                writer.writerow([f"Object_{i + 1}", f"Marker_{j + 1}", *np.round(marker, 2)])
    logger.info(f"Marker positions saved to {path}")


def visualize_decks(holes: NDArray[np.floating], decks: list[NDArray[np.floating]]):
    """Render the marker constellations on top of the available holes.

    Args:
        holes: Hole positions of shape (N, 3) in mm.
        decks: Marker positions of each deck, each of shape (k_markers, 3) in mm.
    """
    ax = plt.figure(figsize=(10, 8)).add_subplot(projection="3d")
    ax.scatter(*holes.T, c="gray", alpha=0.05, label="Available holes")

    colors = ["#E63946", "#1D3557", "#2A9D8F", "#F4A261", "#9B5DE5"]
    for i, deck in enumerate(decks):
        color = colors[i % len(colors)]
        ax.scatter(*deck.T, c=color, s=100, edgecolor="black", label=f"Object {i + 1}")
        for j, k in zip(*np.triu_indices(len(deck), k=1)):
            ax.plot(*deck[[j, k]].T, c=color, alpha=0.6, linewidth=1.5)

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("Optimized Motion Capture Marker Decks")
    ax.legend()
    plt.show()


def main(k_markers: int = 4, n_decks: int = 10, sigma: float = 45.0, show: bool = True):
    """Generate the marker decks and save them to the data folder.

    Args:
        k_markers: Number of markers per deck.
        n_decks: Number of decks to generate.
        sigma: Length scale in mm of the similarity kernel.
        show: Render the markers after saving them.
    """
    possible_marker_positions = marker_positions()
    decks = generate_mocap_decks(possible_marker_positions, k_markers, n_decks, sigma)
    export_decks_to_csv(decks, RECORDINGS_DIR / "mocap_decks.csv")
    if show:
        visualize_decks(possible_marker_positions, decks)


if __name__ == "__main__":
    fire.Fire(main)
