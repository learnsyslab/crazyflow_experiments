"""Select the frames for the reconstruction from a video.

We use regularly spaced timestamps, but pick the sharpest frame available around each spacing.

Run with
    pixi run -e vision python crazyflow_experiments/vision/reconstruct/frames.py <path to recording>
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import fire
import numpy as np

from crazyflow_experiments.utils.paths import RECORDINGS_DIR, SCENES_DIR

logger = logging.getLogger(__name__)


def scan(video: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read every frame for its timestamp and sharpness, without keeping the images.

    Returns:
        Frame timestamps in seconds and their sharpness, one entry per frame.
    """
    capture = cv2.VideoCapture(str(video))
    assert capture.isOpened(), f"cannot open {video}"
    fps = capture.get(cv2.CAP_PROP_FPS)
    assert fps > 0, f"{video} reports no frame rate"
    stamps, sharpness, index = [], [], 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        stamps.append(index / fps)
        sharpness.append(cv2.Laplacian(grey, cv2.CV_64F).var() / (grey.var() + 1e-6))
        index += 1
        if index % 2000 == 0:
            logger.info(f"scanned {index} frames")
    capture.release()
    assert index, f"{video} holds no frames"
    return np.array(stamps), np.array(sharpness)


def select(stamps: np.ndarray, sharpness: np.ndarray, interval: float, window: int) -> np.ndarray:
    """Choose frames spaced in time, taking the sharpest one available around each spacing.

    Args:
        stamps: Frame timestamps in seconds.
        sharpness: Sharpness per frame.
        interval: Seconds between selected frames.
        window: Frames searched either side of each spacing for a sharper one.

    Returns:
        Sorted indices of the chosen frames.
    """
    targets, anchor = [], None
    for index in range(len(stamps)):
        if anchor is None or stamps[index] - stamps[anchor] >= interval:
            targets.append(index)
            anchor = index

    available, chosen = set(range(len(stamps))), []
    for target in targets:
        nearby = [j for j in range(target - window, target + window + 1) if j in available]
        if not nearby:
            continue
        best = max(nearby, key=lambda j: sharpness[j])
        available.discard(best)
        chosen.append(best)
    return np.array(sorted(chosen))


def write_frames(video: Path, wanted: dict[int, str], images: Path):
    """Decode the recording again and write out only the selected frames.

    Args:
        video: Path to the recording.
        wanted: Frame index mapped to the filename it is written under.
        images: Directory the frames are written to.
    """
    images.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index in wanted:
            cv2.imwrite(str(images / wanted[index]), frame)
        index += 1
    capture.release()


def main(
    recording: str,
    scene: str | None = None,
    out: str | None = None,
    interval: float = 0.5,
    window: int = 4,
):
    """Select and write the frames a reconstruction of this recording should run on.

    Args:
        recording: Name of a recording under recordings/, or a path to one elsewhere.
        scene: Name of the scene the frames are written to. Defaults to the recording's stem.
        out: Explicit directory for the frames, overriding scene.
        interval: Seconds between selected frames. Denser spacing helps loop closure and costs
            reconstruction time roughly quadratically.
        window: Frames searched either side of each spacing for a sharper one.
    """
    # A bare name resolves under the recordings directory, anything with a parent is taken as given
    named = Path(recording).expanduser()
    source = named if named.parent != Path(".") else RECORDINGS_DIR / named
    assert source.exists(), f"no recording at {source}"
    output = Path(out).expanduser() if out else SCENES_DIR / (scene or source.stem)
    output.mkdir(parents=True, exist_ok=True)

    stamps, sharpness = scan(source)
    logger.info(
        f"{len(stamps)} frames over {stamps[-1]:.1f} s, mean sharpness {sharpness.mean():.3f}"
    )
    subset = select(stamps, sharpness, interval, window)
    assert len(subset), f"no frames selected from {source}"
    names = {int(index): f"frame_{index:06d}.png" for index in subset}
    logger.info(
        f"selected {len(subset)} frames, sharpness {sharpness[subset].mean():.3f} against "
        f"{sharpness.mean():.3f} over all frames"
    )

    write_frames(source, names, output / "images")
    (output / "frames.json").write_text(
        json.dumps(
            {
                "video": str(source),
                "interval": interval,
                "window": window,
                "duration_s": float(stamps[-1]),
                "scanned": len(stamps),
                "names": sorted(names.values()),
            },
            indent=1,
        )
    )
    logger.info(f"wrote {len(subset)} frames to {output / 'images'}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    fire.Fire(main)
