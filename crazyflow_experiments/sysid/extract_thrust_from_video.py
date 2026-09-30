"""Extract thrust readings from a screen recording of the thrust stand.

The recording shows a terminal that prints one normalized thrust command per line, as
``px4_random_thrust.py`` and ``PX4_thrust_sweep.sh`` do, and a webcam view of the seven-segment
display of the thrust stand. The script asks for the terminal pane and the four corners of the
display digits, finds the moment every command line is printed, reads the printed number, reads the
display right before the next command is printed and writes the command-thrust pairs to a csv next to
the video.

Usage:
    pixi run -e sysid python crazyflow_experiments/sysid/extract_thrust_from_video.py \
        results/sysid/random_2026-09-04.webm

Workflow:
    1. Drag a rectangle around the terminal output pane and press Enter.
    2. The video is scanned for the moments a new ``d.dd`` line appears at the bottom of the pane.
       The digits of all printed commands are grouped by appearance and a window shows one glyph
       per group. Type 0-9 for the highlighted glyph, a/d select the neighbouring glyph, Backspace
       clears a label, Enter accepts once every glyph has its digit.
    3. On the frame with the highest command, drag a rough rectangle around the display, then
       click the four corners of the block of digits (top-left, top-right, bottom-right,
       bottom-left). Follow the slant of the digits, that removes both perspective and italics. A
       preview shows the segment boxes on the rectified digits. Drag the corners, or select one with
       1-4 and nudge it with w/a/s/d, until every box sits on its segment and no box is yellow, then
       press Enter.
    4. The display is decoded for every command and a review sheet shows the black-and-white images
       that were decoded, one page at a time (n/p change the page). Yellow boxes mark segments too
       close to the lit/unlit threshold. The contrast and brightness sliders change the
       black-and-white image until the segments are black on white. Click a segment box to force it
       lit, click again to force it unlit, a third click returns it to automatic. Press Enter to save
       once every step has a value, or ``q`` to abort.

The display is tracked through the video by template matching, so the reading follows the display
when the propeller wash shakes the camera. The regions are stored in ``<video>.rois.json`` (with the
tracking template in ``<video>.template.png``) and the labelled glyphs in ``<video>.glyphs.npz``,
all reused with ``--reuse-rois``.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass, field
from itertools import zip_longest
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import cv2
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import NDArray

# Terminal reading. The pane is read after every noticeable change, the bottom text line is parsed
# as ``d.dd`` and its digit glyphs are grouped by appearance, so that the user labels each digit once.
CHANGE_LEVEL = 40  # grey levels a pixel must change by to count as changed
COLLAPSE_WINDOW = 0.35  # seconds within which changes belong to the same redraw of the pane
READ_DELAY = 0.2  # seconds after the first change of a redraw at which the pane is read
INK_CONTRAST = 60  # grey levels a text pixel differs from the terminal background
CURSOR_KERNEL = 7  # solid blocks larger than this are the block cursor, glyph strokes are thinner
GLYPH_SIZE = (16, 24)  # width, height that every digit glyph is resized to before comparison
GLYPH_MATCH = 0.85  # correlation above which two glyphs show the same digit
SAME_TEXT_PIXELS = 20  # fewer changed pixels between two crops of a text line mean the same text
IRREGULAR_GAP = 1.5  # gaps between prints longer than this many periods are not consecutive steps

DIGITS = 4
DIGIT_GAP = 0.45  # gap between digits relative to the digit width, measured on the Mayatech display
DIGIT_HEIGHT = 80
DIGIT_WIDTH = 48
DISPLAY_WIDTH = round(DIGIT_WIDTH * (DIGITS + (DIGITS - 1) * DIGIT_GAP))
DISPLAY_PAD = 12  # margin around the rectified digits that the alignment refinement may use
# Segment boxes in digit coordinates (x0, y0, x1, y1), order a b c d e f g, measured on the Mayatech
# display. The boxes lie inside their segments and do not overlap, so a lit segment does not leak into
# the reading of its unlit neighbours.
SEGMENT_BOXES = (
    (0.30, 0.02, 0.70, 0.10),
    (0.76, 0.14, 0.97, 0.39),
    (0.76, 0.55, 0.97, 0.83),
    (0.30, 0.87, 0.70, 0.95),
    (0.03, 0.55, 0.24, 0.83),
    (0.03, 0.14, 0.24, 0.39),
    (0.30, 0.43, 0.70, 0.51),
)
SEGMENT_PATTERNS = {
    (1, 1, 1, 1, 1, 1, 0): 0,
    (0, 1, 1, 0, 0, 0, 0): 1,
    (1, 1, 0, 1, 1, 0, 1): 2,
    (1, 1, 1, 1, 0, 0, 1): 3,
    (0, 1, 1, 0, 0, 1, 1): 4,
    (1, 0, 1, 1, 0, 1, 1): 5,
    (1, 0, 1, 1, 1, 1, 1): 6,
    (1, 1, 1, 0, 0, 0, 0): 7,
    (1, 1, 1, 1, 1, 1, 1): 8,
    (1, 1, 1, 1, 0, 1, 1): 9,
}
BLANK = (0,) * 7
LOW_CONFIDENCE = 0.4  # readings below this are marked for review
# Review sheet layout: tiles of the rectified display at TILE_ZOOM with a text banner, in columns.
# A page shows COLUMNS x PAGE_ROWS tiles, the sheet of a long recording is paged.
TILE_ZOOM = 2
BANNER_HEIGHT = 52
HEADER_HEIGHT = 102
COLUMNS = 4
PAGE_ROWS = 5
TILES_PER_PAGE = COLUMNS * PAGE_ROWS
FONT_SCALE = 0.7
VALUE_SCALE = 1.4  # the guessed number is drawn at twice the size of the other text
GLYPH_ZOOM = 6  # magnification of the glyphs in the labelling window


@dataclass
class Regions:
    """Normalized image regions selected by the user."""

    terminal: tuple[float, float, float, float]  # x, y, width, height
    lcd_corners: list[tuple[float, float]]  # top-left, top-right, bottom-right, bottom-left
    display: tuple[float, float, float, float] | None = None  # rough box around the display
    template_time: float | None = None  # time of the frame the corners and display were marked in

    def save(self, path: Path, video: Path, size: tuple[int, int]) -> None:
        """Write the regions as a json sidecar."""
        data = {
            "video": video.name,
            "source_size": {"width": size[0], "height": size[1]},
            "terminal": dict(zip(("x", "y", "width", "height"), self.terminal)),
            "lcd_corners": [{"x": x, "y": y} for x, y in self.lcd_corners],
            "display": dict(zip(("x", "y", "width", "height"), self.display or ())) or None,
            "template_time": self.template_time,
        }
        path.write_text(json.dumps(data, indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> Regions:
        """Read the regions from a json sidecar."""
        data = json.loads(path.read_text())
        required = ("terminal", "lcd_corners", "display", "template_time")
        if any(key not in data for key in required):
            raise KeyError(f"{path} is from an older version, run again without --reuse-rois")
        terminal, display = data["terminal"], data["display"]
        return cls(
            terminal=(terminal["x"], terminal["y"], terminal["width"], terminal["height"]),
            lcd_corners=[(p["x"], p["y"]) for p in data["lcd_corners"]],
            display=(display["x"], display["y"], display["width"], display["height"]),
            template_time=data["template_time"],
        )

    def terminal_slice(self, width: int, height: int) -> tuple[slice, slice]:
        """Return the pixel slices of the terminal pane for a frame of the given size."""
        x, y, w, h = self.terminal
        return slice(round(y * height), round((y + h) * height)), slice(
            round(x * width), round((x + w) * width)
        )

    def display_box(self, width: int, height: int) -> tuple[int, int, int, int]:
        """Return the rough display box in pixels (x, y, width, height)."""
        if self.display is None:
            raise ValueError("The display box has not been selected")
        x, y, w, h = self.display
        return round(x * width), round(y * height), round(w * width), round(h * height)

    def lcd_transform(
        self, width: int, height: int, offset: tuple[float, float] = (0.0, 0.0)
    ) -> NDArray[np.float32]:
        """Return the perspective transform from the frame to the rectified display.

        ``offset`` moves the corners by that many pixels, it compensates camera motion.
        """
        source = np.array(self.lcd_corners, dtype=np.float32) * np.array(
            [width, height], dtype=np.float32
        ) + np.array(offset, dtype=np.float32)
        target = np.array(
            [(0, 0), (DISPLAY_WIDTH, 0), (DISPLAY_WIDTH, DIGIT_HEIGHT), (0, DIGIT_HEIGHT)],
            dtype=np.float32,
        )
        return cv2.getPerspectiveTransform(source, target + DISPLAY_PAD)


class ValueLine(NamedTuple):
    """A ``d.dd`` line at the bottom of the terminal pane."""

    top: int  # first pixel row of the line in the pane
    image: NDArray[np.uint8]  # grey crop of the text
    above: NDArray[np.uint8] | None  # grey crop of the text line above, at the same columns
    glyphs: list[NDArray[np.uint8]]  # the three digit glyphs, resized to GLYPH_SIZE


class TerminalEvent(NamedTuple):
    """The bottom line of the terminal after one redraw of the pane."""

    time: float  # first change of the redraw
    line: ValueLine | None  # None if the bottom line is not a value


@dataclass
class Command:
    """One printed thrust command."""

    time: float
    line: ValueLine
    groups: tuple[int, int, int] = (0, 0, 0)  # glyph group of each digit
    value: float = 0.0


class Adjust(NamedTuple):
    """Settings applied before decoding, tuned with the sliders of the review."""

    contrast: float = 1.0  # 1 leaves the black-and-white image as it is, larger values spread it
    brightness: int = 0  # grey level added to the black-and-white image, -255..255
    max_value: int = 1700  # highest thrust in g that is accepted as a reading


class Decoded(NamedTuple):
    """Result of decoding one rectified display."""

    value: int | None  # None if the segments do not form valid digits or exceed the maximum
    confidence: (
        float  # 1 if every segment sits at the mean of its class, 0 if one is on the threshold
    )
    segments: NDArray[np.bool_]  # (DIGITS, 7) lit segments
    darkness: NDArray[np.float64]  # (DIGITS, 7) segment darkness
    threshold: float  # darkness above which a segment counts as lit
    separation: float  # distance between the mean darkness of lit and unlit segments
    image: NDArray[np.uint8]  # the black-and-white image that was decoded
    ignored: NDArray[np.bool_]  # (DIGITS,) leading digits without a clearly lit segment, blank


@dataclass
class Reading:
    """Decoded display value of one command step."""

    value: int | None
    confidence: float
    n_frames: int
    n_agree: int
    image: NDArray[np.uint8]  # rectified display of the selected frame
    decoded: Decoded  # decoding of the selected frame, with the forced segments applied
    forced: dict[tuple[int, int], bool] = field(default_factory=dict)  # (digit, segment) -> lit


# region Video access


def iter_frames(video: Path) -> Iterator[tuple[float, NDArray[np.uint8]]]:
    """Yield (time in seconds, frame) for every frame of the video.

    Screen recordings have a variable frame rate and do not seek reliably, so all access is
    sequential.
    """
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video {video}")
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        yield capture.get(cv2.CAP_PROP_POS_MSEC) / 1000, frame
    capture.release()


def frame_at(video: Path, time_s: float) -> NDArray[np.uint8]:
    """Return the first frame at or after ``time_s``, or the last frame of the video."""
    frame = None
    for t, frame in iter_frames(video):
        if t >= time_s:
            return frame
    if frame is None:
        raise RuntimeError(f"Could not decode a frame from {video}")
    return frame


def frame_size(video: Path) -> tuple[int, int]:
    """Return the (width, height) of the video."""
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video {video}")
    size = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    return size


# endregion

# region Region selection


def preview_scale(frame: NDArray[np.uint8], max_size: tuple[int, int] = (1600, 900)) -> float:
    """Return the scale that fits the frame into ``max_size``."""
    height, width = frame.shape[:2]
    return min(1.0, max_size[0] / width, max_size[1] / height)


def select_rectangle(frame: NDArray[np.uint8], title: str) -> tuple[int, int, int, int]:
    """Let the user drag a rectangle on the frame and return it in frame pixels."""
    scale = preview_scale(frame)
    preview = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    print(f"{title} Drag the rectangle, then press Enter (c cancels).")
    x, y, w, h = cv2.selectROI(title, preview, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    if w <= 0 or h <= 0:
        raise RuntimeError("Selection was cancelled or empty")
    return round(x / scale), round(y / scale), round(w / scale), round(h / scale)


def select_corners(
    frame: NDArray[np.uint8], title: str, step: int, command: float
) -> tuple[list[tuple[float, float]], tuple[int, int, int, int]]:
    """Let the user mark the display and return the digit corners and the display box in pixels.

    The user first drags a rough rectangle around the display including some housing around it,
    which is then magnified for the corner clicks. The rectangle also serves as the template that
    tracks the display when the camera moves. Once four corners are placed, a preview shows the
    rectified digits with the fitted segment grid and the decoded value, and the corners can be
    dragged until the grid sits on the digits.
    """
    print(f"{title}: drag a rectangle around the display, include some of the housing around it.")
    x, y, w, h = select_rectangle(frame, f"{title}: display with some housing around it")
    crop = frame[y : y + h, x : x + w]
    zoom = min(16.0, 1000 / w, 700 / h)
    canvas = cv2.resize(crop, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
    points: list[list[float]] = []
    state = {"drag": -1, "active": -1, "dirty": False}
    window = f"{title}: click the 4 corners of the DIGITS (TL, TR, BR, BL); Enter accepts, r resets"
    preview_window = "Preview: every box must sit on its segment, yellow boxes are ambiguous"
    print(
        "Click the four corners of the block of digits (not of the glass) in the order top-left, "
        "top-right, bottom-right, bottom-left. This frame shows the highest command. Follow the "
        "slant of the digits. Drag a point to move it, or select it with 1-4 and nudge it with "
        "w/a/s/d. r resets, Enter accepts, Esc aborts."
    )

    def on_mouse(event: int, px: int, py: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            near = [i for i, (qx, qy) in enumerate(points) if abs(qx - px) + abs(qy - py) < 12]
            if near:
                state["drag"] = state["active"] = near[0]
            elif len(points) < 4:
                points.append([px, py])
                state["drag"] = state["active"] = len(points) - 1
                state["dirty"] = True
        elif event == cv2.EVENT_MOUSEMOVE and state["drag"] >= 0:
            points[state["drag"]] = [px, py]
            state["dirty"] = True
        elif event == cv2.EVENT_LBUTTONUP:
            state["drag"] = -1

    def frame_corners() -> list[tuple[float, float]]:
        return [(x + px / zoom, y + py / zoom) for px, py in points]

    def show_preview() -> None:
        height, width = frame.shape[:2]
        corners = [(cx / width, cy / height) for cx, cy in frame_corners()]
        display = rectify(frame, Regions((0, 0, 1, 1), corners).lcd_transform(width, height))
        decoded = decode_display(display)
        reading = Reading(decoded.value, decoded.confidence, 1, 1, display, decoded)
        cv2.imshow(preview_window, draw_reading(reading, step, command, 3))

    nudges = {ord("a"): (-1, 0), ord("d"): (1, 0), ord("w"): (0, -1), ord("s"): (0, 1)}
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)
    while True:
        image = canvas.copy()
        for i, (px, py) in enumerate(points):
            color = (0, 255, 255) if i == state["active"] else (0, 0, 255)
            cv2.circle(image, (round(px), round(py)), 4, color, -1)
            label = ("TL", "TR", "BR", "BL")[i]
            cv2.putText(
                image, label, (round(px), round(py)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )
        if len(points) > 1:
            polygon = np.array(points).round().astype(np.int32)
            cv2.polylines(image, [polygon], len(points) == 4, (0, 255, 255), 1)
        cv2.imshow(window, image)
        if len(points) == 4 and state["dirty"] and state["drag"] < 0:
            state["dirty"] = False
            show_preview()
        key = cv2.waitKey(30) & 0xFF
        if key == ord("r"):
            points.clear()
            state["active"] = -1
        elif key in (ord("1"), ord("2"), ord("3"), ord("4")) and key - ord("1") < len(points):
            state["active"] = key - ord("1")
        elif key in nudges and state["active"] >= 0:
            points[state["active"]][0] += nudges[key][0]
            points[state["active"]][1] += nudges[key][1]
            state["dirty"] = True
        elif key == 27:
            cv2.destroyAllWindows()
            raise RuntimeError("Corner selection aborted")
        elif key in (13, 10) and len(points) == 4:
            break
    cv2.destroyAllWindows()
    return frame_corners(), (x, y, w, h)


# endregion

# region Terminal reading


def mask_runs(mask: NDArray[np.bool_]) -> list[tuple[int, int]]:
    """Return the (start, stop) index ranges of the true runs of a 1d mask."""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def ink_mask(gray: NDArray[np.uint8]) -> NDArray[np.bool_]:
    """Return the text pixels of a terminal pane, without the block cursor.

    Text differs from the background, dark or light, by more than INK_CONTRAST grey levels. The block
    cursor is the only solid patch larger than a glyph stroke, a morphological opening isolates it.
    It is removed because it touches the line above it and would merge with it.
    """
    background = np.full_like(gray, int(np.median(gray)))
    ink = cv2.absdiff(gray, background) > INK_CONTRAST
    kernel = np.ones((CURSOR_KERNEL, CURSOR_KERNEL), np.uint8)
    blocks = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    return ink & (blocks == 0)


def parse_value_line(gray: NDArray[np.uint8]) -> ValueLine | None:
    """Read the bottom text line of the terminal pane if it has the form ``d.dd``.

    The line must consist of four glyphs, the second a dot at the baseline and the others of full
    digit height. Returns None if the bottom line is anything else. The line above is cropped at
    the same columns, it tells a repeated value apart from a redraw once the terminal scrolls.
    """
    ink = ink_mask(gray)
    lines = mask_runs(ink.any(axis=1))
    if not lines:
        return None
    top, bottom = lines[-1]
    height = bottom - top
    glyphs = mask_runs(ink[top:bottom].any(axis=0))
    if len(glyphs) != 4:
        return None
    for i, (x0, x1) in enumerate(glyphs):
        rows = np.flatnonzero(ink[top:bottom, x0:x1].any(axis=1))
        extent = rows[-1] - rows[0] + 1
        if i == 1 and (rows[0] < 0.6 * height or extent > 0.4 * height):
            return None  # not a dot at the baseline
        if i != 1 and extent < 0.7 * height:
            return None  # not a digit of full height
    x0, x1 = glyphs[0][0], glyphs[-1][1]
    above = gray[lines[-2][0] : lines[-2][1], x0:x1] if len(lines) > 1 else None
    digits = [
        cv2.resize(gray[top:bottom, a:b], GLYPH_SIZE, interpolation=cv2.INTER_AREA)
        for i, (a, b) in enumerate(glyphs)
        if i != 1
    ]
    return ValueLine(top, gray[top:bottom, x0:x1], above, digits)


def same_text(a: NDArray[np.uint8] | None, b: NDArray[np.uint8] | None) -> bool:
    """Return whether two crops of the terminal show the same text.

    Screen recorders re-encode only the parts of the screen that changed, so unchanged text comes
    back pixel for pixel, while a different digit moves dozens of pixels.
    """
    if a is None or b is None:
        return a is None and b is None
    return a.shape == b.shape and int((cv2.absdiff(a, b) > CHANGE_LEVEL).sum()) < SAME_TEXT_PIXELS


def pane_changed(
    gray: NDArray[np.uint8], previous: NDArray[np.uint8], min_pixels: int, min_width: int
) -> bool:
    """Return whether the terminal pane changed noticeably between two frames.

    A printed line changes hundreds of pixels across the width of several characters, the blinking
    cursor only a thin bar, so both a pixel count and a bounding box width are required.
    """
    changed = cv2.absdiff(gray, previous) > CHANGE_LEVEL
    if changed.sum() < min_pixels:
        return False
    columns = np.flatnonzero(changed.any(axis=0))
    return columns[-1] - columns[0] + 1 >= min_width


def scan_terminal(
    video: Path, regions: Regions, min_pixels: int, min_width: int
) -> list[TerminalEvent]:
    """Read the bottom line of the terminal after every noticeable change of the pane.

    Changes within COLLAPSE_WINDOW of each other belong to the same redraw. The pane is read
    READ_DELAY after the first change of a redraw, when the redraw is complete.
    """
    events: list[TerminalEvent] = []
    previous = None
    gray = None
    pending = None  # first change of a redraw whose pane has not been read yet
    next_report = 0.0
    for t, frame in iter_frames(video):
        rows, cols = regions.terminal_slice(frame.shape[1], frame.shape[0])
        gray = cv2.cvtColor(frame[rows, cols], cv2.COLOR_BGR2GRAY)
        if pending is not None and t >= pending + READ_DELAY:
            events.append(TerminalEvent(pending, parse_value_line(gray)))
            pending = None
        if previous is not None and pane_changed(gray, previous, min_pixels, min_width):
            last = pending if pending is not None else (events[-1].time if events else -np.inf)
            if t - last > COLLAPSE_WINDOW:
                pending = t
        previous = gray
        if t >= next_report:
            print(f"\rScanning terminal: {t:5.1f} s", end="", flush=True)
            next_report += 5.0
    if pending is not None and gray is not None:
        events.append(TerminalEvent(pending, parse_value_line(gray)))
    n_values = sum(event.line is not None for event in events)
    print(f"\rScanning terminal: done, {len(events)} redraws, {n_values} end in a value line")
    return events


def find_commands(events: list[TerminalEvent]) -> list[Command]:
    """Return the redraws at which a new command line appeared.

    A redraw shows a new command when its bottom line differs from the one at the previous command,
    in position or in text. Once the terminal scrolls, the position no longer changes and a value
    equal to the previous one only shows in the line above it, which then holds the previous value,
    so that line is compared as well. Other redraws, the cursor blinking or the pointer crossing the
    pane, show the same lines and are skipped.
    """
    commands: list[Command] = []
    for event in events:
        line = event.line
        if line is None:
            continue
        if commands:
            last = commands[-1].line
            if (
                line.top == last.top
                and same_text(line.image, last.image)
                and same_text(line.above, last.above)
            ):
                continue
        commands.append(Command(event.time, line))
    if not commands:
        raise RuntimeError(
            "No d.dd line found at the bottom of the terminal pane. Check that the terminal region "
            "covers the printed commands, or adjust --min-pixels and --min-width."
        )
    return commands


def glyph_similarity(a: NDArray, b: NDArray) -> float:
    """Return the normalized correlation of two glyph images, 1 for identical glyphs."""
    a = a.astype(np.float64) - a.mean()
    b = b.astype(np.float64) - b.mean()
    norm = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / norm) if norm > 0 else 0.0


@dataclass
class GlyphLibrary:
    """Digit glyphs grouped by appearance, each group labelled with its digit by the user.

    A terminal renders the same digit identically in every line, so glyphs of one digit correlate
    almost perfectly while different digits do not. Every group holds the running mean of its glyphs.
    """

    sums: list[NDArray[np.float64]] = field(default_factory=list)
    counts: list[int] = field(default_factory=list)
    labels: list[int | None] = field(default_factory=list)

    def prototype(self, group: int) -> NDArray[np.uint8]:
        """Return the mean glyph of a group."""
        return (self.sums[group] / self.counts[group]).round().clip(0, 255).astype(np.uint8)

    def match(self, glyph: NDArray[np.uint8]) -> int:
        """Return the group of a glyph, adding a new group if none matches."""
        scores = [glyph_similarity(glyph, self.prototype(k)) for k in range(len(self.sums))]
        if scores and max(scores) >= GLYPH_MATCH:
            group = int(np.argmax(scores))
            self.sums[group] += glyph
            self.counts[group] += 1
            return group
        self.sums.append(glyph.astype(np.float64))
        self.counts.append(1)
        self.labels.append(None)
        return len(self.sums) - 1

    def save(self, path: Path) -> None:
        """Write the groups with their labels, -1 marks an unlabelled group."""
        np.savez(
            path,
            prototypes=np.stack([self.prototype(k) for k in range(len(self.sums))]),
            counts=np.array(self.counts),
            labels=np.array([-1 if label is None else label for label in self.labels]),
        )

    @classmethod
    def load(cls, path: Path) -> GlyphLibrary:
        """Read the groups written by :meth:`save`."""
        data = np.load(path)
        counts = data["counts"].tolist()
        sums = [p.astype(np.float64) * n for p, n in zip(data["prototypes"], counts)]
        labels = [None if label < 0 else int(label) for label in data["labels"]]
        return cls(sums, counts, labels)


def label_glyphs(library: GlyphLibrary) -> None:
    """Ask the user for the digit of every glyph group that has no label yet."""
    if all(label is not None for label in library.labels):
        return
    window = "Digits: type 0-9 for the highlighted glyph, a/d move, Backspace clears, Enter accepts"
    print(
        "Type the digit shown by the highlighted glyph. a/d select the previous/next glyph, "
        "Backspace clears a label, Enter accepts once every glyph is labelled, Esc aborts."
    )
    n = len(library.sums)
    active = library.labels.index(None)
    pad = 8
    width, height = GLYPH_SIZE[0] * GLYPH_ZOOM, GLYPH_SIZE[1] * GLYPH_ZOOM
    while True:
        tiles = []
        for k in range(n):
            tile = np.zeros((height + 2 * pad + 64, width + 2 * pad, 3), dtype=np.uint8)
            glyph = cv2.resize(
                library.prototype(k), (width, height), interpolation=cv2.INTER_NEAREST
            )
            tile[pad : pad + height, pad : pad + width] = cv2.cvtColor(glyph, cv2.COLOR_GRAY2BGR)
            color = (0, 255, 255) if k == active else (255, 255, 255)
            corner = (width + 2 * pad - 3, height + 2 * pad - 3)
            cv2.rectangle(tile, (2, 2), corner, color, 2 if k == active else 1)
            label = "?" if library.labels[k] is None else str(library.labels[k])
            put_text(tile, label, (pad, height + 2 * pad + 34), color, VALUE_SCALE)
            put_text(tile, f"x{library.counts[k]}", (pad, height + 2 * pad + 58), (160, 160, 160))
            tiles.append(tile)
        cv2.imshow(window, np.hstack(tiles))
        key = cv2.waitKey(30) & 0xFF
        if ord("0") <= key <= ord("9"):
            library.labels[active] = key - ord("0")
            unlabelled = [k for k, label in enumerate(library.labels) if label is None]
            following = [k for k in unlabelled if k > active] or unlabelled or [active + 1]
            active = min(following[0], n - 1)
        elif key == ord("a"):
            active = max(active - 1, 0)
        elif key == ord("d"):
            active = min(active + 1, n - 1)
        elif key == 8:
            library.labels[active] = None
        elif key in (27, ord("q")):
            cv2.destroyAllWindows()
            raise RuntimeError("Digit labelling aborted")
        elif key in (13, 10) and all(label is not None for label in library.labels):
            break
    cv2.destroyAllWindows()


def read_commands(commands: list[Command], library: GlyphLibrary) -> None:
    """Group the digit glyphs of every command, ask for their labels and set the values."""
    for command in commands:
        command.groups = tuple(library.match(glyph) for glyph in command.line.glyphs)  # type: ignore[assignment]
    label_glyphs(library)
    for command in commands:
        digits = [library.labels[group] for group in command.groups]
        command.value = round(digits[0] + digits[1] / 10 + digits[2] / 100, 2)


def drop_redraws(commands: list[Command], period: float) -> list[Command]:
    """Drop commands that repeat the previous value within half a period.

    They are redraws of the same line that the text comparison did not catch, for example the
    pointer moving across the value.
    """
    kept: list[Command] = []
    for command in commands:
        if kept and command.value == kept[-1].value and command.time - kept[-1].time < period / 2:
            print(f"Warning: dropped a redraw of {command.value:.2f} at {command.time:.2f} s")
            continue
        kept.append(command)
    return kept


def step_ends(commands: list[Command], events: list[TerminalEvent], period: float) -> list[float]:
    """Return the time at which every command ends.

    A command ends when the next one is printed. When the next print is more than IRREGULAR_GAP
    periods away, the test was stopped or restarted in between, and the command ends at the first
    redraw after it (the message that stopped the test) or one period after the print if the pane
    did not change.
    """
    ends: list[float] = []
    times = [event.time for event in events]
    for command, following in zip_longest(commands, commands[1:]):
        if following is not None and following.time - command.time <= IRREGULAR_GAP * period:
            ends.append(following.time)
            continue
        later = [t for t in times if t > command.time + COLLAPSE_WINDOW]
        end = later[0] if later and later[0] <= command.time + IRREGULAR_GAP * period else None
        ends.append(command.time + period if end is None else end)
        print(
            f"Warning: no command follows {command.value:.2f} printed at {command.time:.2f} s "
            f"within {IRREGULAR_GAP * period:.1f} s, its step ends at {ends[-1]:.2f} s"
        )
    return ends


# endregion

# region Display decoding


def rectify(frame: NDArray[np.uint8], transform: NDArray[np.float32]) -> NDArray[np.uint8]:
    """Warp the display into the rectified digit frame, with a margin on all sides."""
    size = (DISPLAY_WIDTH + 2 * DISPLAY_PAD, DIGIT_HEIGHT + 2 * DISPLAY_PAD)
    return cv2.warpPerspective(frame, transform, size, flags=cv2.INTER_LINEAR)


def segment_boxes() -> NDArray[np.int64]:
    """Return the pixel boxes (x0, y0, x1, y1) of all segments, shape (DIGITS, 7, 4)."""
    pitch = DIGIT_WIDTH * (1 + DIGIT_GAP)
    boxes = np.zeros((DIGITS, 7, 4))
    for d in range(DIGITS):
        x0, y0 = DISPLAY_PAD + d * pitch, DISPLAY_PAD
        for s, (bx0, by0, bx1, by1) in enumerate(SEGMENT_BOXES):
            boxes[d, s] = (
                x0 + bx0 * DIGIT_WIDTH,
                y0 + by0 * DIGIT_HEIGHT,
                x0 + bx1 * DIGIT_WIDTH,
                y0 + by1 * DIGIT_HEIGHT,
            )
    return np.round(boxes).astype(np.int64)


def to_bw(display: NDArray[np.uint8], adjust: Adjust = Adjust()) -> NDArray[np.uint8]:
    """Convert the rectified display to black-and-white and apply contrast and brightness.

    The brightest colour channel serves as grey level: lit segments are dark in every channel, while
    the red housing and the green display stay bright, so neither is mistaken for a segment. The
    contrast is applied around the middle grey, then the brightness is added.
    """
    gray = display.max(axis=2).astype(np.float64)
    gray = adjust.contrast * (gray - 128) + 128 + adjust.brightness
    return gray.clip(0, 255).astype(np.uint8)


def segment_darkness(bw: NDArray[np.uint8]) -> NDArray[np.float64]:
    """Return how dark each segment is relative to its digit background, shape (DIGITS, 7).

    The darkness is close to zero for unlit segments and well above zero for lit ones.
    """
    bright = bw.astype(np.float64)
    boxes = segment_boxes()
    darkness = np.zeros((DIGITS, 7))
    for d in range(DIGITS):
        x0, x1 = boxes[d, :, 0].min(), boxes[d, :, 2].max()
        y0, y1 = boxes[d, :, 1].min(), boxes[d, :, 3].max()
        cell = bright[y0:y1, x0:x1].ravel()
        rank = int(0.9 * (cell.size - 1))  # 90th percentile, most of the cell is background
        background = max(np.partition(cell, rank)[rank], 1.0)
        for s, (bx0, by0, bx1, by1) in enumerate(boxes[d]):
            darkness[d, s] = (background - bright[by0:by1, bx0:bx1].mean()) / background
    return darkness


def split_segments(darkness: NDArray[np.float64]) -> tuple[float, float]:
    """Split the darkness values into lit and unlit segments.

    Returns the threshold that maximizes the between-class variance (Otsu) and the separation of
    the two class means.
    """
    values = np.sort(darkness.ravel())
    n = len(values)
    counts = np.arange(1, n)  # size of the low class for every split
    low_mean = np.cumsum(values)[:-1] / counts
    high_mean = (values.sum() - np.cumsum(values)[:-1]) / (n - counts)
    variances = counts * (n - counts) * (high_mean - low_mean) ** 2
    i = int(np.argmax(variances))
    return float((values[i] + values[i + 1]) / 2), float(high_mean[i] - low_mean[i])


def segments_value(on: NDArray[np.bool_]) -> int | None:
    """Return the number formed by the lit segments, None if a digit pattern is invalid."""
    value = 0
    seen_digit = False
    for pattern in map(tuple, on.astype(int)):
        if pattern == BLANK and not seen_digit:
            continue
        if pattern not in SEGMENT_PATTERNS:
            return None
        seen_digit = True
        value = value * 10 + SEGMENT_PATTERNS[pattern]
    return value if seen_digit else None


def decode_display(display: NDArray[np.uint8], adjust: Adjust = Adjust()) -> Decoded:
    """Decode the rectified display into a number.

    The display is converted to black-and-white with the given contrast and brightness, then the
    segments are split into lit and unlit at the darkness that separates them best. Leading digits
    without a single clearly lit segment are blank, their ambiguous segments are ignored. Values
    above the maximum are rejected.
    """
    bw = to_bw(display, adjust)
    darkness = segment_darkness(bw)
    threshold, separation = split_segments(darkness)
    on = darkness > threshold
    margin = LOW_CONFIDENCE * separation / 2
    ignored = np.zeros(DIGITS, dtype=bool)
    for d in range(DIGITS):
        if (darkness[d] > threshold + margin).any():
            break
        ignored[d] = True
        on[d] = False
    margins = np.abs(darkness - threshold)[~ignored]
    if separation < 0.15 or darkness.max() < 0.2 or margins.size == 0:
        return Decoded(None, 0.0, on, darkness, threshold, separation, bw, ignored)
    confidence = float(margins.min() / (separation / 2))
    value = segments_value(on)
    if value is not None and value > adjust.max_value:
        value = None
    return Decoded(value, confidence, on, darkness, threshold, separation, bw, ignored)


def force_segments(
    reading: Reading, forced: dict[tuple[int, int], bool], max_value: int
) -> Reading:
    """Return the reading with the given segments forced lit or unlit.

    The value follows the forced segments, a digit with a forced segment is no longer ignored, and
    the confidence only counts the segments that were left to the automatic split.
    """
    decoded = reading.decoded
    on = decoded.segments.copy()
    ignored = decoded.ignored.copy()
    free = np.ones_like(on)
    for (digit, segment), lit in forced.items():
        on[digit, segment] = lit
        free[digit, segment] = False
        ignored[digit] = False
    free &= ~ignored[:, None]
    margins = np.abs(decoded.darkness - decoded.threshold)[free]
    if decoded.separation < 0.15:
        confidence = 0.0
    elif margins.size == 0:
        confidence = 1.0
    else:
        confidence = float(margins.min() / (decoded.separation / 2))
    value = segments_value(on) if forced or decoded.value is not None else None
    if value is not None and value > max_value:
        value = None
    changed = decoded._replace(value=value, confidence=confidence, segments=on, ignored=ignored)
    return Reading(
        value, confidence, reading.n_frames, reading.n_agree, reading.image, changed, dict(forced)
    )


def read_step(displays: list[NDArray[np.uint8]], adjust: Adjust = Adjust()) -> Reading:
    """Read the display from the frames of one settle window.

    The thrust may still be rising in the window, so the latest value that is shown in at least two
    frames wins over the most frequent one. Single frames are ignored because the display may be
    caught mid-update.
    """
    decoded = [decode_display(display, adjust) for display in displays]
    values = [d.value for d in decoded if d.value is not None]
    if not values:
        best = max(range(len(decoded)), key=lambda i: decoded[i].confidence)
        return Reading(None, 0.0, len(displays), 0, displays[best], decoded[best])
    counts = Counter(values)
    stable = [value for value in reversed(values) if counts[value] >= 2]
    value = stable[0] if stable else counts.most_common(1)[0][0]
    # Pick the last frame that shows the value, it is closest to the next print.
    index = max(i for i, d in enumerate(decoded) if d.value == value)
    return Reading(
        value,
        decoded[index].confidence,
        len(displays),
        counts[value],
        displays[index],
        decoded[index],
    )


def track_display(
    frame: NDArray[np.uint8],
    template: NDArray[np.uint8],
    box: tuple[int, int, int, int],
    margin: int,
) -> tuple[int, int, float]:
    """Locate the display template in the frame near its original box.

    Returns the displacement of the display in pixels and the match score in [-1, 1]. Normalized
    cross-correlation is dominated by the housing and glass edges, so the digits changing does not
    disturb the match.
    """
    x, y, w, h = box
    x0, y0 = max(x - margin, 0), max(y - margin, 0)
    x1, y1 = min(x + w + margin, frame.shape[1]), min(y + h + margin, frame.shape[0])
    region = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    result = cv2.matchTemplate(region, gray, cv2.TM_CCOEFF_NORMED)
    _, score, _, (mx, my) = cv2.minMaxLoc(result)
    return x0 + mx - x, y0 + my - y, float(score)


def collect_displays(
    video: Path,
    regions: Regions,
    windows: list[tuple[float, float]],
    template: NDArray[np.uint8] | None,
    margin: int,
) -> list[list[NDArray[np.uint8]]]:
    """Return the rectified display of every frame inside each time window.

    When a template is given, the display is located in every frame first and the digit corners
    follow it, so the reading survives camera motion from the propeller wash. Screen recordings only
    store frames when the screen changes, so a window may contain no frame at all. The latest frame
    before the window then stands in, it shows the display as it was until the next change.
    """
    displays: list[list[NDArray[np.uint8]]] = [[] for _ in windows]
    shifts: list[tuple[int, int, float]] = []

    def rectified(frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        height, width = frame.shape[:2]
        offset = (0.0, 0.0)
        if template is not None:
            dx, dy, score = track_display(
                frame, template, regions.display_box(width, height), margin
            )
            shifts.append((dx, dy, score))
            offset = (dx, dy)
        return rectify(frame, regions.lcd_transform(width, height, offset))

    index = 0
    latest = None
    for t, frame in iter_frames(video):
        while index < len(windows) and t > windows[index][1]:
            if not displays[index] and latest is not None:
                displays[index].append(rectified(latest))
            index += 1
        if index == len(windows):
            break
        latest = frame
        if t >= windows[index][0]:
            displays[index].append(rectified(frame))
        print(f"\rReading display: step {min(index + 1, len(windows))}/{len(windows)}", end="")
    print()
    for frames in displays[index:]:  # the video ended inside or before the remaining windows
        if not frames and latest is not None:
            frames.append(rectified(latest))
    empty = [i for i, frames in enumerate(displays) if not frames]
    if empty:
        raise RuntimeError(f"No frames found in the settle windows of steps {empty}")
    if shifts:
        moves = np.array([(dx, dy) for dx, dy, _ in shifts])
        scores = np.array([score for _, _, score in shifts])
        print(
            f"Tracked the display in {len(shifts)} frames: it moved up to "
            f"{np.abs(moves).max():.0f} px, lowest match score {scores.min():.2f}"
            + (" (weak matches, check the review sheet)" if scores.min() < 0.5 else "")
        )
        if np.abs(moves).max() >= margin - 1:
            print("Warning: the display reached the edge of the search area, raise --track-margin")
    return displays


# endregion

# region Review and output


def put_text(
    image: NDArray[np.uint8],
    text: str,
    origin: tuple[int, int],
    color: tuple,
    scale: float = FONT_SCALE,
) -> int:
    """Draw a line of text in the review font and return its width in pixels."""
    thickness = 2 if scale > FONT_SCALE else 1
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
    return cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]


def draw_reading(
    reading: Reading, index: int, command: float, zoom: int = TILE_ZOOM
) -> NDArray[np.uint8]:
    """Render one review tile: the decoded black-and-white display, the segment boxes, the value.

    Lit segments are drawn green, unlit ones red. Segments whose darkness lies close to the
    lit/unlit threshold are drawn yellow, they are the ones that limit the confidence. Segments
    forced by the user are drawn with a thick line, ignored leading digits are grey.
    """
    bw = cv2.cvtColor(reading.decoded.image, cv2.COLOR_GRAY2BGR)
    image = cv2.resize(bw, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
    # The clicked digit block, the tile keeps a margin of DISPLAY_PAD pixels around it.
    corner = (DISPLAY_PAD * zoom, DISPLAY_PAD * zoom)
    opposite = ((DISPLAY_PAD + DISPLAY_WIDTH) * zoom, (DISPLAY_PAD + DIGIT_HEIGHT) * zoom)
    cv2.rectangle(image, corner, opposite, (255, 255, 255), 1)
    decoded = reading.decoded
    for d, digit in enumerate(segment_boxes()):
        for s, (x0, y0, x1, y1) in enumerate(digit):
            color = (0, 255, 0) if decoded.segments[d, s] else (0, 0, 255)
            forced = (d, s) in reading.forced
            margin = LOW_CONFIDENCE * max(decoded.separation, 1e-6) / 2
            if not forced and abs(decoded.darkness[d, s] - decoded.threshold) < margin:
                color = (0, 255, 255)
            if decoded.ignored[d]:
                color = (110, 110, 110)  # blank leading digit, its segments are ignored
            thickness = 2 if forced else 1
            cv2.rectangle(image, (x0 * zoom, y0 * zoom), (x1 * zoom, y1 * zoom), color, thickness)
    banner = np.zeros((BANNER_HEIGHT, image.shape[1], 3), dtype=np.uint8)
    uncertain = reading.value is None or reading.confidence < LOW_CONFIDENCE
    color = (0, 200, 255) if uncertain else (255, 255, 255)
    baseline = BANNER_HEIGHT - 12
    x = 6 + put_text(banner, f"{index:3d} cmd {command:.2f}: ", (6, baseline), color) + 6
    value = "?" if reading.value is None else f"{reading.value} g"
    x += put_text(banner, value, (x, baseline), color, VALUE_SCALE) + 12
    details = f"({reading.n_agree}/{reading.n_frames}, conf {reading.confidence:.2f})"
    put_text(banner, details, (x, baseline), color)
    return np.vstack((banner, image))


def review_sheet(
    readings: list[Reading], commands: list[float], first: int = 0, columns: int = COLUMNS
) -> NDArray[np.uint8]:
    """Arrange the review tiles of the given steps in a grid, numbered from ``first``."""
    tiles = [
        draw_reading(reading, first + i, command)
        for i, (reading, command) in enumerate(zip(readings, commands))
    ]
    blank = np.zeros_like(tiles[0])
    tiles += [blank] * (-len(tiles) % columns)
    rows = [np.hstack(tiles[i : i + columns]) for i in range(0, len(tiles), columns)]
    return np.vstack(rows)


def review_header(adjust: Adjust, width: int, page: str, notice: str = "") -> NDArray[np.uint8]:
    """Render the explanation lines above the review sheet, with an optional orange notice."""
    header = np.zeros((HEADER_HEIGHT, width, 3), dtype=np.uint8)
    lines = (
        f"Contrast {adjust.contrast:.2f}, brightness {adjust.brightness:+d}, max {adjust.max_value} g."
        "  Green = lit, red = unlit, yellow = ambiguous, grey = blank leading digit, "
        "thick = forced by you. Orange labels are flagged steps.",
        notice
        or "Click a box to force it: 1st click lit, 2nd click unlit, 3rd click back to automatic. "
        "Sliders adjust the black-and-white image. n/p change the page, Enter saves, q aborts.",
        page,
    )
    for i, line in enumerate(lines):
        color = (0, 200, 255) if i == 1 and notice else (255, 255, 255)
        put_text(header, line, (8, 26 + 30 * i), color)
    return header


def list_steps(steps: list[int], limit: int = 20) -> str:
    """Format step indices for a header line, abbreviating long lists."""
    if not steps:
        return "none"
    shown = " ".join(map(str, steps[:limit]))
    return shown if len(steps) <= limit else f"{shown} (+{len(steps) - limit} more)"


def locate_box(x: float, y: float, n_tiles: int) -> tuple[int, int, int] | None:
    """Map a click on the unscaled sheet (below the header) to (tile, digit, segment)."""
    tile_width = (DISPLAY_WIDTH + 2 * DISPLAY_PAD) * TILE_ZOOM
    tile_height = BANNER_HEIGHT + (DIGIT_HEIGHT + 2 * DISPLAY_PAD) * TILE_ZOOM
    column, row = int(x // tile_width), int(y // tile_height)
    tile = row * COLUMNS + column
    if not 0 <= column < COLUMNS or not 0 <= tile < n_tiles:
        return None
    local_x = x - column * tile_width
    local_y = y - row * tile_height - BANNER_HEIGHT
    slack = 3 * TILE_ZOOM  # thin bars are easier to hit with a little tolerance
    for d, digit in enumerate(segment_boxes() * TILE_ZOOM):
        for s, (x0, y0, x1, y1) in enumerate(digit):
            if x0 - slack <= local_x <= x1 + slack and y0 - slack <= local_y <= y1 + slack:
                return tile, d, s
    return None


def print_table(commands: list[float], readings: list[Reading]) -> None:
    """Print the readings of all steps."""
    print(f"{'idx':>3} {'cmd':>5} {'thrust':>7} {'agree':>7} {'conf':>5}")
    for i, (command, reading) in enumerate(zip(commands, readings)):
        value = "?" if reading.value is None else str(reading.value)
        uncertain = reading.value is None or reading.confidence < LOW_CONFIDENCE
        flag = "  <- check" if uncertain else ""
        forced = f"  {len(reading.forced)} forced" if reading.forced else ""
        print(
            f"{i:3d} {command:5.2f} {value:>7} {reading.n_agree:3d}/{reading.n_frames:<3d} "
            f"{reading.confidence:5.2f}{flag}{forced}"
        )


def review(
    displays: list[list[NDArray[np.uint8]]], commands: list[float], adjust: Adjust
) -> tuple[list[Reading], Adjust] | None:
    """Show the readings and let the user tune the image and force segments by clicking.

    The sheet is paged with TILES_PER_PAGE steps per page. Returns the accepted readings and the
    image adjustment, or None if the user aborts. Saving is refused while a step has no value, and
    a window closed with its close button reopens.
    """
    window = "Review"
    contrast_slider = "contrast x100"
    brightness_slider = "brightness +255"
    max_slider = "max thrust x10 g"
    forced: dict[int, dict[tuple[int, int], bool]] = {}
    clicks: list[tuple[int, int]] = []
    n_pages = -(-len(displays) // TILES_PER_PAGE)
    state = {"scale": 1.0, "page": 0}

    def on_mouse(event: int, px: int, py: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((px, py))

    def read_sliders() -> Adjust:
        contrast = max(cv2.getTrackbarPos(contrast_slider, window), 10) / 100
        brightness = cv2.getTrackbarPos(brightness_slider, window) - 255
        return Adjust(contrast, brightness, cv2.getTrackbarPos(max_slider, window) * 10)

    def page_slice() -> slice:
        start = state["page"] * TILES_PER_PAGE
        return slice(start, start + TILES_PER_PAGE)

    def show(readings: list[Reading], adjust: Adjust, notice: str = "") -> None:
        steps = page_slice()
        flagged = [
            i
            for i, reading in enumerate(readings)
            if reading.value is None or reading.confidence < LOW_CONFIDENCE
        ]
        page = f"Page {state['page'] + 1}/{n_pages}, steps {steps.start}-{steps.stop - 1}."
        page += f"  Flagged steps: {list_steps(flagged)}"
        sheet = review_sheet(readings[steps], commands[steps], steps.start)
        sheet = np.vstack((review_header(adjust, sheet.shape[1], page, notice), sheet))
        state["scale"] = preview_scale(sheet, (1800, 1000))
        scale = state["scale"]
        cv2.imshow(
            window, cv2.resize(sheet, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        )

    def open_window(adjust: Adjust) -> None:
        cv2.namedWindow(window)
        cv2.createTrackbar(
            contrast_slider, window, round(adjust.contrast * 100), 800, lambda v: None
        )
        cv2.createTrackbar(brightness_slider, window, adjust.brightness + 255, 510, lambda v: None)
        cv2.createTrackbar(max_slider, window, round(adjust.max_value / 10), 999, lambda v: None)
        cv2.setMouseCallback(window, on_mouse)

    open_window(adjust)
    shown = None
    readings: list[Reading] = []
    while True:
        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:  # closed with the X button
            open_window(shown or adjust)
            shown = None
        adjust = read_sliders()
        if adjust != shown:
            shown = adjust
            readings = [
                force_segments(read_step(frames, adjust), forced.get(i, {}), adjust.max_value)
                for i, frames in enumerate(displays)
            ]
            show(readings, adjust)
        while clicks:
            px, py = clicks.pop(0)
            steps = page_slice()
            n_tiles = len(readings[steps])
            hit = locate_box(px / state["scale"], py / state["scale"] - HEADER_HEIGHT, n_tiles)
            if hit is None:
                continue
            tile, digit, segment = hit
            step = steps.start + tile
            current = forced.setdefault(step, {}).get((digit, segment))
            if current is None:
                forced[step][(digit, segment)] = True
            elif current:
                forced[step][(digit, segment)] = False
            else:
                del forced[step][(digit, segment)]
            reading = read_step(displays[step], adjust)
            readings[step] = force_segments(reading, forced[step], adjust.max_value)
            show(readings, adjust)
        key = cv2.waitKey(50) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
            return None
        if key in (ord("n"), ord("p")):
            step = 1 if key == ord("n") else -1
            state["page"] = min(max(state["page"] + step, 0), n_pages - 1)
            show(readings, adjust)
        if key in (13, 10):
            missing = [i for i, reading in enumerate(readings) if reading.value is None]
            if missing:
                notice = f"Steps {list_steps(missing)} have no value yet, fix them before saving."
                show(readings, adjust, notice)
                continue
            cv2.destroyAllWindows()
            print_table(commands, readings)
            return readings, adjust


def write_csv(
    path: Path, commands: list[Command], readings: list[Reading], allow_missing: bool
) -> None:
    """Write the steps as ``pwm(normalized),thrust(g),time(s)`` rows.

    The first column keeps the name of the ramp csvs, ``fit_thrust_curve.py`` reads the first two
    columns. The time is the moment the command was printed in the recording.
    """
    missing = [i for i, reading in enumerate(readings) if reading.value is None]
    if missing and not allow_missing:
        raise RuntimeError(
            f"Steps {missing} have no value. Correct them in the review or pass --allow-missing."
        )
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("pwm(normalized)", "thrust(g)", "time(s)"))
        for command, reading in zip(commands, readings):
            value = "" if reading.value is None else reading.value
            writer.writerow((f"{command.value:.2f}", value, f"{command.time:.2f}"))
    print(f"Wrote {len(readings)} rows to {path}")


# endregion


def run(args: argparse.Namespace) -> None:
    """Run the extraction for one video."""
    video: Path = args.video
    sidecar = video.with_suffix(".rois.json")
    glyph_path = video.with_suffix(".glyphs.npz")
    output = args.output or video.with_suffix(".csv")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} exists, pass --overwrite to replace it")
    size = frame_size(video)

    regions = None
    if args.reuse_rois:
        regions = Regions.load(sidecar)
    else:
        frame = frame_at(video, args.preview_time)
        x, y, w, h = select_rectangle(frame, "1/2 Terminal output pane")
        regions = Regions((x / size[0], y / size[1], w / size[0], h / size[1]), [])

    events = scan_terminal(video, regions, args.min_pixels, args.min_width)
    commands = find_commands(events)
    library = GlyphLibrary()
    if args.reuse_rois and glyph_path.exists():
        library = GlyphLibrary.load(glyph_path)
    read_commands(commands, library)
    library.save(glyph_path)
    print(f"Saved the labelled digit glyphs to {glyph_path}")
    times = [command.time for command in commands]
    if len(commands) < 2 and not args.period:
        raise RuntimeError("Only one command found, pass --period to set the step length")
    period = args.period or float(np.median(np.diff(times)))
    commands = drop_redraws(commands, period)
    ends = step_ends(commands, events, period)
    values = [command.value for command in commands]
    print(
        f"Read {len(commands)} commands from {commands[0].time:.2f} s to {commands[-1].time:.2f} s, "
        f"one every {period:.2f} s:"
    )
    for i in range(0, len(values), 20):
        print("  " + " ".join(f"{value:.2f}" for value in values[i : i + 20]))

    template_path = video.with_suffix(".template.png")
    highest = int(np.argmax(values))
    if not regions.lcd_corners:
        regions.template_time = ends[highest] - args.margin
        frame = frame_at(video, regions.template_time)
        corners, (x, y, w, h) = select_corners(
            frame, "2/2 Thrust display", highest, values[highest]
        )
        regions.lcd_corners = [(px / size[0], py / size[1]) for px, py in corners]
        regions.display = (x / size[0], y / size[1], w / size[0], h / size[1])
        regions.save(sidecar, video, size)
        cv2.imwrite(str(template_path), frame[y : y + h, x : x + w])
        print(f"Saved regions to {sidecar} and the display template to {template_path}")
    template = None
    if args.track_margin > 0:
        if template_path.exists():
            template = cv2.imread(str(template_path))
        else:
            x, y, w, h = regions.display_box(*size)
            template = frame_at(video, regions.template_time)[y : y + h, x : x + w]

    windows = [(end - args.margin - args.settle_window, end - args.margin) for end in ends]
    displays = collect_displays(video, regions, windows, template, args.track_margin)
    adjust = Adjust(args.contrast, args.brightness, args.max_thrust)
    readings = [read_step(frames, adjust) for frames in displays]
    allow_missing = args.allow_missing
    if args.no_review:
        print_table(values, readings)
    else:
        accepted = review(displays, values, adjust)
        if accepted is None:
            print("Aborted, nothing written.")
            return
        readings, adjust = accepted
    if args.save_review:
        review_path = video.with_name(f"{video.stem}_review.png")
        cv2.imwrite(str(review_path), review_sheet(readings, values))
        print(f"Saved review sheet to {review_path}")
    if adjust != Adjust():
        print(
            f"Settings used: pass --contrast {adjust.contrast:.2f} --brightness {adjust.brightness} "
            f"--max-thrust {adjust.max_value} to reuse them"
        )
    write_csv(output, commands, readings, allow_missing)


def main() -> None:
    """Parse the arguments and run the extraction."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("video", type=Path, help="screen recording of the thrust test")
    parser.add_argument("--output", type=Path, help="csv path (default: <video>.csv)")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing csv")
    parser.add_argument(
        "--reuse-rois", action="store_true",
        help="use the regions in <video>.rois.json and the digit labels in <video>.glyphs.npz",
    )  # fmt: skip
    parser.add_argument(
        "--preview-time",
        type=float,
        default=30.0,
        help="time of the frame for selecting the terminal",
    )
    parser.add_argument(
        "--margin", type=float, default=0.05,
        help="seconds before the next print at which the settle window ends",
    )  # fmt: skip
    parser.add_argument(
        "--settle-window", type=float, default=0.5,
        help="length in seconds of the window before the next print whose frames are decoded",
    )  # fmt: skip
    parser.add_argument(
        "--period", type=float, default=0.0,
        help="seconds between two commands, 0 measures it as the median gap between the prints",
    )  # fmt: skip
    parser.add_argument(
        "--min-pixels", type=int, default=60,
        help="changed pixels in the terminal pane that count as a redraw",
    )  # fmt: skip
    parser.add_argument(
        "--min-width", type=int, default=15,
        help="minimum width in pixels of the changed terminal area, wider than the cursor",
    )  # fmt: skip
    parser.add_argument(
        "--track-margin", type=int, default=150,
        help="pixels around the display box searched for camera motion, 0 disables tracking",
    )  # fmt: skip
    parser.add_argument(
        "--contrast", type=float, default=1.0,
        help="contrast of the black-and-white display before decoding (1 = unchanged)",
    )  # fmt: skip
    parser.add_argument(
        "--brightness", type=int, default=0,
        help="grey level added to the black-and-white display before decoding",
    )  # fmt: skip
    parser.add_argument(
        "--max-thrust", type=int, default=1700,
        help="highest thrust in g accepted as a reading, larger values are rejected",
    )  # fmt: skip
    parser.add_argument("--no-review", action="store_true", help="write the csv without review")
    parser.add_argument(
        "--save-review", action="store_true", help="save the review sheet as <video>_review.png"
    )
    parser.add_argument(
        "--allow-missing", action="store_true", help="write empty cells for unreadable steps"
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
