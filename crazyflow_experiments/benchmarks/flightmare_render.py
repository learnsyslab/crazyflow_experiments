"""Benchmark Flightmare Unity rendering throughput.

Flightmare uses a Unity player to render images, and connects to it over ZMQ. The Python bindings
do not expose this, so we use ZMQ directly.

Run with
    pixi run -e flightmare python crazyflow_experiments/benchmarks/flightmare_render.py
    pixi run -e flightmare python crazyflow_experiments/benchmarks/flightmare_render.py --render
"""

from __future__ import annotations

import ctypes
import itertools
import json
import logging
import os
import signal
import subprocess
import time
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import fire
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import zmq

from crazyflow_experiments.benchmarks.common import (
    NUMBER,
    REPEAT,
    RESOLUTIONS,
    RESULTS_DIR,
    WORLD_RANGE,
    benchmark_function,
    checkpoint,
    load_result_keys,
    run_isolated,
    run_sweep,
    save_results,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

UNITY_BINARY = (
    Path(os.environ["FLIGHTMARE_PATH"]) / "flightrender/RPG_Flightmare/RPG_Flightmare.x86_64"
)

if "DISPLAY" not in os.environ:  # The Unity player renders through OpenGL and needs an X display
    displays = sorted(Path("/tmp/.X11-unix").iterdir())
    assert displays, "Unity needs an X display to render"
    os.environ["DISPLAY"] = f":{displays[0].name[1:]}"

# flightlib binds both sockets and the Unity player connects to them
PUB_PORT = 10253
SUB_PORT = 10254
TIMEOUT = 60.0  # seconds to wait for the Unity connection and for each render
PR_SET_PDEATHSIG = 1

SCENE_ID = 1  # warehouse
DRONE_HEIGHT = 1.0  # metres above the warehouse floor, Unity is y-up
CAMERA = {
    "channels": 3,
    "fov": 90.0,
    "nearClipPlane": [0.01, 0.01, 0.01, 0.01],
    "farClipPlane": [1000.0, 100.0, 1000.0, 1000.0],
    "T_BC": np.eye(4).ravel().tolist(),
    "isDepth": False,
    "depthScale": 0.2,
    "outputIndex": 0,
}
MODES = ("rgb", "rgbd")


def _die_with_parent():
    """Kill the Unity player with the worker that launched it, so it cannot join the next worker."""
    ctypes.CDLL("libc.so.6").prctl(PR_SET_PDEATHSIG, signal.SIGKILL)


def _launch_unity(path: Path) -> subprocess.Popen:
    """Start the Unity player in batch mode."""
    return subprocess.Popen(
        [str(path), "-batchmode"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=_die_with_parent,
    )


def _vehicles(n_worlds: int, resolution: tuple[int, int], mode: str) -> list[dict]:
    """Describe one hovering vehicle with a forward-facing camera per world."""
    assert mode in MODES, f"Unknown render mode: {mode}"
    width, height = resolution
    layers = [mode == "rgbd", False, False]  # depth, segmentation, optical flow
    camera = {**CAMERA, "width": width, "height": height, "enabledLayers": layers}
    return [
        {
            "ID": f"quadrotor{i}",
            "position": [0.0, DRONE_HEIGHT, 0.0],
            "rotation": [0.0, 0.0, 0.0, 1.0],
            "size": [1.0, 1.0, 1.0],
            "cameras": [{**camera, "ID": f"quadrotor{i}_0"}],
            "lidars": [],
            "hasCollisionCheck": True,
        }
        for i in range(n_worlds)
    ]


def _connect(vehicles: list[dict]) -> tuple[zmq.Socket, zmq.Socket]:
    """Bind the bridge sockets and repeat the scene settings until Unity reports ready."""
    context = zmq.Context()
    pub, sub = context.socket(zmq.PUB), context.socket(zmq.SUB)
    pub.set_hwm(6)
    sub.set_hwm(6)
    pub.bind(f"tcp://*:{PUB_PORT}")
    sub.bind(f"tcp://*:{SUB_PORT}")
    sub.subscribe("")
    settings = json.dumps({"scene_id": SCENE_ID, "vehicles": vehicles, "objects": []}).encode()
    deadline = time.monotonic() + TIMEOUT
    ready = False
    while not ready:
        pub.send_multipart([b"Pose", settings])
        if sub.poll(200):  # The ready message is the only reply with a single key
            reply = json.loads(sub.recv_multipart()[0])
            ready = len(reply) == 1 and reply["ready"]
        if time.monotonic() > deadline:
            raise TimeoutError(f"Unity did not connect within {TIMEOUT} s")
    while sub.poll(0):  # Drop renders Unity produced while the settings were repeated
        sub.recv_multipart()
    return pub, sub


def _render(
    pub: zmq.Socket, sub: zmq.Socket, pose: dict, frame_id: int, resolution: tuple[int, int]
) -> list[NDArray]:
    """Publish the vehicle poses and return the images of the requested frame in wire order.

    Every camera yields an RGB uint8 image of shape [H, W, 3], followed by a float32 depth map of
    shape [H, W] when the depth layer is enabled.
    """
    pose["frame_id"] = frame_id
    pub.send_multipart([b"Pose", json.dumps(pose).encode()])
    while True:
        if not sub.poll(TIMEOUT * 1000):
            raise TimeoutError(f"Unity did not render frame {frame_id} within {TIMEOUT} s")
        frames = sub.recv_multipart(copy=False)
        if json.loads(frames[0].bytes)["frame_id"] == frame_id:
            break
    width, height = resolution
    n_layers = 1 + pose["vehicles"][0]["cameras"][0]["enabledLayers"][0]
    assert len(frames) == 1 + len(pose["vehicles"]) * n_layers, f"Unity sent {len(frames)} frames"
    return [
        np.frombuffer(frame.buffer, np.uint8).reshape(height, width, 3)
        if i % n_layers == 0
        else np.frombuffer(frame.buffer, np.float32).reshape(height, width)
        for i, frame in enumerate(frames[1:])
    ]


def benchmark_render(
    n_worlds: int,
    resolution: tuple[int, int],
    mode: str,
    repeat: int = REPEAT,
    number: int = NUMBER,
) -> NDArray:
    """Benchmark Unity rendering for the given world count, resolution, and mode.

    Args:
        n_worlds: Number of camera-equipped vehicles rendered per frame.
        resolution: Render resolution as (width, height).
        mode: Either ``"rgb"`` or ``"rgbd"``.
        repeat: Number of timing repetitions.
        number: Number of renders per repetition.

    Returns:
        Array of per-execution wall-clock times (seconds).
    """
    logger.info(f"Benchmarking {mode} Unity render with {n_worlds} worlds, resolution {resolution}")
    unity = _launch_unity(UNITY_BINARY)
    vehicles = _vehicles(n_worlds, resolution, mode)
    pub, sub = _connect(vehicles)
    pose = {"frame_id": 0, "vehicles": vehicles, "objects": []}
    frame_ids = itertools.count(1)

    def loop_fn(steps: int):
        for _ in range(steps):
            _render(pub, sub, pose, next(frame_ids), resolution)

    def warmup():
        loop_fn(1)

    timings = benchmark_function(warmup, loop_fn, repeat, number)
    unity.kill()
    return timings


def run_benchmarks(
    world_range: list[int],
    resolutions: list[tuple[int, int]],
    modes: list[str],
    results_dir: Path = RESULTS_DIR,
    repeat: int = REPEAT,
    number: int = NUMBER,
) -> pl.DataFrame:
    """Run all Unity render benchmarks and save results to parquet.

    Args:
        world_range: World counts to sweep.
        resolutions: Render resolutions to sweep.
        modes: Render modes to sweep.
        results_dir: Directory the parquet results are read from and written to.
        repeat: Number of timing repetitions.
        number: Number of renders per repetition.

    Returns:
        DataFrame with columns ``n_worlds``, ``resolution``, ``mode``, ``timings``.
    """
    existing_keys = load_result_keys("flightmare_render", results_dir)
    configs = [
        cfg
        for cfg in (
            {"n_worlds": w, "resolution": f"{r[0]}x{r[1]}", "mode": m}
            for m in modes
            for r in resolutions
            for w in world_range
        )
        if frozenset(cfg.items()) not in existing_keys
    ]
    data = run_sweep(
        configs,
        lambda cfg: run_isolated(
            benchmark_render,
            {
                **cfg,
                "resolution": tuple(int(x) for x in cfg["resolution"].split("x")),
                "repeat": repeat,
                "number": number,
            },
        ),
        on_result=partial(checkpoint, "flightmare_render", results_dir),
    )
    if not data:
        logger.info("No new benchmark data collected.")
        return pl.DataFrame()
    df = pl.DataFrame(data)
    save_results(df, "flightmare_render", results_dir)
    logger.info("Saved results.")
    return df


def print_summary(df: pl.DataFrame):
    """Print a human-readable benchmark statistics summary.

    Args:
        df: DataFrame with columns ``n_worlds``, ``resolution``, ``mode``, ``timings``.
    """
    summary = df.with_columns(
        [
            (pl.col("n_worlds") / pl.col("timings").list.mean()).alias("fps_mean"),
            (pl.col("n_worlds") / pl.col("timings").list.min()).alias("fps_best"),
            (pl.col("n_worlds") / pl.col("timings").list.max()).alias("fps_worst"),
            (pl.col("n_worlds") / pl.col("timings").list.eval(pl.element()))
            .list.std()
            .alias("fps_std"),
        ]
    ).select(["mode", "resolution", "n_worlds", "fps_mean", "fps_std", "fps_best", "fps_worst"])

    print("\n=== Flightmare Unity Render Benchmark Summary ===")
    for (mode, resolution), group in summary.group_by(["mode", "resolution"], maintain_order=True):
        print(f"\n[{mode.upper()} | {resolution}]")
        print(
            f"  {'n_worlds':>10}  {'fps_mean':>12}  {'fps_std':>10}"
            f"  {'fps_best':>12}  {'fps_worst':>12}"
        )
        for row in group.sort("n_worlds").iter_rows(named=True):
            print(
                f"  {row['n_worlds']:>10}  {row['fps_mean']:>12.1f}  {row['fps_std']:>10.1f}"
                f"  {row['fps_best']:>12.1f}  {row['fps_worst']:>12.1f}"
            )


def inspect_env(resolution: tuple[int, int], mode: str):
    """Show the image Unity renders from a single drone camera.

    Args:
        resolution: Render resolution as (width, height).
        mode: Either ``"rgb"`` or ``"rgbd"``.
    """
    unity = _launch_unity(UNITY_BINARY)
    vehicles = _vehicles(1, resolution, mode)
    pub, sub = _connect(vehicles)
    images = _render(pub, sub, {"frame_id": 0, "vehicles": vehicles, "objects": []}, 1, resolution)
    unity.kill()

    fig, axes = plt.subplots(1, len(images), figsize=(6 * len(images), 6), squeeze=False)
    # Unity's image origin is the lower left corner, so the rows are flipped for display
    axes[0][0].imshow(images[0][::-1])
    axes[0][0].set_title(f"Unity RGB ({resolution[0]}x{resolution[1]})")
    axes[0][0].axis("off")
    if mode == "rgbd":
        # flightlib scales the depth layer by 100 to convert it to metres
        depth = axes[0][1].imshow(images[1][::-1] * 100.0, cmap="viridis")
        fig.colorbar(depth, ax=axes[0][1], label="depth [m]")
        axes[0][1].set_title("Unity depth")
        axes[0][1].axis("off")
    plt.tight_layout()
    plt.show()


def main(
    render: bool = False,
    n_worlds: int | None = None,
    resolution: tuple[int, int] | None = None,
    mode: str | None = None,
    results_dir: str | Path = RESULTS_DIR,
    repeat: int = REPEAT,
    number: int = NUMBER,
):
    """Entry point for the Flightmare Unity render benchmark.

    Args:
        render: Display the rendered image instead of benchmarking.
        n_worlds: Single world count to benchmark, or None to sweep all of WORLD_RANGE.
        resolution: Single (width, height) resolution, or None to sweep all of RESOLUTIONS.
        mode: Single render mode ("rgb" or "rgbd"), or None to sweep both.
        results_dir: Directory the parquet results are read from and written to.
        repeat: Number of timing repetitions.
        number: Number of renders per repetition.
    """
    resolutions = [resolution] if resolution is not None else RESOLUTIONS
    modes = [mode] if mode is not None else list(MODES)
    if render:
        inspect_env(resolutions[0], modes[0])
        return
    world_range = [n_worlds] if n_worlds is not None else WORLD_RANGE
    df = run_benchmarks(world_range, resolutions, modes, Path(results_dir), repeat, number)
    assert not df.is_empty(), "No benchmark data collected, see the worker errors above"
    print_summary(df)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logger.setLevel(logging.INFO)
    fire.Fire(main)
