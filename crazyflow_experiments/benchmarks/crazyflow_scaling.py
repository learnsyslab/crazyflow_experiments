"""Benchmark how Crazyflow scales over the accelerators of one node.

Run with
    pixi run -e crazyflow python crazyflow_experiments/benchmarks/crazyflow_scaling.py
"""

from __future__ import annotations

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

import logging
from functools import partial
from pathlib import Path

import fire
import jax
from crazyflow.control import Control

from crazyflow_experiments.benchmarks.common import (
    DRONE_RANGE,
    NUMBER,
    REPEAT,
    RESULTS_DIR,
    check_parameters,
    checkpoint,
    load_result_keys,
    run_isolated,
    run_sweep,
)
from crazyflow_experiments.benchmarks.crazyflow_dynamics import benchmark_crazyflow

logger = logging.getLogger(__name__)

DEVICE_RANGE = [1, 2, 4]  # devices the worlds are sharded over
WORLD_RANGE = [2**i for i in range(0, 31, 2)]
CONTROLS = [Control.attitude, Control.rotor_vel]
RESULT_NAMES = {"attitude": "crazyflow_scaling", "rotor_vel": "crazyflow_scaling_rotor_level"}


def run_benchmarks(
    device_range: list[int],
    world_range: list[int],
    drone_range: list[int],
    control: Control = Control.attitude,
    device: str = "gpu",
    results_dir: Path = RESULTS_DIR,
    repeat: int = REPEAT,
    number: int = NUMBER,
):
    """Sweep over device counts, world counts and drone counts for one control interface."""
    visible = len(jax.devices(device))
    if skipped := [n for n in device_range if n > visible]:
        logger.warning(f"Skipping {skipped} device counts, only {visible} {device} devices found")
    name = RESULT_NAMES[control]
    existing_keys = load_result_keys(name, results_dir)
    configs = [
        cfg
        for cfg in (
            {"n_worlds": w, "n_drones": d, "device": device, "n_devices": n}
            for n in device_range
            for w in world_range
            for d in drone_range
        )
        if cfg["n_devices"] <= visible
        and cfg["n_worlds"] % cfg["n_devices"] == 0
        and frozenset(cfg.items()) not in existing_keys
        and not check_parameters(cfg["n_worlds"], cfg["n_drones"], "crazyflow")
    ]
    data = run_sweep(
        configs,
        lambda cfg: run_isolated(
            benchmark_crazyflow,
            {**cfg, "control": control, "repeat": repeat, "number": number, "use_xla_opts": False},
        ),
        on_result=partial(checkpoint, name, results_dir),
    )
    if not data:
        logger.info(f"No new benchmark data collected for {control}.")
        return
    logger.info(f"Saved {len(data)} results for {control}.")


def main(
    n_devices: int | None = None,
    n_worlds: int | None = None,
    n_drones: int | None = None,
    control: Control | None = None,
    device: str = "gpu",
    results_dir: str | Path = RESULTS_DIR,
    repeat: int = REPEAT,
    number: int = NUMBER,
):
    """Run the Crazyflow multi-device scaling benchmark.

    Args:
        n_devices: Device count to benchmark, or None to sweep all of them.
        n_worlds: World count to benchmark, or None to sweep all of them.
        n_drones: Drone count to benchmark, or None to sweep all of them.
        control: Control interface the timed step is driven through, or None to sweep both.
        device: JAX device string the shards are placed on.
        results_dir: Directory the parquet results are read from and written to.
        repeat: Number of timing repetitions.
        number: Number of steps executed per repetition.
    """
    device_range = [n_devices] if n_devices is not None else DEVICE_RANGE
    world_range = [n_worlds] if n_worlds is not None else WORLD_RANGE
    drone_range = [n_drones] if n_drones is not None else DRONE_RANGE
    for ctrl in [control] if control is not None else CONTROLS:
        run_benchmarks(
            device_range, world_range, drone_range, ctrl, device, Path(results_dir), repeat, number
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(main)
