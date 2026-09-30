"""Benchmark Flightmare dynamics.

Flightmare does not have an attitude (rate) controller exposed in Python, so it's not directly
comparable to the other sims except for l2f.

Run with
    pixi run -e flightmare python crazyflow_experiments/benchmarks/flightmare_dynamics.py
"""

from __future__ import annotations

import logging
import os
import tempfile
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import fire
import numpy as np
import polars as pl
import yaml
from flightgym import QuadrotorEnv_v1

from crazyflow_experiments.benchmarks.common import (
    CONTROL_FREQ,
    NUMBER,
    REPEAT,
    RESULTS_DIR,
    WORLD_RANGE,
    benchmark_function,
    check_parameters,
    checkpoint,
    load_result_keys,
    run_isolated,
    run_sweep,
    save_results,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

QUADROTOR_CONFIG = Path(os.environ["FLIGHTMARE_PATH"]) / "flightlib/configs/quadrotor_env.yaml"


def benchmark_flightmare(
    n_worlds: int, device: str, repeat: int = REPEAT, number: int = NUMBER
) -> NDArray:
    """Benchmark Flightmare on the given device with n_worlds parallel environments."""
    logger.info(f"Benchmarking flightmare on {device} with {n_worlds} worlds")
    config = yaml.safe_load(QUADROTOR_CONFIG.read_text())
    config["quadrotor_env"]["sim_dt"] = 1 / CONTROL_FREQ

    # Each environment reads its quadrotor configuration from FLIGHTMARE_PATH
    with tempfile.TemporaryDirectory() as flightmare_path:
        config_dir = Path(flightmare_path) / "flightlib/configs"
        config_dir.mkdir(parents=True)
        with open(config_dir / "quadrotor_env.yaml", "w") as f:
            yaml.safe_dump(config, f)
        os.environ["FLIGHTMARE_PATH"] = flightmare_path
        vec_config = {
            "env": {
                "seed": 1,
                "scene_id": 0,
                "num_envs": n_worlds,
                "num_threads": os.cpu_count(),
                "render": False,
            }
        }
        env = QuadrotorEnv_v1(yaml.safe_dump(vec_config), False)

    n_obs, n_act = env.getObsDim(), env.getActDim()
    obs = np.zeros((n_worlds, n_obs), dtype=np.float32)
    reward = np.zeros((n_worlds, 1), dtype=np.float32)
    done = np.zeros((n_worlds, 1), dtype=bool)
    extra_info = np.zeros((n_worlds, len(env.getExtraInfoNames())), dtype=np.float32)
    # Flightmare centers its action range on hover, so zero actions hold the rotors at hover thrust
    action = np.zeros((n_worlds, n_act), dtype=np.float32)

    def loop_fn(steps: int):
        for _ in range(steps):
            env.step(action, obs, reward, done, extra_info)

    def warmup():
        env.reset(obs)
        loop_fn(1)  # The first step spawns the OpenMP thread pool

    return benchmark_function(warmup, loop_fn, repeat, number)


def run_benchmarks(
    devices: list[str],
    world_range: list[int],
    results_dir: Path = RESULTS_DIR,
    repeat: int = REPEAT,
    number: int = NUMBER,
):
    """Run full benchmark suite for the given devices and world range."""
    existing_keys = load_result_keys("flightmare", results_dir)
    configs = [
        cfg
        for cfg in (
            {"n_worlds": w, "n_drones": 1, "device": dev} for dev in devices for w in world_range
        )
        if frozenset(cfg.items()) not in existing_keys
        and not check_parameters(cfg["n_worlds"], 1, "flightmare")
    ]
    data = run_sweep(
        configs,
        lambda cfg: run_isolated(
            benchmark_flightmare,
            {
                "n_worlds": cfg["n_worlds"],
                "device": cfg["device"],
                "repeat": repeat,
                "number": number,
            },
        ),
        on_result=partial(checkpoint, "flightmare", results_dir),
    )
    if not data:
        logger.info("No new benchmark data collected.")
        return
    save_results(pl.DataFrame(data), "flightmare", results_dir)
    logger.info("Saved results.")


def main(
    device: str | None = None,
    n_worlds: int | None = None,
    results_dir: str | Path = RESULTS_DIR,
    repeat: int = REPEAT,
    number: int = NUMBER,
):
    """Run Flightmare benchmark.

    Args:
        device: Device to benchmark, or None to sweep all devices. Flightmare only runs on the CPU.
        n_worlds: Single world count to benchmark, or None to sweep all of WORLD_RANGE.
        results_dir: Directory the parquet results are read from and written to.
        repeat: Number of timing repetitions.
        number: Number of steps executed per repetition.
    """
    devices = [device] if device is not None else ["cpu"]
    world_range = [n_worlds] if n_worlds is not None else WORLD_RANGE
    run_benchmarks(
        devices=devices,
        world_range=world_range,
        results_dir=Path(results_dir),
        repeat=repeat,
        number=number,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logger.setLevel(logging.INFO)
    fire.Fire(main)
