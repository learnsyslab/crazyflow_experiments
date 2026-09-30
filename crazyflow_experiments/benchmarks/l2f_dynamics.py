"""Benchmark l2f dynamics.

l2f commands the rotors directly, so its step integrates the dynamics without an attitude
controller, and it compiles one vectorized environment per world count out of a fixed set of sizes.
The missing attitude controller makes l2f hard to compare to other benchmarks.

Run with
    pixi run -e l2f python crazyflow_experiments/benchmarks/l2f_dynamics.py
"""

import json
import logging
from functools import partial
from pathlib import Path

import fire
import l2f
import numpy as np
import polars as pl
from numpy.typing import NDArray

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

logger = logging.getLogger(__name__)


def benchmark_l2f(
    n_worlds: int, device: str, repeat: int = REPEAT, number: int = NUMBER
) -> NDArray:
    """Benchmark l2f on the given device with n_worlds parallel environments."""
    logger.info(f"Benchmarking l2f on {device} with {n_worlds} worlds")
    vector = l2f.vector_selector[n_worlds]
    l2f_device = l2f.Device()
    rng = vector.VectorRng()
    env = vector.VectorEnvironment()
    params = vector.VectorParameters()
    state, next_state = vector.VectorState(), vector.VectorState()

    vector.initialize_rng(l2f_device, rng, 0)
    vector.initialize_environment(l2f_device, env)
    vector.initial_parameters(l2f_device, env, params)

    # l2f compiles its integration step in and exposes no parameter fields, so we rewrite it
    # through the parameter json to match the control frequency of the other benchmarks
    config = json.loads(
        l2f.parameters_to_json(l2f_device, env.environments[0], params.parameters[0])
    )
    config["integration"]["dt"] = 1 / CONTROL_FREQ
    hover_throttle = config["dynamics"]["hovering_throttle_relative"]
    config = json.dumps(config)
    for environment, parameter in zip(env.environments, params.parameters):
        l2f.parameters_from_json(l2f_device, environment, config, parameter)
    written = l2f.parameters_to_json(l2f_device, env.environments[0], params.parameters[0])
    assert json.loads(written)["integration"]["dt"] == 1 / CONTROL_FREQ

    # l2f drives the rotors directly, and its actions are the throttles normalized to [-1, 1]
    action = np.full((n_worlds, 4), 2 * hover_throttle - 1, dtype=np.float32)

    def loop_fn(steps: int):
        nonlocal state, next_state
        for _ in range(steps):
            vector.step(l2f_device, env, params, state, action, next_state, rng)
            state, next_state = next_state, state

    def warmup():
        vector.initial_state(l2f_device, env, params, state)
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
    existing_keys = load_result_keys("l2f", results_dir)
    configs = [
        cfg
        for cfg in (
            {"n_worlds": w, "n_drones": 1, "device": dev} for dev in devices for w in world_range
        )
        if frozenset(cfg.items()) not in existing_keys
        and not check_parameters(cfg["n_worlds"], 1, "l2f")
    ]
    data = run_sweep(
        configs,
        lambda cfg: run_isolated(
            benchmark_l2f,
            {
                "n_worlds": cfg["n_worlds"],
                "device": cfg["device"],
                "repeat": repeat,
                "number": number,
            },
        ),
        on_result=partial(checkpoint, "l2f", results_dir),
    )
    if not data:
        logger.info("No new benchmark data collected.")
        return
    save_results(pl.DataFrame(data), "l2f", results_dir)
    logger.info("Saved results.")


def main(
    device: str | None = None,
    n_worlds: int | None = None,
    results_dir: str | Path = RESULTS_DIR,
    repeat: int = REPEAT,
    number: int = NUMBER,
):
    """Run l2f benchmark.

    Args:
        device: Device to benchmark, or None to sweep all devices. l2f only runs on the CPU.
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
