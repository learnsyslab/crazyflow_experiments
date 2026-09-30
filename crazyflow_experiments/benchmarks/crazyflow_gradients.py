"""Benchmark Crazyflow gradient computation throughput.

Run with
    pixi run -e crazyflow python crazyflow_experiments/benchmarks/crazyflow_gradients.py
"""

from __future__ import annotations

import logging
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import crazyflow.sim.functional as F
import fire
import jax
import jax.numpy as jnp
import polars as pl
from crazyflow.sim import Sim

from crazyflow_experiments.benchmarks.common import (
    CONTROL_FREQ,
    DRONE_RANGE,
    NUMBER,
    REPEAT,
    RESULTS_DIR,
    SIM_FREQ,
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
    from jax import Array
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def benchmark_crazyflow_grad(
    n_worlds: int, n_drones: int, device: str, repeat: int = REPEAT, number: int = NUMBER
) -> NDArray:
    """Benchmark gradient throughput through 10 simulation steps of the distance to [0, 0, 1].

    Args:
        n_worlds: Number of parallel worlds.
        n_drones: Number of drones per world.
        device: JAX device string ("cpu" or "gpu").
        repeat: Number of timing repetitions.
        number: Number of gradient computations per repetition.

    Returns:
        Array of timing measurements (seconds per gradient computation).
    """
    logger.info(
        f"Benchmarking crazyflow_grad on {device} with {n_worlds} worlds, {n_drones} drones"
    )
    sim = Sim(
        n_worlds=n_worlds,
        n_drones=n_drones,
        device=device,
        freq=SIM_FREQ,
        attitude_freq=CONTROL_FREQ,
        fused_mjx_model=True,
    )
    # Ensure we don't get zero gradients from clipping positions on hitting the floor
    sim.data = sim.data.replace(
        states=sim.data.states.replace(pos=sim.data.states.pos.at[..., 2].set(2.0))
    )
    sim.build_default_data()
    sim_step = sim.build_step_fn()
    data = sim.data
    jax_device = jax.devices(device)[0]
    cmd = jnp.zeros((n_worlds, n_drones, 4), dtype=jnp.float32, device=jax_device)
    cmd = cmd.at[..., 3].set(sim.data.params.mass[0] * 9.81 * 1.5)
    target = jnp.array([0.0, 0.0, 1.0], device=jax_device)

    def loss_fn(action: jax.Array, sim_data: jax.Array) -> jax.Array:
        sim_data = F.attitude_control(sim_data, action)
        sim_data = sim_step(sim_data, n_steps=10)
        return jnp.mean(jnp.sqrt(jnp.sum((sim_data.states.pos - target) ** 2, axis=-1)))

    @jax.jit
    def fori_loop(num_steps: int) -> Array:
        def single_step(_: int, grad: Array) -> Array:
            return jax.grad(loss_fn, argnums=0)(cmd - 0.01 * grad, data)

        return jax.lax.fori_loop(0, num_steps, single_step, jnp.zeros_like(cmd))

    def loop_fn(steps: int):
        jax.block_until_ready(fori_loop(steps))

    def warmup():
        grad = jax.block_until_ready(fori_loop(1))  # Ensure JIT compiled dynamics
        assert not jnp.any(jnp.isnan(grad)), f"Gradient contains NaN on warmup: {grad}"
        assert not jnp.all(grad == 0), f"Gradient is all zeros on warmup: {grad}"

    return benchmark_function(warmup, loop_fn, repeat, number)


def run_benchmarks(
    devices: list[str],
    world_range: list[int],
    drone_range: list[int],
    results_dir: Path = RESULTS_DIR,
    repeat: int = REPEAT,
    number: int = NUMBER,
):
    """Sweep over devices x world_range x drone_range and collect timing data.

    Each configuration is run in an isolated spawned process so that a crash (e.g. OOM) does not
    terminate the parent. Loads existing results from disk and skips already-completed configs,
    merging new results before saving.

    Args:
        devices: List of JAX device strings to benchmark (e.g. ["cpu", "gpu"]).
        world_range: List of world counts to sweep.
        drone_range: List of drone counts to sweep.
        results_dir: Directory the parquet results are read from and written to.
        repeat: Number of timing repetitions.
        number: Number of gradient computations per repetition.
    """
    existing_keys = load_result_keys("crazyflow_grad", results_dir)
    configs = [
        cfg
        for cfg in (
            {"n_worlds": w, "n_drones": d, "device": dev}
            for dev in devices
            for w in world_range
            for d in drone_range
        )
        if frozenset(cfg.items()) not in existing_keys
        and not check_parameters(cfg["n_worlds"], cfg["n_drones"], "crazyflow_grad")
    ]
    data = run_sweep(
        configs,
        lambda cfg: run_isolated(
            benchmark_crazyflow_grad, {**cfg, "repeat": repeat, "number": number}
        ),
        on_result=partial(checkpoint, "crazyflow_grad", results_dir),
    )
    if not data:
        logger.info("No new benchmark data collected.")
        return
    save_results(pl.DataFrame(data), "crazyflow_grad", results_dir)
    logger.info("Saved results.")


def main(
    device: str | None = "gpu",
    n_worlds: int | None = None,
    n_drones: int | None = 1,
    results_dir: str | Path = RESULTS_DIR,
    repeat: int = REPEAT,
    number: int = NUMBER,
):
    """Run Crazyflow gradient benchmark.

    Args:
        device: JAX device to benchmark ("cpu" or "gpu").
        n_worlds: Single world count to benchmark, or None to sweep all of WORLD_RANGE.
        n_drones: Single drone count to benchmark, or None to sweep all of DRONE_RANGE.
        results_dir: Directory the parquet results are read from and written to.
        repeat: Number of timing repetitions.
        number: Number of gradient computations per repetition.
    """
    world_range = [n_worlds] if n_worlds is not None else WORLD_RANGE
    drone_range = [n_drones] if n_drones is not None else DRONE_RANGE
    devices = [device] if device is not None else ["cpu", "gpu"]
    run_benchmarks(devices, world_range, drone_range, Path(results_dir), repeat, number)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logger.setLevel(logging.INFO)
    fire.Fire(main)
