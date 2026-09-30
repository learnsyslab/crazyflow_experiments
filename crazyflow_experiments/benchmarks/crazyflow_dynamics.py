"""Benchmark crazyflow dynamics.

Run with
    pixi run -e crazyflow python crazyflow_experiments/benchmarks/crazyflow_dynamics.py
"""

from __future__ import annotations

import logging
import math
import os
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

# Preallocating the device leaves the autotuner no room to profile its kernels, which caps the
# feasible configs well below what the card holds. Must be set before jax is imported.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import crazyflow.sim.functional as F
import fire
import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
from crazyflow.control import Control
from crazyflow.control.transform import motor_force2rotor_vel
from crazyflow.dynamics import Dynamics, load_params
from crazyflow.sim import Sim
from crazyflow.sim.integration import Integrator
from crazyflow.sim.sharding import WORLD_AXIS, world_mesh
from jax.sharding import NamedSharding, PartitionSpec

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
    from crazyflow.sim.data import SimData
    from jax import Array
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

EIGEN_POOL_PREFIX = "tf_XLAEigen"  # Name XLA:CPU gives the threads of its intra-op pool
WORLDS_PER_DEVICE_RATIO = {"attitude": 256, "rotor_vel": 16}
CORE_RESIDENT_WORLDS = 4096  # Max worlds before cache misses become a problem
OVERSUBSCRIBED_WORLDS_PER_DEVICE_RATIO = 2048  # Worlds per core once we have more shards than cores
RESULT_NAMES = {"attitude": "crazyflow", "rotor_vel": "crazyflow_rotor_level"}


def _cpu_sharding(n_worlds: int, worlds_per_device_ratio: int) -> tuple[int, int]:
    """Resolve how many host devices to shard a world count over, and on how many CPUs."""
    cores = os.cpu_count() // 2  # SMT siblings cost more than they add
    devices = min(cores, max(1, math.isqrt(n_worlds // worlds_per_device_ratio)))
    if n_worlds > devices * CORE_RESIDENT_WORLDS:  # Shards get too big for core caches
        devices = min(n_worlds // OVERSUBSCRIBED_WORLDS_PER_DEVICE_RATIO, 2048)  # XLA limit
    return devices, max(2, min(cores, devices))


def configure_cpu(n_worlds: int, worlds_per_device_ratio: int) -> int:
    """Force XLA to use the specified number of CPU devices."""
    n_devices, n_cpus = _cpu_sharding(n_worlds, worlds_per_device_ratio)
    assert n_worlds % n_devices == 0, f"{n_worlds} worlds do not split over {n_devices} devices"
    os.sched_setaffinity(0, set(range(n_cpus)))
    flags = f"--xla_force_host_platform_device_count={n_devices}"
    flags += " --xla_cpu_use_fusion_emitters=false"
    os.environ["XLA_FLAGS"] = (f"{os.environ.get('XLA_FLAGS', '')} {flags}").strip()
    assert len(jax.devices("cpu")) == n_devices, "XLA was initialized before it was configured"
    return n_devices


def pin_worker_threads():
    """Give each thread of XLA's intra-op pool a CPU of its own.

    Returns without pinning if pools have more threads than CPUs.

    Note:
        XLA lazily creates the pool, so this must run after executing the step function once.
    """
    cpus = sorted(os.sched_getaffinity(0))
    threads = (t for t in Path("/proc/self/task").iterdir())
    threads = filter(lambda t: (t / "comm").read_text().startswith(EIGEN_POOL_PREFIX), threads)
    threads = sorted(threads, key=lambda t: t.name)
    if len(threads) > len(cpus):
        return
    for cpu, thread in zip(cpus, threads):
        os.sched_setaffinity(int(thread.name), {cpu})


def _hover_rotor_vel(drone: str) -> float:
    """Return the rotor velocity in RPM at which the drone hovers."""
    params = load_params(Dynamics.first_principles, drone)
    thrust = params["mass"] * -params["gravity_vec"][2] / 4
    return float(motor_force2rotor_vel(np.float64(thrust), np.array(params["rpm2thrust"])))


def benchmark_crazyflow(
    n_worlds: int,
    n_drones: int,
    device: str,
    control: Control = Control.attitude,
    repeat: int = REPEAT,
    number: int = NUMBER,
    use_xla_opts: bool = True,
    n_devices: int | None = None,
) -> NDArray:
    """Benchmark Crazyflow simulation.

    Args:
        n_worlds: Number of parallel worlds.
        n_drones: Number of drones per world.
        device: JAX device string ("cpu" or "gpu").
        control: Control interface the timed step is driven through.
        repeat: Number of timing repetitions.
        number: Number of steps executed per repetition.
        use_xla_opts: Shard the CPU run over host devices and apply the XLA flags.
        n_devices: Devices to shard the worlds over.

    Returns:
        Array of timing measurements (seconds per step).
    """
    logger.info(
        f"Benchmarking crazyflow {control} on {device} with {n_worlds} worlds, {n_drones} drones"
    )
    use_xla_opts = use_xla_opts and device == "cpu"
    if n_devices is None:
        ratio = WORLDS_PER_DEVICE_RATIO[control]
        n_devices = configure_cpu(n_worlds, ratio) if use_xla_opts else 1
    assert n_worlds % n_devices == 0, f"{n_worlds} worlds do not split over {n_devices} devices"
    devices = jax.devices(device)
    assert len(devices) >= n_devices, f"{n_devices} devices requested, {len(devices)} found"
    # Creating the sim with sharding up-front avoids allocating on one device at first.
    mesh = world_mesh(devices[:n_devices]) if n_devices > 1 else None
    if control == Control.attitude:
        sim = Sim(
            n_worlds=n_worlds,
            n_drones=n_drones,
            device=device,
            freq=SIM_FREQ,
            attitude_freq=CONTROL_FREQ,
            fused_mjx_model=True,
            mesh=mesh,
        )
        cmd = jnp.zeros((n_worlds, n_drones, 4), dtype=jnp.float32).at[..., 3].set(0.4)
        control_fn, substeps = F.attitude_control, sim.freq // sim.control_freq
    elif control == Control.rotor_vel:
        sim = Sim(
            n_worlds=n_worlds,
            n_drones=n_drones,
            control=Control.rotor_vel,
            integrator=Integrator.rk4,
            device=device,
            freq=CONTROL_FREQ,
            fused_mjx_model=True,
            mesh=mesh,
        )
        cmd = jnp.full((n_worlds, n_drones, 4), _hover_rotor_vel(sim.drone), dtype=jnp.float32)
        control_fn, substeps = F.rotor_vel_control, 1
    else:
        raise ValueError(f"Unknown control interface: {control}")
    sim_step = sim.build_step_fn()
    sim_reset = sim.build_reset_fn()
    if mesh is None:
        cmd = jax.device_put(cmd, devices[0])
    else:
        cmd = jax.device_put(cmd, NamedSharding(mesh, PartitionSpec(WORLD_AXIS)))
    sim_data, default_data = sim.data, sim.default_data

    @jax.jit
    def fori_loop(data: SimData, cmd: Array, num_steps: int) -> SimData:
        def single_step(_: int, data: SimData) -> SimData:
            return sim_step(control_fn(data, cmd), substeps)

        return jax.lax.fori_loop(0, num_steps, single_step, data)

    if use_xla_opts:
        jax.block_until_ready(fori_loop(sim_data, cmd, 1))  # The pool exists only after a run
        pin_worker_threads()

    def loop_fn(steps: int):
        jax.block_until_ready(fori_loop(sim_data, cmd, steps))

    def warmup():
        nonlocal sim_data
        sim_data = sim_reset(sim_data, default_data)
        loop_fn(1)  # Ensure JIT compiled dynamics

    return benchmark_function(warmup, loop_fn, repeat, number)


def run_benchmarks(
    devices: list[str],
    world_range: list[int],
    drone_range: list[int],
    control: Control = Control.attitude,
    results_dir: Path = RESULTS_DIR,
    repeat: int = REPEAT,
    number: int = NUMBER,
    use_xla_opts: bool = True,
):
    """Sweep over devices, world counts and drone counts, skipping configurations already on disk.

    Each configuration runs in an isolated process so that a crash does not terminate the parent.

    Args:
        devices: List of JAX device strings to benchmark (e.g. ["cpu", "gpu"]).
        world_range: List of world counts to sweep.
        drone_range: List of drone counts to sweep.
        control: Control interface the timed step is driven through.
        results_dir: Directory the parquet results are read from and written to.
        repeat: Number of timing repetitions.
        number: Number of steps executed per repetition.
        use_xla_opts: Shard the CPU run over host devices and apply the XLA flags.
    """
    assert control in ("attitude", "rotor_vel"), f"Unsupported control interface: {control}"
    name = RESULT_NAMES[control]
    existing_keys = load_result_keys(name, results_dir)
    configs = [
        cfg
        for cfg in (
            {"n_worlds": w, "n_drones": d, "device": dev}
            for dev in devices
            for w in world_range
            for d in drone_range
        )
        if frozenset(cfg.items()) not in existing_keys
        and not check_parameters(cfg["n_worlds"], cfg["n_drones"], name)
    ]
    data = run_sweep(
        configs,
        lambda cfg: run_isolated(
            benchmark_crazyflow,
            {
                **cfg,
                "control": control,
                "repeat": repeat,
                "number": number,
                "use_xla_opts": use_xla_opts,
            },
        ),
        on_result=partial(checkpoint, name, results_dir),
    )
    if not data:
        logger.info("No new benchmark data collected.")
        return
    save_results(pl.DataFrame(data), name, results_dir)
    logger.info("Saved results.")


def main(
    device: str | None = None,
    n_worlds: int | None = None,
    n_drones: int | None = None,
    control: Control = Control.attitude,
    results_dir: str | Path = RESULTS_DIR,
    repeat: int = REPEAT,
    number: int = NUMBER,
    use_xla_opts: bool = True,
):
    """Run Crazyflow benchmark.

    Args:
        device: JAX device to benchmark ("cpu" or "gpu").
        n_worlds: Single world count to benchmark, or None to sweep all of WORLD_RANGE.
        n_drones: Single drone count to benchmark, or None to sweep all of DRONE_RANGE.
        control: Control interface the timed step is driven through.
        results_dir: Directory the parquet results are read from and written to.
        repeat: Number of timing repetitions.
        number: Number of steps executed per repetition.
        use_xla_opts: Shard the CPU run over host devices and apply the XLA flags.
    """
    world_range = [n_worlds] if n_worlds is not None else WORLD_RANGE
    drone_range = [n_drones] if n_drones is not None else DRONE_RANGE
    devices = [device] if device is not None else ["cpu", "gpu"]
    run_benchmarks(
        devices, world_range, drone_range, control, Path(results_dir), repeat, number, use_xla_opts
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logger.setLevel(logging.INFO)
    fire.Fire(main)
