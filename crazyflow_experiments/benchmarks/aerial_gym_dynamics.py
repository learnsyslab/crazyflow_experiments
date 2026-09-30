"""Benchmark aerial-gym dynamics.

Run with
    pixi run -e aerial-gym python crazyflow_experiments/benchmarks/aerial_gym_dynamics.py
"""

from __future__ import annotations

import logging
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import fire
import polars as pl
from aerial_gym.config.sim_config.base_sim_config import BaseSimConfig
from aerial_gym.registry.sim_registry import sim_config_registry
from aerial_gym.sim.sim_builder import SimBuilder  # Must be imported before torch

from crazyflow_experiments.benchmarks.common import (
    CONTROL_FREQ,
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

import torch  # isort: skip, must be after aerial_gym import

if TYPE_CHECKING:
    from numpy.typing import NDArray


logger = logging.getLogger(__name__)

RESULT_NAMES = {"attitude": "aerial_gym", "rotor_vel": "aerial_gym_rotor_level"}


class BenchmarkSimParamsCPU(BaseSimConfig):
    """CPU config with matching physics and control frequencies."""

    class sim(BaseSimConfig.sim):
        dt = 1 / CONTROL_FREQ
        substeps = SIM_FREQ // CONTROL_FREQ
        use_gpu_pipeline = False

    class viewer(BaseSimConfig.viewer):
        headless = True


class BenchmarkSimParamsGPU(BaseSimConfig):
    """GPU config with matching physics and control frequencies."""

    class sim(BaseSimConfig.sim):
        dt = 1 / CONTROL_FREQ
        substeps = SIM_FREQ // CONTROL_FREQ
        use_gpu_pipeline = True

    class viewer(BaseSimConfig.viewer):
        headless = True


sim_config_registry.register("benchmark_sim_cpu", BenchmarkSimParamsCPU)
sim_config_registry.register("benchmark_sim_gpu", BenchmarkSimParamsGPU)


def benchmark_aerial_gym(
    n_worlds: int,
    device: str,
    control: str = "attitude",
    repeat: int = REPEAT,
    number: int = NUMBER,
) -> NDArray:
    """Benchmark aerial-gym."""
    logger.info(f"Benchmarking aerial-gym {control} on {device} with {n_worlds} worlds")

    torch_device = device if device == "cpu" else f"{device}:0"
    sim_name = "benchmark_sim_cpu" if device == "cpu" else "benchmark_sim_gpu"

    env = SimBuilder().build_env(
        sim_name=sim_name,
        env_name="empty_env",
        robot_name="base_quadrotor",
        controller_name="lee_attitude_control" if control == "attitude" else "no_control",
        args=None,
        device=torch_device,
        num_envs=n_worlds,
        headless=True,
        use_warp=True,
    )
    if control == "attitude":
        actions = torch.zeros((n_worlds, 4), device=torch_device)
        actions[..., 0] = 1.1
    else:
        actions = torch.full((n_worlds, 4), 0.25 * 9.81 / 4, device=torch_device)

    def loop_fn(steps: int):
        for _ in range(steps):
            env.step(actions=actions)

    def warmup():
        env.reset()
        loop_fn(2)  # Ensure warmup

    return benchmark_function(warmup, loop_fn, repeat, number)


def run_benchmarks(
    world_range: list[int],
    devices: list[str],
    control: str = "attitude",
    results_dir: Path = RESULTS_DIR,
    repeat: int = REPEAT,
    number: int = NUMBER,
):
    """Run full benchmark suite for the given world and device ranges."""
    assert control in ("attitude", "rotor_vel"), f"Unsupported control interface: {control}"
    name = RESULT_NAMES[control]
    existing_keys = load_result_keys(name, results_dir)
    configs = [
        cfg
        for cfg in (
            {"n_worlds": w, "n_drones": 1, "device": dev} for dev in devices for w in world_range
        )
        if frozenset(cfg.items()) not in existing_keys
        and not check_parameters(cfg["n_worlds"], 1, name)
    ]
    data = run_sweep(
        configs,
        lambda cfg: run_isolated(
            benchmark_aerial_gym,
            {
                "n_worlds": cfg["n_worlds"],
                "device": cfg["device"],
                "control": control,
                "repeat": repeat,
                "number": number,
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
    control: str = "attitude",
    results_dir: str | Path = RESULTS_DIR,
    repeat: int = REPEAT,
    number: int = NUMBER,
):
    """Run aerial-gym benchmark.

    Args:
        device: Device to benchmark ("cpu" or "cuda"), or None to sweep both.
        n_worlds: Single world count to benchmark, or None to sweep all of WORLD_RANGE.
        control: Control interface the timed step is driven through.
        results_dir: Directory the parquet results are read from and written to.
        repeat: Number of timing repetitions.
        number: Number of steps executed per repetition.
    """
    devices = [device] if device is not None else ["cpu", "cuda"]
    world_range = [n_worlds] if n_worlds is not None else WORLD_RANGE
    run_benchmarks(
        world_range=world_range,
        devices=devices,
        control=control,
        results_dir=Path(results_dir),
        repeat=repeat,
        number=number,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logger.setLevel(logging.INFO)
    fire.Fire(main)
