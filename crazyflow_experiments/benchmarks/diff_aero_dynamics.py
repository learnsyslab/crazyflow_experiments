"""Benchmark diff-aero dynamics.

Run with
    pixi run -e diff-aero python crazyflow_experiments/benchmarks/diff_aero_dynamics.py
"""

import logging
import sys
import tomllib
from functools import partial
from pathlib import Path

_diff_aero_path = Path(__file__).parents[2] / "submodules"
assert (_diff_aero_path / "diffaero").exists(), f"diff-aero not found below: {_diff_aero_path}"
if str(_diff_aero_path) not in sys.path:
    sys.path.insert(0, str(_diff_aero_path))

import fire  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402
from diffaero.env import build_env  # noqa: E402
from diffaero.utils.randomizer import RandomizerManager  # noqa: E402
from numpy.typing import NDArray  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from crazyflow_experiments.benchmarks.common import (  # noqa: E402
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

logger = logging.getLogger(__name__)


def create_env_config(n_worlds: int, n_drones: int) -> DictConfig:
    """Create environment configuration for diff-aero."""
    cfg_name = "diff_aero_single_cfg.toml" if n_drones == 1 else "diff_aero_multi_cfg.toml"
    with open(Path(__file__).parent / cfg_name, "rb") as f:
        cfg = tomllib.load(f)
    cfg["n_envs"] = n_worlds
    cfg["dynamics"]["n_envs"] = n_worlds
    cfg["render"]["n_envs"] = n_worlds
    cfg["n_agents"] = n_drones
    cfg["render"]["n_agents"] = n_drones
    cfg["dynamics"]["n_agents"] = n_drones
    cfg["dt"] = 1 / CONTROL_FREQ
    cfg["dynamics"]["dt"] = 1 / CONTROL_FREQ
    cfg["dynamics"]["n_substeps"] = SIM_FREQ // CONTROL_FREQ
    return DictConfig(cfg)


def benchmark_diff_aero(
    n_worlds: int, n_drones: int, device: str, repeat: int = REPEAT, number: int = NUMBER
) -> NDArray:
    """Benchmark diff-aero."""
    logger.info(f"Benchmarking diff-aero on {device} with {n_worlds} worlds, {n_drones} drones")

    env = None
    action = None
    torch_device = None

    @torch.no_grad()
    def loop_fn(steps: int):
        for _ in range(steps):
            env.step(action)

    def warmup():
        nonlocal env, action, torch_device

        torch_device = torch.device(device)
        cfg = create_env_config(n_worlds, n_drones)
        # Diffaero has a bug where randomizers are not cleared between env builds because they are
        # stored in a class variable list. We clear them here to avoid accumulation across
        # benchmarks.
        RandomizerManager.randomizers.clear()
        env = build_env(cfg, device=torch_device)
        # Give slightly more thrust than needed to hover
        hover_thrust = 9.81 * 0.15 * 0.25
        action = torch.zeros(n_worlds, n_drones, env.action_dim, device=torch_device)
        action[..., 0] = hover_thrust * 1.05
        if n_drones == 1:
            action = action[:, 0, :]

        with torch.no_grad():
            env.reset()
        loop_fn(2)  # For some reason, diff-aero has another delay without this second warmup step

    try:
        return benchmark_function(warmup, loop_fn, repeat, number)
    finally:
        if env is not None:
            del env
            if torch_device is not None and torch_device.type == "cuda":
                torch.cuda.empty_cache()


def run_benchmarks(
    devices: list[str],
    world_range: list[int],
    drone_range: list[int],
    results_dir: Path = RESULTS_DIR,
    repeat: int = REPEAT,
    number: int = NUMBER,
):
    """Run full benchmark suite for the given devices, world, and drone ranges."""
    existing_keys = load_result_keys("diff_aero", results_dir)
    configs = [
        cfg
        for cfg in (
            {"n_worlds": w, "n_drones": d, "device": dev}
            for dev in devices
            for w in world_range
            for d in drone_range
        )
        if frozenset(cfg.items()) not in existing_keys
        and not check_parameters(cfg["n_worlds"], cfg["n_drones"], "diff-aero")
    ]
    data = run_sweep(
        configs,
        lambda cfg: run_isolated(benchmark_diff_aero, {**cfg, "repeat": repeat, "number": number}),
        on_result=partial(checkpoint, "diff_aero", results_dir),
    )
    if not data:
        logger.info("No new benchmark data collected.")
        return
    for d in data:
        d["timings"] = np.asarray(d["timings"], dtype=np.float64)
    save_results(pl.DataFrame(data), "diff_aero", results_dir)
    logger.info("Saved results.")


def main(
    device: str | None = None,
    n_worlds: int | None = None,
    n_drones: int | None = None,
    results_dir: str | Path = RESULTS_DIR,
    repeat: int = REPEAT,
    number: int = NUMBER,
):
    """Run diff-aero benchmark.

    Args:
        device: Device to benchmark ("cpu" or "cuda"), or None to sweep all available.
        n_worlds: Single world count to benchmark, or None to sweep all of WORLD_RANGE.
        n_drones: Single drone count to benchmark, or None to sweep all of DRONE_RANGE.
        results_dir: Directory the parquet results are read from and written to.
        repeat: Number of timing repetitions.
        number: Number of steps executed per repetition.
    """
    if device is not None:
        devices = [device]
    else:
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    world_range = [n_worlds] if n_worlds is not None else WORLD_RANGE
    drone_range = [n_drones] if n_drones is not None else DRONE_RANGE
    run_benchmarks(
        devices=devices,
        world_range=world_range,
        drone_range=drone_range,
        results_dir=Path(results_dir),
        repeat=repeat,
        number=number,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logger.setLevel(logging.INFO)
    fire.Fire(main)
