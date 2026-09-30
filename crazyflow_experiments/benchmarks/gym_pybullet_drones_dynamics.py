"""Benchmark gym-pybullet-drones dynamics.

Run with
    pixi run -e pybullet python crazyflow_experiments/benchmarks/gym_pybullet_drones_dynamics.py
"""

import logging
from functools import partial
from pathlib import Path

import fire
import gymnasium
import numpy as np
import polars as pl
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.utils.enums import DroneModel
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

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

logger = logging.getLogger(__name__)

RESULT_NAMES = {"attitude": "gym_pybullet_drones", "rotor_vel": "gym_pybullet_drones_rotor_level"}


def check_action_space(n_drones: int) -> tuple:
    """Check the action space of the environment."""
    if n_drones == 1:
        env = HoverAviary()
    else:
        env = MultiHoverAviary(num_drones=n_drones)

    action_space = env.action_space
    logger.info(f"Action space for {n_drones} drone(s): {action_space}")
    env.close()
    return action_space


def benchmark_gym_pybullet_drones(
    n_worlds: int,
    n_drones: int,
    device: str,  # noqa: ARG001 (device always cpu)
    control: str = "attitude",
    repeat: int = REPEAT,
    number: int = NUMBER,
) -> NDArray:
    """Benchmark gym-pybullet-drones."""
    logger.info(
        f"Benchmarking gym-pybullet-drones {control} with {n_worlds} worlds, {n_drones} drones"
    )

    env = None
    action = None
    rpms = None
    ctrls = None
    obs = None

    target_rpy_rates = np.zeros((3,))

    def attitude_loop_fn(steps: int):
        nonlocal obs
        for _ in range(steps):
            for i in range(n_worlds):
                for j in range(n_drones):
                    thrust = action[i, j, 3] if n_worlds > 1 else action[j, 3]
                    target_euler = action[i, j, :3] if n_worlds > 1 else action[j, :3]
                    t = 1 / CONTROL_FREQ
                    if n_worlds == 1:
                        quat = R.from_euler("XYZ", obs[0, 7:10]).as_quat()
                    else:
                        quat = R.from_euler("XYZ", obs[i, j, 7:10]).as_quat()

                    rpms[i, j, ...] = np.array(
                        ctrls[i][j]._dslPIDAttitudeControl(
                            t, thrust, quat, target_euler, target_rpy_rates
                        )
                    )

            obs, _, _, _, _ = env.step(rpms[0] if n_worlds == 1 else rpms)

    def rotor_vel_loop_fn(steps: int):
        nonlocal obs
        for _ in range(steps):
            obs, _, _, _, _ = env.step(action)

    loop_fn = attitude_loop_fn if control == "attitude" else rotor_vel_loop_fn
    hover = [0.0, 0.0, 0.0, 0.5] if control == "attitude" else [0.0, 0.0, 0.0, 0.0]

    def warmup():
        nonlocal env, action, rpms, ctrls, obs

        action = np.tile(hover, (n_drones, 1))
        rpms = np.zeros((n_worlds, n_drones, 4))
        ctrls = [[DSLPIDControl(DroneModel.CF2X) for _ in range(n_drones)] for _ in range(n_worlds)]
        if n_worlds == 1:
            if n_drones == 1:
                env = HoverAviary(pyb_freq=SIM_FREQ, ctrl_freq=CONTROL_FREQ)
            else:
                env = MultiHoverAviary(
                    num_drones=n_drones, pyb_freq=SIM_FREQ, ctrl_freq=CONTROL_FREQ
                )
        else:
            action = np.tile(action, (n_worlds, 1, 1))
            if n_drones == 1:
                env = gymnasium.vector.AsyncVectorEnv(
                    [
                        lambda: HoverAviary(pyb_freq=SIM_FREQ, ctrl_freq=CONTROL_FREQ)
                        for _ in range(n_worlds)
                    ]
                )
            else:
                env = gymnasium.vector.AsyncVectorEnv(
                    [
                        lambda: MultiHoverAviary(
                            num_drones=n_drones, pyb_freq=SIM_FREQ, ctrl_freq=CONTROL_FREQ
                        )
                        for _ in range(n_worlds)
                    ]
                )

        obs, _ = env.reset()
        obs, _, _, _, _ = env.step(action)

    try:
        return benchmark_function(warmup, loop_fn, repeat, number)
    finally:
        if env is not None:
            env.close()


def run_benchmarks(
    devices: list[str],
    world_range: list[int],
    drone_range: list[int],
    control: str = "attitude",
    results_dir: Path = RESULTS_DIR,
    repeat: int = REPEAT,
    number: int = NUMBER,
):
    """Run full benchmark suite for the given devices, world, and drone ranges."""
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
            benchmark_gym_pybullet_drones,
            {**cfg, "control": control, "repeat": repeat, "number": number},
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
    control: str = "attitude",
    results_dir: str | Path = RESULTS_DIR,
    repeat: int = REPEAT,
    number: int = NUMBER,
):
    """Run gym-pybullet-drones benchmark.

    Args:
        device: Device to benchmark (only "cpu" is supported), or None to use cpu.
        n_worlds: Single world count to benchmark, or None to sweep all of WORLD_RANGE.
        n_drones: Single drone count to benchmark, or None to sweep all of DRONE_RANGE.
        control: Control interface the timed step is driven through.
        results_dir: Directory the parquet results are read from and written to.
        repeat: Number of timing repetitions.
        number: Number of steps executed per repetition.
    """
    devices = [device] if device is not None else ["cpu"]
    world_range = [n_worlds] if n_worlds is not None else WORLD_RANGE
    drone_range = [n_drones] if n_drones is not None else DRONE_RANGE
    run_benchmarks(
        devices=devices,
        world_range=world_range,
        drone_range=drone_range,
        control=control,
        results_dir=Path(results_dir),
        repeat=repeat,
        number=number,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logger.setLevel(logging.INFO)
    fire.Fire(main)
