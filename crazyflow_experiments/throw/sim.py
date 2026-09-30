"""Fly the throw recovery policy in simulation, training it from a simulated throw.

Run with
    pixi run -e throw python crazyflow_experiments/throw/sim.py
"""

import logging
import os
import time
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["SCIPY_ARRAY_API"] = "1"

import fire
import numpy as np
import toml
from crazyflow.control import Control
from crazyflow.dynamics import load_params
from crazyflow.sim import Dynamics, Sim

from crazyflow_experiments.control.simulator import extract_obs
from crazyflow_experiments.core.envs.throw_env import (
    SUCCESS_ANG_VEL_TOL,
    SUCCESS_VEL_TOL,
    SUCCESS_Z,
    SUCCESS_Z_TOL,
    thrust2rotor_vel,
)
from crazyflow_experiments.throw.controller import ThrowController

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main(render: bool = False) -> None:
    """Main deployment loop.

    Args:
        render: Enable/disable rendering the simulation.
    """
    folder = Path(__file__).parent
    config = toml.load(folder / "config.toml")

    throw_controller = ThrowController()
    throw_controller.warmup_jit()

    # Set up deploy env
    if "cf" not in config["drone"]["model"]:
        raise NotImplementedError(f"Deployment for {config['drone']['model']} not implemented")
    sim = Sim(
        n_worlds=1,
        n_drones=1,
        drone=config["drone"]["model"],
        dynamics="first_principles",
        control=Control.rotor_vel,
        freq=500,
        attitude_freq=500,
        device="cpu",
    )
    sim.reset()
    rpm2thrust = load_params(Dynamics.first_principles, sim.drone)["rpm2thrust"]

    ctrl_freq = config["controller"]["ctrl_freq"]
    fps = config["sim"]["fps"]
    logger.info("Drone ready the be thrown!")

    try:
        tstart = time.perf_counter()
        throw_controller.train()
        tend = time.perf_counter()
        logger.info(f"Training finished in {tend - tstart:.3f}s.")

        # Simulate a thow for the amount of time it took to train
        sim.data = sim.data.replace(
            states=sim.data.states.replace(vel=sim.data.states.vel.at[0, 0].set([0.1, 0.1, 7.0]))
        )
        for _ in range(int((tend - tstart) * 50)):  # 50 FPS
            sim.step(sim.freq // 50)
            if render:
                sim.render()
            time.sleep(0.02)

        for i in range(200):  # Max 4 seconds after throw
            obs = extract_obs(sim)

            action = throw_controller(obs)  # Thrust per rotor
            sim.rotor_vel_control(thrust2rotor_vel(action, rpm2thrust)[None, None, :])
            sim.step(sim.freq // ctrl_freq)

            if (
                abs(obs["pos"][2] - SUCCESS_Z) < SUCCESS_Z_TOL
                and np.linalg.norm(obs["vel"]) < SUCCESS_VEL_TOL
                and np.linalg.norm(obs["ang_vel"]) < SUCCESS_ANG_VEL_TOL
            ):
                logger.info("Drone has stabilized. Returning home.")
                break

            if render and (i * fps) % ctrl_freq < fps:  # Keep frame rate within control loop
                sim.render()

    finally:
        sim.close()


if __name__ == "__main__":
    fire.Fire(main)
