"""Race the drone through the track in simulation.

Run with
    pixi run -e racing python crazyflow_experiments/racing/sim.py
    pixi run -e racing python crazyflow_experiments/racing/sim.py --render False
"""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

import fire
import gymnasium
import numpy as np
from crazyflow.sim.visualize import draw_line
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy
from lsy_drone_racing.utils import load_config, load_controller

if TYPE_CHECKING:
    from lsy_drone_racing.control.controller import Controller
    from lsy_drone_racing.envs.drone_race import DroneRaceEnv


logger = logging.getLogger(__name__)


def simulate(
    config: str = "track.toml",
    controller: str | None = None,
    n_runs: int = 1,
    render: bool | None = None,
) -> list[float]:
    """Evaluate the drone controller over multiple episodes.

    Args:
        config: The path to the configuration file. Assumes the file is in `config/`.
        controller: The name of the controller file in `racing/control/` or None. If None,
            the controller specified in the config file is used.
        n_runs: The number of episodes.
        render: Enable/disable rendering the simulation.

    Returns:
        A list of episode times.
    """
    # Load configuration and check if firmare should be used.
    config_path = Path(__file__).parents[0] / "config" / config
    config = load_config(config_path)
    config.sim.xml_path = str(config_path.parent / config.sim.xml_path)
    if render is None:
        render = config.sim.render
    else:
        config.sim.render = render
    # Load the controller module
    control_path = Path(__file__).parents[0] / "control"
    controller_path = control_path / (controller or config.controller.file)
    controller_cls = load_controller(controller_path)  # This returns a class, not an instance
    # Create the racing environment
    env: DroneRaceEnv = gymnasium.make(
        config.env.id,
        freq=config.env.freq,
        sim_config=config.sim,
        sensor_range=config.env.sensor_range,
        control_mode=config.env.control_mode,
        track=config.env.track,
        disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"),
        seed=config.env.seed,
    )
    env = JaxToNumpy(env)

    ep_times = []
    for _ in range(n_runs):  # Run n_runs episodes with the controller
        obs, info = env.reset()
        controller: Controller = controller_cls(obs, info, config)
        i = 0
        fps = 60

        # Visualizations
        trail = deque([], maxlen=25)

        while True:
            curr_time = i / config.env.freq

            action = controller.compute_control(obs, info)

            obs, reward, terminated, truncated, info = env.step(action)
            trail.append(obs["pos"])
            # Update the controller internal state and models.
            controller_finished = controller.step_callback(
                action, obs, reward, terminated, truncated, info
            )
            # Add up reward, collisions
            if terminated or truncated:
                logger.info(f"{terminated=}, {truncated=}, {controller_finished=}")
                break
            if config.sim.render:  # Render the sim if selected.
                draw_line(
                    env.unwrapped.sim,
                    np.array(trail),
                    rgba=np.array([1, 0.4, 0.0, 0.4]),
                    start_size=0.1,
                    end_size=10,
                )
                if ((i * fps) % config.env.freq) < fps:
                    env.unwrapped.sim.render(
                        mode=config.sim.render_mode,
                        camera=env.unwrapped.settings.camera,
                        cam_config=env.unwrapped.settings.cam_config,
                        width=3840,
                        height=2160,
                    )
            i += 1

        controller.episode_callback()  # Update the controller internal state and models.
        log_episode_stats(obs, curr_time)
        controller.episode_reset()
        ep_times.append(
            curr_time if obs["n_gates_passed"] == obs["gate_sequence"].shape[0] else None
        )

    # Close the environment
    env.close()
    return ep_times


def log_episode_stats(obs: dict, curr_time: float):
    """Log the statistics of a single episode."""
    gates_passed = obs["n_gates_passed"]
    finished = gates_passed == obs["gate_sequence"].shape[0]
    logger.info(
        f"Flight time (s): {curr_time}\nFinished: {finished}\nGates passed: {gates_passed}\n"
    )


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger("lsy_drone_racing").setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    fire.Fire(simulate, serialize=lambda _: None)
