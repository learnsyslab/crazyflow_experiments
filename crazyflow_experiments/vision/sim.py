"""Fly the trained vision reach policy in simulation.

Run with
    pixi run -e vision python crazyflow_experiments/vision/sim.py
"""

from __future__ import annotations

import os

# Both must be set before scipy and jax are imported
os.environ["SCIPY_ARRAY_API"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import fire
import numpy as np
import toml
from crazyflow.control import Control
from crazyflow.sim import Sim
from crazyflow.sim.sensors.splat import build_render_splat_fn
from crazyflow.sim.splat import SplatViewer, attach_splats
from scipy.spatial.transform import Rotation as R
from splax.io import load_ply

from crazyflow_experiments.control.simulator import extract_obs
from crazyflow_experiments.control.takeoff import takeoff
from crazyflow_experiments.core.envs.vision_reach_env import (
    VisionReachEnv,
    mount_camera,
    set_inertial,
)
from crazyflow_experiments.vision.controller import ReachController, checkpoint
from crazyflow_experiments.vision.scenes import splat
from crazyflow_experiments.vision.viewer import add_drone, gray2rgb, update_drone_pose

if TYPE_CHECKING:
    from numpy.typing import NDArray

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

HOVER = (0.0, 0.0, 1.2)


def imu(sim: Sim) -> tuple[NDArray, NDArray]:
    """Gyroscope and accelerometer measurements."""
    rotation = R.from_quat(np.asarray(sim.data.states.quat[0, 0]))
    gravity = np.asarray(sim.data.params.gravity_vec)
    specific_acc = np.asarray(sim.data.states_deriv.acc[0, 0]) - gravity
    return np.asarray(sim.data.states.ang_vel[0, 0]), rotation.apply(specific_acc, inverse=True)


def main(
    policy: str = "attitude",
    scene: str = "lab",
    variant: str = "lab_30k",
    view_variant: str = "lab_30k",
    splat_view: bool = True,
    seed: int = 0,
    goals: int = 10,
    max_time: float = float("inf"),
    config: dict | None = None,
):
    """Take off and hand the drone to the trained reach policy.

    Args:
        policy: Run name of the checkpoint to fly.
        scene: Scene the policy's camera renders from.
        variant: Variant of the scene's placed splat the camera renders.
        view_variant: Variant the viewer shows.
        splat_view: Serve the splat viewer, with the drone and the policy's own camera.
        seed: Seed drawing the goals.
        goals: Goals to fly before stopping.
        max_time: Max flight time in s.
        config: Flight config. Read from config.toml if None.
    """
    if config is None:
        config = toml.load(Path(__file__).parent / "config.toml")
    ctrl_freq = config["controller"]["ctrl_freq"]
    render, fps = config["sim"]["render"], config["sim"]["fps"]
    source = splat(scene, variant)
    rng = np.random.default_rng(seed)

    control = Control.state if "state" in config["controller"]["type"] else Control.attitude
    sim = Sim(drone=config["drone"]["model"], control=control, device="gpu")
    mount_camera(sim, VisionReachEnv.CAM_FOVY)
    attach_splats(sim, scene=source)
    sim.data = set_inertial(sim.data, config["drone"]["mass"])
    sim.build_default_data()
    sim.reset()

    controller = ReachController(checkpoint(policy))
    assert controller.mode in config["controller"]["type"]
    goal = rng.uniform(controller.goal_low, controller.goal_high)
    render_splat = build_render_splat_fn(sim, resolution=controller.resolution)

    def frame() -> NDArray:
        """Grayscale image the onboard camera sees right now."""
        return np.asarray(render_splat(sim.data)[0, 0] @ VisionReachEnv.LUMINANCE)

    takeoff(sim, np.array(HOVER), 0, ctrl_freq, render=render)
    controller.reset(frame())
    if splat_view:
        shown = splat(scene, view_variant)
        viewer = SplatViewer(sim)
        if shown != source:
            # The viewer uploaded whatever the camera renders from, so swap it for the finer fit
            viewer.viewer.remove("scene")
            viewer.viewer.add_splats("scene", *load_ply(shown))
            logger.info(f"viewer showing {view_variant}, camera rendering {variant}")
        handles = add_drone(viewer.viewer.server.scene)
        marker = viewer.viewer.server.scene.add_icosphere(
            "goal", radius=controller.goal_radius, color=(255, 60, 60), opacity=0.4
        )
        view = viewer.viewer.server.gui.add_image(gray2rgb(frame()), label="Policy Input")
    target = np.array2string(goal, precision=2, floatmode="fixed")
    logger.info(f"flying {policy} to {target}")

    flown, step = 0, 0
    while flown < goals and step < max_time * ctrl_freq:
        obs = extract_obs(sim)
        obs["frame"], (obs["ang_vel"], obs["acc"]) = frame(), imu(sim)
        obs["goal"] = goal
        if "attitude" in config["controller"]["type"]:
            sim.attitude_control(controller(obs)[None, None, :])
        elif "state" in config["controller"]["type"]:
            sim.state_control(controller(obs)[None, None, :])
        sim.step(sim.freq // ctrl_freq)

        if controller.done(obs):
            flown += 1
            goal = rng.uniform(controller.goal_low, controller.goal_high)
            target = np.array2string(goal, precision=2, floatmode="fixed")
            logger.info(f"reached {flown}, flying to {target}")
        if splat_view:
            update_drone_pose(handles, sim.data)
            marker.position = tuple(goal)
            view.image = gray2rgb(obs["frame"])
        if render and (step * fps) % ctrl_freq < fps:  # Keep the frame rate within the loop
            sim.render()
        step += 1
    logger.info(f"stopped after {flown} goals")
    sim.close()


if __name__ == "__main__":
    fire.Fire(main)
