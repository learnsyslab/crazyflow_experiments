"""Fly the trained vision reach policy on the drone.

Run with
    pixi run -e vision python crazyflow_experiments/vision/deploy.py
"""

from __future__ import annotations

import os

os.environ["SCIPY_ARRAY_API"] = "1"

import logging
import pickle
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import fire
import numpy as np
import rclpy
import toml
from crazyflow.control import Control
from crazyflow.sim import Sim
from crazyflow.sim.sensors.splat import build_render_splat_fn
from crazyflow.sim.splat import SplatViewer, attach_splats
from scipy.spatial.transform import Rotation as R
from splax.io import load_ply

from crazyflow_experiments.control.config import TrajectoryConfig
from crazyflow_experiments.control.interface import control_interface
from crazyflow_experiments.control.loop import sleep_until_next_cycle
from crazyflow_experiments.control.mellinger_controller import MellingerController
from crazyflow_experiments.control.state_controller import StateController
from crazyflow_experiments.core.envs.vision_reach_env import VisionReachEnv, mount_camera
from crazyflow_experiments.deploy.experiment import connect
from crazyflow_experiments.deploy.takeoff import takeoff
from crazyflow_experiments.sim2real.utils import generate_filename_from_config
from crazyflow_experiments.utils.paths import RESULTS_DIR
from crazyflow_experiments.vision.controller import ReachController, checkpoint
from crazyflow_experiments.vision.ros.cam_viewer import CameraViewer
from crazyflow_experiments.vision.scenes import splat
from crazyflow_experiments.vision.viewer import add_drone, gray2rgb, update_drone_pose

if TYPE_CHECKING:
    from cflib.crazyflie.log import LogConfig
    from numpy.typing import NDArray

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

np.set_printoptions(precision=3, suppress=True, floatmode="fixed", linewidth=120)

# Camera calibration matched with the policy's training settings.
calibration = toml.load(Path(__file__).parent / "ros" / "cam_params.toml")
intrinsics = calibration["models"][calibration["model"]]
SENSOR = (calibration["calibration"]["width"], calibration["calibration"]["height"])
if calibration["model"] == "SIMPLE_RADIAL":
    FOCAL = (intrinsics["f"], intrinsics["f"])
else:
    FOCAL = (intrinsics["fx"], intrinsics["fy"])
CENTRE = (intrinsics["cx"], intrinsics["cy"])


def policy_frame(image: NDArray[np.uint8], resolution: tuple[int, int]) -> NDArray[np.floating]:
    """Reduce one camera frame to the grayscale image the policy reads."""
    squeezed = cv2.resize(image, SENSOR, interpolation=cv2.INTER_AREA)
    fov = VisionReachEnv.CAM_FOVY
    width, height = (round(2.0 * f * np.tan(np.deg2rad(fov) / 2.0)) for f in FOCAL)
    left, top = round(CENTRE[0] - width / 2), round(CENTRE[1] - height / 2)
    cropped = np.asarray(squeezed[top : top + height, left : left + width], np.float32) / 255.0
    grey = cropped @ np.asarray(VisionReachEnv.LUMINANCE)
    small = cv2.resize(grey, resolution, interpolation=cv2.INTER_AREA)
    return small.astype(np.float32)


def crazyflie_imu_conversion(obs: dict[str, float]) -> dict[str, float]:
    """Convert the raw IMU readings from the Crazyflie to SI units."""
    converted_obs = {}
    converted_obs["pos"] = np.array([obs[f"stateEstimate.{a}"] for a in "xyz"], np.float32)
    converted_obs["acc"] = 9.81 * np.array([obs[f"acc.{axis}"] for axis in "xyz"], np.float32)
    rpy = np.array([obs[f"stateEstimate.{a}"] for a in ("roll", "pitch", "yaw")], np.float32)
    converted_obs["rpy"] = np.deg2rad(rpy)
    ang_vel = np.array([obs[f"gyro.{axis}"] for axis in "xyz"], np.float32)
    converted_obs["ang_vel"] = np.deg2rad(ang_vel)
    return converted_obs


def main(
    policy: str = "attitude",
    goals: int = 3,
    seed: int = 0,
    render: bool = False,
    calibrate: bool = True,
):
    """Take off and hand the drone to the controller the config names.

    Args:
        policy: Run name of the checkpoint to fly.
        goals: Goals the policy reaches before coming home.
        seed: Seed drawing the goals.
        render: Show the splat viewer.
        calibrate: Calibrate the drone during takeoff.
    """
    folder = Path(__file__).parent
    config = toml.load(folder / "config.toml")

    ctrl_type = config["controller"]["type"]
    flies_policy = "policy" in ctrl_type
    if flies_policy:
        controller = ReachController(checkpoint(policy))
    resolution = controller.resolution if flies_policy else VisionReachEnv.RESOLUTION

    if config["experiment"]["deploy_sitl"] or render:
        sim = Sim(control="state", device="gpu")
        mount_camera(sim, VisionReachEnv.CAM_FOVY)
        attach_splats(sim, scene=splat("lab", "lab_30k"))
        render_rgb = build_render_splat_fn(sim, resolution=resolution)

    if render:
        shown = splat("lab", "lab_30k")
        assert shown.exists(), f"no splat at {shown}"
        viewer = SplatViewer(sim)
        # The viewer uploaded whatever the camera renders from, so swap it for the finer fit
        viewer.viewer.remove("scene")
        viewer.viewer.add_splats("scene", *load_ply(shown))
        handles = add_drone(viewer.viewer.server.scene)
        marker = viewer.viewer.server.scene.add_icosphere(
            "goal", radius=0.2, color=(255, 60, 60), opacity=0.4
        )
        frame = np.asarray(render_rgb(sim.data)[0, 0] @ VisionReachEnv.LUMINANCE)
        view = viewer.viewer.server.gui.add_image(gray2rgb(frame), label="Policy Input")

    trajectory_config = TrajectoryConfig.from_config(config)
    trajectory = trajectory_config.build()
    pos_start, yaw_start = trajectory_config.start_pose(trajectory)

    data = []
    log_variables = ["acc.x", "acc.y", "acc.z", "gyro.x", "gyro.y", "gyro.z"]
    log_variables += [
        "stateEstimate.x",
        "stateEstimate.y",
        "stateEstimate.z",
        "stateEstimate.roll",
        "stateEstimate.pitch",
        "stateEstimate.yaw",
    ]
    obs_onboard = {k: 0.0 for k in log_variables}

    def cf_log_callback(_timestamp: float, data: dict[str, float], _logconf: LogConfig) -> None:
        """Publish one cflib IMU log sample using the host's system timestamp."""
        for k, v in data.items():
            obs_onboard[k] = v

    # Set up initial info
    ctrl_freq = config["controller"]["ctrl_freq"]
    ctrl_interface: Control = control_interface(ctrl_type, onboard=True)
    rng = np.random.default_rng(seed)
    # Prepare controllers
    if flies_policy:
        goal = rng.uniform(controller.goal_low, controller.goal_high)
    elif "mellinger" in ctrl_type:
        initial_obs = {
            "pos": np.zeros(3),
            "quat": np.zeros(4),
            "vel": np.zeros(3),
            "ang_vel": np.zeros(3),
            "rpy": np.zeros(3),
            "t": time.perf_counter(),
        }
        if ctrl_interface == Control.state:
            controller = StateController(initial_obs, trajectory)
        else:
            controller = MellingerController(initial_obs, trajectory, config["drone"]["model"])
    else:
        raise ValueError(f"Cannot deploy the controller type {ctrl_type}")

    # Deploy reuses this ROS context. The bound method has the callback
    # signature expected by cflib's LogConfig.
    rclpy.init()
    camera_viewer = CameraViewer(headless=True)
    camera_spin_thread = threading.Thread(
        target=rclpy.spin, args=(camera_viewer,), name="camera_viewer_ros", daemon=True
    )
    camera_spin_thread.start()

    # Set up deploy env
    model = config["drone"]["model"]
    if "cf" not in model:
        raise NotImplementedError(f"Deployment for {model} not implemented")
    drone = connect(config, ctrl_interface, log_variables, cf_log_callback)
    # Update drone parameters to match heavier drone
    drone.cf.param.set_value("ctrlMel.mass", config["drone"]["mass"])
    drone.cf.param.set_value("ctrlMel.massThrust", config["drone"]["mass_thrust"])
    assert camera_viewer.get_latest_image() is not None, "camera viewer delivered no frame"

    try:
        takeoff(drone, pos_start, yaw_start, calibrate=calibrate)
        if flies_policy:
            # Seed the controller frame buffer on resets
            if config["experiment"]["deploy_sitl"]:
                controller.reset(render_rgb(sim.data)[0, 0] @ VisionReachEnv.LUMINANCE)
            else:
                controller.reset(policy_frame(camera_viewer.get_latest_image(), resolution))

            target = np.array2string(goal, precision=2, floatmode="fixed")
            logger.info(f"flying {policy} to {target}")
        else:
            controller._t_start = time.perf_counter()  # Reset start time
        flown = 0
        t_start = time.perf_counter()
        while rclpy.ok():
            t = time.perf_counter()
            obs = drone.obs()
            obs["cam"] = camera_viewer.get_latest_image()
            obs["t"] = t

            if obs["is_outdated"]:
                logger.info("Measurements outdated. Emergency stop.")
                break

            if flies_policy:
                if render or config["experiment"]["deploy_sitl"]:
                    sim.data = sim.data.replace(
                        states=sim.data.states.replace(
                            pos=obs["pos"][None, None, :], quat=obs["quat"][None, None, :]
                        )
                    )
                if config["experiment"]["deploy_sitl"]:
                    obs["frame"] = render_rgb(sim.data)[0, 0] @ VisionReachEnv.LUMINANCE
                else:
                    obs["frame"] = policy_frame(obs["cam"], resolution)
                if render:
                    update_drone_pose(handles, sim.data)
                    marker.position = tuple(goal)
                    view.image = gray2rgb(obs["frame"])
                obs["goal"] = goal
                print(f"vicon pos {obs['pos']}, vicon rpy {obs['rpy']}")
                obs["pos_vicon"] = obs["pos"]
                obs["quat_vicon"] = obs["quat"]
                obs = obs | crazyflie_imu_conversion(obs_onboard)
                print(f"onboard pos {obs['pos']}, onboard rpy {obs['rpy']}")

            action = controller(obs)

            if ctrl_interface == Control.state and flies_policy:
                action_hw = np.zeros(16)
                action_hw[:3] = action[:3]
                action_hw[3:5] = action[3:5] * 0.1
                action_hw[9:13] = action[9:13]
                action = action_hw
            drone.send_action(action)

            obs["u"] = action
            data.append(obs.copy())

            if controller.done(drone.obs() | {"goal": goal}):
                if not flies_policy:
                    logger.info("Controller terminated. Returning home.")
                    break
                flown += 1
                if flown == goals:
                    logger.info(f"Reached {flown} goals. Returning home.")
                    break
                goal = rng.uniform(controller.goal_low, controller.goal_high)
                target = np.array2string(goal, precision=2, floatmode="fixed")
                logger.info(f"Reached {flown}, flying on to {target}")

            if ctrl_interface == Control.state:
                if t - t_start < 1.0 or not flies_policy:
                    drone.update_estimators()  # Only for absolute controllers needed
                else:  # Policy supplies own estimate
                    drone.update_estimators(controller.pos, controller.quat)

            print(
                f"cam pos {controller.pos}, cam rpy {R.from_quat(controller.quat).as_euler('xyz')}"
            )

            sleep_until_next_cycle(t, ctrl_freq)

        logger.info("Landing...")
        drone.update_estimators()  # do a single update to make sure return home works
        time.sleep(0.05)
        drone.update_estimators()
        drone.return_to_start()  # Since there is no estimate, we will just land

        logger.info("Storing data...")

    finally:
        try:
            drone.close()  # Kill the drone
            # Store the data to a file with a unique name
            counter = 0
            while (
                filepath := (
                    RESULTS_DIR
                    / "vision"
                    / f"{generate_filename_from_config(config, True)}_{counter:02d}.pkl"
                )
            ).exists():
                counter += 1
                if counter > 99:
                    raise FileExistsError("Too many files with the same name exist already.")
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, "wb") as f:
                pickle.dump(data, f)
        finally:
            camera_spin_thread.join(timeout=1.0)
            camera_viewer.stop()
            camera_viewer.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    fire.Fire(main)
