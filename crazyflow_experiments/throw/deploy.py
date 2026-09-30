"""Deploy the throw recovery policy on the real drone, training mid-air after the throw.

Run with
    pixi run -e throw python crazyflow_experiments/throw/deploy.py
"""

import logging
import os
import pickle
import threading
import time
from datetime import datetime
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["SCIPY_ARRAY_API"] = "1"  # TODO remove once scipy uses the array API by default

import jax
import numpy as np
import rclpy
import toml
from crazyflow.control import Control
from scipy.spatial.transform import Rotation as R

from crazyflow_experiments.control.loop import sleep_until_next_cycle
from crazyflow_experiments.deploy.experiment import connect
from crazyflow_experiments.throw import (
    CEILING_MIN_TIME,
    FLIGHT_BOUNDS_X,
    FLIGHT_BOUNDS_Y,
    FLIGHT_BOUNDS_Z,
    RECORDINGS_DIR,
    STABILIZED_MIN_TIME,
    STABILIZED_VEL_Z_TOL,
)
from crazyflow_experiments.throw.controller import IDLE_NORMALIZED_FORCE, ThrowController

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger("cflib").setLevel(logging.WARNING)

MAX_RETURN_TILT = np.deg2rad(60.0)  # Tilted further, the drone cannot fly home and stops instead


def main():
    """Main deployment loop."""
    folder = Path(__file__).parent
    config = toml.load(folder / "config.toml")

    # Set up initial info
    initial_info = {
        "drone_model": config["drone"]["model"],
        "use_x_start": config["controller"]["use_x_start"],
        "use_simple_throw_detection": config["controller"]["use_simple_throw_detection"],
    }

    throw_controller = ThrowController(use_x_start=initial_info["use_x_start"])
    throw_controller.warmup_jit()
    if initial_info["use_simple_throw_detection"]:
        assert not initial_info["use_x_start"], (
            "Simple throw detection requires use_x_start to be False"
        )

    # Set up deploy env
    if "cf" not in config["drone"]["model"]:
        raise NotImplementedError(f"Deployment for {config['drone']['model']} not implemented")
    drone = connect(config, Control.rotor_vel)

    # Precompute param IDs before flight for fast mid-flight transfer
    throw_controller.precompute_param_ids(drone.cf)
    logger.info("Drone ready the be thrown!")

    drone.cf.commander.send_stop_setpoint()
    trained_event = threading.Event()
    collect_obs_event = threading.Event()
    lock = threading.Lock()

    def spin_drone():
        """Keeps sending commands to the drone such that it doesn't time out."""
        drone.cf.param.set_value(
            "stabilizer.controller", 6
        )  # rotor_vel interface, the rotors idle at the action the policy starts from
        while not trained_event.is_set():
            drone.update_estimators()
            drone.cf.commander.send_position_setpoint(*(IDLE_NORMALIZED_FORCE,) * 4)
            time.sleep(0.1)

    threading.Thread(target=spin_drone, daemon=True).start()

    def collect_obs(state: str):
        """Collects observations from the drone for logging and analysis."""
        while not collect_obs_event.is_set():
            obs = drone.obs()
            obs["t"] = time.perf_counter()
            obs["state"] = state
            with lock:
                obs_list.append(obs.copy())
            time.sleep(0.01)  # 50 Hz logging

    obs_list = []
    thread = None
    x_throw = None
    # Outcome of the throw: stabilized succeeds; out_of_bounds and outdated fail, and so does
    # a run that never gets that far, e.g. one stopped with Ctrl+C
    outcome = None
    completed = False

    try:
        K = 0.3
        acc_z = 0.0
        vel_z_prev = None
        thrown = False
        spin_frequency = 100  # Hz
        spin_period = 1.0 / spin_frequency
        thread = threading.Thread(target=collect_obs, daemon=True, args=("pre_throw",))
        thread.start()
        while True:
            t_loop = time.perf_counter()
            obs = drone.obs()
            acc_finite_diff = (obs["vel"][2] - vel_z_prev) / spin_period if vel_z_prev else 0.0
            vel_z_prev = obs["vel"][2]
            acc_z = (1 - K) * acc_z + K * acc_finite_diff  # LFP for acc
            if initial_info["use_simple_throw_detection"] and obs["pos"][2] > 0.5:
                logger.warning("Simple throw detection triggered.")
                break
            if thrown and acc_z < -8:  # Is thrown
                break
            if acc_z > 20:
                thrown = True
            dt = time.perf_counter() - t_loop
            if dt < spin_period:
                time.sleep(spin_period - dt)

        collect_obs_event.set()
        thread.join()
        collect_obs_event.clear()
        thread = threading.Thread(target=collect_obs, daemon=True, args=("thrown",))
        thread.start()

        logger.info("Drone has been thrown, starting training.")
        time.sleep(0.1)  # Wait for estimator to keep up

        obs = drone.obs()
        x_throw = {
            "pos": obs["pos"],
            "quat": obs["quat"],
            "vel": obs["vel"],
            "ang_vel": obs["ang_vel"],
        }
        tstart = time.perf_counter()
        throw_controller.train(None if not initial_info["use_x_start"] else x_throw)  # TRAIN
        tend = time.perf_counter()
        logger.info(f"Training finished in {tend - tstart:.3f}s.")
        collect_obs_event.set()
        thread.join()
        collect_obs_event.clear()
        thread = threading.Thread(target=collect_obs, daemon=True, args=("trained",))
        thread.start()

        # Send trained policy weights to onboard controller
        tstart = time.perf_counter()
        throw_controller.send_parameters()
        drone.cf.param.set_value("ctrlThrow.freq", 500)
        tend = time.perf_counter()
        logger.info(f"send_parameters finished in {tend - tstart:.3f}s.")
        trained_event.set()
        collect_obs_event.set()
        thread.join()
        collect_obs_event.clear()
        thread = threading.Thread(target=collect_obs, daemon=True, args=("stabilizing",))
        thread.start()

        logger.info(f"[{time.perf_counter()}] Switching to onboard controller.")
        drone.cf.param.set_value("stabilizer.controller", 9)  # 9: Throw policy
        drone.send_action(np.zeros(4))

        action = np.zeros(4)  # use onboard policy, action doesn't matter
        ctrl_freq = 100  # Does not really matter, onboard policy is running
        while rclpy.ok():
            t = time.perf_counter()
            obs = drone.obs()

            if obs["is_outdated"]:
                logger.warning("Measurements outdated. Emergency stop.")
                outcome = "outdated"
                break

            x, y, z = obs["pos"]
            in_bounds = (
                FLIGHT_BOUNDS_X[0] <= x <= FLIGHT_BOUNDS_X[1]
                and FLIGHT_BOUNDS_Y[0] <= y <= FLIGHT_BOUNDS_Y[1]
                and FLIGHT_BOUNDS_Z[0] <= z
                # The throw may still carry the drone upwards right after the takeover
                and (z <= FLIGHT_BOUNDS_Z[1] or t - tend <= CEILING_MIN_TIME)
            )
            stabilized = (
                np.abs(obs["vel"][2]) < STABILIZED_VEL_Z_TOL and t - tend > STABILIZED_MIN_TIME
            )
            if not in_bounds or stabilized:
                if not in_bounds:
                    outcome = "out_of_bounds"
                    logger.warning("Drone left the flight bounds.")
                else:
                    outcome = "stabilized"
                    logger.info(f"[{time.perf_counter()}] Drone has stabilized.")
                collect_obs_event.set()
                thread.join()
                body_z = R.from_quat(obs["quat"]).apply([0.0, 0.0, 1.0])
                if np.arccos(np.clip(body_z[2], -1.0, 1.0)) > MAX_RETURN_TILT:
                    logger.warning("Drone too tilted to fly home. Emergency stop.")
                    break
                logger.info("Returning home.")
                drone.cf.param.set_value("stabilizer.controller", 2)  # switch back to Mellinger
                drone.return_to_start(keep_height=True)
                break

            drone.send_action(action)
            drone.update_estimators()
            sleep_until_next_cycle(t, ctrl_freq)

        completed = True

    finally:  # Kill the drone
        drone.close()
        collect_obs_event.set()
        if thread is not None:
            thread.join()
        # Store data AFTER killing the drone to make sure we end safely
        if x_throw is not None:
            _save_recording(throw_controller, x_throw, obs_list, outcome, completed)


def _save_recording(
    throw_controller: ThrowController,
    x_throw: dict,
    obs_list: list[dict],
    outcome: str | None,
    completed: bool,
) -> None:
    """Store a thrown run, with whether it succeeded, and the policy it flew."""
    success = completed and outcome == "stabilized"
    policy = None  # float32 weights before the float16 transfer, None if never trained
    if throw_controller.trained:
        policy = jax.tree.map(np.asarray, throw_controller.agent.actor_states.params)
    datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "x_throw": x_throw,
        "x_start": throw_controller.x_start,
        "policy": policy,
        "success": success,
        "outcome": outcome if completed else "interrupted",
        "obs": obs_list,
    }
    with open(RECORDINGS_DIR / f"x_throw_{datetime_str}.pkl", "wb") as f:
        pickle.dump(data, f)
    logger.info(f"Stored {'successful' if success else 'failed'} throw ({data['outcome']}).")


if __name__ == "__main__":
    main()
