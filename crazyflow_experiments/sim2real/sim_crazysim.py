"""Sim2real tests for CrazySim.

This file must run inside the container built from `sim2real/CrazySimDocker`, because we cannot
create a pixi environment for the sim. Use `sim2real/run_crazysim.py` to run all four paper
conditions for one image variant in one go.

The physics name selects the backend and its variants, e.g. ``crazysim-gz``, ``crazysim-mj`` or
``crazysim-mj-reaction`` (MuJoCo with the rotor reaction torque the backend omits) or
``crazysim-mj-reaction-rotor`` (additionally crazyflow's identified rotor model), see
`load_mj_backend`; ``origpacing`` paces the physics like CrazySim's own loop (real-time factor
≈ 0.91), see `MjCrazyflie`; ``simtick`` declares that the firmware in the image counts its ticks
from the simulator (`crazysim_sim_tick.py`), and then the control loop here also runs on
simulation time, so that the whole chain is independent of the real-time factor.
"""

import logging
import pickle
import time
from functools import partial
from pathlib import Path

import numpy as np
import toml
from crazyflow.control import mellinger_state2attitude, parametrize
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

from crazyflow_experiments.control.config import TrajectoryConfig
from crazyflow_experiments.control.interface import control_interface
from crazyflow_experiments.control.loop import sleep_until_next_cycle
from crazyflow_experiments.control.mellinger_controller import MellingerController
from crazyflow_experiments.control.state_controller import StateController
from crazyflow_experiments.control.takeoff import TakeoffTrajectory
from crazyflow_experiments.sim2real import RECORDINGS_DIR
from crazyflow_experiments.sim2real.sim_crazysim_interface import (
    GzCrazyflie,
    MjCrazyflie,
    SitlCrazyflie,
)
from crazyflow_experiments.sim2real.utils import next_recording_path

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Acceptance of the takeoff: hover within this distance of the start pose and below this tilt
TAKEOFF_MAX_POS_ERROR = 0.3  # m
TAKEOFF_MAX_TILT = np.deg2rad(20.0)  # rad
# A rollout counts as crashed once the drone tilts beyond this angle or drops to the floor
CRASH_TILT = np.deg2rad(60.0)  # rad
CRASH_HEIGHT = 0.1  # m
SIM_TIME_POLL_INTERVAL = 0.0002  # s of wall time between polls of the simulation clock


def takeoff_succeeded(
    obs: dict, pos_start: NDArray[np.floating], max_pos_error: float, max_tilt: float
) -> bool:
    """Check that the drone hovers at the start pose after the takeoff.

    The firmware-in-the-loop attitude loop has little phase margin (see the investigation
    handoff), and the climb-and-translate ramp from the floor tips the drone over in some
    rollouts. Such rollouts never reach the trajectory and are discarded rather than recorded.

    Args:
        obs: Observation at the end of the takeoff.
        pos_start: Start pose of the trajectory.
        max_pos_error: Largest accepted distance to the start pose in m.
        max_tilt: Largest accepted roll or pitch angle in rad.
    """
    pos_error = float(np.linalg.norm(obs["pos"] - pos_start))
    tilt = float(np.max(np.abs(obs["rpy"][:2])))
    if pos_error > max_pos_error or tilt > max_tilt:
        logger.warning(
            f"Takeoff failed: {pos_error:.2f} m from the start pose, tilt {np.rad2deg(tilt):.0f} deg"
        )
        return False
    return True


class Clock:
    """Time source and cycle pacing of the control loop, on the wall clock or on simulation time.

    With the wall-clock firmware the loop must run on the wall clock like the firmware does. With
    the sim-tick firmware everything in the loop advances with the simulator, so the loop is paced
    by simulation time and the recordings carry simulation time; a slow machine then only makes
    the run take longer.
    """

    def __init__(self, sitl_cf: SitlCrazyflie, use_sim_time: bool):
        """Pace by simulation time of the backend or by the wall clock."""
        self.now = sitl_cf.sim_time if use_sim_time else time.perf_counter
        self.use_sim_time = use_sim_time

    def sleep_until_next_cycle(self, cycle_start: float, ctrl_freq: float):
        """Wait until the next control cycle starts."""
        if not self.use_sim_time:
            sleep_until_next_cycle(cycle_start, ctrl_freq)
            return
        deadline = cycle_start + 1 / ctrl_freq
        if self.now() > deadline:
            logger.warning(
                f"Controller execution exceeded the cycle by {self.now() - deadline:.3f} s"
            )
        while self.now() < deadline:
            time.sleep(SIM_TIME_POLL_INTERVAL)


def rollout(config: dict) -> str:
    """Simulate one rollout of the experiment and save its recording if the drone completes it.

    The SITL firmware and the simulator must already be running, see `run_crazysim.py`, which
    also restarts them between repeated rollouts.

    Returns:
        ``"completed"`` if the trajectory was flown and recorded, ``"takeoff_failed"`` if the drone
        did not reach the start pose (see `takeoff_succeeded`), ``"crashed"`` if it tilted beyond
        `CRASH_TILT` or dropped below `CRASH_HEIGHT` during the trajectory. Only completed
        rollouts leave a recording, so that the k-th recording always pairs with the k-th real
        flight.
    """
    physics = config["sim"]["physics"]
    variant = physics.split("-")
    assert variant[0] == "crazysim", f"Not a CrazySim physics name: {physics}"
    if "mj" in variant:
        # "reaction" adds the rotor reaction torque that the backend omits and "rotor" swaps in
        # crazyflow's identified rotor model, see load_mj_backend
        crazyflie_interface = partial(
            MjCrazyflie,
            reaction_torque="reaction" in variant,
            crazyflow_rotor="rotor" in variant,
            original_pacing="origpacing" in variant,
        )
    else:
        crazyflie_interface = GzCrazyflie
    use_sim_time = "simtick" in variant

    data = []

    # Set up initial info
    ctrl_freq = config["controller"]["ctrl_freq"]
    ctrl_interface = control_interface(config["controller"]["type"])
    trajectory_config = TrajectoryConfig.from_config(config)
    trajectory = trajectory_config.build()
    pos_start, yaw_start = trajectory_config.start_pose(trajectory)

    # Set up the simulator connection (Gazebo, or the in-process MuJoCo backend)
    sitl_cf = crazyflie_interface(config["drone"]["model"], ctrl_interface)
    clock = Clock(sitl_cf, use_sim_time)

    # Prepare controllers (not necessary for sim, but for consistency with real)
    initial_obs = sitl_cf.get_obs()
    initial_obs["t"] = clock.now()
    match config["controller"]["type"]:
        case "mellinger":
            controller = MellingerController(
                initial_obs,
                trajectory,
                config["drone"]["model"],
                tilt_max=np.deg2rad(config["controller"]["tilt_max"]) or None,
            )
        case "state":
            controller = StateController(initial_obs, trajectory)
        case _:
            raise ValueError(f"Unknown controller type: {config['controller']['type']}")

    # Take off
    obs = sitl_cf.get_obs()
    trajectory_pos, trajectory_yaw = TakeoffTrajectory(
        obs["pos"], obs["rpy"][2], pos_start, yaw_start
    ).setpoints(ctrl_freq)
    state2attitude = parametrize(mellinger_state2attitude, config["drone"]["model"])
    cmd, pos_err_i = np.zeros(16), np.zeros(3)

    for pos, yaw in zip(trajectory_pos, trajectory_yaw, strict=True):
        t = clock.now()
        obs = sitl_cf.get_obs()
        cmd[:3], cmd[9:13] = pos, R.from_euler("z", yaw).as_quat()
        action, pos_err_i = state2attitude(
            obs["pos"], obs["quat"], obs["vel"], cmd, pos_err_i, ctrl_freq
        )
        sitl_cf.send_action(action)
        clock.sleep_until_next_cycle(t, ctrl_freq)

    if not takeoff_succeeded(sitl_cf.get_obs(), pos_start, TAKEOFF_MAX_POS_ERROR, TAKEOFF_MAX_TILT):
        sitl_cf.close()
        return "takeoff_failed"

    # Experiment
    # Actual experiment controller
    initial_obs = sitl_cf.get_obs()
    initial_obs["t"] = clock.now()
    controller._t_start = initial_obs["t"]  # Reset start time
    controller._t_start -= trajectory_config.start_time  # Negative for a soft start
    controller_finished = False
    while not controller_finished:
        t = clock.now()
        obs = sitl_cf.get_obs()
        obs["t"] = t
        action = controller(obs)
        if config["controller"]["type"] == "state":
            raise NotImplementedError("State control not implemented for CrazySim")
        else:
            sitl_cf.send_action(action)

        obs["u"] = action
        data.append(obs | {"t": obs["t"] - controller._t_start})  # Recorded in trajectory time

        if np.max(np.abs(obs["rpy"][:2])) > CRASH_TILT or obs["pos"][2] < CRASH_HEIGHT:
            logger.warning(f"Crashed {t - initial_obs['t']:.1f} s into the trajectory")
            sitl_cf.close()
            return "crashed"
        controller_finished = controller.done(obs)
        if controller_finished:
            break
        clock.sleep_until_next_cycle(t, ctrl_freq)

    with open(next_recording_path(RECORDINGS_DIR, config), "wb") as f:
        pickle.dump(data, f)

    sitl_cf.close()
    return "completed"


def main(config: dict | None = None) -> None:
    """Record `n_rollouts` completed rollouts, each saved under the next free index.

    Rollouts that fail the takeoff or crash are repeated; the discarded attempts are logged.
    """
    if config is None:
        config = toml.load(Path(__file__).parent / "config.toml")
    outcomes = {"completed": 0, "takeoff_failed": 0, "crashed": 0}
    while outcomes["completed"] < config["sim"]["n_rollouts"]:
        outcomes[rollout(config)] += 1
    logger.info(f"Rollout outcomes: {outcomes}")


if __name__ == "__main__":
    main()
