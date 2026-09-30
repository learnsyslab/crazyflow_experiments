"""Flightmare sim2real tests.

Flightmare uses rotor thrusts as interface. We use crazyflow's controllers to work around this.

Run with
    pixi run -e flightmare python crazyflow_experiments/sim2real/sim_flightmare.py

"""

from __future__ import annotations

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import logging
import pickle
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import toml
import yaml
from crazyflow.control import load_params as load_control_params
from crazyflow.control import mellinger_attitude2force_torque, mellinger_state2attitude, parametrize
from crazyflow.control.mellinger import force_torque2rotor_vel
from crazyflow.control.transform import motor_force2rotor_vel
from crazyflow.dynamics import Dynamics, load_params
from flightgym import QuadrotorEnv_v1
from scipy.spatial.transform import Rotation as R

from crazyflow_experiments.control.config import TrajectoryConfig
from crazyflow_experiments.control.mellinger_controller import MellingerController
from crazyflow_experiments.sim2real import ATTITUDE_FREQ, PHYSICS_FREQ, RECORDINGS_DIR
from crazyflow_experiments.sim2real.utils import next_recording_path

if TYPE_CHECKING:
    from numpy.typing import NDArray

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


FLIGHTMARE_PATH = Path(os.environ["FLIGHTMARE_PATH"])


def state2obs(state: NDArray[np.floating]) -> dict:
    """Convert a Flightmare observation row to an observation."""
    assert state.shape == (12,), f"Unexpected observation shape: {state.shape}"
    rot = R.from_euler("ZYX", state[3:6])
    return {
        "pos": np.array(state[0:3], dtype=float),
        "quat": rot.as_quat(),
        "vel": np.array(state[6:9], dtype=float),
        "ang_vel": np.array(state[9:12], dtype=float),
        "rpy": rot.as_euler("xyz"),
    }


def identified_config(config: dict, params: dict, dt: float) -> dict:
    """Write the identified drone and the integration step into Flightmare's quadrotor config."""
    # Flightmare's allocation matrix in the convention of crazyflow's mixing matrix
    mixing_matrix = np.array([[-1, -1, 1, 1], [-1, 1, 1, -1], [-1, 1, -1, 1]])
    if not np.array_equal(np.asarray(params["mixing_matrix"]), mixing_matrix):
        raise ValueError(f"Flightmare mixes as {mixing_matrix.tolist()}, the drone differs")
    rpm2thrust = np.asarray(params["rpm2thrust"])
    gravity = np.asarray(params["gravity_vec"])
    thrusts = np.array(
        [params["thrust_min"], params["thrust_max"], -gravity[2] * params["mass"] / 4]
    )
    rotor_vel_min, rotor_vel_max, rotor_vel_hover = motor_force2rotor_vel(thrusts, rpm2thrust)
    acc1, acc2, _, _ = params["rotor_dyn_coef"]

    dynamics = config["quadrotor_dynamics"]
    dynamics["mass"] = float(params["mass"])
    dynamics["arm_l"] = float(params["L"] * np.sqrt(2.0))  # Flightmare levers at arm_l * sqrt(0.5)
    dynamics["motor_omega_min"] = float(rotor_vel_min)
    dynamics["motor_omega_max"] = float(rotor_vel_max)
    dynamics["motor_tau"] = float(1 / (acc1 + 2 * acc2 * rotor_vel_hover))  # Symmetric, linearized
    dynamics["thrust_map"] = rpm2thrust[::-1].tolist()
    dynamics["kappa"] = float(params["thrust2torque"])
    config["quadrotor_env"]["sim_dt"] = dt
    return config


class MellingerAttitudeControl:
    """crazyflow's Mellinger attitude loop on Flightmare's rotors with normalized actions."""

    rotor_order = [3, 0, 1, 2]  # Flightmare numbers from the front left, crazyflow from the right

    def __init__(self, drone: str, params: dict):
        """Initialize the controller."""
        self.attitude2force_torque = parametrize(mellinger_attitude2force_torque, drone)
        self.force_torque2rotor_vel = parametrize(force_torque2rotor_vel, drone)
        self.rpm2thrust = np.asarray(params["rpm2thrust"])
        self.hover_thrust = -np.asarray(params["gravity_vec"])[2] * params["mass"] / 4
        self.r_int_error, self.prev_ang_vel = np.zeros(3), np.zeros(3)

    def __call__(self, obs: dict, rpyt: NDArray[np.floating]) -> NDArray[np.floating]:
        """Return Flightmare's action for one attitude step."""
        ang_vel = obs["ang_vel"]
        force, torque, self.r_int_error = self.attitude2force_torque(
            obs["quat"], ang_vel, rpyt, self.prev_ang_vel, self.r_int_error, ATTITUDE_FREQ
        )
        self.prev_ang_vel = ang_vel
        rotor_vel = self.force_torque2rotor_vel(force, torque)
        thrust = np.polynomial.polynomial.polyval(rotor_vel, self.rpm2thrust)
        return (np.asarray(thrust)[self.rotor_order] - self.hover_thrust) / (2 * self.hover_thrust)


class ApproachControl:
    """Fly the drone to the starting position without crashing in flightmare.

    Flightmare resets the drone to a random state. We cannot set the state directly, so we fly it to
    the start pose instead. Since the crazyflie is really unstable in flightmare, we fly a very
    conservative approach trajectory.
    """

    lead = 0.2  # m, max setpoint distance from the drone
    speed = 0.4  # m/s, max feedforward speed
    max_yaw_error = 0.01  # rad, flightmare struggles with the slow motor dynamics of the crazyflie

    def __init__(self, drone: str, ctrl_freq: float):
        """Initialize the controller."""
        self.state2attitude = parametrize(mellinger_state2attitude, drone)
        self.state2attitude.keywords["ki"] = np.array([0.05, 0.05, 0.4])
        self.state2attitude.keywords["int_err_max"] = np.array([2.0, 2.0, 2.0])
        self.ctrl_freq = ctrl_freq
        self.pos_err_i = np.zeros(3)

    def __call__(self, obs: dict, pos: NDArray[np.floating], yaw: float) -> NDArray[np.floating]:
        """Return the attitude and thrust setpoint that flies the drone towards a pose."""
        yaw_error = np.arctan2(np.sin(yaw - obs["rpy"][2]), np.cos(yaw - obs["rpy"][2]))
        yaw = obs["rpy"][2] + np.clip(yaw_error, -self.max_yaw_error, self.max_yaw_error)
        offset = pos - obs["pos"]
        distance = np.linalg.norm(offset)
        way = offset / distance if distance > 1e-6 else np.zeros(3)
        cmd = np.concatenate(
            (
                obs["pos"] + way * min(self.lead, distance),
                way * self.speed if distance > self.lead else np.zeros(3),
                np.zeros(3),
                R.from_euler("z", yaw).as_quat(),
                np.zeros(3),
            )
        )
        # Prevent the integrator to wind up during the approach phase. Hacky, but necessary to reach
        # a fair initial state close to the start pose
        err_i = self.pos_err_i if distance <= self.lead else np.zeros(3)
        rpyt, self.pos_err_i = self.state2attitude(
            obs["pos"], obs["quat"], obs["vel"], cmd, err_i, self.ctrl_freq
        )
        return np.asarray(rpyt)


class Simulator:
    """One Flightmare environment with the buffers its step signature needs."""

    spinup_time = 0.5  # s
    # Reset approach tolerances
    approach_timeout = 30.0  # s
    approach_radius = 0.005  # m
    approach_speed = 0.01  # m/s
    approach_yaw = 0.002  # rad
    # reset redraw threshold
    max_reset_tilt = 0.05  # rad
    max_reset_yaw = 0.02  # rad

    def __init__(self, quadrotor_config: dict):
        """Build the environment from the contents of ``quadrotor_env.yaml``."""
        vec_config = {
            "env": {"seed": 1, "scene_id": 0, "num_envs": 1, "num_threads": 1, "render": False}
        }
        # Flightmare resolves its config through FLIGHTMARE_PATH when the environment is built
        with tempfile.TemporaryDirectory() as flightmare_path:
            config_dir = Path(flightmare_path) / "flightlib/configs"
            config_dir.mkdir(parents=True)
            (config_dir / "quadrotor_env.yaml").write_text(yaml.safe_dump(quadrotor_config))
            os.environ["FLIGHTMARE_PATH"] = flightmare_path
            self.env = QuadrotorEnv_v1(yaml.safe_dump(vec_config), False)
        os.environ["FLIGHTMARE_PATH"] = str(FLIGHTMARE_PATH)
        self.state = np.zeros((1, self.env.getObsDim()), dtype=np.float32)
        self.reward = np.zeros((1, 1), dtype=np.float32)
        self.done = np.zeros((1, 1), dtype=bool)
        self.extra_info = np.zeros((1, len(self.env.getExtraInfoNames())), dtype=np.float32)
        self.crashed = False

    def reset(self, yaw: float) -> dict:
        """Draw reset poses until the drone is level and points along the reference yaw.

        Flightmare draws at random and exposes no state setter, so the pose is redrawn, not set.
        """
        while True:
            self.env.reset(self.state)
            obs = state2obs(self.state[0])
            tilt = np.arccos(R.from_quat(obs["quat"]).as_matrix()[2, 2])
            yaw_error = np.arctan2(np.sin(obs["rpy"][2] - yaw), np.cos(obs["rpy"][2] - yaw))
            if tilt < self.max_reset_tilt and abs(yaw_error) < self.max_reset_yaw:
                return obs

    def step(self, action: NDArray[np.floating]) -> dict:
        """Advance the simulation by one physics step and return the observation."""
        self.env.step(
            action[None, ...].astype(np.float32),
            self.state,
            self.reward,
            self.done,
            self.extra_info,
        )
        if self.done[0, 0] and not self.crashed:
            self.crashed = True  # Flightmare draws a new random pose whenever the drone lands
            logger.warning("Flightmare hit the ground and reset the drone, the rollout is invalid")
        obs = state2obs(self.state[0])
        obs["lost"] = self.crashed  # Persistent so it persists across substeps
        return obs


def step_attitude_loop(
    sim: Simulator,
    attitude_control: MellingerAttitudeControl,
    obs: dict,
    rpyt: NDArray[np.floating],
    n_steps: int,
) -> dict:
    """Run the attitude loop for one outer control period and return the new observation."""
    substeps = PHYSICS_FREQ // ATTITUDE_FREQ
    for _ in range(n_steps):
        action = attitude_control(obs, rpyt)
        for _ in range(substeps):
            obs = sim.step(action)
    return obs


def rollout(config: dict):
    """Simulate one rollout of the experiment and save its recording."""
    if config["controller"]["type"] != "mellinger":
        raise ValueError(f"Unsupported Flightmare controller type: {config['controller']['type']}")
    model = config["drone"]["model"]

    data = []
    # Set up initial info
    ctrl_freq = config["controller"]["ctrl_freq"]
    steps_per_ctrl = ATTITUDE_FREQ // ctrl_freq
    trajectory_config = TrajectoryConfig.from_config(config)
    trajectory = trajectory_config.build()
    pos_start, yaw_start = trajectory_config.start_pose(trajectory)

    # Set up sim
    drone_params = load_control_params("mellinger", model)["core"] | load_params(
        Dynamics.first_principles, model
    )
    quadrotor_config = yaml.safe_load(
        (FLIGHTMARE_PATH / "flightlib/configs/quadrotor_env.yaml").read_text()
    )
    quadrotor_config = identified_config(quadrotor_config, drone_params, 1 / PHYSICS_FREQ)
    sim = Simulator(quadrotor_config)
    obs = sim.reset(yaw_start)
    obs["t"] = 0.0

    # Prepare controllers
    controller = MellingerController(
        obs, trajectory, model, tilt_max=np.deg2rad(config["controller"]["tilt_max"]) or None
    )
    attitude_control = MellingerAttitudeControl(model, drone_params)
    approach_control = ApproachControl(model, ctrl_freq)

    # Spin up the rotors. We need to keep yaw to not destablilize the drone in flightmare
    hover_thrust = -np.asarray(drone_params["gravity_vec"])[2] * drone_params["mass"]
    rpyt = np.array([0.0, 0.0, yaw_start, hover_thrust])
    for _ in range(int(sim.spinup_time * ctrl_freq)):
        obs = step_attitude_loop(sim, attitude_control, obs, rpyt, steps_per_ctrl)

    # Fly to the start pose from the random start pose
    for _ in range(int(sim.approach_timeout * ctrl_freq)):
        rpyt = approach_control(obs, pos_start, yaw_start)
        obs = step_attitude_loop(sim, attitude_control, obs, rpyt, steps_per_ctrl)
        yaw_error = np.arctan2(np.sin(obs["rpy"][2] - yaw_start), np.cos(obs["rpy"][2] - yaw_start))
        if np.linalg.norm(obs["pos"] - pos_start) < sim.approach_radius:
            if (
                np.linalg.norm(obs["vel"]) < sim.approach_speed
                and abs(yaw_error) < sim.approach_yaw
            ):
                break
    else:
        raise RuntimeError(f"The drone did not reach {pos_start} within {sim.approach_timeout} s")

    # Experiment
    i = 0
    obs["t"] = 0.0
    controller._t_start = obs["t"]
    controller._t_start -= trajectory_config.start_time  # Negative for a soft start
    while True:
        action = controller(obs)
        obs = step_attitude_loop(sim, attitude_control, obs, action, steps_per_ctrl)
        i += 1
        obs["t"] = i / ctrl_freq
        obs["u"] = action
        data.append(obs | {"t": obs["t"] - controller._t_start})  # Recorded in trajectory time
        if controller.done(obs):
            break

    sim.env.close()
    if sim.crashed:
        logger.error("Rollout has failed")
    with open(next_recording_path(RECORDINGS_DIR, config), "wb") as f:
        pickle.dump(data, f)


def main(config: dict | None = None):
    """Repeat the rollout `n_rollouts` times, each saved under the next free index."""
    if config is None:
        config = toml.load(Path(__file__).parent / "config.toml")
    for _ in range(config["sim"]["n_rollouts"]):
        rollout(config)


if __name__ == "__main__":
    main()
