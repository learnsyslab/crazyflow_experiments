"""l2f sim2real tests.

l2f uses rotor throttles as interface. We use crazyflow's controllers to work around this.

Run with
    pixi run -e l2f python crazyflow_experiments/sim2real/sim_l2f.py

"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import l2f
import numpy as np
import toml
from crazyflow.control import mellinger_attitude2force_torque, mellinger_state2attitude, parametrize
from crazyflow.control.mellinger import force_torque2rotor_vel
from crazyflow.dynamics import Dynamics, load_params
from scipy.spatial.transform import Rotation as R

from crazyflow_experiments.control.config import TrajectoryConfig
from crazyflow_experiments.control.mellinger_controller import MellingerController
from crazyflow_experiments.control.takeoff import TakeoffTrajectory
from crazyflow_experiments.sim2real import ATTITUDE_FREQ, PHYSICS_FREQ, RECORDINGS_DIR
from crazyflow_experiments.sim2real.utils import next_recording_path

if TYPE_CHECKING:
    from numpy.typing import NDArray

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def state2obs(state: l2f.State) -> dict:
    """Convert an l2f state to an observation, whose quaternion l2f orders w-x-y-z."""
    quat = np.roll(np.array(state.orientation), -1)
    return {
        "pos": np.array(state.position),
        "quat": quat,
        "vel": np.array(state.linear_velocity),
        "ang_vel": np.array(state.angular_velocity),
        "rpy": R.from_quat(quat).as_euler("xyz"),
    }


class MellingerAttitudeControl:
    """crazyflow's Mellinger attitude loop on l2f's rotors."""

    def __init__(self, drone: str, params: dict, dynamics: dict):
        """Initialize the controller."""
        self.attitude2force_torque = parametrize(mellinger_attitude2force_torque, drone)
        self.force_torque2rotor_vel = parametrize(force_torque2rotor_vel, drone)
        self.rpm2thrust = np.asarray(params["rpm2thrust"])
        self.thrust_curve = np.asarray(dynamics["rotor_thrust_coefficients"])
        self.action_min = dynamics["action_limit"]["min"]
        self.action_max = dynamics["action_limit"]["max"]
        self.r_int_error, self.prev_ang_vel = np.zeros(3), np.zeros(3)

    def __call__(self, obs: dict, rpyt: NDArray[np.floating]) -> list[float]:
        """Return l2f's action for one attitude step."""
        ang_vel = obs["ang_vel"]
        force, torque, self.r_int_error = self.attitude2force_torque(
            obs["quat"], ang_vel, rpyt, self.prev_ang_vel, self.r_int_error, ATTITUDE_FREQ
        )
        self.prev_ang_vel = ang_vel
        # Convert the thrust to l2f's thrust curve and then to normalized throttle
        rotor_vel = np.asarray(self.force_torque2rotor_vel(force, torque))
        thrust = np.polynomial.polynomial.polyval(rotor_vel, self.rpm2thrust)
        c0, c1, c2 = self.thrust_curve.T
        throttle = (-c1 + np.sqrt(c1**2 - 4 * c2 * (c0 - thrust))) / (2 * c2)
        span = self.action_max - self.action_min
        return (2 * (throttle - self.action_min) / span - 1).tolist()


class Simulator:
    """One l2f environment with the handles its step signature needs."""

    def __init__(self, dt: float):
        """Build the environment and set the integration step."""
        self.device, self.env = l2f.Device(), l2f.Environment()
        self.params, self.rng = l2f.Parameters(), l2f.Rng()
        self.state, self.next_state = l2f.State(), l2f.State()
        l2f.initialize_environment(self.device, self.env)
        l2f.initialize_rng(self.device, self.rng, 0)
        l2f.initial_parameters(self.device, self.env, self.params)
        config = json.loads(l2f.parameters_to_json(self.device, self.env, self.params))
        config["integration"]["dt"] = dt
        # l2f rejects a partial parameter json, so its own drone is written back with the step
        l2f.parameters_from_json(self.device, self.env, json.dumps(config), self.params)
        self.dynamics = config["dynamics"]

    def reset(self, pos: NDArray[np.floating]) -> dict:
        """Reset the drone onto the floor under the start position, l2f has no ground."""
        l2f.initial_state(self.device, self.env, self.params, self.state)
        self.state.position = [pos[0], pos[1], 0.0]
        return state2obs(self.state)

    def step(self, action: list[float]) -> dict:
        """Advance the simulation by one physics step and return the observation."""
        l2f.step(self.device, self.env, self.params, self.state, action, self.next_state, self.rng)
        self.state.assign(self.next_state)
        return state2obs(self.state)


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
        raise ValueError(f"Unsupported l2f controller type: {config['controller']['type']}")
    model = config["drone"]["model"]

    data = []
    # Set up initial info
    ctrl_freq = config["controller"]["ctrl_freq"]
    steps_per_ctrl = ATTITUDE_FREQ // ctrl_freq
    trajectory_config = TrajectoryConfig.from_config(config)
    trajectory = trajectory_config.build()
    pos_start, yaw_start = trajectory_config.start_pose(trajectory)

    # Set up sim
    drone_params = load_params(Dynamics.first_principles, model)
    sim = Simulator(1 / PHYSICS_FREQ)
    obs = sim.reset(pos_start)
    obs["t"] = 0.0

    # Prepare controllers (not necessary for sim, but for consistency with real)
    controller = MellingerController(
        obs, trajectory, model, tilt_max=np.deg2rad(config["controller"]["tilt_max"]) or None
    )
    attitude_control = MellingerAttitudeControl(model, drone_params, sim.dynamics)

    # Take off
    # Fly drone to hover position (actually not necessary in sim, but for consistency with real)
    trajectory_pos, trajectory_yaw = TakeoffTrajectory(
        obs["pos"], obs["rpy"][2], pos_start, yaw_start
    ).setpoints(ctrl_freq)
    state2attitude = parametrize(mellinger_state2attitude, model)
    cmd, pos_err_i = np.zeros(16), np.zeros(3)

    for pos, yaw in zip(trajectory_pos, trajectory_yaw, strict=True):
        cmd[:3], cmd[9:13] = pos, R.from_euler("z", yaw).as_quat()
        rpyt, pos_err_i = state2attitude(
            obs["pos"], obs["quat"], obs["vel"], cmd, pos_err_i, ctrl_freq
        )
        obs = step_attitude_loop(sim, attitude_control, obs, rpyt, steps_per_ctrl)

    # Experiment
    i = len(trajectory_pos)  # We have advanced during takeoff
    obs["t"] = i / ctrl_freq
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
