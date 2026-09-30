"""Gym PyBullet Drones sim2real tests.

Run with
    pixi run -e pybullet python crazyflow_experiments/sim2real/sim_pybullet.py

The physics name of the config selects the attitude loop and the drag model, see ``VARIANTS``. The
outer position controller runs at the experiment's ``ctrl_freq`` and the attitude loop at
``ATTITUDE_FREQ``, as on the real drone.

gym-pybullet-drones ships the stock Crazyflie in its ``cf2x.urdf``. The identified parameters of
our drone are applied at runtime instead, see ``identified_drone``, so the submodule stays stock.
"""

from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pybullet as p
import toml
from crazyflow.control import load_params as load_control_params
from crazyflow.control import mellinger_attitude2force_torque, mellinger_state2attitude, parametrize
from crazyflow.control.mellinger import force_torque2rotor_vel
from crazyflow.control.transform import motor_force2rotor_vel
from crazyflow.dynamics import Dynamics, load_params
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from scipy.spatial.transform import Rotation as R

from crazyflow_experiments.control.config import TrajectoryConfig
from crazyflow_experiments.control.mellinger_controller import MellingerController
from crazyflow_experiments.control.takeoff import TakeoffTrajectory
from crazyflow_experiments.sim2real import ATTITUDE_FREQ, PHYSICS_FREQ, RECORDINGS_DIR
from crazyflow_experiments.sim2real.utils import next_recording_path

if TYPE_CHECKING:
    from gym_pybullet_drones.envs.BaseAviary import BaseAviary
    from numpy.typing import NDArray

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


GROUND_CLEARANCE = 0.02  # spawn height in m
DSL_TORQUE_CLIP = 3200  # PWM, torque clip hard-coded in DSLPIDControl
FIRMWARE_TORQUE_CLIP = 32000  # PWM, torque clip of the Crazyflie firmware
# Physics name -> (attitude loop, drag model). Attitude loops: "dsl" is DSLPIDControl of
# gym-pybullet-drones on the instantaneous motors of the simulator, "mellinger" is crazyflow's
# Mellinger loop on the same motors, and the "-rotor" suffix puts crazyflow's rotor dynamics in front
# of them. "-fwclip" widens the torque clip of DSLPIDControl from GPD's ±3200 to the firmware's
# ±32000 PWM; with the rotor lag, the narrow clip turns the loop's ringing into a flip. Drag models:
# "none" is Physics.PYB (the paper's setting), "pybullet" is Physics.PYB_DRAG, "identified" applies
# the drag matrix of crazyflow's first principles model.
VARIANTS = {
    "pybullet": ("dsl", "none"),
    "pybullet-pybdrag": ("dsl", "pybullet"),
    "pybullet-drag": ("dsl", "identified"),
    "pybullet-rotor": ("dsl-rotor", "none"),
    "pybullet-rotor-pybdrag": ("dsl-rotor", "pybullet"),
    "pybullet-rotor-drag": ("dsl-rotor", "identified"),
    "pybullet-rotor-fwclip": ("dsl-rotor-fwclip", "none"),
    "pybullet-rotor-fwclip-pybdrag": ("dsl-rotor-fwclip", "pybullet"),
    "pybullet-rotor-fwclip-drag": ("dsl-rotor-fwclip", "identified"),
    "pybullet-mellinger-pybdrag": ("mellinger", "pybullet"),
    "pybullet-mellinger-drag": ("mellinger", "identified"),
    "pybullet-mellinger-rotor-pybdrag": ("mellinger-rotor", "pybullet"),
    "pybullet-mellinger-rotor-drag": ("mellinger-rotor", "identified"),
}


def state2obs(state: NDArray) -> dict:
    """Convert a gym pybullet drones state to an observation.

    For the convention, see BaseAviary._getDroneStateVector.
    """
    assert state.shape == (1, 20), f"Unexpected state shape: {state.shape}"
    state = state.flatten()
    return {
        "pos": state[0:3],
        "quat": state[3:7],
        "vel": state[10:13],
        "ang_vel": state[13:16],
        "rpy": R.from_quat(state[3:7]).as_euler("xyz"),
    }


def identified_drone(
    aviary: type[BaseAviary], params: dict, drag_matrix: NDArray[np.floating] | None
) -> type[BaseAviary]:
    """Return an aviary subclass whose drone carries the identified parameters.

    gym-pybullet-drones reads mass and inertia from the URDF twice: PyBullet loads them into the
    rigid body, and ``BaseAviary`` parses them to derive its constants. Overriding the two hooks
    replaces both copies, so the URDF asset stays stock. The motor torque coefficient follows the
    identified thrust-to-torque ratio (the stock value is 3.4 times too large for our drone).
    Every physics step additionally applies ``rotor_torque``, the torque of the spinning rotors set
    by the attitude loop, and the linear drag of ``drag_matrix`` if given.

    Args:
        aviary: The aviary class to subclass, e.g. ``CtrlAviary``.
        params: Drone parameters from ``crazyflow.drones.load_params``.
        drag_matrix: Linear drag matrix (3, 3) in the body frame, or None for no extra drag.

    Returns:
        The aviary subclass with the parameters applied on every reset.
    """
    mass, inertia = params["mass"], np.asarray(params["J"], dtype=float)

    class IdentifiedAviary(aviary):
        rotor_torque = np.zeros(3)  # Body torque of the rotors, see rotor_inertia_torque

        def _parseURDFParameters(self) -> tuple:
            values = list(super()._parseURDFParameters())
            values[0], values[3], values[4] = mass, inertia, np.linalg.inv(inertia)
            values[6] = values[5] * params["thrust2torque"]  # KM from KF
            return tuple(values)

        def _housekeeping(self) -> None:
            super()._housekeeping()  # Reloads the URDF, so the body needs the parameters again
            for drone_id in self.DRONE_IDS:
                p.changeDynamics(
                    int(drone_id),
                    -1,
                    mass=mass,
                    localInertiaDiagonal=np.diag(inertia).tolist(),
                    physicsClientId=self.CLIENT,
                )

        def _physics(self, rpm: NDArray[np.floating], nth_drone: int) -> None:
            super()._physics(rpm, nth_drone)
            drone_id, client = int(self.DRONE_IDS[nth_drone]), self.CLIENT
            if drag_matrix is not None:
                vel, _ = p.getBaseVelocity(drone_id, physicsClientId=client)
                _, quat = p.getBasePositionAndOrientation(drone_id, physicsClientId=client)
                force = drag_matrix @ R.from_quat(quat).apply(vel, inverse=True)
                p.applyExternalForce(
                    drone_id, -1, force.tolist(), [0, 0, 0], p.LINK_FRAME, physicsClientId=client
                )
            p.applyExternalTorque(
                drone_id, -1, self.rotor_torque.tolist(), p.LINK_FRAME, physicsClientId=client
            )

    IdentifiedAviary.__name__ = f"Identified{aviary.__name__}"
    return IdentifiedAviary


def rotor_dynamics(
    rotor_vel: NDArray[np.floating],
    cmd: NDArray[np.floating],
    dt: float,
    rotor_dyn_coef: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Advance the rotor speeds by one Euler step of crazyflow's rotor model.

    Same spin-up/spin-down model as ``crazyflow.dynamics.first_principles``.

    Args:
        rotor_vel: Current rotor speeds in RPM with shape (4,).
        cmd: Commanded rotor speeds in RPM with shape (4,).
        dt: Integration step in s.
        rotor_dyn_coef: Coefficients ``[k_acc1, k_acc2, k_dec1, k_dec2]`` of the drone.

    Returns:
        The rotor speeds after ``dt``.
    """
    acc1, acc2, dec1, dec2 = rotor_dyn_coef
    rotor_acc = np.where(
        cmd > rotor_vel,
        acc1 * (cmd - rotor_vel) + acc2 * (cmd**2 - rotor_vel**2),
        dec1 * (cmd - rotor_vel) + dec2 * (cmd**2 - rotor_vel**2),
    )
    return rotor_vel + dt * rotor_acc


def rotor_inertia_torque(
    rotor_vel: NDArray[np.floating],
    rotor_acc: NDArray[np.floating],
    ang_vel: NDArray[np.floating],
    spin: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Body torque of the spinning rotors: gyroscopic coupling and reaction to rotor acceleration.

    Same terms as ``crazyflow.dynamics.first_principles``. The reaction torque is what lets the
    yaw loop act faster than the rotor time constant.

    Args:
        rotor_vel: Rotor speeds in RPM with shape (4,).
        rotor_acc: Rotor accelerations in RPM/s with shape (4,).
        ang_vel: Angular velocity of the body in rad/s with shape (3,).
        spin: Signed rotor inertia ``mixing_matrix[2] * prop_inertia`` with shape (4,).

    Returns:
        The torque in the body frame with shape (3,).
    """
    rad = 2 * np.pi / 60
    momentum = np.sum(spin * rotor_vel * rad)
    reaction = np.sum(spin * rotor_acc * rad)
    return np.array([ang_vel[1] * momentum, -ang_vel[0] * momentum, reaction])


class RotorModel:
    """crazyflow's rotor dynamics between the commanded and the spinning rotor speeds.

    gym-pybullet-drones turns RPM into thrust with its own quadratic curve, so the rotor speeds are
    integrated on crazyflow's RPM scale and mapped through crazyflow's thrust curve to the RPM that
    yields the same thrust in PyBullet. The rotors start at rest, as in crazyflow.
    """

    def __init__(self, params: dict, kf: float):
        """Initialize the rotor model.

        Args:
            params: Drone parameters from ``crazyflow.drones.load_params``.
            kf: Thrust coefficient of gym-pybullet-drones (N/RPM^2).
        """
        self.rpm2thrust = np.asarray(params["rpm2thrust"])
        self.rotor_dyn_coef = np.asarray(params["rotor_dyn_coef"])
        self.spin = np.asarray(params["mixing_matrix"])[2] * params["prop_inertia"]
        thrust_limits = np.array([params["thrust_min"], params["thrust_max"]])
        self.rotor_vel_limits = motor_force2rotor_vel(thrust_limits, self.rpm2thrust)
        self.kf = kf
        self.rotor_vel = np.zeros(4)
        self.rotor_torque = np.zeros(3)

    def gpd2rotor_vel(self, rpm: NDArray[np.floating]) -> NDArray[np.floating]:
        """Convert gym-pybullet-drones RPM to the crazyflow RPM of equal thrust."""
        return motor_force2rotor_vel(self.kf * rpm**2, self.rpm2thrust)

    def rotor_vel2gpd(self, rotor_vel: NDArray[np.floating]) -> NDArray[np.floating]:
        """Convert crazyflow RPM to the gym-pybullet-drones RPM of equal thrust."""
        thrust = np.polynomial.polynomial.polyval(rotor_vel, self.rpm2thrust)
        return np.sqrt(np.maximum(thrust, 0.0) / self.kf)

    def __call__(
        self, cmd: NDArray[np.floating], ang_vel: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Advance the rotors by one attitude step towards the commanded speeds.

        Args:
            cmd: Commanded rotor speeds in crazyflow RPM with shape (4,).
            ang_vel: Angular velocity of the body in the body frame with shape (3,).

        Returns:
            The rotor speeds after the step, clipped to the thrust limits of the drone. Also updates
            ``rotor_torque`` with the body torque the rotors exert during the step.
        """
        dt = 1 / ATTITUDE_FREQ
        rotor_vel = rotor_dynamics(self.rotor_vel, cmd, dt, self.rotor_dyn_coef)
        rotor_vel = np.clip(rotor_vel, *self.rotor_vel_limits)
        rotor_acc = (rotor_vel - self.rotor_vel) / dt
        self.rotor_torque = rotor_inertia_torque(rotor_vel, rotor_acc, ang_vel, self.spin)
        self.rotor_vel = rotor_vel
        return rotor_vel


def body_ang_vel(obs: dict) -> NDArray[np.floating]:
    """Angular velocity in the body frame; gym-pybullet-drones reports it in the world frame."""
    return R.from_quat(obs["quat"]).apply(obs["ang_vel"], inverse=True)


class DSLAttitudeControl:
    """Attitude loop of gym-pybullet-drones, optionally with rotor dynamics in front of the motors.

    The loop commands PWM and maps it to RPM with GPD's affine PWM-to-RPM curve. With the rotor
    model, that RPM is the setpoint of the rotor dynamics instead of the speed of the motors.
    """

    def __init__(self, rotors: RotorModel | None = None, torque_clip: float = DSL_TORQUE_CLIP):
        """Initialize the controller.

        Args:
            rotors: Rotor dynamics between the loop and the motors, or None for instantaneous
                motors as in gym-pybullet-drones.
            torque_clip: Clip of the roll, pitch and yaw torque commands in PWM units. The clip is
                hard-coded in DSLPIDControl, so it is widened by scaling the PID gains down and the
                mixer up by the same factor, which leaves the unclipped output unchanged.
        """
        self.ctrl = DSLPIDControl(DroneModel.CF2X)
        scale = torque_clip / DSL_TORQUE_CLIP
        for gains in ("P_COEFF_TOR", "I_COEFF_TOR", "D_COEFF_TOR"):
            setattr(self.ctrl, gains, getattr(self.ctrl, gains) / scale)
        self.ctrl.MIXER_MATRIX = self.ctrl.MIXER_MATRIX * scale
        self.rotors = rotors
        self.rotor_torque = np.zeros(3)  # Only set by the rotor model

    def __call__(self, obs: dict, rpyt: NDArray[np.floating]) -> NDArray[np.floating]:
        """Return the motor RPMs (gym-pybullet-drones scale) for one attitude step."""
        ctrl = self.ctrl
        rpy = R.from_euler("xyz", rpyt[:3]).as_euler("XYZ")
        pwm = (np.sqrt(rpyt[3] / (4 * ctrl.KF)) - ctrl.PWM2RPM_CONST) / ctrl.PWM2RPM_SCALE
        pwm = np.clip(pwm, ctrl.MIN_PWM, ctrl.MAX_PWM)
        rpm = ctrl._dslPIDAttitudeControl(1 / ATTITUDE_FREQ, pwm, obs["quat"], rpy, np.zeros(3))
        if self.rotors is None:
            return rpm
        rotor_vel = self.rotors(self.rotors.gpd2rotor_vel(rpm), body_ang_vel(obs))
        self.rotor_torque = self.rotors.rotor_torque
        return self.rotors.rotor_vel2gpd(rotor_vel)


class MellingerAttitudeControl:
    """crazyflow's Mellinger attitude loop, optionally with rotor dynamics in front of the motors.

    Without the rotor model, the rotor speeds the loop commands are mapped straight to the RPM of
    equal thrust in gym-pybullet-drones, see ``RotorModel``.
    """

    def __init__(self, drone: str, rotors: RotorModel, rotor_dynamics: bool):
        """Initialize the controller.

        Args:
            drone: Drone model name.
            rotors: Rotor model of the drone, used for the RPM conversion in any case.
            rotor_dynamics: Whether to integrate the rotor dynamics as well.
        """
        self.attitude2force_torque = parametrize(mellinger_attitude2force_torque, drone)
        self.force_torque2rotor_vel = parametrize(force_torque2rotor_vel, drone)
        self.rotors = rotors
        self.rotor_dynamics = rotor_dynamics
        self.r_int_error, self.prev_ang_vel = np.zeros(3), np.zeros(3)
        self.rotor_torque = np.zeros(3)  # Only set by the rotor model

    def __call__(self, obs: dict, rpyt: NDArray[np.floating]) -> NDArray[np.floating]:
        """Return the motor RPMs (gym-pybullet-drones scale) for one attitude step."""
        ang_vel = body_ang_vel(obs)
        force, torque, self.r_int_error = self.attitude2force_torque(
            obs["quat"], ang_vel, rpyt, self.prev_ang_vel, self.r_int_error, ATTITUDE_FREQ
        )
        self.prev_ang_vel = ang_vel
        rotor_vel = self.force_torque2rotor_vel(force, torque)
        if self.rotor_dynamics:
            rotor_vel = self.rotors(rotor_vel, ang_vel)
            self.rotor_torque = self.rotors.rotor_torque
        return self.rotors.rotor_vel2gpd(rotor_vel)


def step_attitude_loop(
    env: BaseAviary,
    attitude_control: DSLAttitudeControl | MellingerAttitudeControl,
    obs: dict,
    rpyt: NDArray[np.floating],
    n_steps: int,
) -> dict:
    """Run the attitude loop for one outer control period and return the new observation."""
    for _ in range(n_steps):
        rpms = attitude_control(obs, rpyt)
        env.rotor_torque = attitude_control.rotor_torque
        state, *_ = env.step(rpms[None, ...])
        obs = state2obs(state)
    return obs


def rollout(config: dict) -> None:
    """Simulate one rollout of the experiment and save its recording."""
    if config["sim"]["physics"] not in VARIANTS:
        raise ValueError(f"Physics {config['sim']['physics']} not in {tuple(VARIANTS)}.")
    if config["controller"]["type"] != "mellinger":
        raise ValueError(f"Unsupported GPD controller type: {config['controller']['type']}")
    if config["drone"]["model"] != "cf2x_L250":
        raise ValueError(f"Drone model {config['drone']['model']} not supported in gpd.")
    attitude_loop, drag = VARIANTS[config["sim"]["physics"]]
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
    initial_pos = np.array(pos_start)[None]
    initial_pos[..., 2] = GROUND_CLEARANCE  # At 0.0 the drone jumps from contact dynamics
    drag_matrix = np.asarray(drone_params["drag_matrix"]) if drag == "identified" else None
    aviary = identified_drone(CtrlAviary, drone_params, drag_matrix)
    env = aviary(
        drone_model=DroneModel.CF2X,
        num_drones=1,
        initial_xyzs=initial_pos,
        initial_rpys=np.zeros((1, 3)),
        physics=Physics.PYB_DRAG if drag == "pybullet" else Physics.PYB,
        neighbourhood_radius=10,
        pyb_freq=PHYSICS_FREQ,
        ctrl_freq=ATTITUDE_FREQ,
        gui=config["sim"]["render"],
    )
    state, _ = env.reset()
    obs = state2obs(state)
    obs["t"] = 0.0

    # Prepare controllers (not necessary for sim, but for consistency with real)
    controller = MellingerController(
        obs, trajectory, model, tilt_max=np.deg2rad(config["controller"]["tilt_max"]) or None
    )
    rotors = RotorModel(drone_params, env.KF)
    rotor = "-rotor" in attitude_loop
    if attitude_loop.startswith("dsl"):
        fwclip = attitude_loop.endswith("-fwclip")
        clip = FIRMWARE_TORQUE_CLIP if fwclip else DSL_TORQUE_CLIP
        attitude_control = DSLAttitudeControl(rotors if rotor else None, torque_clip=clip)
    else:
        attitude_control = MellingerAttitudeControl(model, rotors, rotor_dynamics=rotor)

    def render_pause(i: int):
        fps = config["sim"]["fps"]
        if config["sim"]["render"] and ((i * fps) % ctrl_freq) < fps:
            time.sleep(1 / ctrl_freq)

    # Take off
    # Fly drone to hover position (actually not necessary in sim, but for consistency with real)
    trajectory_pos, trajectory_yaw = TakeoffTrajectory(
        obs["pos"], obs["rpy"][2], pos_start, yaw_start
    ).setpoints(ctrl_freq)
    state2attitude = parametrize(mellinger_state2attitude, model)
    cmd, pos_err_i = np.zeros(16), np.zeros(3)

    for i, (pos, yaw) in enumerate(zip(trajectory_pos, trajectory_yaw, strict=True)):
        cmd[:3], cmd[9:13] = pos, R.from_euler("z", yaw).as_quat()
        rpyt, pos_err_i = state2attitude(
            obs["pos"], obs["quat"], obs["vel"], cmd, pos_err_i, ctrl_freq
        )
        obs = step_attitude_loop(env, attitude_control, obs, rpyt, steps_per_ctrl)
        render_pause(i)

    # Experiment
    i = len(trajectory_pos)  # firmware cannot step back, and we have advanced during takeoff
    obs["t"] = i / ctrl_freq
    controller._t_start = obs["t"]
    controller._t_start -= trajectory_config.start_time  # Negative for a soft start
    while True:
        action = controller(obs)
        obs = step_attitude_loop(env, attitude_control, obs, action, steps_per_ctrl)
        i += 1
        obs["t"] = i / ctrl_freq
        obs["u"] = action
        data.append(obs | {"t": obs["t"] - controller._t_start})  # Recorded in trajectory time
        if controller.done(obs):
            break
        render_pause(i)

    env.close()
    with open(next_recording_path(RECORDINGS_DIR, config), "wb") as f:
        pickle.dump(data, f)


def main(config: dict | None = None) -> None:
    """Repeat the rollout `n_rollouts` times, each saved under the next free index."""
    if config is None:
        config = toml.load(Path(__file__).parent / "config.toml")
    for _ in range(config["sim"]["n_rollouts"]):
        rollout(config)


if __name__ == "__main__":
    main()
