"""Analytic expert that flies the drone to a goal from its privileged state.

View the expert with
    pixi run -e vision python crazyflow_experiments/vision/expert.py
"""

from __future__ import annotations

import os

os.environ["SCIPY_ARRAY_API"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import functools
import logging
import time
from typing import TYPE_CHECKING, Callable

import fire
import jax
import jax.numpy as jp
import numpy as np
from crazyflow.control import Control, parametrize
from crazyflow.control.mellinger import state2attitude
from crazyflow.sim.splat import SplatViewer
from scipy.spatial.transform import Rotation as R

from crazyflow_experiments.control.loop import sleep_until_next_cycle
from crazyflow_experiments.core.envs.vision_reach_env import VisionReachEnv, setpoints
from crazyflow_experiments.vision.scenes import splat
from crazyflow_experiments.vision.viewer import add_drone, gray2rgb, update_drone_pose

if TYPE_CHECKING:
    from crazyflow.sim.data import SimData
    from jax import Array

logger = logging.getLogger(__name__)

DELTA_SMOOTH = 0.15  # Distance in m when the expert slows down
YAW_SMOOTH = 0.25
YAW_DEADBAND = 0.2
ACTION_RATE = 0.1  # Largest change between two steps
ATTITUDE_SPEED = 0.5  # Approach speed in m/s the attitude expert asks the cascade to hold
mass, mass_thrust = VisionReachEnv.MASS, VisionReachEnv.MASS_THRUST
CONTROLLER = functools.partial(
    parametrize(state2attitude, VisionReachEnv.DRONE, xp=jp), mass=mass, mass_thrust=mass_thrust
)


def state_command(data: SimData, goal: Array, yaw_fade: float = 0.0) -> tuple[Array, Array]:
    """World frame position increment and yaw increment toward the goal."""
    rot = R.from_quat(data.states.quat[:, 0, :])
    error = goal - data.states.pos[:, 0, :]
    distance = jp.linalg.norm(error, axis=-1, keepdims=True)
    bearing = jp.arctan2(error[:, 1], error[:, 0]) - rot.as_euler("xyz")[:, 2]
    yaw_error = jp.arctan2(jp.sin(bearing), jp.cos(bearing))
    horizontal = jp.linalg.norm(error[:, :2], axis=-1)
    if yaw_fade > 0.0:
        yaw_error = yaw_error * jp.clip((horizontal - YAW_DEADBAND) / yaw_fade, 0.0, 1.0)
    else:
        yaw_error = jp.where(horizontal < YAW_DEADBAND, 0.0, yaw_error)
    return error / (DELTA_SMOOTH + distance), yaw_error / (YAW_SMOOTH + jp.abs(yaw_error))


def smooth_actions(action: Array, previous: Array, fresh: Array) -> Array:
    """Hold the expert to a command rate a drone could follow."""
    limited = previous + jp.clip(action - previous, -ACTION_RATE, ACTION_RATE)
    # A reset teleports the drone, so the jump in its expert action is real and must not be eased
    return jp.where(fresh[:, None], action, limited)


def expert_state(data: SimData, goal: Array, yaw_fade: float = 0.0) -> Array:
    """Normalized expert action driving the drone toward the goal along a smooth path."""
    rot = R.from_quat(data.states.quat[:, 0, :])
    delta, dyaw = state_command(data, goal, yaw_fade)
    body_delta = rot.apply(delta, inverse=True)
    return jp.clip(jp.concatenate([body_delta, dyaw[:, None]], axis=-1), -1.0, 1.0)


def expert_attitude(
    data: SimData,
    goal: Array,
    yaw_fade: float = 0.0,
    brake_gain: float = 0.0,
    step_scale: float = 1.0,
    speed: float = ATTITUDE_SPEED,
) -> Array:
    """Normalized attitude command, taken from the cascade the state interface runs internally."""
    pos, quat, vel = data.states.pos[:, 0, :], data.states.quat[:, 0, :], data.states.vel[:, 0, :]
    rot = R.from_quat(quat)
    delta, dyaw = state_command(data, goal, yaw_fade)
    # The same setpoint the state interface uses
    action = jp.concatenate([rot.apply(delta, inverse=True), dyaw[:, None]], axis=-1)
    action = action.at[:, :3].multiply(step_scale)  # Reduce delta x
    setpoint, yaw = setpoints(pos, rot, jp.clip(action, -1.0, 1.0))
    zeros = jp.zeros((pos.shape[0], 3))
    if brake_gain > 0.0:
        error = goal - pos
        distance = jp.linalg.norm(error, axis=-1, keepdims=True)
        approach = jp.minimum(speed, brake_gain * distance)
        velocity = error / jp.maximum(distance, 1e-6) * approach  # Dampen approach speed
    else:
        # The attitude expert is too slow without a feedforward speed term
        velocity = delta * speed
    quat_cmd = R.from_euler("z", yaw[:, None]).as_quat()
    cmd = jp.concatenate([setpoint, velocity, zeros, quat_cmd, zeros], axis=-1)
    rpyt, _ = CONTROLLER(pos, quat, vel, cmd)
    thrust_min, thrust_max = VisionReachEnv.THRUST_MIN, VisionReachEnv.THRUST_MAX
    thrust = 2.0 * (rpyt[:, 3] - thrust_min) / (thrust_max - thrust_min) - 1.0
    tilt = rpyt[:, :2] / VisionReachEnv.TILT_MAX
    return jp.clip(jp.concatenate([tilt, dyaw[:, None], thrust[:, None]], axis=-1), -1.0, 1.0)


def build_expert(
    control: str,
    yaw_fade: float = 0.0,
    brake_gain: float = 0.0,
    step_scale: float = 1.0,
    speed: float = ATTITUDE_SPEED,
) -> Callable[..., Array]:
    """Select the expert matching the control interface the actions are sent through."""
    if control == Control.attitude:
        return functools.partial(
            expert_attitude,
            yaw_fade=yaw_fade,
            brake_gain=brake_gain,
            step_scale=step_scale,
            speed=speed,
        )
    return functools.partial(expert_state, yaw_fade=yaw_fade)


# The expert defaults are the ones the policies are trained against, see vision/train.py
def view(
    scene: str = "lab",
    variant: str = "lab_30k",
    control: str = "attitude",
    steps: int = 3000,
    yaw_fade: float = 0.2,
    brake_gain: float = 0.2,
    step_scale: float = 1.0,
    speed: float = 0.35,
):
    """Fly the expert to the environment's goals in the splat viewer."""
    env = VisionReachEnv.create(scene=splat(scene, variant), control=control)
    env, (obs, _) = env.reset(env)
    expert = build_expert(control, yaw_fade, brake_gain, step_scale, speed)
    command = jax.jit(
        lambda data, goal, previous, fresh: smooth_actions(expert(data, goal), previous, fresh)
    )

    viewer = SplatViewer(env.sim)
    handles = add_drone(viewer.viewer.server.scene)
    marker = viewer.viewer.server.scene.add_icosphere(
        "goal", radius=0.1, color=(255, 60, 60), opacity=0.4
    )
    panel = viewer.viewer.server.gui.add_image(gray2rgb(obs["frame"][0, 0]), label="Camera")
    logger.info(f"flying the {control} expert in {scene}")

    previous = jp.zeros((1, env.single_action_space.shape[0]))
    fresh = jp.ones((1,), bool)  # The reset teleported the drone, so its first command is a jump
    for _ in range(steps):
        cycle = time.perf_counter()
        action = command(env.data, env.goal_pos, previous, fresh)
        env, (obs, _, terminated, truncated, _) = env.step(env, action)
        previous, fresh = action, terminated | truncated
        update_drone_pose(handles, env.data)
        marker.position = tuple(np.asarray(env.goal_pos[0]))
        panel.image = gray2rgb(obs["frame"][0, 0])
        sleep_until_next_cycle(cycle, env.freq)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(view)
