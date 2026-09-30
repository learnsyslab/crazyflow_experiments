"""Fly a trained vision reach policy from camera frames, the IMU and a goal position."""

from __future__ import annotations

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from typing import TYPE_CHECKING, Any

import jax
import numpy as np
from flax import serialization
from scipy.spatial.transform import Rotation as R

from crazyflow_experiments.control.controller import Controller
from crazyflow_experiments.core.agents.vision_agent import VisionPolicyNet
from crazyflow_experiments.core.envs.vision_reach_env import attitudes, setpoints
from crazyflow_experiments.vision import MODEL_DIR

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


def checkpoint(stem: str) -> Path:
    """Locate a trained reach policy by the run it was saved under."""
    path = MODEL_DIR / f"reach_policy_{stem}.msgpack"
    assert path.exists(), f"no checkpoint at {path}"
    return path


def load_checkpoint(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read a checkpoint's policy parameters and the configuration."""
    saved = serialization.msgpack_restore(path.read_bytes())
    # Convert tuples that get serialized as dicts back to tuples
    config = {
        k: tuple(v.values()) if isinstance(v, dict) else v for k, v in saved["config"].items()
    }
    return saved["params"], config


class ReachController(Controller):
    """Turn rendered frames, an IMU reading and a goal into a state command for the drone."""

    def __init__(self, path: Path):
        """Rebuild the policy from its checkpoint and compile it, ready to be seeded with a frame.

        Args:
            path: Checkpoint written by a training run.
        """
        params, config = load_checkpoint(path)
        assert config["control"] in ("state", "attitude")
        self.mode = config["control"]
        self.resolution = tuple(config["resolution"])
        self.goal_radius = config["goal_radius"] * 2
        self.rest_speed = config["rest_speed"] * 2
        self.use_imu = config["use_imu"]
        self.stride = config["stride"]
        self.goal_low, self.goal_high = np.array(config["goal_low"]), np.array(config["goal_high"])
        self.spawn_low = np.array(config["spawn_low"])
        self.spawn_high = np.array(config["spawn_high"])
        policy = VisionPolicyNet(
            features=config["features"],
            embed_size=config["embed_size"],
            hidden_size=config["hidden_size"],
        )
        width, height = self.resolution
        # Deep enough that striding it gives the frames the policy was trained on, newest first
        depth = (config["stack"] - 1) * self.stride + 1
        self.frames = np.zeros((depth, height, width), np.float32)
        self.actions = np.zeros((config["command_history"], policy.act_dim), np.float32)
        obs = self._observation(np.zeros(3), np.zeros(3), np.zeros(3))
        self.params = serialization.from_state_dict(policy.init(jax.random.key(0), obs), params)
        self.policy = jax.jit(policy.apply)
        # store position and policy estimates
        self.pos = np.zeros(3, np.float32)
        self.quat = np.array([0.0, 0.0, 0.0, 1.0], np.float32)
        self.vel = np.zeros(3, np.float32)
        self.ang_vel = np.zeros(3, np.float32)
        self.warmup_jit()

    def warmup_jit(self):
        """Compile the policy before the first command is due."""
        obs = self._observation(np.zeros(3), np.zeros(3), np.zeros(3))
        jax.block_until_ready(self.policy(self.params, obs))

    def reset(self, frame: NDArray[np.floating]):
        """Reset the controller's frame stack and clear the action history.

        Args:
            frame: Camera frame the reach starts from, grayscale in [0, 1].
        """
        assert frame.shape == self.resolution, f"{frame.shape=} != {self.resolution=}"
        self.frames[:] = frame
        self.actions[:] = 0.0

    def _observation(self, goal: NDArray, ang_vel: NDArray, acc: NDArray) -> dict[str, Any]:
        """Assemble what the policy reads, with a leading batch axis of one."""
        ang_vel, acc = np.asarray(ang_vel, np.float32), np.asarray(acc, np.float32)
        if not self.use_imu:  # Trained blind, so the inertial channels held zeros
            ang_vel, acc = np.zeros(3, np.float32), np.zeros(3, np.float32)
        return {
            "frames": self.frames[:: self.stride][None],
            "ang_vel": ang_vel[None],
            "acc": acc[None],
            "goal": np.asarray(goal, np.float32)[None],
            "actions": self.actions.reshape(1, -1),
        }

    def __call__(self, obs: dict[str, Any]) -> NDArray[np.floating]:
        """Command the drone one step towards its goal."""
        self.frames = np.roll(self.frames, shift=1, axis=0)
        assert obs["frame"].shape == self.resolution, f"{obs['frame'].shape=} != {self.resolution=}"
        self.frames[0] = obs["frame"]
        observation = self._observation(obs["goal"], obs["ang_vel"], obs["acc"])
        prediction, estimate = self.policy(self.params, observation)
        action = np.clip(np.asarray(prediction)[0], -1.0, 1.0)
        self._update_estimate(np.asarray(estimate)[0])
        self.actions = np.roll(self.actions, shift=1, axis=0)
        self.actions[0] = action
        rot = R.from_euler("xyz", obs["rpy"], degrees=False)

        if self.mode == "attitude":
            rpy, thrust = attitudes(rot.as_euler("xyz")[2], action)
            return np.concatenate((rpy, thrust[None]), axis=-1)

        setpoint, yaw = setpoints(obs["pos"], rot, action)
        command = np.zeros(16)
        command[:3] = setpoint
        # command[3:6] = action[:3]
        command[9:13] = R.from_euler("z", yaw).as_quat()
        return command

    def _update_estimate(self, estimate: NDArray[np.floating]):
        """Read the drone's pose and motion out of the auxiliary head."""
        self.pos = estimate[:3]
        self.quat = R.from_matrix(estimate[3:12].reshape(3, 3)).as_quat()
        self.vel = estimate[12:15]
        self.ang_vel = estimate[15:]

    def done(self, obs: dict[str, Any]) -> bool:
        """Report whether the drone has settled on its goal."""
        arrived = np.linalg.norm(obs["pos"] - obs["goal"]) < self.goal_radius
        return bool(arrived and np.linalg.norm(obs["vel"]) < self.rest_speed)
