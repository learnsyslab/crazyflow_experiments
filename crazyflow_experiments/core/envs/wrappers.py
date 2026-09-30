"""wrappers for struct.PyTreeNode-style environments."""

from __future__ import annotations

import functools
import os
from typing import Any, Callable

os.environ["SCIPY_ARRAY_API"] = "1"

import flax.struct as struct
import jax
import jax.numpy as jp
import numpy as np
from gymnasium import spaces
from gymnasium.spaces import flatten_space
from gymnasium.vector.utils import batch_space
from jax import Array
from scipy.spatial.transform import Rotation as R

from crazyflow_experiments.core.envs.reach_pos_env import ReachPosEnv


# region Base
@struct.dataclass
class Wrapper(struct.PyTreeNode):
    """Base class for jittable wrappers that delegates common metadata to the wrapped base.

    Subclasses provide jitted `step` and `reset` callables and may override any of the spaces.
    """

    base: struct.PyTreeNode = struct.field(pytree_node=True)

    @property
    def single_observation_space(self) -> spaces.Space:
        """Return the single-observation Space of the wrapped base."""
        return getattr(self.base, "single_observation_space")

    @property
    def observation_space(self) -> spaces.Space:
        """Return the (possibly batched) observation Space of the wrapped base."""
        return getattr(self.base, "observation_space")

    @property
    def single_action_space(self) -> spaces.Space:
        """Return the single-action Space of the wrapped base."""
        return getattr(self.base, "single_action_space")

    @property
    def action_space(self) -> spaces.Space:
        """Return the (possibly batched) action Space of the wrapped base."""
        return getattr(self.base, "action_space")

    @property
    def num_envs(self) -> int:
        """Return the number of parallel environments from the wrapped base."""
        return getattr(self.base, "num_envs")

    @property
    def unwrapped(self) -> struct.PyTreeNode:
        """Return the unwrapped (innermost) base environment."""
        return getattr(self.base, "unwrapped", self.base)

    @property
    def steps(self) -> Array:
        """Get the current step count for each environment."""
        return getattr(self.base, "steps")

    @staticmethod
    def recursive_replace(env: struct.PyTreeNode, **kwargs: Any) -> struct.PyTreeNode:
        """Recursively replace fields in the innermost base environment."""
        if isinstance(env, Wrapper):
            new_base = Wrapper.recursive_replace(env.base, **kwargs)
            return env.replace(base=new_base)
        return env.replace(**kwargs)

    def render(self, **kwargs: dict) -> None:
        """Returns the render mode from the base vector environment."""
        return self.base.render(**kwargs)

    def close(self, **kwargs: Any) -> None:
        """Close all environments."""
        return self.base.close(**kwargs)


# region NormAct
@struct.dataclass
class NormalizeActions(Wrapper):
    """Wrapper that exposes actions in [-1, 1] and rescales them to the simulator range."""

    base: struct.PyTreeNode = struct.field(pytree_node=True)

    step: Callable = struct.field(pytree_node=False)
    reset: Callable = struct.field(pytree_node=False)

    @property
    def single_action_space(self) -> spaces.Space:
        """Expose a normalized single-action space in [-1, 1]."""
        base_space = self.base.single_action_space
        return spaces.Box(-1.0, 1.0, shape=base_space.shape, dtype=base_space.dtype)

    @property
    def action_space(self) -> spaces.Space:
        """Batched action space matching the wrapper's num_envs."""
        return batch_space(self.single_action_space, self.num_envs)

    @classmethod
    def create(cls, base: struct.PyTreeNode) -> NormalizeActions:
        """Create a NormalizeActions wrapper around `base`.

        Args:
            base: The jittable base environment to wrap.

        Returns:
            NormalizeActions: wrapper with jitted step/reset.
        """
        # Read simulator action bounds from base (may be numpy arrays)
        action_sim_low = jp.array(base.single_action_space.low)
        action_sim_high = jp.array(base.single_action_space.high)

        # Precompute scale and mean for rescaling from [-1,1] -> [low, high]
        scale = (action_sim_high - action_sim_low) / 2.0
        mean = (action_sim_high + action_sim_low) / 2.0

        def reset(
            env: NormalizeActions, *, seed: int | None = None, options: dict | None = None
        ) -> tuple[NormalizeActions, tuple[Any, Any]]:
            base_env, (obs, info) = env.base.reset(env.base, seed=seed, options=options)
            env = env.replace(base=base_env)
            return env, (obs, info)

        def step(env: NormalizeActions, actions: Array) -> tuple[NormalizeActions, tuple[Any, ...]]:
            # actions are expected in [-1, 1]. Clip and rescale to simulator range
            action = jp.clip(actions, -1.0, 1.0) * scale + mean
            base_env, (obs, reward, terminated, truncated, info) = env.base.step(env.base, action)
            env = env.replace(base=base_env)
            return env, (obs, reward, terminated, truncated, info)

        return cls(base=base, step=step, reset=reset)


# region NormActQuadratic
def quadratic_action_coefficients(
    low: Array, high: Array, neutral: Array
) -> tuple[Array, Array, Array]:
    """Compute the coefficients of the f(a) = k0 + k1 * a + k2 * a^2 action mapping.

    The quadratic sends -1, 0 and 1 to low, neutral and high. Where that curve would turn around
    inside [-1, 1], the coefficients fall back to the widest quadratic through neutral that stays
    monotonic, which sacrifices one of the two bounds.

    Args:
        low: Action lower bound, mapped to -1.
        high: Action upper bound, mapped to 1.
        neutral: Action neutral level, mapped to 0.

    Returns:
        The coefficients k0, k1 and k2.
    """
    k0 = neutral
    k1 = (high - low) / 2.0
    k2 = (high + low - 2.0 * neutral) / 2.0
    is_monotonic = jp.abs(k1) >= 2.0 * jp.abs(k2)
    k2_fix_high = (neutral - low) / 2.0
    k2_fix_low = (neutral - high) / 2.0
    k1_fixed = jp.where(k2 > 0, 2.0 * k2_fix_high, -2.0 * k2_fix_low)
    k2_fixed = jp.where(k2 > 0, k2_fix_high, k2_fix_low)
    return k0, jp.where(is_monotonic, k1, k1_fixed), jp.where(is_monotonic, k2, k2_fixed)


@struct.dataclass
class NormalizeActionsQuadratic(Wrapper):
    """Wrapper that exposes actions in [-1, 1] and rescales them to the simulator range."""

    base: struct.PyTreeNode = struct.field(pytree_node=True)

    step: Callable = struct.field(pytree_node=False)
    reset: Callable = struct.field(pytree_node=False)

    @property
    def single_action_space(self) -> spaces.Space:
        """Expose a normalized single-action space in [-1, 1]."""
        base_space = self.base.single_action_space
        return spaces.Box(-1.0, 1.0, shape=base_space.shape, dtype=base_space.dtype)

    @property
    def action_space(self) -> spaces.Space:
        """Batched action space matching the wrapper's num_envs."""
        return batch_space(self.single_action_space, self.num_envs)

    @classmethod
    def create(
        cls,
        base: struct.PyTreeNode,
        neutral: Array | None = None,
        action_low: Array | None = None,
        action_high: Array | None = None,
    ) -> NormalizeActionsQuadratic:
        """Create a NormalizeActionsQuadratic wrapper around `base`.

        Args:
            base: The jittable base environment to wrap.
            neutral: Action neutral level, will be mapped to 0. Use quadratic mapping if not None.
            action_low: Action lower bound, will be mapped to -1.
            action_high: Action upper bound, will be mapped to 1.

        Returns:
            NormalizeActionsQuadratic: wrapper with jitted step/reset.
        """
        # Read simulator action bounds from base (may be numpy arrays)
        low = jp.array(base.single_action_space.low) if action_low is None else jp.array(action_low)
        high = (
            jp.array(base.single_action_space.high)
            if action_high is None
            else jp.array(action_high)
        )
        if neutral is None:
            k0 = (high + low) / 2.0
            k1 = (high - low) / 2.0
            k2 = jp.zeros_like(k0)
        else:
            k0, k1, k2 = quadratic_action_coefficients(low, high, neutral)

        def _reset(
            env: NormalizeActionsQuadratic, *, seed: int | None = None, options: dict | None = None
        ) -> tuple[NormalizeActionsQuadratic, tuple[Any, Any]]:
            base_env, (obs, info) = env.base.reset(env.base, seed=seed, options=options)
            env = env.replace(base=base_env)
            return env, (obs, info)

        def _step(
            env: NormalizeActionsQuadratic, actions: Array
        ) -> tuple[NormalizeActionsQuadratic, tuple[Any, ...]]:
            # actions are expected in [-1, 1]. Clip and rescale to simulator range
            # actions = jp.clip(actions, -1.0, 1.0)
            actions = k0 + k1 * actions + k2 * (actions**2)
            base_env, (obs, reward, terminated, truncated, info) = env.base.step(env.base, actions)
            env = env.replace(base=base_env)
            return env, (obs, reward, terminated, truncated, info)

        return cls(base=base, step=jax.jit(_step), reset=jax.jit(_reset))


# region ZeroYaw
@struct.dataclass
class ZeroYaw(Wrapper):
    """Wrapper to set yaw output to zero."""

    base: struct.PyTreeNode = struct.field(pytree_node=True)

    step: Callable = struct.field(pytree_node=False)
    reset: Callable = struct.field(pytree_node=False)

    @classmethod
    def create(cls, base: struct.PyTreeNode) -> ZeroYaw:
        """Create an ZeroYaw around `base`.

        Args:
            base: The jittable base environment to wrap.

        Returns:
            ZeroYaw: A configured wrapper with jitted step/reset.
        """

        def reset(
            env: ZeroYaw, *, seed: int | None = None, options: dict | None = None
        ) -> tuple[ZeroYaw, tuple[Any, Any]]:
            base_env, (obs, info) = env.base.reset(env.base, seed=seed, options=options)
            env = env.replace(base=base_env)
            return env, (obs, info)

        def step(env: ZeroYaw, actions: Array) -> tuple[ZeroYaw, tuple[Any, ...]]:
            actions = actions.at[..., 2].set(0.0)
            base_env, (obs, reward, terminated, truncated, info) = env.base.step(env.base, actions)
            env = env.replace(base=base_env)
            return env, (obs, reward, terminated, truncated, info)

        return cls(base=base, step=step, reset=reset)


# region AngleReward
@struct.dataclass
class AngleReward(Wrapper):
    """Wrapper to penalize orientation in the reward."""

    base: struct.PyTreeNode = struct.field(pytree_node=True)

    step: Callable = struct.field(pytree_node=False)
    reset: Callable = struct.field(pytree_node=False)

    @classmethod
    def create(cls, base: struct.PyTreeNode, weight: float = 0.08) -> AngleReward:
        """Create an AngleReward around `base`.

        Args:
            base: The jittable base environment to wrap.
            weight: Weight used to penalize angle magnitudes in rewards.

        Returns:
            AngleReward: A configured wrapper with jitted step/reset.
        """

        def reset(
            env: AngleReward, *, seed: int | None = None, options: dict | None = None
        ) -> tuple[AngleReward, tuple[Any, Any]]:
            base_env, (obs, info) = env.base.reset(env.base, seed=seed, options=options)
            env = env.replace(base=base_env)
            return env, (obs, info)

        def _reward(reward: Array, observations: dict[str, Array]) -> Array:
            """Additional angular rewards."""
            return reward - weight * R.from_quat(observations["quat"]).magnitude()

        def step(env: AngleReward, actions: Array) -> tuple[AngleReward, tuple[Any, ...]]:
            base_env, (obs, reward, terminated, truncated, info) = env.base.step(env.base, actions)
            reward = _reward(reward, obs)
            env = env.replace(base=base_env)
            return env, (obs, reward, terminated, truncated, info)

        return cls(base=base, step=step, reset=reset)


# region VelocityPenalty


@struct.dataclass
class VelocityPenalty(Wrapper):
    """Wrapper to penalize drone velocities exceeding a threshold with a quadratic penalty."""

    base: struct.PyTreeNode = struct.field(pytree_node=True)

    step: Callable = struct.field(pytree_node=False)
    reset: Callable = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls, base: struct.PyTreeNode, threshold: float = 2.75, weight: float = 0.1
    ) -> VelocityPenalty:
        """Create a VelocityPenalty around `base`.

        Args:
            base: The jittable base environment to wrap.
            threshold: Velocity magnitude threshold in m/s above which penalty is applied.
            weight: Penalty weight scaling the quadratic excess.

        Returns:
            VelocityPenalty: A configured wrapper with jitted step/reset.
        """

        def reset(
            env: VelocityPenalty, *, seed: int | None = None, options: dict | None = None
        ) -> tuple[VelocityPenalty, tuple[Any, Any]]:
            base_env, (obs, info) = env.base.reset(env.base, seed=seed, options=options)
            env = env.replace(base=base_env)
            return env, (obs, info)

        def step(env: VelocityPenalty, actions: Array) -> tuple[VelocityPenalty, tuple[Any, ...]]:
            base_env, (obs, reward, terminated, truncated, info) = env.base.step(env.base, actions)
            speed = jp.linalg.norm(obs["vel"], axis=-1)  # (num_envs,)
            excess = jp.maximum(0.0, speed - threshold)
            reward = reward - weight * excess**2
            env = env.replace(base=base_env)
            return env, (obs, reward, terminated, truncated, info)

        return cls(base=base, step=step, reset=reset)


# region ActionPenalty
@struct.dataclass
class ActionPenalty(Wrapper):
    """Wrapper to apply action penalty and augment observations with last_action."""

    base: struct.PyTreeNode | AngleReward = struct.field(pytree_node=True)
    last_actions: Array = struct.field(pytree_node=True)
    num_actions: int = struct.field(pytree_node=False)

    step: Callable = struct.field(pytree_node=False)
    reset: Callable = struct.field(pytree_node=False)

    @property
    def single_observation_space(self) -> spaces.Space:
        """Return the base single_observation_space augmented with action_history."""
        spec = {k: v for k, v in self.base.single_observation_space.items()}
        act_dim = self.base.action_space.shape[-1]
        spec["last_actions"] = spaces.Box(
            -np.inf, np.inf, shape=(self.num_actions, act_dim), dtype=np.float32
        )
        return spaces.Dict(spec)

    @property
    def observation_space(self) -> spaces.Space:
        """Batched observation space matching the wrapper's num_envs."""
        return batch_space(self.single_observation_space, self.num_envs)

    @classmethod
    def create(
        cls,
        base: struct.PyTreeNode | AngleReward,
        num_actions: int = 1,
        init_last_actions: tuple | None = None,
        hover_action: Array = jp.zeros((4,)),
        act_coefs: tuple = (0.0,) * 4,
        d_act_coefs: tuple = (0.0,) * 4,
    ) -> ActionPenalty:
        """Add `last_action` to the observation and penalise the reward based on the action.

        Args:
            base: The jittable environment to wrap.
            num_actions: Number of past actions kept in the observation.
            init_last_actions: Actions every episode starts from, the automatic resets included.
            hover_action: Action treated as the neutral hover reference.
            act_coefs: Coefficients penalising the action magnitude.
            d_act_coefs: Coefficients penalising the change between consecutive actions.

        Returns:
            ActionPenalty: Configured wrapper instance.
        """
        num_envs = base.num_envs
        act_dim = base.action_space.shape[-1]
        act_coefs = jp.array(act_coefs, dtype=jp.float32)
        d_act_coefs = jp.array(d_act_coefs, dtype=jp.float32)
        # last_actions is part of the observation dict (computed in property)
        last_actions = jp.zeros((num_envs, num_actions, act_dim), dtype=jp.float32)
        if init_last_actions is not None:
            init_last_actions = jp.array(init_last_actions, dtype=jp.float32)
            last_actions = jp.broadcast_to(init_last_actions, last_actions.shape)

        def reset(
            env: ActionPenalty, *, seed: int | None = None, options: dict | None = None
        ) -> tuple[ActionPenalty, tuple[Any, Any]]:
            base_env, (obs, info) = env.base.reset(env.base, seed=seed, options=options)
            env = env.replace(base=base_env, last_actions=last_actions)
            obs["last_actions"] = env.last_actions
            return env, (obs, info)

        def step(env: ActionPenalty, action: Array) -> tuple[ActionPenalty, tuple[Any, ...]]:
            # Episodes the base step starts anew, marked by environments that reset in the next step
            restarted = env.unwrapped._marked_for_reset
            base_env, (obs, reward, terminated, truncated, info) = env.base.step(env.base, action)
            # energy
            action_deviation = action - hover_action
            reward = reward - jp.sum(act_coefs * (action_deviation**2), axis=-1)
            # smoothness
            action_diff = action - env.last_actions[:, 0, :]
            reward = reward - jp.sum(d_act_coefs * (action_diff**2), axis=-1)
            # update action history
            new_last_actions = jp.roll(env.last_actions, shift=1, axis=1)
            new_last_actions = new_last_actions.at[:, 0, :].set(action)
            new_last_actions = jp.where(restarted[:, None, None], last_actions, new_last_actions)
            env = env.replace(base=base_env, last_actions=new_last_actions)
            # flatten action history for observation: (num_envs, num_actions * act_dim)
            obs["last_actions"] = env.last_actions
            if "final_obs" in info:
                info["final_obs"]["last_actions"] = env.last_actions
            return env, (obs, reward, terminated, truncated, info)

        return cls(
            base=base, last_actions=last_actions, num_actions=num_actions, step=step, reset=reset
        )


# region StackFrames
@struct.dataclass
class StackFrames(Wrapper):
    """Wrapper that replaces the camera frame with a strided stack of the recent frames."""

    base: struct.PyTreeNode = struct.field(pytree_node=True)
    frames: Array = struct.field(pytree_node=True)
    stack: int = struct.field(pytree_node=False)
    drop: float = struct.field(pytree_node=False)
    delay: Array = struct.field(pytree_node=True)

    step: Callable = struct.field(pytree_node=False)
    reset: Callable = struct.field(pytree_node=False)

    @property
    def single_observation_space(self) -> spaces.Space:
        """Return the base space with the single frame replaced by the stack."""
        spec = {k: v for k, v in self.base.single_observation_space.items() if k != "frame"}
        frame = self.base.single_observation_space["frame"]
        spec["frames"] = spaces.Box(
            frame.low.min(), frame.high.max(), (self.stack, *frame.shape[1:]), frame.dtype
        )
        return spaces.Dict(spec)

    @property
    def observation_space(self) -> spaces.Space:
        """Batched observation space matching the wrapper's num_envs."""
        return batch_space(self.single_observation_space, self.num_envs)

    @classmethod
    def create(
        cls,
        base: struct.PyTreeNode,
        stack: int = 1,
        stride: int = 1,
        drop: float = 0.0,
        latency: tuple[int, int] = (0, 0),
    ) -> StackFrames:
        """Stack the recent camera frames into the observation."""
        assert stack > 0 and stride > 0, f"cannot stack {stack} frames every {stride} steps"
        frame = base.single_observation_space["frame"]
        # Keep intermediate frames so they later drop into it, and the delayed ones only so the
        # stack can reach back past them
        low, high = latency
        assert 0 <= low <= high, f"cannot delay between {low} and {high} steps"
        depth = (stack - 1) * stride + 1 + high
        frames = jp.zeros((base.num_envs, depth, *frame.shape[1:]), dtype=jp.float32)
        offsets = jp.arange(stack) * stride  # Newest first, then stride back

        def observe(delay: Array, frames: Array, obs: dict[str, Array]) -> dict[str, Array]:
            index = depth - 1 - (delay[:, None] + offsets)
            stacked = jp.take_along_axis(frames, index[:, :, None, None], axis=1)
            return {k: v for k, v in obs.items() if k != "frame"} | {"frames": stacked}

        def reset(
            env: StackFrames, *, seed: int | None = None, options: dict | None = None
        ) -> tuple[StackFrames, tuple[Any, Any]]:
            base_env, (obs, info) = env.base.reset(env.base, seed=seed, options=options)
            frames = jp.broadcast_to(obs["frame"], env.frames.shape)
            key = jax.random.fold_in(base_env.unwrapped.data.core.rng_key, 2)
            delay = jax.random.randint(key, (base_env.num_envs,), low, high + 1)
            observation = observe(delay, frames, obs)
            return env.replace(base=base_env, frames=frames, delay=delay), (observation, info)

        def step(env: StackFrames, action: Array) -> tuple[StackFrames, tuple[Any, ...]]:
            base_env, (obs, reward, terminated, truncated, info) = env.base.step(env.base, action)
            # A dropped frame means repeating the last frame
            key = jax.random.fold_in(base_env.unwrapped.data.core.rng_key, 1)
            dropped = jax.random.uniform(key, (base_env.num_envs, 1, 1, 1)) < env.drop
            frame = jp.where(dropped, env.frames[:, -1:], obs["frame"])
            rolled = jp.concatenate([env.frames[:, 1:], frame], axis=1)
            # A world that just reset has no past, so we fill it with the first frame
            done = (terminated | truncated)[:, None, None, None]
            frames = jp.where(done, jp.broadcast_to(obs["frame"], env.frames.shape), rolled)
            key = jax.random.fold_in(base_env.unwrapped.data.core.rng_key, 2)
            delay = jax.random.randint(key, (base_env.num_envs,), low, high + 1)
            delay = jp.where(terminated | truncated, delay, env.delay)
            env = env.replace(base=base_env, frames=frames, delay=delay)
            return env, (observe(delay, frames, obs), reward, terminated, truncated, info)

        delay = jp.zeros((base.num_envs,), jp.int32)
        return cls(
            base=base, frames=frames, stack=stack, drop=drop, delay=delay, step=step, reset=reset
        )


# region ActionDelay
@struct.dataclass
class ActionDelay(Wrapper):
    """Wrapper that delays actions.

    Must wrap before ActionHistory to avoid giving the policy stale actions in the observation.
    """

    base: struct.PyTreeNode = struct.field(pytree_node=True)
    queue: Array = struct.field(pytree_node=True)
    delay: Array = struct.field(pytree_node=True)

    step: Callable = struct.field(pytree_node=False)
    reset: Callable = struct.field(pytree_node=False)

    @classmethod
    def create(cls, base: struct.PyTreeNode, latency: tuple[int, int] = (0, 0)) -> ActionDelay:
        """Delay the actions reaching the environment by a per episode number of steps."""
        low, high = latency
        assert 0 <= low <= high, f"cannot delay between {low} and {high} steps"
        queue = jp.zeros((base.num_envs, high + 1, base.single_action_space.shape[-1]))

        def draw(base_env: struct.PyTreeNode) -> Array:
            key = jax.random.fold_in(base_env.unwrapped.data.core.rng_key, 3)
            return jax.random.randint(key, (base_env.num_envs,), low, high + 1)

        def reset(
            env: ActionDelay, *, seed: int | None = None, options: dict | None = None
        ) -> tuple[ActionDelay, tuple[Any, Any]]:
            base_env, (obs, info) = env.base.reset(env.base, seed=seed, options=options)
            queue = jp.zeros_like(env.queue)
            return env.replace(base=base_env, queue=queue, delay=draw(base_env)), (obs, info)

        def step(env: ActionDelay, action: Array) -> tuple[ActionDelay, tuple[Any, ...]]:
            queue = jp.concatenate([env.queue[:, 1:], action[:, None, :]], axis=1)
            # The drone flies the command it was handed this episode's delay ago
            index = queue.shape[1] - 1 - env.delay
            delayed = jp.take_along_axis(queue, index[:, None, None], axis=1)[:, 0]
            base_env, (obs, reward, terminated, truncated, info) = env.base.step(env.base, delayed)
            done = terminated | truncated
            queue = jp.where(done[:, None, None], 0.0, queue)
            delay = jp.where(done, draw(base_env), env.delay)
            env = env.replace(base=base_env, queue=queue, delay=delay)
            return env, (obs, reward, terminated, truncated, info)

        return cls(
            base=base,
            queue=queue,
            delay=jp.zeros((base.num_envs,), jp.int32),
            step=step,
            reset=reset,
        )


# region ActionHistory
@struct.dataclass
class ActionHistory(Wrapper):
    """Wrapper that adds the recent actions to the observation, newest first."""

    base: struct.PyTreeNode = struct.field(pytree_node=True)
    actions: Array = struct.field(pytree_node=True)
    length: int = struct.field(pytree_node=False)

    step: Callable = struct.field(pytree_node=False)
    reset: Callable = struct.field(pytree_node=False)

    @property
    def single_observation_space(self) -> spaces.Space:
        """Return the base space with the recent actions added."""
        space = self.base.single_action_space
        width = self.length * space.shape[-1]
        spec = dict(self.base.single_observation_space)
        spec["actions"] = spaces.Box(space.low.min(), space.high.max(), (width,), space.dtype)
        return spaces.Dict(spec)

    @property
    def observation_space(self) -> spaces.Space:
        """Batched observation space matching the wrapper's num_envs."""
        return batch_space(self.single_observation_space, self.num_envs)

    @classmethod
    def create(cls, base: struct.PyTreeNode, length: int = 1) -> ActionHistory:
        """Add past actions to the observation."""
        assert length > 0, f"cannot keep {length} actions"
        actions = jp.zeros((base.num_envs, length, base.single_action_space.shape[-1]))

        def observe(actions: Array, obs: dict[str, Array]) -> dict[str, Array]:
            return obs | {"actions": actions.reshape(actions.shape[0], -1)}

        def reset(
            env: ActionHistory, *, seed: int | None = None, options: dict | None = None
        ) -> tuple[ActionHistory, tuple[Any, Any]]:
            base_env, (obs, info) = env.base.reset(env.base, seed=seed, options=options)
            actions = jp.zeros_like(env.actions)
            return env.replace(base=base_env, actions=actions), (observe(actions, obs), info)

        def step(env: ActionHistory, action: Array) -> tuple[ActionHistory, tuple[Any, ...]]:
            base_env, (obs, reward, terminated, truncated, info) = env.base.step(env.base, action)
            taken = jp.concatenate([action[:, None, :], env.actions[:, :-1]], axis=1)
            # A world that just reset has taken nothing yet in its new episode
            actions = jp.where((terminated | truncated)[:, None, None], 0.0, taken)
            env = env.replace(base=base_env, actions=actions)
            return env, (observe(actions, obs), reward, terminated, truncated, info)

        return cls(base=base, actions=actions, length=length, step=step, reset=reset)


# region PrivilegedCriticObs
@struct.dataclass
class PrivilegedCriticObs(Wrapper):
    """Wrapper for asymmetric actor-critic with privileged observations for TD3.

    The critic additionally sees the rotor velocities.
    """

    base: struct.PyTreeNode = struct.field(pytree_node=True)
    step: Callable = struct.field(pytree_node=False)
    reset: Callable = struct.field(pytree_node=False)

    @property
    def single_observation_space(self) -> spaces.Space:
        """Base single_observation_space augmented with `~privileged_obs`."""
        spec = {k: v for k, v in self.base.single_observation_space.items()}
        spec["~privileged_obs"] = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        return spaces.Dict(spec)

    @property
    def observation_space(self) -> spaces.Space:
        """Batched observation space matching the wrapper's num_envs."""
        return batch_space(self.single_observation_space, self.num_envs)

    @classmethod
    def create(cls, base: struct.PyTreeNode) -> PrivilegedCriticObs:
        """Create a PrivilegedCriticObs wrapper."""

        def _reset(
            env: PrivilegedCriticObs, *, seed: int | None = None, options: dict | None = None
        ) -> tuple[PrivilegedCriticObs, tuple[Any, Any]]:
            base_env, (obs, info) = env.base.reset(env.base, seed=seed, options=options)
            env = env.replace(base=base_env)

            # Add privileged obs: rotor velocities (key sorts last alphabetically)
            rotor_vel = env.unwrapped.data.states.rotor_vel[:, 0, :]  # (num_envs, 4)
            obs["~privileged_obs"] = rotor_vel

            return env, (obs, info)

        def _step(
            env: PrivilegedCriticObs, action: Array
        ) -> tuple[PrivilegedCriticObs, tuple[Any, ...]]:
            base_env, (obs, reward, terminated, truncated, info) = env.base.step(env.base, action)
            env = env.replace(base=base_env)
            rotor_vel = env.unwrapped.data.states.rotor_vel[:, 0, :]  # (num_envs, 4), post-reset
            obs["~privileged_obs"] = rotor_vel
            if "final_obs" in info:
                info["final_obs"]["~privileged_obs"] = info["final_rotor_vel"]

            return env, (obs, reward, terminated, truncated, info)

        return cls(base=base, step=jax.jit(_step), reset=jax.jit(_reset))


# region NormalizeObs
@struct.dataclass
class NormalizeObs(Wrapper):
    """Wrapper to normalize flattened observations with mean and scale.

    Apply this after FlattenJaxObservation.
    """

    base: struct.PyTreeNode = struct.field(pytree_node=True)

    step: Callable = struct.field(pytree_node=False)
    reset: Callable = struct.field(pytree_node=False)

    @classmethod
    def create(cls, base: struct.PyTreeNode, mean: Array, scale: Array) -> NormalizeObs:
        """Create a NormalizeObs wrapper.

        Args:
            base: The jittable environment to wrap (after FlattenJaxObservation).
            mean: Array of shape (obs_dim,) to subtract from observations.
            scale: Array of shape (obs_dim,) to divide observations by.
        """
        assert mean.shape == scale.shape == base.single_observation_space.shape, (
            "Mean/scale must match obs shape."
        )

        def _reset(
            env: NormalizeObs, *, seed: int | None = None, options: dict | None = None
        ) -> tuple[NormalizeObs, tuple[Any, Any]]:
            base_env, (obs, info) = env.base.reset(env.base, seed=seed, options=options)
            env = env.replace(base=base_env)
            norm_obs = (obs - mean) / scale
            return env, (norm_obs, info)

        def _step(env: NormalizeObs, action: Array) -> tuple[NormalizeObs, tuple[Any, ...]]:
            base_env, (obs, reward, terminated, truncated, info) = env.base.step(env.base, action)
            env = env.replace(base=base_env)
            norm_obs = (obs - mean) / scale
            if "final_obs" in info:
                info["final_obs"] = (info["final_obs"] - mean) / scale
            return env, (norm_obs, reward, terminated, truncated, info)

        return cls(base=base, step=jax.jit(_step), reset=jax.jit(_reset))


# region FlattenObs
@struct.dataclass
class FlattenJaxObservation(Wrapper):
    """Wrapper to flatten dict observations into a single array."""

    base: Any = struct.field(pytree_node=True)

    @property
    def single_observation_space(self) -> spaces.Space:
        """Flattened single-observation space derived from the base."""
        return flatten_space(self.base.single_observation_space)

    @property
    def observation_space(self) -> spaces.Space:
        """Flattened (batched) observation space derived from the base."""
        return flatten_space(self.base.observation_space)

    step: Callable = struct.field(pytree_node=False)
    reset: Callable = struct.field(pytree_node=False)

    @classmethod
    def create(cls, base: Any) -> FlattenJaxObservation:
        """Create a FlattenJaxObservation that concatenates dict observations.

        Args:
            base: The jittable environment to wrap.

        Returns:
            FlattenJaxObservation: Configured wrapper instance.
        """

        def flatten_obs(observations: dict[str, Array]) -> Array:
            keys = sorted(observations.keys())
            return jp.concatenate(
                [jp.reshape(observations[k], (observations[k].shape[0], -1)) for k in keys], axis=-1
            )

        def reset(
            env: FlattenJaxObservation, *, seed: int | None = None, options: dict | None = None
        ) -> tuple[FlattenJaxObservation, tuple[Array, Any]]:
            base_env, (obs, info) = env.base.reset(env.base, seed=seed, options=options)
            flat_obs = flatten_obs(obs)
            env = env.replace(base=base_env)
            return env, (flat_obs, info)

        def step(
            env: FlattenJaxObservation, action: Array
        ) -> tuple[FlattenJaxObservation, tuple[Array, Any]]:
            base_env, (obs, reward, terminated, truncated, info) = env.base.step(env.base, action)
            flat_obs = flatten_obs(obs)
            if "final_obs" in info:
                info["final_obs"] = flatten_obs(info["final_obs"])
            env = env.replace(base=base_env)
            return env, (flat_obs, reward, terminated, truncated, info)

        return cls(base=base, step=step, reset=reset)


# region TerminateOnMaxAngVel
@struct.dataclass
class TerminateOnMaxAngVel(Wrapper):
    """Wrapper to terminate episodes when max angular velocity is exceeded."""

    base: struct.PyTreeNode = struct.field(pytree_node=True)
    max_ang_vel: float = struct.field(pytree_node=False)

    step: Callable = struct.field(pytree_node=False)
    reset: Callable = struct.field(pytree_node=False)

    @classmethod
    def create(cls, base: struct.PyTreeNode, max_ang_vel: float = 100.0) -> TerminateOnMaxAngVel:
        """Create a TerminateOnMaxAngVel around `base`.

        Args:
            base: The jittable base environment to wrap.
            max_ang_vel: Maximum allowed angular velocity magnitude before termination.

        Returns:
            TerminateOnMaxAngVel: Configured wrapper instance.
        """

        def reset(
            env: TerminateOnMaxAngVel, *, seed: int | None = None, options: dict | None = None
        ) -> tuple[TerminateOnMaxAngVel, tuple[Any, Any]]:
            base_env, (obs, info) = env.base.reset(env.base, seed=seed, options=options)
            env = env.replace(base=base_env)
            return env, (obs, info)

        def step(
            env: TerminateOnMaxAngVel, action: Array
        ) -> tuple[TerminateOnMaxAngVel, tuple[Any, ...]]:
            base_env, (obs, reward, terminated, truncated, info) = env.base.step(env.base, action)
            ang_vel = obs["ang_vel"]  # (num_envs, 3)
            max_vel = jp.linalg.norm(ang_vel, axis=-1)  # (num_envs,)
            terminated = jp.logical_or(terminated, max_vel > env.max_ang_vel)
            env = env.replace(base=base_env)
            return env, (obs, reward, terminated, truncated, info)

        return cls(base=base, max_ang_vel=max_ang_vel, step=step, reset=reset)


# region QuatToMatrixObs
@struct.dataclass
class QuatToMatrixObs(Wrapper):
    """Wrapper to convert quaternion observations to rotation matrices."""

    base: struct.PyTreeNode = struct.field(pytree_node=True)

    step: Callable = struct.field(pytree_node=False)
    reset: Callable = struct.field(pytree_node=False)

    @property
    def single_observation_space(self) -> spaces.Space:
        """Base single_observation_space with 'quat' replaced by 'rot_mat'."""
        spec = {k: v for k, v in self.base.single_observation_space.items()}
        quat_space = spec.pop("quat")
        rot_mat_shape = quat_space.shape[:-1] + (3, 3)
        spec["rot_mat"] = spaces.Box(low=-1.0, high=1.0, shape=rot_mat_shape, dtype=np.float32)
        return spaces.Dict(spec)

    @property
    def observation_space(self) -> spaces.Space:
        """Batched observation space matching the wrapper's num_envs."""
        return batch_space(self.single_observation_space, self.num_envs)

    @classmethod
    def create(cls, base: struct.PyTreeNode) -> QuatToMatrixObs:
        """Create a QuatToMatrixObs around `base`.

        Args:
            base: The jittable base environment to wrap.

        Returns:
            QuatToMatrixObs: Configured wrapper instance.
        """

        def reset(
            env: QuatToMatrixObs, *, seed: int | None = None, options: dict | None = None
        ) -> tuple[QuatToMatrixObs, tuple[Any, Any]]:
            base_env, (obs, info) = env.base.reset(env.base, seed=seed, options=options)
            obs["rot_mat"] = R.from_quat(obs.pop("quat")).as_matrix()
            env = env.replace(base=base_env)
            return env, (obs, info)

        def step(env: QuatToMatrixObs, action: Array) -> tuple[QuatToMatrixObs, tuple[Any, ...]]:
            base_env, (obs, reward, terminated, truncated, info) = env.base.step(env.base, action)
            obs["rot_mat"] = R.from_quat(obs.pop("quat")).as_matrix()
            if "final_obs" in info:
                quat = info["final_obs"].pop("quat")
                info["final_obs"]["rot_mat"] = R.from_quat(quat).as_matrix()
            env = env.replace(base=base_env)
            return env, (obs, reward, terminated, truncated, info)

        return cls(base=base, step=step, reset=reset)


# region GoalRelativeObs
@struct.dataclass
class GoalRelativeObs(Wrapper):
    """Wrapper that reports position and velocity relative to the current trajectory target.

    Adapter for goal-conditioned policies to work with trajectory envs.
    """

    base: struct.PyTreeNode = struct.field(pytree_node=True)

    step: Callable = struct.field(pytree_node=False)
    reset: Callable = struct.field(pytree_node=False)

    @classmethod
    def create(cls, base: struct.PyTreeNode) -> GoalRelativeObs:
        """Create a GoalRelativeObs around the base env."""
        assert hasattr(base.unwrapped, "target_pos"), "Base env must expose target_pos"

        def _relative(env: GoalRelativeObs, obs: dict[str, Array]) -> dict[str, Array]:
            obs["pos"] = obs["pos"] - env.unwrapped.target_pos
            obs["vel"] = obs["vel"] - env.unwrapped.target_vel
            return obs

        def _reset(
            env: GoalRelativeObs, *, seed: int | None = None, options: dict | None = None
        ) -> tuple[GoalRelativeObs, tuple[Any, Any]]:
            base_env, (obs, info) = env.base.reset(env.base, seed=seed, options=options)
            env = env.replace(base=base_env)
            return env, (_relative(env, obs), info)

        def _step(env: GoalRelativeObs, action: Array) -> tuple[GoalRelativeObs, tuple[Any, ...]]:
            base_env, (obs, reward, terminated, truncated, info) = env.base.step(env.base, action)
            env = env.replace(base=base_env)
            if "final_obs" in info:
                info["final_obs"] = _relative(env, info["final_obs"])
            return env, (_relative(env, obs), reward, terminated, truncated, info)

        return cls(base=base, step=_step, reset=_reset)


# region ClipObs
@struct.dataclass
class ClipObs(Wrapper):
    """Wrapper to clip observations to specified bounds."""

    base: struct.PyTreeNode = struct.field(pytree_node=True)
    low: Array = struct.field(pytree_node=True)
    high: Array = struct.field(pytree_node=True)

    step: Callable = struct.field(pytree_node=False)
    reset: Callable = struct.field(pytree_node=False)

    @classmethod
    def create(cls, base: struct.PyTreeNode, low: float, high: float) -> ClipObs:
        """Create a ClipObs around `base`.

        Args:
            base: The jittable base environment to wrap.
            low: Minimum value for clipping observations.
            high: Maximum value for clipping observations.

        Returns:
            ClipObs: Configured wrapper instance.
        """
        low = jp.array(low, dtype=jp.float32)
        high = jp.array(high, dtype=jp.float32)

        def reset(
            env: ClipObs, *, seed: int | None = None, options: dict | None = None
        ) -> tuple[ClipObs, tuple[Any, Any]]:
            base_env, (obs, info) = env.base.reset(env.base, seed=seed, options=options)
            clipped_obs = {k: jp.clip(v, env.low, env.high) for k, v in obs.items()}
            env = env.replace(base=base_env)
            return env, (clipped_obs, info)

        def step(env: ClipObs, action: Array) -> tuple[ClipObs, tuple[Any, ...]]:
            base_env, (obs, reward, terminated, truncated, info) = env.base.step(env.base, action)
            clipped_obs = {k: jp.clip(v, env.low, env.high) for k, v in obs.items()}
            env = env.replace(base=base_env)
            return env, (clipped_obs, reward, terminated, truncated, info)

        return cls(base=base, low=low, high=high, step=step, reset=reset)


# region RecordData


def _record_write(
    buffer: Array, values: Array, step_idx: Array, env_idx: Array, recording: Array
) -> Array:
    """Write the values of each recording env into its own row, leaving the other rows untouched."""
    keep = buffer[step_idx, env_idx]
    return buffer.at[step_idx, env_idx].set(jp.where(recording[:, None], values, keep))


@struct.dataclass
class RecordData(Wrapper):
    """Wrapper that records debugging data."""

    base: struct.PyTreeNode = struct.field(pytree_node=True)

    _record_act: Array = struct.field(pytree_node=True)
    _record_pos: Array = struct.field(pytree_node=True)
    _record_goal: Array = struct.field(pytree_node=True)
    _record_rpy: Array = struct.field(pytree_node=True)
    _record_steps: Array = struct.field(pytree_node=True)
    _recording: Array = struct.field(pytree_node=True)

    step: Callable = struct.field(pytree_node=False)
    reset: Callable = struct.field(pytree_node=False)

    @classmethod
    def create(cls, base: struct.PyTreeNode) -> RecordData:
        """Create a RecordData wrapper around `base`.

        Args:
            base: The jittable environment to wrap.

        Returns:
            RecordData: Configured wrapper instance.
        """
        assert hasattr(base.unwrapped, "max_episode_time"), (
            "Base env must have max_episode_time attribute"
        )
        max_T = int(base.unwrapped.max_episode_time * base.unwrapped.freq)
        num_envs = int(base.num_envs)
        act_dim = int(base.action_space.shape[-1])
        pos_dim = 3

        # initialize buffers # TODO: this takes a lot of memory
        empty_act = jp.zeros((max_T, num_envs, act_dim), dtype=jp.float32)
        empty_pos = jp.zeros((max_T, num_envs, pos_dim), dtype=jp.float32)
        empty_goal = jp.zeros((max_T, num_envs, pos_dim), dtype=jp.float32)
        empty_rpy = jp.zeros((max_T, num_envs, pos_dim), dtype=jp.float32)
        empty_steps = jp.zeros((num_envs,), dtype=jp.int32)
        all_recording = jp.ones((num_envs,), dtype=jp.bool_)
        env_idx = jp.arange(num_envs)

        def _extract_data(base: RecordData) -> dict[str, Array]:
            """Extract recorded data as a dict of arrays."""
            raw = base.unwrapped
            pos = raw.data.states.pos[:, 0, :]
            if hasattr(raw, "trajectories"):
                goal = raw.trajectories[jp.arange(raw.steps.shape[0]), raw.steps.squeeze(1)]
            elif hasattr(raw, "goal_pos"):
                goal = raw.goal_pos
            else:
                goal = pos
            rpy = R.from_quat(raw.data.states.quat[:, 0, :]).as_euler("xyz")
            return pos, goal, rpy

        def _reset(
            env: RecordData, *, seed: int | None = None, options: dict | None = None
        ) -> tuple[RecordData, tuple[Any, Any]]:
            base_env, (obs, info) = env.base.reset(env.base, seed=seed, options=options)
            env = env.replace(
                base=base_env,
                _record_act=empty_act,
                _record_pos=empty_pos,
                _record_goal=empty_goal,
                _record_rpy=empty_rpy,
                _record_steps=empty_steps,
                _recording=all_recording,
            )
            return env, (obs, info)

        def _step(env: RecordData, action: Array) -> tuple[RecordData, tuple[Any, ...]]:
            # step the wrapped environment
            base_env, (obs, reward, terminated, truncated, info) = env.base.step(env.base, action)

            pos, goal, rpy = _extract_data(base_env)

            # Envs autoreset on the step that ends their episode, so the obs of that step already
            # belongs to the next episode and is dropped
            recording = env._recording & ~(terminated | truncated)
            write = functools.partial(
                _record_write, step_idx=env.steps[:, 0], env_idx=env_idx, recording=recording
            )

            env = env.replace(
                base=base_env,
                _record_act=write(env._record_act, action),
                _record_pos=write(env._record_pos, pos),
                _record_goal=write(env._record_goal, goal),
                _record_rpy=write(env._record_rpy, rpy),
                _record_steps=env._record_steps + recording.astype(jp.int32),
                _recording=recording,
            )
            return env, (obs, reward, terminated, truncated, info)

        return cls(
            base=base,
            _record_act=empty_act,
            _record_pos=empty_pos,
            _record_goal=empty_goal,
            _record_rpy=empty_rpy,
            _record_steps=empty_steps,
            _recording=all_recording,
            step=jax.jit(_step),
            reset=jax.jit(_reset),
        )

    def pos_error(self) -> np.ndarray:
        """Compute the goal distance in meters of every recorded step, flattened over all envs."""
        pos = np.array(self._record_pos)  # shape: (T, num_envs, 3)
        goal = np.array(self._record_goal)  # shape: (T, num_envs, 3)
        error = np.linalg.norm(pos - goal, axis=-1)  # shape: (T, num_envs)
        recorded = np.arange(error.shape[0])[:, None] < np.array(self._record_steps)
        assert recorded.any(), "No steps recorded"
        return error[recorded]

    def calc_rmse(self) -> float:
        """Compute RMSE between recorded position and goal (return in meters)."""
        return float(np.sqrt(np.mean(self.pos_error() ** 2)))


class CurriculumWrapper(Wrapper):
    """Reset positions once a certain success threshold is reached."""

    base: struct.PyTreeNode = struct.field(pytree_node=True)
    ema_success: float = struct.field(pytree_node=True)
    active: bool = struct.field(pytree_node=True)
    success_threshold: float = struct.field(pytree_node=False)
    ema_alpha: float = struct.field(pytree_node=False)
    goal_resample_interval: int = struct.field(pytree_node=False)

    reset: Callable = struct.field(pytree_node=False)
    step: Callable = struct.field(pytree_node=False)

    @staticmethod
    def create(
        base: struct.PyTreeNode,
        success_threshold: float,
        ema_alpha: float,
        goal_resample_interval: float,
        resample_min: Array,
        resample_max: Array,
        freq: int,
    ) -> struct.PyTreeNode:
        """Create a CurriculumWrapper around `base`."""
        assert isinstance(base.unwrapped, ReachPosEnv), "CurriculumWrapper must wrap a ReachPosEnv"
        ema_success = jp.array(0.0, dtype=jp.float32)
        active = jp.array(False)

        def reset(
            env: CurriculumWrapper, *, seed: int | None = None, options: dict | None = None
        ) -> tuple[CurriculumWrapper, tuple[Any, Any]]:
            base_env, (obs, info) = env.base.reset(env.base, seed=seed, options=options)
            env = env.replace(base=base_env, active=jp.array(False), ema_success=jp.array(0.0))
            return env, (obs, info)

        def resample_goals(env: CurriculumWrapper) -> CurriculumWrapper:
            data = env.unwrapped.data
            rng_key, subkey = jax.random.split(data.core.rng_key)
            data = data.replace(core=data.core.replace(rng_key=rng_key))
            env = env.recursive_replace(env, data=data)
            goal_pos = jax.random.uniform(
                subkey, shape=(env.num_envs, 3), minval=resample_min, maxval=resample_max
            )
            assert goal_pos.shape == env.unwrapped.goal_pos.shape
            steps = (data.core.steps[:, 0] // (data.core.freq // freq)).astype(jp.int32)
            resample = (steps % env.goal_resample_interval == 0) & (steps > 0)
            goal_pos = jp.where(resample[:, None], goal_pos, env.unwrapped.goal_pos)
            return env.recursive_replace(env, goal_pos=goal_pos)

        def step(
            env: CurriculumWrapper, action: Array
        ) -> tuple[CurriculumWrapper, tuple[Any, ...]]:
            base_env, (obs, reward, terminated, truncated, info) = env.base.step(env.base, action)
            success = jp.mean(info["success"], where=terminated | truncated)
            # If no env terminated or truncated, keep previous success to avoid NaNs
            success = jp.where(jp.isnan(success), env.ema_success, success)
            ema_success = env.ema_alpha * success + (1 - env.ema_alpha) * env.ema_success
            active = env.active | (ema_success >= env.success_threshold)
            env = env.replace(ema_success=ema_success, active=active, base=base_env)
            env = jax.lax.cond(env.active, resample_goals, lambda x: x, env)
            return env, (obs, reward, terminated, truncated, info)

        return CurriculumWrapper(
            base,
            ema_success,
            active,
            success_threshold,
            ema_alpha,
            goal_resample_interval,
            reset,
            step,
        )
