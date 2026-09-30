"""Behavior cloning agent for vision based flight, implemented in Flax."""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jp
import optax
from flax import linen as nn
from flax import struct
from flax.linen.initializers import orthogonal, zeros
from flax.training import train_state
from jax import Array


class ImageEncoder(nn.Module):
    """Strided convolutional encoder for a stack of grayscale frames."""

    features: tuple[int, ...] = (32, 64, 128, 128)
    embed_size: int = 256

    @nn.compact
    def __call__(self, frames: Array) -> Array:
        """Encode a frame stack of shape (batch, stack, height, width)."""
        x = jp.moveaxis(frames, 1, -1).astype(jp.bfloat16)  # Move the stack into the channel axis
        for features in self.features:
            x = nn.Conv(features, (3, 3), strides=2, dtype=jp.bfloat16)(x)
            x = nn.GroupNorm(num_groups=8, dtype=jp.bfloat16)(x)
            x = nn.relu(x)
        # The action depends on where things are in the frame, so flatten instead of pooling
        flat = nn.Dense(self.embed_size, dtype=jp.bfloat16)(x.reshape(x.shape[0], -1))
        return nn.relu(flat).astype(jp.float32)


class VisionPolicyNet(nn.Module):
    """Policy mapping a frame stack, an IMU reading and the goal to action increments.

    As auxiliary output, it also predicts the drone's pose and velocities.
    """

    features: tuple[int, ...] = (32, 64, 128, 128)
    embed_size: int = 256
    hidden_size: int = 256
    act_dim: int = 4

    @nn.compact
    def __call__(self, obs: dict[str, Array]) -> tuple[Array, Array]:
        """Predict an action in [-1, 1] and the drone state from an observation dict."""
        embedding = ImageEncoder(features=self.features, embed_size=self.embed_size)(obs["frames"])
        inputs = [embedding, obs["ang_vel"], obs["acc"], obs["goal"], obs["actions"]]
        x = jp.concatenate(inputs, axis=-1)
        x = nn.relu(nn.Dense(self.hidden_size, kernel_init=orthogonal(), bias_init=zeros)(x))
        state = nn.Dense(18, kernel_init=orthogonal(0.01), bias_init=zeros)(x)
        action = nn.tanh(nn.Dense(self.act_dim, kernel_init=orthogonal(0.01), bias_init=zeros)(x))
        return action, state


class Agent(struct.PyTreeNode):
    """Behavior cloning agent holding the policy parameters and the optimizer state."""

    policy_state: train_state.TrainState = struct.field(pytree_node=True)
    # Stored as static so JIT treats them as compile-time constants and reinit avoids recompilation.
    policy: nn.Module = struct.field(pytree_node=False)
    tx: optax.GradientTransformation = struct.field(pytree_node=False)
    action: Callable = struct.field(pytree_node=False)
    aux_weight: float = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        key: Array,
        obs: dict[str, Array],
        policy: nn.Module,
        decay_steps: int,
        learning_rate: float = 3e-4,
        max_grad_norm: float = 1.0,
        aux_weight: float = 0.0,
    ) -> Agent:
        """Create an agent with parameters initialized from a sample observation.

        Args:
            key: Random key for parameter initialization.
            obs: Sample observation used to infer the input shapes.
            policy: Policy module mapping an observation to an action in [-1, 1].
            decay_steps: Cosine decay steps.
            learning_rate: Adam learning rate.
            max_grad_norm: Global norm the gradients are clipped to.
            aux_weight: Weight on the state prediction.

        Returns:
            An agent with initialized policy and optimizer state.
        """
        params = policy.init(key, obs)
        rate = optax.cosine_decay_schedule(learning_rate, decay_steps)
        tx = optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(rate))
        policy_state = train_state.TrainState.create(apply_fn=policy.apply, params=params, tx=tx)

        def action(agent: Agent, obs: dict[str, Array]) -> Array:
            """Deterministic action for an observation batch."""
            return agent.policy_state.apply_fn(agent.policy_state.params, obs)[0]

        return cls(
            policy_state=policy_state,
            policy=policy,
            tx=tx,
            action=jax.jit(action),
            aux_weight=aux_weight,
        )

    def update(
        self, obs: dict[str, Array], target: Array, state: Array
    ) -> tuple[Agent, dict[str, Array]]:
        """Take one gradient step on the imitation loss and the state prediction.

        Args:
            obs: Batch of observations.
            target: Expert actions in [-1, 1] with shape (batch, act_dim).
            state: True drone state as a position, a flattened rotation matrix, a velocity and
                an angular velocity, shape (batch, 18).

        Returns:
            The updated agent and a metrics dict holding the imitation loss, the position and
            velocity errors in SI units and the rotation error as a Frobenius norm.
        """

        def loss_fn(params: dict) -> tuple[Array, dict[str, Array]]:
            action, predicted = self.policy_state.apply_fn(params, obs)
            imitation = jp.mean(jp.abs(action - target))
            pos = jp.mean(jp.abs(predicted[:, :3] - state[:, :3]))
            # Frobenius norm of the rotation difference, which grows with the angle between them
            rot = jp.mean(jp.linalg.norm(predicted[:, 3:12] - state[:, 3:12], axis=-1))
            vel = jp.mean(jp.abs(predicted[:, 12:15] - state[:, 12:15]))
            ang_vel = jp.mean(jp.abs(predicted[:, 15:] - state[:, 15:]))
            metrics = {"loss": imitation, "pos": pos, "rot": rot, "vel": vel, "ang_vel": ang_vel}
            return imitation + self.aux_weight * (pos + rot + vel + ang_vel), metrics

        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(self.policy_state.params)
        return self.replace(policy_state=self.policy_state.apply_gradients(grads=grads)), metrics
