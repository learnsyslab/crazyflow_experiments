"""BPTT agent implementation using Flax."""

from __future__ import annotations

import jax
import jax.numpy as jp
import optax
from flax import linen as nn
from flax import struct
from flax.linen.initializers import orthogonal, zeros
from flax.training import train_state
from jax import Array


class ActorNet(nn.Module):
    """Class defining the actor-critic model."""

    hidden_size: int = 64
    act_dim: int = 4
    num_layers: int = 2

    @nn.compact
    def __call__(self, obs: Array) -> tuple[Array, Array]:
        """Simple MLP model for actor-critic."""
        # Actor mean
        x = obs
        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden_size, kernel_init=orthogonal(), bias_init=zeros)(x)
            x = nn.relu(x)
        mean = nn.Dense(self.act_dim, kernel_init=orthogonal(0.01), bias_init=zeros)(x)
        mean = nn.tanh(mean)

        return mean


class Agent(struct.PyTreeNode):
    """SHAC agent class with actor and critic networks."""

    actor_states: train_state.TrainState = struct.field(pytree_node=True)
    # Stored as pytree_node=False (static) so JAX JIT treats them as compile-time constants.
    # reinit() reuses these exact Python objects across seeds to avoid recompilation.
    actor: ActorNet = struct.field(pytree_node=False)
    tx: optax.GradientTransformation = struct.field(pytree_node=False)

    def sample_action(self, params: dict, obs: Array, key: Array) -> tuple:
        """Actor network mean action and log stds."""
        mean = self.actor.apply(params, obs)
        return (mean, jp.zeros_like(mean), jp.zeros_like(mean)), key

    def mean_action(self, params: dict, obs: Array) -> Array:
        """Actor network mean action."""
        return self.actor.apply(params, obs)

    @classmethod
    def create(
        cls,
        key: jax.random.PRNGKey,
        obs_dim: int,
        act_dim: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        actor_lr: float | optax.Schedule = 3e-4,
    ) -> Agent:
        """Initialize the SHAC agent's actor and critic networks."""
        actor = ActorNet(hidden_size=hidden_size, act_dim=act_dim, num_layers=num_layers)
        tx = optax.adamw(learning_rate=actor_lr, eps=1e-5)
        dummy_obs = jp.zeros((1, obs_dim), dtype=jp.float32)
        actor_params = actor.init(key, dummy_obs)
        actor_states = train_state.TrainState.create(
            apply_fn=actor.apply, params=actor_params, tx=tx
        )
        return cls(actor_states=actor_states, actor=actor, tx=tx)

    def reinit(self, key: jax.random.PRNGKey, obs_dim: int) -> Agent:
        """Return a new agent with fresh parameters but the same actor and optimizer.

        Every static field stays the same Python object, so JAX reuses the compiled cache instead of
        recompiling for each seed.
        """
        dummy_obs = jp.zeros((1, obs_dim), dtype=jp.float32)
        fresh_params = self.actor.init(key, dummy_obs)
        actor_states = train_state.TrainState.create(
            apply_fn=self.actor.apply, params=fresh_params, tx=self.tx
        )
        return self.replace(actor_states=actor_states)
