"""PPO agent implementation using Flax."""

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


class ActorNet(nn.Module):
    """Class defining the actor-critic model."""

    hidden_size: int = 64
    act_dim: int = 4

    @nn.compact
    def __call__(self, obs: Array) -> tuple[Array, Array]:
        """Simple MLP model for actor-critic."""
        # Actor mean
        x = obs
        x = nn.Dense(self.hidden_size, kernel_init=orthogonal(), bias_init=zeros)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_size, kernel_init=orthogonal(), bias_init=zeros)(x)
        x = nn.relu(x)
        mean = nn.Dense(self.act_dim, kernel_init=orthogonal(0.01), bias_init=zeros)(x)
        mean = nn.tanh(mean)
        # Actor logstd
        actor_logstd = self.param(
            "actor_logstd", lambda rng, shape: jp.zeros(shape, dtype=jp.float32), (1, self.act_dim)
        )
        logstd = jp.broadcast_to(actor_logstd, (mean.shape[0], self.act_dim))

        return mean, logstd


class CriticNet(nn.Module):
    """Class defining the critic model."""

    hidden_size: int = 64

    @nn.compact
    def __call__(self, obs: Array) -> Array:
        """Simple MLP model for critic."""
        x = obs
        x = nn.Dense(self.hidden_size, kernel_init=orthogonal(), bias_init=zeros)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_size, kernel_init=orthogonal(), bias_init=zeros)(x)
        x = nn.relu(x)
        value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=zeros)(x)
        return value.squeeze(-1)


class Agent(struct.PyTreeNode):
    """PPO agent class with actor and critic networks."""

    actor_states: train_state.TrainState = struct.field(pytree_node=True)
    critic_states: train_state.TrainState = struct.field(pytree_node=True)
    # Stored as pytree_node=False (static) so JAX JIT treats them as compile-time constants.
    # reinit() reuses these exact Python objects across seeds to avoid recompilation.
    actor: ActorNet = struct.field(pytree_node=False)
    critic: CriticNet = struct.field(pytree_node=False)
    actor_tx: optax.GradientTransformation = struct.field(pytree_node=False)
    critic_tx: optax.GradientTransformation = struct.field(pytree_node=False)
    init_logstd: float = struct.field(pytree_node=False)
    mean_action: Callable = struct.field(pytree_node=False)
    sample_action: Callable = struct.field(pytree_node=False)
    get_action_logprob: Callable = struct.field(pytree_node=False)
    get_value: Callable = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        key: jax.random.PRNGKey,
        obs_dim: int,
        act_dim: int,
        hidden_size: int = 64,
        actor_lr: float | optax.Schedule = 3e-4,
        critic_lr: float | optax.Schedule = 1e-3,
        init_logstd: float = -1.0,
    ) -> dict:
        """Initialize the PPO agent's actor and critic networks."""
        actor = ActorNet(hidden_size=hidden_size, act_dim=act_dim)
        critic = CriticNet(hidden_size=hidden_size)
        k1, k2 = jax.random.split(key)
        dummy_obs = jp.zeros((1, obs_dim), dtype=jp.float32)
        actor_params = actor.init(k1, dummy_obs)
        actor_params["params"]["actor_logstd"] = jp.full(
            (1, act_dim), init_logstd, dtype=jp.float32
        )
        critic_params = critic.init(k2, dummy_obs)
        actor_tx = optax.adamw(learning_rate=actor_lr, eps=1e-5)
        critic_tx = optax.adamw(learning_rate=critic_lr, eps=1e-5)
        actor_states = train_state.TrainState.create(
            apply_fn=actor.apply, params=actor_params, tx=actor_tx
        )
        critic_states = train_state.TrainState.create(
            apply_fn=critic.apply, params=critic_params, tx=critic_tx
        )

        # Build jittable actor and critic inference functions
        def _sample_action(params: dict, obs: Array, key: Array) -> tuple[Array, Array]:
            """Get stochastic action sample for collecting rollout."""
            mean, logstd = actor.apply(params, obs)
            std = jp.exp(logstd)
            new_key, sub = jax.random.split(key)
            eps = jax.random.normal(sub, mean.shape, dtype=mean.dtype)
            action = mean + std * eps
            logp = -0.5 * (eps**2 + 2.0 * logstd + jp.log(2.0 * jp.pi))
            logp = jp.sum(logp, axis=-1)
            entropy = jp.sum(0.5 * (1.0 + jp.log(2.0 * jp.pi)) + logstd, axis=-1)
            return (action, logp, entropy), new_key

        def _get_action_logprob(params: dict, obs: Array, action: Array) -> tuple[Array, Array]:
            """Get log probability and entropy of given action for training."""
            mean, logstd = actor.apply(params, obs)
            std = jp.exp(logstd)
            logp = -0.5 * (
                ((action - mean) / (std + 1e-8)) ** 2 + 2.0 * logstd + jp.log(2.0 * jp.pi)
            )
            logp = jp.sum(logp, axis=-1)
            entropy = jp.sum(0.5 * (1.0 + jp.log(2.0 * jp.pi)) + logstd, axis=-1)
            return logp, entropy

        def _mean_action(params: dict, obs: Array) -> Array:
            """Get deterministic action (mean)."""
            mean, logstd = actor.apply(params, obs)
            return mean

        def _get_value(params: dict, obs: Array) -> Array:
            return jp.squeeze(critic.apply(params, obs))

        return cls(
            actor_states=actor_states,
            critic_states=critic_states,
            actor=actor,
            critic=critic,
            actor_tx=actor_tx,
            critic_tx=critic_tx,
            init_logstd=init_logstd,
            mean_action=jax.jit(_mean_action),
            sample_action=jax.jit(_sample_action),
            get_action_logprob=jax.jit(_get_action_logprob),
            get_value=jax.jit(_get_value),
        )

    def reinit(self, key: jax.random.PRNGKey, obs_dim: int) -> Agent:
        """Return a new agent with fresh parameters but the same networks, optimizers, and closures.

        Every static field stays the same Python object, so JAX reuses the compiled cache instead of
        recompiling for each seed.
        """
        k1, k2 = jax.random.split(key)
        dummy_obs = jp.zeros((1, obs_dim), dtype=jp.float32)
        fresh_actor_params = self.actor.init(k1, dummy_obs)
        fresh_actor_params["params"]["actor_logstd"] = jp.full(
            (1, self.actor.act_dim), self.init_logstd, dtype=jp.float32
        )
        fresh_critic_params = self.critic.init(k2, dummy_obs)
        actor_states = train_state.TrainState.create(
            apply_fn=self.actor.apply, params=fresh_actor_params, tx=self.actor_tx
        )
        critic_states = train_state.TrainState.create(
            apply_fn=self.critic.apply, params=fresh_critic_params, tx=self.critic_tx
        )
        return self.replace(actor_states=actor_states, critic_states=critic_states)


if __name__ == "__main__":
    """Test the agent implementation."""
    # initialization
    obs_dim, act_dim = 13, 4
    agent = Agent.create(
        jax.random.PRNGKey(0),
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_size=64,
        actor_lr=3e-4,
        critic_lr=1e-3,
    )

    obs = jp.ones((2, obs_dim), dtype=jp.float32)

    # rollout - sample actions
    key = jax.random.PRNGKey(1)
    (action, logp, entropy), key = agent.get_action_sample(agent.actor_states.params, obs, key)
    value = agent.get_value(agent.critic_states.params, obs)
    print("Rollout:")
    print(action, logp, entropy, value)

    # optimization - get log probabilities
    chosen_action = action
    logp, entropy = agent.get_action_logprob(agent.actor_states.params, obs, chosen_action)
    value = agent.get_value(agent.critic_states.params, obs)
    print("Optimization:")
    print(chosen_action, logp, entropy, value)

    # deployment - get deterministic action
    action = agent.get_action_mean(agent.actor_states.params, obs)
    print("Deployment:")
    print(action)
