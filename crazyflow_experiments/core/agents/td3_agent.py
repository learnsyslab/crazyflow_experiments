"""Twin Delayed Deep Deterministic Policy Gradient agent implemented in Flax."""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING, Any, Callable

import flax.struct as struct
import jax
import jax.lax as lax
import jax.numpy as jp
import optax
from flax import linen as nn
from flax.linen.initializers import orthogonal, zeros
from flax.training import train_state
from jax import Array

if TYPE_CHECKING:
    from pathlib import Path


class ActorNet(nn.Module):
    """Deterministic actor network for TD3."""

    hidden_size: int = 64
    act_dim: int = 4
    num_layers: int = 2

    @nn.compact
    def __call__(self, obs: Array) -> Array:
        """Forward pass returning deterministic action in [-1, 1]."""
        x = obs
        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden_size, kernel_init=orthogonal(), bias_init=zeros)(x)
            x = nn.relu(x)
        action = nn.Dense(self.act_dim, kernel_init=orthogonal(0.01), bias_init=zeros)(x)
        action = nn.tanh(action)  # Bound to [-1, 1]
        return action


class CriticNet(nn.Module):
    """Q-network for TD3. Takes concatenated (obs, action) as input."""

    hidden_size: int = 64
    num_layers: int = 2

    @nn.compact
    def __call__(self, obs: Array, action: Array) -> Array:
        """Forward pass returning Q-value."""
        x = jp.concatenate([obs, action], axis=-1)
        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden_size, kernel_init=orthogonal(), bias_init=zeros)(x)
            x = nn.relu(x)
        q_value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=zeros)(x)
        return q_value.squeeze(-1)


class TD3Agent(struct.PyTreeNode):
    """TD3 agent with asymmetric actor-critic observations."""

    actor_states: train_state.TrainState = struct.field(pytree_node=True)
    critic1_state: train_state.TrainState = struct.field(pytree_node=True)
    critic2_state: train_state.TrainState = struct.field(pytree_node=True)
    target_actor_params: dict = struct.field(pytree_node=True)
    target_critic1_params: dict = struct.field(pytree_node=True)
    target_critic2_params: dict = struct.field(pytree_node=True)

    # Stored as pytree_node=False (static) so JAX JIT treats them as compile-time constants.
    # reinit() reuses these exact Python objects across seeds to avoid recompilation.
    actor: ActorNet = struct.field(pytree_node=False)
    critic: CriticNet = struct.field(pytree_node=False)
    actor_tx: optax.GradientTransformation = struct.field(pytree_node=False)
    critic_tx: optax.GradientTransformation = struct.field(pytree_node=False)
    actor_obs_dim: int = struct.field(pytree_node=False)

    # Inference functions (not part of pytree)
    mean_action: Callable = struct.field(pytree_node=False)
    sample_action: Callable = struct.field(pytree_node=False)
    random_action: Callable = struct.field(pytree_node=False)
    get_q: Callable = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        key: jax.random.PRNGKey,
        obs_dim: int,
        act_dim: int,
        actor_obs_dim: int | None = None,
        hidden_size: int = 64,
        num_layers: int = 2,
        hidden_size_critic: int = 128,
        num_layers_critic: int = 3,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
    ) -> TD3Agent:
        """Initialize TD3 agent with actor, twin critics, and target networks."""
        # Create networks (single critic instance, params are separate)
        actor = ActorNet(hidden_size=hidden_size, act_dim=act_dim, num_layers=num_layers)
        critic = CriticNet(hidden_size=hidden_size_critic, num_layers=num_layers_critic)

        # Initialize parameters
        actor_obs_dim = actor_obs_dim if actor_obs_dim is not None else obs_dim
        k1, k2, k3 = jax.random.split(key, 3)
        dummy_obs = jp.zeros((1, obs_dim), dtype=jp.float32)
        dummy_actor_obs = dummy_obs[..., :actor_obs_dim]  # Slice for actor input
        dummy_action = jp.zeros((1, act_dim), dtype=jp.float32)

        actor_params = actor.init(k1, dummy_actor_obs)
        critic1_params = critic.init(k2, dummy_obs, dummy_action)
        critic2_params = critic.init(k3, dummy_obs, dummy_action)

        # Create optimizers
        actor_tx = optax.adamw(learning_rate=actor_lr, eps=1e-5)
        critic_tx = optax.adamw(learning_rate=critic_lr, eps=1e-5)

        # Create train states (both critics share the same apply_fn)
        actor_states = train_state.TrainState.create(
            apply_fn=actor.apply, params=actor_params, tx=actor_tx
        )
        critic1_state = train_state.TrainState.create(
            apply_fn=critic.apply, params=critic1_params, tx=critic_tx
        )
        critic2_state = train_state.TrainState.create(
            apply_fn=critic.apply, params=critic2_params, tx=critic_tx
        )

        # Initialize target networks with same parameters
        target_actor_params = actor_params
        target_critic1_params = critic1_params
        target_critic2_params = critic2_params

        def _mean_action(params: dict, obs: Array) -> Array:
            """Get deterministic action (for evaluation). Slices first actor_obs_dim dims."""
            return actor.apply(params, obs[..., :actor_obs_dim])

        def _sample_action(
            params: dict, obs: Array, key: Array, std: float, noise_clip: float = jp.inf
        ) -> tuple[Array, Array]:
            """Get action with exploration noise (for training). Slices first actor_obs_dim dims."""
            mean = actor.apply(params, obs[..., :actor_obs_dim])
            new_key, sub = jax.random.split(key)
            noise = jax.random.normal(sub, mean.shape, dtype=mean.dtype) * std
            noise = jp.clip(noise, -noise_clip, noise_clip)
            action = jp.clip(mean + noise, -1.0, 1.0)
            return action, new_key

        def _random_action(num_envs: int, key: Array) -> tuple[Array, Array]:
            """Get action with exploration noise (for training). Slices first actor_obs_dim dims."""
            new_key, sub = jax.random.split(key)
            action = jax.random.uniform(
                sub, (num_envs, act_dim), minval=-1.0, maxval=1.0, dtype=jp.float32
            )
            return action, new_key

        def _get_q(params: dict, obs: Array, action: Array) -> Array:
            """Get Q value (works with either critic's params)."""
            return critic.apply(params, obs, action)

        return cls(
            actor_states=actor_states,
            critic1_state=critic1_state,
            critic2_state=critic2_state,
            target_actor_params=target_actor_params,
            target_critic1_params=target_critic1_params,
            target_critic2_params=target_critic2_params,
            actor=actor,
            critic=critic,
            actor_tx=actor_tx,
            critic_tx=critic_tx,
            actor_obs_dim=actor_obs_dim,
            mean_action=jax.jit(_mean_action),  # jitted for faster inference in real experiments
            sample_action=_sample_action,
            random_action=_random_action,
            get_q=_get_q,
        )

    def reinit(self, key: jax.random.PRNGKey, obs_dim: int) -> TD3Agent:
        """Return a new agent with fresh parameters but the same networks, optimizers, and closures.

        Every static field stays the same Python object, so JAX reuses the compiled cache instead of
        recompiling for each seed.
        """
        k1, k2, k3 = jax.random.split(key, 3)
        dummy_obs = jp.zeros((1, obs_dim), dtype=jp.float32)
        dummy_actor_obs = dummy_obs[..., : self.actor_obs_dim]
        dummy_action = jp.zeros((1, self.actor.act_dim), dtype=jp.float32)

        actor_params = self.actor.init(k1, dummy_actor_obs)
        critic1_params = self.critic.init(k2, dummy_obs, dummy_action)
        critic2_params = self.critic.init(k3, dummy_obs, dummy_action)

        actor_states = train_state.TrainState.create(
            apply_fn=self.actor.apply, params=actor_params, tx=self.actor_tx
        )
        critic1_state = train_state.TrainState.create(
            apply_fn=self.critic.apply, params=critic1_params, tx=self.critic_tx
        )
        critic2_state = train_state.TrainState.create(
            apply_fn=self.critic.apply, params=critic2_params, tx=self.critic_tx
        )
        return self.replace(
            actor_states=actor_states,
            critic1_state=critic1_state,
            critic2_state=critic2_state,
            target_actor_params=actor_params,
            target_critic1_params=critic1_params,
            target_critic2_params=critic2_params,
        )


@struct.dataclass
class Transition:
    """Transition of a single environment step, batched over envs."""

    obs: Array
    action: Array
    reward: Array
    next_obs: Array
    terminated: Array


class ReplayBuffer(struct.PyTreeNode):
    """Preallocated circular replay buffer."""

    observations: Array = struct.field(pytree_node=True)
    actions: Array = struct.field(pytree_node=True)
    rewards: Array = struct.field(pytree_node=True)
    next_observations: Array = struct.field(pytree_node=True)
    terminated: Array = struct.field(pytree_node=True)
    ptr: Array = struct.field(pytree_node=True)
    size: Array = struct.field(pytree_node=True)

    add: Callable = struct.field(pytree_node=False)
    sample: Callable = struct.field(pytree_node=False)

    @classmethod
    def create(cls, capacity: int, obs_dim: int, act_dim: int) -> ReplayBuffer:
        """Create an empty preallocated replay buffer."""

        def _add(buffer: ReplayBuffer, transitions: Transition) -> ReplayBuffer:
            """Add batched transitions (n_steps, n_envs, ...) to buffer."""
            flat = jax.tree_util.tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), transitions)
            batch_size = flat.reward.shape[0]
            indices = (buffer.ptr + jp.arange(batch_size)) % capacity
            return buffer.replace(
                observations=buffer.observations.at[indices].set(flat.obs),
                actions=buffer.actions.at[indices].set(flat.action),
                rewards=buffer.rewards.at[indices].set(flat.reward),
                next_observations=buffer.next_observations.at[indices].set(flat.next_obs),
                terminated=buffer.terminated.at[indices].set(flat.terminated),
                ptr=(buffer.ptr + batch_size) % capacity,
                size=jp.minimum(buffer.size + batch_size, capacity),
            )

        def _sample(buffer: ReplayBuffer, batch_size: int, key: Array) -> Transition:
            """Sample a random batch of transitions."""
            indices = jax.random.randint(key, (batch_size,), 0, jp.minimum(buffer.size, capacity))
            return Transition(
                obs=buffer.observations[indices],
                action=buffer.actions[indices],
                reward=buffer.rewards[indices],
                next_obs=buffer.next_observations[indices],
                terminated=buffer.terminated[indices],
            )

        return cls(
            observations=jp.zeros((capacity, obs_dim), dtype=jp.float32),
            actions=jp.zeros((capacity, act_dim), dtype=jp.float32),
            rewards=jp.zeros((capacity,), dtype=jp.float32),
            next_observations=jp.zeros((capacity, obs_dim), dtype=jp.float32),
            terminated=jp.zeros((capacity,), dtype=jp.bool_),
            ptr=jp.array(0, dtype=jp.int32),
            size=jp.array(0, dtype=jp.int32),
            add=_add,
            sample=_sample,
        )


def update_policy(
    agent: TD3Agent,
    buffer: ReplayBuffer,
    key: Array,
    epochs: int,
    batch_size: int,
    gamma: float,
    tau: float,
    policy_noise: float,
    noise_clip: float,
    policy_delay: int,
) -> tuple[TD3Agent, Array, Array, Array]:
    """Perform `epochs` TD3 policy updates with an actor update every `policy_delay` steps."""
    sample_keys = jax.random.split(key, epochs + 1)
    key = sample_keys[0]
    sample_keys = sample_keys[1:]

    def sample_batch(key: Array) -> Transition:
        return buffer.sample(buffer, batch_size, key)

    batches = jax.vmap(sample_batch)(sample_keys)

    def update_batch_epochs(
        carry: tuple[TD3Agent, Array], batch_offset: int
    ) -> tuple[tuple[TD3Agent, Array], tuple[Array, Array]]:
        agent, key = carry

        def update_critic(
            carry: tuple[TD3Agent, Array], local_idx: int
        ) -> tuple[tuple[TD3Agent, Array], Array]:
            agent, key = carry
            idx = batch_offset * policy_delay + local_idx

            obs = batches.obs[idx]
            actions = batches.action[idx]
            rewards = batches.reward[idx]
            next_obs = batches.next_obs[idx]
            terminated = batches.terminated[idx]

            target_actions, key = agent.sample_action(
                agent.target_actor_params, next_obs, key, std=policy_noise, noise_clip=noise_clip
            )

            # Compute target Q
            target_q1 = agent.get_q(agent.target_critic1_params, next_obs, target_actions)
            target_q2 = agent.get_q(agent.target_critic2_params, next_obs, target_actions)
            target_q = rewards + gamma * (1.0 - terminated.astype(jp.float32)) * jp.minimum(
                target_q1, target_q2
            )

            # Twin critic loss
            def critic_loss_fn(c1_params: dict[str, Any], c2_params: dict[str, Any]) -> Array:
                q1 = agent.get_q(c1_params, obs, actions)
                q2 = agent.get_q(c2_params, obs, actions)
                return jp.mean((q1 - target_q) ** 2) + jp.mean((q2 - target_q) ** 2)

            (critic_loss, (c1_grads, c2_grads)) = jax.value_and_grad(
                critic_loss_fn, argnums=(0, 1)
            )(agent.critic1_state.params, agent.critic2_state.params)

            critic1_state = agent.critic1_state.apply_gradients(grads=c1_grads)
            critic2_state = agent.critic2_state.apply_gradients(grads=c2_grads)
            agent = agent.replace(critic1_state=critic1_state, critic2_state=critic2_state)

            return (agent, key), critic_loss

        (agent, key), critic_losses = lax.scan(
            update_critic, (agent, key), jp.arange(policy_delay), unroll=policy_delay
        )  # unroll update_critic() loop

        # Actor Update (*1)
        def update_actor(agent: TD3Agent, obs: Array) -> tuple[TD3Agent, Array]:
            def actor_loss_fn(params: dict[str, Any]) -> Array:
                a = agent.mean_action(params, obs)
                return -jp.mean(agent.get_q(agent.critic1_state.params, obs, a))

            actor_loss, a_grads = jax.value_and_grad(actor_loss_fn)(agent.actor_states.params)
            actor_states = agent.actor_states.apply_gradients(grads=a_grads)

            # Soft update targets
            agent = agent.replace(
                actor_states=actor_states,
                target_actor_params=optax.incremental_update(
                    actor_states.params, agent.target_actor_params, tau
                ),
                target_critic1_params=optax.incremental_update(
                    agent.critic1_state.params, agent.target_critic1_params, tau
                ),
                target_critic2_params=optax.incremental_update(
                    agent.critic2_state.params, agent.target_critic2_params, tau
                ),
            )
            return agent, actor_loss

        last_idx = batch_offset * policy_delay + policy_delay - 1
        obs = batches.obs[last_idx]
        agent, actor_loss = update_actor(agent, obs)

        return (agent, key), (jp.mean(critic_losses), actor_loss)

    num_actor_updates = epochs // policy_delay
    (agent, key), (critic_losses, actor_losses) = lax.scan(
        update_batch_epochs, (agent, key), jp.arange(num_actor_updates)
    )

    mean_critic_loss = jp.mean(critic_losses)
    mean_actor_loss = jp.mean(actor_losses)

    return agent, mean_critic_loss, mean_actor_loss, key


def load_actor_params(model_path: Path) -> dict:
    """Load the actor parameter tree of a checkpoint."""
    with open(model_path, "rb") as f:
        return pickle.load(f)["actor"]


def load_actor(model_path: Path) -> TD3Agent:
    """Create an agent whose actor holds the parameters of a checkpoint.

    The architecture follows the checkpoint. Critics stay at their initial values, since they play no
    role at inference time.
    """
    params = load_actor_params(model_path)
    num_layers = len([k for k in params["params"] if k.startswith("Dense_")]) - 1  # Last is output
    obs_dim, hidden_size = params["params"]["Dense_0"]["kernel"].shape
    act_dim = params["params"][f"Dense_{num_layers}"]["kernel"].shape[1]
    agent = TD3Agent.create(
        key=jax.random.PRNGKey(0),
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
    )
    return agent.replace(actor_states=agent.actor_states.replace(params=params))
