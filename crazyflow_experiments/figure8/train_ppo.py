"""A PPO implementation based on Flax.

With dr_scale > 0, training runs under domain randomization: every world draws its own mass,
inertia, rotor dynamics, thrust and torque curves and constant external wrench at reset, and
noise is added
to the wrench at every sim step. The spreads are `domain_randomization.DynamicsRanges`, scaled
by `dr_scale`, and `eval_dr_scale` evaluates the trained policy under randomization.

Run with
    pixi run -e figure8 python crazyflow_experiments/figure8/train_ppo.py
    pixi run -e figure8 python crazyflow_experiments/figure8/train_ppo.py --dr_scale 1.0
"""

from __future__ import annotations

import functools
import logging
import pickle
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Literal

import fire
import flax
import flax.struct as struct
import jax
import jax.numpy as jp
import numpy as np
import optax
import wandb
from jax import Array

from crazyflow_experiments.core.agents.ppo_agent import Agent
from crazyflow_experiments.core.envs import domain_randomization as dr
from crazyflow_experiments.core.envs.figure8_env import FigureEightEnv
from crazyflow_experiments.core.envs.wrappers import (
    ActionPenalty,
    AngleReward,
    FlattenJaxObservation,
    NormalizeActions,
    RecordData,
    ZeroYaw,
)
from crazyflow_experiments.figure8 import MODEL_DIR, checkpoint_name
from crazyflow_experiments.utils.paths import RESULTS_DIR

logger = logging.getLogger(__name__)


Preset = Literal["10s", "5_5s"]


# region Arguments
# https://github.com/yufei4hua/rl_diffsim/blob/f57b3eea267e89d0d01bfc98632db3aa10810d73/rl_diffsim/ppo/train_ppo_figure8.py
@dataclass(frozen=True)
class Args:
    """Class to store configurations."""

    seed: int = 42
    """seed of the experiment"""
    jax_device: str = "gpu"
    """environment device"""
    exp_name: str = "ppo_f8"
    """the name of the experiment"""
    wandb_project_name: str = "ppo-f8"
    """the wandb's project name"""
    wandb_entity: str | None = None
    """the entity (team) of wandb's project"""
    wandb_group: str | None = None
    """the group of the experiment for wandb logging"""

    # Algorithm specific arguments
    total_timesteps: int = 2_000_000
    """total timesteps of the experiments"""
    num_envs: int = 2048
    """the number of parallel game environments"""
    num_steps: int = 16
    """the number of steps to run in each environment per policy rollout"""
    num_minibatches: int = 16
    """the number of mini-batches"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    actor_lr: float = 8e-4
    """the learning rate of the actor optimizer"""
    critic_lr: float = 5e-3
    """the learning rate of the critic optimizer"""
    gamma: float = 0.92
    """the discount factor gamma"""
    gae_lambda: float = 0.94
    """the lambda for the general advantage estimation"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    clip_coef: float = 0.7
    """the surrogate clipping coefficient"""
    ent_coef: float = 0.007
    """coefficient of the entropy"""
    vf_coef: float = 0.7
    """coefficient of the value function"""
    hidden_size: int = 24
    """the hidden size of actor and critic networks"""
    init_logstd: float = -1.0
    """initial log standard deviation for the actor"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    # Wrapper settings
    trajectory_time: float = 10.0
    """time the drone takes to complete one figure eight"""
    n_samples: int = 11
    samples_dt: float = 0.06
    rpy_coef: float = 0.1
    act_coefs: tuple = (0.1, 0.1, 0.0, 0.02)
    d_act_coefs: tuple = (0.9, 0.9, 0.0, 0.6)
    """reward coefficients for training"""

    # Domain randomization
    dr_scale: float = 0.0
    """factor on the nominal randomization spreads of the environment, 0 trains without them"""

    @staticmethod
    def create(preset: Preset = "10s", **kwargs: Any) -> Args:
        """Create arguments class for the given trajectory preset."""
        args = Args(**{**PRESETS[preset], **kwargs})
        batch_size = int(args.num_envs * args.num_steps)
        minibatch_size = int(batch_size // args.num_minibatches)
        num_iterations = args.total_timesteps // batch_size
        return replace(
            args,
            batch_size=batch_size,
            minibatch_size=minibatch_size,
            num_iterations=num_iterations,
        )


# Values that differ between the two trajectory durations.
PRESETS: dict[Preset, dict] = {
    "10s": {
        "seed": 42,
        "total_timesteps": 2_000_000,
        "num_envs": 2048,
        "num_steps": 16,
        "num_minibatches": 16,
        "actor_lr": 8e-4,
        "critic_lr": 5e-3,
        "gamma": 0.92,
        "update_epochs": 10,
        "clip_coef": 0.7,
        "hidden_size": 24,
        "trajectory_time": 10.0,
        "d_act_coefs": (0.9, 0.9, 0.0, 0.6),
    },
    "5_5s": {
        "seed": 42,
        "total_timesteps": 10_000_000,
        "num_envs": 1024,
        "num_steps": 32,
        "num_minibatches": 32,
        "actor_lr": 3e-4,
        "critic_lr": 4e-3,
        "gamma": 0.96,
        "update_epochs": 15,
        "clip_coef": 0.5,
        "hidden_size": 32,
        "trajectory_time": 5.5,
        "d_act_coefs": (0.9, 0.9, 0.0, 0.2),
    },
}


def reward_coefs(args: Args) -> dict:
    """Return the environment and reward coefficients carried by args."""
    return {
        "trajectory_time": args.trajectory_time,
        "n_samples": args.n_samples,
        "samples_dt": args.samples_dt,
        "rpy_coef": args.rpy_coef,
        "act_coefs": args.act_coefs,
        "d_act_coefs": args.d_act_coefs,
    }


# region MakeEnvs
def make_jitted_envs(
    num_envs: int,
    coefs: dict,
    jax_device: str = "cpu",
    reset_rotor: bool = True,
    reset_random: bool = False,
    dr_scale: float = 0.0,
) -> FigureEightEnv:
    """Make environments for training RL policy.

    Args:
        num_envs: Number of parallel environments.
        coefs: Environment and reward coefficients, see `reward_coefs`.
        jax_device: JAX device string.
        reset_rotor: Whether to spin up the rotors on reset.
        reset_random: Whether to randomize the initial position and velocity.
        dr_scale: Factor on the nominal randomization spreads of the dynamics and disturbances.
            0 keeps the nominal dynamics.
    """
    randomization = dr.DynamicsRanges().scale(dr_scale) if dr_scale > 0 else None
    env: FigureEightEnv = FigureEightEnv.create(
        max_episode_time=20.0,
        trajectory_time=coefs["trajectory_time"],
        n_samples=coefs["n_samples"],
        samples_dt=coefs["samples_dt"],
        num_envs=num_envs,
        freq=50,
        drone="cf21B_500",
        dynamics="first_principles",
        device=jax_device,
        reset_rotor=reset_rotor,
        reset_randomization=None if reset_random else lambda data, default_data, mask: data,
        reset_velocity=True,
        multi_start=True,
        randomization=randomization,
    )

    env = NormalizeActions.create(env)
    env = ZeroYaw.create(env)
    env = AngleReward.create(env, weight=coefs["rpy_coef"])
    env = ActionPenalty.create(
        env,
        num_actions=1,
        init_last_actions=jp.array([[0.0, 0.0, 0.0, 0.0]]),
        act_coefs=coefs["act_coefs"],
        d_act_coefs=coefs["d_act_coefs"],
    )
    env = FlattenJaxObservation.create(env)
    return env


# region Utils
@flax.struct.dataclass
class RolloutData:
    """Class for storing rollout data."""

    observations: Array
    actions: Array
    logprobs: Array
    rewards: Array
    dones: Array
    values: Array
    sum_rewards: Array
    advantages: Array
    returns: Array


# region Rollout
@functools.partial(jax.jit, static_argnames=("args",))
def collect_rollout(
    envs: struct.PyTreeNode,
    args: Args,
    agent: Agent,
    next_obs: Array,
    next_done: Array,
    sum_rewards: Array,
    key: Array,
) -> tuple[struct.PyTreeNode, RolloutData, Array, Array, Array, Array]:
    """Collect a rollout of length args.num_steps. Returns data dict and next obs/done/key."""

    # loop over args.num_steps
    def step_once(carry: tuple, _) -> tuple[tuple, tuple]:
        env, key, sum_rewards, obs, dones = carry
        # 1. get action from policy
        (action, logprob, _), key = agent.sample_action(agent.actor_states.params, obs, key)
        value = agent.get_value(agent.critic_states.params, obs)

        # 2. step environment
        env, (next_obs, reward, terminations, truncations, _) = env.step(env, action)
        next_sum_rewards = jp.where(dones, 0.0, sum_rewards) + reward
        next_dones = terminations | truncations

        return (env, key, next_sum_rewards, next_obs, next_dones), RolloutData(
            observations=obs,
            actions=action,
            logprobs=logprob,
            rewards=reward,
            dones=dones,
            values=value,
            sum_rewards=sum_rewards,
            advantages=jp.zeros_like(reward),
            returns=jp.zeros_like(reward),
        )

    (envs, key, sum_rewards, next_obs, next_done), rollout_data = jax.lax.scan(
        step_once, (envs, key, sum_rewards, next_obs, next_done), length=args.num_steps
    )

    return envs, rollout_data, next_obs, next_done, sum_rewards, key


# region GAE
@functools.partial(jax.jit, static_argnames=("args",))
def compute_gae(
    args: Args, agent: Agent, next_obs: Array, next_done: Array, data: RolloutData
) -> RolloutData:
    """Compute GAE advantages and returns."""
    last_value = agent.get_value(agent.critic_states.params, next_obs)

    advantages = jp.zeros((args.num_envs,))
    dones = jp.concatenate([data.dones, next_done[None, :]], axis=0)
    values = jp.concatenate([data.values, last_value[None, :]], axis=0)

    def compute_gae_once(
        carry: Array, inp: tuple[Array, Array, Array, Array]
    ) -> tuple[Array, Array]:
        """Compute one step of GAE in scan."""
        advantages = carry
        nextdone, nextvalues, curvalues, reward = inp
        nextnonterminal = 1.0 - nextdone

        delta = reward + args.gamma * nextvalues * nextnonterminal - curvalues
        advantages = delta + args.gamma * args.gae_lambda * nextnonterminal * advantages
        return advantages, advantages

    _, advantages = jax.lax.scan(
        compute_gae_once,
        advantages,
        (dones[1:], values[1:], values[:-1], data.rewards),
        reverse=True,
    )
    data = data.replace(advantages=advantages, returns=advantages + data.values)
    return data


# region PG
@functools.partial(jax.jit, static_argnames=("args",))
def update_policy(
    args: Args, agent: Agent, data: RolloutData, key: Array
) -> tuple[tuple[Agent, Array], tuple[Array, Array, Array, Array, Array]]:
    """Performs PPO updates. Returns losses."""

    # loss function
    def loss_fn(
        actor_params: dict,
        critic_params: dict,
        obs_mb: Array,
        acts_mb: Array,
        old_logprobs_mb: Array,
        adv_mb: Array,
        ret_mb: Array,
        val_mb: Array,
    ) -> tuple[Array, tuple[Array, Array, Array, Array]]:
        # ppo policy loss
        logp, entropy = agent.get_action_logprob(actor_params, obs_mb, acts_mb)
        ratio = jp.exp(logp - old_logprobs_mb)
        pg_loss1 = -adv_mb * ratio
        pg_loss2 = -adv_mb * jp.clip(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef)
        pg_loss = jp.mean(jp.maximum(pg_loss1, pg_loss2))
        entropy_loss = -jp.mean(entropy)

        # ppo value loss (clip by default)
        value = agent.get_value(critic_params, obs_mb).reshape(-1)
        v_unclipped = (value - ret_mb) ** 2
        v_clipped = val_mb + jp.clip(value - val_mb, -args.clip_coef, args.clip_coef)
        v_clipped_loss = (v_clipped - ret_mb) ** 2
        v_loss = 0.5 * jp.mean(jp.maximum(v_unclipped, v_clipped_loss))

        loss = pg_loss + args.ent_coef * entropy_loss + args.vf_coef * v_loss
        approx_kl = jp.mean((ratio - 1.0) - (logp - old_logprobs_mb))
        return loss, (pg_loss, v_loss, entropy_loss, approx_kl)

    grad_fn = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)

    # batch: loop over epochs
    def update_epoch(
        carry: tuple[Agent, Array], _
    ) -> tuple[tuple[Agent, Array], tuple[Array, Array, Array, Array]]:
        agent, key = carry
        key, subkey = jax.random.split(key)

        def process_data(x: Array) -> Array:
            x = x.reshape((-1,) + x.shape[2:])
            x = jax.random.permutation(subkey, x)
            x = jp.reshape(x, (args.num_minibatches, -1) + x.shape[1:])
            return x

        minibatches = jax.tree_util.tree_map(process_data, data)

        # minibatch: scan minibatches
        def update_minibatch(
            carry: Agent, minibatch: RolloutData
        ) -> tuple[Agent, tuple[Array, Array, Array, Array]]:
            agent = carry
            (_, (pg_loss, v_loss, entropy_loss, approx_kl)), (g_actor, g_critic) = grad_fn(
                agent.actor_states.params,
                agent.critic_states.params,
                minibatch.observations,
                minibatch.actions,
                minibatch.logprobs,
                minibatch.advantages,
                minibatch.returns,
                minibatch.values,
            )
            return agent.replace(
                actor_states=agent.actor_states.apply_gradients(grads=g_actor),
                critic_states=agent.critic_states.apply_gradients(grads=g_critic),
            ), (pg_loss, v_loss, entropy_loss, approx_kl)

        agent, (pg_loss, v_loss, entropy_loss, approx_kl) = jax.lax.scan(
            update_minibatch, agent, minibatches
        )

        return (agent, key), (pg_loss, v_loss, entropy_loss, approx_kl)

    (agent, key), (pg_loss, v_loss, entropy_loss, approx_kl) = jax.lax.scan(
        update_epoch, (agent, key), (), length=args.update_epochs
    )

    y_pred = data.values.reshape(-1)
    y_true = data.returns.reshape(-1)
    var_y = jp.var(y_true)
    explained_var = jp.where(var_y == 0, jp.nan, 1.0 - jp.var(y_true - y_pred) / var_y)
    return (agent, key), (pg_loss, v_loss, entropy_loss, approx_kl, explained_var)


# region Train
def make_train_fn(args: Args) -> Callable:
    """Create a JIT-compiled training function that runs args.num_iterations via lax.scan.

    Both setup_train and the training call reuse this exact function object, so the scan context is
    compiled once.
    """

    def train_iteration(carry: tuple, _) -> tuple[tuple, tuple]:
        envs, agent, key, next_obs, next_done, sum_rewards = carry

        # 1. collect rollouts
        envs, data, next_obs, next_done, sum_rewards, key = collect_rollout(
            envs=envs,
            args=args,
            agent=agent,
            next_obs=next_obs,
            next_done=next_done,
            sum_rewards=sum_rewards,
            key=key,
        )

        # 2. compute GAE
        data = compute_gae(args, agent, next_obs, next_done, data)

        # 3. update policy
        (agent, key), (pg_loss, v_loss, entropy_loss, approx_kl, explained_var) = update_policy(
            args, agent, data, key
        )

        return (envs, agent, key, next_obs, next_done, sum_rewards), (
            data,
            pg_loss,
            v_loss,
            entropy_loss,
            approx_kl,
            explained_var,
        )

    @jax.jit
    def train_fn(
        envs: FigureEightEnv,
        agent: Agent,
        key: Array,
        next_obs: Array,
        next_done: Array,
        sum_rewards: Array,
    ) -> tuple:
        (envs, agent, key, next_obs, next_done, sum_rewards), outputs = jax.lax.scan(
            train_iteration,
            (envs, agent, key, next_obs, next_done, sum_rewards),
            jp.arange(args.num_iterations),
        )
        return (envs, agent, key, next_obs, next_done, sum_rewards), outputs

    return train_fn


def setup_train(args: Args, jax_device: str = "cpu") -> tuple[Callable, FigureEightEnv, Agent]:
    """Create environments and JIT-compile train_fn via a warmup run.

    Later calls that differ only in seed skip recompilation.

    Args:
        args: Training configuration. Hyperparameters that affect XLA compilation (num_iterations,
            num_envs, num_steps, hidden_size) must match those used in subsequent train_ppo() calls.
        jax_device: JAX device string.

    Returns:
        Tuple of (compiled train_fn, template envs ready to be reset, warmup agent for reinit).
    """
    r_coefs = reward_coefs(args)
    envs = make_jitted_envs(
        num_envs=args.num_envs,
        jax_device=jax_device,
        coefs=r_coefs,
        reset_rotor=True,
        dr_scale=args.dr_scale,
    )
    train_fn = make_train_fn(args)

    warmup_seed = args.seed + 10_000
    warmup_key = jax.random.PRNGKey(warmup_seed)
    init_key, _ = jax.random.split(warmup_key)
    train_steps = args.num_iterations * args.update_epochs * args.num_minibatches
    if args.anneal_lr:
        actor_lr = optax.linear_schedule(
            init_value=args.actor_lr, end_value=0.0, transition_steps=train_steps
        )
        critic_lr = optax.linear_schedule(
            init_value=args.critic_lr, end_value=0.0, transition_steps=train_steps
        )
    else:
        actor_lr = args.actor_lr
        critic_lr = args.critic_lr
    warmup_agent = Agent.create(
        key=init_key,
        obs_dim=envs.single_observation_space.shape[0],
        act_dim=envs.single_action_space.shape[0],
        hidden_size=args.hidden_size,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        init_logstd=args.init_logstd,
    )
    warmup_envs, (warmup_obs, _) = envs.reset(envs, seed=warmup_seed)
    jax.block_until_ready(
        train_fn(
            warmup_envs,
            warmup_agent,
            warmup_key,
            warmup_obs,
            jp.zeros(args.num_envs, dtype=bool),
            jp.zeros((args.num_envs,)),
        )
    )
    return train_fn, envs, warmup_agent


def train_ppo(
    args: Args,
    model_path: Path,
    jax_device: str,
    train_fn: Callable,
    envs: FigureEightEnv,
    template_agent: Agent,
    wandb_enabled: bool = False,
) -> tuple[list[float], float]:
    """Train a PPO policy with an end-to-end JAX/Flax compiled training loop.

    Takes the compiled function, environments and template agent from setup_train, and
    reinitialises the agent so every seed reuses the same compiled code.
    """
    # setup training
    if wandb_enabled and wandb.run is None:
        init_kwargs = {"project": args.wandb_project_name, "config": vars(args), "dir": RESULTS_DIR}
        if args.wandb_entity:
            init_kwargs["entity"] = args.wandb_entity
        if args.wandb_group:
            init_kwargs["group"] = args.wandb_group
        wandb.init(**init_kwargs)
    sum_rewards_hist = []

    setup_start_time = time.time()
    key = jax.random.PRNGKey(args.seed)
    logger.info(f"Training on device: {jax_device}")

    init_key, key = jax.random.split(key)
    agent = template_agent.reinit(init_key, envs.single_observation_space.shape[0])
    logger.info(f"Make envs and agent took {time.time() - setup_start_time:.5f} s")

    # start the training
    train_start_time = time.time()
    envs, (next_obs, _) = envs.reset(envs, seed=args.seed)
    next_done = jp.zeros(args.num_envs, dtype=bool)
    sum_rewards = jp.zeros((args.num_envs,))

    (
        (envs, agent, key, next_obs, next_done, sum_rewards),
        (all_data, all_pg_loss, all_v_loss, all_entropy_loss, all_approx_kl, all_explained_var),
    ) = train_fn(envs, agent, key, next_obs, next_done, sum_rewards)

    next_obs.block_until_ready()
    training_time = time.time() - train_start_time
    total_steps = args.num_iterations * args.batch_size
    logger.info(f"Training for {total_steps} steps took {training_time:.2f} seconds.")

    # Post-training logging
    if wandb_enabled:
        # Log aggregated metrics
        for iter_idx in range(args.num_iterations):
            global_step = iter_idx * args.batch_size
            data = jax.tree_util.tree_map(lambda x: x[iter_idx], all_data)

            # Log episode rewards
            for batch_step, (sum_reward, done) in enumerate(zip(data.sum_rewards, data.dones)):
                if jp.any(done):
                    wandb.log(
                        {"train/reward": jp.mean(sum_reward[done])},
                        step=global_step + batch_step * args.num_envs,
                    )
                    sum_rewards_hist.append(float(jp.mean(sum_reward[done])))

            # Log training metrics
            wandb.log(
                {
                    "losses/pg_loss": float(jp.mean(all_pg_loss[iter_idx])),
                    "losses/v_loss": float(jp.mean(all_v_loss[iter_idx])),
                    "losses/entropy": float(-jp.mean(all_entropy_loss[iter_idx])),
                    "losses/approx_kl": float(jp.mean(all_approx_kl[iter_idx])),
                    "losses/explained_variance": float(jp.mean(all_explained_var[iter_idx])),
                    "charts/SPS": int(global_step / training_time) if training_time > 0 else 0,
                },
                step=global_step + args.batch_size,
            )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    params = {
        "actor": agent.actor_states.params,
        "critic": agent.critic_states.params,
        "samples_dt": args.samples_dt,
    }
    with open(model_path, "wb") as f:
        pickle.dump(params, f)
    logger.info(f"model saved to {model_path}")

    return sum_rewards_hist, training_time


# region Evaluate
def load_agent(args: Args, envs: struct.PyTreeNode, model_path: Path) -> Agent:
    """Build an agent for the spaces of envs and load the checkpoint at model_path into it."""
    agent = Agent.create(
        key=jax.random.PRNGKey(0),
        obs_dim=envs.single_observation_space.shape[0],
        act_dim=envs.single_action_space.shape[0],
        hidden_size=args.hidden_size,
        init_logstd=args.init_logstd,
    )
    with open(model_path, "rb") as f:
        params = pickle.load(f)
    return agent.replace(
        actor_states=agent.actor_states.replace(params=params["actor"]),
        critic_states=agent.critic_states.replace(params=params["critic"]),
    )


def evaluate_ppo(
    args: Args, n_eval: int, model_path: Path, render: bool
) -> tuple[float, list, list]:
    """Run `n_eval` episodes in the nominal env with deterministic actions from `model_path`."""
    r_coefs = reward_coefs(args)
    eval_env = make_jitted_envs(
        num_envs=1, jax_device=args.jax_device, coefs=r_coefs, reset_rotor=True, dr_scale=0.0
    )
    eval_env = RecordData.create(eval_env)
    agent = load_agent(args, eval_env, model_path)

    episode_rewards = []
    episode_lengths = []
    ep_seed = args.seed

    for _ in range(n_eval):
        eval_env, (obs, _) = eval_env.reset(eval_env, seed=(ep_seed := ep_seed + 1))
        done = False
        episode_reward = 0.0
        steps = 0
        while not done:
            action = agent.mean_action(agent.actor_states.params, obs)
            eval_env, (obs, reward, terminated, truncated, _) = eval_env.step(eval_env, action)
            if render:
                eval_env.render()
            done = terminated | truncated
            episode_reward += float(np.asarray(reward).item())
            steps += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)

    rmse_pos = eval_env.calc_rmse()
    logger.info(f"Eval Mean Reward: {np.mean(episode_rewards):.2f}, RMSE: {rmse_pos * 1000:.3f} mm")

    eval_env.close()

    return rmse_pos, episode_rewards, episode_lengths


def evaluate_ppo_batched(
    args: Args, model_path: Path, dr_scale: float, num_envs: int = 256
) -> dict[str, float]:
    """Fly one episode per env with deterministic actions and summarize tracking and crashes.

    Every env starts at its own point of the trajectory and, with a positive dr_scale, with its own
    draw of the dynamics and disturbances. The RMSE is computed over the envs that did not crash, so
    that early terminations do not shrink it, and the crash rate counts the envs that left the
    flight volume before the episode ended.

    Args:
        args: Training configuration of the checkpoint.
        model_path: Checkpoint to evaluate.
        dr_scale: Factor on the nominal randomization spreads, 0 evaluates the nominal dynamics.
        num_envs: Number of parallel episodes.

    Returns:
        The position RMSE in mm, the crash rate, the mean return and the mean episode length.
    """
    envs = make_jitted_envs(
        num_envs=num_envs,
        jax_device=args.jax_device,
        coefs=reward_coefs(args),
        reset_rotor=True,
        dr_scale=dr_scale,
    )
    agent = load_agent(args, envs, model_path)
    max_steps = int(envs.unwrapped.max_episode_time * envs.unwrapped.freq)

    @jax.jit
    def rollout(envs: struct.PyTreeNode, params: dict) -> tuple[Array, Array, Array, Array]:
        envs, (obs, _) = envs.reset(envs, seed=args.seed + 1)

        def step(carry: tuple, _: None) -> tuple[tuple, None]:
            envs, obs, sum_rewards, sq_error, n_steps, active, crashed = carry
            action = agent.mean_action(params, obs)
            envs, (obs, reward, terminated, truncated, _) = envs.step(envs, action)
            raw = envs.unwrapped
            error = jp.linalg.norm(raw.data.states.pos[:, 0, :] - raw.target_pos, axis=-1)
            sum_rewards = sum_rewards + jp.where(active, reward, 0.0)
            crashed = crashed | (active & terminated)
            # The step that ends an episode already shows the reset state, so it is not tracked
            active = active & ~(terminated | truncated)
            sq_error = sq_error + jp.where(active, error**2, 0.0)
            n_steps = n_steps + active
            return (envs, obs, sum_rewards, sq_error, n_steps, active, crashed), None

        zeros = jp.zeros(num_envs)
        n_steps = jp.zeros(num_envs, dtype=jp.int32)
        active, crashed = jp.ones(num_envs, dtype=bool), jp.zeros(num_envs, dtype=bool)
        carry = (envs, obs, zeros, zeros, n_steps, active, crashed)
        (_, _, sum_rewards, sq_error, n_steps, _, crashed), _ = jax.lax.scan(
            step, carry, length=max_steps
        )
        return sum_rewards, sq_error, n_steps, crashed

    results = rollout(envs, agent.actor_states.params)
    sum_rewards, sq_error, n_steps, crashed = jax.tree.map(np.asarray, results)
    survived = ~crashed
    tracked_steps = np.sum(n_steps[survived])
    rmse = np.sqrt(np.sum(sq_error[survived]) / tracked_steps) if tracked_steps else float("nan")
    metrics = {
        "pos_rmse_mm": float(rmse * 1000),
        "crash_rate": float(np.mean(crashed)),
        "mean_reward": float(np.mean(sum_rewards)),
        "mean_steps": float(np.mean(n_steps)),
    }
    report = ", ".join(f"{k}={v:.3f}" for k, v in metrics.items())
    logger.info(f"Eval of {num_envs} episodes at DR scale {dr_scale:g}: {report}")
    envs.close()
    return metrics


# region Main
def main(
    preset: Preset = "10s",
    wandb_enabled: bool = True,
    train: bool = True,
    n_eval: int = 1,
    render: bool = False,
    n_runs: int = 1,
    seed: int | None = None,
    save_dir: str | None = None,
    dr_scale: float = 0.0,
    eval_dr_scale: float = 0.0,
):
    """Main entry.

    Flags:
      preset: trajectory duration preset, either "10s" or "5_5s"
      wandb_enabled: log metrics to wandb
      train: run training
      n_eval: number of evaluation episodes to run after training (or standalone)
      render: whether to render the environment during evaluation
      save_dir: directory for the checkpoint, defaults to the saves folder
      n_runs: number of successive training runs (run k uses seed = seed + k)
      seed: starting seed for the first run, defaults to the preset's seed
      dr_scale: factor on the nominal randomization spreads for training, 0 trains without them
      eval_dr_scale: factor for an additional evaluation under randomization, 0 skips it
    """
    template_args = Args.create(preset, dr_scale=dr_scale)
    jax_device = template_args.jax_device
    seed = template_args.seed if seed is None else seed

    # Build the JIT-compiled train function and envs once, so every run reuses them.
    # Evaluation loads its own env and agent, so this is skipped with nothing to train.
    if train:
        train_fn, envs, template_agent = setup_train(template_args, jax_device=jax_device)

    for run_idx in range(n_runs):
        run_seed = seed + run_idx
        args = Args.create(preset, seed=run_seed, dr_scale=dr_scale)
        saves = MODEL_DIR if save_dir is None else Path(save_dir)
        model_path = saves / checkpoint_name(
            args.exp_name, args.trajectory_time, dr_scale, run_seed if n_runs > 1 else None
        )

        if n_runs > 1:
            logger.info(f"Run {run_idx + 1}/{n_runs} (seed={run_seed})")

        if train:
            train_ppo(args, model_path, jax_device, train_fn, envs, template_agent, wandb_enabled)

        logs = {}
        if n_eval > 0:
            rmse_pos, episode_rewards, episode_lengths = evaluate_ppo(
                args, n_eval, model_path, render
            )
            logs = {
                "eval/mean_rewards": np.mean(episode_rewards),
                "eval/mean_steps": np.mean(episode_lengths),
                "eval/pos_rmse_mm": rmse_pos,
            }
        if eval_dr_scale > 0:
            metrics = evaluate_ppo_batched(args, model_path, eval_dr_scale)
            logs |= {f"eval_dr/{key}": value for key, value in metrics.items()}
            logs["eval_dr/scale"] = eval_dr_scale
        if wandb_enabled and train and wandb.run is not None:
            if logs:
                wandb.log(logs)
            wandb.finish()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(main)
