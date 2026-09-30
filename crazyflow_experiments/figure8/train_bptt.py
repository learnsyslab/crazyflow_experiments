"""A BPTT implementation based on Flax.

Run with
    pixi run -e figure8 python crazyflow_experiments/figure8/train_bptt.py
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

from crazyflow_experiments.core.agents.bptt_agent import Agent
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
# Note: We absolutely push how fast BPTT can train, which makes it brittle to seeds. Robust training
# will require retuning
@dataclass(frozen=True)
class Args:
    """Class to store configurations."""

    seed: int = 42
    """seed of the experiment"""
    jax_device: str = "cpu"
    """environment device"""
    exp_name: str = "bptt_f8"
    """the name of the experiment"""
    wandb_project_name: str = "bptt-f8"
    """the wandb's project name"""
    wandb_entity: str | None = None
    """the entity (team) of wandb's project"""
    wandb_group: str | None = None
    """the group of the experiment for wandb logging"""

    # Algorithm specific arguments
    total_timesteps: int = 100_000
    """total timesteps of the experiments"""
    num_envs: int = 16
    """the number of parallel game environments"""
    num_steps: int = 40
    """the number of steps to run in each environment per policy rollout"""
    anneal_actor_lr: bool = False
    """Toggle learning rate annealing for policy networks"""
    actor_lr: float = 4.6e-2
    """the learning rate of the actor optimizer"""
    gamma: float = 1.0
    """the discount factor gamma"""
    hidden_size: int = 6
    """the hidden size of actor and critic networks"""
    init_logstd: float = -1.0
    """initial log standard deviation for the actor"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    # Wrapper settings
    trajectory_time: float = 10.0
    """time the drone takes to complete one figure eight"""
    n_samples: int = 11
    samples_dt: float = 0.06
    rpy_coef: float = 0.1
    act_coefs: tuple = (0.12, 0.12, 0.0, 0.02)
    d_act_coefs: tuple = (1.4, 1.4, 0.0, 0.8)
    """reward coefficients for training"""

    @staticmethod
    def create(preset: Preset = "10s", **kwargs: Any) -> Args:
        """Create arguments class for the given trajectory preset."""
        args = Args(**{**PRESETS[preset], **kwargs})
        batch_size = int(args.num_envs * args.num_steps)
        num_iterations = args.total_timesteps // batch_size
        act_coefs = (args.act_coefs[0],) * 2 + (0.0, args.act_coefs[3])
        d_act_coefs = (args.d_act_coefs[0],) * 2 + (0.0, args.d_act_coefs[3])
        return replace(
            args,
            batch_size=batch_size,
            num_iterations=num_iterations,
            act_coefs=act_coefs,
            d_act_coefs=d_act_coefs,
        )


# Values that differ between the two trajectory durations.
PRESETS: dict[Preset, dict] = {
    "10s": {
        "seed": 42,
        "total_timesteps": 100_000,
        "num_steps": 40,
        "actor_lr": 4.6e-2,
        "hidden_size": 6,
        "trajectory_time": 10.0,
        "n_samples": 11,
        "samples_dt": 0.06,
        "act_coefs": (0.12, 0.12, 0.0, 0.02),
        "d_act_coefs": (1.4, 1.4, 0.0, 0.8),
    },
    "5_5s": {
        "seed": 42,
        "total_timesteps": 800_000,
        "num_steps": 64,
        "actor_lr": 1.0e-2,
        "hidden_size": 12,
        "trajectory_time": 5.5,
        "n_samples": 12,
        "samples_dt": 0.04,
        "act_coefs": (0.22, 0.18, 0.0, 0.03),
        "d_act_coefs": (1.4, 1.4, 0.0, 0.2),
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
    reset_rotor: bool = False,
    reset_random: bool = False,
) -> FigureEightEnv:
    """Make environments for training RL policy."""
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
    rewards: Array
    dones: Array
    sum_rewards: Array
    entropy: Array
    losses: Array


# region Policy Update
@functools.partial(jax.jit, static_argnames=("args",))
def update_policy(
    envs: struct.PyTreeNode,
    args: Args,
    agent: Agent,
    next_obs: Array,
    next_done: Array,
    sum_rewards: Array,
    key: Array,
) -> tuple[
    tuple[struct.PyTreeNode, Agent, Array], tuple[Array, tuple[RolloutData, Array, Array, Array]]
]:
    """BPTT policy updates."""

    def collect_rollout(
        envs: struct.PyTreeNode,
        actor_params: dict,
        next_obs: Array,
        next_done: Array,
        sum_rewards: Array,
        key: Array,
    ) -> tuple[struct.PyTreeNode, RolloutData, Array, Array, Array, Array, Array]:
        """Collect a rollout of length args.num_steps. Returns data dict and next obs/done/key."""
        discounts = jp.ones((args.num_envs,))

        # loop over args.num_steps
        def step_once(carry: tuple, _) -> tuple[tuple, tuple]:
            env, key, sum_rewards, discounts, obs, dones = carry
            # 1. get action from policy
            (action, _, entropy), key = agent.sample_action(actor_params, obs, key)

            # 2. step environment & compute stepwise loss
            env, (next_obs, reward, terminations, truncations, _) = env.step(env, action)

            next_sum_rewards = jp.where(dones, 0.0, sum_rewards) + reward
            loss = -discounts * reward
            next_discounts = jp.where(dones, 1.0, discounts * args.gamma)
            next_dones = terminations | truncations

            return (env, key, next_sum_rewards, next_discounts, next_obs, next_dones), RolloutData(
                observations=obs,
                actions=action,
                rewards=reward,
                dones=dones,
                sum_rewards=sum_rewards,
                entropy=jp.mean(entropy),
                losses=loss,
            )

        (envs, key, sum_rewards, next_discounts, next_obs, next_done), rollout_data = jax.lax.scan(
            step_once,
            (envs, key, sum_rewards, discounts, next_obs, next_done),
            length=args.num_steps,
        )

        return envs, rollout_data, next_discounts, next_obs, next_done, sum_rewards, key

    def policy_loss_fn(
        envs: struct.PyTreeNode,
        actor_params: dict,
        next_obs: Array,
        next_done: Array,
        sum_rewards: Array,
        key: Array,
    ) -> tuple[Array, tuple[struct.PyTreeNode, RolloutData, Array, Array, Array, Array]]:
        # shac policy loss
        envs, data, next_discounts, next_obs, next_done, sum_rewards, key = collect_rollout(
            envs, actor_params, next_obs, next_done, sum_rewards, key
        )
        # compute loss as in SHAC paper Eq(5)
        losses = jp.sum(data.losses)
        return losses / (args.num_envs * args.num_steps), (
            envs,
            data,
            next_obs,
            next_done,
            sum_rewards,
            key,
        )

    policy_grad_fn = jax.value_and_grad(policy_loss_fn, argnums=(1,), has_aux=True)
    (p_loss, (envs, data, next_obs, next_done, sum_rewards, key)), (g_actor,) = policy_grad_fn(
        envs, agent.actor_states.params, next_obs, next_done, sum_rewards, key
    )

    agent = agent.replace(actor_states=agent.actor_states.apply_gradients(grads=g_actor))
    return (envs, agent, key), (p_loss, (data, next_obs, next_done, sum_rewards))


# region Train
def make_train_fn(args: Args) -> Callable:
    """Create a JIT-compiled training function that runs args.num_iterations via lax.scan.

    Both setup_train and the training call reuse this exact function object, so the scan context is
    compiled once.
    """

    def train_iteration(carry: tuple, _) -> tuple[tuple, tuple]:
        envs, agent, key, next_obs, next_done, sum_rewards = carry
        (envs, agent, key), (p_loss, (data, next_obs, next_done, sum_rewards)) = update_policy(
            envs=envs,
            args=args,
            agent=agent,
            next_obs=next_obs,
            next_done=next_done,
            sum_rewards=sum_rewards,
            key=key,
        )
        return (envs, agent, key, next_obs, next_done, sum_rewards), (p_loss, data)

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
            num_envs, num_steps, hidden_size) must match those used in subsequent train_bptt()
            calls.
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
        reset_random=False,
    )
    train_fn = make_train_fn(args)

    warmup_seed = args.seed + 10_000
    warmup_key = jax.random.PRNGKey(warmup_seed)
    init_key, _ = jax.random.split(warmup_key)
    actor_lr: float | optax.Schedule = args.actor_lr
    if args.anneal_actor_lr:
        actor_lr = optax.linear_schedule(
            init_value=args.actor_lr, end_value=0.0, transition_steps=args.num_iterations
        )
    warmup_agent = Agent.create(
        key=init_key,
        obs_dim=envs.single_observation_space.shape[0],
        act_dim=envs.single_action_space.shape[0],
        hidden_size=args.hidden_size,
        actor_lr=actor_lr,
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


def train_bptt(
    args: Args,
    model_path: Path,
    jax_device: str,
    train_fn: Callable,
    envs: FigureEightEnv,
    template_agent: Agent,
    wandb_enabled: bool = False,
) -> tuple[list[float], float]:
    """Train a BPTT policy with an end-to-end JAX/Flax compiled training loop.

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

    (envs, agent, key, next_obs, next_done, sum_rewards), (p_losses, all_data) = train_fn(
        envs, agent, key, next_obs, next_done, sum_rewards
    )

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
                    "losses/p_loss": float(jp.mean(p_losses[iter_idx])),
                    "losses/entropy": float(jp.mean(data.entropy)),
                    "charts/SPS": int(global_step / training_time) if training_time > 0 else 0,
                },
                step=global_step + args.batch_size,
            )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    params = {"actor": agent.actor_states.params, "samples_dt": args.samples_dt}
    with open(model_path, "wb") as f:
        pickle.dump(params, f)
    logger.info(f"model saved to {model_path}")

    return sum_rewards_hist, training_time


# region Evaluate
def evaluate_bptt(
    args: Args, n_eval: int, model_path: Path, render: bool
) -> tuple[float, list, list]:
    """Run `n_eval` episodes with deterministic actions from the checkpoint at `model_path`."""
    r_coefs = reward_coefs(args)
    eval_env = make_jitted_envs(
        num_envs=1, jax_device=args.jax_device, coefs=r_coefs, reset_rotor=True
    )
    eval_env = RecordData.create(eval_env)

    agent = Agent.create(
        key=jax.random.PRNGKey(0),
        obs_dim=eval_env.single_observation_space.shape[0],
        act_dim=eval_env.single_action_space.shape[0],
        hidden_size=args.hidden_size,
        init_logstd=args.init_logstd,
    )
    with open(model_path, "rb") as f:
        params = pickle.load(f)
    agent = agent.replace(actor_states=agent.actor_states.replace(params=params["actor"]))

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
    """
    template_args = Args.create(preset)
    jax_device = template_args.jax_device
    seed = template_args.seed if seed is None else seed

    # Build the JIT-compiled train function and envs once, so every run reuses them.
    # Evaluation loads its own env and agent, so this is skipped with nothing to train.
    if train:
        train_fn, envs, template_agent = setup_train(template_args, jax_device=jax_device)

    for run_idx in range(n_runs):
        run_seed = seed + run_idx
        args = Args.create(preset, seed=run_seed)
        saves = MODEL_DIR if save_dir is None else Path(save_dir)
        model_path = saves / checkpoint_name(
            args.exp_name, args.trajectory_time, seed=run_seed if n_runs > 1 else None
        )

        if n_runs > 1:
            logger.info(f"Run {run_idx + 1}/{n_runs} (seed={run_seed})")

        if train:
            train_bptt(args, model_path, jax_device, train_fn, envs, template_agent, wandb_enabled)

        if n_eval > 0:
            rmse_pos, episode_rewards, episode_lengths = evaluate_bptt(
                args, n_eval, model_path, render
            )
            if wandb_enabled and train:
                logs = {
                    "eval/mean_rewards": np.mean(episode_rewards),
                    "eval/mean_steps": np.mean(episode_lengths),
                    "eval/pos_rmse_mm": rmse_pos,
                }
                wandb.log(logs)
                wandb.finish()
        elif wandb_enabled and train and wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(main)
