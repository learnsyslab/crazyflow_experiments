"""TD3 training for the reach-position task with direct rotor velocity setpoints.

Training is non-deterministic, since deterministic GPU kernels cost roughly a factor of five in
throughput.

Run with
    pixi run -e learn2fly python crazyflow_experiments/learn2fly/train.py
"""

from __future__ import annotations

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import logging
import pickle
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import fire
import flax.struct as struct
import jax
import jax.lax as lax
import jax.numpy as jp
import numpy as np
import wandb
from crazyflow.control import Control
from crazyflow.sim import Dynamics
from jax import Array

from crazyflow_experiments.core.agents.td3_agent import (
    ReplayBuffer,
    TD3Agent,
    Transition,
    load_actor,
    update_policy,
)
from crazyflow_experiments.core.envs.drone_env import HOVER_RPM
from crazyflow_experiments.core.envs.figure8_env import FigureEightEnv
from crazyflow_experiments.core.envs.reach_pos_env import ReachPosEnv
from crazyflow_experiments.core.envs.wrappers import (
    ActionPenalty,
    AngleReward,
    CurriculumWrapper,
    FlattenJaxObservation,
    GoalRelativeObs,
    NormalizeActionsQuadratic,
    NormalizeObs,
    PrivilegedCriticObs,
    QuatToMatrixObs,
    RecordData,
    VelocityPenalty,
)
from crazyflow_experiments.learn2fly import MODEL_DIR
from crazyflow_experiments.utils.paths import RESULTS_DIR

logger = logging.getLogger(__name__)


SIM_SETTINGS = {
    "freq": 100,
    "sim_freq": 500,
    "drone": "cf2x_L250",  # Use cf2x brushed for fair comparison since it was used in L2F
    "dynamics": Dynamics.first_principles,
    "control": Control.rotor_vel,
    "reset_rotor": True,  # Start with the rotors spinning at hover speed
}

FIGURE8_EVAL_NUM_ENVS = 64
FIGURE8_TRAJECTORY_TIMES = {"3.5s": 3.5, "5.5s": 5.5, "15s": 15.0}


# region Arguments
# https://wandb.ai/amacati/td3-reach-rv-sweep/sweeps/lsavewi1/runs/3q9azz27/overview
@dataclass(frozen=True)
class Args:
    """Configuration for TD3 training."""

    seed: int = 42
    jax_device: str = "gpu"
    exp_name: str = "td3_rprv"
    wandb_project_name: str = "td3-reach-rv"
    wandb_entity: str | None = None
    wandb_group: str | None = None

    total_timesteps: int = 20_000_000
    num_envs: int = 4096
    num_steps: int = 1
    """Steps per env per rollout."""
    updates_epochs: int = 8
    """Gradient updates per rollout."""
    buffer_size: int = 4096 * 1024
    batch_size: int = 2048
    learning_starts: int = 98_304
    """Timesteps of random exploration before training starts."""
    actor_lr: float = 1.34e-4
    critic_lr: float = 1.8e-3
    gamma: float = 0.984
    tau: float = 0.005
    """Polyak averaging rate of the target networks."""
    policy_delay: int = 1
    """Number of critic updates between actor updates."""
    exploration_noise: float = 0.3
    policy_noise: float = 0.05
    """Standard deviation of the target policy smoothing noise."""
    noise_clip: float = 0.1
    """clip target policy noise"""
    curriculum: bool = False

    # Network architecture
    hidden_size: int = 64
    num_layers: int = 2
    hidden_size_critic: int = 128
    num_layers_critic: int = 3

    # Envs settings
    pos_min: Array = (-0.5, -0.5, 1.2)
    pos_max: Array = (0.5, 0.5, 1.8)
    goal_pmin: Array = (-0.0, -0.0, 1.5)
    goal_pmax: Array = (0.0, 0.0, 1.5)
    vel_min: float = -0.0898
    vel_max: float = 0.0898
    ang_vel_min: Array = (-0.55,) * 3
    ang_vel_max: Array = (0.55,) * 3

    # Wrapper settings
    num_last_actions: int = 4
    rpy_coef: float = 0.368
    act_coefs: tuple = (0.077,) * 4
    d_act_coefs: tuple = (0.173,) * 4
    vel_penalty_weight: float = 0.0
    # Curriculum settings
    success_threshold: float = 0.75
    ema_alpha: float = 0.01
    goal_resample_interval: int = 80

    def __post_init__(self):
        """Check that the actor updates divide the gradient updates of a rollout."""
        if self.updates_epochs % self.policy_delay != 0:
            raise ValueError(
                f"updates_epochs ({self.updates_epochs}) must be divisible by policy_delay "
                f"({self.policy_delay})"
            )

    @property
    def rollout_batchsize(self) -> int:
        """Steps collected per rollout."""
        return self.num_steps * self.num_envs

    @property
    def num_iterations(self) -> int:
        """Training iterations."""
        return (self.total_timesteps - self.learning_starts) // self.rollout_batchsize


# region MakeEnvs
def _action_obs_wrappers(env: struct.PyTreeNode, args: Args) -> struct.PyTreeNode:
    """Add the wrappers for action and observation transforms."""
    env = NormalizeActionsQuadratic.create(env, neutral=jp.full(4, HOVER_RPM))
    env = ActionPenalty.create(
        env,
        num_actions=args.num_last_actions,
        act_coefs=args.act_coefs,
        d_act_coefs=args.d_act_coefs,
    )
    return QuatToMatrixObs.create(env)


def make_reach_pos_envs(num_envs: int, args: Args) -> ReachPosEnv:
    """Create wrapped environments for TD3 training."""
    env: ReachPosEnv = ReachPosEnv.create(
        num_envs=num_envs,
        device=args.jax_device,
        max_episode_time=7.0,
        **SIM_SETTINGS,
        pos_min=jp.array(args.pos_min),
        pos_max=jp.array(args.pos_max),
        goal_pmin=jp.array(args.goal_pmin),
        goal_pmax=jp.array(args.goal_pmax),
        ang_vel_min=jp.array(args.ang_vel_min),
        ang_vel_max=jp.array(args.ang_vel_max),
        vel_min=args.vel_min,
        vel_max=args.vel_max,
    )
    if args.curriculum:
        env = CurriculumWrapper.create(
            env,
            success_threshold=args.success_threshold,
            ema_alpha=args.ema_alpha,
            goal_resample_interval=args.goal_resample_interval,
            resample_min=jp.array(args.pos_min),
            resample_max=jp.array(args.pos_max),
            freq=100,
        )
    env = AngleReward.create(env, weight=args.rpy_coef)
    if args.vel_penalty_weight > 0.0:
        env = VelocityPenalty.create(env, threshold=2.75, weight=args.vel_penalty_weight)
    env = _action_obs_wrappers(env, args)
    env = PrivilegedCriticObs.create(env)
    env = FlattenJaxObservation.create(env)
    obs_dim = env.single_observation_space.shape[0]
    env = NormalizeObs.create(
        env,
        mean=jp.concatenate([jp.zeros(obs_dim - 4), jp.full(4, HOVER_RPM)]),
        scale=jp.concatenate([jp.ones(obs_dim - 4), jp.full(4, 3000.0)]),
    )

    return env


# region Utils
@struct.dataclass
class RolloutData:
    """Episode statistics of a single rollout step, batched over envs."""

    terminated: Array
    truncated: Array
    success: Array
    episode_length: Array
    sum_rewards: Array


# region Rollout
def collect_rollout(
    envs: struct.PyTreeNode,
    args: Args,
    agent: TD3Agent,
    obs: Array,
    sum_rewards: Array,
    key: Array,
    buffer: ReplayBuffer,
    random_action: bool,
) -> tuple[struct.PyTreeNode, ReplayBuffer, Array, Array, Array, RolloutData]:
    """Collect rollout data and add it to the buffer."""
    n_steps = args.num_steps if not random_action else args.learning_starts // args.num_envs

    def step_once(
        carry: tuple[struct.PyTreeNode, Array, Array, Array], _: Any
    ) -> tuple[tuple[struct.PyTreeNode, Array, Array, Array], tuple[Transition, RolloutData]]:
        env, key, sum_rewards, obs = carry

        if random_action:
            action, key = agent.random_action(args.num_envs, key)
        else:
            action, key = agent.sample_action(
                agent.actor_states.params, obs, key, std=args.exploration_noise
            )

        env, (next_obs, reward, terminated, truncated, info) = env.step(env, action)
        sum_rewards = sum_rewards + reward
        next_sum_rewards = jp.where(terminated | truncated, 0.0, sum_rewards)

        # Environments use SAME_STEP autoreset: next_obs is the post-reset initial observation.
        # info["final_obs"] holds the true pre-reset final state, which is what the TD bootstrap
        # should use. For terminated transitions the bootstrap is masked out anyway, but for
        # truncated transitions we must bootstrap from the true final state, not the reset state, to
        # avoid a systematic bias in the value estimate.
        transition = Transition(
            obs=obs, action=action, reward=reward, next_obs=info["final_obs"], terminated=terminated
        )
        data = RolloutData(
            terminated=terminated,
            truncated=truncated,
            success=info["success"],
            episode_length=info["episode_steps"],
            sum_rewards=sum_rewards,  # Store before reset for logging
        )

        return (env, key, next_sum_rewards, next_obs), (transition, data)

    (envs, key, sum_rewards, obs), (transitions, rollout_data) = lax.scan(
        step_once, (envs, key, sum_rewards, obs), length=n_steps
    )

    buffer = buffer.add(buffer, transitions)
    return envs, buffer, obs, sum_rewards, key, rollout_data


# region make_train_fn


def make_train_fn(args: Args) -> Callable:
    """Create a JIT-compiled training function."""

    def train_iteration(carry: tuple, _) -> tuple[tuple, tuple]:
        envs, agent, buffer, obs, sum_rewards, key = carry
        envs, buffer, obs, sum_rewards, key, rollout_data = collect_rollout(
            envs, args, agent, obs, sum_rewards, key, buffer, random_action=False
        )
        agent, critic_loss, actor_loss, key = update_policy(
            agent,
            buffer,
            key,
            epochs=args.updates_epochs,
            batch_size=args.batch_size,
            gamma=args.gamma,
            tau=args.tau,
            policy_noise=args.policy_noise,
            noise_clip=args.noise_clip,
            policy_delay=args.policy_delay,
        )
        return (envs, agent, buffer, obs, sum_rewards, key), (rollout_data, critic_loss, actor_loss)

    @jax.jit
    def train_fn(
        envs: ReachPosEnv,
        agent: TD3Agent,
        obs: Array,
        sum_rewards: Array,
        key: Array,
        buffer: ReplayBuffer,
    ) -> tuple[TD3Agent, tuple]:
        envs, buffer, obs, sum_rewards, key, pre_rollout_data = collect_rollout(
            envs, args, agent, obs, sum_rewards, key, buffer, random_action=True
        )
        (
            (envs, agent, buffer, obs, sum_rewards, key),
            (all_rollout_data, all_critic_loss, all_actor_loss),
        ) = lax.scan(
            train_iteration,
            (envs, agent, buffer, obs, sum_rewards, key),
            length=args.num_iterations,
        )
        return agent, (pre_rollout_data, all_rollout_data, all_critic_loss, all_actor_loss)

    return train_fn


def setup_train(args: Args) -> tuple[Callable, ReachPosEnv, TD3Agent]:
    """Create environments and JIT-compile train_fn through a warmup run.

    Args:
        args: Training configuration.

    Returns:
        The compiled train_fn, the template envs and the warmup agent to reinit from.
    """
    envs = make_reach_pos_envs(args.num_envs, args)
    train_fn = make_train_fn(args)

    warmup_seed = args.seed + 10_000
    warmup_key = jax.random.PRNGKey(warmup_seed)
    init_key, _ = jax.random.split(warmup_key)
    obs_dim = envs.single_observation_space.shape[0]
    warmup_agent = TD3Agent.create(
        key=init_key,
        obs_dim=obs_dim,
        act_dim=envs.single_action_space.shape[0],
        actor_obs_dim=obs_dim - 4,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        hidden_size_critic=args.hidden_size_critic,
        num_layers_critic=args.num_layers_critic,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
    )
    buffer = ReplayBuffer.create(
        capacity=args.buffer_size, obs_dim=obs_dim, act_dim=envs.single_action_space.shape[0]
    )
    buffer = jax.device_put(buffer)

    warmup_envs, (warmup_obs, _) = envs.reset(envs, seed=warmup_seed)
    warmup_sum_rewards = jp.zeros((args.num_envs,), dtype=jp.float32)

    jax.block_until_ready(
        train_fn(warmup_envs, warmup_agent, warmup_obs, warmup_sum_rewards, warmup_key, buffer)
    )
    return train_fn, envs, warmup_agent


# region train_td3
def _masked_mean(values: np.ndarray, mask: np.ndarray, count: np.ndarray) -> np.ndarray:
    """Average the masked entries of the last axis, leaving zeros where the mask is empty."""
    return np.where(mask, values, 0.0).sum(axis=-1) / np.maximum(count, 1)


def train_td3(
    args: Args,
    model_path: Path,
    wandb_enabled: bool,
    train_fn: Callable,
    envs: ReachPosEnv,
    agent: TD3Agent,
) -> tuple[list[float], float, TD3Agent]:
    """Train TD3 agent."""
    if wandb_enabled and wandb.run is None:
        config = vars(args) | {
            "rollout_batchsize": args.rollout_batchsize,
            "num_iterations": args.num_iterations,
        }
        init_kwargs = {"project": args.wandb_project_name, "config": config, "dir": RESULTS_DIR}
        if args.wandb_entity:
            init_kwargs["entity"] = args.wandb_entity
        if args.wandb_group:
            init_kwargs["group"] = args.wandb_group
        wandb.init(**init_kwargs)

    setup_start = time.time()
    key = jax.random.PRNGKey(args.seed)
    logger.info(f"Training on device: {args.jax_device}")

    # Reinitialize the parameters of the template agent for this run
    key, init_key = jax.random.split(key)
    obs_dim = envs.single_observation_space.shape[0]
    agent = agent.reinit(init_key, obs_dim)

    # Create buffer
    buffer = ReplayBuffer.create(
        capacity=args.buffer_size, obs_dim=obs_dim, act_dim=envs.single_action_space.shape[0]
    )
    buffer = jax.device_put(buffer)
    logger.info(f"Setup took {time.time() - setup_start:.2f}s")

    # Reset environments
    envs, (obs, _) = envs.reset(envs, seed=args.seed)
    sum_rewards = jp.zeros((args.num_envs,), dtype=jp.float32)

    logger.info(
        f"Training: {args.num_iterations} iterations (N={args.num_steps}, M={args.updates_epochs})"
    )

    train_start = time.time()
    logger.info("Start training TD3...")
    agent, (pre_rollout_data, all_rollout_data, all_critic_loss, all_actor_loss) = train_fn(
        envs, agent, obs, sum_rewards, key, buffer
    )
    jax.block_until_ready(agent.actor_states.params)
    training_time = time.time() - train_start
    sps = args.total_timesteps / training_time
    logger.info(f"Training {args.total_timesteps} steps took {training_time:.2f}s ({sps:.0f} SPS)")

    # Convert to numpy once, so the logging below does not repeatedly transfer from the GPU
    prefill = jax.tree.map(np.asarray, pre_rollout_data)
    stats = jax.tree.map(np.asarray, all_rollout_data)  # (num_iterations, num_steps, num_envs)
    critic_losses, actor_losses = np.asarray(all_critic_loss), np.asarray(all_actor_loss)

    # Masked mean over the envs that finished. Dividing by a clamped count keeps NaNs out.
    prefill_done = prefill.terminated | prefill.truncated
    prefill_count = prefill_done.sum(axis=-1)
    prefill_rewards = _masked_mean(prefill.sum_rewards, prefill_done, prefill_count)
    reward_history = prefill_rewards[prefill_count > 0].tolist()

    done = stats.terminated | stats.truncated
    count = done.sum(axis=-1)
    any_done = count > 0
    rewards = _masked_mean(stats.sum_rewards, done, count)
    episode_lengths = _masked_mean(stats.episode_length, done, count)
    # Both rates are taken over all ended episodes, so timeouts count as failures
    success_rates = _masked_mean(stats.success, stats.terminated, count)
    timeout_rates = stats.truncated.sum(axis=-1) / np.maximum(count, 1)

    if len(reward_history) > 0 and wandb_enabled:
        wandb.log({"train/episode_reward": np.mean(reward_history)}, step=args.learning_starts)

    global_steps = (
        args.learning_starts + (np.arange(args.num_iterations) + 1) * args.rollout_batchsize
    )
    if wandb_enabled:
        for iter_idx in range(args.num_iterations):
            global_step = int(global_steps[iter_idx])

            for step_idx in range(args.num_steps):
                if any_done[iter_idx, step_idx]:
                    reward_history.append(float(rewards[iter_idx, step_idx]))
                    wandb.log(
                        {
                            "train/episode_reward": float(rewards[iter_idx, step_idx]),
                            "train/success_rate": float(success_rates[iter_idx, step_idx]),
                            "train/timeout_rate": float(timeout_rates[iter_idx, step_idx]),
                            "train/episode_length": float(episode_lengths[iter_idx, step_idx]),
                        },
                        step=global_step,
                    )

            if iter_idx % 10 == 0:
                wandb.log(
                    {
                        "losses/critic_loss": float(critic_losses[iter_idx]),
                        "losses/actor_loss": float(actor_losses[iter_idx]),
                    },
                    step=global_step,
                )

        wandb.log({"charts/SPS": sps}, step=args.total_timesteps)

    # Save model
    if model_path is not None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        params = {
            "actor": agent.actor_states.params,
            "critic1": agent.critic1_state.params,
            "critic2": agent.critic2_state.params,
        }
        with open(model_path, "wb") as f:
            pickle.dump(params, f)
        logger.info(f"Model saved to {model_path}")

    return reward_history, training_time, agent


# region Evaluate
def _log_eval_to_wandb(
    eval_env: RecordData, prefix: str, rmse: float, rewards: list[float], lengths: list[float]
):
    """Log the evaluation metrics and the trajectory of env 0 to WandB."""
    wandb.log(
        {
            f"{prefix}/rmse_mm": rmse * 1000,
            f"{prefix}/mean_reward": float(np.mean(rewards)),
            f"{prefix}/mean_steps": float(np.mean(lengths)),
        }
    )
    num_steps = int(eval_env._record_steps[0])
    pos = np.array(eval_env._record_pos)[:num_steps, 0]
    goal = np.array(eval_env._record_goal)[:num_steps, 0]
    actions = np.array(eval_env._record_act)[:num_steps, 0]
    pos_err = np.linalg.norm(pos - goal, axis=-1)
    T = pos.shape[0]

    # Scatter rather than line_series, which sorts by x and would scramble a figure-8
    xy_table = wandb.Table(
        data=[[float(pos[t, 0]), float(pos[t, 1]), "Actual"] for t in range(T)]
        + [[float(goal[t, 0]), float(goal[t, 1]), "Reference"] for t in range(T)],
        columns=["x", "y", "series"],
    )
    wandb.log(
        {f"{prefix}/xy_trajectory": wandb.plot.scatter(xy_table, "x", "y", title="XY Trajectory")}
    )

    # A custom x-axis makes these render as line charts without adding *_table summary entries
    step_key = f"{prefix}/eval_step"
    wandb.define_metric(step_key)
    for metric in ["pos_err", "rotor_1", "rotor_2", "rotor_3", "rotor_4"]:
        wandb.define_metric(f"{prefix}/{metric}", step_metric=step_key)
    for t in range(T):
        wandb.log(
            {
                step_key: t,
                f"{prefix}/pos_err": float(pos_err[t]),
                f"{prefix}/rotor_1": float(actions[t, 0]),
                f"{prefix}/rotor_2": float(actions[t, 1]),
                f"{prefix}/rotor_3": float(actions[t, 2]),
                f"{prefix}/rotor_4": float(actions[t, 3]),
            }
        )


def _eval_episodes(
    agent: TD3Agent, eval_env: RecordData, n_eval: int, seed: int, render: bool = False
) -> tuple[RecordData, float, list[float], list[float]]:
    """Run episodes with the deterministic policy until every env has ended its episode."""
    episode_rewards, episode_lengths, pos_errors = [], [], []

    for episode in range(n_eval):
        eval_env, (obs, _) = eval_env.reset(eval_env, seed=seed + episode + 1)
        finished = np.zeros((eval_env.num_envs,), dtype=bool)
        end_steps = np.zeros((eval_env.num_envs,), dtype=np.int32)
        episode_reward = 0.0
        steps = 0

        while not finished.all():  # All envs have to finish
            action = agent.mean_action(agent.actor_states.params, obs)
            eval_env, (obs, reward, terminated, truncated, _) = eval_env.step(eval_env, action)
            steps += 1

            if render:
                fps = 50
                if ((steps * fps) % eval_env.unwrapped.freq) < fps:
                    eval_env.render()

            # Envs past their first episode no longer contribute to stats
            episode_reward += float(np.asarray(reward)[~finished].mean())
            done = np.asarray(terminated | truncated) & ~finished
            end_steps = np.where(done, steps, end_steps)
            finished |= done

        episode_rewards.append(episode_reward)
        episode_lengths.append(float(end_steps.mean()))
        pos_errors.append(eval_env.pos_error())

    if render:
        eval_env.close()
    rmse_pos = float(np.sqrt(np.mean(np.concatenate(pos_errors) ** 2)))
    return eval_env, rmse_pos, episode_rewards, episode_lengths


def eval_reach_pos(
    args: Args,
    n_eval: int,
    model_path: Path | None = None,
    render: bool = False,
    wandb_enabled: bool = False,
    agent: TD3Agent | None = None,
    eval_env: RecordData | None = None,
) -> tuple[float, list[float], list[float]]:
    """Evaluate trained TD3 policy on the reach-position task."""
    if eval_env is None:
        eval_env = RecordData.create(make_reach_pos_envs(1, args))

    agent = agent if agent is not None else load_actor(model_path)
    eval_env, rmse_pos, episode_rewards, episode_lengths = _eval_episodes(
        agent, eval_env, n_eval, args.seed, render
    )
    if wandb_enabled:
        _log_eval_to_wandb(eval_env, "eval(reach)", rmse_pos, episode_rewards, episode_lengths)

    return rmse_pos, episode_rewards, episode_lengths


def make_figure8_env(num_envs: int, args: Args, trajectory_time: float) -> FigureEightEnv:
    """Create a wrapped XY-plane figure-8 env for tracking evaluation."""
    env: FigureEightEnv = FigureEightEnv.create(
        num_envs=num_envs,
        device=args.jax_device,
        max_episode_time=trajectory_time,
        **SIM_SETTINGS,
        trajectory_time=trajectory_time,
        plane="xy",
        n_samples=0,  # The policy tracks the current target, without a lookahead
    )

    env = GoalRelativeObs.create(env)
    return FlattenJaxObservation.create(_action_obs_wrappers(env, args))


def make_figure8_envs(args: Args) -> dict[str, RecordData]:
    """Create three XY-plane figure-8 evaluation environments with different trajectory times."""
    return {
        label: RecordData.create(make_figure8_env(FIGURE8_EVAL_NUM_ENVS, args, time))
        for label, time in FIGURE8_TRAJECTORY_TIMES.items()
    }


def eval_figure8(
    args: Args,
    n_eval: int,
    model_path: Path | None = None,
    agent: TD3Agent | None = None,
    render: bool = False,
    wandb_enabled: bool = False,
    eval_envs: dict[str, RecordData] | None = None,
) -> dict[str, tuple[float, list[float], list[float]]]:
    """Evaluate the policy on figure-8 tracking."""
    if eval_envs is None:
        eval_envs = make_figure8_envs(args)

    if agent is None:
        agent = load_actor(model_path)

    results = {}
    for label, eval_env in eval_envs.items():
        eval_env, rmse_pos, episode_rewards, episode_lengths = _eval_episodes(
            agent, eval_env, n_eval, args.seed, render
        )
        logger.info(
            f"Tracking [{label}] Mean Reward: {np.mean(episode_rewards):.2f}, "
            f"RMSE: {rmse_pos * 1000:.3f} mm"
        )
        if wandb_enabled:
            _log_eval_to_wandb(
                eval_env, f"eval({label})", rmse_pos, episode_rewards, episode_lengths
            )
        results[label] = (rmse_pos, episode_rewards, episode_lengths)

    return results


# region Main
def main(
    wandb_enabled: bool = True,
    do_train: bool = True,
    n_eval: int = 1,
    render: bool = False,
    save: bool = True,
    save_dir: str = str(MODEL_DIR),
    seed: int | None = None,
    n_runs: int = 1,
    wandb_group: str | None = None,
    curriculum: bool = False,
):
    """Main entry point.

    Args:
        wandb_enabled: Log metrics to W&B.
        do_train: Run training (False to evaluate a saved model only).
        n_eval: Number of evaluation episodes after training (0 to skip).
        render: Render the environment during evaluation.
        save: Save the trained model to disk.
        save_dir: Directory the checkpoints are written to.
        seed: Base RNG seed. Each run draws a unique seed from this.
        n_runs: Number of independent training runs.
        wandb_group: W&B group name for comparing multiple runs.
        curriculum: Flag to turn on curriculum learning.
    """
    base_seed = Args.seed if seed is None else seed
    rng = random.Random(base_seed)

    kwargs: dict[str, Any] = {"curriculum": curriculum}
    if wandb_group is not None:
        kwargs["wandb_group"] = wandb_group

    # Compile training function and build eval envs once for all runs.
    template_args = Args(**kwargs, seed=base_seed)
    train_fn, train_envs, template_agent = setup_train(template_args)

    fig8_eval_envs = None
    reach_eval_env = None
    if n_eval > 0:
        fig8_eval_envs = make_figure8_envs(template_args)
        reach_eval_env = RecordData.create(make_reach_pos_envs(1, template_args))

    for run_idx in range(n_runs):
        run_seed = rng.randint(0, 2**31 - 1)
        args = Args(**kwargs, seed=run_seed)
        model_stem = args.exp_name if n_runs == 1 else f"{args.exp_name}_run{run_idx}"
        model_path = Path(save_dir) / f"{model_stem}_model.ckpt" if save else None

        logger.info(f"Run {run_idx + 1}/{n_runs}: seed={run_seed}")

        trained_agent = None
        if do_train:
            _, _, trained_agent = train_td3(
                args,
                model_path,
                wandb_enabled,
                train_fn=train_fn,
                envs=train_envs,
                agent=template_agent,
            )

        if n_eval > 0:
            eval_agent = trained_agent if do_train else None
            eval_figure8(
                args,
                n_eval,
                model_path=model_path,
                agent=eval_agent,
                render=render,
                wandb_enabled=wandb_enabled and do_train,
                eval_envs=fig8_eval_envs,
            )
            eval_reach_pos(
                args,
                n_eval,
                model_path=model_path,
                agent=eval_agent,
                render=render,
                wandb_enabled=wandb_enabled and do_train,
                eval_env=reach_eval_env,
            )

        if wandb_enabled and wandb.run is not None:
            wandb.finish()

    if fig8_eval_envs is not None:
        for env in fig8_eval_envs.values():
            env.close()
    if reach_eval_env is not None:
        reach_eval_env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("jax").setLevel(logging.WARNING)
    fire.Fire(main)
