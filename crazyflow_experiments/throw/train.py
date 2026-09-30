"""Train a throw recovery policy with BPTT.

Run with
    pixi run -e throw python crazyflow_experiments/throw/train.py
"""

from __future__ import annotations

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["SCIPY_ARRAY_API"] = "1"

import functools
import logging
import pickle
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import fire
import flax
import flax.struct as struct
import jax
import jax.numpy as jp
import numpy as np
from crazyflow.dynamics import load_params
from crazyflow.sim import Dynamics
from jax import Array
from scipy.spatial.transform import Rotation as R

import wandb
from crazyflow_experiments.core.agents.bptt_agent_deterministic import Agent
from crazyflow_experiments.core.envs.throw_env import ThrowEnv
from crazyflow_experiments.core.envs.wrappers import (
    ActionPenalty,
    AngleReward,
    ClipObs,
    FlattenJaxObservation,
    NormalizeActions,
    QuatToMatrixObs,
    TerminateOnMaxAngVel,
)
from crazyflow_experiments.throw import X_START_NOISE
from crazyflow_experiments.utils.paths import RESULTS_DIR, SAVES_DIR

logger = logging.getLogger(__name__)

MODEL_DIR = SAVES_DIR / "throw"
OBS_CLIP = 20.0  # Bound on every observation entry seen by the policy
ENV_FREQ = 100  # Hz, rate at which the policy acts
IDLE_ACTION = -1.0  # Normalized rotor thrust of the free fall before the policy takes over
_PARAMS = load_params(Dynamics.first_principles, "cf21B_500")
_HOVER_THRUST = -float(_PARAMS["gravity_vec"][2]) * float(_PARAMS["mass"]) / 4  # N per rotor
_THRUST_MIN, _THRUST_MAX = float(_PARAMS["thrust_min"]), float(_PARAMS["thrust_max"])
# Normalized rotor thrust that holds the drone in hover
HOVER_ACTION = 2 * (_HOVER_THRUST - _THRUST_MIN) / (_THRUST_MAX - _THRUST_MIN) - 1


def flat_obs(pos: Array, quat: Array, vel: Array, ang_vel: Array, last_action: Array) -> Array:
    """Turn a drone state into a flattened observation, converting the quaternion to a matrix."""
    rot_mat = R.from_quat(quat).as_matrix()
    obs = {
        "ang_vel": ang_vel,
        "last_action": last_action,
        "rot_mat": rot_mat.reshape(*rot_mat.shape[:-2], 9),
        "vel": vel,
        "z": pos[..., 2:3],
    }
    return jp.clip(jp.concatenate([obs[k] for k in sorted(obs)], axis=-1), -OBS_CLIP, OBS_CLIP)


X_START = {
    "pos": jp.array([[[0.0, 0.0, 2.2]]]),
    "quat": jp.array([[[0.0, 0.0, 0.0, 1.0]]]),
    "vel": jp.array([[[0.0, 0.0, -1.0]]]),
    "ang_vel": jp.array([[[1.0, 0.5, 0.0]]]),
}


# region Arguments
# Defaults set based on the best performing hyperparameters found in the W&B sweep, except for the
# height and velocity reward, changed so the policy holds its height after the throw
@dataclass(frozen=True)
class Args:
    """Class to store configurations."""

    seed: int = 42
    """seed of the experiment"""
    jax_device: str = "cpu"
    """environment device"""
    exp_name: str = "bptt_throw"
    """the name of the experiment"""
    wandb_project_name: str = "rl-bptt-throw"
    """the wandb's project name"""
    wandb_entity: str | None = None
    """optional wandb entity (team)"""
    wandb_group: str | None = None
    """optional wandb group"""

    # Algorithm specific arguments
    total_timesteps: int = 180_000
    """total timesteps of the experiments"""
    num_envs: int = 32
    """the number of parallel game environments"""
    num_steps: int = 48
    """the number of steps to run in each environment per policy rollout"""
    actor_lr: float = 0.015670855096441366
    """the learning rate of the actor optimizer"""
    gamma: float = 0.9397693036348193
    """the discount factor gamma"""
    hidden_size: int = 8
    """the hidden size of actor and critic networks"""

    # Reward coefficients
    height_weight: float = 0.2
    vel_weight: float = 1.5
    vel_coef: float = 1.0
    floor_weight: float = 1.0
    floor_coef: float = 10.0
    ceiling_weight: float = 10.0
    ceiling_coef: float = 1.0
    fall_weight_below: float = 0.3
    fall_weight_above: float = 0.2

    # Wrapper settings
    angle_weight: float = 0.4438053990762441
    act_coefs: tuple = (0.016885701484858687,) * 4
    d_act_coefs: tuple = (0.07708131243138613,) * 4
    """reward coefficients for training"""

    @property
    def batch_size(self) -> int:
        """Environment steps collected per training iteration."""
        return self.num_envs * self.num_steps

    @property
    def num_iterations(self) -> int:
        """Training iterations."""
        return self.total_timesteps // self.batch_size


# region MakeEnvs
def make_jitted_envs(
    args: Args,
    jax_device: str = "cpu",
    override_num_envs: int | None = None,
    use_x_start: bool = False,
) -> ThrowEnv:
    """Make rotor-thrust environments for training RL policy."""
    env: ThrowEnv = ThrowEnv.create(
        max_episode_time=4.0,
        num_envs=args.num_envs if override_num_envs is None else override_num_envs,
        freq=ENV_FREQ,
        sim_freq=200,
        device=jax_device,
        control_type="rotor_thrust",
        vel_coef=args.vel_coef,
        height_weight=args.height_weight,
        vel_weight=args.vel_weight,
        floor_weight=args.floor_weight,
        floor_coef=args.floor_coef,
        ceiling_weight=args.ceiling_weight,
        ceiling_coef=args.ceiling_coef,
        fall_weight_below=args.fall_weight_below,
        fall_weight_above=args.fall_weight_above,
        use_x_start=use_x_start,
        x_start_noise=X_START_NOISE,
    )
    env = NormalizeActions.create(env)
    env = AngleReward.create(env, weight=args.angle_weight)
    env = ActionPenalty.create(
        env,
        num_actions=1,
        init_last_actions=jp.full((1, 4), IDLE_ACTION),
        hover_action=jp.full((4,), HOVER_ACTION),
        act_coefs=args.act_coefs,
        d_act_coefs=args.d_act_coefs,
    )
    env = TerminateOnMaxAngVel.create(env, max_ang_vel=100.0)
    env = QuatToMatrixObs.create(env)
    env = ClipObs.create(env, low=-OBS_CLIP, high=OBS_CLIP)
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
    terminations: Array
    truncations: Array
    successes: Array
    states: dict
    distance: Array
    sum_rewards: Array
    entropy: Array
    losses: Array


# region Policy Update
def update_policy(
    envs: struct.PyTreeNode,
    args: Args,
    agent: Agent,
    next_obs: Array,
    next_done: Array,
    sum_rewards: Array,
    key: Array,
) -> float:
    """BPTT policy updates."""

    def collect_rollout(
        envs: struct.PyTreeNode,
        actor_params: dict,
        next_obs: Array,
        next_done: Array,
        sum_rewards: Array,
        key: Array,
    ) -> tuple[RolloutData, Array, Array, Array, Array]:
        """Collect a rollout of length args.num_steps. Returns data dict and next obs/done/key."""
        discounts = jp.ones((args.num_envs,))

        def step_once(carry: tuple, _) -> tuple[tuple, tuple]:
            env, key, sum_rewards, discounts, obs, dones = carry
            (action, _, entropy), key = agent.sample_action(actor_params, obs, key)

            env, (next_obs, reward, terminations, truncations, info) = env.step(env, action)

            sum_rewards = sum_rewards + reward
            next_sum_rewards = jp.where(dones, 0.0, sum_rewards)
            loss = -discounts * reward
            next_discounts = jp.where(dones, 1.0, discounts * args.gamma)
            next_dones = terminations | truncations
            states = {
                "pos": env.unwrapped.data.states.pos[:, 0, :],
                "quat": env.unwrapped.data.states.quat[:, 0, :],
                "vel": env.unwrapped.data.states.vel[:, 0, :],
                "ang_vel": env.unwrapped.data.states.ang_vel[:, 0, :],
            }

            return (env, key, next_sum_rewards, next_discounts, next_obs, next_dones), RolloutData(
                observations=obs,
                actions=action,
                rewards=reward,
                dones=dones,
                terminations=terminations,
                truncations=truncations,
                successes=info["success"],
                states=states,
                distance=info["distance"],
                sum_rewards=sum_rewards,
                entropy=jp.mean(entropy),
                losses=loss,
            )

        (envs, key, sum_rewards, next_discounts, next_obs, next_done), rollouts = jax.lax.scan(
            step_once,
            (envs, key, sum_rewards, discounts, next_obs, next_done),
            length=args.num_steps,
        )

        return envs, rollouts, next_discounts, next_obs, next_done, sum_rewards, key

    def policy_loss_fn(
        envs: struct.PyTreeNode,
        actor_params: dict,
        next_obs: Array,
        next_done: Array,
        sum_rewards: Array,
        key: Array,
    ) -> tuple[Array, tuple[Array, Array, Array, Array]]:
        envs, data, next_discounts, next_obs, next_done, sum_rewards, key = collect_rollout(
            envs, actor_params, next_obs, next_done, sum_rewards, key
        )
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


# region Utils
def _set_x_start(envs: struct.PyTreeNode, x_start: dict) -> struct.PyTreeNode:
    """Recursively update x_start fields in the innermost ThrowEnv through the wrapper chain.

    The wrapper hierarchy is a static Python structure at JAX trace time, so isinstance checks
    resolve correctly. Only the leaf x_start array values are traced by JAX.

    Args:
        envs: The (possibly wrapped) environment.
        x_start: Dict with keys "pos", "quat", "vel", "ang_vel", each shape (1, 1, dim).

    Returns:
        Updated env with x_start injected into the inner ThrowEnv.
    """
    if isinstance(envs, ThrowEnv):
        return envs.replace(
            x_start_pos=x_start["pos"],
            x_start_quat=x_start["quat"],
            x_start_vel=x_start["vel"],
            x_start_ang_vel=x_start["ang_vel"],
        )
    return envs.replace(base=_set_x_start(envs.base, x_start))


@functools.partial(jax.jit, static_argnames=("seed",))
def reset_at_x_start(envs: ThrowEnv, x_start: dict, seed: int) -> tuple[ThrowEnv, Array]:
    """Reset the envs around x_start and return them with their first observation.

    Compiled on purpose: BPTT amplifies the rounding differences between a compiled and an eager
    reset into different policies, so deployment and replay must share this function to train the
    same policy.
    """
    envs, (next_obs, _) = envs.reset(_set_x_start(envs, x_start), seed=seed)
    return envs, next_obs


# region Train
def make_train_fns(args: Args, collect_stats: bool = True) -> tuple[Callable, Callable]:
    """Create train_iteration and train_fn closures for the given args.

    Args:
        args: Training configuration.
        collect_stats: If True, scan outputs include full RolloutData for logging. If False,
            only per-iteration policy losses are returned. The branch is resolved at JAX trace
            time, so unused RolloutData computations are eliminated by XLA when False.
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
        if collect_stats:
            return (envs, agent, key, next_obs, next_done, sum_rewards), (p_loss, data)
        return (envs, agent, key, next_obs, next_done, sum_rewards), p_loss

    @jax.jit
    def train_fn(
        envs: struct.PyTreeNode,
        agent: Agent,
        key: Array,
        next_obs: Array,
        next_done: Array,
        sum_rewards: Array,
        x_start: dict | None = None,
    ) -> tuple[tuple, tuple]:
        if x_start is not None:
            envs = _set_x_start(envs, x_start)
        (envs, agent, key, next_obs, next_done, sum_rewards), outputs = jax.lax.scan(
            train_iteration,
            (envs, agent, key, next_obs, next_done, sum_rewards),
            jp.arange(args.num_iterations),
        )
        return (envs, agent, key, next_obs, next_done, sum_rewards), outputs

    return train_iteration, train_fn


def setup_train(
    args: Args, jax_device: str = "cpu", collect_stats: bool = False, x_start: dict | None = None
) -> tuple[Callable, ThrowEnv, Agent]:
    """Create environments and JIT-compile the train function via a warmup run.

    Compiles train_fn once so that subsequent train() calls with the same hyperparameters
    (different seeds only) skip recompilation and run at full speed.

    Args:
        args: Training configuration. Hyperparameters that affect XLA compilation
            (num_iterations, num_envs, num_steps, hidden_size) must match those used in
            subsequent train() calls.
        jax_device: JAX device string.
        collect_stats: Must match the collect_stats value used in subsequent train() calls.
        x_start: Optional fixed initial state dict. Must match the x_start passed to train();
            passing x_start here ensures the warmup traces the same JIT branch as the actual runs.

    Returns:
        Tuple of (compiled train_fn, template envs ready to be reset, warmup agent for reinit).
    """
    envs = make_jitted_envs(args, jax_device=jax_device, use_x_start=x_start is not None)
    _, train_fn = make_train_fns(args, collect_stats=collect_stats)

    # Compile train_fn via a warmup run using a separate seed (result discarded)
    warmup_seed = args.seed + 10_000
    warmup_envs, (warmup_obs, _) = envs.reset(envs, seed=warmup_seed)
    warmup_key = jax.random.PRNGKey(warmup_seed)
    init_key, _ = jax.random.split(warmup_key)
    warmup_agent = Agent.create(
        key=init_key,
        obs_dim=warmup_envs.single_observation_space.shape[0],
        act_dim=warmup_envs.single_action_space.shape[0],
        hidden_size=args.hidden_size,
        actor_lr=args.actor_lr,
    )
    next_done = jp.zeros(args.num_envs, dtype=bool)
    sum_rewards_warmup = jp.zeros((args.num_envs,))
    jax.block_until_ready(
        train_fn(
            warmup_envs,
            warmup_agent,
            warmup_key,
            warmup_obs,
            next_done,
            sum_rewards_warmup,
            x_start,
        )
    )

    return train_fn, envs, warmup_agent


def train(
    args: Args,
    train_fn: Callable,
    envs: ThrowEnv,
    template_agent: Agent,
    model_path: Path | None,
    wandb_enabled: bool = False,
    collect_stats: bool = False,
    x_start: dict | None = None,
) -> tuple[float, Agent]:
    """Train a single run.

    The environments are reset with this run's seed and the agent is re-initialised, so JAX does not
    need to recompile.

    Args:
        args: Training configuration, including the seed for this run.
        train_fn: Compiled train function from setup_train.
        envs: Template environments from setup_train, built with the same hyperparameters.
        template_agent: Warmup agent from setup_train, whose optimiser reinit reuses.
        model_path: Path to save the trained model, or None to skip saving.
        wandb_enabled: Whether to log metrics to W&B.
        collect_stats: Whether to collect and log per-iteration rollout statistics. Must match
            what setup_train compiled with.
        x_start: Optional fixed initial state dict (pos, quat, vel, ang_vel).

    Returns:
        Tuple of (training_time, agent).
    """
    if wandb_enabled and wandb.run is None:
        config = vars(args) | {"batch_size": args.batch_size, "num_iterations": args.num_iterations}
        init_kwargs = {"project": args.wandb_project_name, "config": config, "dir": RESULTS_DIR}
        if args.wandb_entity:
            init_kwargs["entity"] = args.wandb_entity
        if args.wandb_group:
            init_kwargs["group"] = args.wandb_group
        wandb.init(**init_kwargs)

    key = jax.random.PRNGKey(args.seed)

    init_key, key = jax.random.split(key)
    # setup_train built template_agent with the same optimiser, which reinit reuses.
    agent = template_agent.reinit(init_key, envs.single_observation_space.shape[0])

    if x_start is not None:  # The first episodes have to start around x_start as well
        envs, next_obs = reset_at_x_start(envs, x_start, seed=args.seed)
    else:
        envs, (next_obs, _) = envs.reset(envs, seed=args.seed)
    next_done = jp.zeros(args.num_envs, dtype=bool)
    sum_rewards = jp.zeros((args.num_envs,))

    train_start_time = time.time()
    (envs, agent, key, next_obs, next_done, sum_rewards), outputs = train_fn(
        envs, agent, key, next_obs, next_done, sum_rewards, x_start
    )
    next_obs.block_until_ready()
    training_time = time.time() - train_start_time
    total_steps = args.num_iterations * args.batch_size
    logger.info(f"Training for {total_steps} steps took {training_time:.2f} seconds.")

    p_losses, all_data = outputs if collect_stats else (None, None)

    if collect_stats:
        ep_step_counts = np.zeros(args.num_envs, dtype=int)
        ep_ever_succeeded = np.zeros(args.num_envs, dtype=bool)

        for iter_idx in range(args.num_iterations):
            reward_buf: list[float] = []
            success_buf: list[bool] = []
            distance_buf: list[float] = []
            ep_length_buf: list[int] = []
            global_step = iter_idx * args.batch_size
            data = jax.tree_util.tree_map(lambda x: x[iter_idx], all_data)

            for sum_reward, termination, truncation, success, distance in zip(
                data.sum_rewards, data.terminations, data.truncations, data.successes, data.distance
            ):
                ep_step_counts += 1
                ep_ever_succeeded |= np.asarray(success)
                next_done = termination | truncation
                if jp.any(next_done):
                    done_mask = np.asarray(next_done)
                    success_buf.extend(ep_ever_succeeded[done_mask])
                    distance_buf.extend(distance[done_mask])
                    reward_buf.extend(sum_reward[next_done])
                    ep_length_buf.extend(ep_step_counts[done_mask])
                    ep_step_counts[done_mask] = 0
                    ep_ever_succeeded[done_mask] = False

            if wandb_enabled:
                wandb.log(
                    {
                        "train/reward": np.mean(reward_buf) if reward_buf else np.nan,
                        "train/success": np.mean(success_buf) if success_buf else np.nan,
                        "train/ep_length": np.mean(ep_length_buf) if ep_length_buf else np.nan,
                        "train/distance": np.mean(distance_buf) if distance_buf else np.nan,
                        "losses/p_loss": jp.mean(p_losses[iter_idx]),
                        "losses/entropy": jp.mean(data.entropy),
                        "charts/SPS": int(global_step / training_time) if training_time > 0 else 0,
                    },
                    step=global_step + args.batch_size,
                )

    if model_path is not None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        params = {"actor": agent.actor_states.params}
        with open(model_path, "wb") as f:
            pickle.dump(params, f)
        logger.info(f"Model saved to {model_path}")

    return training_time, agent


# region Evaluate
def load_agent(args: Args, eval_env: ThrowEnv, model_path: Path) -> Agent:
    """Load an agent from a saved checkpoint."""
    agent = Agent.create(
        key=jax.random.PRNGKey(0),
        obs_dim=eval_env.single_observation_space.shape[0],
        act_dim=eval_env.single_action_space.shape[0],
        hidden_size=args.hidden_size,
    )
    with open(model_path, "rb") as f:
        params = pickle.load(f)
    return agent.replace(actor_states=agent.actor_states.replace(params=params["actor"]))


def evaluate(
    args: Args,
    n_eval: int,
    eval_env: ThrowEnv,
    agent: Agent,
    render: bool = False,
    x_start: dict | None = None,
) -> tuple[list, list, list, list]:
    """Evaluate a policy over n_eval episodes.

    Args:
        args: Training configuration, whose seed resets the environment.
        n_eval: Number of episodes to evaluate.
        eval_env: Environment to evaluate in.
        agent: Policy to evaluate.
        render: Whether to render during evaluation.
        x_start: Optional fixed initial state dict.

    Returns:
        Tuple of (episode_rewards, episode_lengths, episode_successes, episode_distances).
    """
    if x_start is not None:
        eval_env = _set_x_start(eval_env, x_start)

    episode_rewards = []
    episode_lengths = []
    episode_successes = []
    episode_distances = []
    sum_rewards = np.zeros(eval_env.unwrapped.num_envs)
    step_counts = np.zeros(sum_rewards.shape[0], dtype=int)
    ever_succeeded = np.zeros(sum_rewards.shape[0], dtype=bool)

    eval_env, (obs, _) = eval_env.reset(eval_env, seed=args.seed)
    i = 0
    while len(episode_rewards) < n_eval:
        action = agent.mean_action(agent.actor_states.params, obs)
        eval_env, (obs, reward, terminated, truncated, info) = eval_env.step(eval_env, action)
        if render and i % max(1, int(eval_env.unwrapped.freq / 60)) == 0:
            eval_env.render(world=0)
        reward = np.asarray(reward)
        terminated = np.asarray(terminated)
        truncated = np.asarray(truncated)
        success = np.asarray(info["success"])
        distances = np.asarray(info["distance"])
        sum_rewards += reward
        step_counts += 1
        ever_succeeded |= success
        for j in np.where(terminated | truncated)[0]:
            if len(episode_rewards) < n_eval:
                episode_rewards.append(float(sum_rewards[j]))
                episode_lengths.append(int(step_counts[j]))
                episode_successes.append(bool(ever_succeeded[j]))
                episode_distances.append(float(distances[j]))
            sum_rewards[j] = 0.0
            step_counts[j] = 0
            ever_succeeded[j] = False
        i += 1

    return episode_rewards, episode_lengths, episode_successes, episode_distances


# region Main
def main(
    wandb_enabled: bool = True,
    do_train: bool = True,
    n_eval: int = 64,
    render: bool = False,
    use_x_start: bool = False,
    steps: int | None = None,
    seed: int | None = None,
    n_runs: int = 1,
    wandb_group: str | None = None,
    save: bool = False,
    save_dir: str | None = None,
):
    """Entry point for single runs and multi-run groups.

    Flags:
      wandb_enabled: log metrics to wandb
      do_train: run training (set False to evaluate a saved model only)
      n_eval: number of evaluation episodes to run after training (0 to skip)
      render: whether to render the environment during evaluation
      use_x_start: condition episode starts on a fixed throw state
      steps: override total_timesteps
      seed: base RNG seed (each run draws a unique JAX seed from this)
      n_runs: number of independent training runs
      wandb_group: W&B group name (useful for comparing multiple runs)
      save_dir: directory for the checkpoint, defaults to the saves folder
    """
    base_seed = Args.seed if seed is None else seed
    rng = random.Random(base_seed)

    kwargs: dict[str, Any] = {} if steps is None else {"total_timesteps": steps}
    if wandb_group is not None:
        kwargs["wandb_group"] = wandb_group

    x_start = X_START if use_x_start else None

    # Build the template args and compile once for all runs
    template_args = Args(**kwargs, seed=base_seed)
    train_fn, train_envs, template_agent = setup_train(
        template_args,
        jax_device=template_args.jax_device,
        collect_stats=wandb_enabled,
        x_start=x_start,
    )
    eval_env = None
    if n_eval > 0:
        eval_env = make_jitted_envs(
            template_args,
            jax_device=template_args.jax_device,
            override_num_envs=n_eval,
            use_x_start=x_start is not None,
        )

    for run_idx in range(n_runs):
        run_seed = rng.randint(0, 2**31 - 1)
        args = Args(**kwargs, seed=run_seed)
        model_stem = args.exp_name if n_runs == 1 else f"{args.exp_name}_run{run_idx}"
        saves = MODEL_DIR if save_dir is None else Path(save_dir)
        model_path = saves / f"{model_stem}_model.ckpt" if save else None

        logger.info(f"Run {run_idx + 1}/{n_runs}: seed={run_seed}")

        agent = None
        if do_train:
            _, agent = train(
                args,
                train_fn,
                train_envs,
                template_agent,
                model_path,
                wandb_enabled=wandb_enabled,
                collect_stats=wandb_enabled,
                x_start=x_start,
            )

        if n_eval > 0:
            agent = agent if do_train else load_agent(args, eval_env, model_path)
            episode_rewards, episode_lengths, episode_successes, episode_distances = evaluate(
                args, n_eval, eval_env, agent, render=render, x_start=x_start
            )
            logger.info(
                f"Eval ({n_eval} eps): "
                f"success={np.mean(episode_successes):.1%}  "
                f"ep_length={np.mean(episode_lengths):.1f} +/- {np.std(episode_lengths):.1f} steps "
                f"distance={np.mean(episode_distances):.3f} +/- {np.std(episode_distances):.3f} m"
            )
            if wandb_enabled and wandb.run is not None:
                wandb.log(
                    {
                        "eval/mean_rewards": np.mean(episode_rewards),
                        "eval/mean_steps": np.mean(episode_lengths),
                        "eval/success": np.mean(episode_successes),
                        "eval/distance": np.mean(episode_distances),
                    }
                )

        if wandb_enabled and wandb.run is not None:
            wandb.finish()

    if eval_env is not None:
        eval_env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(main)
