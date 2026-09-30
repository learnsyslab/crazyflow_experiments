"""Train a vision based reach policy by imitating an analytic expert.

Run with
    pixi run -e vision python crazyflow_experiments/vision/train.py
"""

import os

os.environ["SCIPY_ARRAY_API"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import functools
import json
import logging
from dataclasses import asdict, dataclass
from typing import Any, Callable, NamedTuple

import fire
import flax.struct as struct
import jax
import jax.numpy as jp
import numpy as np
import wandb
from flax import serialization
from jax import Array
from scipy.spatial.transform import Rotation as R

from crazyflow_experiments.core.agents.vision_agent import Agent, VisionPolicyNet
from crazyflow_experiments.core.envs import domain_randomization as dr
from crazyflow_experiments.core.envs.augmentation import CameraNoise
from crazyflow_experiments.core.envs.vision_reach_env import VisionReachEnv
from crazyflow_experiments.core.envs.wrappers import ActionDelay, ActionHistory, StackFrames
from crazyflow_experiments.utils.paths import RESULTS_DIR
from crazyflow_experiments.vision import MODEL_DIR
from crazyflow_experiments.vision.expert import build_expert, smooth_actions
from crazyflow_experiments.vision.scenes import splat

logger = logging.getLogger(__name__)


# region Arguments
@dataclass(frozen=True)
class Args:
    """Configuration of a training run."""

    seed: int = 0
    """seed of the experiment"""
    exp_name: str = "base"
    """checkpoint stem"""
    wandb_enabled: bool = True
    """log to wandb"""
    wandb_project: str = "vision-reach"
    """wandb project"""
    run_name: str | None = None
    """wandb run name"""
    wandb_group: str | None = None
    """wandb group"""

    # Environment settings
    scene: str = "lab"
    """scene to render"""
    variant: str = "lab_30k"
    """splat variant to render"""
    control: str = "attitude"
    """control interface, either state or attitude"""
    resolution: tuple[int, int] = (64, 64)
    """policy image resolution"""
    stack: int = 3
    """frames per observation"""
    stride: int = 4
    """steps between stacked frames"""
    command_history: int = 1
    """commands shown, newest first"""
    episode_time: float = 10.0
    """episode cutoff in s"""
    bounds_low: tuple[float, float, float] = (-1.5, -1.2, 0.1)
    """lower corner of the flyable space, beyond which an episode ends"""
    bounds_high: tuple[float, float, float] = (2.0, 1.4, 2.3)
    """upper corner of the flyable space"""
    goal_low: tuple[float, float, float] = (-1.3, -1.0, 0.8)
    """lower corner of the box goals are drawn from"""
    goal_high: tuple[float, float, float] = (1.8, 1.2, 1.6)
    """upper corner of the goal box"""
    spawn_low: tuple[float, float, float] = (-1.3, -1.0, 0.2)
    """lower corner of the box the drone spawns in"""
    spawn_high: tuple[float, float, float] = (1.8, 1.2, 2.2)
    """upper corner of the goal box"""
    goal_radius: float = 0.20
    """goal radius in m"""
    rest_speed: float = 0.1
    """settled speed in m/s"""
    yaw_fade: float = 0.2
    """distance in m over which the expert eases off its turn near the goal, zero to cut it"""
    expert_brake_gain: float = 0.2
    """expert approach speed per m of distance left, which sets how early it starts braking"""
    expert_speed: float = 0.35
    """largest approach speed in m/s the expert asks for"""
    expert_step_scale: float = 0.6
    """fraction of the full position setpoint offset the expert commands, which caps its speed"""
    use_imu: bool = True
    """show the inertial channels"""

    # Randomization settings
    randomize_imu: bool = True
    """drift and bias the inertial readings"""
    randomize_photometry: bool = True
    """vary brightness, contrast, gamma, vignette and exposure"""
    randomize_pixel: bool = True
    """add pixel noise and brightness ramps to the frames"""
    randomize_sharpen: bool = True
    """vary the fine detail"""
    splat_variants: tuple[str, ...] = ()
    """additional reconstructions of the scene"""
    randomize_shutter: bool = True
    """shear the frames with the rolling shutter"""
    randomize_drop: bool = True
    """randomly drop frames"""
    randomize_latency: bool = True
    """delay the frames and the commands"""
    randomize_wind: bool = True
    """push the drone with wind and gusts"""
    randomize_mount: bool = True
    """offset and tilt the camera on its mount"""
    randomize_inertial: bool = True
    """draw the mass and the inertia"""
    mass_range: float = 0.05
    """fraction the mass is drawn over"""
    frame_drop: float = 0.05
    """dropped frame chance"""
    camera_latency: tuple[int, int] = (1, 3)
    """steps the camera frame lags, drawn per episode"""
    action_latency: tuple[int, int] = (0, 1)
    """dropped frame chance"""

    # Collection settings
    num_envs: int = 1024
    """parallel worlds"""
    rollout_steps: int = 100
    """steps per world per epoch"""
    epochs: int = 500
    """epochs of collection and fitting"""
    expert_fraction: float = 0.1
    """worlds the expert drives"""
    capacity: int = 2**20
    """buffer capacity in samples"""

    # Fitting settings
    batch_size: int = 256
    """batch size"""
    updates_per_epoch: int = 1000
    """gradient steps per epoch"""
    learning_rate: float = 3e-4
    """learning rate"""
    max_grad_norm: float = 1.0
    """gradient clip norm"""
    aux_weight: float = 1.0
    """weight on the state prediction"""
    features: tuple[int, ...] = (48, 96, 192, 192)
    """encoder channels"""
    embed_size: int = 384
    """embedding width"""
    hidden_size: int = 384
    """trunk width"""

    # Evaluation settings
    eval_envs: int = 1024
    """episodes per evaluation"""
    eval_steps: int = 500
    """steps per evaluation rollout"""
    eval_every: int = 25
    """epochs between evaluations"""


# region Buffer
@struct.dataclass
class Buffer:
    """Ring buffer of policy inputs and expert actions."""

    obs: dict[str, Array]
    target: Array
    state: Array
    index: Array
    size: Array


def pack(key: str, value: Array) -> Array:
    """Compress a policy input for storage."""
    if key == "frames":
        return jp.round(value * 255).astype(jp.uint8)
    return value.astype(jp.float16)


def unpack(key: str, value: Array) -> Array:
    """Restore a stored policy input to the range the network expects."""
    if key == "frames":
        return value.astype(jp.float32) / 255.0
    return value.astype(jp.float32)


def create_buffer(obs: dict[str, Array], capacity: int, act_dim: int) -> Buffer:
    """Allocate a buffer matching the observation shapes, holding inputs compressed."""
    return Buffer(
        obs={k: jp.zeros((capacity, *v.shape[1:]), dtype=pack(k, v).dtype) for k, v in obs.items()},
        target=jp.zeros((capacity, act_dim)),
        state=jp.zeros((capacity, 18), dtype=jp.float16),
        index=jp.zeros((), dtype=jp.int32),
        size=jp.zeros((), dtype=jp.int32),
    )


# The buffer is donated, so the write lands in the memory the old one occupied.
@functools.partial(jax.jit, donate_argnums=0)
def insert(buffer: Buffer, obs: dict[str, Array], target: Array, state: Array) -> Buffer:
    """Write a flat batch of compressed samples into the buffer, overwriting the oldest."""
    capacity, count = buffer.target.shape[0], target.shape[0]
    idx = (buffer.index + jp.arange(count)) % capacity
    return buffer.replace(
        obs={k: v.at[idx].set(obs[k]) for k, v in buffer.obs.items()},
        target=buffer.target.at[idx].set(target),
        state=buffer.state.at[idx].set(state.astype(jp.float16)),
        index=(buffer.index + count) % capacity,
        size=jp.minimum(buffer.size + count, capacity),
    )


def sample(buffer: Buffer, key: Array, batch_size: int) -> tuple[dict[str, Array], Array, Array]:
    """Draw a minibatch from the filled region of the buffer."""
    idx = jax.random.randint(key, (batch_size,), 0, buffer.size)
    obs = {k: unpack(k, v[idx]) for k, v in buffer.obs.items()}
    return obs, buffer.target[idx], buffer.state[idx].astype(jp.float32)


@functools.partial(jax.jit, static_argnames=("updates", "batch_size"))
def train_epoch(
    agent: Agent, buffer: Buffer, key: Array, updates: int, batch_size: int
) -> tuple[Agent, dict[str, Array]]:
    """Take a sequence of gradient steps on minibatches drawn from the buffer."""

    def step(agent: Agent, key: Array) -> tuple[Agent, dict[str, Array]]:
        return agent.update(*sample(buffer, key, batch_size))

    return jax.lax.scan(step, agent, jax.random.split(key, updates))


# region Rollouts
class Step(NamedTuple):
    """What one environment step reports back to the caller of _advance."""

    obs: dict[str, Array]
    policy_obs: dict[str, Array]
    expert: Array
    state: Array  # pos, rot vel, ang_vel, the auxiliary target
    error: Array
    success: Array
    terminated: Array
    truncated: Array


def _advance(
    env: ActionHistory,
    agent: Agent,
    expert: Callable[..., Array],
    obs: dict[str, Array],
    expert_driven: Array,
    encode: Callable[[dict[str, Array]], dict[str, Array]],
    previous: Array,
    fresh: Array,
) -> tuple[ActionHistory, Step]:
    """Step the environment once, returning the policy input and both actions at that state."""
    data = env.unwrapped.data
    goal = env.unwrapped.goal_pos
    label = smooth_actions(expert(data, goal), previous, fresh)
    rot = R.from_quat(data.states.quat[:, 0, :]).as_matrix()
    state = jp.concatenate(
        [
            data.states.pos[:, 0, :],
            rot.reshape(rot.shape[0], -1),
            data.states.vel[:, 0, :],
            data.states.ang_vel[:, 0, :],
        ],
        axis=-1,
    )
    policy_obs = encode(obs)
    student = agent.action(agent, policy_obs)
    error = jp.mean(jp.abs(student - label), axis=-1)
    action = jp.where(expert_driven[:, None], label, student)
    env, (next_obs, _, terminated, truncated, info) = env.step(env, action)
    step = Step(next_obs, policy_obs, label, state, error, info["success"], terminated, truncated)
    return env, step


def collect(
    env: ActionHistory,
    agent: Agent,
    expert: Callable[..., Array],
    obs: dict[str, Array],
    expert_driven: Array,
    steps: int,
    encode: Callable[[dict[str, Array]], dict[str, Array]],
) -> tuple[ActionHistory, dict[str, Array], tuple[dict[str, Array], Array, Array]]:
    """Roll out and record the expert action at every visited state."""

    def step(
        carry: tuple[ActionHistory, dict[str, Array], Array, Array], _: None
    ) -> tuple[tuple[ActionHistory, dict[str, Array], Array, Array], tuple]:
        env, obs, previous, fresh = carry
        env, out = _advance(env, agent, expert, obs, expert_driven, encode, previous, fresh)
        done = out.terminated | out.truncated
        stored = {k: pack(k, v) for k, v in out.policy_obs.items()}  # Compress here to save memory
        return (env, out.obs, out.expert, done), (stored, out.expert, out.state)

    actions = jp.zeros((expert_driven.shape[0], env.single_action_space.shape[0]))
    start = (env, obs, actions, jp.ones_like(expert_driven))
    (env, obs, _, _), records = jax.lax.scan(step, start, xs=None, length=steps)
    return env, obs, records


def flatten(records: tuple) -> tuple[dict[str, Array], Array, Array]:
    """Merge the time and world axes of a recorded rollout into a flat batch."""
    obs, expert, state = records
    obs = {k: v.reshape(-1, *v.shape[2:]) for k, v in obs.items()}
    return obs, expert.reshape(-1, expert.shape[-1]), state.reshape(-1, state.shape[-1])


# region Evaluation
class Rollout(NamedTuple):
    """Per step record of an evaluation rollout, with a leading time and world axis."""

    error: Array  # Mean absolute deviation from the expert action
    success: Array
    terminated: Array
    truncated: Array
    distance: Array  # To the goal
    speed: Array


def evaluate(
    env: ActionHistory,
    agent: Agent,
    expert: Callable[..., Array],
    expert_driven: Array,
    steps: int,
    encode: Callable[[dict[str, Array]], dict[str, Array]],
    seed: int,
) -> Rollout:
    """Roll out and score the policy."""

    def step(
        carry: tuple[ActionHistory, dict[str, Array], Array, Array], _: None
    ) -> tuple[tuple[ActionHistory, dict[str, Array], Array, Array], Rollout]:
        env, obs, previous, fresh = carry
        # Read the state before the step, since a world that finishes is reset within it
        pos = env.unwrapped.data.states.pos[:, 0, :]
        distance = jp.linalg.norm(pos - env.unwrapped.goal_pos, axis=-1)
        speed = jp.linalg.norm(env.unwrapped.data.states.vel[:, 0, :], axis=-1)
        env, out = _advance(env, agent, expert, obs, expert_driven, encode, previous, fresh)
        done = out.terminated | out.truncated
        record = Rollout(out.error, out.success, out.terminated, out.truncated, distance, speed)
        return (env, out.obs, out.expert, done), record

    env, (obs, _) = env.reset(env, seed=seed)
    actions = jp.zeros((expert_driven.shape[0], env.single_action_space.shape[0]))
    start = (env, obs, actions, jp.ones_like(expert_driven))
    _, records = jax.lax.scan(step, start, xs=None, length=steps)
    return records


def score(rollout: Rollout, goal_radius: float) -> dict[str, float]:
    """Reduce a rollout to the first episode of every world and say how each of them ended."""
    error, success, terminated, truncated, distance, speed = (
        np.asarray(field) for field in rollout
    )
    done = terminated | truncated
    worlds = np.arange(done.shape[1])
    finished = done.any(axis=0)
    end = np.argmax(done, axis=0)  # First step each world finishes in
    won = success[end, worlds] & finished
    crashed = terminated[end, worlds] & finished & ~won
    # The state one step before the end, the last one this episode was measured in
    parked, drift = distance[end, worlds], speed[end, worlds]
    # Whether the goal region was ever entered, so a miss separates from a failure to settle
    steps = np.arange(done.shape[0])[:, None]
    inside = ((distance < goal_radius) & (steps <= end)).any(axis=0) | won
    return {
        "success": float(won.mean()),
        "crashed": float(crashed.mean()),
        "timeout": float((finished & ~terminated[end, worlds]).mean()),
        "unfinished": float((~finished).mean()),
        "reached goal region": float(inside.mean()),
        "missed after reaching": float((inside & ~won).mean()),
        "final distance": float(parked.mean()),
        "median final distance": float(np.median(parked)),
        "final speed": float(drift.mean()),
        "steps to success": float(np.median(end[won])) if won.any() else float("nan"),
        "label error": float(error.mean()),
    }


# region Builders
def build_randomization(
    args: Args,
) -> tuple[dr.Ranges, CameraNoise, float, tuple[int, int], tuple[int, int]]:
    """Spread of every randomized quantity, with the ones a run asks for switched on."""
    spread, effects = dr.Ranges(), CameraNoise()
    ranges = dr.Ranges(
        gyro_bias=spread.gyro_bias if args.randomize_imu else 0.0,
        gyro_noise=spread.gyro_noise if args.randomize_imu else (0.0, 0.0, 0.0),
        accel_bias=spread.accel_bias if args.randomize_imu else 0.0,
        accel_noise=spread.accel_noise if args.randomize_imu else (0.0, 0.0, 0.0),
        gyro_drift=spread.gyro_drift if args.randomize_imu else 0.0,
        accel_drift=spread.accel_drift if args.randomize_imu else 0.0,
        wind_max=spread.wind_max if args.randomize_wind else 0.0,
        gust_std=spread.gust_std if args.randomize_wind else 0.0,
        torque_std=spread.torque_std if args.randomize_wind else 0.0,
        cam_offset_std=spread.cam_offset_std if args.randomize_mount else 0.0,
        cam_angle_std=spread.cam_angle_std if args.randomize_mount else 0.0,
        cam_fovy_range=spread.cam_fovy_range if args.randomize_mount else 0.0,
        brightness_std=spread.brightness_std if args.randomize_photometry else 0.0,
        contrast_std=spread.contrast_std if args.randomize_photometry else 0.0,
        gamma_std=spread.gamma_std if args.randomize_photometry else 0.0,
        vignette_max=spread.vignette_max if args.randomize_photometry else 0.0,
        exposure_max=spread.exposure_max if args.randomize_photometry else 0.0,
        exposure_min=spread.exposure_min,
        mass_range=args.mass_range if args.randomize_inertial else 0.0,
        inertia_range=spread.inertia_range if args.randomize_inertial else 0.0,
    )
    noise = CameraNoise(
        sharpen=effects.sharpen if args.randomize_sharpen else 0.0,
        pixel_noise=effects.pixel_noise if args.randomize_pixel else 0.0,
        ramp_std=effects.ramp_std if args.randomize_pixel else 0.0,
        readout_time=effects.readout_time if args.randomize_shutter else 0.0,
    )
    drop = args.frame_drop if args.randomize_drop else 0.0
    camera = args.camera_latency if args.randomize_latency else (0, 0)
    action = args.action_latency if args.randomize_latency else (0, 0)
    return ranges, noise, drop, camera, action


def build_env(args: Args, n_envs: int) -> ActionHistory:
    """Build the reach environment for a number of worlds, with the policy's inputs assembled."""
    ranges, noise, drop, camera, action = build_randomization(args)
    env = VisionReachEnv.create(
        scene=splat(args.scene, args.variant),
        variants=[splat(args.scene, n) for n in args.splat_variants],
        n_envs=n_envs,
        max_episode_time=args.episode_time,
        control=args.control,
        ranges=ranges,
        noise=noise,
        goal_radius=args.goal_radius,
        rest_speed=args.rest_speed,
        resolution=args.resolution,
        bounds_low=args.bounds_low,
        bounds_high=args.bounds_high,
        goal_low=args.goal_low,
        goal_high=args.goal_high,
        spawn_low=args.spawn_low,
        spawn_high=args.spawn_high,
    )
    env = ActionDelay.create(env, latency=action)
    env = StackFrames.create(env, stack=args.stack, stride=args.stride, drop=drop, latency=camera)
    return ActionHistory.create(env, length=args.command_history)


def build_policy(args: Args) -> VisionPolicyNet:
    """Build the policy for the configured architecture."""
    return VisionPolicyNet(
        features=args.features, embed_size=args.embed_size, hidden_size=args.hidden_size
    )


def blind_imu(obs: dict[str, Array], use_imu: bool) -> dict[str, Array]:
    """Zero the inertial channels, leaving their shape so only their information is removed."""
    if use_imu:
        return obs
    return obs | {"ang_vel": jp.zeros_like(obs["ang_vel"]), "acc": jp.zeros_like(obs["acc"])}


# region Training
def main(**kwargs: Any) -> list[dict[str, float]]:
    """Train the policy with expert labeled data and report both metrics."""
    args = Args(**kwargs)
    key = jax.random.key(args.seed)
    env = build_env(args, args.num_envs)
    eval_env = build_env(args, args.eval_envs)
    env, (obs, _) = env.reset(env, seed=args.seed)

    expert = build_expert(
        args.control,
        args.yaw_fade,
        args.expert_brake_gain,
        args.expert_step_scale,
        args.expert_speed,
    )
    encode = functools.partial(blind_imu, use_imu=args.use_imu)
    policy_obs = encode(obs)
    key, agent_key = jax.random.split(key)
    agent = Agent.create(
        agent_key,
        policy_obs,
        build_policy(args),
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        aux_weight=args.aux_weight,
        decay_steps=args.epochs * args.updates_per_epoch,
    )
    buffer = create_buffer(policy_obs, args.capacity, env.single_action_space.shape[0])
    config = asdict(args)
    if args.wandb_enabled:
        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            group=args.wandb_group,
            config=config,
            dir=RESULTS_DIR,
        )
    collect_jit = jax.jit(collect, static_argnames=("steps", "encode", "expert"))
    evaluate_jit = jax.jit(evaluate, static_argnames=("steps", "encode", "expert", "seed"))
    draw_scene = jax.jit(env.unwrapped.draw_scene)
    history = []
    best = (-1.0, -1, agent.policy_state.params)  # Success, epoch and the parameters behind it
    stem = args.run_name or args.exp_name
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    progress = MODEL_DIR / f"reach_policy_{stem}.progress.json"

    for epoch_index in range(args.epochs):
        key, splat_key = jax.random.split(key)
        env = draw_scene(env, splat_key)
        # First epoch is plain behavior cloning. Afterwards the student acts and the expert labels
        # the states, with a few worlds kept on the expert's own distribution.
        key, drive_key = jax.random.split(key)
        fraction = 1.0 if epoch_index == 0 else args.expert_fraction
        expert_driven = jax.random.uniform(drive_key, (args.num_envs,)) < fraction
        env, obs, records = collect_jit(
            env, agent, expert, obs, expert_driven, args.rollout_steps, encode
        )
        buffer = insert(buffer, *flatten(records))

        key, batch_key = jax.random.split(key)
        agent, steps = train_epoch(
            agent, buffer, batch_key, args.updates_per_epoch, args.batch_size
        )
        # Averaged over the last tenth of the epoch, so the reading is not dragged by the start
        recent = {k: float(jp.mean(v[-args.updates_per_epoch // 10 :])) for k, v in steps.items()}
        train_loss = recent["loss"]
        angle = np.degrees(2 * np.arcsin(np.clip(recent["rot"] / (2 * np.sqrt(2)), 0, 1)))
        train = {
            "train/loss": train_loss,
            "train/position error (cm)": recent["pos"] * 100,
            "train/rotation error (deg)": float(angle),
            "train/velocity error (cm/s)": recent["vel"] * 100,
            "train/angular velocity error (deg/s)": np.degrees(recent["ang_vel"]),
        }

        due = args.eval_every and epoch_index % args.eval_every == 0
        if not due and epoch_index != args.epochs - 1:
            logger.info(f"epoch {epoch_index:3d}  loss {train_loss:.4f}")
            if args.wandb_enabled:
                wandb.log(train, step=epoch_index)
            continue

        # Open loop error on expert states
        expert_only = jp.ones((args.eval_envs,), dtype=bool)
        expert_rollout = evaluate_jit(
            eval_env, agent, expert, expert_only, args.eval_steps, encode, args.seed
        )
        action_error = float(jp.mean(expert_rollout.error)) * 100  # As fraction of the action range

        # Closed loop performance with the student flying alone
        student_only = jp.zeros((args.eval_envs,), dtype=bool)
        student_rollout = evaluate_jit(
            eval_env, agent, expert, student_only, args.eval_steps, encode, args.seed
        )
        outcome = score(student_rollout, args.goal_radius)
        metrics = train | {
            "eval/action error (%)": action_error,
            "eval/success": outcome["success"],
            "eval/crashed": outcome["crashed"],
            "eval/timeout": outcome["timeout"],
            "eval/reached goal region": outcome["reached goal region"],
            "eval/missed after reaching": outcome["missed after reaching"],
            "eval/final distance": outcome["final distance"],
            "eval/final speed": outcome["final speed"],
            "eval/steps to success": outcome["steps to success"],
        }
        if outcome["success"] > best[0]:
            best = (outcome["success"], epoch_index, agent.policy_state.params)
        history.append({"epoch": epoch_index} | metrics)
        progress.write_text(json.dumps(history))
        if args.wandb_enabled:
            wandb.log(metrics, step=epoch_index)
        logger.info(
            f"epoch {epoch_index:3d}  loss {train_loss:.4f}  action error {action_error:.2f}%"
            f"  success {outcome['success']:.2%}  reached {outcome['reached goal region']:.2%}"
            f"  crashed {outcome['crashed']:.2%}  distance {outcome['final distance']:.3f} m"
            f"  pose {train['train/position error (cm)']:.1f} cm "
            f"{train['train/rotation error (deg)']:.1f} deg"
        )

    path = MODEL_DIR / f"reach_policy_{stem}.msgpack"
    # Save the config with the params to recreate the network etc.
    checkpoint = {"params": best[2], "config": config}
    path.write_bytes(serialization.to_bytes(checkpoint))
    logger.info(f"wrote {path}, best {best[0]:.3f} at epoch {best[1]}")
    if args.wandb_enabled:
        artifact = wandb.Artifact(f"reach_policy_{stem}", type="model", metadata=config)
        artifact.add_file(str(path))
        wandb.log_artifact(artifact)
        wandb.finish()
    return history


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    fire.Fire(main)
