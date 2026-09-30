"""Replay throw training and record how often each checkpoint recovers the real drone.

Every recorded throw is retrained with several seeds, so the success rate over the throws per seed
gives a spread across seeds, where a single seed only has the 0 or 1 of each throw.

Run with
    pixi run -e throw python crazyflow_experiments/throw/collect.py
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from dataclasses import dataclass, replace
from typing import Any, Callable

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("SCIPY_ARRAY_API", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from pathlib import Path

import fire
import jax
import jax.numpy as jp
import numpy as np
from crazyflow.control import Control
from crazyflow.dynamics import load_params
from crazyflow.sim import Dynamics, Sim

from crazyflow_experiments.core.agents.bptt_agent_deterministic import Agent
from crazyflow_experiments.core.envs.throw_env import create_action_space, thrust2rotor_vel
from crazyflow_experiments.throw import (
    CEILING_MIN_TIME,
    FLIGHT_BOUNDS_X,
    FLIGHT_BOUNDS_Y,
    FLIGHT_BOUNDS_Z,
    RECORDINGS_DIR,
    STABILIZED_MIN_TIME,
    STABILIZED_VEL_Z_TOL,
    SUCCESS_CURVES_PATH,
)
from crazyflow_experiments.throw.controller import SIM_ROTOR_VEL
from crazyflow_experiments.throw.train import (
    IDLE_ACTION,
    Args,
    flat_obs,
    make_jitted_envs,
    make_train_fns,
    reset_at_x_start,
)

logger = logging.getLogger(__name__)

THROWN_TO_TRAIN_DELAY = 0.1
EVAL_POLICY_FREQ = 500
# The real throw runs until it stabilizes or fails. A simulated one still flying after this long
# would be stopped by the pilot, so it fails
EVAL_HORIZON = 10.0  # s
# The onboard controller feeds back the action of its previous 500 Hz step, starting from the idle
# action the drone fell with
EVAL_ACTION_HISTORY_STRIDE = 1
EVAL_INITIAL_LAST_ACTION = IDLE_ACTION
_RPM2THRUST = jp.asarray(load_params(Dynamics.first_principles, "cf21B_500")["rpm2thrust"])


@dataclass
class ThrowTrace:
    """Recorded throw aligned to policy transfer, with full pre-transfer context."""

    file_path: Path
    t: np.ndarray
    pos: np.ndarray
    quat: np.ndarray
    failed: bool
    training_duration: float
    transfer_time: float
    takeover_state: dict[str, np.ndarray]


@dataclass
class ReplayArtifacts:
    """Reusable compiled objects for offline training replay."""

    args: Args
    train_env_template: Any
    success_scan: Any


def _obs_to_arrays(obs: list[dict]) -> dict[str, np.ndarray]:
    return {key: np.asarray([sample[key] for sample in obs]) for key in obs[0]}


def _first_state_index(states: np.ndarray, state_name: str) -> int | None:
    matches = np.flatnonzero(states == state_name)
    return int(matches[0]) if len(matches) else None


def _extract_throw_trace(file_path: Path) -> ThrowTrace:
    data = pickle.loads(file_path.read_bytes())
    obs = _obs_to_arrays(data["obs"])

    t = obs["t"]
    t = t - t[0]
    pos = obs["pos"]
    quat = obs["quat"]
    vel = obs["vel"]
    ang_vel = obs["ang_vel"]
    states = obs["state"]

    thrown_idx = _first_state_index(states, "thrown")
    trained_idx = _first_state_index(states, "trained")
    stabilizing_idx = _first_state_index(states, "stabilizing")
    if thrown_idx is None or trained_idx is None or stabilizing_idx is None:
        raise ValueError(f"{file_path} is missing expected state markers")
    if not thrown_idx < trained_idx < stabilizing_idx:
        raise ValueError(f"{file_path} has its state markers out of order")

    training_start_time = t[thrown_idx] + THROWN_TO_TRAIN_DELAY
    keep_aligned = t >= training_start_time
    states_aligned = states[keep_aligned]
    z_aligned = pos[keep_aligned, 2]
    stabilizing_mask = states_aligned == "stabilizing"
    if "success" in data:  # Recordings from before deploy.py stored the outcome lack the flag
        failed = not data["success"]
    else:
        crashed = z_aligned[stabilizing_mask] < FLIGHT_BOUNDS_Z[0]
        failed = bool(np.any(crashed)) if np.any(stabilizing_mask) else False

    transfer_time = float(t[stabilizing_idx] - t[trained_idx])
    # Use one sample before the first stabilizing marker to approximate the actual handover.
    takeover_idx = int(np.clip(stabilizing_idx, trained_idx, len(t) - 1))
    takeover_state = {
        "pos": pos[takeover_idx].copy(),
        "quat": quat[takeover_idx].copy(),
        "vel": vel[takeover_idx].copy(),
        "ang_vel": ang_vel[takeover_idx].copy(),
    }
    return ThrowTrace(
        file_path=file_path,
        t=t[keep_aligned] - t[stabilizing_idx],
        pos=pos[keep_aligned],
        quat=quat[keep_aligned],
        failed=failed,
        training_duration=max(t[trained_idx] - t[thrown_idx] - THROWN_TO_TRAIN_DELAY, 1e-6),
        transfer_time=transfer_time,
        takeover_state=takeover_state,
    )


def _load_throw_traces(recordings: Path) -> list[ThrowTrace]:
    throw_files = sorted(recordings.glob("x_throw_*.pkl"))
    if not throw_files:
        raise FileNotFoundError(f"No x_throw_*.pkl files found in {recordings}")
    return [_extract_throw_trace(file_path) for file_path in throw_files]


def _make_eval_sim(args: Args) -> tuple[Any, Any, jp.ndarray, jp.ndarray, int, float]:
    eval_sim = Sim(
        n_worlds=1,
        n_drones=1,
        drone="cf21B_500",
        dynamics=Dynamics.first_principles,
        control=Control.rotor_vel,
        device=args.jax_device,
        freq=500,
    )
    action_space = create_action_space(eval_sim.drone, control_type="rotor_thrust")
    action_low = jp.asarray(action_space.low, dtype=jp.float32)
    action_high = jp.asarray(action_space.high, dtype=jp.float32)
    action_scale = (action_high - action_low) / 2.0
    action_mean = (action_high + action_low) / 2.0
    n_substeps = eval_sim.freq // EVAL_POLICY_FREQ
    dt = 1.0 / EVAL_POLICY_FREQ
    return eval_sim.data, eval_sim.build_step_fn(), action_scale, action_mean, n_substeps, dt


def _inject_exact_state_into_sim_data(
    sim_data: Any, state: dict[str, np.ndarray | jp.ndarray]
) -> Any:
    pos = jp.asarray(state["pos"], dtype=sim_data.states.pos.dtype).reshape(
        sim_data.states.pos.shape
    )
    quat = jp.asarray(state["quat"], dtype=sim_data.states.quat.dtype).reshape(
        sim_data.states.quat.shape
    )
    vel = jp.asarray(state["vel"], dtype=sim_data.states.vel.dtype).reshape(
        sim_data.states.vel.shape
    )
    ang_vel = jp.asarray(state["ang_vel"], dtype=sim_data.states.ang_vel.dtype).reshape(
        sim_data.states.ang_vel.shape
    )
    rotor_vel = jp.full(
        sim_data.states.rotor_vel.shape, SIM_ROTOR_VEL, dtype=sim_data.states.rotor_vel.dtype
    )
    control_rotor_vel = jp.full(
        sim_data.controls.rotor_vel.shape, SIM_ROTOR_VEL, dtype=sim_data.controls.rotor_vel.dtype
    )
    return sim_data.replace(
        core=sim_data.core.replace(steps=jp.zeros_like(sim_data.core.steps), mjx_synced=False),
        states=sim_data.states.replace(
            pos=pos, quat=quat, vel=vel, ang_vel=ang_vel, rotor_vel=rotor_vel
        ),
        controls=sim_data.controls.replace(rotor_vel=control_rotor_vel),
    )


def _advance_eval_sim(
    sim_data: Any,
    raw_action: jp.ndarray,
    action_scale: jp.ndarray,
    action_mean: jp.ndarray,
    sim_step: Any,
    n_substeps: int,
) -> Any:
    thrust_cmd = jp.clip(raw_action, -1.0, 1.0) * action_scale + action_mean
    rotor_vel_cmd = thrust2rotor_vel(thrust_cmd, _RPM2THRUST).reshape(
        sim_data.controls.rotor_vel.shape
    )
    sim_data = sim_data.replace(controls=sim_data.controls.replace(rotor_vel=rotor_vel_cmd))
    return sim_step(sim_data, n_substeps)


def _out_of_bounds(sim_data: Any, elapsed: jp.ndarray) -> jp.ndarray:
    """Return whether the drone left the room's flight volume or went non-finite.

    The ceiling only counts CEILING_MIN_TIME after the takeover, as on the real drone, since the
    throw may still carry the drone upwards.
    """
    x, y, z = (sim_data.states.pos[:, 0, i] for i in range(3))
    outside_xy = (
        (x < FLIGHT_BOUNDS_X[0])
        | (x > FLIGHT_BOUNDS_X[1])
        | (y < FLIGHT_BOUNDS_Y[0])
        | (y > FLIGHT_BOUNDS_Y[1])
    )
    above_ceiling = (z > FLIGHT_BOUNDS_Z[1]) & (elapsed > CEILING_MIN_TIME)
    return (z < FLIGHT_BOUNDS_Z[0]) | above_ceiling | outside_xy | jp.isnan(z)


def make_deploy_evaluation(args: Args) -> Callable[[Agent, dict[str, jp.ndarray]], jp.ndarray]:
    """Return a function that flies an agent from a takeover state and scores it like deploy.py.

    The drone runs the policy at the onboard rate from the takeover state with idle rotors. It
    succeeds once it stabilizes and fails once it crashes, leaves the flight volume or is still
    flying at the horizon, whichever comes first.
    """
    eval_sim_data, eval_sim_step, eval_action_scale, eval_action_mean, eval_n_substeps, eval_dt = (
        _make_eval_sim(args)
    )
    max_eval_steps = int(EVAL_HORIZON / eval_dt) + 1

    def evaluate(agent: Agent, eval_state: dict[str, jp.ndarray]) -> jp.ndarray:
        sim_data = _inject_exact_state_into_sim_data(eval_sim_data, eval_state)
        init_action_history = jp.full(
            (1, EVAL_ACTION_HISTORY_STRIDE, eval_action_mean.shape[0]),
            EVAL_INITIAL_LAST_ACTION,
            dtype=jp.float32,
        )

        def cond_fn(
            carry: tuple[Any, jp.ndarray, jp.ndarray, jp.ndarray, jp.ndarray],
        ) -> jp.ndarray:
            _, _, _, done, step = carry
            return jp.logical_and(~done, step < max_eval_steps)

        def body_fn(
            carry: tuple[Any, jp.ndarray, jp.ndarray, jp.ndarray, jp.ndarray],
        ) -> tuple[Any, ...]:
            sim_data, action_history, ever_succeeded, _, step = carry
            last_actions = action_history[
                :, EVAL_ACTION_HISTORY_STRIDE - 1 :: EVAL_ACTION_HISTORY_STRIDE, :
            ]
            obs = flat_obs(
                sim_data.states.pos[:, 0, :],
                sim_data.states.quat[:, 0, :],
                sim_data.states.vel[:, 0, :],
                sim_data.states.ang_vel[:, 0, :],
                last_actions.reshape(last_actions.shape[0], -1),
            )
            action = agent.mean_action(agent.actor_states.params, obs)
            sim_data = _advance_eval_sim(
                sim_data,
                action,
                eval_action_scale,
                eval_action_mean,
                eval_sim_step,
                eval_n_substeps,
            )
            action_history = jp.roll(action_history, shift=1, axis=1).at[:, 0, :].set(action)
            elapsed = (step + 1).astype(jp.float32) * eval_dt
            # Same outcome as deploy.py: failing first loses, stabilizing wins
            vel_z = sim_data.states.vel[0, 0, 2]
            stabilized = jp.logical_and(
                jp.abs(vel_z) < STABILIZED_VEL_Z_TOL, elapsed >= STABILIZED_MIN_TIME
            )
            failed = _out_of_bounds(sim_data, elapsed)[0] | ~jp.all(jp.isfinite(action))
            succeeded = ~failed & stabilized
            done = failed | stabilized
            return sim_data, action_history, succeeded, done, step + 1

        init_carry = (sim_data, init_action_history, jp.array(False), jp.array(False), jp.array(0))
        final_carry = jax.lax.while_loop(cond_fn, body_fn, init_carry)
        return final_carry[2]

    return evaluate


def _make_replay_artifacts(args: Args, use_x_start: bool) -> ReplayArtifacts:
    train_env_template = make_jitted_envs(args, jax_device=args.jax_device, use_x_start=use_x_start)
    train_iteration, _train_fn = make_train_fns(args, collect_stats=False)
    evaluate = make_deploy_evaluation(args)

    @jax.jit
    def success_scan(
        carry: tuple[Any, ...], eval_state: dict[str, jp.ndarray]
    ) -> tuple[Any, jp.ndarray]:
        def scan_body(
            train_carry: tuple[Any, ...], update_idx: jp.ndarray
        ) -> tuple[tuple[Any, ...], jp.ndarray]:
            train_carry, _loss = train_iteration(train_carry, update_idx)
            success = evaluate(train_carry[1], eval_state)
            return train_carry, success

        return jax.lax.scan(scan_body, carry, jp.arange(args.num_iterations))

    return ReplayArtifacts(
        args=args, train_env_template=train_env_template, success_scan=success_scan
    )


def _initial_training_carry(
    replay: ReplayArtifacts, x_start: dict[str, np.ndarray] | None
) -> tuple[Any, ...]:
    args = replay.args
    train_envs = replay.train_env_template
    key = jax.random.PRNGKey(args.seed)
    init_key, key = jax.random.split(key)
    agent = Agent.create(
        key=init_key,
        obs_dim=train_envs.single_observation_space.shape[0],
        act_dim=train_envs.single_action_space.shape[0],
        hidden_size=args.hidden_size,
        actor_lr=args.actor_lr,
    )
    if x_start is None:
        train_envs, (next_obs, _info) = train_envs.reset(train_envs, seed=args.seed)
    else:  # The same compiled reset as ThrowController, so the replay trains the flown policy
        x_start = {name: jp.asarray(value) for name, value in x_start.items()}
        train_envs, next_obs = reset_at_x_start(train_envs, x_start, seed=args.seed)
    next_done = jp.zeros(args.num_envs, dtype=bool)
    sum_rewards = jp.zeros((args.num_envs,))
    return train_envs, agent, key, next_obs, next_done, sum_rewards


def _compute_success_curves(
    trace: ThrowTrace, replays: dict[bool, ReplayArtifacts], args: Args, seeds: list[int]
) -> np.ndarray:
    """Replay the training of a recorded throw once per seed, in the mode it flew.

    Returns the success of every checkpoint, one row per seed. The recording's x_start is None
    when the policy trained on random throws. Recordings that hold the flown policy must match the
    replay with the seed of Args, otherwise the curves would belong to another policy.
    """
    recording = pickle.loads(trace.file_path.read_bytes())
    use_x_start = recording["x_start"] is not None
    if use_x_start not in replays:
        replays[use_x_start] = _make_replay_artifacts(args, use_x_start)
    replay = replays[use_x_start]
    eval_state = {key: jp.asarray(value) for key, value in trace.takeover_state.items()}
    curves = []
    for seed in seeds:
        seeded = replace(replay, args=replace(args, seed=seed))
        carry = _initial_training_carry(seeded, recording["x_start"])
        (_, agent, *_), successes = replay.success_scan(carry, eval_state)
        if seed == args.seed and recording.get("policy") is not None:
            flown = jax.tree.leaves(recording["policy"])
            replayed = jax.tree.leaves(agent.actor_states.params)
            if not all(
                np.array_equal(a, np.asarray(b)) for a, b in zip(flown, replayed, strict=True)
            ):
                raise ValueError(f"Replayed policy of {trace.file_path} differs from the flown one")
        curves.append(np.asarray(successes))
    return np.stack(curves)


def main(
    n_seeds: int = 10, recordings_dir: str | None = None, curves_path: str | None = None
) -> None:
    """Replay every recorded throw with n_seeds seeds and write the success curves next to them.

    The seeds count up from the seed of Args, so the first one is the seed that flew.
    """
    recordings = RECORDINGS_DIR if recordings_dir is None else Path(recordings_dir)
    curves = recordings / SUCCESS_CURVES_PATH.name if curves_path is None else Path(curves_path)
    traces = _load_throw_traces(recordings)

    args = Args()
    seeds = [args.seed + i for i in range(n_seeds)]
    replays: dict[bool, ReplayArtifacts] = {}  # Built once per training mode among the recordings
    successes = [_compute_success_curves(trace, replays, args, seeds) for trace in traces]

    success_curves = {
        "successes": np.stack(successes).tolist(),  # Indexed [throw][seed][checkpoint]
        "seeds": seeds,
        "traces": [
            {
                "t": trace.t.tolist(),
                "pos_z": trace.pos[:, 2].tolist(),
                "quat": trace.quat.tolist(),
                "training_duration": float(trace.training_duration),
                "transfer_time": float(trace.transfer_time),
                "failed": trace.failed,
            }
            for trace in traces
        ],
    }
    curves.parent.mkdir(parents=True, exist_ok=True)
    curves.write_text(json.dumps(success_curves))
    logger.info(f"Wrote {len(traces)} x {n_seeds} success curves to {curves}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(main)
