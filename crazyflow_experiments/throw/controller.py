"""Rotor-velocity controller that recovers from being thrown, trained on-the-fly from x_throw.

Before the drone is thrown, call warmup_jit() to pre-compile all JAX functions. After the throw,
call train(x_throw) to predict the takeover state and train the policy on the resulting
x_start state. The entire training pipeline runs in compiled JAX.
"""

from __future__ import annotations

import logging
import struct
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jp
import numpy as np
from cflib.crtp.crtpstack import CRTPPacket
from crazyflow.control import Control
from crazyflow.dynamics import load_params
from crazyflow.sim import Dynamics, Sim
from crazyflow.utils import leaf_replace

from crazyflow_experiments.control.controller import Controller
from crazyflow_experiments.core.agents.bptt_agent_deterministic import Agent
from crazyflow_experiments.core.envs.throw_env import thrust2rotor_vel
from crazyflow_experiments.throw.train import (
    IDLE_ACTION,
    Args,
    flat_obs,
    make_jitted_envs,
    make_train_fns,
    reset_at_x_start,
)

if TYPE_CHECKING:
    import cflib.crazyflie
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Keep the compiled training on disk, so later starts load it instead of compiling it again. Every
# compilation is cached, the short ones add up to most of the startup
jax.config.update("jax_compilation_cache_dir", str(Path.home() / ".cache/crazyflow_jax_cache"))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)

# CRTP param port constants
PARAM_PORT = 0x02
PARAM_WRITE_CH = 2  # Channel for parameter writes

# Max CRTP payload is ~29 bytes. Each float16 param = 2 bytes ID + 2 bytes value = 4 bytes
MAX_PARAMS_PER_PACKET = 7

# Simulation constants for the ballistic prediction
SIM_FREQ = 500  # Simulation frequency (Hz)
# The drone falls with its rotors idling at the minimum thrust, the idle action
_PARAMS = load_params(Dynamics.first_principles, "cf21B_500")
SIM_ROTOR_VEL = float(thrust2rotor_vel(_PARAMS["thrust_min"], _PARAMS["rpm2thrust"]))  # rpm
IDLE_NORMALIZED_FORCE = float(_PARAMS["thrust_min"] / _PARAMS["thrust_max"])
TRAINING_LATENCY = 0.5  # seconds from throw detection to policy activation


def _get_param_id(cf: cflib.crazyflie.Crazyflie, full_name: str) -> int:
    """Get the numeric parameter ID from the Crazyflie's parameter TOC.

    Args:
        cf: Connected Crazyflie with TOC downloaded.
        full_name: Full parameter name in "group.name" format.

    Returns:
        The 16-bit parameter ID.

    Raises:
        KeyError: If parameter not found in TOC.
    """
    group, name = full_name.split(".")
    toc = cf.param.toc.toc
    if group not in toc or name not in toc[group]:
        raise KeyError(f"Parameter '{full_name}' not found in TOC")
    element = toc[group][name]
    return element.ident


def _send_params(cf: cflib.crazyflie.Crazyflie, param_ids: list[int], values: list[float]) -> None:
    """Send batched float16 parameters to the Crazyflie.

    Args:
        cf: Connected Crazyflie.
        param_ids: Pre-computed list of 16-bit parameter IDs (from precompute_param_ids).
        values: List of float values, same length as param_ids. Will be converted to float16.
    """
    # Send in batches
    for i in range(0, len(param_ids), MAX_PARAMS_PER_PACKET):
        batch_ids = param_ids[i : i + MAX_PARAMS_PER_PACKET]
        batch_vals = values[i : i + MAX_PARAMS_PER_PACKET]

        # Build packet data: [id (2 bytes)][value (2 bytes)]...
        data = bytearray()
        for param_id, value in zip(batch_ids, batch_vals):
            data.extend(struct.pack("<H", param_id))  # uint16 little-endian
            data.extend(struct.pack("<e", value))  # float16 little-endian

        pk = CRTPPacket()
        pk.port = PARAM_PORT
        pk.channel = PARAM_WRITE_CH
        pk.data = bytes(data)
        cf.send_packet(pk)


def _build_param_names(hidden_size: int, obs_dim: int, act_dim: int) -> list[str]:
    """Return the onboard parameter names for the policy weights, in transmission order.

    Args:
        hidden_size: Width of the hidden layers.
        obs_dim: Observation dimension, the first layer's input.
        act_dim: Action dimension, the last layer's output.

    Returns:
        Names ordered W0, b0, W1, b1, W2, b2.
    """
    layers = (
        ("W0", obs_dim * hidden_size),
        ("b0", hidden_size),
        ("W1", hidden_size * hidden_size),
        ("b1", hidden_size),
        ("W2", hidden_size * act_dim),
        ("b2", act_dim),
    )
    return [
        f"ctrlThrow.{name}_{i:0{len(str(size - 1))}d}" for name, size in layers for i in range(size)
    ]


def _make_ballistic_prediction_fn(ballistic_sim: Sim, n_steps: int) -> Callable[[dict], dict]:
    """Create a JIT-compiled function that forward-sims x_throw for n_steps (rotors off).

    Args:
        ballistic_sim: A Sim instance without controller stages in its step_pipeline.
        n_steps: Number of simulation steps to run (= TRAINING_LATENCY * sim_freq).

    Returns:
        A jit-compiled callable: x_throw (dict) -> x_start (dict).
    """
    sim_step = ballistic_sim.build_step_fn()
    sim_data = ballistic_sim.data

    @jax.jit
    def forward(x_throw: dict) -> dict:
        """Predict the state n_steps later, with the rotors held at their idle speed.

        Args:
            x_throw: Dict with keys "pos", "quat", "vel", "ang_vel", each shape (1, 1, dim).
                     Quaternion convention: [x, y, z, w].

        Returns:
            x_start: Same dict structure, state after n_steps.
        """
        data = sim_data.replace(
            states=leaf_replace(
                sim_data.states,
                None,
                pos=x_throw["pos"],
                quat=x_throw["quat"],
                vel=x_throw["vel"],
                ang_vel=x_throw["ang_vel"],
                rotor_vel=jp.full((1, 1, 4), SIM_ROTOR_VEL),
            ),
            controls=leaf_replace(
                sim_data.controls, None, rotor_vel=jp.full((1, 1, 4), SIM_ROTOR_VEL)
            ),
        )
        data = sim_step(data, n_steps)
        x_start = {
            "pos": data.states.pos,
            "quat": data.states.quat,
            "vel": data.states.vel,
            "ang_vel": data.states.ang_vel,
        }
        return x_start

    return forward


class ThrowController(Controller):
    """Controller that trains a stabilization policy on-the-fly after a throw.

    Workflow:
        1. Create ThrowController (builds envs and the ballistic prediction sim).
        2. Call warmup_jit() to pre-compile all JAX functions (must finish before throw).
        3. After the drone is thrown, record x_throw (pos, quat, vel, ang_vel).
        4. Call train(x_throw), predicts x_start, trains policy (~1 s).
        5. Call the controller with an observation to get actions from the trained policy.
    """

    def __init__(self, use_x_start: bool = False) -> None:
        """Build the training envs, the compiled train function and the ballistic prediction sim."""
        self.control_type = "rotor_thrust"
        args = Args()
        self.last_action = np.full(4, IDLE_ACTION, dtype=np.float32)
        self._args = args

        # Build training envs. Random throw resets overwrite the whole state and ignore x_start
        self.use_x_start = use_x_start
        key = jax.random.PRNGKey(args.seed)
        envs = make_jitted_envs(args, jax_device="cpu", use_x_start=self.use_x_start)
        self._obs_dim = envs.single_observation_space.shape[0]
        self._act_dim = envs.single_action_space.shape[0]
        init_key, key = jax.random.split(key)
        self.agent = Agent.create(
            key=init_key,
            obs_dim=self._obs_dim,
            act_dim=self._act_dim,
            hidden_size=args.hidden_size,
            actor_lr=args.actor_lr,
        )
        _, self.train_fn = make_train_fns(args, collect_stats=False)

        # The x_start resets have to start from the predicted state, so they are reset again once
        # it is known. The random throw resets never read it and are reset right away
        self._envs = envs
        envs, (next_obs, _) = envs.reset(envs, seed=args.seed)
        next_done = jp.zeros(args.num_envs, dtype=bool)
        sum_rewards = jp.zeros((args.num_envs,))
        self._training_args = {
            "envs": envs,
            "key": key,
            "next_obs": next_obs,
            "next_done": next_done,
            "sum_rewards": sum_rewards,
        }
        self.trained = False
        self.x_start = None  # Takeover state the policy trained from, set by train()
        low, high = envs.unwrapped.single_action_space.low, envs.unwrapped.single_action_space.high
        self.mean = (low + high) / 2.0
        self.scale = (high - low) / 2.0

        # Build the ballistic prediction sim (1 world, same dynamics, no controllers)
        ballistic_sim = Sim(
            drone="cf21B_500",
            dynamics=Dynamics.first_principles,
            control=Control.rotor_vel,
            device="cpu",
            freq=SIM_FREQ,
            state_freq=100,
            attitude_freq=500,
            force_torque_freq=500,
        )
        n_steps = int(TRAINING_LATENCY * ballistic_sim.freq)
        self._ballistic_prediction = _make_ballistic_prediction_fn(ballistic_sim, n_steps)
        logger.info(
            f"ThrowController initialized. Ballistic sim: {n_steps} steps "
            f"({TRAINING_LATENCY:.1f} s at {ballistic_sim.freq} Hz)."
        )

        # Cached param IDs for fast send (populated by precompute_param_ids)
        self._param_ids: list[int] | None = None
        self._cf: cflib.crazyflie.Crazyflie | None = None

    def __call__(self, obs: dict[str, NDArray[np.floating]]) -> NDArray[np.floating]:
        """Compute the next action from the trained policy.

        Args:
            obs: Current observation dict from the drone.
            _: Optional info dict (unused).

        Returns:
            Action array in the simulator's native action space.
        """
        if not self.trained:
            raise ValueError("Agent not trained. Call train(x_throw) first.")
        policy_obs = flat_obs(obs["pos"], obs["quat"], obs["vel"], obs["ang_vel"], self.last_action)
        raw_act = self.agent.mean_action(self.agent.actor_states.params, policy_obs)
        self.last_action = raw_act
        return self._action_tf(raw_act)

    def precompute_param_ids(self, cf: cflib.crazyflie.Crazyflie) -> None:
        """Precompute and cache parameter IDs from the Crazyflie's TOC.

        Call this once after the drone is connected and before the flight, so that
        send_parameters() can run as fast as possible mid-flight.

        Args:
            cf: Connected Crazyflie with TOC downloaded.
        """
        param_names = _build_param_names(
            hidden_size=self._args.hidden_size, obs_dim=self._obs_dim, act_dim=self._act_dim
        )
        self._param_ids = [_get_param_id(cf, name) for name in param_names]
        self._cf = cf
        logger.info(f"Precomputed {len(self._param_ids)} parameter IDs.")

    def send_parameters(self) -> None:
        """Send trained policy parameters to the onboard controller.

        Uses pre-computed parameter IDs, so this only packs values and sends CRTP packets.

        Raises:
            ValueError: If precompute_param_ids() was not called first.
            ValueError: If the agent has not been trained yet.
        """
        # Extract policy params: W0 (15x8), b0 (8), W1 (8x8), b1 (8), W2 (8x4), b2 (4)
        params = self.agent.actor_states.params["params"]
        values: list[float] = []
        values.extend(np.array(params["Dense_0"]["kernel"]).flatten().tolist())
        values.extend(np.array(params["Dense_0"]["bias"]).tolist())
        values.extend(np.array(params["Dense_1"]["kernel"]).flatten().tolist())
        values.extend(np.array(params["Dense_1"]["bias"]).tolist())
        values.extend(np.array(params["Dense_2"]["kernel"]).flatten().tolist())
        values.extend(np.array(params["Dense_2"]["bias"]).tolist())
        logger.debug(f"Params healthy (no NaNs): {not np.isnan(values).any()}")
        _send_params(self._cf, self._param_ids, values)
        logger.info(f"Sent {len(values)} policy parameters to drone.")

    def train(self, x_throw: dict | None = None) -> None:
        """Train the stabilization policy, optionally conditioned on x_throw.

        If x_throw is provided, predicts x_start with the ballistic prediction and trains the
        policy starting from x_start (plus small domain randomization). If x_throw is None, trains
        from the default hover-like x_start.

        Args:
            x_throw: Dict with keys "pos", "quat", "vel", "ang_vel" (shape (1,1,dim) each), or
                     None to use default x_start.
        """
        if x_throw is not None and not self.use_x_start:
            raise ValueError("x_throw given, but the random throw resets would ignore it")
        training_args = self._training_args
        if x_throw is not None:
            self.x_start = self._ballistic_prediction(x_throw)
            logger.info(f"Computed x_start: {self.x_start}")
            envs, next_obs = reset_at_x_start(self._envs, self.x_start, seed=self._args.seed)
            training_args = training_args | {"envs": envs, "next_obs": next_obs}
        (_, self.agent, _, _, _, _), _ = self.train_fn(
            **training_args, agent=self.agent, x_start=self.x_start
        )
        jax.block_until_ready(self.agent.actor_states.params)
        self.trained = True

    def warmup_jit(self) -> None:
        """Pre-compile all JAX functions. Must be called before the drone is thrown."""
        x_throw = None
        x_start = None
        if self.use_x_start:
            x_throw = self._default_x_throw()
        # Warmup the ballistic prediction
        if x_throw is not None:
            x_start = self._ballistic_prediction(x_throw)
            jax.block_until_ready(x_start)
        logger.info("Ballistic sim JIT compiled.")
        # Warmup the reset at x_start and train_fn
        training_args = self._training_args
        if x_start is not None:
            envs, next_obs = reset_at_x_start(self._envs, x_start, seed=self._args.seed)
            training_args = training_args | {"envs": envs, "next_obs": next_obs}
        jax.block_until_ready(self.train_fn(**training_args, agent=self.agent, x_start=x_start))
        logger.info("train_fn JIT compiled.")

    def _default_x_throw(self) -> dict:
        """Return a default hover-like x_start (shape (1,1,dim) each)."""
        return {
            "pos": jp.array([0.0, 0.0, 2.2], dtype=jp.float32),
            "quat": jp.array([0.0, 0.0, 0.0, 1.0], dtype=jp.float32),
            "vel": jp.zeros((3,), dtype=jp.float32),
            "ang_vel": jp.zeros((3,), dtype=jp.float32),
        }

    def _action_tf(self, actions: NDArray) -> NDArray:
        """Rescale and clip actions from [-1, 1] to simulator action space."""
        actions = np.clip(np.array(actions), -1.0, 1.0) * self.scale + self.mean
        return actions.astype(np.float32)

    def done(self, obs: dict[str, NDArray[np.floating]]) -> bool:
        """Never finish on its own, the throw is stopped by the deployment script."""
        return False
