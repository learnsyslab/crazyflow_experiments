"""Train the figure-8 policies and check that they and the MPC track the trajectory."""

from __future__ import annotations

import pickle
import subprocess
import sys
from typing import TYPE_CHECKING, Callable

import pytest

from tests.conftest import PACKAGE_ROOT, REPO_ROOT, module_requires_env

module_requires_env("figure8")

import jax.numpy as jp  # noqa: E402
import numpy as np  # noqa: E402
import toml  # noqa: E402

from crazyflow_experiments.core.envs import domain_randomization as dr  # noqa: E402
from crazyflow_experiments.core.envs.figure8_env import FigureEightEnv  # noqa: E402
from crazyflow_experiments.figure8 import controller as figure8_controller  # noqa: E402
from crazyflow_experiments.figure8 import sim as figure8_sim  # noqa: E402
from crazyflow_experiments.figure8.train_bptt import Args as BPTTArgs  # noqa: E402
from crazyflow_experiments.figure8.train_bptt import evaluate_bptt  # noqa: E402
from crazyflow_experiments.figure8.train_ppo import Args as PPOArgs  # noqa: E402
from crazyflow_experiments.figure8.train_ppo import (  # noqa: E402
    evaluate_ppo,
    evaluate_ppo_batched,
)
from tests.sim2real_fixtures import tracking_error  # noqa: E402

if TYPE_CHECKING:
    from pathlib import Path

TIMEOUT_S = 600
CONFIG_PATH = PACKAGE_ROOT / "figure8" / "config.toml"
MAX_TRACKING_ERROR = 0.15  # m, of every controller flying the figure-8 in simulation

ENV_SETTINGS = {
    "num_envs": 4,
    "max_episode_time": 1.0,
    "dynamics": "first_principles",
    "freq": 50,
    "reset_rotor": True,
    "multi_start": True,
}


def run_training(module: str, save_dir: Path, *flags: str) -> Path:
    """Train with wandb disabled and return the checkpoint it wrote.

    Args:
        module: Training module name inside crazyflow_experiments.figure8.
        save_dir: Directory the checkpoint is written to.
        flags: Flags appended to the invocation.

    Returns:
        Path of the written checkpoint.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            f"crazyflow_experiments.figure8.{module}",
            "--wandb_enabled",
            "False",
            "--save_dir",
            str(save_dir),
            *flags,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
    )
    assert result.returncode == 0, f"{module} exited {result.returncode}\n{result.stderr}"
    checkpoints = sorted(save_dir.glob("*.ckpt"))
    assert checkpoints, f"{module} wrote no checkpoint into {save_dir}"
    return checkpoints[0]


@pytest.fixture(scope="module")
def trained_checkpoint(
    tmp_path_factory: pytest.TempPathFactory,
) -> Callable[[str, str, float], Path]:
    """Train each requested policy once and share its temporary checkpoint across tests."""
    checkpoints: dict[tuple[str, str, float], Path] = {}

    def get_checkpoint(module: str, preset: str, dr_scale: float = 0.0) -> Path:
        key = (module, preset, dr_scale)
        if key not in checkpoints:
            save_dir = tmp_path_factory.mktemp(f"{module}_{preset}_dr{dr_scale:g}")
            flags = ["--preset", preset]
            if dr_scale:
                flags.extend(["--dr_scale", str(dr_scale)])
            checkpoints[key] = run_training(module, save_dir, *flags)
        return checkpoints[key]

    return get_checkpoint


def single_lap_config(controller: str) -> dict:
    """Load the figure8 config, shortened to a single lap of the 10 s figure-8 without rendering."""
    config = toml.load(CONFIG_PATH)
    config["controller"]["type"] = controller
    config["experiment"]["n_runs"] = 1
    config["experiment"]["time"] = 10.0
    config["sim"]["render"] = False
    return config


@pytest.mark.parametrize("controller", ["mellinger", "mpc", "ppo", "bptt"])
def test_sim_tracks_the_trajectory(
    controller: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    trained_checkpoint: Callable[[str, str, float], Path],
):
    """Every controller flies the configured trajectory in simulation and follows the reference."""
    if controller in ("ppo", "bptt"):
        checkpoint = trained_checkpoint(f"train_{controller}", "10s", 0.0)
        monkeypatch.setattr(figure8_controller, "MODEL_DIR", checkpoint.parent)
    config = single_lap_config(controller)
    monkeypatch.setattr(figure8_sim, "RECORDINGS_DIR", tmp_path)
    figure8_sim.main(config=config)
    recordings = list(tmp_path.glob("*.pkl"))
    assert len(recordings) == 1, f"expected one recording, got {recordings}"
    with recordings[0].open("rb") as file:
        data = pickle.load(file)
    error = tracking_error(config, data)
    assert error.max() < MAX_TRACKING_ERROR, f"{controller} peaked at {error.max() * 100:.1f}cm"


def check_checkpoint(path: Path):
    """Assert the checkpoint holds actor parameters that can be loaded back."""
    with path.open("rb") as file:
        params = pickle.load(file)
    assert "actor" in params, f"{path.name} holds {sorted(params)}, missing the actor parameters"
    assert params["actor"], f"{path.name} holds empty actor parameters"


@pytest.mark.parametrize(("preset", "max_rmse_mm"), [("10s", 10.0), ("5_5s", 10.0)])
def test_train_bptt(
    trained_checkpoint: Callable[[str, str, float], Path], preset: str, max_rmse_mm: float
):
    """Train a BPTT policy for both trajectory presets and check how well it tracks."""
    checkpoint = trained_checkpoint("train_bptt", preset, 0.0)
    check_checkpoint(checkpoint)
    rmse_mm = evaluate_bptt(BPTTArgs.create(preset), 1, checkpoint, False)[0] * 1000
    assert rmse_mm < max_rmse_mm, f"tracking error {rmse_mm:.1f} mm exceeds {max_rmse_mm} mm"


@pytest.mark.parametrize(("preset", "max_rmse_mm"), [("10s", 10.0), ("5_5s", 25.0)])
def test_train_ppo(
    trained_checkpoint: Callable[[str, str, float], Path], preset: str, max_rmse_mm: float
):
    """Train a PPO policy for both trajectory presets and check how well it tracks."""
    checkpoint = trained_checkpoint("train_ppo", preset, 0.0)
    check_checkpoint(checkpoint)
    rmse_mm = evaluate_ppo(PPOArgs.create(preset), 1, checkpoint, False)[0] * 1000
    assert rmse_mm < max_rmse_mm, f"tracking error {rmse_mm:.1f} mm exceeds {max_rmse_mm} mm"


def test_train_ppo_randomized(trained_checkpoint: Callable[[str, str, float], Path]):
    """Train PPO under domain randomization and check nominal and randomized tracking."""
    checkpoint = trained_checkpoint("train_ppo", "10s", 1.0)
    check_checkpoint(checkpoint)
    assert checkpoint.name == "ppo_f8_10s_dr1.0_model.ckpt", checkpoint.name

    rmse_mm = evaluate_ppo(PPOArgs.create("10s"), 1, checkpoint, False)[0] * 1000
    assert rmse_mm < 15.0, f"nominal tracking error {rmse_mm:.1f} mm exceeds 15 mm"
    metrics = evaluate_ppo_batched(PPOArgs.create("10s"), checkpoint, dr_scale=1.0, num_envs=64)
    assert set(metrics) == {"pos_rmse_mm", "crash_rate", "mean_reward", "mean_steps"}
    assert metrics["crash_rate"] == 0.0, f"policy crashes under randomization: {metrics}"
    assert metrics["pos_rmse_mm"] < 35.0, f"randomized tracking error too large: {metrics}"


def test_randomized_env():
    """Randomize the dynamics per world and check the draws, the masked reset and the noise."""
    ranges = dr.DynamicsRanges()
    num_envs = ENV_SETTINGS["num_envs"]
    env = FigureEightEnv.create(**ENV_SETTINGS, randomization=ranges)
    env, _ = env.reset(env, seed=0)
    nominal, params = env.sim.default_data.params, env.data.params

    def factors(name: str) -> np.ndarray:
        """Scale factors of a randomized parameter relative to nominal, one row per world."""
        return np.asarray(getattr(params, name) / getattr(nominal, name)).reshape(num_envs, -1)

    inertia = np.diagonal(np.asarray(params.J / nominal.J), axis1=-2, axis2=-1)
    rotor = factors("rotor_dyn_coef")
    for name, values, spread in [
        ("mass", factors("mass"), ranges.mass),
        ("inertia", inertia.reshape(num_envs, -1), ranges.inertia),
        ("rotor dynamics", rotor, ranges.rotor_dynamics),
        ("thrust curve", factors("rpm2thrust"), ranges.thrust_curve),
        ("torque curve", factors("rpm2torque"), ranges.torque_curve),
    ]:
        # Zero coefficients stay zero, so their factors are 0 / 0 and are skipped
        finite = np.isfinite(values)
        assert np.unique(values[:, -1]).size > 1, f"{name} is the same in every world: {values}"
        assert np.all(np.abs(values[finite] - 1.0) <= spread + 1e-6), f"{name} spread: {values}"
    for name in ("rotor_dyn_coef", "rpm2thrust", "rpm2torque"):
        world = factors(name)[0]
        finite = world[np.isfinite(world)]
        assert np.unique(finite).size == finite.size, f"{name} factors repeat within a world"
    assert np.allclose(np.asarray(params.J_inv @ params.J), np.eye(3), atol=1e-4), "J_inv is stale"

    wrench = env.data.plugins["wrench"]
    force, torque = np.asarray(wrench.force), np.asarray(wrench.torque)
    assert np.any(force != 0) and np.all(np.abs(force) <= ranges.force), force
    assert np.any(torque != 0) and np.all(np.abs(torque) <= ranges.torque), torque
    assert np.allclose(np.asarray(env.data.states.force), force), "the wrench is not applied"

    action = jp.zeros(env.action_space.shape)
    for _ in range(3):
        env, (obs, *_) = env.step(env, action)
    assert np.all(np.isfinite(np.asarray(obs["pos"]))), obs["pos"]
    assert np.allclose(np.asarray(env.data.plugins["wrench"].force), force), "the wrench drifted"
    assert not np.allclose(np.asarray(env.data.states.force), force), "the wrench has no noise"

    mask = jp.array([True, False, False, False])
    reset = env.sim._reset(env.data, env.sim.default_data, mask)
    # Exact comparisons: unmasked worlds pass through jnp.where untouched, and allclose's atol
    # would hide the redraw of the tiny torque curve coefficients
    for name in ("mass", "J", "rotor_dyn_coef", "rpm2thrust", "rpm2torque"):
        before, after = np.asarray(getattr(params, name)), np.asarray(getattr(reset.params, name))
        assert not np.array_equal(after[0], before[0]), f"{name} of the reset world is unchanged"
        assert np.array_equal(after[1:], before[1:]), f"{name} of the other worlds changed"
    assert not np.array_equal(np.asarray(reset.plugins["wrench"].force[0]), force[0])
    assert np.array_equal(np.asarray(reset.plugins["wrench"].force[1:]), force[1:])


def test_randomized_env_zero_scale():
    """Zero spreads reproduce the plain env, and no randomization leaves its pipelines untouched."""
    plain = FigureEightEnv.create(**ENV_SETTINGS)
    assert plain.data.plugins == {} and plain.randomization is None
    assert "disturb_wrench" not in plain.sim.step_pipeline
    assert "randomize_mass" not in plain.sim.reset_pipeline

    zero = FigureEightEnv.create(**ENV_SETTINGS, randomization=dr.DynamicsRanges().scale(0.0))
    plain, (obs_plain, _) = plain.reset(plain, seed=1)
    zero, (obs_zero, _) = zero.reset(zero, seed=1)
    nominal = zero.sim.default_data.params
    for name in ("mass", "J", "J_inv", "rotor_dyn_coef", "rpm2thrust", "rpm2torque"):
        drawn = np.asarray(getattr(zero.data.params, name))
        base = np.asarray(getattr(nominal, name))
        assert np.array_equal(drawn, np.broadcast_to(base, drawn.shape)), f"{name} deviates"
    action = jp.tile(jp.array([[0.0, 0.0, 0.0, 0.43]]), (ENV_SETTINGS["num_envs"], 1))
    for _ in range(20):
        plain, (obs_plain, *_) = plain.step(plain, action)
        zero, (obs_zero, *_) = zero.step(zero, action)
    assert np.allclose(np.asarray(obs_plain["pos"]), np.asarray(obs_zero["pos"]), atol=1e-6)
