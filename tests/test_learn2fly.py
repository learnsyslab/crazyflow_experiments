"""Train the TD3 reach-position policy and check that it reaches its goal and flies in the sim."""

from __future__ import annotations

import pickle
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

from tests.conftest import PACKAGE_ROOT, REPO_ROOT, module_requires_env

module_requires_env("learn2fly")

import toml  # noqa: E402

from crazyflow_experiments.core.agents.td3_agent import load_actor  # noqa: E402
from crazyflow_experiments.core.envs.wrappers import RecordData  # noqa: E402
from crazyflow_experiments.learn2fly import controller as learn2fly_controller  # noqa: E402
from crazyflow_experiments.learn2fly import sim as learn2fly_sim  # noqa: E402
from crazyflow_experiments.learn2fly.train import (  # noqa: E402
    Args,
    eval_figure8,
    eval_reach_pos,
    make_reach_pos_envs,
)
from tests.sim2real_fixtures import tracking_error  # noqa: E402

if TYPE_CHECKING:
    from pathlib import Path

TIMEOUT_S = 600
CONFIG_PATH = PACKAGE_ROOT / "learn2fly" / "config.toml"
# Bounds are ~2x what the policy achieves. Anything outside is a major regression
MAX_REACH_RMSE_MM = 700.0
MAX_TRACKING_RMSE_MM = {"3.5s": 1600.0, "5.5s": 400.0, "15s": 150.0}
# The policy chases a moving goal, so it lags behind the trajectory of the simulated flight
MAX_SIM_TRACKING_ERROR = 1.0  # m


def run_training(save_dir: Path) -> Path:
    """Train with wandb disabled and return the checkpoint it wrote."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crazyflow_experiments.learn2fly.train",
            "--wandb_enabled",
            "False",
            "--n_eval",
            "0",
            "--save_dir",
            str(save_dir),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
    )
    assert result.returncode == 0, f"training exited {result.returncode}\n{result.stderr}"
    checkpoints = sorted(save_dir.glob("*.ckpt"))
    assert checkpoints, f"training wrote no checkpoint into {save_dir}"
    return checkpoints[0]


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Train the policy once for the module, into the checkpoint the deployed controller loads."""
    checkpoint = run_training(tmp_path_factory.mktemp("learn2fly"))
    assert checkpoint.name == "td3_rprv_model.ckpt", checkpoint.name
    return checkpoint


def test_train_td3(checkpoint: Path):
    """The trained reach-position policy flies to the goal and tracks a figure-8."""
    with checkpoint.open("rb") as file:
        params = pickle.load(file)
    assert set(params) == {"actor", "critic1", "critic2"}, (
        f"{checkpoint.name} holds {sorted(params)}"
    )

    args = Args()
    agent = load_actor(checkpoint)
    reach_env = RecordData.create(make_reach_pos_envs(1, args))
    rmse_mm = eval_reach_pos(args, 5, agent=agent, eval_env=reach_env)[0] * 1000
    assert rmse_mm < MAX_REACH_RMSE_MM, f"reach error {rmse_mm:.1f} mm exceeds {MAX_REACH_RMSE_MM}"

    for label, (rmse, _, _) in eval_figure8(args, 1, agent=agent).items():
        bound = MAX_TRACKING_RMSE_MM[label]
        assert rmse * 1000 < bound, f"{label} tracking error {rmse * 1000:.1f} mm exceeds {bound}"


def test_sim_tracks_the_trajectory(
    checkpoint: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The simulated flight of `sim.py` follows the configured trajectory with the policy."""
    monkeypatch.setattr(learn2fly_controller, "MODEL_DIR", checkpoint.parent)
    monkeypatch.setattr(learn2fly_sim, "RECORDINGS_DIR", tmp_path)
    config = toml.load(CONFIG_PATH)
    config["experiment"]["n_runs"] = 1
    config["experiment"]["time"] = 10.0
    config["sim"]["render"] = False
    learn2fly_sim.main(config)
    recordings = list(tmp_path.glob("*.pkl"))
    assert len(recordings) == 1, f"expected one recording, got {recordings}"
    with recordings[0].open("rb") as file:
        data = pickle.load(file)
    error = tracking_error(config, data)
    assert error.max() < MAX_SIM_TRACKING_ERROR, f"policy peaked at {error.max() * 100:.1f}cm"
