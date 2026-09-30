"""Train the vision reach policy and fly it in simulation."""

from __future__ import annotations

import os

os.environ["SCIPY_ARRAY_API"] = "1"  # Must be set before anything imports scipy

from typing import TYPE_CHECKING

import pytest

from tests.conftest import PACKAGE_ROOT, module_requires_env

module_requires_env("vision")

import toml  # noqa: E402

from crazyflow_experiments.vision import controller, sim, train  # noqa: E402

if TYPE_CHECKING:
    from pathlib import Path

CONFIG_PATH = PACKAGE_ROOT / "vision" / "config.toml"
EXP_NAME = "test"
FLIGHT_TIME = 2.0  # s

# Enough of a run to cover both the cloning epoch and the student driven ones
TRAIN_SETTINGS = {
    "exp_name": EXP_NAME,
    "wandb_enabled": False,
    "num_envs": 8,
    "rollout_steps": 10,
    "epochs": 2,
    "capacity": 1000,
    "batch_size": 8,
    "updates_per_epoch": 10,
    "eval_envs": 8,
    "eval_steps": 20,
    "eval_every": 1,
}


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[dict]]:
    """Train a short run into a temporary model directory and return what it wrote."""
    model_dir = tmp_path_factory.mktemp("vision")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(train, "MODEL_DIR", model_dir)
        history = train.main(**TRAIN_SETTINGS)
    return model_dir / f"reach_policy_{EXP_NAME}.msgpack", history


def test_train(trained: tuple[Path, list[dict]]):
    """Training reports its metrics and leaves a checkpoint the controller can rebuild from."""
    checkpoint, history = trained
    assert len(history) == TRAIN_SETTINGS["epochs"], f"{len(history)} epochs evaluated"
    for metrics in history:
        assert 0.0 <= metrics["eval/success"] <= 1.0, metrics
        assert metrics["train/loss"] > 0.0, metrics

    params, config = controller.load_checkpoint(checkpoint)
    assert params, f"{checkpoint.name} holds no policy parameters"
    assert config["exp_name"] == EXP_NAME and config["control"] == "attitude", config


def test_sim(trained: tuple[Path, list[dict]], monkeypatch: pytest.MonkeyPatch):
    """The trained policy flies in simulation without erroring."""
    checkpoint, _ = trained
    monkeypatch.setattr(controller, "MODEL_DIR", checkpoint.parent)
    config = toml.load(CONFIG_PATH)
    config["sim"]["render"] = False
    sim.main(policy=EXP_NAME, splat_view=False, max_time=FLIGHT_TIME, config=config)
