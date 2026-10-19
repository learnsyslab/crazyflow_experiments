"""Train, simulate and evaluate the throw recovery policy."""

from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING

from tests.conftest import REPO_ROOT, module_requires_env

module_requires_env("throw")

import jax  # noqa: E402
import numpy as np  # noqa: E402

from crazyflow_experiments.throw.collect import make_deploy_evaluation  # noqa: E402
from crazyflow_experiments.throw.train import (  # noqa: E402
    Args,
    load_agent,
    make_jitted_envs,
)
from tests.throw_fixtures import write_recording  # noqa: E402

if TYPE_CHECKING:
    from pathlib import Path

TIMEOUT_S = 900
N_EVAL = 64
TRAIN_SEED = 7  # Variance is high, so pin the seed to avoid flaky tests
MIN_SUCCESS = 0.3


def run_script(module: str, *flags: str):
    """Run a throw module with flags and fail with its stderr if it exits non-zero."""
    result = subprocess.run(
        [sys.executable, "-m", f"crazyflow_experiments.throw.{module}", *flags],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
    )
    assert result.returncode == 0, f"{module} exited {result.returncode}\n{result.stderr}"


def test_train(tmp_path: Path):
    """Train a policy, then check the saved checkpoint recovers from the throw often enough.

    The throws are random takeover states of the training distribution, scored like deploy.py.
    """
    run_script(
        "train",
        "--wandb_enabled",
        "False",
        "--n_eval",
        "0",
        "--save",
        "True",
        "--save_dir",
        str(tmp_path),
        "--seed",
        str(TRAIN_SEED),
    )
    checkpoints = sorted(tmp_path.glob("*.ckpt"))
    assert checkpoints, f"train wrote no checkpoint into {tmp_path}"

    args = Args()
    throws = make_jitted_envs(args, override_num_envs=N_EVAL)
    agent = load_agent(args, throws, checkpoints[0])
    throws, _ = throws.reset(throws, seed=args.seed)
    states = throws.unwrapped.data.states
    takeover = {key: getattr(states, key)[:, 0] for key in ("pos", "quat", "vel", "ang_vel")}
    evaluate = jax.jit(jax.vmap(make_deploy_evaluation(args), in_axes=(None, 0)))
    success_rate = np.mean(evaluate(agent, takeover))
    assert success_rate > MIN_SUCCESS, f"recovered {success_rate:.0%}, want over {MIN_SUCCESS:.0%}"


def test_sim():
    """The simulation trains a policy in-flight and stabilises without erroring."""
    run_script("sim", "--render", "False")


def test_collect_success_curves(tmp_path: Path):
    """Replaying recorded throws produces success curves across checkpoints and seeds."""
    recordings, curves = tmp_path / "rec", tmp_path / "curves.json"
    recordings.mkdir()
    for name in ("x_throw_a.pkl", "x_throw_b.pkl"):
        write_recording(recordings / name)

    run_script(
        "collect",
        "--recordings_dir",
        str(recordings),
        "--curves_path",
        str(curves),
        "--n_seeds",
        "2",
    )
    recorded = json.loads(curves.read_text())
    assert len(recorded["successes"]) == len(recorded["traces"]) == 2
    assert len(recorded["successes"][0]) == len(recorded["seeds"]) == 2
    assert len(recorded["successes"][0][0]) == Args().num_iterations
