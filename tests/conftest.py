"""Shared fixtures and environment gating for the experiment test suite.

Each script declares the Pixi environment it runs in. Running the suite in another environment skips
it with the reason, so the same suite covers every environment without failing for the wrong one.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[1]
PACKAGE_ROOT = REPO_ROOT / "crazyflow_experiments"
PIXI_ENV = os.environ.get("PIXI_ENVIRONMENT_NAME")

JAX_CACHE_DIR = Path(tempfile.gettempdir()) / f"jax_cache-{os.getuid()}"

os.environ["SCIPY_ARRAY_API"] = "1"
os.environ["WANDB_MODE"] = "disabled"  # Disable WandB to not write test runs to the cloud
os.environ["MPLBACKEND"] = "Agg"  # The suite is headless, and interactive backends crash it
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # Preallocation makes back to back runs crash
os.environ["JAX_COMPILATION_CACHE_DIR"] = str(JAX_CACHE_DIR)
os.environ["JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"] = "-1"
os.environ["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "0"


def requires_env(env: str) -> pytest.MarkDecorator:
    """Skip the calling test unless the active Pixi environment is one of envs."""
    return pytest.mark.skipif(PIXI_ENV != env, reason=f"needs env {env!r}, running in {PIXI_ENV!r}")


def module_requires_env(env: str):
    """Skip the calling module unless the active Pixi environment is env.

    Called at the top of a module before it imports anything the environment provides, so the
    module is skipped rather than failing to import elsewhere.
    """
    if PIXI_ENV != env:
        pytest.skip(f"needs env {env!r}, running in {PIXI_ENV!r}", allow_module_level=True)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Root of the checkout under test."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def package_root() -> Path:
    """Root of the crazyflow_experiments package under test."""
    return PACKAGE_ROOT
