"""Test the MPPI obstacles controller."""

from __future__ import annotations

import subprocess
import sys

from tests.conftest import REPO_ROOT, module_requires_env

module_requires_env("mppi")


TIMEOUT_S = 600


def test_mppi():
    """Test that the MPPI obstacles controller runs without errors."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crazyflow_experiments.mppi_obstacles.sim",
            "--render",
            "False",
            "--save",
            "False",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
    )
    assert result.returncode == 0, f"MPPI failed:\n{result.stderr}\nOutput:\n{result.stdout}"
