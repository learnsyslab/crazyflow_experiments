"""Run each benchmark and check the results it saves.

Every run is pinned to a small configuration and saves into a pytest temporary directory, so a full
pass costs seconds and leaves the published results untouched. A benchmark gets a test here once its
CLI accepts the subset flags, otherwise the run falls back to the full sweep and takes hours.

`stats.py` is deliberately not covered.
"""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING

import polars as pl

from tests.conftest import REPO_ROOT, requires_env

if TYPE_CHECKING:
    from pathlib import Path

TIMEOUT_S = 300

# The full sweeps reach millions of environments, so every run is pinned to a few environments and
# two timing repetitions.
SUBSET = ("--device", "cpu", "--n_worlds", "4", "--n_drones", "2", "--repeat", "2", "--number", "2")
# Simulators that run a single drone per environment take no drone count
SUBSET_SINGLE_DRONE = ("--device", "cpu", "--n_worlds", "4", "--repeat", "2", "--number", "2")
# Render benchmarks sweep resolutions instead of drone counts
SUBSET_RENDER = (
    "--device",
    "cpu",
    "--n_worlds",
    "2",
    "--resolution",
    "16,16",
    "--repeat",
    "2",
    "--number",
    "2",
)
SUBSET_MODE_RENDER = ("--n_worlds", "2", "--resolution", "16,16", "--repeat", "2", "--number", "2")

COLUMNS = ("n_worlds", "device", "timings")  # Columns every benchmark must save
MODE_RENDER_COLUMNS = ("n_worlds", "resolution", "mode", "timings")  # Renderings have extra columns


def run_benchmark(module: str, results_dir: Path, *flags: str):
    """Run a benchmark on the subset configuration and save its results to results_dir.

    Args:
        module: Benchmark module name inside crazyflow_experiments.benchmarks.
        results_dir: Directory the benchmark saves its results to.
        flags: Flags replacing the default subset, for benchmarks with a different CLI.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            f"crazyflow_experiments.benchmarks.{module}",
            *(flags or SUBSET),
            "--results_dir",
            str(results_dir),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
    )
    assert result.returncode == 0, f"{module} exited {result.returncode}"


def check_results(results_dir: Path, columns: tuple[str, ...] = COLUMNS):
    """Assert the benchmark saved results and that they carry the expected measurements."""
    saved = sorted(results_dir.glob("*.parquet"))
    assert saved, f"no results saved, found {sorted(p.name for p in results_dir.iterdir())}"
    for path in saved:
        frame = pl.read_parquet(path)
        missing = [column for column in columns if column not in frame.columns]
        assert not missing, f"{path.name} has columns {frame.columns}, missing {missing}"
        assert len(frame) > 0, f"{path.name} has no rows"
        assert all(len(timings) > 0 for timings in frame["timings"]), (
            f"{path.name} records empty timings, so nothing was measured"
        )


@requires_env("crazyflow")
def test_crazyflow_dynamics(tmp_path: Path):
    """Benchmark the crazyflow dynamics."""
    run_benchmark("crazyflow_dynamics", tmp_path)
    check_results(tmp_path)


@requires_env("crazyflow")
def test_crazyflow_dynamics_rotor_level(tmp_path: Path):
    """Benchmark the crazyflow dynamics driven by rotor velocity commands."""
    run_benchmark("crazyflow_dynamics", tmp_path, *SUBSET_SINGLE_DRONE, "--control", "rotor_vel")
    check_results(tmp_path)


@requires_env("crazyflow")
def test_crazyflow_gradients(tmp_path: Path):
    """Benchmark the crazyflow gradient computation."""
    run_benchmark("crazyflow_gradients", tmp_path)
    check_results(tmp_path)


@requires_env("crazyflow")
def test_crazyflow_render(tmp_path: Path):
    """Benchmark the crazyflow depth rendering."""
    run_benchmark("crazyflow_render", tmp_path, *SUBSET_RENDER)
    check_results(tmp_path)


@requires_env("vision")
def test_crazyflow_splat_render(tmp_path: Path):
    """Benchmark the crazyflow gaussian splat rendering."""
    run_benchmark("crazyflow_splat_render", tmp_path, *SUBSET_MODE_RENDER)
    check_results(tmp_path, MODE_RENDER_COLUMNS)


@requires_env("flightmare")
def test_flightmare_dynamics(tmp_path: Path):
    """Benchmark the flightmare dynamics."""
    run_benchmark("flightmare_dynamics", tmp_path, *SUBSET_SINGLE_DRONE)
    check_results(tmp_path)


@requires_env("flightmare")
def test_flightmare_render(tmp_path: Path):
    """Benchmark the flightmare Unity rendering."""
    run_benchmark("flightmare_render", tmp_path, *SUBSET_MODE_RENDER)
    check_results(tmp_path, MODE_RENDER_COLUMNS)


@requires_env("l2f")
def test_l2f_dynamics(tmp_path: Path):
    """Benchmark the l2f dynamics."""
    run_benchmark("l2f_dynamics", tmp_path, *SUBSET_SINGLE_DRONE)
    check_results(tmp_path)


@requires_env("diff-aero")
def test_diff_aero_dynamics(tmp_path: Path):
    """Benchmark the diff-aero dynamics."""
    run_benchmark("diff_aero_dynamics", tmp_path)
    check_results(tmp_path)


@requires_env("diff-aero")
def test_diff_aero_gradients(tmp_path: Path):
    """Benchmark the diff-aero gradient computation."""
    run_benchmark("diff_aero_gradients", tmp_path)
    check_results(tmp_path)


@requires_env("diff-aero")
def test_diff_aero_render(tmp_path: Path):
    """Benchmark the diff-aero camera ray casting."""
    run_benchmark("diff_aero_render", tmp_path, *SUBSET_RENDER)
    check_results(tmp_path)


@requires_env("aerial-gym")
def test_aerial_gym_dynamics(tmp_path: Path):
    """Benchmark the aerial-gym dynamics."""
    run_benchmark("aerial_gym_dynamics", tmp_path, *SUBSET_SINGLE_DRONE)
    check_results(tmp_path)


@requires_env("aerial-gym")
def test_aerial_gym_dynamics_rotor_level(tmp_path: Path):
    """Benchmark the aerial-gym dynamics driven by rotor velocity commands."""
    run_benchmark("aerial_gym_dynamics", tmp_path, *SUBSET_SINGLE_DRONE, "--control", "rotor_vel")
    check_results(tmp_path)


@requires_env("pybullet")
def test_gym_pybullet_drones_dynamics(tmp_path: Path):
    """Benchmark the gym-pybullet-drones dynamics."""
    run_benchmark("gym_pybullet_drones_dynamics", tmp_path)
    check_results(tmp_path)


@requires_env("pybullet")
def test_gym_pybullet_drones_dynamics_rotor_level(tmp_path: Path):
    """Benchmark the gym-pybullet-drones dynamics driven by rotor velocity commands."""
    run_benchmark(
        "gym_pybullet_drones_dynamics", tmp_path, *SUBSET_SINGLE_DRONE, "--control", "rotor_vel"
    )
    check_results(tmp_path)
