"""Common benchmarking functions."""

from __future__ import annotations

import logging
import multiprocessing
import shutil
import socket
import subprocess
import timeit
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from queue import Empty
from typing import TYPE_CHECKING, Callable

import numpy as np
import polars as pl

from crazyflow_experiments.utils import paths

if TYPE_CHECKING:
    from numpy.typing import NDArray


METADATA = ("host", "cpu", "gpu", "measured_at", "commit", "dirty")  # never identify a result
TIMEOUT = 600  # seconds
CONTROL_FREQ = 500  # Hz
SIM_FREQ = 500  # Hz
WORLD_RANGE = [2**i for i in range(0, 25, 2)]  # number of parallel environments
DRONE_RANGE = [2**i for i in range(13)]
RESOLUTIONS: list[tuple[int, int]] = [(64, 64)]  # render benchmark resolutions (width, height)
REPEAT = 50  # number of repetitions for benchmarking
NUMBER = 50  # number of executions per repetition

logger = logging.getLogger(__name__)

RESULTS_DIR = paths.RESULTS_DIR / "benchmark"


class WorkerError(Exception):
    """Raised when an isolated worker process fails, times out, or produces no result."""


def _isolated_target(fn: Callable, config: dict, queue: multiprocessing.Queue):
    """Top-level worker target (must be module-level to be picklable with spawn)."""
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger(fn.__module__).setLevel(logging.INFO)
    queue.put(fn(**config))


def run_isolated(fn: Callable, config: dict) -> NDArray:
    """Run ``fn(**config)`` in an isolated spawned process and return the result.

    Uses ``multiprocessing.get_context("spawn")`` so each call gets a fresh interpreter, safe for
    libraries (Isaac Gym, JAX, PyBullet) that allow only one instance per process.

    Args:
        fn: Module-level callable that accepts ``**config`` and returns an NDArray.
        config: Keyword arguments forwarded to *fn*.

    Returns:
        The NDArray returned by *fn* inside the worker process.

    Raises:
        WorkerError: On timeout, non-zero exit code, or missing result.
    """
    ctx = multiprocessing.get_context("spawn")
    queue: multiprocessing.Queue = ctx.Queue()
    proc = ctx.Process(target=_isolated_target, args=(fn, config, queue))
    proc.start()
    proc.join(timeout=TIMEOUT)
    if proc.is_alive():
        proc.kill()
        proc.join()
        raise WorkerError(f"timed out after {TIMEOUT}s for {config}")
    if proc.exitcode != 0:
        raise WorkerError(f"exit code {proc.exitcode} for {config}")
    try:
        return queue.get_nowait()
    except Empty:
        raise WorkerError(f"no result returned for {config}")


def _is_dominated(config: dict, exhausted: list[dict]) -> bool:
    """Return True if config is dominated by any exhausted config.

    A config is dominated when it runs on the same device, has at least as many worlds and drones as
    the exhausted config, and all other keys match exactly.
    """
    for e in exhausted:
        if config.get("device") != e.get("device"):
            continue
        if config.get("n_worlds", 0) < e.get("n_worlds", 0):
            continue
        if config.get("n_drones", 0) < e.get("n_drones", 0):
            continue
        other_keys = set(e.keys()) - {"n_worlds", "n_drones", "device"}
        if all(config.get(k) == e[k] for k in other_keys) or not other_keys:
            return True
    return False


def benchmark_function(warmup_fn: Callable, loop_fn: Callable, repeat: int, number: int) -> NDArray:
    """Time a loop of simulation steps and return how long each step took.

    Args:
        warmup_fn: Callable that resets the simulator and warms it up before each repetition.
        loop_fn: Callable that runs the number of steps it is given.
        repeat: Number of repetitions.
        number: Number of steps per repetition.

    Returns:
        Array of per-step wall-clock times (seconds), one entry per repetition.
    """
    timer = timeit.Timer(stmt=partial(loop_fn, number), setup=warmup_fn)
    return np.array(timer.repeat(repeat=repeat, number=1)) / number


def run_sweep(
    configs: list[dict],
    benchmark_fn: Callable[[dict], NDArray],
    on_result: Callable[[dict], None] | None = None,
) -> list[dict]:
    """Iterate pre-generated configs and collect benchmark timings.

    A config that exhausts resources is recorded, and every later config dominated by it is skipped.

    Args:
        configs: Ordered list of config dicts. Each dict is stored in the returned results.
        benchmark_fn: Callable that receives a config dict and returns per-step timing measurements.
            An empty array signals self-detected infeasibility.
        on_result: Called with each result as soon as it is measured.

    Returns:
        List of ``{**config, "timings": list}`` dicts for every config that produced at least one
        timing measurement.
    """
    results = []
    exhausted: list[dict] = []
    for config in configs:
        if _is_dominated(config, exhausted):
            logger.info("Skipping config %s due to prior resource exhaustion.", config)
            continue
        try:
            timings = benchmark_fn(config)
        except WorkerError as e:
            logger.warning(f"Error for config {config}, skipping dominated configs.")
            logger.warning(f"Exception: {e}")
            exhausted.append(config)
            continue
        if len(timings) > 0:
            result = {**config, "timings": timings.tolist()}
            results.append(result)
            if on_result is not None:
                on_result(result)
        else:
            exhausted.append(config)
    return results


def metadata() -> dict[str, str]:
    """Describe the machine and the checkout a measurement is taken on."""
    nvidia_smi = shutil.which("nvidia-smi")
    gpu = "none"
    if nvidia_smi is not None:
        query = [nvidia_smi, "--query-gpu=name", "--format=csv,noheader"]
        gpu = subprocess.run(query, capture_output=True, text=True, check=True).stdout
    cpu = [ln for ln in Path("/proc/cpuinfo").read_text().splitlines() if "model name" in ln]
    commit, dirty = "unversioned", "True"
    git_executable = shutil.which("git")
    if git_executable is not None:
        git = [git_executable, "-C", str(paths.REPO_ROOT)]
        top_level = subprocess.run(
            [*git, "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=False
        )
        if (
            top_level.returncode == 0
            and Path(top_level.stdout.strip()).resolve() == paths.REPO_ROOT
        ):
            commit = subprocess.run(
                [*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
            ).stdout.strip()
            status = subprocess.run(
                [*git, "status", "--porcelain"], capture_output=True, text=True, check=True
            )
            dirty = str(bool(status.stdout.strip()))
    return {
        "host": socket.gethostname(),
        "cpu": cpu[0].split(":", 1)[1].strip(),
        "gpu": gpu.splitlines()[0].strip(),
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": commit,
        "dirty": dirty,
    }


def load_result_keys(name: str, results_dir: Path = RESULTS_DIR) -> set[frozenset]:
    """Return the set of existing result keys from a parquet file for skip-checking.

    Args:
        name: Benchmark name used as the parquet filename stem.
        results_dir: Directory holding the parquet files.

    Returns:
        One frozenset of ``(column, value)`` pairs per existing row, covering all non-timings
        columns so the key is independent of column order. Empty if the file does not exist.
    """
    path = results_dir / f"{name}.parquet"
    if not path.exists():
        return set()
    df = pl.read_parquet(path)
    key_cols = [c for c in df.columns if c != "timings" and c not in METADATA]
    return {frozenset((k, row[k]) for k in key_cols) for row in df.iter_rows(named=True)}


def save_results(df: pl.DataFrame, name: str, results_dir: Path = RESULTS_DIR):
    """Save benchmark results, replacing rows whose keys match and keeping all others.

    Args:
        df: DataFrame with benchmark results.
        name: Benchmark name used as the parquet filename stem.
        results_dir: Directory the parquet files are written to.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{name}.parquet"
    df = df.with_columns(**{k: pl.lit(v) for k, v in metadata().items()})
    if path.exists():
        key_cols = [c for c in df.columns if c != "timings" and c not in METADATA]
        existing = pl.read_parquet(path)
        existing = existing.join(df.select(key_cols), on=key_cols, how="anti")
        df = pl.concat([existing, df])
    df.write_parquet(path)


def checkpoint(name: str, results_dir: Path, result: dict):
    """Checkpoint a sweep result."""
    save_results(pl.DataFrame([result]), name, results_dir)


def check_parameters(n_worlds: int, n_drones: int, framework: str) -> bool:
    """Check if the given combination of parameters is valid for benchmarking."""
    if framework == "gym_pybullet_drones":
        return False
    elif framework == "crazyflow":
        return False
    elif framework in ("crazyflow_rotor_level", "gym_pybullet_drones_rotor_level"):
        return n_drones > 1  # Compared against l2f and Flightmare, which are single-drone only
    elif framework == "diff-aero":
        return False
    elif framework in ("aerial_gym", "aerial_gym_rotor_level"):
        # Aerial-gym uses single drone per environment
        return n_drones > 1
    elif framework == "flightmare":
        # Flightmare uses a single drone per environment
        return n_drones > 1
    elif framework == "l2f":
        # l2f uses a single drone per environment and compiles vectorized environments up to 8192
        return n_drones > 1 or n_worlds > 8192
    elif framework == "crazyflow_grad":
        return False
    elif framework == "diff_aero_grad":
        return False
    raise ValueError(f"Unknown framework: {framework}")
