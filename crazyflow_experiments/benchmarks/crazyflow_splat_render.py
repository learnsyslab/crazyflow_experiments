"""Benchmark Crazyflow gaussian-splat rendering throughput.

Run with
    pixi run -e vision python crazyflow_experiments/benchmarks/crazyflow_splat_render.py
    pixi run -e vision python crazyflow_experiments/benchmarks/crazyflow_splat_render.py --sensor
"""

from __future__ import annotations

import logging
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import fire
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import splax
from crazyflow.sim import Sim
from crazyflow.sim.sensors.splat import build_render_splat_fn, build_render_splat_rgbd_fn
from crazyflow.sim.splat import attach_splats

from crazyflow_experiments.benchmarks.common import (
    NUMBER,
    REPEAT,
    RESOLUTIONS,
    RESULTS_DIR,
    WORLD_RANGE,
    benchmark_function,
    checkpoint,
    load_result_keys,
    run_isolated,
    run_sweep,
    save_results,
)

if TYPE_CHECKING:
    from typing import Callable

    from crazyflow.sim.data import SimData
    from jax import Array
    from numpy.typing import NDArray


logger = logging.getLogger(__name__)

MAX_WORLDS = 1_000_000

# The drone hall scene used by the Crazyflow splat examples, aligned to the simulation world frame.
SPLAT_URL = "https://huggingface.co/datasets/amacati/splats/resolve/main/robot_hall.ply"

# Height of the drone above the MuJoCo ground plane after reset (metres).
DRONE_HEIGHT = 1.0

MODES = ("rgb", "rgbd")


def _build_sim(
    n_worlds: int, resolution: tuple[int, int], mode: str
) -> tuple[Sim, Callable[[SimData], Array]]:
    """Build a splat sim of the drone hall and the render function for the requested mode.

    Args:
        n_worlds: Number of parallel simulation worlds.
        resolution: Render resolution as (width, height).
        mode: Either ``"rgb"`` or ``"rgbd"``.

    Returns:
        The simulation with its drones lifted to the hover height, and the baked render function.
    """
    assert mode in MODES, f"Unknown render mode: {mode}"
    sim = Sim(n_worlds=n_worlds, n_drones=1, device="gpu")
    attach_splats(sim, scene=splax.io.fetch(SPLAT_URL))
    pos = sim.data.states.pos.at[:, :, 2].set(DRONE_HEIGHT)
    sim.data = sim.data.replace(states=sim.data.states.replace(pos=pos))
    build_fn = build_render_splat_fn if mode == "rgb" else build_render_splat_rgbd_fn
    return sim, build_fn(sim, resolution=resolution)


def benchmark_render(
    n_worlds: int,
    resolution: tuple[int, int],
    mode: str,
    repeat: int = REPEAT,
    number: int = NUMBER,
) -> NDArray:
    """Benchmark splat rendering for the given world count, resolution, and mode.

    Args:
        n_worlds: Number of parallel simulation worlds.
        resolution: Render resolution as (width, height).
        mode: Either ``"rgb"`` or ``"rgbd"``.
        repeat: Number of timing repetitions.
        number: Number of renders per repetition.

    Returns:
        Array of per-execution wall-clock times (seconds). Empty if aborted.
    """
    logger.info(f"Benchmarking {mode} splats with {n_worlds} worlds, resolution {resolution}")
    sim, render_fn = _build_sim(n_worlds, resolution, mode)
    sim_data = sim.data

    @jax.jit
    def fori_loop(sim_data: SimData, num_steps: int) -> Array:
        def single_render(_, carry: tuple[SimData, Array]) -> tuple[SimData, Array]:
            sim_data, acc = carry
            img = render_fn(sim_data)
            # Prevent XLA from optimizing the render call away
            mean = jnp.mean(img)
            scale = jnp.finfo(jnp.float32).eps
            pos = sim_data.states.pos.at[..., 2].add(mean * scale)
            return sim_data.replace(states=sim_data.states.replace(pos=pos)), acc + mean

        _, total = jax.lax.fori_loop(0, num_steps, single_render, (sim_data, jnp.float32(0.0)))
        return total

    def loop_fn(steps: int):
        jax.block_until_ready(fori_loop(sim_data, steps))

    def warmup():
        loop_fn(1)  # Trigger JIT compilation

    return benchmark_function(warmup, loop_fn, repeat, number)


def run_benchmarks(
    world_range: list[int],
    resolutions: list[tuple[int, int]],
    modes: list[str],
    results_dir: Path = RESULTS_DIR,
    repeat: int = REPEAT,
    number: int = NUMBER,
) -> pl.DataFrame:
    """Run all splat render benchmarks and save results to parquet.

    Args:
        world_range: World counts to sweep.
        resolutions: Render resolutions to sweep.
        modes: Render modes to sweep.
        results_dir: Directory the parquet results are read from and written to.
        repeat: Number of timing repetitions.
        number: Number of renders per repetition.

    Returns:
        DataFrame with columns ``n_worlds``, ``resolution``, ``mode``, ``timings``.
    """
    existing_keys = load_result_keys("crazyflow_splat_render", results_dir)
    configs = [
        cfg
        for cfg in (
            {"n_worlds": w, "resolution": f"{r[0]}x{r[1]}", "mode": m}
            for m in modes
            for r in resolutions
            for w in world_range
        )
        if frozenset(cfg.items()) not in existing_keys and not cfg["n_worlds"] > MAX_WORLDS
    ]
    data = run_sweep(
        configs,
        lambda cfg: run_isolated(
            benchmark_render,
            {
                **cfg,
                "resolution": tuple(int(x) for x in cfg["resolution"].split("x")),
                "repeat": repeat,
                "number": number,
            },
        ),
        on_result=partial(checkpoint, "crazyflow_splat_render", results_dir),
    )
    if not data:
        logger.info("No new benchmark data collected.")
        return pl.DataFrame()
    df = pl.DataFrame(data)
    save_results(df, "crazyflow_splat_render", results_dir)
    logger.info("Saved results.")
    return df


def print_summary(df: pl.DataFrame):
    """Print a human-readable benchmark statistics summary.

    Args:
        df: DataFrame with columns ``n_worlds``, ``resolution``, ``mode``, ``timings``.
    """
    summary = df.with_columns(
        [
            (pl.col("n_worlds") / pl.col("timings").list.mean()).alias("fps_mean"),
            (pl.col("n_worlds") / pl.col("timings").list.min()).alias("fps_best"),
            (pl.col("n_worlds") / pl.col("timings").list.max()).alias("fps_worst"),
            (pl.col("n_worlds") / pl.col("timings").list.eval(pl.element()))
            .list.std()
            .alias("fps_std"),
        ]
    ).select(["mode", "resolution", "n_worlds", "fps_mean", "fps_std", "fps_best", "fps_worst"])

    print("\n=== Crazyflow Splat Render Benchmark Summary ===")
    for (mode, resolution), group in summary.group_by(["mode", "resolution"], maintain_order=True):
        print(f"\n[{mode.upper()} | {resolution}]")
        print(
            f"  {'n_worlds':>10}  {'fps_mean':>12}  {'fps_std':>10}"
            f"  {'fps_best':>12}  {'fps_worst':>12}"
        )
        for row in group.sort("n_worlds").iter_rows(named=True):
            print(
                f"  {row['n_worlds']:>10}  {row['fps_mean']:>12.1f}  {row['fps_std']:>10.1f}"
                f"  {row['fps_best']:>12.1f}  {row['fps_worst']:>12.1f}"
            )


def inspect_env(resolution: tuple[int, int], mode: str):
    """Show the splat image a single-world sim renders from the drone camera.

    Args:
        resolution: Render resolution as (width, height).
        mode: Either ``"rgb"`` or ``"rgbd"``.
    """
    sim, render_fn = _build_sim(1, resolution, mode)
    img = np.array(jax.block_until_ready(render_fn(sim.data)))[0, 0]

    logger.info("=== Crazyflow Splat Environment Inspection ===")
    logger.info(f"  n_gaussians : {sim.data.plugins['splats'].means.shape[0]}")
    logger.info(f"  drone height: {DRONE_HEIGHT} m")
    logger.info(f"  resolution  : {resolution[0]}x{resolution[1]}")
    logger.info(f"  image shape : {img.shape}  (expected [H, W, {3 if mode == 'rgb' else 4}])")

    fig, axes = plt.subplots(1, 1 if mode == "rgb" else 2, figsize=(6, 6), squeeze=False)
    axes[0][0].imshow(np.clip(img[..., :3], 0.0, 1.0))
    axes[0][0].set_title(f"Splat RGB ({resolution[0]}x{resolution[1]})")
    axes[0][0].axis("off")
    if mode == "rgbd":
        depth = axes[0][1].imshow(img[..., 3], cmap="viridis")
        fig.colorbar(depth, ax=axes[0][1], label="depth [m]")
        axes[0][1].set_title("Splat depth")
        axes[0][1].axis("off")
    plt.tight_layout()
    plt.show()


def main(
    sensor: bool = False,
    n_worlds: int | None = None,
    resolution: tuple[int, int] | None = None,
    mode: str | None = None,
    results_dir: str | Path = RESULTS_DIR,
    repeat: int = REPEAT,
    number: int = NUMBER,
):
    """Entry point for the Crazyflow splat render benchmark.

    Args:
        sensor: Display the rendered splat image instead of benchmarking.
        n_worlds: Single world count to benchmark, or None to sweep all of WORLD_RANGE.
        resolution: Single (width, height) resolution, or None to sweep all of RESOLUTIONS.
        mode: Single render mode ("rgb" or "rgbd"), or None to sweep both.
        results_dir: Directory the parquet results are read from and written to.
        repeat: Number of timing repetitions.
        number: Number of renders per repetition.
    """
    resolutions = [resolution] if resolution is not None else RESOLUTIONS
    modes = [mode] if mode is not None else list(MODES)
    if sensor:
        inspect_env(resolutions[0], modes[0])
        return
    world_range = [n_worlds] if n_worlds is not None else WORLD_RANGE
    df = run_benchmarks(world_range, resolutions, modes, Path(results_dir), repeat, number)
    print_summary(df)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logger.setLevel(logging.INFO)
    fire.Fire(main)
