"""Benchmark diff-aero camera raycasting / depth-rendering throughput.

The benchmarked operation is ``ObstacleAvoidance.update_sensor_data()``, which calls the ``Camera``
ray-casting sensor and stores the resulting depth image in ``env.sensor_tensor``.

Resolution is fixed to 64 x 64 rays.

Run with
    pixi run -e diff-aero python crazyflow_experiments/benchmarks/diff_aero_render.py
    pixi run -e diff-aero python crazyflow_experiments/benchmarks/diff_aero_render.py --render
"""

import logging
import sys
import tomllib
from functools import partial
from pathlib import Path

diff_aero_path = Path(__file__).parents[2] / "submodules"
assert (diff_aero_path / "diffaero").exists(), f"diff-aero not found below: {diff_aero_path}"
if str(diff_aero_path) not in sys.path:
    sys.path.insert(0, str(diff_aero_path))

import fire  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402
from diffaero.env import build_env  # noqa: E402
from diffaero.utils.randomizer import RandomizerManager  # noqa: E402
from numpy.typing import NDArray  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from crazyflow_experiments.benchmarks.common import (  # noqa: E402
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

logger = logging.getLogger(__name__)

# (width, height) tuples to width = W (horizontal rays), height = H (vertical rays).
# Sensor return shape for each resolution: [n_envs, H, W].
# Ray-casting allocates [n_envs, n_obstacles, H*W] intermediate tensors.
# Cap the world count to avoid OOM. The RuntimeError handler covers edge cases.
MAX_WORLDS = 10_000


def check_parameters(n_worlds: int, resolution: tuple[int, int]) -> bool:
    """Return True if the combination should be skipped due to anticipated resource exhaustion.

    Args:
        n_worlds: Number of parallel simulation worlds.
        resolution: Render resolution as (width, height).

    Returns:
        True if the parameter combination should be skipped.
    """
    return n_worlds > MAX_WORLDS


def create_env_config(
    n_worlds: int, resolution: tuple[int, int], headless: bool = True
) -> DictConfig:
    """Create environment configuration for diff-aero obstacle avoidance with camera sensor.

    Args:
        n_worlds: Number of parallel simulation worlds.
        resolution: Camera resolution as (width, height).
        headless: Whether to run headless (no visual window).

    Returns:
        DictConfig ready for ``build_env``.
    """
    with open(Path(__file__).parent / "diff_aero_render_cfg.toml", "rb") as f:
        cfg = tomllib.load(f)

    cfg["n_envs"] = n_worlds
    cfg["n_agents"] = 1
    cfg["dynamics"]["n_envs"] = n_worlds
    cfg["dynamics"]["n_agents"] = 1
    cfg["dynamics"]["dt"] = cfg["dt"]
    cfg["render"]["n_envs"] = n_worlds
    cfg["render"]["n_agents"] = 1
    cfg["render"]["headless"] = headless
    cfg["render"]["dt"] = cfg["dt"]
    cfg["render"]["env_spacing"] = cfg["length"]["default"]

    # Camera resolution: width = W (columns), height = H (rows).
    # Return shape of depth image: [n_envs, H, W].
    cfg["sensor"]["n_envs"] = n_worlds
    cfg["sensor"]["n_agents"] = 1
    cfg["sensor"]["width"] = resolution[0]
    cfg["sensor"]["height"] = resolution[1]

    return DictConfig(cfg)


def benchmark_render(
    n_worlds: int,
    resolution: tuple[int, int],
    device: str,
    repeat: int = REPEAT,
    number: int = NUMBER,
) -> NDArray:
    """Benchmark diff-aero camera ray casting for the given world count and resolution.

    Args:
        n_worlds: Number of parallel simulation worlds.
        resolution: Camera resolution as (width, height).
        device: PyTorch device string, e.g. ``"cpu"`` or ``"cuda"``.
        repeat: Number of timing repetitions.
        number: Number of renders per repetition.

    Returns:
        Array of per-execution wall-clock times (seconds). Empty if aborted.
    """
    logger.info(f"Benchmarking render on {device} with {n_worlds} worlds, resolution {resolution}")

    env = None
    action = None
    torch_device = None

    @torch.no_grad()
    def loop_fn(steps: int):
        for _ in range(steps):
            env.update_sensor_data()
            if torch_device.type == "cuda":
                torch.cuda.synchronize()

    def warmup():
        nonlocal env, action, torch_device
        torch_device = torch.device(device)
        cfg = create_env_config(n_worlds, resolution)
        # Clear accumulated randomizers between benchmark runs (diffaero class-variable bug).
        RandomizerManager.randomizers.clear()
        env = build_env(cfg, device=torch_device)
        action = torch.zeros(n_worlds, env.action_dim, device=torch_device)
        with torch.no_grad():
            env.reset()
        loop_fn(1)  # Warmup / trigger JIT compilation

    try:
        return benchmark_function(warmup, loop_fn, repeat, number)
    finally:
        if env is not None:
            del env
            if torch_device is not None and torch_device.type == "cuda":
                torch.cuda.empty_cache()


def run_benchmarks(
    devices: list[str],
    world_range: list[int],
    resolutions: list[tuple[int, int]],
    results_dir: Path = RESULTS_DIR,
    repeat: int = REPEAT,
    number: int = NUMBER,
) -> pl.DataFrame:
    """Run all render benchmarks and save results to parquet.

    Args:
        devices: Devices to benchmark.
        world_range: World counts to sweep.
        resolutions: Render resolutions to sweep.
        results_dir: Directory the parquet results are read from and written to.
        repeat: Number of timing repetitions.
        number: Number of renders per repetition.

    Returns:
        DataFrame with columns ``n_worlds``, ``n_drones``, ``resolution``, ``device``, ``timings``.
    """
    existing_keys = load_result_keys("diff_aero_render", results_dir)
    configs = [
        cfg
        for cfg in (
            {"n_worlds": w, "resolution": f"{r[0]}x{r[1]}", "device": dev}
            for dev in devices
            for r in resolutions
            for w in world_range
        )
        if frozenset(cfg.items()) not in existing_keys
        and not check_parameters(cfg["n_worlds"], cfg["resolution"])
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
        on_result=partial(checkpoint, "diff_aero_render", results_dir),
    )
    if not data:
        logger.info("No new benchmark data collected.")
        return pl.DataFrame()
    df = pl.DataFrame(data)
    save_results(df, "diff_aero_render", results_dir)
    logger.info("Saved results.")
    return df


def print_summary(df: pl.DataFrame):
    """Print a human-readable benchmark statistics summary.

    Args:
        df: DataFrame with columns ``n_worlds``, ``resolution``, ``device``, ``timings``.
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
    ).select(["device", "resolution", "n_worlds", "fps_mean", "fps_std", "fps_best", "fps_worst"])

    print("\n=== diff-aero Render Benchmark Summary ===")
    for (device, resolution), group in summary.group_by(
        ["device", "resolution"], maintain_order=True
    ):
        print(f"\n[{device.upper()} | {resolution}]")
        print(
            f"  {'n_worlds':>10}  {'fps_mean':>12}  {'fps_std':>10}"
            f"  {'fps_best':>12}  {'fps_worst':>12}"
        )
        for row in group.sort("n_worlds").iter_rows(named=True):
            print(
                f"  {row['n_worlds']:>10}  {row['fps_mean']:>12.1f}  {row['fps_std']:>10.1f}"
                f"  {row['fps_best']:>12.1f}  {row['fps_worst']:>12.1f}"
            )


def inspect_env(render: bool = False, sensor: bool = False):
    """Log the obstacle breakdown, sensor resolution and depth statistics for a single environment.

    Args:
        render: Whether to open a visual window. Requires the ``diff-aero`` pixi environment with
            Isaac Gym installed.
        sensor: Whether to display the depth image from the camera sensor via matplotlib.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    resolution = (64, 64)  # (width, height) used for inspection
    n_worlds = 1

    cfg = create_env_config(n_worlds, resolution, headless=not render)
    RandomizerManager.randomizers.clear()
    torch_device = torch.device(device)
    env = build_env(cfg, device=torch_device)

    with torch.no_grad():
        env.reset()

    om = env.obstacle_manager
    cam = env.sensor

    logger.info("=== diff-aero Environment Inspection ===")
    logger.info(f"  Device  : {device}")
    logger.info(f"  Env name: {cfg.name}")
    logger.info(
        f"  Sensor  : {cfg.sensor.name}  "
        f"width={cfg.sensor.width} (W)  height={cfg.sensor.height} (H)  "
        f"->  {cfg.sensor.width * cfg.sensor.height} rays per env"
    )
    logger.info(f"  max_dist: {cfg.sensor.max_dist} m")

    logger.info("--- Obstacle Manager ---")
    logger.info(f"  n_spheres  : {om.n_spheres}")
    logger.info(f"  n_cubes    : {om.n_cubes}  (includes walls={om.walls}, ceiling={om.ceiling})")
    logger.info(f"  n_obstacles: {om.n_obstacles}  (spheres + cubes total)")
    logger.info(f"  walls      : {om.walls}")
    logger.info(f"  ceiling    : {om.ceiling}")

    # Call the sensor directly to capture the full return tuple.
    # z_ground_plane=None disables ground-plane ray-clipping for this inspection call.
    with torch.no_grad():
        depth, contact_points = cam(
            obstacle_manager=om, pos=env.p, quat_xyzw=env.q, z_ground_plane=None
        )

    logger.info("--- Sensor Output Shapes ---")
    logger.info(f"  depth          : {tuple(depth.shape)}  dtype={depth.dtype}")
    logger.info(f"  contact_points : {tuple(contact_points.shape)}  dtype={contact_points.dtype}")
    logger.info(
        f"  Expected depth shape : [n_envs={n_worlds}, H={cfg.sensor.height}, W={cfg.sensor.width}]"
    )
    logger.info(
        f"  Expected contact_points shape : "
        f"[n_envs={n_worlds}, H*W={cfg.sensor.height * cfg.sensor.width}, 3]"
    )
    logger.info(
        f"  depth  min={depth.min().item():.4f}  max={depth.max().item():.4f}"
        f"  (0 = no obstacle within max_dist, 1 = obstacle at sensor origin)"
    )

    if sensor:
        depth_img = np.array(depth[0].cpu())  # [H, W]
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(depth_img, cmap="viridis", vmin=0.0, vmax=1.0)
        fig.colorbar(im, ax=ax, label="normalised depth [0=far, 1=close]")
        ax.set_title(f"diff-aero camera depth ({cfg.sensor.width}x{cfg.sensor.height})")
        ax.axis("off")
        plt.tight_layout()
        plt.show()

    if render:
        logger.info("--- Rendering (Ctrl+C to stop) ---")
        action = torch.zeros(n_worlds, env.action_dim, device=torch_device)
        with torch.no_grad():
            while True:
                env.step(action)


def main(
    render: bool = False,
    sensor: bool = False,
    device: str | None = None,
    n_worlds: int | None = None,
    resolution: tuple[int, int] | None = None,
    results_dir: str | Path = RESULTS_DIR,
    repeat: int = REPEAT,
    number: int = NUMBER,
):
    """Main entry point for the diff-aero render benchmark script.

    Args:
        render: Whether to open a visual window for environment inspection instead of running
            benchmarks. Requires the ``diff-aero`` pixi environment with Isaac Gym installed.
        sensor: Whether to display the depth image from the camera sensor via matplotlib instead of
            running benchmarks.
        device: Device to benchmark, or None to sweep the CPU and, when present, CUDA.
        n_worlds: Single world count to benchmark, or None to sweep all of WORLD_RANGE.
        resolution: Single (width, height) resolution, or None to sweep all of RESOLUTIONS.
        results_dir: Directory the parquet results are read from and written to.
        repeat: Number of timing repetitions.
        number: Number of renders per repetition.
    """
    if render or sensor:
        inspect_env(render=render, sensor=sensor)
        return
    if device is not None:
        devices = [device]
    else:
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    world_range = [n_worlds] if n_worlds is not None else WORLD_RANGE
    resolutions = [resolution] if resolution is not None else RESOLUTIONS
    df = run_benchmarks(devices, world_range, resolutions, Path(results_dir), repeat, number)
    print_summary(df)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logger.setLevel(logging.INFO)

    fire.Fire(main)
