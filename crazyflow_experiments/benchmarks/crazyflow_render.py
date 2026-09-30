"""Benchmark Crazyflow raycasting / depth-rendering throughput.

Obstacles are loaded into the MuJoCo spec before building the MJX model so that scene complexity
matches the diff-aero obstacle_avoidance benchmark.

Run with
    pixi run -e crazyflow python crazyflow_experiments/benchmarks/crazyflow_render.py
    pixi run -e crazyflow python crazyflow_experiments/benchmarks/crazyflow_render.py --render
    pixi run -e crazyflow python crazyflow_experiments/benchmarks/crazyflow_render.py --sensor
"""

from __future__ import annotations

import logging
import tomllib
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import fire
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import polars as pl
from crazyflow.sim import Sim
from crazyflow.sim.sensors.depth import _camera_rays, _render_rays, build_render_depth_fn
from crazyflow.sim.sim import sync_sim2mjx
from scipy.spatial.transform import Rotation

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
    from crazyflow.sim.sim import SimData
    from jax import Array
    from numpy.typing import NDArray


logger = logging.getLogger(__name__)

MAX_WORLDS = 1_000_000

# All obstacle geometry and placement parameters are read from the diff-aero configuration file
# rather than being declared a second time here.  This enforces a *single source of truth* for
# every shared benchmark parameter.
# We intentionally do *not* create a separate crazyflow_render_cfg.toml: that file would be a
# near-duplicate of diff_aero_render_cfg.toml and would require manual synchronisation, which
# is error-prone and adds unnecessary maintenance overhead.
with open(Path(__file__).parent / "diff_aero_render_cfg.toml", "rb") as _f:
    _SCENE_CFG = tomllib.load(_f)

_OBS_CFG = _SCENE_CFG["obstacles"]

_SPHERE_XML = """\
<mujoco>
  <worldbody>
    <body name="obstacle">
      <geom type="sphere" size="{radius:.4f}" rgba="0.8 0.3 0.1 1" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>"""

_BOX_XML = """\
<mujoco>
  <worldbody>
    <body name="obstacle">
      <geom type="box" size="{hx:.4f} {hy:.4f} {hz:.4f}" rgba="0.1 0.3 0.8 1"
            contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>"""

# Height of the drone above the MuJoCo ground plane after reset (metres).
DRONE_HEIGHT = 1.0


def add_obstacles(sim: Sim, seed: int = 0):
    """Add random sphere and box obstacles to the sim spec and rebuild the MJX model.

    Counts, sizes and bounds match the diff-aero render benchmark, and every obstacle is placed
    fully above the ground plane.

    Args:
        sim: Crazyflow Sim instance whose spec to extend.
        seed: RNG seed for reproducible obstacle placement and sizing.
    """
    n_obstacles: int = _SCENE_CFG["n_obstacles"]
    sphere_pct: float = _OBS_CFG["sphere_percentage"]
    n_spheres = int(n_obstacles * sphere_pct)
    n_cubes = n_obstacles - n_spheres

    scale = 1.0
    sphere_r_min = _OBS_CFG["sphere_radius_range"][0] * scale
    sphere_r_max = _OBS_CFG["sphere_radius_range"][1] * scale
    cube_lw_min = _OBS_CFG["cube_lw_range"][0] * scale
    cube_lw_max = _OBS_CFG["cube_lw_range"][1] * scale
    cube_h_min = _OBS_CFG["cube_h_range"][0] * scale
    cube_h_max = _OBS_CFG["cube_h_range"][1] * scale

    height_scale: float = _SCENE_CFG["height_scale"]
    L: float = _SCENE_CFG["length"]["default"]
    env_half_xy: float = 1.0  # place obstacles close to the drone so they appear in the depth image
    env_z_max: float = height_scale * L  # ceiling height above ground

    randomize_cube_pose: bool = _OBS_CFG["randomize_cube_pose"]
    cube_rp_range_deg: float = _OBS_CFG["cube_roll_pitch_range"]

    rng = np.random.default_rng(seed)
    frame = sim.spec.worldbody.add_frame()

    for i in range(n_spheres):
        radius = rng.uniform(sphere_r_min, sphere_r_max)
        # Place centre so the lowest point of the sphere is at or above z = 0.
        z = rng.uniform(radius, max(radius + 0.1, env_z_max - radius))
        xy = rng.uniform(-env_half_xy, env_half_xy, size=2)
        obstacle_spec = mujoco.MjSpec.from_string(_SPHERE_XML.format(radius=radius))
        attached = frame.attach_body(obstacle_spec.body("obstacle"), "", f":sphere{i}")
        attached.pos = [float(xy[0]), float(xy[1]), float(z)]

    for i in range(n_cubes):
        lw = rng.uniform(cube_lw_min, cube_lw_max)
        h = rng.uniform(cube_h_min, cube_h_max)
        hxy, hz = lw / 2, h / 2
        # Place centre so the bottom face of the box is at or above z = 0.
        z = rng.uniform(hz, max(hz + 0.1, env_z_max - hz))
        xy = rng.uniform(-env_half_xy, env_half_xy, size=2)
        obstacle_spec = mujoco.MjSpec.from_string(_BOX_XML.format(hx=hxy, hy=hxy, hz=hz))
        attached = frame.attach_body(obstacle_spec.body("obstacle"), "", f":box{i}")
        attached.pos = [float(xy[0]), float(xy[1]), float(z)]
        if randomize_cube_pose:
            roll = rng.uniform(-cube_rp_range_deg, cube_rp_range_deg)
            pitch = rng.uniform(-cube_rp_range_deg, cube_rp_range_deg)
            tilt = Rotation.from_euler("zyx", [0.0, pitch, roll], degrees=True)
            attached.quat = tilt.as_quat(scalar_first=True)  # MuJoCo wxyz

    sim.build_mjx()


def _set_drone_height(sim: Sim, z: float = DRONE_HEIGHT):
    """Set the z-position of all drones in all worlds after reset.

    Args:
        sim: Crazyflow Sim instance with data already reset.
        z: Target height above the ground plane in metres.
    """
    pos = sim.data.states.pos.at[:, :, 2].set(z)
    sim.data = sim.data.replace(states=sim.data.states.replace(pos=pos))


def check_parameters(n_worlds: int, resolution: tuple[int, int]) -> bool:
    """Return True if the combination should be skipped due to anticipated resource exhaustion.

    Args:
        n_worlds: Number of parallel simulation worlds.
        resolution: Render resolution as (width, height).

    Returns:
        True if the parameter combination should be skipped.
    """
    return n_worlds > MAX_WORLDS


def benchmark_render(
    n_worlds: int,
    resolution: tuple[int, int],
    device: str,
    repeat: int = REPEAT,
    number: int = NUMBER,
) -> NDArray:
    """Benchmark depth rendering for the given world count, resolution, and device.

    Each iteration perturbs the drone position by a negligible amount derived from the previous
    depth sum, so XLA cannot hoist the render out of the timing loop.

    Args:
        n_worlds: Number of parallel simulation worlds.
        resolution: Render resolution as (width, height).
        device: JAX device string, e.g. ``"cpu"`` or ``"gpu"``.
        repeat: Number of timing repetitions.
        number: Number of renders per repetition.

    Returns:
        Array of per-execution wall-clock times (seconds). Empty if aborted.
    """
    logger.info(f"Benchmarking render on {device} with {n_worlds} worlds, resolution {resolution}")
    # The fused drone body carries no cameras, so the full model is required for raycasting.
    sim = Sim(n_worlds=n_worlds, n_drones=1, device=device)
    add_obstacles(sim)

    rays = _camera_rays(resolution=resolution, fov_y=jnp.pi / 4)[None, ...]
    ray_fn = jax.jit(
        partial(
            _render_rays,
            mjx_model=sim.mjx_model,
            camera=0,
            geomgroup=(1, 1, 0, 0, 1, 1, 1, 1),
            rays=rays,
        )
    )
    _set_drone_height(sim)
    sim_data = sim.data

    @jax.jit
    def fori_loop(sim_data: SimData, num_steps: int) -> Array:
        def single_render(_, carry: tuple[SimData, Array]) -> tuple[SimData, Array]:
            sim_data, acc = carry
            # Sync MuJoCo state (kinematics, camlight, collision), same cost as real pipeline.
            _, mjx_data = sync_sim2mjx(sim_data, sim.mjx_data, sim.mjx_model)
            depth = ray_fn(mjx_data=mjx_data)
            # Accumulate depth sum. The data-dependent acc feeds back into the carry so XLA
            # cannot hoist the sync+render out of the loop via LICM.
            new_acc = acc + jnp.sum(depth)
            # Negligible drone-position perturbation (sub-nanometre) to break loop invariance.
            scale = jnp.finfo(jnp.float32).eps
            new_sim_data = sim_data.replace(
                states=sim_data.states.replace(
                    pos=sim_data.states.pos.at[..., 2].add(new_acc * scale)
                ),
                core=sim_data.core.replace(mjx_synced=sim_data.core.mjx_synced.at[...].set(False)),
            )
            return (new_sim_data, new_acc)

        _, total = jax.lax.fori_loop(0, num_steps, single_render, (sim_data, jnp.float32(0.0)))
        return total

    def loop_fn(steps: int):
        jax.block_until_ready(fori_loop(sim_data, steps))

    def warmup():
        _set_drone_height(sim)
        loop_fn(1)  # Trigger JIT compilation

    return benchmark_function(warmup, loop_fn, repeat, number)


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
        DataFrame with columns ``n_worlds``, ``resolution``, ``device``, ``timings``.
    """
    existing_keys = load_result_keys("crazyflow_render", results_dir)
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
        on_result=partial(checkpoint, "crazyflow_render", results_dir),
    )
    if not data:
        logger.info("No new benchmark data collected.")
        return pl.DataFrame()
    df = pl.DataFrame(data)
    save_results(df, "crazyflow_render", results_dir)
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

    print("\n=== Crazyflow Render Benchmark Summary ===")
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
    """Log obstacle counts and sensor output shapes for a single-world sim.

    Args:
        render: Open the MuJoCo 3D viewer.
        sensor: Display the depth image from the camera raycasting function via matplotlib.
    """
    sim = Sim(n_worlds=1, n_drones=1, device="cpu")
    add_obstacles(sim)
    resolution = RESOLUTIONS[0]
    render_fn = build_render_depth_fn(sim.mjx_model, camera=0, resolution=resolution)
    sim.reset()
    _set_drone_height(sim)

    depth = np.array(jax.block_until_ready(render_fn(sim)))  # [1, H, W]

    n_obstacles = _SCENE_CFG["n_obstacles"]
    n_spheres = int(n_obstacles * _OBS_CFG["sphere_percentage"])
    n_cubes = n_obstacles - n_spheres

    logger.info("=== Crazyflow Environment Inspection ===")
    logger.info(f"  n_spheres : {n_spheres}")
    logger.info(f"  n_cubes   : {n_cubes}")
    logger.info(f"  n_geoms in mjx_model: {sim.mjx_model.ngeom}")
    logger.info(f"  drone height: {DRONE_HEIGHT} m")
    logger.info(
        f"  Sensor resolution : {resolution[0]}x{resolution[1]}"
        f" = {resolution[0] * resolution[1]} rays"
    )
    logger.info(
        f"  depth shape : {depth.shape}"
        f"  (expected [n_worlds=1, H={resolution[1]}, W={resolution[0]}])"
    )
    logger.info(f"  depth dtype : {depth.dtype}")
    logger.info(f"  depth min={depth.min():.4f}  max={depth.max():.4f}  (metres, -1 = no hit)")

    if sensor:
        depth_img = depth[0]  # [H, W]
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(depth_img, cmap="viridis")
        fig.colorbar(im, ax=ax, label="depth [m]")
        ax.set_title(f"Crazyflow camera depth ({resolution[0]}x{resolution[1]})")
        ax.axis("off")
        plt.tight_layout()
        plt.show()

    if render:
        logger.info("--- Rendering (Ctrl+C to stop) ---")
        while True:
            sim.render(mode="human", world=0)


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
    """Entry point for the Crazyflow render benchmark.

    Args:
        render: Open the MuJoCo 3D viewer to inspect the loaded scene instead of benchmarking.
        sensor: Display the depth image from the camera raycasting function instead of benchmarking.
        device: JAX device to benchmark ("cpu" or "gpu"), or None to sweep both.
        n_worlds: Single world count to benchmark, or None to sweep all of WORLD_RANGE.
        resolution: Single (width, height) resolution, or None to sweep all of RESOLUTIONS.
        results_dir: Directory the parquet results are read from and written to.
        repeat: Number of timing repetitions.
        number: Number of renders per repetition.
    """
    if render or sensor:
        inspect_env(render=render, sensor=sensor)
        return
    devices = [device] if device is not None else ["cpu", "gpu"]
    world_range = [n_worlds] if n_worlds is not None else WORLD_RANGE
    resolutions = [resolution] if resolution is not None else RESOLUTIONS
    df = run_benchmarks(devices, world_range, resolutions, Path(results_dir), repeat, number)
    print_summary(df)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logger.setLevel(logging.INFO)
    fire.Fire(main)
