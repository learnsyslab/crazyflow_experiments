"""Report headline throughput numbers from the benchmark results.

Run with
    pixi run -e crazyflow python crazyflow_experiments/benchmarks/stats.py
"""

import logging

import numpy as np
import polars as pl
from scipy.stats import ttest_ind

from crazyflow_experiments.utils import paths

logger = logging.getLogger(__name__)

_RESULTS_DIR = paths.RESULTS_DIR / "benchmark"


def with_fps(df: pl.DataFrame) -> pl.DataFrame:
    """Add fps column as n_worlds / mean(timings)."""
    return df.with_columns((pl.col("n_worlds") / pl.col("timings").list.mean()).alias("fps"))


def analyze_crazyflow(df_cf: pl.DataFrame):
    """Log the world count and throughput ceilings reached by Crazyflow."""
    max_worlds = df_cf["n_worlds"].max()
    fps_at_max = df_cf.filter(pl.col("n_worlds") == max_worlds)["fps"].max()
    logger.info("=== Crazyflow ===")
    logger.info(f"Max n_worlds:            {max_worlds:>12,}")
    logger.info(f"FPS at max n_worlds:     {fps_at_max:>12,.0f}")
    logger.info(f"Max FPS overall:         {df_cf['fps'].max():>12,.0f}")


def analyze_vs_gpd(df_cf: pl.DataFrame, df_gpd: pl.DataFrame):
    """Log per-world-count throughput ratios of Crazyflow over gym-pybullet-drones."""
    logger.info("")
    logger.info("=== Crazyflow vs gym-pybullet-drones (CPU, n_drones=1) ===")
    logger.info(f"{'n_worlds':>10}  {'GPD FPS':>12}  {'CF FPS':>12}  {'Ratio':>8}")
    for n in sorted(df_gpd["n_worlds"].unique().to_list()):
        gpd_fps = df_gpd.filter((pl.col("device") == "cpu") & (pl.col("n_worlds") == n))["fps"][0]
        cf_fps = df_cf.filter((pl.col("device") == "cpu") & (pl.col("n_worlds") == n))["fps"][0]
        logger.info(f"{n:>10,}  {gpd_fps:>12,.0f}  {cf_fps:>12,.0f}  {cf_fps / gpd_fps:>7.1f}x")

    max_worlds_gpd = df_gpd["n_worlds"].max()
    max_worlds_cf = df_cf["n_worlds"].max()
    logger.info("")
    logger.info(f"Max n_worlds GPD:        {max_worlds_gpd:>12,}")
    logger.info(f"Max n_worlds CF (GPU):   {max_worlds_cf:>12,}")
    logger.info(f"n_worlds ratio (CF/GPD): {max_worlds_cf / max_worlds_gpd:>11.1f}x")


def analyze_gradients(df_grad: pl.DataFrame):
    """Log the world count and throughput ceilings reached by Crazyflow gradients."""
    max_worlds = df_grad["n_worlds"].max()
    logger.info("")
    logger.info("=== Crazyflow gradients (GPU) ===")
    logger.info(f"Max n_worlds:            {max_worlds:>12,}")
    logger.info(
        f"FPS at max n_worlds:     {df_grad.filter(pl.col('n_worlds') == max_worlds)['fps'][0]:>12,.0f}"
    )
    logger.info(f"Max FPS overall:         {df_grad['fps'].max():>12,.0f}")


def analyze_speedup(
    baseline: str, df_cf: pl.DataFrame, df_base: pl.DataFrame, alpha: float = 0.05
) -> float:
    """Test in every configuration both reached whether Crazyflow outpaces a baseline."""
    key = ["n_worlds", "n_drones"]
    pairs = df_cf.select(*key, "timings").join(
        df_base.select(*key, "timings"), on=key, suffix="_base"
    )
    results = []
    for row in pairs.iter_rows(named=True):
        log_cf, log_base = (
            np.log(row["n_worlds"] / np.asarray(row[column]))
            for column in ("timings", "timings_base")
        )
        result = ttest_ind(log_cf, log_base, equal_var=False, alternative="greater")
        bound = np.exp(result.confidence_interval(1 - alpha).low)
        results.append({**{k: row[k] for k in key}, "p": result.pvalue, "bound": bound})
    results = pl.DataFrame(results)
    weakest_p = results.row(results["p"].arg_max(), named=True)
    weakest_bound = results.row(results["bound"].arg_min(), named=True)
    logger.info("")
    logger.info(f"=== Crazyflow vs {baseline} (one-sided log Welch t-tests, IUT) ===")
    logger.info(
        f"{results.height} configurations, drones {sorted(results['n_drones'].unique())}, "
        f"{df_cf.join(df_base, on=key, how='anti').height} reached by Crazyflow alone"
    )
    logger.info(
        f"IUT p = {weakest_p['p']:.1e}, "
        f"largest at {weakest_p['n_worlds']:,} worlds x {weakest_p['n_drones']} drones"
    )
    logger.info(
        f"Faster everywhere by at least {weakest_bound['bound']:.1f}x ({1 - alpha:.0%} bound), "
        f"tightest at {weakest_bound['n_worlds']:,} worlds x {weakest_bound['n_drones']} drones"
    )
    return weakest_p["p"]


def analyze_annotations(
    frames: dict[str, pl.DataFrame], claims: tuple[int, ...] = (25, 33, 18), n_worlds: int = 1024
):
    """Test the speedups over DiffAero claimed at the benchmark's anchor world count.

    Each claim states that Crazyflow outpaces DiffAero at the anchor world count of the benchmark
    figure by at least the claimed factor. A one-sided log Welch t-test against the shifted null of
    a speedup no larger than that factor tests the claim, and the Holm correction over the claims
    controls the family-wise error rate of reporting them together.

    Args:
        frames: Raw benchmark frames keyed by result name.
        claims: Claimed speedups of CPU dynamics, GPU dynamics and gradients.
        n_worlds: World count at which the claims are evaluated.
    """
    panels = (
        ("dynamics CPU", "crazyflow", "cpu", "diff_aero", "cpu"),
        ("dynamics GPU", "crazyflow", "gpu", "diff_aero", "cuda"),
        ("gradient GPU", "crazyflow_grad", "gpu", "diff_aero_grad", "cuda"),
    )
    rows, p_values = [], []
    for (panel, cf, cf_device, base, base_device), claim in zip(panels, claims, strict=True):
        timings_cf, timings_base = (
            frames[name]
            .filter(
                (pl.col("device") == device)
                & (pl.col("n_drones") == 1)
                & (pl.col("n_worlds") == n_worlds)
            )["timings"][0]
            .to_numpy()
            for name, device in ((cf, cf_device), (base, base_device))
        )
        log_cf, log_base = np.log(n_worlds / timings_cf), np.log(n_worlds / timings_base)
        speedup = np.exp(log_cf.mean() - log_base.mean())
        label = round(timings_base.mean() / timings_cf.mean())
        rows.append((panel, n_worlds, label, claim, speedup))
        p_values.append(
            [
                ttest_ind(
                    log_cf - np.log(c), log_base, equal_var=False, alternative="greater"
                ).pvalue
                for c in (1, claim)
            ]
        )
    # Holm scales the i-th smallest of k p-values by k - i and keeps the running maximum
    p_values = np.asarray(p_values)
    k = len(p_values)
    order = np.argsort(p_values, axis=0)
    ranked = np.take_along_axis(p_values, order, axis=0) * (k - np.arange(k))[:, None]
    holm = np.empty_like(p_values)
    np.put_along_axis(holm, order, np.minimum(1.0, np.maximum.accumulate(ranked, axis=0)), axis=0)
    logger.info("")
    logger.info(f"=== Claimed speedups over DiffAero (Holm over {k} claims) ===")
    logger.info(
        f"{'Panel':<13}  {'n_worlds':>8}  {'Label':>6}  {'Claim':>6}  {'Speedup':>8}  "
        f"{'p > 1':>9}  {'p > claim':>9}"
    )
    for (panel, n_worlds, label, claim, speedup), (p_faster, p_claim) in zip(rows, holm):
        logger.info(
            f"{panel:<13}  {n_worlds:>8,}  {label:>5}x  {claim:>5}x  {speedup:>7.2f}x  "
            f"{p_faster:>9.1e}  {p_claim:>9.2g}"
        )


def _log_ratios(baseline: str, df_baseline: pl.DataFrame, df_cf: pl.DataFrame):
    """Log Crazyflow throughput against a baseline for every world count both reached."""
    logger.info(f"{'n_worlds':>10}  {baseline + ' FPS':>12}  {'CF FPS':>12}  {'Ratio':>8}")
    for n in sorted(set(df_baseline["n_worlds"]) & set(df_cf["n_worlds"])):
        baseline_fps = df_baseline.filter(pl.col("n_worlds") == n)["fps"][0]
        cf_fps = df_cf.filter(pl.col("n_worlds") == n)["fps"][0]
        logger.info(
            f"{n:>10,}  {baseline_fps:>12,.0f}  {cf_fps:>12,.0f}  {cf_fps / baseline_fps:>7.1f}x"
        )


def analyze_rendering(df_cf_render: pl.DataFrame, df_da_render: pl.DataFrame):
    """Log per-world-count depth render throughput ratios of Crazyflow over diff-aero."""
    cf = df_cf_render.filter(pl.col("device") == "gpu")
    da = df_da_render.filter(pl.col("device") == "cuda")
    logger.info("")
    logger.info("=== Depth rendering: Crazyflow vs diff-aero (64x64, GPU) ===")
    _log_ratios("DA", da, cf)

    da_fps = da.filter(pl.col("n_worlds") == 1024)["fps"][0]
    cf_fps = cf.filter(pl.col("n_worlds") == 1024)["fps"][0]
    logger.info("")
    logger.info(
        f"At 1k drones: DA {da_fps:,.0f} FPS, CF {cf_fps:,.0f} FPS, ratio {cf_fps / da_fps:.1f}x"
    )


def analyze_splat_vs_unity(df_splat: pl.DataFrame, df_fm_render: pl.DataFrame):
    """Log per-world-count throughput ratios of Crazyflow splats over Flightmare Unity per mode."""
    for mode in ("rgb", "rgbd"):
        cf = df_splat.filter(pl.col("mode") == mode)
        fm = df_fm_render.filter(pl.col("mode") == mode)
        logger.info("")
        logger.info(
            f"=== {mode.upper()} rendering: Crazyflow splats vs Flightmare Unity (64x64) ==="
        )
        _log_ratios("FM", fm, cf)
        logger.info("")
        for name, df in (("FM", fm), ("CF", cf)):
            max_worlds = df["n_worlds"].max()
            fps = df.filter(pl.col("n_worlds") == max_worlds)["fps"][0]
            logger.info(f"Max n_worlds {name}: {max_worlds:>12,}  FPS at max: {fps:>12,.0f}")


def main():
    """Load every benchmark parquet and log the headline comparisons."""
    df_cf = with_fps(
        pl.read_parquet(_RESULTS_DIR / "crazyflow.parquet").filter(pl.col("n_drones") == 1)
    )
    df_gpd = with_fps(
        pl.read_parquet(_RESULTS_DIR / "gym_pybullet_drones.parquet").filter(
            pl.col("n_drones") == 1
        )
    )
    df_grad = with_fps(
        pl.read_parquet(_RESULTS_DIR / "crazyflow_grad.parquet").filter(pl.col("n_drones") == 1)
    )

    analyze_crazyflow(df_cf)
    analyze_vs_gpd(df_cf, df_gpd)
    analyze_gradients(df_grad)
    frames = {
        name: pl.read_parquet(_RESULTS_DIR / f"{name}.parquet")
        for name in (
            "crazyflow",
            "crazyflow_grad",
            "diff_aero",
            "diff_aero_grad",
            "aerial_gym",
            "gym_pybullet_drones",
        )
    }
    # the comparisons of panels A to D, each on the device both simulators ran on
    comparisons = (
        ("DiffAero, dynamics CPU", "crazyflow", "cpu", "diff_aero", "cpu"),
        ("Aerial Gym, dynamics CPU", "crazyflow", "cpu", "aerial_gym", "cpu"),
        ("GPD, dynamics CPU", "crazyflow", "cpu", "gym_pybullet_drones", "cpu"),
        ("DiffAero, dynamics GPU", "crazyflow", "gpu", "diff_aero", "cuda"),
        ("Aerial Gym, dynamics GPU", "crazyflow", "gpu", "aerial_gym", "cuda"),
        ("DiffAero, gradient GPU", "crazyflow_grad", "gpu", "diff_aero_grad", "cuda"),
    )
    p_values = [
        analyze_speedup(
            baseline,
            frames[cf].filter(pl.col("device") == cf_device),
            frames[base].filter(pl.col("device") == base_device),
        )
        for baseline, cf, cf_device, base, base_device in comparisons
    ]
    logger.info("")
    logger.info(f"IUT over all comparisons of panels A to D: p = {max(p_values):.1e}")
    analyze_annotations(frames)
    analyze_rendering(
        with_fps(pl.read_parquet(_RESULTS_DIR / "crazyflow_render.parquet")),
        with_fps(pl.read_parquet(_RESULTS_DIR / "diff_aero_render.parquet")),
    )
    analyze_splat_vs_unity(
        with_fps(pl.read_parquet(_RESULTS_DIR / "crazyflow_splat_render.parquet")),
        with_fps(pl.read_parquet(_RESULTS_DIR / "flightmare_render.parquet")),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logger.setLevel(logging.INFO)
    main()
