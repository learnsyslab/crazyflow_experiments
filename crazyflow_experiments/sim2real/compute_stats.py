"""Test whether one simulator tracks the real drone more closely than another.

Run with `pixi run -e sim2real python crazyflow_experiments/sim2real/compute_stats.py` after
`compute_rmses.py` has written the `<drone>_RMSEs.toml` files.

Data
    Every simulator is rolled out once per real flight of trajectory ``j`` and the ``i``-th rollout
    is compared against the ``i``-th of the ``n_j`` real flights, see `compute_rmses.py`. This
    yields RMSEs ``r^A_{j,i}`` for simulator ``A`` and ``r^B_{j,i}`` for simulator ``B``,
    ``i = 1..n_j``, where the index ``i`` refers to the same real flight for both simulators. The
    two samples are therefore paired by real flight, and the paired log differences

        d_{j,i} = log(r^A_{j,i}) - log(r^B_{j,i}) = log(r^A_{j,i} / r^B_{j,i})

    are the quantity of interest. The RMSE is a positive scale quantity whose spread grows with its
    magnitude, and the simulators differ by factors rather than by fixed offsets, so the test is run
    on the log scale, where the paired differences have a common variance and the result is a
    statement about the ratio of the RMSEs. Each pair uses its own rollout, so the randomness of a
    non-deterministic simulator such as CrazySim enters every difference independently and the
    pairs stay independent. Crashed rollouts are kept in the sample. The hypotheses below are then
    about the RMSE expected over both the real flights and the simulator's own randomness.

Per-trajectory test
    For each trajectory ``j`` a one-sided paired t-test is run with

        H0_j: mu_j >= 0   (A is not closer to the real drone than B on trajectory j)
        H1_j: mu_j <  0   (A is closer to the real drone than B on trajectory j)

    where ``mu_j`` is the expected paired log difference, i.e. ``mu_j < 0`` means the geometric
    mean RMSE of A is smaller than that of B. The statistic and lower-tail p-value are

        t_j = mean(d_j) / (std(d_j, ddof=1) / sqrt(n_j)),    p_j = F_{t, n_j - 1}(t_j),

    with ``F_{t, nu}`` the CDF of Student's t distribution with ``nu`` degrees of freedom. The test
    assumes the paired log differences are approximately normal.

Intersection-union test (IUT)
    A is only called better than B if it is better on every trajectory. The global hypotheses are

        H0: union_j H0_j          (A is not better on at least one trajectory)
        H1: intersection_j H1_j   (A is better on every trajectory)

    H0 is rejected at level ``alpha`` if and only if every H0_j is rejected at level ``alpha``, so
    the IUT p-value is

        p_IUT = max_j p_j.

    Because rejecting requires all individual tests to reject, the IUT has level ``alpha`` without
    any multiplicity correction (Berger, 1982).

Geometric mean ratio (GMR)
    The effect size on trajectory ``j`` is the geometric mean ratio

        GMR_j = exp(mean(d_j)) = GM(r^A_j) / GM(r^B_j),

    the factor by which the RMSE of A is smaller (``GMR_j < 1``) or larger than that of B. Its
    two-sided confidence interval at level ``c`` is the t interval of ``mu_j`` mapped back through
    the exponential,

        exp(mean(d_j) -/+ F^-1_{t, n_j - 1}((1 + c) / 2) * std(d_j, ddof=1) / sqrt(n_j)).

    The 95 % interval lies below one exactly when ``p_j < 0.025``, so it is stricter than the
    one-sided test at ``alpha = 0.05``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import toml
from scipy.stats import ttest_rel

from crazyflow_experiments.sim2real import RECORDINGS_DIR

if TYPE_CHECKING:
    from pathlib import Path

REFERENCE = (
    "crazyflow"  # Every simulator A with this prefix is tested against every other simulator B
)
ALPHA = 0.05
CONFIDENCE = 0.95  # Level of the two-sided confidence interval of the geometric mean ratio
SUMMARY_KEYS = {"iut", "gmr", "gmr_ci"}  # Every other key of a test is a trajectory
HEADER = (
    "# One-sided paired t-tests of the log RMSE per trajectory, one table per crazyflow model,\n"
    "# H1: the crazyflow model has a lower RMSE than the listed simulator. 'iut' is the\n"
    "# intersection-union p-value, the maximum over the trajectories the simulator completed; a\n"
    "# trajectory it has no rollouts for is left out. 'gmr' is the geometric mean ratio of the\n"
    "# RMSE of the crazyflow model over that of the listed simulator per trajectory, in the order\n"
    "# of the p-values, below one where crazyflow is closer, and 'gmr_ci' its two-sided\n"
    f"# {CONFIDENCE:.0%} confidence interval.\n\n"
)


def load_rmses(path: Path) -> dict[str, dict[str, np.ndarray]]:
    """Load the per-run RMSEs from a `compute_rmses.py` output file.

    Args:
        path: The `<drone>_RMSEs.toml` file.

    Returns:
        RMSE arrays indexed by simulator and trajectory. The mean and std entries are dropped.
    """
    data = toml.load(path)
    return {
        simulator: {
            key: np.asarray(value) for key, value in rmses.items() if isinstance(value, list)
        }
        for simulator, rmses in data.items()
    }


def iut(rmses_a: dict[str, np.ndarray], rmses_b: dict[str, np.ndarray]) -> dict[str, float]:
    """Test if simulator A has a lower RMSE than simulator B on every trajectory, on the log scale.

    Args:
        rmses_a: Per-trajectory RMSEs of simulator A, one value per real deployment.
        rmses_b: Per-trajectory RMSEs of simulator B, paired with `rmses_a` by deployment.

    Returns:
        The one-sided paired p-value per trajectory and their maximum under the key "iut".
    """
    trajectories = sorted(rmses_a.keys() & rmses_b.keys())
    p_values = {
        key: float(ttest_rel(np.log(rmses_a[key]), np.log(rmses_b[key]), alternative="less").pvalue)
        for key in trajectories
    }
    p_values["iut"] = max(p_values.values())
    return p_values


def geometric_mean_ratio(
    rmses_a: np.ndarray, rmses_b: np.ndarray
) -> tuple[float, tuple[float, float]]:
    """Geometric mean ratio of the RMSEs of simulator A over B and its confidence interval.

    Args:
        rmses_a: RMSEs of simulator A on one trajectory, one value per real deployment.
        rmses_b: RMSEs of simulator B on the same trajectory, paired with `rmses_a` by deployment.

    Returns:
        The ratio and the bounds of its two-sided confidence interval at level `CONFIDENCE`.
    """
    log_a, log_b = np.log(rmses_a), np.log(rmses_b)
    ci = ttest_rel(log_a, log_b).confidence_interval(CONFIDENCE)
    return float(np.exp(np.mean(log_a - log_b))), (float(np.exp(ci.low)), float(np.exp(ci.high)))


def compare(
    rmses_a: dict[str, np.ndarray], rmses_b: dict[str, np.ndarray]
) -> dict[str, float | list]:
    """Test simulator A against simulator B and add the geometric mean ratio per trajectory.

    Args:
        rmses_a: Per-trajectory RMSEs of simulator A, one value per real deployment.
        rmses_b: Per-trajectory RMSEs of simulator B, paired with `rmses_a` by deployment.

    Returns:
        The p-values of `iut`, the geometric mean ratios under "gmr" and their confidence intervals
        under "gmr_ci", both listed in the sorted order of the trajectories.
    """
    trajectories = sorted(rmses_a.keys() & rmses_b.keys())
    ratios = [geometric_mean_ratio(rmses_a[key], rmses_b[key]) for key in trajectories]
    return iut(rmses_a, rmses_b) | {
        "gmr": [ratio for ratio, _ in ratios],
        "gmr_ci": [ci for _, ci in ratios],
    }


def main() -> None:
    """Test every Crazyflow model against every other simulator for each drone model."""
    for path in sorted(RECORDINGS_DIR.glob("*/*_RMSEs.toml")):  # One folder per drone model
        rmses = load_rmses(path)
        # Only the simulators that have rollouts for the model enter the tests
        references = [a for a, values in rmses.items() if a.startswith(REFERENCE) and values]
        others = [b for b, values in rmses.items() if not b.startswith(REFERENCE) and values]
        results = {a: {b: compare(rmses[a], rmses[b]) for b in others} for a in references}
        out_path = path.with_name(path.name.replace("_RMSEs", "_stats"))
        out_path.write_text(HEADER + toml.dumps(results), encoding="utf-8")
        if not others:
            print(f"\n{path.stem}: only crazyflow has RMSEs, nothing to test against")
            continue

        for reference, tests in results.items():
            print(f"\n{path.stem}: H1: {reference} has a lower RMSE than the other simulator")
            trajectories = sorted({key for test in tests.values() for key in test} - SUMMARY_KEYS)
            width = max(len(s) for s in tests) + 2
            print(" " * width + "".join(f"{key:>14}" for key in [*trajectories, "iut"]))
            for simulator, test in tests.items():
                cells = "".join(
                    f"{test[key]:>14.2e}" if key in test else f"{'---':>14}"
                    for key in [*trajectories, "iut"]
                )  # --- where the simulator has no rollouts for the trajectory
                print(f"{simulator:<{width}}{cells}")
            better = [b for b, test in tests.items() if test["iut"] <= ALPHA]
            print(f"{reference} is better at alpha={ALPHA} than: " + ", ".join(better))

            print(f"\nGeometric mean RMSE ratio {reference} / simulator [{CONFIDENCE:.0%} CI]")
            print(" " * width + "".join(f"{key:>28}" for key in trajectories))
            for simulator, test in tests.items():
                own_trajectories = sorted(test.keys() - SUMMARY_KEYS)  # The order of the lists
                ratios = {
                    key: f"{ratio:#.3g} [{low:#.3g}, {high:#.3g}]"
                    for key, ratio, (low, high) in zip(
                        own_trajectories, test["gmr"], test["gmr_ci"], strict=True
                    )
                }
                cells = "".join(f"{ratios.get(key, '---'):>28}" for key in trajectories)
                print(f"{simulator:<{width}}{cells}")


if __name__ == "__main__":
    main()
