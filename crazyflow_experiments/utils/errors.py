"""Position errors of flown trajectories.

Every experiment that reports how well a flight tracked its reference, or how close a simulated
flight came to a real one, computes the numbers here, so that the plots, tables and tests all use
the same definitions. All errors are in metres.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from crazyflow_experiments.control.trajectories import Trajectory


def tracking_error(
    t: NDArray[np.floating], pos: NDArray[np.floating], trajectory: Trajectory
) -> NDArray[np.floating]:
    """Euclidean distance of every sample to the reference position at its trajectory time.

    Args:
        t: Trajectory times of the samples, zero where the trajectory starts.
        pos: Positions of the samples, shape (len(t), 3).
        trajectory: Reference the flight tracked.

    Returns:
        The error of every sample in m.
    """
    t = np.asarray(t, dtype=float)
    pos = np.asarray(pos, dtype=float)
    if t.ndim != 1 or pos.shape != (t.size, 3):
        raise ValueError(
            f"Expected t of shape (n,) and pos of shape (n, 3), got {t.shape} and {pos.shape}."
        )
    reference = np.asarray(trajectory.pos(t), dtype=float)
    return np.linalg.norm(pos - reference, axis=-1)


def tracking_rmse(
    t: NDArray[np.floating], pos: NDArray[np.floating], trajectory: Trajectory
) -> float:
    """Root mean square of the `tracking_error` over all samples, in m."""
    if t.size == 0:
        raise ValueError("Cannot compute the tracking RMSE of an empty flight.")
    return float(np.sqrt(np.mean(tracking_error(t, pos, trajectory) ** 2)))


def sim2real_rmse(
    t_sim: NDArray[np.floating],
    pos_sim: NDArray[np.floating],
    t_real: NDArray[np.floating],
    pos_real: NDArray[np.floating],
) -> float:
    """Position RMSE between a simulated and a real flight, in m.

    The simulated flight is interpolated onto the timestamps of the real flight, whose samples
    stay untouched, and the root mean square of the Euclidean distance between them is returned.
    The simulation has to cover the whole real flight, a rollout that stops early is an error
    rather than a flight that silently holds its last position.

    Args:
        t_sim: Timestamps of the simulated flight, on the same clock as `t_real`.
        pos_sim: Positions of the simulated flight, shape (len(t_sim), 3).
        t_real: Timestamps of the real flight.
        pos_real: Positions of the real flight, shape (len(t_real), 3).
    """
    if t_real[0] < t_sim[0] or t_real[-1] > t_sim[-1]:
        raise ValueError(
            f"The simulation covers [{t_sim[0]:.3f}, {t_sim[-1]:.3f}] s but the real flight "
            f"[{t_real[0]:.3f}, {t_real[-1]:.3f}] s."
        )
    sim = np.stack([np.interp(t_real, t_sim, pos_sim[:, i]) for i in range(3)], axis=-1)
    return float(np.sqrt(np.mean(np.linalg.norm(sim - pos_real, axis=-1) ** 2)))


def run_mask(
    t: NDArray[np.floating], cycle_time: float, n_runs: int, run: int | None = None
) -> NDArray[np.bool_]:
    """Select the samples inside the tracked runs of a periodic trajectory.

    Args:
        t: Trajectory times of the samples, zero where the trajectory starts.
        cycle_time: Duration of one run in s.
        n_runs: Number of runs the trajectory repeats.
        run: Zero-based index of a single run to select. All runs when None.

    Returns:
        A boolean mask over `t`, True inside the selected window.
    """
    if run is None:
        t_start, t_end = 0.0, cycle_time * n_runs
    else:
        if not 0 <= run < n_runs:
            raise ValueError(f"Run {run} is out of range for {n_runs} runs.")
        t_start, t_end = run * cycle_time, (run + 1) * cycle_time
    t = np.asarray(t, dtype=float)
    return (t >= t_start) & (t <= t_end)
