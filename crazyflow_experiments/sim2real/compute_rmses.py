"""Compute per-deployment sim-to-real position RMSEs and save them as TOML.

Every simulator is rolled out once per real flight (`n_rollouts` in `config.toml`) and the k-th
rollout is compared against the k-th real flight. This pairing keeps the RMSE samples of a
non-deterministic simulator such as CrazySim independent across real flights; for deterministic
simulators the rollouts are identical and the pairing changes nothing.

The RMSE is taken over the middle run of the trajectory. Recordings are timed in trajectory time,
zero where the trajectory starts, so the soft start and the first run, in which the flight still
settles onto the trajectory, lie before the window, and the last run after it.
"""

import pickle
import re
from pathlib import Path

import numpy as np
import toml
from numpy.typing import NDArray

from crazyflow_experiments.control.config import TrajectoryConfig
from crazyflow_experiments.sim2real import RECORDINGS_DIR
from crazyflow_experiments.sim2real.utils import controller_type, models
from crazyflow_experiments.utils.errors import run_mask, sim2real_rmse

CONFIG_PATH = Path(__file__).parent / "config.toml"
HEADER = (
    "# Position RMSE values in millimetres (mm) for each trajectory, one value per real flight,\n"
    "# each compared against the simulator rollout with the same index over the middle run of\n"
    "# the trajectory. `_crashes` counts the rollouts that lost the drone before the end of that\n"
    "# run; they stay in the sample. `physics` names the recordings the simulator's values come\n"
    "# from, e.g. which crazyflow dynamics.\n\n"
)
SIMULATORS = {
    # Both crazyflow models, named by their dynamics
    "crazyflow_first_principles": ("first_principles",),
    "crazyflow_so_rpy_rotor_drag": ("so_rpy_rotor_drag",),
    "pybullet": ("pybullet", "gpd"),
    "crazysim": ("crazysim",),
    "l2f": ("l2f",),
    "flightmare": ("flightmare",),
    # Extra variants for the GPD ablation, see sim_pybullet.VARIANTS
    # "pybullet-drag": ("pybullet-drag",),
    # "pybullet-mellinger-rotor-drag": ("pybullet-mellinger-rotor-drag",),
    # Extra variants for CrazySim ablation
    # "crazysim-gz": ("crazysim-gz",),
    # "crazysim-gz-fix": ("crazysim-gz-fix",),
    # "crazysim-mj": ("crazysim-mj",),
    # "crazysim-mj-fix": ("crazysim-mj-fix",),
}


def load_recording(path: Path) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Load a recording.

    Args:
        path: Recording pickle, a list of observation dicts with "t" and "pos" entries.

    Returns:
        Trajectory times, zero where the trajectory starts, and positions of all samples.
    """
    with open(path, "rb") as f:
        data = pickle.load(f)
    t = np.array([obs["t"] for obs in data], dtype=float)  # Trajectory time
    pos = np.array([obs["pos"] for obs in data], dtype=float)
    return t, pos


def crashed(path: Path, t_end: float) -> bool:
    """Whether a rollout lost the drone before trajectory time `t_end`.

    A drone at or below the ground has crashed; a simulator that recovers the drone itself marks
    the samples where it lost it under "lost".
    """
    with open(path, "rb") as f:
        data = pickle.load(f)
    t = np.array([obs["t"] for obs in data], dtype=float)
    lost = np.array([obs["pos"][2] <= 0.0 or bool(obs.get("lost", False)) for obs in data])
    return bool(np.any(lost & (t <= t_end)))


def middle_run(
    path: Path, cycle_time: float, n_runs: int
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Load a real flight and crop it to the middle run of its trajectory.

    Args:
        path: Recording pickle, see `load_recording`.
        cycle_time: Duration of one run of the trajectory in s.
        n_runs: Number of runs the trajectory repeats.

    Returns:
        Trajectory times and positions of the samples inside the middle run.
    """
    t, pos = load_recording(path)
    window = run_mask(t, cycle_time, n_runs, run=n_runs // 2)
    if not window.any():
        raise ValueError(f"{path.name} covers [{t[0]:.2f}, {t[-1]:.2f}] s, not the middle run.")
    return t[window], pos[window]


def find_conditions(
    directory: Path, controller: str, model: str
) -> list[tuple[str, str, float, list[Path]]]:
    """Group deployment recordings by trajectory condition."""
    pattern = re.compile(
        rf"^{re.escape(controller)}_(?P<trajectory>.+?)_[^_]+_[xyz]{{2}}_"
        rf"(?P<time>\d+(?:\.\d+)?)s_{re.escape(model)}_deploy_(?P<run>\d+)\.pkl$"
    )
    conditions: dict[tuple[str, float, str], list[tuple[int, Path]]] = {}

    for path in directory.glob(f"{controller}_*_{model}_deploy_*.pkl"):
        match = pattern.match(path.name)
        if match is None:
            continue
        stem = path.name.rsplit("_deploy_", 1)[0]
        identity = match.group("trajectory"), float(match.group("time")), stem
        conditions.setdefault(identity, []).append((int(match.group("run")), path))

    result = []
    for (trajectory, cycle_time, stem), deployments in sorted(
        conditions.items(), key=lambda item: (item[0][0], -item[0][1])
    ):
        duration = f"{cycle_time:g}".replace(".", "p")
        key = f"{trajectory}_{duration}s"
        paths = [path for _, path in sorted(deployments)]
        result.append((key, stem, cycle_time, paths))
    return result


def simulator_paths(
    directory: Path, stem: str, backends: tuple[str, ...]
) -> tuple[str | None, list[Path]]:
    """The first backend that has numbered rollouts and those rollouts, sorted by index."""
    for backend in backends:
        if paths := sorted(directory.glob(f"{stem}_{backend}_[0-9][0-9].pkl")):
            return backend, paths
    return None, []


def compute_model(
    directory: Path, controller: str, model: str, flight: TrajectoryConfig
) -> dict[str, dict[str, list[float] | float]]:
    """Compute all trajectory RMSE values for one drone model.

    Args:
        directory: Folder with the recordings of the model.
        controller: Controller type of the flights.
        model: Drone model of the flights.
        flight: Number of runs of the flown trajectories, the RMSE covers the middle
            run.

    Returns:
        The RMSEs per simulator and trajectory, one per real flight, with their mean and std, and
        the number of paired rollouts that crashed before the end of the middle run.
    """
    data: dict[str, dict[str, list[float] | float]] = {simulator: {} for simulator in SIMULATORS}
    n_runs = flight.n_runs

    for key, stem, cycle_time, deployment_paths in find_conditions(directory, controller, model):
        deployments = [middle_run(path, cycle_time, n_runs) for path in deployment_paths]

        for simulator, backends in SIMULATORS.items():
            backend, paths = simulator_paths(directory, stem, backends)
            if not paths:
                continue
            data[simulator]["physics"] = backend  # The recordings behind the values
            if len(paths) < len(deployments):
                raise FileNotFoundError(
                    f"{simulator} has {len(paths)} rollouts for {stem}, but there are "
                    f"{len(deployments)} real flights to pair them with."
                )
            # The rollouts stay whole, `sim2real_rmse` interpolates them onto the real samples
            simulations = [load_recording(path) for path in paths]
            rmses = [
                sim2real_rmse(*simulation, *deployment) * 1000.0
                for simulation, deployment in zip(simulations, deployments)
            ]
            data[simulator][key] = rmses
            data[simulator][f"{key}_mean"] = float(np.mean(rmses))
            data[simulator][f"{key}_std"] = float(np.std(rmses))
            t_end = (n_runs // 2 + 1) * cycle_time  # End of the middle run
            data[simulator][f"{key}_crashes"] = sum(crashed(path, t_end) for path in paths)

    return data


def main() -> None:
    """Compute and write one RMSE file per drone model, next to the recordings of the model."""
    flight = TrajectoryConfig.from_config(toml.load(CONFIG_PATH))  # Number of runs

    for model in models:
        directory = RECORDINGS_DIR / model  # One folder of recordings per drone model
        directory.mkdir(parents=True, exist_ok=True)
        data = compute_model(directory, controller_type, model, flight)
        drone_name = "B500" if model == "cf21B_500" else model.rsplit("_", 1)[-1]
        path = directory / f"{drone_name}_RMSEs.toml"
        path.write_text(HEADER + toml.dumps(data), encoding="utf-8")


if __name__ == "__main__":
    main()
