"""Utils for sim2real experiments."""

from itertools import count, product
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

### Generate the parameter grid for all experiments
controller_type = "mellinger"  # mellinger, state
# "cf2x_L250", "cf2x_P250", "cf2x_T350", "cf21B_500", "racer", "hb_x500"
models = ["cf2x_L250", "cf2x_T350", "cf21B_500", "hb_x500"]
# "first_principles", "so_rpy_rotor_drag", "pybullet", "crazysim", "l2f", "flightmare"
physics = ["so_rpy_rotor_drag"]
# Origin the real flights of a model were centred on, where it differs from config.toml: the
# Crazyflies fly at 1.25 m in the motion capture room, the x500 at 3 m
origins = {"hb_x500": [0.0, 0.0, 3.0]}
# Main sim2real experiments for the paper: type, scaling, plane, time per lap
conditions = [
    ("ellipse", [3.0, 3.0, 0.0], "xy", 9.0),
    ("ellipse", [3.0, 3.0, 0.0], "xy", 3.5),
    ("figure8", [2.0, 0.0, 1.0], "xz", 10.0),
    ("figure8", [2.0, 0.0, 1.0], "xz", 5.5),
]


def param_grid(models: list[str], physics: list[str]) -> list[tuple]:
    """Build the sweep over the conditions for every drone model and physics backend."""
    return [
        (model, phys, *condition) for model, phys, condition in product(models, physics, conditions)
    ]


def generate_filename_from_config(
    config: dict, deploy: bool = False, skip_physics: bool = False
) -> str:
    """Generates the filename for the experiment based on the config."""
    filename = f"{config['controller']['type']}"
    filename += f"_{config['experiment']['trajectory']}"
    filename += "_" + "-".join(str(x) for x in config["experiment"]["scaling"])
    filename += f"_{config['experiment']['plane']}"
    filename += f"_{config['experiment']['time']}s"
    filename += f"_{config['drone']['model']}"
    if not skip_physics:
        if deploy:
            filename += "_deploy"
        else:
            filename += f"_{config['sim']['physics']}"
    return filename


def next_recording_path(directory: Path, config: dict, deploy: bool = False) -> Path:
    """Reserve the next free numbered recording path for an experiment.

    Deployments and simulator rollouts are both stored as `<model>/<experiment>_<source>_NN.pkl`,
    in a subdirectory named after the drone model, where the two-digit index counts the
    repetitions of the same experiment. Repeating the simulations lets the sim-to-real analysis
    pair the k-th rollout with the k-th real flight.

    Args:
        directory: Root of the recordings. The file lands in its subdirectory named after the
            drone model, created if it does not exist.
        config: Experiment config, see `generate_filename_from_config`.
        deploy: Whether the recording is a real flight rather than a simulator rollout.

    Returns:
        The path with the lowest index that does not exist yet.
    """
    directory = directory / config["drone"]["model"]
    directory.mkdir(parents=True, exist_ok=True)
    stem = generate_filename_from_config(config, deploy)
    paths = (directory / f"{stem}_{i:02d}.pkl" for i in count())
    return next(path for path in paths if not path.exists())


def list2dict(dicts: list[dict[str, float]]) -> dict[str, NDArray]:
    """Transpose a list of dicts with identical keys into a dict of NDArrays."""
    result = {k: [] for k in dicts[0].keys()}
    for d in dicts:
        for k in d.keys():
            result[k].append(d[k])
    for k, v in result.items():
        result[k] = np.array(v)
    return result
