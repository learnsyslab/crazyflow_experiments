"""Fit the simplified so_rpy model to the converted recordings of a Crazyflie or a PX4 vehicle.

The csvs come from `data_converter_mocap.py` for a Crazyflie in `results/sysid/rosbags/` and from
`data_converter_px4.py` for a PX4 vehicle in `results/sysid/ulogs/`. Both hold the same columns, but
the Crazyflie commands its attitude in deg and its thrust as 16 bit pwm, the PX4 vehicle in rad and
as normalized thrust command. Every drone parameter comes from the table of the model `config.toml` selects in
`config_drones.toml`: the mass, the pwm range and the thrust curve of one motor in N over the
normalized command p = pwm / pwm_max,
thrust = a p^2 + b p + c, clipped to the thrust limits. A Crazyflie's curve is linear,
[0, thrust_max, 0], a PX4 vehicle's comes from the thrust stand (`fit_thrust_to_thrust.py`).

The filtered derivatives are cached as `<recording>_SVF.csv` next to the csvs, delete them to
recompute after changing the config.

The identified parameters are written to `results/sysid/params_<model>_<dynamics>_<date>.toml` as
the `[<model>]` table of crazyflow's `dynamics/<dynamics>/params.toml`, ready to be pasted there.

Usage:

    pixi run -e sysid python crazyflow_experiments/sysid/fit_simplified_model.py [--dynamics DYN]

`--dynamics` picks the translational model, `so_rpy` for the plain thrust, `so_rpy_rotor` with the
rotor lag and `so_rpy_rotor_drag` with the body drag on top, which is the default.

Set `FILENAME_TRAIN` and `FILENAME_VALID` to the recordings, `FILENAME_VALID = None` fits without
validation data.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import toml
from crazyflow.dynamics.utils.data_utils import derivatives_svf, preprocessing
from crazyflow.dynamics.utils.identification import sys_id_rotation, sys_id_translation

from crazyflow_experiments.control.config import drone_params
from crazyflow_experiments.sysid import RECORDINGS_DIR
from crazyflow_experiments.sysid.utils import csv2dict, dict2csv

if TYPE_CHECKING:
    from numpy.typing import NDArray

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Recordings in results/sysid/rosbags or results/sysid/ulogs, without the .csv suffix
FILENAME_TRAIN = "data_x500_sysid_train_2026-09-09"
FILENAME_VALID: str | None = None
# FILENAME_TRAIN = "data_cf2x_T350_sysid_train_2026-05-07"  # cf10
# FILENAME_VALID = "data_cf2x_T350_sysid_valid_2026-05-07"  # cf10
# FILENAME_TRAIN = "data_cf2x_L250_sysid_train_2026-03-04"
# FILENAME_VALID = "data_cf2x_L250_sysid_valid_2026-03-04"

DYNAMICS = ("so_rpy", "so_rpy_rotor", "so_rpy_rotor_drag")
# The state variable filters cut off at 6 and 8 Hz, so the poses must arrive well above twice that
MIN_RATE = 20.0  # Hz


def load_drone_params(config: dict) -> dict[str, float | NDArray]:
    """Read the drone parameters the fit needs, the `config_drones.toml` table of the config's model.

    Returns:
        mass in kg, pwm_min and pwm_max of the thrust command, thrust_min and thrust_max of one motor
        in N and thrust_curve as the coefficients [a, b, c] of the thrust of one motor in N over the
        normalized command.
    """
    drone = drone_params(config["drone"]["model"])
    params = {
        key: float(drone[key]) for key in ("mass", "pwm_min", "pwm_max", "thrust_min", "thrust_max")
    }
    params["thrust_curve"] = np.asarray(drone["thrust_curve"], dtype=np.float64)
    if params["thrust_curve"].shape != (3,):
        raise ValueError(f"thrust_curve must be [a, b, c], got {drone['thrust_curve']}")
    if not 0 <= params["pwm_min"] < params["pwm_max"]:
        raise ValueError(f"Need 0 <= pwm_min < pwm_max, got {drone['pwm_min']}, {drone['pwm_max']}")
    if not params["thrust_min"] < params["thrust_max"]:
        raise ValueError(
            f"Need thrust_min < thrust_max, got {drone['thrust_min']}, {drone['thrust_max']}"
        )
    return params


def pwm2thrust(cmd_pwm: NDArray, params: dict[str, float | NDArray]) -> NDArray:
    """Collective thrust in N for the commanded pwm.

    The pwm is clipped to its range and normalized, the thrust curve gives the thrust of one motor,
    which is clipped to its limits like on the drone, and the four motors add up.
    """
    a, b, c = params["thrust_curve"]
    pwm = np.clip(cmd_pwm, params["pwm_min"], params["pwm_max"]) / params["pwm_max"]
    thrust_per_motor = a * pwm**2 + b * pwm + c  # The inverse of deploy.px4.thrust2pwm
    return 4 * np.clip(thrust_per_motor, params["thrust_min"], params["thrust_max"])


def load_recording(filename: str, params: dict[str, float | NDArray], px4: bool) -> dict:
    """Load a converted recording with its filtered derivatives, computing and caching them once.

    Args:
        filename: Name of the recording in `results/sysid/rosbags` or `results/sysid/ulogs`,
            without the csv suffix.
        params: Drone parameters from `load_drone_params`.
        px4: Whether the recording is of a PX4 vehicle, whose commands are in rad rather than deg.
    """
    filepath_svf = RECORDINGS_DIR / f"{filename}_SVF.csv"
    if not filepath_svf.exists():
        paths = [RECORDINGS_DIR / sub / f"{filename}.csv" for sub in ("rosbags", "ulogs")]
        found = [path for path in paths if path.exists()]
        if not found:
            raise FileNotFoundError(f"No {filename}.csv in {[str(path.parent) for path in paths]}")
        data = csv2dict(found[0])
        logger.info(f"Loaded {filename}")
        rate = 1 / np.mean(np.diff(data["time"]))
        if rate < MIN_RATE:
            raise ValueError(
                f"{filename} is sampled at {rate:.1f} Hz, the filters need at least {MIN_RATE} Hz. "
                "Convert the recording from a faster pose source, for a PX4 vehicle a ulog with "
                "vehicle_local_position logged at full rate"
            )
        data["cmd_f"] = pwm2thrust(data["cmd_pwm"], params)
        if not px4:  # A Crazyflie is commanded in deg, a PX4 vehicle in rad
            data["cmd_rpy"] = np.deg2rad(data["cmd_rpy"])
        data = derivatives_svf(preprocessing(data))
        dict2csv(data, filepath_svf)
        logger.info(f"Preprocessed {filename} and stored the derivatives at {filepath_svf}")
    data = csv2dict(filepath_svf)
    logger.info(f"Loaded the preprocessed {filename}")
    return data


def params_table(
    translation: dict[str, float], rotation: dict[str, NDArray]
) -> dict[str, float | list[float] | list[list[float]]]:
    """Arrange the identified parameters like a drone table of crazyflow's `params.toml`."""
    table = {"acc_coef": 0.0, "cmd_f_coef": float(translation["cmd_f_coef"])}
    if "thrust_time_coef" in translation:
        table["thrust_time_coef"] = float(translation["thrust_time_coef"])
    if "drag_xy_coef" in translation:
        drag = [float(translation[key]) for key in ("drag_xy_coef", "drag_xy_coef", "drag_z_coef")]
        table["drag_matrix"] = np.diag(drag).tolist()
    for key in ("rpy_coef", "rpy_rates_coef", "cmd_rpy_coef"):
        table[key] = [float(value) for value in rotation[key]]
    return table


def main(dynamics: str) -> None:
    """Fit the translational and rotational so_rpy dynamics to the configured recordings.

    Args:
        dynamics: Translational dynamics to identify, one of `DYNAMICS`.
    """
    assert dynamics in ("so_rpy", "so_rpy_rotor", "so_rpy_rotor_drag"), (
        f"Unknown dynamics {dynamics}, must be one of {DYNAMICS}"
    )
    config = toml.load(Path(__file__).parent / "config.toml")
    model = config["drone"]["model"]
    params = load_drone_params(config)
    # A Crazyflie flies over the radio, every other model is a PX4 vehicle, see deploy.experiment
    px4 = not model.startswith("cf")
    assert model in FILENAME_TRAIN, (
        f"The training data does not contain the configured drone model {model}, "
        f"please check the recording name and the config file."
    )
    if FILENAME_VALID is not None:
        assert model in FILENAME_VALID, (
            f"The validation data does not contain the configured drone model {model}, "
            f"please check the recording name and the config file."
        )
    logger.info(f"Fitting {model} with mass {params['mass']} kg")

    data = load_recording(FILENAME_TRAIN, params, px4)
    data_valid = load_recording(FILENAME_VALID, params, px4) if FILENAME_VALID else None

    params_translation = sys_id_translation(
        dynamics=dynamics, mass=params["mass"], data=data, data_validation=data_valid, plot=True
    )
    params_rotation = sys_id_rotation(data=data, data_validation=data_valid, plot=True)

    path = RECORDINGS_DIR / f"params_{model}_{dynamics}_{datetime.now():%Y-%m-%d}.toml"
    path.write_text(toml.dumps({model: params_table(params_translation, params_rotation)}))
    logger.info(f"Stored the identified parameters at {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dynamics", choices=DYNAMICS, default="so_rpy_rotor_drag", help="dynamics to identify"
    )
    main(parser.parse_args().dynamics)
