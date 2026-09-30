"""Throw-and-stabilize policies trained with BPTT."""

import tomllib
from pathlib import Path

from crazyflow_experiments.utils.paths import RESULTS_DIR

CONFIG = tomllib.loads((Path(__file__).parent / "config.toml").read_text())

RECORDINGS_DIR = RESULTS_DIR / "throw"  # Throw recordings written by deploy.py
SUCCESS_CURVES_PATH = RECORDINGS_DIR / "success_curves.json"

_EXPERIMENT = CONFIG["experiment"]
STABILIZED_VEL_Z_TOL = _EXPERIMENT["stabilized_vel_z_tol"]
STABILIZED_MIN_TIME = _EXPERIMENT["stabilized_min_time"]
FLIGHT_BOUNDS_X = tuple(_EXPERIMENT["flight_bounds_x"])
FLIGHT_BOUNDS_Y = tuple(_EXPERIMENT["flight_bounds_y"])
FLIGHT_BOUNDS_Z = tuple(_EXPERIMENT["flight_bounds_z"])
CEILING_MIN_TIME = _EXPERIMENT["ceiling_min_time"]
# Noise of the x_start resets, see ThrowEnv.create
X_START_NOISE = CONFIG["controller"]["x_start_randomization"]
