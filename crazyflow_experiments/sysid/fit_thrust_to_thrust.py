"""Fit the measured thrust of the stand as a function of the commanded thrust.

The data comes from ``px4_random_thrust.py``, which sends random normalized thrust commands through
PX4's thrust model and battery compensation, extracted from the screen recording with
``extract_thrust_from_video.py``. Only the motor on the stand spins, so every measurement is the
thrust of one motor. The csvs hold the stand reading in gram, which is converted to newton here. If
the compensation were perfect, the measured thrust would be a line through the origin. Three models
are fitted and compared: a quadratic, a line with an offset and a line through the origin. The
console table lists their parameters, RMSE and R², followed by the fits of the combined data as
``thrust_curve`` lines for ``[drone]`` in the sysid and sim2real config files.

Usage:
    pixi run -e sysid python crazyflow_experiments/sysid/fit_thrust_to_thrust.py
"""

from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from crazyflow_experiments.utils import paths

plt.rcParams["figure.figsize"] = 8, 5
plt.rcParams["axes.grid"] = True
plt.rcParams["figure.dpi"] = 96
plt.rcParams["savefig.dpi"] = 240

DATE = "2026-09-05"
FILENAMES = f"random_{DATE}*.csv"
RESULTS_DIR = paths.RESULTS_DIR / "sysid"
PLOTS_DIR = paths.PLOTS_DIR / "sysid"
GRAVITY = 9.81  # N/kg, converts the stand reading in gram to newton
# Model name -> the powers of the command in its polynomial, highest first. The design matrix has one
# column per power and the fit is linear least squares in the coefficients.
MODELS = {"quadratic": (2, 1, 0), "linear": (1, 0), "proportional": (1,)}
LINE_STYLES = {"quadratic": "-", "linear": "--", "proportional": ":"}


class Fit(NamedTuple):
    """A polynomial fitted to the command-thrust pairs."""

    name: str
    powers: tuple[int, ...]
    coefficients: NDArray[np.float64]
    rmse: float  # N
    r2: float

    def predict(self, command: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate the fitted polynomial."""
        return design(command, self.powers) @ self.coefficients

    def formula(self, precision: int = 3) -> str:
        """Return the polynomial as text, e.g. ``-1.524 x² + 14.091 x - 0.428``."""
        text = ""
        for power, coefficient in zip(self.powers, self.coefficients):
            variable = {0: "", 1: " x", 2: " x²"}[power]
            if text:
                text += " - " if coefficient < 0 else " + "
            elif coefficient < 0:
                text += "-"
            text += f"{abs(coefficient):.{precision}f}{variable}"
        return text


def design(command: NDArray[np.float64], powers: tuple[int, ...]) -> NDArray[np.float64]:
    """Return the design matrix with one column per power of the command."""
    return np.stack([command**power for power in powers], axis=1)


def read_file(fname: Path) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Read the commanded normalized thrust and the measured thrust per motor in N."""
    command, thrust_gram = np.genfromtxt(
        fname, delimiter=",", skip_header=1, usecols=(0, 1), unpack=True
    )
    return command, thrust_gram / 1000 * GRAVITY


def fit_model(name: str, command: NDArray[np.float64], thrust: NDArray[np.float64]) -> Fit:
    """Fit one of the models in ``MODELS`` by least squares."""
    powers = MODELS[name]
    coefficients, *_ = np.linalg.lstsq(design(command, powers), thrust, rcond=None)
    residuals = thrust - design(command, powers) @ coefficients
    rmse = float(np.sqrt(np.mean(residuals**2)))
    r2 = float(1 - (residuals**2).sum() / ((thrust - thrust.mean()) ** 2).sum())
    return Fit(name, powers, coefficients, rmse, r2)


def print_table(label: str, fits: list[Fit], n_points: int) -> None:
    """Print the parameters and the goodness of fit of every model."""
    print(f"{label} ({n_points} points)")
    print(f"  {'model':<13} {'RMSE (N)':>8} {'R²':>7}  thrust per motor (N) =")
    for fit in fits:
        print(f"  {fit.name:<13} {fit.rmse:8.3f} {fit.r2:7.4f}  {fit.formula()}")


def print_config_lines(fits: list[Fit]) -> None:
    """Print every model as a ``thrust_curve`` entry of the config files, [a, b, c] for a x² + b x + c."""
    print("\n[drone] thrust_curve, per motor in N:")
    for fit in fits:
        by_power = dict(zip(fit.powers, fit.coefficients))
        coefficients = ", ".join(f"{by_power.get(power, 0.0):.4f}" for power in (2, 1, 0))
        print(f"  thrust_curve = [{coefficients}]  # {fit.name}, {DATE}")


fig, ax = plt.subplots()
command_space = np.linspace(0.0, 1.0)
files = sorted(RESULTS_DIR.glob(FILENAMES))
all_command = []
all_thrust = []

for i, fname in enumerate(files):
    command, thrust = read_file(fname)
    all_command.append(command)
    all_thrust.append(thrust)
    label = fname.stem.removeprefix("random_")
    fits = [fit_model(name, command, thrust) for name in MODELS]
    print_table(label, fits, len(command))
    ax.plot(command, thrust, ".", color=f"C{i}")
    for fit in fits:
        ax.plot(
            command_space,
            fit.predict(command_space),
            LINE_STYLES[fit.name],
            color=f"C{i}",
            label=f"{label} {fit.name}: {fit.formula(2)}",
        )

combined_command = np.concatenate(all_command)
combined_thrust = np.concatenate(all_thrust)
combined_fits = [fit_model(name, combined_command, combined_thrust) for name in MODELS]
if len(files) > 1:
    print_table("Combined", combined_fits, len(combined_command))
    for fit in combined_fits:
        ax.plot(
            command_space,
            fit.predict(command_space),
            LINE_STYLES[fit.name],
            color="black",
            label=f"Combined {fit.name}: {fit.formula(2)}",
        )
print_config_lines(combined_fits)

ax.set_xlabel("Commanded normalized thrust")
ax.set_ylabel("Measured thrust per motor (N)")
ax.set_title("PX4 Commanded vs. Measured Thrust")
ax.legend()
fig.tight_layout()
plt.show()
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(PLOTS_DIR / f"px4-thrust-to-thrust-{DATE}.png")
