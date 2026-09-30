"""Script fitting the PX4 thrust curve.

The ramp csvs from ``extract_thrust_from_video.py`` hold the stand reading in gram for the one motor on
the stand, which is converted to newton here, so α is the maximum thrust per motor in N.

In large parts copied from https://github.com/PX4/PX4-user_guide/blob/main/assets/config/mc/ThrustCurve.ipynb

Usage:
    pixi run -e sysid python crazyflow_experiments/sysid/fit_pwm_to_thrust.py
"""

from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy import optimize

from crazyflow_experiments.utils import paths

plt.rcParams["figure.figsize"] = 8, 5
plt.rcParams["axes.grid"] = True
plt.rcParams["figure.dpi"] = 96
plt.rcParams["savefig.dpi"] = 240

FILENAMES = "ramp_2026-09-02_*.csv"
RESULTS_DIR = paths.RESULTS_DIR / "sysid"
PLOTS_DIR = paths.PLOTS_DIR / "sysid"
PWM_MIN, PWM_MAX = 0.0, 1.0
PWM_MASK_MIN, PWM_MASK_MAX = 0.0, 1.0
GRAVITY = 9.81  # N/kg, converts the stand reading in gram to newton


def read_file(fname: Path) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Read the normalized PWM and the measured thrust per motor in N."""
    pwm, thrust_gram = np.genfromtxt(
        fname, delimiter=",", skip_header=1, usecols=(0, 1), unpack=True
    )
    return pwm, thrust_gram / 1000 * GRAVITY


def thrust_model_func(pwm: NDArray[np.float64], α: float, k: float) -> NDArray[np.float64]:
    """Evaluate the PX4 thrust curve model."""
    pwm_rel = (pwm - PWM_MIN) / (PWM_MAX - PWM_MIN)
    return α * (k * pwm_rel**2 + (1 - k) * pwm_rel)


def fixed_alpha_thrust_model(
    pwm: NDArray[np.float64], k: float, *, alpha: float
) -> NDArray[np.float64]:
    """Evaluate the thrust curve with alpha fixed for the combined fit."""
    return thrust_model_func(pwm, alpha, k)


fig, ax = plt.subplots()
pwm_space = np.linspace(PWM_MASK_MIN, PWM_MASK_MAX)
filename_prefix, filename_suffix = FILENAMES.split("*")
all_pwm = []
all_thrust = []
all_alpha = []

for i, fname in enumerate(sorted(RESULTS_DIR.glob(FILENAMES))):
    pwm, thrust = read_file(fname)
    mask = (pwm >= PWM_MASK_MIN) & (pwm <= PWM_MASK_MAX)
    all_pwm.append(pwm[mask])
    all_thrust.append(thrust[mask])

    (α, k), _ = optimize.curve_fit(thrust_model_func, pwm[mask], thrust[mask])
    all_alpha.append(α)
    thrust_model = partial(thrust_model_func, α=α, k=k)

    c = f"C{i}"
    label = fname.name.removeprefix(filename_prefix).removesuffix(filename_suffix)

    ax.plot(pwm, thrust, ".", color=c)
    ax.plot(
        pwm_space, thrust_model(pwm_space), "-", color=c, label=f"{label} (α={α:.2f}, k={k:.2f})"
    )

combined_pwm = np.concatenate(all_pwm)
combined_thrust = np.concatenate(all_thrust)
combined_alpha = np.mean(all_alpha)
combined_model = partial(fixed_alpha_thrust_model, alpha=combined_alpha)
(combined_k,), _ = optimize.curve_fit(combined_model, combined_pwm, combined_thrust)
ax.plot(
    pwm_space,
    combined_model(pwm_space, combined_k),
    "--",
    color="black",
    label=f"Combined (α={combined_alpha:.2f}, k={combined_k:.2f})",
)

ax.set_xlabel("Normalized PWM")
ax.set_ylabel("Measured thrust per motor (N)")

ax.set_title("PX4 Thrust Curve Compensation")
ax.legend()
fig.tight_layout()
plt.show()
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(PLOTS_DIR / f"px4-thrust-curve-{FILENAMES.split('_')[1]}.png")
