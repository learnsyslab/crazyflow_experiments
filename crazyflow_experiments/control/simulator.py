"""Reading the simulated drone state."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    from crazyflow import Sim


def extract_obs(sim: Sim) -> dict[str, Any]:
    """Build an observation of the first drone from the current sim state."""
    quat = np.array(sim.data.states.quat[0, 0])
    return {
        "pos": np.array(sim.data.states.pos[0, 0]),
        "quat": quat,
        "vel": np.array(sim.data.states.vel[0, 0]),
        "ang_vel": np.array(sim.data.states.ang_vel[0, 0]),
        "rpy": R.from_quat(quat).as_euler("xyz"),
        "t": float(sim.data.core.steps[0, 0]) / sim.freq,
    }
