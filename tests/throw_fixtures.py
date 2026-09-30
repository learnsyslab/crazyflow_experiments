"""Synthetic throw recordings for the tests that replay them."""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path


def write_recording(path: Path):
    """Write a throw recording."""
    t = np.arange(0.0, 1.5, 0.01)
    phases = (("pre_throw", -1.0), ("thrown", 0.0), ("trained", 0.4), ("stabilizing", 0.5))
    obs = [
        {
            "t": float(ti),
            "state": max((p for p in phases if p[1] <= ti), key=lambda p: p[1])[0],
            "pos": np.array([0.0, 0.0, 2.0], dtype=np.float32),
            "quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "vel": np.array([0.0, 0.0, -1.0], dtype=np.float32),
            "ang_vel": np.array([0.1, 0.0, 0.0], dtype=np.float32),
        }
        for ti in t
    ]
    x_throw = {
        "pos": np.array([0.0, 0.0, 2.2], dtype=np.float32),
        "quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "vel": np.array([0.0, 0.0, -1.0], dtype=np.float32),
        "ang_vel": np.array([0.5, 0.0, 0.0], dtype=np.float32),
    }
    x_start = {key: value[None, None] for key, value in x_throw.items()}
    path.write_bytes(pickle.dumps({"x_throw": x_throw, "x_start": x_start, "obs": obs}))
