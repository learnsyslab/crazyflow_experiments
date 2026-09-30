"""Shared viser visualization tools."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import crazyflow
import numpy as np
import trimesh

if TYPE_CHECKING:
    from crazyflow.sim.data import SimData
    from numpy.typing import NDArray

DRONE_MESH = Path(crazyflow.__file__).parent / "drones/assets/cf21B/cf21B_full.stl"
MESH_SCALE = 0.001  # the assets are modelled in millimetres
ZOOM = 6  # whole pixel magnification, so the viewer's own expand shows it big and unblurred


def add_drone(scene: Any) -> tuple[Any, Any]:
    """Put the drone body and its axes into the scene."""
    assert DRONE_MESH.exists(), f"no drone mesh at {DRONE_MESH}"
    body = trimesh.load_mesh(DRONE_MESH)
    body.apply_scale(MESH_SCALE)
    return (
        scene.add_mesh_trimesh("drone", body),
        scene.add_frame("drone_axes", axes_length=0.15, axes_radius=0.004),
    )


def update_drone_pose(handles: tuple[Any, ...], data: SimData, world: int = 0):
    """Move the drone handles onto the pose the simulation currently holds."""
    pos = np.asarray(data.states.pos[world, 0])
    # The simulator orders quaternions xyzw and the viewer wants wxyz
    quat = np.asarray(data.states.quat[world, 0])[[3, 0, 1, 2]]
    for handle in handles:
        handle.position, handle.wxyz = tuple(pos), tuple(quat)


def gray2rgb(frame: NDArray[np.floating], zoom: int = ZOOM) -> NDArray[np.uint8]:
    """Transform a grayscale image to RGB."""
    grey = (np.asarray(frame) * 255).astype(np.uint8)
    assert grey.ndim == 2, f"expected a grayscale image, got {grey.ndim} dims"
    blocky = np.repeat(np.repeat(grey, zoom, axis=0), zoom, axis=1)
    return np.repeat(blocky[..., None], 3, axis=-1)
