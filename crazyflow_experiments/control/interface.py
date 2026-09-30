"""Mapping from a controller type to the Crazyflow control interface it commands."""

from crazyflow.control import Control

from crazyflow_experiments.control.config import ATTITUDE_CONTROLLERS


def control_interface(ctrl_type: str, onboard: bool = False) -> Control:
    """Return the interface a controller of the given type sends its actions to."""
    # Membership test over the values: `ctrl_type in Control` requires Python >= 3.12, but the
    # CrazySim container runs Python 3.11.
    if ctrl_type in {control.value for control in Control}:
        return Control(ctrl_type)
    if ctrl_type == "td3":
        # The firmware runs the network and takes its goal through the state interface
        return Control.state if onboard else Control.rotor_vel
    if "state" in ctrl_type:
        return Control.state
    if "attitude" in ctrl_type:
        return Control.attitude
    if ctrl_type not in ATTITUDE_CONTROLLERS:
        raise ValueError(f"Unknown controller type {ctrl_type}")
    return Control.attitude
