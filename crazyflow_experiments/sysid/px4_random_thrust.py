"""Command random collective thrust to PX4 with its battery compensation in the loop.

The PX4 shell command ``actuator_test`` writes to the output driver, which still applies the thrust
model (``THR_MDL_FAC``), but the battery scaling (``MC_BAT_SCALE_EN``) lives in the rate controller
and never acts on it. To measure the thrust the vehicle really produces for a normalized thrust
command, this script runs on the companion computer and sends attitude setpoints through MAVROS in
offboard mode. The attitude setpoint always equals the current attitude, so the attitude controller
asks for no torque and every motor receives the collective thrust command, passed through the
battery compensation and the thrust model.

Every few seconds a new random thrust in [0, 1] is drawn and printed as a line on the terminal, and
nothing else is printed while the test runs, so a screen recording of this terminal and of the thrust
stand can be read with ``extract_thrust_from_video.py``. The test runs until the battery is low, the
duration is over or Ctrl+C is pressed, then the thrust is set to zero and the vehicle is disarmed.

The script is standalone, it only needs the ROS 2 Python packages of MAVROS and numpy.

The battery cell count is inferred from the voltage at the start, which only works with a battery
that is reasonably charged. The count must be confirmed on the terminal before the vehicle arms.

Safety: only the motor on the thrust stand may spin. The vehicle arms and drives that motor at up to
full thrust.

Pre-run checks on the vehicle (restore everything afterwards):
    1. Outputs. In QGroundControl open Vehicle Setup > Actuators and set the outputs of the three
       motors that are not on the stand to "Disabled". Leave the geometry untouched, the allocator
       must keep its four motors so the collective thrust reaches the tested motor unchanged.
    2. Parameters (test value, our default value, reason).
       COM_DISARM_LAND  -1  2  the land detector reports "landed" on the stand and would disarm
       COM_DISARM_PRFLT -1  10 no auto disarm while waiting without a takeoff
       COM_RC_IN_MODE    3  3 keep the RC transmitter on and set to OFFBOARD mode
       COM_ARM_CHK_ESCS  0  0 the disabled outputs report no ESC telemetry
       COM_OBL_RC_ACT    0  0 (default) offboard loss falls back to position mode, not to landing
       NAV_DLL_ACT       0  0  no data link loss failsafe; without a GCS connected the vehicle
                               otherwise sits in AUTO.LOITER and refuses other modes
       MC_BAT_SCALE_EN   1  1 the battery compensation that this test measures
       BAT1_R_INTERNAL  -1  -1  estimate the internal resistance online instead of a fixed value;
                               the load compensated voltage behind MC_BAT_SCALE_EN uses it and the
                               estimate needs the current sensor of the power module
       BAT1_N_CELLS      4  4  cell count of the pack, the estimator works per cell
       THR_MDL_FAC       the thrust model to evaluate; it also acts on ``actuator_test``, so a raw
                         thrust curve measured with the sweep script needs 0 here
       On the MAVLink console: ``param show THR_MDL_FAC``, ``param set THR_MDL_FAC 0.54``,
       ``param save``; ``param show COM_*`` lists a group.
    3. Thrust stand, camera and screen recording ready, battery charged so the cell count is
       unambiguous.

Usage (on the companion computer, with MAVROS running):
    python crazyflow_experiments/sysid/px4_random_thrust.py
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from typing import TYPE_CHECKING

import numpy as np
import rclpy
from mavros_msgs.msg import AttitudeTarget, State, StatusText
from mavros_msgs.srv import CommandBool, CommandLong, SetMode
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import BatteryState, Imu

if TYPE_CHECKING:
    from rclpy.client import Client

SETPOINT_RATE = 50.0  # Hz, offboard mode needs a steady setpoint stream
WARMUP = 2.0  # seconds of setpoints before switching to offboard mode
CELL_VOLTAGE_RANGE = (3.6, 4.25)  # V, a lithium cell that is charged enough for the test
# The MAVROS services fail with Fast DDS on the companion computer (fixed size response buffers),
# the client has to run with Cyclone DDS.
REQUIRED_RMW = "rmw_cyclonedds_cpp"
MAV_CMD_COMPONENT_ARM_DISARM = 400
FORCE_DISARM_MAGIC = 21196.0  # param2 that disarms even when PX4 believes it is flying


class RandomThrustNode(Node):
    """Streams attitude setpoints that hold the current attitude at a chosen thrust."""

    def __init__(self, namespace: str) -> None:
        """Subscribe to the vehicle state and publish attitude setpoints under ``namespace``."""
        super().__init__("px4_random_thrust")
        self.thrust = 0.0
        self.stop = False  # set by Ctrl+C, ends the test cleanly
        self.orientation = None
        self.state = State()
        self.battery = None
        # MAVROS publishes sensor topics best effort, a best effort subscription receives from
        # both best effort and reliable publishers.
        qos = qos_profile_sensor_data
        self.create_subscription(Imu, f"{namespace}/mavros/imu/data", self._on_imu, qos)
        self.create_subscription(State, f"{namespace}/mavros/state", self._on_state, qos)
        self.create_subscription(BatteryState, f"{namespace}/mavros/battery", self._on_battery, qos)
        self.attitude_pub = self.create_publisher(
            AttitudeTarget, f"{namespace}/mavros/setpoint_raw/attitude", 10
        )
        self.set_mode_client = self.create_client(SetMode, f"{namespace}/mavros/set_mode")
        self.arm_client = self.create_client(CommandBool, f"{namespace}/mavros/cmd/arming")
        self.command_client = self.create_client(CommandLong, f"{namespace}/mavros/cmd/command")
        self.create_subscription(
            StatusText, f"{namespace}/mavros/statustext/recv", self._on_statustext, qos
        )
        self.create_timer(1 / SETPOINT_RATE, self._publish_setpoint)

    def _on_imu(self, msg: Imu) -> None:
        self.orientation = msg.orientation

    def _on_state(self, msg: State) -> None:
        self.state = msg

    def _on_battery(self, msg: BatteryState) -> None:
        self.battery = msg

    def _on_statustext(self, msg: StatusText) -> None:
        print(f"PX4: {msg.text}", file=sys.stderr, flush=True)

    def _publish_setpoint(self) -> None:
        """Send the current attitude as setpoint together with the commanded thrust."""
        if self.orientation is None:
            return
        msg = AttitudeTarget()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.orientation = self.orientation
        msg.thrust = float(self.thrust)
        msg.type_mask = (
            AttitudeTarget.IGNORE_ROLL_RATE
            | AttitudeTarget.IGNORE_PITCH_RATE
            | AttitudeTarget.IGNORE_YAW_RATE
        )
        self.attitude_pub.publish(msg)

    def spin_for(self, seconds: float) -> None:
        """Process callbacks for the given time."""
        end = time.monotonic() + seconds
        while time.monotonic() < end and not self.stop:
            rclpy.spin_once(self, timeout_sec=0.02)

    def wait_for_data(self, timeout: float = 10.0) -> None:
        """Wait until the attitude, the vehicle state and the battery have been received."""
        end = time.monotonic() + timeout
        while self.orientation is None or self.battery is None or not self.state.connected:
            if time.monotonic() > end:
                raise TimeoutError(
                    "No IMU, battery or vehicle state from MAVROS. Is MAVROS running and connected?"
                )
            rclpy.spin_once(self, timeout_sec=0.1)

    def call(self, client: Client, request: object, what: str) -> object:
        """Call a MAVROS service and return its response, or fail with a clear message."""
        if not client.wait_for_service(timeout_sec=5.0):
            raise TimeoutError(f"The MAVROS service for {what} is not available")
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if future.result() is None:
            raise RuntimeError(
                f"No answer from MAVROS for {what}. With Fast DDS the services never answer, "
                f"run with RMW_IMPLEMENTATION={REQUIRED_RMW}."
            )
        return future.result()

    def set_offboard(self) -> None:
        """Switch to offboard mode and wait for the vehicle to report it."""
        request = SetMode.Request()
        request.custom_mode = "OFFBOARD"
        if not self.call(self.set_mode_client, request, "the flight mode").mode_sent:
            raise RuntimeError("MAVROS did not send the offboard mode request")
        end = time.monotonic() + 5.0
        while self.state.mode != "OFFBOARD":
            if time.monotonic() > end:
                raise RuntimeError(
                    f"Vehicle did not switch to OFFBOARD, it is in {self.state.mode}. PX4's reason "
                    "is in the 'PX4:' lines on stderr."
                )
            rclpy.spin_once(self, timeout_sec=0.05)

    def arm(self) -> None:
        """Arm the vehicle and wait for the state to follow."""
        request = CommandBool.Request()
        request.value = True
        response = self.call(self.arm_client, request, "arming")
        if not response.success:
            raise RuntimeError(
                f"PX4 rejected arming (result {response.result}). PX4's reason is in the 'PX4:' "
                "lines on stderr."
            )
        self.wait_armed(True)

    def force_disarm(self) -> None:
        """Disarm even though PX4 believes it is flying, as the emergency stop of a ground station."""
        request = CommandLong.Request()
        request.command = MAV_CMD_COMPONENT_ARM_DISARM
        request.param1 = 0.0
        request.param2 = FORCE_DISARM_MAGIC
        response = self.call(self.command_client, request, "disarming")
        if not response.success:
            raise RuntimeError(f"PX4 rejected the force disarm (result {response.result})")
        self.wait_armed(False)

    def wait_armed(self, armed: bool) -> None:
        """Wait until the vehicle reports the given armed state."""
        end = time.monotonic() + 5.0
        while self.state.armed != armed:
            if time.monotonic() > end:
                raise RuntimeError("Vehicle did not change its armed state")
            rclpy.spin_once(self, timeout_sec=0.05)

    def voltage(self) -> float:
        """Return the latest battery voltage in V."""
        return float(self.battery.voltage)


def infer_cells(voltage: float) -> int:
    """Infer the number of cells in series from the pack voltage of a charged battery."""
    low, high = CELL_VOLTAGE_RANGE
    candidates = [n for n in range(1, 13) if low <= voltage / n <= high]
    if not candidates:
        raise RuntimeError(
            f"No cell count gives {low} to {high} V per cell at {voltage:.2f} V. "
            "Charge the battery or check the voltage measurement."
        )
    if len(candidates) > 1:
        raise RuntimeError(
            f"{voltage:.2f} V fits {candidates} cells, charge the battery to make the count unique."
        )
    return candidates[0]


def confirm_cells(voltage: float) -> int:
    """Infer the cell count and let the user confirm it on the terminal."""
    cells = infer_cells(voltage)
    # input() writes its prompt to stderr, print it so it is visible when stderr is redirected.
    print(
        f"Battery at {voltage:.2f} V, that is {cells} cells at {voltage / cells:.2f} V each. "
        "Type y to confirm: ",
        end="",
        flush=True,
    )
    answer = input()
    if answer.strip().lower() != "y":
        raise RuntimeError("Cell count not confirmed, aborting")
    return cells


def run(args: argparse.Namespace) -> None:
    """Arm in offboard mode and command random thrust values until the battery is low."""
    if os.environ.get("RMW_IMPLEMENTATION") != REQUIRED_RMW:
        print(
            f"Warning: RMW_IMPLEMENTATION is not {REQUIRED_RMW}, the MAVROS services will hang "
            "with Fast DDS on this setup.",
            file=sys.stderr,
        )
    # rclpy must not tear down its context on Ctrl+C, the test has to disarm the vehicle first.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = RandomThrustNode(args.namespace)
    signal.signal(signal.SIGINT, lambda signum, frame: setattr(node, "stop", True))
    rng = np.random.default_rng(args.seed)
    try:
        node.wait_for_data()
        cells = confirm_cells(node.voltage())
        node.thrust = 0.0
        node.spin_for(WARMUP)  # offboard mode is only accepted with a running setpoint stream
        node.set_offboard()
        node.arm()
        start = time.monotonic()
        thrust = 0.0  # the first interval is silent, it marks the start of the recording
        while True:
            node.thrust = thrust
            print(f"{thrust:.2f}", flush=True)
            node.spin_for(args.interval)
            if node.stop:
                print("Stopped")
                break
            if not node.state.armed or node.state.mode != "OFFBOARD":
                print(f"Vehicle left the test (armed={node.state.armed}, mode={node.state.mode})")
                break
            if node.voltage() / cells < args.min_cell_voltage:
                print(f"Battery low: {node.voltage():.2f} V, stopping")
                break
            if args.duration and time.monotonic() - start > args.duration * 60:
                print("Duration over, stopping")
                break
            thrust = round(float(rng.uniform(args.min_thrust, args.max_thrust)), 2)
    finally:
        if rclpy.ok():
            node.thrust = 0.0
            node.stop = False  # let the setpoints run down before disarming
            node.spin_for(1.0)
            if node.state.armed:
                node.force_disarm()
            node.destroy_node()
            rclpy.shutdown()


def main() -> None:
    """Parse the arguments and run the test."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--namespace", default="", help="MAVROS namespace, e.g. /drone1")
    parser.add_argument(
        "--min-cell-voltage", type=float, default=3.5, help="stop below this cell voltage in V"
    )
    parser.add_argument(
        "--interval", type=float, default=4.0, help="seconds between two thrust values"
    )
    parser.add_argument(
        "--duration", type=float, default=0.0, help="stop after this many minutes, 0 = only battery"
    )
    parser.add_argument("--min-thrust", type=float, default=0.0, help="lowest random thrust")
    parser.add_argument("--max-thrust", type=float, default=1.0, help="highest random thrust")
    parser.add_argument("--seed", type=int, default=0, help="seed of the random thrust sequence")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
