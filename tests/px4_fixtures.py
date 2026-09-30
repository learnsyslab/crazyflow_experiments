"""Fake MAVROS node for the PX4 interface tests.

Publishes a static state estimate, home position, GNSS solution and motion capture pose on the MAVROS
topics in its ENU frame, answers the message interval and the setpoint_raw parameter services and
records every setpoint it receives, so the interface can be exercised without a vehicle.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import GPSRAW, AttitudeTarget, HomePosition, PositionTarget, State
from mavros_msgs.srv import MessageInterval
from rcl_interfaces.msg import SetParametersResult
from rcl_interfaces.srv import SetParameters
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Imu

LOCAL_POSITION_NED = 32


class FakeMavros(Node):
    """Stand-in for the MAVROS node of a hovering vehicle. All values are in MAVROS's ENU frame."""

    def __init__(self):
        """Create the publishers, the service and the recording subscriptions, and spin."""
        super().__init__("fake_mavros")
        self.mode = "AUTO.LOITER"
        self.home = np.array([0.5, -1.0, 2.0])
        self.pos = np.array([1.0, 2.0, 3.0])
        self.yaw = 0.5
        self.vel = np.array([0.1, 0.2, 0.3])
        self.gyro = np.array([0.01, 0.02, 0.03])
        self.rate = 30.0  # Hz, PX4's default local position rate for the onboard link
        self.gps_rate = 20.0  # Hz, the rate GPS_RAW_INT is streamed at, see px4.GPS_RATE
        self.vision_rate = 50.0  # Hz, the motion capture pose the tracking node feeds to PX4
        self.gps = {  # An RTK fixed solution as GPS_RAW_INT carries it
            "fix_type": 6, "lat": 435000000, "lon": -795000000, "alt": 120000, "eph": 80,
            "epv": 120, "satellites_visible": 22, "h_acc": 14, "v_acc": 21, "vel_acc": 30,
            "dgps_age": 1200, "dgps_numch": 18,
        }  # fmt: skip
        self.sensors_on = True
        self.vision_on = True
        self.interval_requests: list[tuple[int, float]] = []
        self.parameters: dict[str, float] = {}
        self._position_targets: list[tuple[float, PositionTarget]] = []
        self._attitude_targets: list[tuple[float, AttitudeTarget]] = []
        self._lock = threading.Lock()
        qos = qos_profile_sensor_data
        # MAVROS publishes the state and the home position reliable and transient local
        state_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._pose_pub = self.create_publisher(PoseStamped, "/mavros/local_position/pose", qos)
        self._vel_pub = self.create_publisher(
            TwistStamped, "/mavros/local_position/velocity_local", qos
        )
        self._imu_pub = self.create_publisher(Imu, "/mavros/imu/data", qos)
        self._state_pub = self.create_publisher(State, "/mavros/state", state_qos)
        self._home_pub = self.create_publisher(
            HomePosition, "/mavros/home_position/home", state_qos
        )
        self._gps_pub = self.create_publisher(GPSRAW, "/mavros/gpsstatus/gps1/raw", qos)
        self._vision_pub = self.create_publisher(PoseStamped, "/mavros/vision_pose/pose", qos)
        self.create_subscription(
            PositionTarget, "/mavros/setpoint_raw/local", self._on_position_target, qos
        )
        self.create_subscription(
            AttitudeTarget, "/mavros/setpoint_raw/attitude", self._on_attitude_target, qos
        )
        self.create_service(MessageInterval, "/mavros/set_message_interval", self._on_interval)
        self.create_service(
            SetParameters, "/mavros/setpoint_raw/set_parameters", self._on_set_parameters
        )
        self._sensor_timer = self.create_timer(1 / self.rate, self._publish_sensors)
        self.create_timer(1.0, self.publish_state)
        self.create_timer(0.5, self._publish_home)
        self.create_timer(1 / self.gps_rate, self._publish_gps)
        self.create_timer(1 / self.vision_rate, self._publish_vision)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def set_mode(self, mode: str):
        """Switch the flight mode like the RC mode switch and announce it right away."""
        self.mode = mode
        self.publish_state()

    def targets(self, kind: str, since: float = 0.0) -> list[PositionTarget | AttitudeTarget]:
        """Return the received position or attitude targets, optionally since a monotonic time."""
        with self._lock:
            items = self._position_targets if kind == "position" else self._attitude_targets
            return [msg for t, msg in items if t >= since]

    def publish_state(self):
        """Publish the vehicle state like the MAVROS heartbeat handler."""
        msg = State()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.connected = True
        msg.armed = True
        msg.guided = self.mode == "OFFBOARD"
        msg.mode = self.mode
        self._state_pub.publish(msg)

    def close(self):
        """Stop spinning and destroy the node."""
        self._stop.set()
        self._thread.join()
        self._executor.shutdown()
        self.destroy_node()

    def _spin(self):
        while not self._stop.is_set() and rclpy.ok():
            self._executor.spin_once(timeout_sec=0.05)

    def _on_position_target(self, msg: PositionTarget):
        with self._lock:
            self._position_targets.append((time.monotonic(), msg))

    def _on_attitude_target(self, msg: AttitudeTarget):
        with self._lock:
            self._attitude_targets.append((time.monotonic(), msg))

    def _on_set_parameters(
        self, request: SetParameters.Request, response: SetParameters.Response
    ) -> SetParameters.Response:
        for parameter in request.parameters:
            self.parameters[parameter.name] = parameter.value.double_value
        response.results = [SetParametersResult(successful=True) for _ in request.parameters]
        return response

    def _on_interval(
        self, request: MessageInterval.Request, response: MessageInterval.Response
    ) -> MessageInterval.Response:
        self.interval_requests.append((request.message_id, request.message_rate))
        if request.message_id == LOCAL_POSITION_NED:
            self.rate = request.message_rate
            self._sensor_timer.cancel()
            self._sensor_timer = self.create_timer(1 / self.rate, self._publish_sensors)
        response.success = True
        return response

    def _publish_home(self):
        msg = HomePosition()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.position.x, msg.position.y, msg.position.z = self.home
        self._home_pub.publish(msg)

    def _publish_gps(self):
        if not self.sensors_on:
            return
        msg = GPSRAW(**self.gps)
        msg.header.stamp = self.get_clock().now().to_msg()
        self._gps_pub.publish(msg)

    def _publish_vision(self):
        # The motion capture pose on its way into PX4, published by the tracking node next to MAVROS
        if not self.vision_on:
            return
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = self.pos
        self._vision_pub.publish(msg)

    def _publish_sensors(self):
        if not self.sensors_on:
            return
        stamp = self.get_clock().now().to_msg()
        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = self.pos
        x, y, z, w = R.from_euler("z", self.yaw).as_quat()
        pose.pose.orientation.x, pose.pose.orientation.y = x, y
        pose.pose.orientation.z, pose.pose.orientation.w = z, w
        self._pose_pub.publish(pose)
        twist = TwistStamped()
        twist.header.stamp = stamp
        twist.twist.linear.x, twist.twist.linear.y, twist.twist.linear.z = self.vel
        # MAVROS puts the world frame angular velocity here, which the interface must not use
        twist.twist.angular.x, twist.twist.angular.y, twist.twist.angular.z = 9.0, 9.0, 9.0
        self._vel_pub.publish(twist)
        imu = Imu()
        imu.header.stamp = stamp
        imu.orientation = pose.pose.orientation
        imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z = self.gyro
        self._imu_pub.publish(imu)
