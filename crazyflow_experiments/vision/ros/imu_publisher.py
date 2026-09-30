"""Publish Crazyflie IMU log data as ROS messages."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu

if TYPE_CHECKING:
    from cflib.crazyflie.log import LogConfig


class ImuPublisher(Node):
    """Adapt a cflib log callback to a ROS ``sensor_msgs/Imu`` topic."""

    GRAVITY = 9.81

    def __init__(self, drone_name: str):
        super().__init__(f"{drone_name}_imu_publisher")

        self._frame_id = f"{drone_name}/imu"
        self._clock = Clock(clock_type=ClockType.SYSTEM_TIME)
        self._publisher = self.create_publisher(
            Imu, f"/drones/{drone_name}/imu", qos_profile_sensor_data
        )

    def cf_log_callback(
        self, _drone_timestamp: float, data: dict[str, float], _logconf: LogConfig
    ) -> None:
        """Publish one cflib IMU log sample using the host's system timestamp."""
        msg = Imu()
        msg.header.stamp = self._clock.now().to_msg()
        msg.header.frame_id = self._frame_id

        # No orientation estimate is included in the Crazyflie log block.
        msg.orientation_covariance[0] = -1.0

        msg.linear_acceleration.x = data["acc.x"] * self.GRAVITY
        msg.linear_acceleration.y = data["acc.y"] * self.GRAVITY
        msg.linear_acceleration.z = data["acc.z"] * self.GRAVITY

        msg.angular_velocity.x = np.deg2rad(data["gyro.x"])
        msg.angular_velocity.y = np.deg2rad(data["gyro.y"])
        msg.angular_velocity.z = np.deg2rad(data["gyro.z"])

        self._publisher.publish(msg)
