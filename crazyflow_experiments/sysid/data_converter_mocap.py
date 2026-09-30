"""Converts rosbags of flights under motion capture into a csv file.

The rosbags are recorded during the identification flights of `collect_data.py` as described in the
README, with the motion capture pose of the drone and the commands the deployment logs. The drone
can be a Crazyflie or a PX4 vehicle, the only requirement is that it flies in the motion capture
space, so its pose is published on `/tf`. `data_converter_px4.py` is the counterpart for a PX4
vehicle outside of it, whose pose comes from its onboard estimate through MAVROS.

Topics:
    - `/tf`: the motion capture pose of the drone, the transform whose child frame is the drone name
      `cf`. Every pose becomes one csv row with its position in m and its quaternion in SciPy's
      scalar-last order [x, y, z, w].
    - `/drones/<cf>/command`: the attitude command the deployment logs, [roll, pitch, yaw, thrust]
      in the units of the drone. Every row holds the latest command before the pose as `cmd_rpy0`
      to `cmd_rpy2` and `cmd_pwm`, `fit_simplified_model.py` converts them with the drone's
      parameters.

Usage:

    pixi run -e sysid python crazyflow_experiments/sysid/data_converter_mocap.py

Set `cf` to the drone name and `bag_paths` to the recordings, or `process_all_bags` to convert every
bag in the rosbag directory. The script plots every recording first and asks for the start and end
time in seconds since the first message of the bag, then writes `<bag>.csv` next to the bag.
"""

import csv
import os
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
from scipy.spatial.transform import Rotation as R

from crazyflow_experiments.sysid import RECORDINGS_DIR

cf = "cf10"  # Set to correct cf number
bags_path = RECORDINGS_DIR / "rosbags"
process_all_bags = False  # Decide if you want to process all bags in the bags path

if process_all_bags:
    bag_paths = [x[0] for x in os.walk(bags_path)]
    del bag_paths[0]
else:
    # Replace with actual path
    bag_paths = [
        # f"{bags_path}/data_{cf}L_sysid_train_2025-04-28",
        # f"{bags_path}/data_{cf}L_sysid_valid_2025-04-28",
        # f"{bags_path}/data_cf2x_P250_sysid_train_2025-11-14",  # cf52
        # f"{bags_path}/data_cf2x_P250_sysid_valid_2025-11-14",  # cf52
        # f"{bags_path}/data_cf2x_L250_sysid_train_2025-11-15",  # cf52
        # f"{bags_path}/data_cf2x_L250_sysid_valid_2025-11-15",  # cf52
        f"{bags_path}/data_cf2x_T350_sysid_train_2026-05-07",  # cf10
        f"{bags_path}/data_cf2x_T350_sysid_valid_2026-05-07",  # cf10
    ]

output_csvs = [f"{bag_path}.csv" for bag_path in bag_paths]


def extract_and_plot(bag_path: str):
    """Read the poses of a bag and plot them, to pick the start and end time of the flight."""
    typestore = get_typestore(Stores.ROS2_FOXY)

    tf_data = []
    command_data = deque()
    timestamp0 = 0

    with Reader(bag_path) as reader:
        for connection, timestamp, rawdata in reader.messages():
            timestamp = timestamp / 1e9  # Convert ns to s
            if timestamp0 == 0:
                timestamp0 = timestamp
            timestamp -= timestamp0

            msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
            if connection.topic == "/tf":  # Extract transform specifically
                for transform in msg.transforms:
                    if transform.child_frame_id == cf:
                        tf_data.append((timestamp, transform))
            elif connection.topic == f"/drones/{cf}/command":
                command_data.append((timestamp, msg.data))

    plot_tf_data(tf_data)


def extract_and_save_to_csv(
    bag_path: str, output_csv: str, start_time: float | None = None, end_time: float | None = None
):
    """Write one csv row per pose between the start and end time with the latest command before it.

    Poses before the first command are dropped.
    """
    typestore = get_typestore(Stores.ROS2_FOXY)

    tf_data = []
    command_data = deque()
    timestamp0 = 0

    with Reader(bag_path) as reader:
        for connection, timestamp, rawdata in reader.messages():
            timestamp = timestamp / 1e9  # Convert ns to s
            if timestamp0 == 0:
                timestamp0 = timestamp
            timestamp -= timestamp0
            if start_time and timestamp < start_time:
                continue
            if end_time and timestamp > end_time:
                break

            msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
            if connection.topic == "/tf":  # Extract transform specifically for cf6
                for transform in msg.transforms:
                    if transform.child_frame_id == cf:
                        tf_data.append((timestamp, transform))
            elif connection.topic == f"/drones/{cf}/command":
                command_data.append((timestamp, msg.data))

    with open(output_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "time",
                "pos0",
                "pos1",
                "pos2",
                "quat0",
                "quat1",
                "quat2",
                "quat3",
                "cmd_rpy0",
                "cmd_rpy1",
                "cmd_rpy2",
                "cmd_pwm",
            ]
        )

        last_command = None
        for tf_timestamp, tf_msg in tf_data:
            while command_data and command_data[0][0] <= tf_timestamp:
                last_command = command_data.popleft()[1]

            if last_command is not None:
                writer.writerow(
                    [
                        tf_timestamp,
                        tf_msg.transform.translation.x,
                        tf_msg.transform.translation.y,
                        tf_msg.transform.translation.z,
                        tf_msg.transform.rotation.x,
                        tf_msg.transform.rotation.y,
                        tf_msg.transform.rotation.z,
                        tf_msg.transform.rotation.w,
                        last_command[0],
                        last_command[1],
                        last_command[2],
                        last_command[3],
                    ]
                )


def plot_tf_data(tf_data: list, command_data: deque | None = None):
    """Plot the motion capture position and Euler angles over time."""
    timestamps = [t for t, _ in tf_data]
    x_values = [msg.transform.translation.x for _, msg in tf_data]
    y_values = [msg.transform.translation.y for _, msg in tf_data]
    z_values = [msg.transform.translation.z for _, msg in tf_data]

    qx_values = np.array([msg.transform.rotation.x for _, msg in tf_data])
    qy_values = np.array([msg.transform.rotation.y for _, msg in tf_data])
    qz_values = np.array([msg.transform.rotation.z for _, msg in tf_data])
    qw_values = np.array([msg.transform.rotation.w for _, msg in tf_data])
    quat = np.stack((qx_values, qy_values, qz_values, qw_values)).T
    rpy = R.from_quat(quat).as_euler("xyz")

    fig, ax = plt.subplots(2, figsize=(10, 8))
    ax[0].plot(timestamps, x_values, label="Position X")
    ax[0].plot(timestamps, y_values, label="Position Y")
    ax[0].plot(timestamps, z_values, label="Position Z")
    ax[0].set_xlabel("Time [s]")
    ax[0].set_ylabel("Postion [m]")
    ax[0].legend()
    ax[1].plot(timestamps, rpy[:, 0], label="Roll")
    ax[1].plot(timestamps, rpy[:, 1], label="Pitch")
    ax[1].plot(timestamps, rpy[:, 2], label="Yaw")
    ax[1].set_xlabel("Time [s]")
    ax[1].set_ylabel("Rotation [rad]")
    ax[1].legend()
    fig.suptitle(f"Raw Vicon Data for {cf}")
    plt.show()


if __name__ == "__main__":
    for i in range(len(bag_paths)):
        print(f"{i + 1}/{len(bag_paths)}")
        bag_path = bag_paths[i]
        output_csv = output_csvs[i]

        # First, look at the plot and decide where to start and stop
        extract_and_plot(bag_path)
        start_time = float(input("start time: "))
        end_time = float(input("end time (-1 for end): "))
        if end_time <= 0:
            end_time = None

        # Then actually cut and store the data
        extract_and_save_to_csv(bag_path, output_csv, start_time=start_time, end_time=end_time)
