#!/usr/bin/env python3

"""Display images published by the capture-card ROS node."""

import argparse
import os
import threading
import time
from pathlib import Path
from typing import Optional

# Match the publisher's enlarged Fast DDS buffers. Preserve an explicit setting
# supplied by the user or launch environment.
os.environ.setdefault(
    "FASTDDS_BUILTIN_TRANSPORTS", "DEFAULT?max_msg_size=64KB&sockets_size=4MB&non_blocking=true"
)

import cv2
import numpy as np
import rclpy
import toml
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


class CameraViewer(Node):
    """Receive camera frames and display them or retain the latest RGB image."""

    WINDOW_NAME = "Magewell HDZero"
    STATS_INTERVAL = 2.0

    def __init__(self, headless: bool = False):
        """Subscribe to camera frames with an optional display window."""
        super().__init__("camera_viewer")
        self.headless = headless

        folder = Path(__file__).parent.parent
        config = toml.load(folder / "config.toml")
        drone_id = config["drone"]["id"]
        drone_name = f"cf{drone_id}"

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.subscription = self.create_subscription(
            Image, f"/drones/{drone_name}/cam", self._on_image, qos
        )

        # Keep ROS reception independent from the GUI. The subscription callback
        # only replaces this one-message mailbox, while the main thread owns all
        # OpenCV window calls.
        self._lock = threading.Lock()
        self._image_ready = threading.Event()
        self._latest_msg = None
        self._latest_image = None

        self._received_frames = 0
        self._received_bytes = 0
        self._displayed_frames = 0
        self._mailbox_overwrites = 0
        self._latest_age_ms = None
        self._latest_display_ms = None

        self._last_stats_time = time.monotonic()
        self._last_received_frames = 0
        self._last_received_bytes = 0
        self._last_displayed_frames = 0
        self._last_mailbox_overwrites = 0

        if not self.headless:
            # WINDOW_NORMAL makes the window interactively resizable. KEEPRATIO
            # prevents the camera image from being stretched while resizing it.
            cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    def _on_image(self, msg: Image) -> None:
        if self.headless:
            frame = self._decode_image(msg, as_rgb=True)
            if frame is not None:
                # Detach the stored RGB image from the ROS message buffer.
                frame = frame.copy()

            with self._lock:
                if frame is not None:
                    self._latest_image = frame
                self._received_frames += 1
                self._received_bytes += len(msg.data)

            return

        with self._lock:
            if self._latest_msg is not None:
                self._mailbox_overwrites += 1

            self._latest_msg = msg
            self._received_frames += 1
            self._received_bytes += len(msg.data)
            self._image_ready.set()

    def get_latest_image(self) -> Optional[np.ndarray]:
        """Return a copy of the latest RGB image, or None before one arrives."""
        with self._lock:
            if self._latest_image is None:
                return None
            return self._latest_image.copy()

    def _take_latest_image(self, timeout: float) -> Optional[Image]:
        if not self._image_ready.wait(timeout):
            return None

        with self._lock:
            msg = self._latest_msg
            self._latest_msg = None
            self._image_ready.clear()

        return msg

    def _decode_image(self, msg: Image, as_rgb: bool = False) -> Optional[np.ndarray]:
        if msg.encoding not in ("bgr8", "rgb8"):
            self.get_logger().warning(
                f"Unsupported image encoding: {msg.encoding}; expected bgr8 or rgb8"
            )
            return None

        row_size = msg.width * 3
        if msg.step < row_size:
            self.get_logger().warning(
                f"Invalid image step {msg.step}; expected at least {row_size}"
            )
            return None

        data = np.frombuffer(msg.data, dtype=np.uint8)
        required_size = msg.height * msg.step
        if data.size < required_size:
            self.get_logger().warning(
                f"Incomplete image data: got {data.size} bytes, expected {required_size}"
            )
            return None

        rows = data[:required_size].reshape(msg.height, msg.step)
        frame = rows[:, :row_size].reshape(msg.height, msg.width, 3)

        if msg.encoding == "bgr8" and as_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif msg.encoding == "rgb8" and not as_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        return frame

    def display_once(self, timeout: float = 0.1) -> bool:
        """Display the newest frame and return whether the viewer should continue."""
        msg = self._take_latest_image(timeout)
        display_start = time.perf_counter()

        if msg is not None:
            frame = self._decode_image(msg)
            if frame is not None:
                cv2.imshow(self.WINDOW_NAME, frame)

                stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
                age_ms = None
                if stamp_ns:
                    age_ms = (self.get_clock().now().nanoseconds - stamp_ns) / 1e6

                with self._lock:
                    self._displayed_frames += 1
                    self._latest_age_ms = age_ms

        # ESC or q exits, as in readcam.py.
        key = cv2.waitKey(1) & 0xFF
        display_ms = (time.perf_counter() - display_start) * 1000.0

        with self._lock:
            self._latest_display_ms = display_ms

        self._report_stats()

        if key == 27 or key == ord("q"):
            self.get_logger().info("Stopping camera viewer")
            return False

        return True

    def _report_stats(self):
        now = time.monotonic()
        elapsed = now - self._last_stats_time
        if elapsed < self.STATS_INTERVAL:
            return

        with self._lock:
            received_frames = self._received_frames
            received_bytes = self._received_bytes
            displayed_frames = self._displayed_frames
            mailbox_overwrites = self._mailbox_overwrites
            age_ms = self._latest_age_ms
            display_ms = self._latest_display_ms

        receive_fps = (received_frames - self._last_received_frames) / elapsed
        receive_mbps = (received_bytes - self._last_received_bytes) * 8.0 / elapsed / 1e6
        display_fps = (displayed_frames - self._last_displayed_frames) / elapsed
        overwrites = mailbox_overwrites - self._last_mailbox_overwrites

        age_text = "n/a" if age_ms is None else f"{age_ms:.1f} ms"
        display_text = "n/a" if display_ms is None else f"{display_ms:.1f} ms"
        self.get_logger().info(
            f"Viewer stats: receive {receive_fps:.1f} fps / {receive_mbps:.1f} Mbit/s, "
            f"display {display_fps:.1f} fps, mailbox overwrites {overwrites}, "
            f"frame age {age_text}, last GUI update {display_text}"
        )

        self._last_stats_time = now
        self._last_received_frames = received_frames
        self._last_received_bytes = received_bytes
        self._last_displayed_frames = displayed_frames
        self._last_mailbox_overwrites = mailbox_overwrites

    def stop(self):
        """Close the display window when one was opened."""
        if not self.headless:
            cv2.destroyAllWindows()


def main(headless: bool = False):
    """Run the camera subscriber until the viewer or ROS stops."""
    rclpy.init()
    node = CameraViewer(headless=headless)
    spin_thread = None

    try:
        if headless:
            rclpy.spin(node)
        else:
            spin_thread = threading.Thread(
                target=rclpy.spin, args=(node,), name="camera_viewer_ros", daemon=True
            )
            spin_thread.start()

            while rclpy.ok() and node.display_once():
                pass
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()
        if spin_thread is not None:
            spin_thread.join(timeout=1.0)
        node.stop()
        node.destroy_node()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--headless",
        action="store_true",
        help="store the latest image without creating an OpenCV window",
    )
    cli_args = parser.parse_args()
    main(headless=cli_args.headless)
