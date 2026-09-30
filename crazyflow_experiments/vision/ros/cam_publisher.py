"""Basic script that waits for new images from the Magewell capture card and published them to ROS2.

In case the Magewell capture card does not open (RuntimeError: Could not open /dev/video0),
you might need to repair the driver manully:
cd /tmp
sudo mwcap-repair.sh
"""

#!/usr/bin/env python3

import argparse
import os
import threading
import time
from pathlib import Path

# A 1280x720 BGR image is about 2.76 MB. Enlarge Fast DDS's default transport
# buffers before rclpy creates a participant; sockets_size=4MB gives its SHM
# transport an 8MB segment. Keep the default transport set so ordinary ROS
# tools remain compatible. An explicit user setting still wins.
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


class CaptureCardNode(Node):
    STATS_INTERVAL = 2.0

    def __init__(self, publish_distorted: bool = False):
        super().__init__("capture_card")

        folder = Path(__file__).parent.parent
        config = toml.load(folder / "config.toml")
        drone_id = config["drone"]["id"]
        drone_name = f"cf{drone_id}"

        self.device = "/dev/video0"
        self.width = 1280
        self.height = 720
        self.fps = 60.0
        self.frame_period = 1.0 / self.fps
        self.publish_distorted = publish_distorted

        self._calibration_model = None
        self._calibration_size = None
        self._camera_matrix = None
        self._distortion_coefficients = None
        self._undistort_size = None
        self._undistort_maps = None

        if not self.publish_distorted:
            self._load_calibration(Path(__file__).with_name("cam_params.toml"))

        # For live video we don't want ROS building another queue of old images.
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.publisher = self.create_publisher(Image, f"/drones/{drone_name}/cam", qos)

        # ------------------------------------------------------------------
        # Open camera
        # ------------------------------------------------------------------

        self.cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open {self.device}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        # Keep the V4L2 queue small.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        reported_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.get_logger().info(
            "Camera opened: "
            f"{self.cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x"
            f"{self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}; "
            f"publish limit {self.fps:.2f} fps "
            f"(driver reports {reported_fps:.2f} fps)"
        )
        self.get_logger().info(
            f"ROS middleware: {rclpy.get_rmw_implementation_identifier()}; "
            f"Fast DDS transports: {os.environ['FASTDDS_BUILTIN_TRANSPORTS']}"
        )
        if self.publish_distorted:
            self.get_logger().info("Publishing distorted camera images")
        else:
            self.get_logger().info(
                f"Publishing undistorted camera images using {self._calibration_model} "
                f"calibration ({self._calibration_size[0]}x{self._calibration_size[1]})"
            )

        # ------------------------------------------------------------------
        # One-frame mailbox shared between camera thread and ROS executor
        # ------------------------------------------------------------------

        self._lock = threading.Lock()

        self._latest_frame = None
        self._publishing = False

        self._dropped_frames = 0
        self._last_reported_drops = 0
        self._last_warning_time = 0.0

        self._captured_frames = 0
        self._rate_limited_frames = 0
        self._published_frames = 0
        self._last_stats_time = time.monotonic()
        self._last_stats_captured = 0
        self._last_stats_rate_limited = 0
        self._last_stats_published = 0
        self._last_stats_dropped = 0
        self._message_build_time = 0.0
        self._message_build_time_max = 0.0
        self._publish_time = 0.0
        self._publish_time_max = 0.0

        self._stop_event = threading.Event()

        # This is the "hook":
        #
        # camera thread:
        #       frame becomes ready
        #              |
        #              v
        #       guard.trigger()
        #
        # ROS executor wakes:
        #              |
        #              v
        #       _on_frame_ready()
        #
        self._frame_guard = self.create_guard_condition(self._on_frame_ready)

        self._capture_thread = threading.Thread(
            target=self._capture_loop, name="hdzero_capture", daemon=True
        )
        self._capture_thread.start()

    def _load_calibration(self, path: Path):
        config = toml.load(path)
        model = str(config["model"]).upper()

        supported_models = {"SIMPLE_RADIAL", "OPENCV", "FULL_OPENCV"}
        if model not in supported_models:
            supported = ", ".join(sorted(supported_models))
            raise ValueError(f"Unsupported camera model {model!r}; choose one of: {supported}")

        calibration = config["calibration"]
        width = int(calibration["width"])
        height = int(calibration["height"])
        if width <= 0 or height <= 0:
            raise ValueError("Calibration width and height must be positive")

        params = config["models"][model]
        cx = float(params["cx"])
        cy = float(params["cy"])

        if model == "SIMPLE_RADIAL":
            fx = fy = float(params["f"])
            distortion_coefficients = [float(params["k"]), 0.0, 0.0, 0.0]
        elif model == "OPENCV":
            fx = float(params["fx"])
            fy = float(params["fy"])
            distortion_coefficients = [float(params[name]) for name in ("k1", "k2", "p1", "p2")]
        else:
            fx = float(params["fx"])
            fy = float(params["fy"])
            distortion_coefficients = [
                float(params[name]) for name in ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")
            ]

        self._calibration_model = model
        self._calibration_size = (width, height)
        self._camera_matrix = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
        )
        self._distortion_coefficients = np.asarray(distortion_coefficients, dtype=np.float64)

    def _undistort(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        frame_size = (width, height)

        if self._undistort_size != frame_size:
            calibration_width, calibration_height = self._calibration_size
            scale_x = width / calibration_width
            scale_y = height / calibration_height

            camera_matrix = self._camera_matrix.copy()
            camera_matrix[0, 0] *= scale_x
            camera_matrix[0, 2] *= scale_x
            camera_matrix[1, 1] *= scale_y
            camera_matrix[1, 2] *= scale_y

            self._undistort_maps = cv2.initUndistortRectifyMap(
                camera_matrix,
                self._distortion_coefficients,
                None,
                camera_matrix,
                frame_size,
                cv2.CV_16SC2,
            )
            self._undistort_size = frame_size
            self.get_logger().info(
                f"Built undistortion map for {width}x{height} images "
                f"(intrinsic scale {scale_x:.3f}x{scale_y:.3f})"
            )

        map1, map2 = self._undistort_maps
        return cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)

    # ----------------------------------------------------------------------
    # Camera thread
    # ----------------------------------------------------------------------

    def _capture_loop(self):
        next_publish_time = time.monotonic()

        while not self._stop_event.is_set():
            # For a live V4L2 camera this is the blocking acquisition point.
            # There is no sleep/poll loop while the camera is operating normally.
            ok, frame = self.cap.read()

            if not ok:
                # Avoid a busy loop if the device starts failing.
                self._stop_event.wait(0.01)
                continue

            capture_time = time.monotonic()

            # Some capture cards ignore CAP_PROP_FPS and return frames faster
            # than requested. Continue reading so their V4L2 queue stays fresh,
            # but only hand a frame to ROS at the configured rate.
            if capture_time < next_publish_time:
                with self._lock:
                    self._captured_frames += 1
                    self._rate_limited_frames += 1
                continue

            next_publish_time += self.frame_period
            if next_publish_time <= capture_time:
                # Do not burst old frame slots after a camera stall.
                next_publish_time = capture_time + self.frame_period

            with self._lock:
                self._captured_frames += 1
                # If ROS hasn't finished with the previous frame, do NOT make
                # a queue. Throw away the stale frame and keep the newest one.
                if self._latest_frame is not None or self._publishing:
                    self._dropped_frames += 1

                self._latest_frame = frame

            # Wake the ROS executor.
            self._frame_guard.trigger()

    # ----------------------------------------------------------------------
    # Runs in ROS executor thread
    # ----------------------------------------------------------------------

    def _on_frame_ready(self):
        with self._lock:
            frame = self._latest_frame

            if frame is None:
                return

            self._latest_frame = None
            self._publishing = True

        start = time.perf_counter()
        message_build_elapsed = 0.0
        publish_elapsed = 0.0
        published = False

        try:
            message_build_start = time.perf_counter()
            if not self.publish_distorted:
                frame = self._undistort(frame)

            msg = Image()

            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "capture_card"

            msg.height = frame.shape[0]
            msg.width = frame.shape[1]

            # OpenCV VideoCapture normally gives us BGR uint8.
            msg.encoding = "bgr8"

            msg.is_bigendian = False
            msg.step = frame.strides[0]

            # ROS message owns its data, so this makes a copy.
            msg.data = frame.tobytes()
            message_build_elapsed = time.perf_counter() - message_build_start

            publish_start = time.perf_counter()
            self.publisher.publish(msg)
            publish_elapsed = time.perf_counter() - publish_start
            published = True

        finally:
            elapsed = time.perf_counter() - start

            with self._lock:
                self._publishing = False

                # A frame may have arrived while we were publishing.
                frame_waiting = self._latest_frame is not None
                dropped = self._dropped_frames

                if published:
                    self._published_frames += 1
                    self._message_build_time += message_build_elapsed
                    self._message_build_time_max = max(
                        self._message_build_time_max, message_build_elapsed
                    )
                    self._publish_time += publish_elapsed
                    self._publish_time_max = max(self._publish_time_max, publish_elapsed)

            # Important:
            # GuardConditions aren't being used as a counting semaphore here.
            # If a frame arrived while this callback was executing, explicitly
            # wake the executor again.
            if frame_waiting:
                self._frame_guard.trigger()

        # ------------------------------------------------------------------
        # Can't-keep-up warning
        # ------------------------------------------------------------------

        now = time.monotonic()

        new_drops = dropped - self._last_reported_drops

        if new_drops > 0 and now - self._last_warning_time >= 1.0:
            self.get_logger().warning(
                f"Camera publisher cannot keep up: "
                f"{new_drops} frame(s) overwritten. "
                f"Last publish path took {elapsed * 1000.0:.2f} ms; "
                f"frame budget is {self.frame_period * 1000.0:.2f} ms."
            )

            self._last_reported_drops = dropped
            self._last_warning_time = now

        self._report_stats(now)

    def _report_stats(self, now):
        elapsed = now - self._last_stats_time
        if elapsed < self.STATS_INTERVAL:
            return

        with self._lock:
            captured = self._captured_frames
            rate_limited = self._rate_limited_frames
            published = self._published_frames
            dropped = self._dropped_frames
            message_build_time = self._message_build_time
            message_build_time_max = self._message_build_time_max
            publish_time = self._publish_time
            publish_time_max = self._publish_time_max

            self._message_build_time = 0.0
            self._message_build_time_max = 0.0
            self._publish_time = 0.0
            self._publish_time_max = 0.0

        new_captured = captured - self._last_stats_captured
        new_rate_limited = rate_limited - self._last_stats_rate_limited
        new_published = published - self._last_stats_published
        new_dropped = dropped - self._last_stats_dropped

        message_build_avg_ms = message_build_time / new_published * 1000.0 if new_published else 0.0
        publish_avg_ms = publish_time / new_published * 1000.0 if new_published else 0.0

        self.get_logger().info(
            f"Publisher stats: capture {new_captured / elapsed:.1f} fps, "
            f"publish {new_published / elapsed:.1f} fps, "
            f"rate-limited {new_rate_limited}, overwritten {new_dropped}, "
            f"message build avg/max {message_build_avg_ms:.2f}/"
            f"{message_build_time_max * 1000.0:.2f} ms, DDS publish avg/max "
            f"{publish_avg_ms:.2f}/{publish_time_max * 1000.0:.2f} ms"
        )

        self._last_stats_time = now
        self._last_stats_captured = captured
        self._last_stats_rate_limited = rate_limited
        self._last_stats_published = published
        self._last_stats_dropped = dropped

    # ----------------------------------------------------------------------

    def stop(self):
        self._stop_event.set()

        # Usually cap.read() returns every ~16.7 ms at 60 fps, so the thread
        # naturally exits almost immediately.
        self._capture_thread.join(timeout=1.0)

        self.cap.release()


def main(publish_distorted: bool = False):
    rclpy.init()

    node = CaptureCardNode(publish_distorted=publish_distorted)

    try:
        # This is not busy spinning. The executor sleeps in its wait set until
        # something (our GuardCondition, shutdown, etc.) wakes it.
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--publish_distorted",
        action="store_true",
        help="publish the raw distorted image instead of applying camera calibration",
    )
    cli_args = parser.parse_args()
    main(publish_distorted=cli_args.publish_distorted)
