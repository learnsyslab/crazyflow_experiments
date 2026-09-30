"""Converts the ulogs of a PX4 vehicle into the csv files of the sysid pipeline.

The counterpart of `data_converter_mocap.py` for a PX4 vehicle. It reads the flight controller's
own log rather than a rosbag of the MAVROS topics: every sample carries the microsecond timestamp of
the flight controller, so the derivatives the fits need are not disturbed by the MAVLink link. It
writes the same csv columns, so `fit_simplified_model.py` consumes the result unchanged.

Localization: the flight is flown with `localization = "gps"` or `"mocap"` in `[drone.px4]`, and
the log says which, so the converter needs no flag. The estimator's own fusion flags in
`estimator_status_flags` decide: `cs_ev_pos` marks the external vision fusion PX4 runs on the motion
capture pose MAVROS feeds it, `cs_gnss_pos` marks GNSS. The mode picks the origin of the global
frame and the signals that say when the estimate cannot be trusted.

Topics:
    - `vehicle_local_position`: PX4's position estimate in NED, one csv row per sample. Needs the
      full rate of 100 Hz, which the logger only writes with `vehicle_local_position 0` in
      `/fs/microsd/etc/logging/logger_topics.txt`, the default set logs it at 10 Hz.
    - `vehicle_attitude`: the attitude estimate, the latest sample before every position.
    - `vehicle_attitude_setpoint`: the attitude and thrust setpoints as PX4 received them, the
      latest before every position, as roll, pitch and yaw in rad in the global frame and the
      normalized thrust command in [0, 1] as `cmd_pwm`.
    - `home_position`: with `"gps"` the origin of the global frame, the home position at the start
      of the window, the takeoff position like in `deploy/px4.py`. With `"mocap"` the origin is the
      estimator origin, which is the motion capture origin, so the topic is not read: PX4 never gets
      a global position indoors, so its home position is not the takeoff position.
    - `vehicle_control_mode`: finds the trajectory. `deploy/experiment.py` flies to the start pose
      with position setpoints, then the controller sends attitude setpoints from the soft start to
      the end of the trajectory, then `hold` sends position setpoints again. The stretch of offboard
      flight without position control is the trajectory, and the window defaults to it, from the
      soft start out of the takeoff hover to the zero-size trajectory that ends every set with a
      hover.
    - `estimator_status_flags`: the fusion flags that tell the two localizations apart.
    - `vehicle_gps_position` and `estimator_aid_src_gnss_pos` with `"gps"`: the fix type and the
      GNSS fusion status, to cut the stretches where the estimate cannot be trusted.
    - `vehicle_visual_odometry` and `estimator_aid_src_ev_pos` with `"mocap"`: the motion capture
      poses as PX4 received them and the external vision fusion status, for the same purpose.

Frames: PX4 works in NED with the body axes forward, right, down, the pipeline in the global frame
north, west, up with the body axes forward, left, up. Both pairs differ by a half turn about the x
axis, so positions flip their y and z and a rotation R becomes D R D with D = diag(1, -1, -1).

Cutting: a second around every estimator reset is dropped, since the estimate jumps there. The rest
depends on the localization. With `"gps"`, a loss of the RTK fix is dropped only when the estimate
suffered: when the estimator rejected GNSS positions afterwards, when its own position uncertainty
grew beyond `EPH_MAX_GNSS`, or when the loss lasted longer than `LONG_FIX_LOSS`. With `"mocap"`, a
gap in the pose stream longer than `MAX_VISION_GAP`, the staleness the deployment watches, is
dropped under the same rule, when it lasted longer than `LONG_VISION_LOSS` or the estimate suffered,
and every stretch where the estimator rejected the pose, dead reckoned, or reported an uncertainty
above `EPH_MAX_VISION` is dropped as well. A short loss that the estimator absorbed is kept and
reported in both modes, because the derivative filters need contiguous data and a splice costs more
than a small blip. The remaining valid stretches are written as separate csvs, `<log>.csv` for a
single stretch, `<log>_<n>.csv` otherwise.

Usage:

    pixi run -e sysid python crazyflow_experiments/sysid/data_converter_px4.py

Set `log_paths` to the logs in `results/sysid/ulogs`, or `process_all_logs` to convert every ulog
there. Name a log after its flight before converting, e.g. `data_x500_sysid_train_2026-09-08.ulg`,
because `fit_simplified_model.py` checks the drone model in the csv name. The script plots every log
with the detected trajectory window and the cut stretches, asks for the start and end time in
seconds since the start of the log with the detected values as defaults, then writes the csvs next
to the log. The default window is the trajectory widened by `WINDOW_MARGIN` on both sides, because
`vehicle_control_mode` is logged at 2 Hz and the mode switch is only seen at the next sample; the
flight hovers on both sides of the trajectory, so the margin only adds hover. Reading the logs needs
pyulog.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from pyulog import ULog
from scipy.spatial.transform import Rotation as R

from crazyflow_experiments.sysid import RECORDINGS_DIR

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

logs_path = RECORDINGS_DIR / "ulogs"
process_all_logs = False  # Decide if you want to process all logs in the logs path

if process_all_logs:
    log_paths = sorted(logs_path.glob("*.ulg"))
else:
    # Replace with actual path
    log_paths = [
        logs_path / "data_x500_sysid_train_2026-09-13_mocap.ulg",
        logs_path / "data_x500_sysid_valid_2026-09-13_mocap.ulg",
    ]

CSV_HEADER = [
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

TOPICS = (
    "vehicle_local_position",
    "vehicle_attitude",
    "vehicle_attitude_setpoint",
    "home_position",
    "vehicle_control_mode",
    "estimator_status_flags",
    "vehicle_gps_position",
    "estimator_aid_src_gnss_pos",
    "vehicle_visual_odometry",
    "estimator_aid_src_ev_pos",
)
REQUIRED_TOPICS = TOPICS[:3]  # Without these the log holds no flight, `home_position` only for gps
RTK_FIXED = 6  # fix_type of an RTK fixed solution
LONG_FIX_LOSS = 3.0  # s, a longer loss of the RTK fix is cut even if the estimator kept fusing
AFTER_FIX_RETURN = 6.0  # s, GNSS rejections within this time after the fix returns extend the cut
EPH_MAX_GNSS = 0.3  # m, position uncertainty of the estimator above which GNSS data is cut
MAX_VISION_GAP = 0.2  # s, a longer gap in the motion capture stream is a dropout, as in deploy/px4
LONG_VISION_LOSS = 0.5  # s, a longer dropout is cut even if the estimator kept fusing
EPH_MAX_VISION = 0.15  # m, position uncertainty above which motion capture data is cut
MARGIN = 0.5  # s, cut before a loss of the measurements and after the estimator recovers
AROUND_RESET = 1.0  # s, cut on either side of an estimator state reset
WINDOW_MARGIN = 0.5  # s, widen the detected trajectory, vehicle_control_mode is logged at 2 Hz
MAX_GAP = 0.1  # s, a longer gap between positions splits the data
MIN_SEGMENT = 10.0  # s, shorter valid stretches are not written
# NED with body axes forward, right, down to north, west, up with body axes forward, left, up:
# a half turn about x, which is its own inverse
GLOBAL_FROM_NED = R.from_euler("x", np.pi)
RESET_COUNTERS = (
    "xy_reset_counter",
    "z_reset_counter",
    "vxy_reset_counter",
    "vz_reset_counter",
    "heading_reset_counter",
)


@dataclass
class Recording:
    """The pose and command samples of one flight in the global frame.

    Timestamps are in s since the start of the log.

    Attributes:
        t_pos: Timestamps of the positions as (N,).
        pos: Positions in m as (N, 3).
        pos_valid: Whether the estimator reported the position as valid, as (N,).
        t_att: Timestamps of the attitudes as (M,).
        quat: Attitudes as SciPy quaternions [x, y, z, w] as (M, 4).
        t_cmd: Timestamps of the attitude setpoints as (K,).
        cmd_rpy: Setpoint roll, pitch and yaw in rad as (K, 3).
        cmd_thrust: Normalized setpoint thrust in [0, 1] as (K,).
        invalid: Stretches (start, end, reason) in s to drop.
        trajectory: Start and end in s of the trajectory the controller flew, widened by
            `WINDOW_MARGIN`, None if the log holds no offboard flight with attitude setpoints.
        localization: Where the estimate came from, "gps" or "mocap".
    """

    t_pos: NDArray
    pos: NDArray
    pos_valid: NDArray
    t_att: NDArray
    quat: NDArray
    t_cmd: NDArray
    cmd_rpy: NDArray
    cmd_thrust: NDArray
    invalid: list[tuple[float, float, str]]
    trajectory: tuple[float, float] | None
    localization: str


def rotation_from_px4(q: NDArray) -> R:
    """Rotations of the global frame from PX4 quaternions [w, x, y, z] of NED to the body frame."""
    return GLOBAL_FROM_NED * R.from_quat(q[:, [1, 2, 3, 0]]) * GLOBAL_FROM_NED


def topic(ulog: ULog, name: str, start: float) -> dict[str, NDArray]:
    """The fields of a topic with its timestamps converted to s since the start of the log."""
    data = dict(ulog.get_dataset(name).data)
    data["time"] = data["timestamp"] / 1e6 - start
    if "timestamp_sample" in data and np.any(data["timestamp_sample"] > 0):
        data["time"] = data["timestamp_sample"] / 1e6 - start
    return data


def blocks(flags: NDArray, time: NDArray) -> list[tuple[float, float]]:
    """Start and end times of the stretches where the flags are set."""
    edges = np.diff(np.concatenate([[0], flags.astype(int), [0]]))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return [(time[i], time[min(j, len(time) - 1)]) for i, j in zip(starts, ends)]


def trajectory_window(mode: dict[str, NDArray]) -> tuple[float, float] | None:
    """Start and end of the longest offboard stretch flown with attitude setpoints.

    Widened by `WINDOW_MARGIN` on both sides: `vehicle_control_mode` is logged at 2 Hz, so the mode
    switch is only seen at the next sample, and the flight hovers on both sides of the trajectory,
    so the margin recovers the switch instant and adds nothing but hover.
    """
    attitude_only = (mode["flag_control_offboard_enabled"] > 0) & (
        mode["flag_control_position_enabled"] == 0
    )
    stretches = blocks(attitude_only, mode["time"])
    if not stretches:
        return None
    start, end = max(stretches, key=lambda stretch: stretch[1] - stretch[0])
    return start - WINDOW_MARGIN, end + WINDOW_MARGIN


def localization(ulog: ULog, logged: set[str]) -> str:
    """Whether the estimate rested on the motion capture system or on GNSS.

    The estimator's own fusion flags decide: `cs_ev_pos` marks the external vision fusion PX4 runs
    on the motion capture pose, `cs_gnss_pos` marks GNSS. A log without the flags falls back to the
    presence of the motion capture stream.
    """
    if "estimator_status_flags" not in logged:
        return "mocap" if "vehicle_visual_odometry" in logged else "gps"
    flags = ulog.get_dataset("estimator_status_flags").data
    return "mocap" if np.sum(flags["cs_ev_pos"]) > np.sum(flags["cs_gnss_pos"]) else "gps"


def reset_intervals(lpos: dict[str, NDArray]) -> list[tuple[float, float, str]]:
    """Stretches to drop around the estimator state resets, where the estimate jumps."""
    intervals = []
    for counter in RESET_COUNTERS:
        for i in np.flatnonzero(np.diff(lpos[counter].astype(int)) != 0):
            reset = lpos["time"][i + 1]
            intervals.append((reset - AROUND_RESET, reset + AROUND_RESET, f"{counter} reset"))
    return intervals


def gnss_intervals(
    gps: dict[str, NDArray], lpos: dict[str, NDArray], aid: dict[str, NDArray] | None
) -> list[tuple[float, float, str]]:
    """Stretches to drop where the RTK fix was lost and the estimate suffered.

    Args:
        gps: `vehicle_gps_position` with times.
        lpos: `vehicle_local_position` with times.
        aid: `estimator_aid_src_gnss_pos` with times, None if the log lacks it.
    """
    intervals = []
    rejected = None
    if aid is not None:
        rejected = aid["time"][(aid["fused"] == 0) | (aid["innovation_rejected"] > 0)]
    for lost, returned in blocks(gps["fix_type"] < RTK_FIXED, gps["time"]):
        end = returned
        reasons = []
        if returned - lost > LONG_FIX_LOSS:
            reasons.append(f"fix lost for {returned - lost:.1f} s")
        if rejected is not None:
            late = rejected[(rejected >= lost) & (rejected <= returned + AFTER_FIX_RETURN)]
            if len(late):
                end = max(end, late[-1])
                reasons.append("GNSS positions rejected")
        after = (lpos["time"] >= lost) & (lpos["time"] <= returned + AFTER_FIX_RETURN)
        uncertain = lpos["time"][after & (lpos["eph"] > EPH_MAX_GNSS)]
        if len(uncertain):
            end = max(end, uncertain[-1])
            reasons.append(f"position uncertainty above {EPH_MAX_GNSS} m")
        if reasons:
            intervals.append((lost - MARGIN, end + MARGIN, ", ".join(reasons)))
        else:
            report_absorbed("RTK fix", lost, returned, lpos)
    return intervals


def vision_intervals(
    vision: dict[str, NDArray], lpos: dict[str, NDArray], aid: dict[str, NDArray] | None
) -> list[tuple[float, float, str]]:
    """Stretches to drop where the motion capture pose was missing and the estimate suffered.

    A gap in the pose stream longer than `MAX_VISION_GAP` is a dropout, the staleness
    `deploy/px4.py` watches during the flight. It is dropped when it lasted longer than
    `LONG_VISION_LOSS` or the estimator rejected the pose afterwards, and kept otherwise. Every
    stretch where the estimator rejected the pose, dead reckoned or grew uncertain is dropped on its
    own, so a degraded estimate is cut even without a gap in the stream.

    Args:
        vision: `vehicle_visual_odometry` with times.
        lpos: `vehicle_local_position` with times.
        aid: `estimator_aid_src_ev_pos` with times, None if the log lacks it.
    """
    intervals = []
    rejected = None
    if aid is not None:
        bad = (aid["fused"] == 0) | (aid["innovation_rejected"] > 0)
        rejected = aid["time"][bad]
        for start, end in blocks(bad, aid["time"]):
            intervals.append((start - MARGIN, end + MARGIN, "motion capture pose rejected"))
    for start, end in blocks(lpos["dead_reckoning"] > 0, lpos["time"]):
        intervals.append((start - MARGIN, end + MARGIN, "estimator dead reckoning"))
    for start, end in blocks(lpos["eph"] > EPH_MAX_VISION, lpos["time"]):
        reason = f"position uncertainty above {EPH_MAX_VISION} m"
        intervals.append((start - MARGIN, end + MARGIN, reason))
    gaps = np.flatnonzero(np.diff(vision["time"]) > MAX_VISION_GAP)
    for lost, returned in zip(vision["time"][gaps], vision["time"][gaps + 1]):
        reasons = []
        if returned - lost > LONG_VISION_LOSS:
            reasons.append(f"motion capture pose lost for {returned - lost:.1f} s")
        if rejected is not None and np.any((rejected >= lost) & (rejected <= returned + MARGIN)):
            reasons.append("motion capture pose rejected")
        if reasons:
            intervals.append((lost - MARGIN, returned + MARGIN, ", ".join(reasons)))
        else:
            report_absorbed("motion capture pose", lost, returned, lpos)
    return intervals


def report_absorbed(measurement: str, lost: float, returned: float, lpos: dict[str, NDArray]):
    """Report a loss of the measurements that the estimator absorbed, which is kept in the data."""
    eph = lpos["eph"][(lpos["time"] >= lost) & (lpos["time"] <= returned + MARGIN)]
    print(
        f"{measurement} lost {lost:.1f}-{returned:.1f} s, the estimator absorbed it "
        f"(uncertainty up to {np.max(eph, initial=0.0):.3f} m), kept"
    )


def merge_intervals(intervals: list[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    """The intervals sorted by start, with the overlapping ones joined and their reasons kept."""
    merged: list[tuple[float, float, str]] = []
    for start, end, reason in sorted(intervals):
        if merged and start <= merged[-1][1]:
            last = merged[-1]
            merged[-1] = (last[0], max(last[1], end), f"{last[2]}; {reason}")
        else:
            merged.append((start, end, reason))
    return merged


def read_log(
    log_path: Path, start_time: float | None = None, end_time: float | None = None
) -> Recording:
    """Read the pose and command topics of a PX4 flight from a ulog.

    Args:
        log_path: The ulog file.
        start_time: Drop samples before this time in s since the start of the log. None takes the
            start of the trajectory, or of the log if there is none.
        end_time: Drop samples after this time in s since the start of the log. None takes the end
            of the trajectory, or of the log if there is none.

    Returns:
        The recording in the global frame, whose origin is the home position at the start of the
        window with `"gps"` and the motion capture origin with `"mocap"`.
    """
    ulog = ULog(str(log_path), message_name_filter_list=list(TOPICS))
    logged = {data.name for data in ulog.data_list}
    for name in REQUIRED_TOPICS:
        if name not in logged:
            raise ValueError(f"{log_path} holds no {name}")
    start = ulog.start_timestamp / 1e6
    lpos, att, cmd = (topic(ulog, name, start) for name in REQUIRED_TOPICS)
    mode = localization(ulog, logged)

    trajectory = None
    if "vehicle_control_mode" in logged:
        trajectory = trajectory_window(topic(ulog, "vehicle_control_mode", start))
    if trajectory is None:
        print("No offboard flight with attitude setpoints in the log, the window defaults to all")
        trajectory_start, trajectory_end = lpos["time"][0], lpos["time"][-1]
    else:
        trajectory_start, trajectory_end = trajectory
    start_time = trajectory_start if start_time is None else start_time
    end_time = trajectory_end if end_time is None else end_time

    # With "mocap" the estimator origin is the motion capture origin, so the frame already fits.
    # With "gps" the origin is the home position PX4 had when the window starts, home moves while
    # disarmed.
    origin_ned = np.zeros(3)
    if mode == "gps":
        if "home_position" not in logged:
            raise ValueError(f"{log_path} holds no home_position, the origin of the global frame")
        home = topic(ulog, "home_position", start)
        i_home = max(latest_before(np.array([start_time]), home["time"])[0], 0)
        origin_ned = np.array([home["x"][i_home], home["y"][i_home], home["z"][i_home]])
        if not home["valid_lpos"][i_home]:
            raise ValueError(f"{log_path} has no valid home position at {start_time:.1f} s")

    invalid = reset_intervals(lpos)
    measurements = "vehicle_gps_position" if mode == "gps" else "vehicle_visual_odometry"
    aid_name = "estimator_aid_src_gnss_pos" if mode == "gps" else "estimator_aid_src_ev_pos"
    if measurements in logged:
        aid = topic(ulog, aid_name, start) if aid_name in logged else None
        source = topic(ulog, measurements, start)
        detect = gnss_intervals if mode == "gps" else vision_intervals
        invalid += detect(source, lpos, aid)
    else:
        print(f"No {measurements} in the log, losses of the measurements are not cut")
    invalid = merge_intervals(invalid)

    def window(data: dict[str, NDArray]) -> NDArray:
        return (data["time"] >= start_time) & (data["time"] <= end_time)

    in_pos, in_att, in_cmd = window(lpos), window(att), window(cmd)
    if not np.any(in_pos) or not np.any(in_att) or not np.any(in_cmd):
        raise ValueError(f"{log_path} has no position, attitude or setpoint in the selected window")
    pos_ned = np.stack([lpos["x"], lpos["y"], lpos["z"]], axis=1)[in_pos]
    q_att = np.stack([att[f"q[{i}]"] for i in range(4)], axis=1)[in_att]
    q_cmd = np.stack([cmd[f"q_d[{i}]"] for i in range(4)], axis=1)[in_cmd]
    return Recording(
        t_pos=lpos["time"][in_pos],
        pos=GLOBAL_FROM_NED.apply(pos_ned - origin_ned),
        pos_valid=(lpos["xy_valid"][in_pos] > 0) & (lpos["z_valid"][in_pos] > 0),
        t_att=att["time"][in_att],
        quat=rotation_from_px4(q_att).as_quat(),
        t_cmd=cmd["time"][in_cmd],
        cmd_rpy=rotation_from_px4(q_cmd).as_euler("xyz"),
        # thrust_body points down in the body frame, so the collective thrust command is negative
        cmd_thrust=np.clip(-cmd["thrust_body[2]"][in_cmd], 0.0, 1.0),
        invalid=invalid,
        trajectory=trajectory,
        localization=mode,
    )


def latest_before(t_query: NDArray, t_samples: NDArray) -> NDArray:
    """Index of the latest sample at or before every query time, -1 where there is none."""
    return np.searchsorted(t_samples, t_query, side="right") - 1


def valid_segments(recording: Recording) -> list[NDArray]:
    """Index arrays of the contiguous stretches of positions that can be written.

    A position counts when the estimator reported it valid, an attitude and a setpoint precede it
    and it lies outside every invalid stretch. A gap longer than `MAX_GAP` splits a stretch, and
    stretches shorter than `MIN_SEGMENT` are dropped.
    """
    i_att = latest_before(recording.t_pos, recording.t_att)
    i_cmd = latest_before(recording.t_pos, recording.t_cmd)
    valid = recording.pos_valid & (i_att >= 0) & (i_cmd >= 0)
    for start, end, _ in recording.invalid:
        valid &= (recording.t_pos < start) | (recording.t_pos > end)
    indices = np.flatnonzero(valid)
    if len(indices) == 0:
        return []
    breaks = np.flatnonzero(np.diff(recording.t_pos[indices]) > MAX_GAP) + 1
    segments = np.split(indices, breaks)
    return [s for s in segments if recording.t_pos[s[-1]] - recording.t_pos[s[0]] >= MIN_SEGMENT]


def save_to_csv(recording: Recording, segment: NDArray, output_csv: Path):
    """Write one csv row per position of the segment with the latest attitude and setpoint."""
    i_att = latest_before(recording.t_pos[segment], recording.t_att)
    i_cmd = latest_before(recording.t_pos[segment], recording.t_cmd)
    with open(output_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(CSV_HEADER)
        for i, j, k in zip(segment, i_att, i_cmd):
            writer.writerow(
                [
                    recording.t_pos[i],
                    *recording.pos[i],
                    *recording.quat[j],
                    *recording.cmd_rpy[k],
                    recording.cmd_thrust[k],
                ]
            )


def plot_recording(recording: Recording, title: str):
    """Plot position, attitude and setpoints, the trajectory window and the cut stretches."""
    rpy = R.from_quat(recording.quat).as_euler("xyz")

    fig, ax = plt.subplots(3, figsize=(10, 10), sharex=True)
    for i, label in enumerate(("Position X", "Position Y", "Position Z")):
        ax[0].plot(recording.t_pos, recording.pos[:, i], label=label)
    ax[0].set_ylabel("Position [m]")
    for i, label in enumerate(("Roll", "Pitch", "Yaw")):
        ax[1].plot(recording.t_att, rpy[:, i], label=label)
        ax[1].plot(recording.t_cmd, recording.cmd_rpy[:, i], "--", label=f"{label} setpoint")
    ax[1].set_ylabel("Rotation [rad]")
    ax[2].plot(recording.t_cmd, recording.cmd_thrust, label="Thrust setpoint")
    ax[2].set_ylabel("Normalized thrust")
    ax[2].set_xlabel("Time since the start of the log [s]")
    extra = []
    if recording.invalid:
        lost = "fix loss" if recording.localization == "gps" else "motion capture loss"
        extra.append(Patch(color="red", alpha=0.15, label=f"Cut: estimator reset or {lost}"))
    if recording.trajectory is not None:
        extra.append(plt.Line2D([], [], color="k", ls=":", label="Trajectory start and end"))
    for axis in ax:
        for start, end, _ in recording.invalid:
            axis.axvspan(start, end, color="red", alpha=0.15)
        if recording.trajectory is not None:
            for edge in recording.trajectory:
                axis.axvline(edge, color="k", ls=":")
    ax[0].legend(handles=ax[0].get_legend_handles_labels()[0] + extra, ncol=2)
    ax[1].legend(ncol=2)
    ax[2].legend()
    fig.suptitle(title)
    plt.show()


def ask_time(prompt: str, default: float) -> float:
    """Ask for a time in s, an empty answer takes the default."""
    answer = input(f"{prompt} [{default:.1f}]: ").strip()
    return default if answer == "" else float(answer)


if __name__ == "__main__":
    for n, log_path in enumerate(log_paths):
        print(f"{n + 1}/{len(log_paths)}: {log_path}")

        # First, look at the plot and decide where to start and stop
        full = read_log(log_path, start_time=-np.inf, end_time=np.inf)
        print(f"Estimate localized with {full.localization}")
        if full.trajectory is not None:
            print(f"Trajectory flown {full.trajectory[0]:.1f}-{full.trajectory[1]:.1f} s")
        for start, end, reason in full.invalid:
            print(f"Cut {start:.1f}-{end:.1f} s: {reason}")
        plot_recording(full, title=f"Raw estimate and setpoint data of {log_path.name}")
        default_start, default_end = full.trajectory or (full.t_pos[0], full.t_pos[-1])
        start_time = ask_time("start time", default_start)
        end_time = ask_time("end time", default_end)

        # Then actually cut and store the data
        recording = read_log(log_path, start_time, end_time)
        segments = valid_segments(recording)
        if not segments:
            raise ValueError(f"{log_path} has no valid stretch of {MIN_SEGMENT} s in the window")
        for k, segment in enumerate(segments):
            suffix = "" if len(segments) == 1 else f"_{k}"
            output_csv = log_path.with_name(f"{log_path.stem}{suffix}.csv")
            save_to_csv(recording, segment, output_csv)
            span = recording.t_pos[segment[-1]] - recording.t_pos[segment[0]]
            print(f"Saved {output_csv}: {len(segment)} rows, {span:.1f} s")
