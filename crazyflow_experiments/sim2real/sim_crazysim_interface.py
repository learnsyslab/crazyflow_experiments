"""Interfaces to the CrazySim SITL firmware and its physics backends.

`SitlCrazyflie` owns the cflib link to the software-in-the-loop firmware and everything both
physics backends share, so the firmware receives identical commands regardless of the backend.
`GzCrazyflie` adds the Gazebo side (ground-truth odometry over gz transport), `MjCrazyflie` runs
CrazySim's MuJoCo backend in-process and reads ground truth straight from ``MjData``.

All of this runs only inside the container built from `sim2real/CrazySimDocker`, see
`run_crazysim.py`.
"""

from __future__ import annotations

import os

os.environ["SCIPY_ARRAY_API"] = "1"

import importlib.util
import logging
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

import cflib.crtp
import numpy as np
from cflib.crazyflie import Crazyflie
from crazyflow.control import Control
from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    from types import ModuleType

    from gz.msgs.odometry_pb2 import Odometry
    from numpy.typing import NDArray


# Set logging to capture connection errors
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

CRAZYSIM_ROOT = Path(os.environ.get("CRAZYSIM_ROOT", "/root/CrazySim"))
MJ_BACKEND_DIR = (
    CRAZYSIM_ROOT / "crazyflie-firmware/tools/crazyflie-simulation/simulator_files/mujoco"
)
CF2_BINARY = CRAZYSIM_ROOT / "crazyflie-firmware/sitl_make/build/cf2"
FIRMWARE_PORT = 19950
HANDSHAKE_TIMEOUT = 20.0  # s to wait for the firmware to register with the MuJoCo sim
MAX_BACKLOG = 0.002  # s the MuJoCo stepping loop may lag its schedule before the backlog is dropped
GIL_SWITCH_INTERVAL = 0.0005  # s, keeps the control loop from blocking the stepping thread for 5 ms


class SitlCrazyflie:
    """Crazyflie SITL firmware interface over the cflib UDP link.

    Owns everything the simulation backends share: the cflib connection, the firmware settings,
    the arming sequence, the action-to-setpoint conversion, and the teardown. Subclasses connect
    a physics backend and implement :meth:`get_obs`.
    """

    def __init__(self, drone_model: str, ctrl_interface: Control, uri: str = "udp://0.0.0.0:19850"):
        """Connect to the SITL firmware and prepare it for flight."""
        assert drone_model == "cf2x_L250"
        self.ctrl_interface = ctrl_interface
        self.uri = uri
        self.is_connected = False
        self.drone = self._connect_to_drone(uri)
        self._reset_drone()

    def _connect_to_drone(self, uri: str) -> Crazyflie:
        cflib.crtp.init_drivers()

        drone = Crazyflie(rw_cache="./cache_sim")

        def cb_fully_connected(_):
            self.is_connected = True

        drone.fully_connected.add_callback(cb_fully_connected)
        drone.connected.add_callback(lambda _: logger.info(f"Connected to {uri}"))
        drone.disconnected.add_callback(lambda _: logger.warning("Disconnected"))
        drone.connection_failed.add_callback(
            lambda _, msg: logger.error(f"Connection failed: {msg}")
        )
        drone.connection_lost.add_callback(lambda _, msg: logger.warning(f"Connection lost: {msg}"))

        logger.info(f"Connecting to {uri}...")
        drone.open_link(uri)

        # Wait for connection with timeout
        start = time.time()
        while self.is_connected is False:
            time.sleep(0.1)
            if time.time() - start > 10.0:
                raise TimeoutError(f"Connection to {uri} timed out. Is the sim running?")

        logger.info("Ready to fly!")
        return drone

    def _reset_drone(self):
        self._apply_drone_settings()
        self.drone.commander.send_setpoint(0, 0, 0, 0)  # unlock drone
        self.drone.platform.send_arming_request(True)  # unlock brushless drone
        self.drone.commander.send_setpoint(0.0, 0.0, 0.0, 7000)  # send min thrust to start motors
        time.sleep(1.5)  # Wait for settings to be applied and propellers to spin up

    def _apply_drone_settings(self):
        """Apply firmware settings to the drone.

        Note:
            These settings are also required to make the high-level drone commander work properly.
        """
        # Estimator setting;  1: complementary, 2: kalman -> Manual test: kalman significantly better!
        self.drone.param.set_value("stabilizer.estimator", 2)
        # Choose controller: 1: PID; 2:Mellinger
        self.drone.param.set_value("stabilizer.controller", 2)
        # rate: 0, angle: 1
        self.drone.param.set_value("flightmode.stabModeRoll", 1)
        self.drone.param.set_value("flightmode.stabModePitch", 1)
        self.drone.param.set_value("flightmode.stabModeYaw", 1)
        time.sleep(0.1)  # Wait for settings to be applied

    def send_action(self, action: NDArray):
        """Send the action to the drone."""
        if not self.is_connected:
            return

        if self.ctrl_interface == Control.attitude:
            # Convert: Rad -> Deg for RPY, and Thrust -> PWM
            r_deg = np.rad2deg(action[0])
            p_deg = np.rad2deg(action[1])
            y_deg = np.rad2deg(action[2])
            thrust_pwm = int(self._thrust2pwm(action[3]))

            self.drone.commander.send_setpoint(r_deg, p_deg, y_deg, thrust_pwm)
        else:
            raise NotImplementedError("State control not implemented for CrazySim")

    def update_estimators(self, obs: dict):
        """Sends position and orientations to update the onboard estimators."""
        # Note: cflib expects (x, y, z)
        if self.is_connected:
            self.drone.extpos.send_extpos(*obs["pos"])

    def close(self):
        """Closes the deployment."""
        # Send a stop setpoint first to kill motors
        self.drone.commander.send_stop_setpoint()
        time.sleep(0.1)
        self.drone.close_link()

    def get_obs(self) -> dict:
        """Return the latest observation dictionary."""
        raise NotImplementedError("Simulation backends must implement get_obs")

    def sim_time(self) -> float:
        """Return the current simulation time in seconds."""
        raise NotImplementedError("Simulation backends must implement sim_time")

    def _thrust2pwm(self, thrust: float) -> float:
        # Simple linear mapping based on your drone params
        THRUST_MAX = 0.12
        PWM_MIN = 7000
        PWM_MAX = 65535
        pwm = thrust / 4 / THRUST_MAX * PWM_MAX
        return np.clip(pwm, PWM_MIN, PWM_MAX)


class GzCrazyflie(SitlCrazyflie):
    """Gazebo backend: the SITL link plus ground-truth odometry from the gz transport topic."""

    def __init__(
        self,
        drone_model: str,
        ctrl_interface: Control,
        uri: str = "udp://0.0.0.0:19850",
        topic_name: str = "/cf_0/odom",
    ):
        """Connect to the SITL firmware and subscribe to the Gazebo odometry topic."""
        # Import here: the gz-python bindings are only needed for the Gazebo backend.
        from gz.transport import Node, SubscribeOptions

        super().__init__(drone_model, ctrl_interface, uri)

        self.node = Node()
        self.topic = topic_name
        self.latest_obs = {}
        self.start_time = None
        self._sim_time = 0.0

        if not self.node.subscribe(
            self.topic, self._odometry_callback, "gz.msgs.Odometry", SubscribeOptions()
        ):
            raise ConnectionError(f"Could not subscribe to {self.topic}")

        logger.info(f"[GazeboObs] Listening to {self.topic}...")

        # Block until we get the first message from Gazebo
        wait_start = time.time()
        while not self.latest_obs:
            if time.time() - wait_start > 5.0:
                logger.warning("Gazebo not publishing? Check if simulation is paused.")
            time.sleep(0.1)

    def get_obs(self) -> dict:
        """Return the latest observation dictionary."""
        return self.latest_obs

    def sim_time(self) -> float:
        """Simulation time of the latest odometry message (published at 200 Hz)."""
        return self._sim_time

    def _odometry_callback(self, msg: Odometry):
        p = msg.pose.position
        pos = np.array([p.x, p.y, p.z])
        q = msg.pose.orientation
        quat = np.array([q.x, q.y, q.z, q.w])

        # Efficient RPY calculation (simpler than full scipy object if speed matters)
        rpy = R.from_quat(quat).as_euler("xyz")

        v = msg.twist.linear
        vel = np.array([v.x, v.y, v.z])
        w = msg.twist.angular
        ang_vel = np.array([w.x, w.y, w.z])

        current_time = msg.header.stamp.sec + (msg.header.stamp.nsec * 1e-9)
        if self.start_time is None:
            self.start_time = current_time
        t = current_time - self.start_time
        self._sim_time = current_time

        self.latest_obs = {
            "pos": pos,
            "quat": quat,
            "vel": vel,
            "ang_vel": ang_vel,
            "rpy": rpy,
            "t": t,
        }


def load_mj_backend(reaction_torque: bool = False, crazyflow_rotor: bool = False) -> ModuleType:
    """Load CrazySim's MuJoCo backend module from the container installation.

    Args:
        reaction_torque: Add the body reaction torque of the accelerating rotors
            (J_prop * sum_i dir_i * rotor_acc_i about the body z-axis). The backend computes the
            rotor acceleration but never applies this torque, so its only yaw authority is the
            rotor drag torque, which lags by the rotor time constant. Without the term the
            firmware's yaw loop is unstable at the identified time constant; crazyflow's
            first_principles dynamics include it.
        crazyflow_rotor: Integrate the rotor speeds with crazyflow's identified rotor model
            (``rotor_dyn_coef`` of the drone parameters: linear spin-up, quadratic spin-down,
            tau_up 0.136 s and tau_down 0.108 s at hover) instead of the backend's symmetric
            first-order model with ``rotor_dyn_coef_simple`` (tau 0.145 s). The firmware's rate
            loop has almost no phase margin against a 0.145 s first-order rotor, and the
            symmetric model makes its attitude loop ring up in flight.
    """
    spec = importlib.util.spec_from_file_location("crazysim_mj", MJ_BACKEND_DIR / "crazysim.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if crazyflow_rotor:
        from crazyflow.dynamics import Dynamics, load_params

        params = load_params(Dynamics.first_principles, "cf2x_L250")
        acc1, acc2, dec1, dec2 = params["rotor_dyn_coef"]

        def update_motors_crazyflow(self: object):
            # Same force/torque conversion as the backend's update_motors, only the rotor speed
            # integration is replaced by crazyflow's model.
            params = self._params
            with self._motor_lock:
                rpm_ref = self._rpm_ref.copy()
            rpm = self._rpm
            rpm_dot = np.where(
                rpm_ref >= rpm,
                acc1 * (rpm_ref - rpm) + acc2 * (rpm_ref**2 - rpm**2),
                dec1 * (rpm_ref - rpm) + dec2 * (rpm_ref**2 - rpm**2),
            )
            rpm_new = np.clip(rpm + rpm_dot * self.dt, 0.0, params.max_rpm)
            self._rpm_dot = (rpm_new - rpm) / self.dt
            self._rpm = rpm_new
            ge_factor = self._compute_ground_effect() if self._ground_effect else 1.0
            a_t, b_t, c_t = params.rpm2thrust
            a_q, b_q, c_q = params.rpm2torque
            for i in range(4):
                thrust = max(a_t + b_t * rpm_new[i] + c_t * rpm_new[i] ** 2, 0.0) * ge_factor
                torque = module.MOTOR_DIR[i] * (a_q + b_q * rpm_new[i] + c_q * rpm_new[i] ** 2)
                self.data.ctrl[self._act_force[i]] = thrust
                self.data.ctrl[self._act_torque[i]] = torque

        module.DroneAgent.update_motors = update_motors_crazyflow
    if reaction_torque:
        apply_aero_effects = module.DroneAgent.apply_aero_effects

        def apply_aero_and_reaction(self: object):
            apply_aero_effects(self)
            rotor_acc = self._rpm_dot * module.RPM_TO_RADS
            torque_z = float(np.sum(module.MOTOR_DIR * self._params.prop_inertia * rotor_acc))
            self.data.qfrc_applied[self._qvel_adr + 5] += torque_z

        module.DroneAgent.apply_aero_effects = apply_aero_and_reaction
    return module


class MjCrazyflie(SitlCrazyflie):
    """MuJoCo backend: the SITL link plus CrazySim's MuJoCo sim, run in-process.

    CrazySim ships a second physics backend under `simulator_files/mujoco` (a MuJoCo scene plus a
    motor-level rotor model with parameters from the vendored utiasDSL drone-models). It is loaded
    lazily, so this module imports without the mujoco package installed.
    """

    def __init__(
        self,
        drone_model: str,
        ctrl_interface: Control,
        uri: str = "udp://0.0.0.0:19850",
        reaction_torque: bool = False,
        crazyflow_rotor: bool = False,
        original_pacing: bool = False,
    ):
        """Start the MuJoCo sim and the SITL firmware, then connect to the firmware.

        Args:
            drone_model: Drone model name, only cf2x_L250 is supported.
            ctrl_interface: Control interface the actions are sent through.
            uri: cflib URI of the SITL firmware.
            reaction_torque: Add the rotor reaction torque the backend omits, see
                `load_mj_backend`.
            crazyflow_rotor: Use crazyflow's identified rotor model, see `load_mj_backend`.
            original_pacing: Pace the physics like CrazySim's own loop does (sleep ``dt`` minus
                the step time after every step), which runs at a real-time factor of about 0.91
                here, instead of against absolute deadlines. Only for reproducing the paper-era
                behaviour, see `_run_sim`.
        """
        assert drone_model == "cf2x_L250"
        self._original_pacing = original_pacing
        backend = load_mj_backend(reaction_torque, crazyflow_rotor)
        self._mujoco = backend.mujoco
        model_path = MJ_BACKEND_DIR / backend.MODEL_PATHS[drone_model]
        self.sim = backend.CrazySimMuJoCo(
            model_path=str(model_path),
            host="0.0.0.0",
            base_port=FIRMWARE_PORT,
            spawn_positions=[(0.0, 0.0)],
            visualize=False,
            timestep=0.001,
            model_type=drone_model,
        )
        self.agent = self.sim.agents[0]
        self._sim_thread = threading.Thread(target=self._run_sim, daemon=True)
        self._sim_thread.start()
        time.sleep(0.5)

        cf2_dir = Path("/tmp/cf0")
        cf2_dir.mkdir(parents=True, exist_ok=True)
        with open(cf2_dir / "cf2_out.log", "w") as cf2_log:
            self._cf2 = subprocess.Popen(
                [str(CF2_BINARY), str(FIRMWARE_PORT)],
                cwd=cf2_dir,
                stdout=cf2_log,
                stderr=subprocess.STDOUT,
            )
        t_start = time.time()
        while self.agent._firmware_addr is None:
            time.sleep(0.1)
            if time.time() - t_start > HANDSHAKE_TIMEOUT:
                raise TimeoutError("No firmware handshake with the MuJoCo sim")
        logger.info("Firmware handshake done")
        time.sleep(1.0)

        super().__init__(drone_model, ctrl_interface, uri)

    def _run_sim(self):
        """Step the MuJoCo backend like `CrazySimMuJoCo.run` does, headless and wall-clock paced.

        Re-implemented here rather than calling ``run`` to pace the steps against absolute
        deadlines, see below.
        """
        sim, agent, mujoco = self.sim, self.agent, self._mujoco
        agent.handshake()
        sim._running = True
        agent.start_threads()
        # Schedule the steps against absolute deadlines so that step time and sleep overshoot do not
        # accumulate. CrazySim's own loop sleeps ``dt - elapsed`` after every step, which runs the
        # sim at a real-time factor of ~0.91 here. The firmware integrates the gyro over its
        # wall-clock ticks, so a real-time factor below one inflates its attitude estimate by
        # 1 / RTF (measured: 10 % attitude gain error at RTF 0.91). Sub-millisecond overshoots are
        # absorbed by the absolute schedule; a longer stall (e.g. another thread holding the GIL)
        # is dropped, so the sim stays a constant offset behind the wall clock, which the firmware
        # cannot see. Catching up with a faster sensor stream instead makes the firmware's
        # wall-clock EKF diverge.
        sys.setswitchinterval(GIL_SWITCH_INTERVAL)
        next_step = time.perf_counter()
        while sim._running:
            t_step = time.perf_counter()
            next_step += sim.dt
            agent.update_motors()
            agent.apply_aero_effects()
            agent.update_led_materials()
            mujoco.mj_step(sim.model, sim.data)
            agent.send_sensor_data()
            now = time.perf_counter()
            if self._original_pacing:
                next_step = t_step + sim.dt  # CrazySim's loop: the overshoot accumulates
            elif now - next_step > MAX_BACKLOG:
                next_step = now
            if now < next_step:
                time.sleep(next_step - now)
        agent.stop()

    def sim_time(self) -> float:
        """Simulation time of the MuJoCo backend."""
        return float(self.sim.data.time)

    def get_obs(self) -> dict:
        """Return the ground-truth observation dictionary, read directly from MjData."""
        data = self.sim.data
        body_id, qvel_adr = self.agent._body_id, self.agent._qvel_adr
        rot = R.from_quat(data.xquat[body_id], scalar_first=True)  # MuJoCo stores wxyz
        return {
            "pos": np.array(data.xpos[body_id]),
            "quat": rot.as_quat(),
            "vel": np.array(data.qvel[qvel_adr : qvel_adr + 3]),
            "ang_vel": np.array(data.qvel[qvel_adr + 3 : qvel_adr + 6]),
            "rpy": rot.as_euler("xyz"),
            "t": time.perf_counter(),
        }

    def close(self):
        """Stop the drone, the MuJoCo sim and the firmware process."""
        super().close()
        self.sim._running = False
        self._cf2.terminate()
