"""Model predictive contouring control for the race track.

The controller follows a cubic spline through the gates. We solve with acados over the abstracted
drone dynamics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import casadi as cs
import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from crazyflow.dynamics.core import Dynamics
from crazyflow.dynamics.core import load_params as load_dynamics_params
from crazyflow.dynamics.so_rpy_rotor_drag import symbolic_dynamics_euler
from crazyflow.dynamics.utils.rotation import ang_vel2rpy_rates
from lsy_drone_racing.control import Controller
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from crazyflow_experiments.utils.paths import REPO_ROOT

if TYPE_CHECKING:
    from numpy.typing import NDArray


def create_acados_model(parameters: dict) -> AcadosModel:
    """Create an acados model from the symbolic drone dynamics."""
    X_dot, X, U, _ = symbolic_dynamics_euler(
        mass=parameters["mass"],
        gravity_vec=parameters["gravity_vec"],
        J=parameters["J"],
        J_inv=parameters["J_inv"],
        thrust_time_coef=parameters["thrust_time_coef"],
        acc_coef=parameters["acc_coef"],
        cmd_f_coef=parameters["cmd_f_coef"],
        rpy_coef=parameters["rpy_coef"],
        rpy_rates_coef=parameters["rpy_rates_coef"],
        cmd_rpy_coef=parameters["cmd_rpy_coef"],
        drag_matrix=parameters["drag_matrix"],
    )

    # Add progress variable *theta* for MPCC
    theta = cs.MX.sym("theta")
    X = cs.vertcat(X, theta)
    v_theta = cs.MX.sym("v_theta")
    U = cs.vertcat(U, v_theta)
    X_dot = cs.vertcat(X_dot, v_theta)

    # Initialize the nonlinear model for NMPC formulation
    model = AcadosModel()
    model.name = "racing_mpcc"
    model.f_expl_expr = X_dot
    model.f_impl_expr = None
    model.x = X
    model.u = U

    return model


def create_ocp_solver(
    Tf: float, N: int, parameters: dict, waypoints: dict[str, NDArray], verbose: bool = False
) -> tuple[AcadosOcpSolver, AcadosOcp]:
    """Create an acados optimal control problem and solver."""
    ocp = AcadosOcp()
    build_dir = REPO_ROOT / "c_generated_code"
    ocp.code_export_directory = str(build_dir)

    # Set model
    ocp.model = create_acados_model(parameters)

    # Get Dimensions
    nx = ocp.model.x.rows()

    # Extract model
    x = ocp.model.x
    u = ocp.model.u
    # State is pos, rpy, vel, rpy_rates, thrust over four entries, theta. Input is rpy, thrust,
    # v_theta.
    pos = x[0:3]
    rpy_rates = x[9:12]
    theta = x[16]
    v_theta = u[4]

    # Set dimensions
    ocp.solver_options.N_horizon = N

    # Reference interpolant from the waypoints
    lut_px = cs.interpolant("lut_px", "linear", [waypoints["time"]], waypoints["pos"][:, 0])
    lut_py = cs.interpolant("lut_py", "linear", [waypoints["time"]], waypoints["pos"][:, 1])
    lut_pz = cs.interpolant("lut_pz", "linear", [waypoints["time"]], waypoints["pos"][:, 2])
    pos_ref = cs.vertcat(lut_px(theta), lut_py(theta), lut_pz(theta))
    vel_ref = cs.vertcat(
        cs.gradient(pos_ref[0], theta),
        cs.gradient(pos_ref[1], theta),
        cs.gradient(pos_ref[2], theta),
    )
    u_ref = cs.vertcat(
        cs.MX(0),
        cs.MX(0),
        cs.MX(0),
        cs.MX(-parameters["gravity_vec"][-1] * parameters["mass"]),
        cs.MX(1.0),
    )

    ## Set Cost
    # For more Information regarding Cost Function Definition in Acados:
    # https://github.com/acados/acados/blob/main/docs/problem_formulation/problem_formulation_ocp_mex.pdf

    # Cost Type
    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    # Weights
    Q_c = np.eye(3) * 100  # contour
    Q_l = np.eye(3) * 1_000  # lag
    Q_omega = np.eye(3) * 0.5  # rpy rates velocity
    mu = 4.0  # progress, 1 is slow, 3 is normal, 4 is fast
    R_u = np.diag(
        [
            1.0,  # rpy
            1.0,  # rpy
            10.0,  # rpy
            1.0,  # thrust
            1.0,  # v_theta
        ]
    )

    # Errors
    e = cs.vertcat(pos - pos_ref)
    normal = vel_ref
    normal = normal / cs.norm_2(normal)

    # Contouring & lag error estimation
    e_l_hat = cs.dot(normal, e) * normal
    e_c_hat = e - e_l_hat

    # Costs
    stage_cost = (
        e_c_hat.T @ Q_c @ e_c_hat
        + e_l_hat.T @ Q_l @ e_l_hat
        - mu * v_theta
        + rpy_rates.T @ Q_omega @ rpy_rates
        + (u - u_ref).T @ R_u @ (u - u_ref)  # the actual inputs to the system
    )
    # No inputs in the terminal cost
    terminal_cost = e_c_hat.T @ Q_c @ e_c_hat + e_l_hat.T @ Q_l @ e_l_hat

    ocp.model.cost_expr_ext_cost = stage_cost
    ocp.model.cost_expr_ext_cost_e = terminal_cost

    # Set Input Constraints (rpy < 1 rad, thrust in limits)
    ocp.constraints.lbu = np.array([-1.0, -1.0, -1.0, parameters["thrust_min"] * 4, 0.25])
    ocp.constraints.ubu = np.array([1.0, 1.0, 1.0, parameters["thrust_max"] * 4, 4.0])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3, 4])

    # We have to set x0 even though we will overwrite it later on.
    ocp.constraints.x0 = np.zeros((nx))

    # Solver Options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"  # FULL_, PARTIAL_ ,_HPIPM, _QPOASES
    ocp.solver_options.hpipm_mode = "BALANCE"  # BALANCE, SPEED, ROBUST, SPEED_ABS
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"  # SQP, SQP_RTI
    ocp.solver_options.tol = 1e-6

    ocp.solver_options.qp_solver_cond_N = N
    ocp.solver_options.warm_start = True
    ocp.solver_options.qp_solver_warm_start = True
    ocp.solver_options.nlp_solver_warm_start_first_qp = True

    ocp.solver_options.qp_solver_iter_max = 20
    ocp.solver_options.nlp_solver_max_iter = 50

    # set prediction horizon
    ocp.solver_options.tf = Tf

    acados_ocp_solver = AcadosOcpSolver(
        ocp,
        json_file=str(build_dir / "racing_mpcc.json"),
        verbose=verbose,
        build=True,
        generate=True,
    )

    return acados_ocp_solver, ocp


class AttitudeMPCC(Controller):
    """Model predictive contouring controller for the attitude interface."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the attitude controller.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: Additional environment information from the reset.
            config: The configuration of the environment.
        """
        super().__init__(obs, info, config)
        self._N = 25  # steps
        self._T = 1.0  # s
        self._env_freq = config.env.freq

        # Same waypoints as in the trajectory controller. Determined by trial and error.
        waypoints = np.array(
            [
                [0.15, 0.3, 0.05],
                [-0.5, 0.2, 0.6],  # Entry gate 1
                [-1.4, 1.5, 1.2],  # Entry gate 2
                [0.1, 1.5, 0.8],  # Entry gate 3
                [-0.1, 0.1, 1.0],  # Connection gate 3-4
                [1.5, -1.0, 1.5],  # Behind gate 4
                [2.5, -1.5, 1.6],  # Runout
                [2.5, -1.5, 1.6],  # Runout
            ]
        )
        self._t_total = 3  # s
        t = np.linspace(0, self._t_total, len(waypoints))
        self._des_pos_spline = CubicSpline(
            t, waypoints, bc_type=((1, [-0.5, -0.2, 0.5]), (2, [0.0, 0.0, 0.0]))
        )
        self._des_vel_spline = self._des_pos_spline.derivative()
        self.waypoints = {
            "time": np.linspace(0, self._t_total, self._env_freq * self._t_total),
            "pos": self._des_pos_spline(
                np.linspace(0, self._t_total, self._env_freq * self._t_total)
            ),
            "vel": self._des_vel_spline(
                np.linspace(0, self._t_total, self._env_freq * self._t_total)
            ),
        }

        drone = config.sim.drone
        self.drone_params = load_dynamics_params(Dynamics.so_rpy_rotor_drag, drone)
        self._acados_ocp_solver, self._ocp = create_ocp_solver(
            self._T, self._N, self.drone_params, self.waypoints
        )

        self._theta = 0
        self._theta_max = self.waypoints["time"][-1] - self._T * 1.1
        self._thrust = 0

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired collective thrust and roll/pitch/yaw of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The orientation as roll, pitch, yaw angles, and the collective thrust [r_des, p_des, y_des, t_des] as a numpy array.
        """
        # Setting initial state
        obs["rpy"] = R.from_quat(obs["quat"]).as_euler("xyz")
        obs["drpy"] = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])
        # the env does not provide proper thrust observations, so we build them ourselves

        x0 = np.concatenate(
            (
                obs["pos"],
                obs["rpy"],
                obs["vel"],
                obs["drpy"],
                [self._thrust, 0, 0, 0],
                [self._theta],
            )
        )
        self._acados_ocp_solver.set(0, "lbx", x0)
        self._acados_ocp_solver.set(0, "ubx", x0)

        # Solving problem and getting first input
        self._acados_ocp_solver.solve()
        u0 = self._acados_ocp_solver.get(0, "u")

        # Update thrust state
        thrust_dot = 1 / self.drone_params["thrust_time_coef"] * (u0[-2] - self._thrust)
        self._thrust += thrust_dot / self._env_freq
        self._theta += u0[-1] / self._env_freq

        return u0[:-1]

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Return if the controller has finished."""
        return self._theta >= self._theta_max
