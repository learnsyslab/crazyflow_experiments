"""Model predictive trajectory tracking through the collective thrust and attitude interface.

The acados solver optimizes the attitude and thrust commands over a receding horizon of the
reference trajectory, using the crazyflow drone model with rotor and drag dynamics.
"""

from __future__ import annotations  # Python 3.10 type hints

from typing import TYPE_CHECKING

import numpy as np
import scipy
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from crazyflow.dynamics.core import Dynamics
from crazyflow.dynamics.core import load_params as load_dynamics_params
from crazyflow.dynamics.so_rpy_rotor_drag import symbolic_dynamics_euler
from crazyflow.dynamics.utils.rotation import ang_vel2rpy_rates
from scipy.spatial.transform import Rotation as R

from crazyflow_experiments.control.controller import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from crazyflow_experiments.control.trajectories import Trajectory


def create_acados_model(parameters: dict) -> AcadosModel:
    """Creates an acados model from the symbolic drone dynamics."""
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

    # Initialize the nonlinear model for NMPC formulation
    model = AcadosModel()
    model.name = "basic_example_mpc"
    model.f_expl_expr = X_dot
    model.f_impl_expr = None
    model.x = X
    model.u = U

    return model


def create_ocp_solver(
    Tf: float, N: int, parameters: dict, verbose: bool = False
) -> tuple[AcadosOcpSolver, AcadosOcp]:
    """Creates an acados Optimal Control Problem and Solver."""
    ocp = AcadosOcp()

    # Set model
    ocp.model = create_acados_model(parameters)

    # Get Dimensions
    nx = ocp.model.x.rows()
    nu = ocp.model.u.rows()
    ny = nx + nu
    ny_e = nx

    # Set dimensions
    ocp.solver_options.N_horizon = N

    ## Set Cost
    # For more Information regarding Cost Function Definition in Acados:
    # https://github.com/acados/acados/blob/main/docs/problem_formulation/problem_formulation_ocp_mex.pdf
    #

    # Cost Type
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    # Weights
    # State weights
    Q = np.diag(
        [
            80.0,  # pos
            80.0,  # pos
            80.0,  # pos
            0.0,  # rpy
            0.0,  # rpy
            0.0,  # rpy
            0.0,  # vel
            0.0,  # vel
            0.0,  # vel
            0.0,  # drpy
            0.0,  # drpy
            0.0,  # drpy
            0.0,  # thrust
            0.0,  # thrust
            0.0,  # thrust
            0.0,  # thrust
        ]
    )
    # Input weights (reference is upright orientation and hover thrust)
    R = np.diag(
        [
            1.0,  # rpy
            1.0,  # rpy
            1.0,  # rpy
            1.0,  # thrust
        ]
    )

    Q_e = Q.copy()
    ocp.cost.W = scipy.linalg.block_diag(Q, R)
    ocp.cost.W_e = Q_e

    Vx = np.zeros((ny, nx))
    Vx[0:nx, 0:nx] = np.eye(nx)  # Select all states
    ocp.cost.Vx = Vx

    Vu = np.zeros((ny, nu))
    Vu[nx : nx + nu, :] = np.eye(nu)  # Select all actions
    ocp.cost.Vu = Vu

    Vx_e = np.zeros((ny_e, nx))
    Vx_e[0:nx, 0:nx] = np.eye(nx)  # Select all states
    ocp.cost.Vx_e = Vx_e

    # Set initial references (we will overwrite these later on to make the controller track the traj.)
    ocp.cost.yref, ocp.cost.yref_e = np.zeros((ny,)), np.zeros((ny_e,))

    # Set State Constraints (rpy < 30 deg)
    # ocp.constraints.lbx = np.array([-0.5, -0.5, -0.5])
    # ocp.constraints.ubx = np.array([0.5, 0.5, 0.5])
    # ocp.constraints.idxbx = np.array([3, 4, 5])

    # Set Input Constraints (rpy < 30 deg, thrust in limits)
    ocp.constraints.lbu = np.array([-0.5, -0.5, -0.5, parameters["thrust_min"] * 4])
    ocp.constraints.ubu = np.array([0.5, 0.5, 0.5, parameters["thrust_max"] * 4])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])

    # We have to set x0 even though we will overwrite it later on.
    ocp.constraints.x0 = np.zeros((nx))

    # Solver Options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"  # FULL_, PARTIAL_ ,_HPIPM, _QPOASES
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"  # SQP, SQP_RTI
    ocp.solver_options.tol = 1e-6

    ocp.solver_options.qp_solver_cond_N = N
    ocp.solver_options.qp_solver_warm_start = 1

    ocp.solver_options.qp_solver_iter_max = 20
    ocp.solver_options.nlp_solver_max_iter = 50

    # set prediction horizon
    ocp.solver_options.tf = Tf

    acados_ocp_solver = AcadosOcpSolver(
        ocp,
        json_file="c_generated_code/lsy_example_mpc2.json",
        verbose=verbose,
        build=True,
        generate=True,
    )

    return acados_ocp_solver, ocp


class ModelPredictiveController(Controller):
    """MPC commanding collective thrust and attitude."""

    def __init__(
        self,
        initial_obs: dict[str, NDArray[np.floating]],
        trajectory: Trajectory,
        drone_model: str,
        ctrl_freq: float,
    ):
        """Initialize the controller and build the acados solver.

        Args:
            initial_obs: The initial observation, its time `t` starts the trajectory.
            trajectory: The trajectory the drone follows.
            drone_model: Drone model.
            ctrl_freq: Control frequency.
        """
        self._N = 25  # steps
        self._T = 2.5  # s
        self._dt = self._T / self._N
        self.dt_array = np.arange(0, self._N + 1) * self._dt
        # difference in frequency between prediction and environment
        self._ctrl_freq = ctrl_freq
        self._delta_f = self._ctrl_freq * self._dt
        assert np.isclose(self._delta_f % 1, 0), (
            f"Controller and prediction frequency don't have a common multiple, was {self._ctrl_freq}*{self._dt}={self._delta_f}"
        )
        self._delta_f = int(self._delta_f)

        self.trajectory = trajectory

        drone = drone_model
        self.drone_params = load_dynamics_params(Dynamics.so_rpy_rotor_drag, drone)
        self._acados_ocp_solver, self._ocp = create_ocp_solver(self._T, self._N, self.drone_params)
        self._nx = self._ocp.model.x.rows()
        self._nu = self._ocp.model.u.rows()
        self._ny = self._nx + self._nu
        self._ny_e = self._nx

        self._t_start = initial_obs["t"]
        self._t_end = trajectory.duration
        self._last_thrust = 0.0

    def __call__(self, obs: dict[str, NDArray[np.floating]]) -> NDArray[np.floating]:
        """Compute the next desired collective thrust and roll/pitch/yaw of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.

        Returns:
            The orientation as roll, pitch, yaw angles, and the collective thrust [r_des, p_des, y_des, t_des] as a numpy array.
        """
        t = obs["t"] - self._t_start

        # Setting initial state
        obs["rpy"] = R.from_quat(obs["quat"]).as_euler("xyz")
        obs["drpy"] = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])
        # the env does not provide proper thrust observations, so we build them ourselves
        x0 = np.concatenate(
            (obs["pos"], obs["rpy"], obs["vel"], obs["drpy"], [self._last_thrust, 0, 0, 0])
        )
        self._acados_ocp_solver.set(0, "lbx", x0)
        self._acados_ocp_solver.set(0, "ubx", x0)

        # Generating state reference
        horizon = self.dt_array + t
        des_pos = np.asarray(self.trajectory.pos(horizon))
        des_vel = np.asarray(self.trajectory.vel(horizon))
        zeros = np.zeros_like(des_pos)

        # Setting state reference
        yref = np.zeros((self._N, self._ny))
        yref[:, 0:3] = des_pos[:-1]
        yref[:, 3:6] = zeros[:-1]
        yref[:, 6:9] = des_vel[:-1]
        yref[:, 9:12] = zeros[:-1]
        # yref[:, 12:16] = self._waypoints_thrust[i : i + self._N * self._delta_f : self._delta_f]
        yref[:, 16:19] = zeros[:-1]
        yref[:, 19] = -self.drone_params["mass"] * self.drone_params["gravity_vec"][2]
        for j in range(self._N):
            self._acados_ocp_solver.set(j, "yref", yref[j])

        # Setting final state reference
        yref_e = np.zeros((self._ny_e))
        yref_e[0:3] = des_pos[-1]
        yref_e[3:6] = zeros[-1]
        yref_e[6:9] = des_vel[-1]
        yref_e[9:12] = zeros[-1]
        # yref_e[12:16] = self._waypoints_thrust[i + self._N * self._delta_f]
        self._acados_ocp_solver.set(self._N, "y_ref", yref_e)

        # Solving problem and getting first input
        self._acados_ocp_solver.solve()
        u0 = self._acados_ocp_solver.get(0, "u")
        x = []
        for j in range(self._N):
            x.append(self._acados_ocp_solver.get(j, "x"))
        self.x_pred = np.array(x)

        # Update thrust state
        thrust_dot = 1 / self.drone_params["thrust_time_coef"] * (u0[-1] - self._last_thrust)
        self._last_thrust += thrust_dot / self._ctrl_freq

        return u0

    def done(self, obs: dict[str, NDArray[np.floating]]) -> bool:
        """Return True once the planned trajectory is done."""
        return obs["t"] - self._t_start >= self._t_end
