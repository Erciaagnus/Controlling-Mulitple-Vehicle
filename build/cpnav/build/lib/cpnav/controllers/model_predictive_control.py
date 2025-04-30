from typing import Tuple

import numpy as np
from cvxpy import Variable
from scipy.linalg import block_diag
import cvxpy as cp
from control import StateSpace
from filterpy.kalman import KalmanFilter

from cpnav.controllers.interface_controller import ControllerInterface


class ModelPredictiveControl(ControllerInterface):
    """
    Model Predictive Controller (MPC) class for controlling a system based on a state-space model.
    The controller optimizes control inputs while respecting constraints on inputs, output, and changes in inputs.
    """

    def __init__(self, model: StateSpace, h_p: int, h_u: int, u_min: np.ndarray, u_max: np.ndarray,
                 du_min: np.ndarray, du_max: np.ndarray, y_min: np.ndarray, y_max: np.ndarray,
                 q: np.ndarray, r: np.ndarray, q_kalman: np.ndarray, r_kalman: np.ndarray):
        """
        Initializes the Model Predictive Controller (MPC) with the given system model and controller settings.

        Args:
            model (StateSpace): State-space model of the system.
            h_p (int): Prediction horizon length.
            h_u (int): Control horizon length.
            u_min, u_max (np.ndarray): Minimum and maximum control input bounds with shape (nu,).
            du_min, du_max (np.ndarray): Minimum and maximum control input change bounds with shape (nu,).
            y_min, y_max (np.ndarray): Minimum and maximum system output bounds with shape (ny,) or (ny * h_p,).
            q, r (np.ndarray): Weighting matrices for output (q) and control input (r) with shapes (ny, ny) and (nu, nu).
            q_kalman, r_kalman (np.ndarray): Kalman filter process and measurement noise covariance matrices,
                                             with shapes (nx, nx) and (ny, ny), respectively.
        """
        # Vehicle model
        self.model: StateSpace = model

        # Matrix sizes (number of states, inputs, outputs)
        self._nu: int = model.B.shape[1]  # Number of control inputs (scalar)
        self._nx: int = model.A.shape[0]  # Number of states (scalar)
        self._ny: int = model.C.shape[0]  # Number of outputs (scalar)

        # Horizon lengths
        self.h_p: int = h_p  # Prediction horizon (scalar)
        self._h_u: int = h_u  # Control horizon (scalar)

        # Previous control input (initialized to zeros, shape: (nu,))
        self._u_k_minus_one: np.ndarray = np.zeros(self._nu)

        # Control input limits (minimum and maximum values, shape: (nu,))
        self._u_min: np.ndarray = u_min
        assert self._u_min.shape == (
            self._nu,), f"u_min shape incorrect, should be {(self._nu,)}, is {u_min.shape}"

        self._u_max: np.ndarray = u_max
        assert self._u_max.shape == (
            self._nu,), f"u_max shape incorrect, should be {(self._nu,)}, is {u_max.shape}"

        # Control input change limits (minimum and maximum changes, shape: (nu,))
        self._du_min: np.ndarray = du_min
        assert self._du_min.shape == (
            self._nu,), f"du_min shape incorrect, should be {(self._nu,)}, is {du_min.shape}"

        self._du_max: np.ndarray = du_max
        assert self._du_max.shape == (
            self._nu,), f"du_max shape incorrect, should be {(self._nu,)}, is {du_max.shape}"

        # System output limits (minimum and maximum values, shape: (ny,) or (ny * h_p,))
        assert y_min.shape == (self._ny,) or y_min.shape == (self._ny * self.h_p,), \
            f"y_min shape incorrect, should be {(self._ny,)}, is {y_min.shape}"
        if y_min.shape == (self._ny,):
            # Extend limit to prediction horizon (shape: (ny * h_p,))
            y_min = np.tile(y_min, self.h_p)
        self._y_min = y_min

        assert y_max.shape == (self._ny,) or y_max.shape == (self._ny * self.h_p,), \
            f"y_max shape incorrect, should be {(self._ny,)}, is {y_max.shape}"
        if y_max.shape == (self._ny,):
            # Extend limit to prediction horizon (shape: (ny * h_p,))
            y_max = np.tile(y_max, self.h_p)
        self._y_max = y_max

        # Initialize Kalman filter with FilterPy for state estimation
        assert q_kalman.shape == (self._nx, self._nx), \
            f"q_kalman shape incorrect, should be {(self._nx, self._nx)}, is {q_kalman.shape}"
        assert r_kalman.shape == (self._ny, self._ny), \
            f"r_kalman shape incorrect, should be {(self._ny, self._ny)}, is {r_kalman.shape}"

        # Kalman Filter setup
        self._observer: KalmanFilter = KalmanFilter(dim_x=model.A.shape[0], dim_z=model.C.shape[0])
        self._observer.F = model.A  # State transition matrix (shape: (nx, nx))
        self._observer.H = model.C  # Measurement matrix (shape: (ny, nx))
        self._observer.Q = q_kalman  # Process noise covariance (shape: (nx, nx))
        self._observer.R = r_kalman  # Measurement noise covariance (shape: (ny, ny))
        self._observer.x = np.zeros((model.A.shape[0],))  # Initial state estimate (shape: (nx,))

        # MPC matrices for state and output predictions
        assert q.shape == (
            self._ny,
            self._ny), f"q shape incorrect, should be {(self._ny, self._ny)}, is {q.shape}"
        assert r.shape == (
            self._nu,
            self._nu), f"r shape incorrect, should be {(self._nu, self._nu)}, is {r.shape}"

        # Build the prediction matrices for the system model
        psi: np.ndarray = np.vstack([np.linalg.matrix_power(model.A, i) for i in
                                     range(1, self.h_p + 1)])  # Shape: (h_p, nx)

        gamma: np.ndarray = np.vstack([
            sum(np.linalg.matrix_power(model.A, j) @ model.B for j in range(i))
            for i in range(1, self.h_p + 1)
        ])  # Shape: (h_p, nu)

        theta: np.ndarray = np.vstack([
            np.hstack([
                sum(np.linalg.matrix_power(model.A, k) @ model.B for k in range(i - j + 1))
                if j <= i else np.zeros_like(model.B)
                for j in range(1, self._h_u + 1)
            ])
            for i in range(1, self.h_p + 1)
        ])  # Shape: (h_p, h_u * nu)

        # Extended output matrices for prediction horizon (shape: (h_p * ny, nx))
        self._c_hp: np.ndarray = block_diag(*[model.C for _ in range(self.h_p)])

        # Extended weighting matrices for prediction and control
        # (shapes: (h_p * ny, h_p * ny) and (h_u * nu, h_u * nu))
        self._q: np.ndarray = block_diag(*[q] * self.h_p)
        self._r: np.ndarray = block_diag(*[r] * self._h_u)

        # Prediction matrices for system outputs
        self._psi_y: np.ndarray = self._c_hp @ psi  # Shape: (h_p * ny, nx)
        self._gamma_y: np.ndarray = self._c_hp @ gamma  # Shape: (h_p * ny, nu)
        self._theta_y: np.ndarray = self._c_hp @ theta  # Shape: (h_p * ny, h_u * nu)

    def setup(self, x_init: np.ndarray) -> None:
        """
        Set up the initial state for the controller, including initializing the Kalman filter.

        Args:
            x_init (np.ndarray): Initial state estimate for the system (shape: (nx,)).
        """
        self._observer.x = x_init

    def step(self, y_measured: np.ndarray, y_reference: np.ndarray, y_min: np.ndarray | None = None,
             y_max: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform one step of Model Predictive Control (MPC), optimizing control inputs and applying the
        control to the system based on the current measurements and reference.

        Args:
            y_measured (np.ndarray): Measured system outputs (shape: (ny,)).
            y_reference (np.ndarray): Desired reference trajectory for the system outputs (shape: (ny,) or (ny * h_p,)).
            y_min (np.ndarray | None): Lower bound for system output constraints (shape: (ny,) or (ny * h_p,)).
            y_max (np.ndarray | None): Upper bound for system output constraints (shape: (ny,) or (ny * h_p,)).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Control input (u, shape: (nu,)) and predicted system output (y, shape: (ny * h_p,)).
        """

        # Assert correct dimensions of the measured system outputs
        assert y_measured.shape == (self._ny,), \
            f"y_measured shape incorrect, should be {(self._ny,)}, is {y_measured.shape}"

        # Assert correct dimensions of the reference trajectory (either single time-step or prediction horizon)
        assert y_reference.shape == (self._ny,) or y_reference.shape == (self._ny * self.h_p,), \
            (f"y_reference shape incorrect, should be {(self._ny,)} or {(self._ny * self.h_p,)}, "
             f"is {y_reference.shape}")

        # If y_reference is a single time step, extend it to the prediction horizon
        if y_reference.shape == (self._ny,):
            y_reference = np.tile(y_reference, self.h_p)  # Shape: (ny * h_p,)

        # Handle output bounds if provided (extend if they are single time-step)
        if y_min is not None:
            assert y_min.shape == (self._ny,) or y_min.shape == (self._ny * self.h_p,), \
                f"y_min shape incorrect, should be {(self._ny,)}, is {y_min.shape}"
            if y_min.shape == (self._ny,):
                y_min = np.tile(y_min, self.h_p)  # Shape: (ny * h_p,)
            self._y_min = y_min

        if y_max is not None:
            assert y_max.shape == (self._ny,) or y_max.shape == (self._ny * self.h_p,), \
                f"y_max shape incorrect, should be {(self._ny,)}, is {y_max.shape}"
            if y_max.shape == (self._ny,):
                y_max = np.tile(y_max, self.h_p)  # Shape: (ny * h_p,)
            self._y_max = y_max

        # Prediction and Correction Step in Kalman Filter
        self._observer.predict(
            self._u_k_minus_one)  # Prediction based on previous control input (shape: (nu,))
        self._observer.update(y_measured)  # Update based on the measured output (shape: (ny,))
        x_k: np.ndarray = self._observer.x  # State estimate (shape: (nx,))

        # Free dynamics output (shape: (ny * h_p,))
        y_free: np.ndarray = self._psi_y @ x_k + self._gamma_y @ self._u_k_minus_one  # Shape: (ny * h_p,)

        # Objective function terms
        q: np.ndarray = 2 * self._theta_y.T @ self._q @ (
                y_free - y_reference)  # Linear part (shape: (h_u * nu,))
        P: np.ndarray = 2 * (
                self._theta_y.T @ self._q @ self._theta_y + self._r)  # Quadratic part (shape: (h_u * nu, h_u * nu))

        # Control constraints
        G_u: np.ndarray = np.vstack([
            np.kron(np.tril(np.ones((self._h_u, self._h_u))), np.eye(self._nu)),
            -np.kron(np.tril(np.ones((self._h_u, self._h_u))), np.eye(self._nu))
        ])  # Shape: (2 * h_u * nu, h_u * nu)

        h_u: np.ndarray = np.hstack([
            np.tile(self._u_max, self._h_u) - np.tile(self._u_k_minus_one, self._h_u),
            -(np.tile(self._u_min, self._h_u) - np.tile(self._u_k_minus_one, self._h_u))
        ])  # Shape: (2 * h_u * nu,)

        # Variable bounds
        lb: np.ndarray = np.kron(np.ones(self._h_u), self._du_min)  # Shape: (h_u,)
        ub: np.ndarray = np.kron(np.ones(self._h_u), self._du_max)  # Shape: (h_u,)

        # Output constraints (for system outputs)
        G_y: np.ndarray = np.vstack(
            [self._theta_y, -self._theta_y])  # Shape: (2 * h_p * ny, h_u * nu)
        h_y: np.ndarray = np.hstack([self._y_max, -self._y_min]) - np.hstack(
            [y_free, -y_free])  # Shape: (2 * h_p * ny,)

        # -----------------------------------------------------------

        G_u_slack = np.hstack([G_u, np.zeros((G_u.shape[0], 1))])
        G_y_slack = np.hstack([G_y, -1 * np.ones((G_y.shape[0], 1))])

        G: np.ndarray = np.vstack([G_u_slack, G_y_slack]) 

        # Define optimization problem
        n = P.shape[0]  # Number of optimization variables (excluding slack)
        x = cp.Variable(n + 1)  # Optimization variable (control inputs, shape: (h_u * nu,))
        
        h: np.ndarray = np.hstack([h_u, h_y]) 

        P_slack = np.zeros((n + 1, n + 1))
        P_slack[:n, :n] = P  # Original quadratic cost for control inputs
        P_slack[-1, -1] = 1e6  # Large penalty for slack variable

        q_slack = np.zeros(n + 1)
        q_slack[:n] = q  # Original linear cost for control inputs

        # Define and solve the optimization problem
        prob = cp.Problem(
            cp.Minimize((1 / 2) * cp.quad_form(x, P_slack) + q_slack.T @ x), 
            [G @ x <= h, x[:-1] <= ub, lb <= x[:-1], 0 <= x[-1]]
        )

        # -----------------------------------------------------------

        # Initializing delta_u_solution for storing the solution
        delta_u_solution: np.ndarray = np.zeros((self._nu * self._h_u,))  # Shape: (h_u * nu,)

        try:
            prob.solve(verbose=False)  # Solve the optimization problem
            if prob.status != 'optimal':
                print("Warning: No feasible solution found, setting u=0.")
                delta_u_solution[:self._nu + 1] = -self._u_k_minus_one
            else:
                xi: float = float(delta_u_solution[-1])  # Slack variable value
                delta_u_solution = x.value[:-1]  # Get control input deltas #TODO CHANGED x.value[:-1] -> x.value
                
                if xi > 1e-15:
                    print(f'Warning: Needed slack variable xi = {xi}')
        except ValueError as e:
            print(f'Warning: {e}.')

        # Compute the control input change (delta_u)
        d_u_k: np.ndarray = delta_u_solution[:self._nu]  # Shape: (nu,)

        # Compute the control input (u) and predicted output (y)
        u: np.ndarray = self._u_k_minus_one + d_u_k  # Shape: (nu,)
        
        y: np.ndarray = self._psi_y @ x_k + self._gamma_y @ self._u_k_minus_one + self._theta_y @ delta_u_solution  # Shape: (ny * h_p,)

        # Save last control input
        self._u_k_minus_one = u

        return u, y  # Return control input (u, shape: (nu,)) and predicted output (y, shape: (ny * h_p,))
