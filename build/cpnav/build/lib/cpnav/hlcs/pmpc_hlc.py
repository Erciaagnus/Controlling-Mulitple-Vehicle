from typing import List, Dict, Callable

import numpy as np

from cpnav.controllers import ModelPredictiveControl
from cpnav.hlcs import InterfaceHlc
from cpnav.models import LongitudinalModel
from cpnav.utils import outer_lane_path, MeasurementTransformer
from cpnav.internal_types import PathPoint, VehicleStateList


class PmpcHlc(InterfaceHlc):
    """
    High-Level Controller (HLC) for Position Model Predictive Control (PMPC) in a cooperative driving system.
    This class implements the InterfaceHlc and provides functionality for controlling a single vehicle using
    Model Predictive Control (MPC).
    """

    def __init__(self, vehicle_ids: List[int], t_s: float):
        """
        Initializes the PMPC HLC with vehicle IDs and sampling time.
        This controller is designed to use MPC for controlling a single vehicle.

        Args:
            vehicle_ids (List[int]): List of vehicle IDs for which the controller will be responsible.
            t_s (float): The sampling time, defining how frequently the controller updates.
        """
        super().__init__(vehicle_ids, t_s)

        # Ensure only one vehicle is used for PMPC
        assert len(vehicle_ids) == 1, "Only one vehicle is allowed for PMPC."

        # Parameters for the controller
        self._path_points: List[PathPoint] = outer_lane_path()
        self._mt = MeasurementTransformer(self._path_points, vehicle_ids)

        # Setup the Longitudinal Model for the vehicle
        vehicle_model: LongitudinalModel = LongitudinalModel(self.t_s)

        # TODO Setup the MPC

        # Parameters for MPC prediction horizon and control horizon
        h_p: int = 20  # TODO Prediction horizon
        self.h_p = h_p
        h_u: int = 10  # TODO Control horizon

        # Speed and acceleration limits
        v_min: np.array = np.array([0.0])  # TODO
        self.v_min = v_min
        v_max: np.array = np.array([1.5])  # TODO
        self.v_max = v_max
        a_min: np.array = np.array([-1.0])  # TODO
        a_max: np.array = np.array([.5])  # TODO

        # Control input bounds (change in velocity)
        dv_min: np.array = a_min * t_s  # TODO
        dv_max: np.array = a_max * t_s  # TODO

        # Cost function weights
        q: np.array = np.eye(2)  # TODO
        r: np.array = np.eye(1)  # TODO

        self.validate_positive_semidefinite(q, "q")
        self.validate_positive_semidefinite(r, "r")

        # Kalman filter weights for state estimation 
        # Set based on noise of the system, increase q for larger process noise, increase r for larger measurement noise
        q_kalman: np.ndarray = np.eye(3)  # TODO
        r_kalman: np.ndarray = np.eye(2)  # TODO

        # Measurement bounds for prediction

        y_min: np.ndarray = np.array([np.sum(v_min * t_s * h_p), np.sum(v_min)]) # has to be a 1x2 Matrix, [0] for dist, [1] for vel
        self.y_min = y_min
        y_max: np.ndarray = np.array([np.sum(v_max * t_s * h_p), np.sum(v_max)]) # has to be a 1x2 Matrix, [0] for dist, [1] for vel
        self.y_max = y_max

        # Set up the Model Predictive Controller (MPC)
        self._mpc = ModelPredictiveControl(vehicle_model.model, h_p, h_u, v_min, v_max, dv_min,
                                           dv_max, y_min, y_max, q, r, q_kalman, r_kalman)

        # Initial position
        self._y_zero: float | None = None

        # Reference generation function (target trajectory)
        self._s_ref: Callable[[np.ndarray], np.ndarray] = lambda t: 1.1 * t + .5 * np.sin(t) + self._y_zero
        self._ds_ref: Callable[[np.ndarray], np.ndarray] = lambda t: .5 * np.cos(t) + 1.1

        self._y_ref: Callable[[np.ndarray], np.ndarray] = lambda t: np.array([np.array([self._s_ref(t_), self._ds_ref(t_)]) for t_ in t]).flatten()


    def validate_positive_semidefinite(self, matrix: np.ndarray, name: str):
        """
        Validates that a matrix is positive semidefinite and logs a warning if it is not.
        """
        eigenvalues = np.linalg.eigvals(matrix)
        if np.any(eigenvalues < 0):
            print(f"Warning: {name} is not positive semidefinite. Eigenvalues: {eigenvalues}", flush=True)
        else:
            print(f"{name} is positive semidefinite.", flush=True)

    def on_first_timestep(self, vehicle_state_list: VehicleStateList) -> None:
        """
        Method to handle the logic for the first timestep in the simulation.
        This method sets up the initial conditions for the controller and the vehicle model.

        Args:
            vehicle_state_list (VehicleStateList): The list of current vehicle states at the first timestep.
        """
        # Allocate initial state vector
        x: np.ndarray = np.zeros(self._mpc.model.A.shape[0])

        # Get distance and speed of the vehicle on the path
        y: np.ndarray = self._mt.measure_longitudinal(vehicle_state_list)
        assert len(y) == 2, "Received measurements for more than one vehicle."

        # Fill in distance and speed into the state vector
        x[:2] = y

        # Save initial position for reference trajectory generation
        self._y_zero = x[0]

        # Set up the MPC with the initial state
        self._mpc.setup(x)

    def on_each_timestep(self, vehicle_state_list: VehicleStateList, t_expired_s: float) -> Dict[
        int, float]:
        """
        Method to handle the logic at each timestep during the simulation, specifically using Model Predictive Control.

        Args:
            vehicle_state_list (VehicleStateList): The list of current vehicle states.
            t_expired_s (float): The amount of time that has passed since the start of the simulation in seconds.

        Returns:
            Dict[int, float]: A dictionary where the key is the vehicle ID and the value is the control action (acceleration or steering) for the vehicle at this timestep.
        """
        # Get distance and speed of the vehicle on the path
        y: np.ndarray = self._mt.measure_longitudinal(vehicle_state_list)
        
        # Compute the control action using MPC
        t1: float = t_expired_s + self.t_s
        t2: float = t_expired_s + self._mpc.h_p * self.t_s
        u: np.ndarray

        self.y_min = np.array([y[0], self.v_min[0]])
        self.y_max = np.array([y[0] + np.sum(self.v_max * self.t_s * self.h_p), self.v_max[0]])

        u, _ = self._mpc.step(
            y_measured = y, 
            y_reference = self._y_ref(np.linspace(t1, t2 + self.t_s, self._mpc.h_p, endpoint=False)), 
            y_min = self.y_min, 
            y_max = self.y_max)

        assert len(u) == 1, "Received control for more than one vehicle."


        # Return the control action for the vehicle
        return {self.vehicle_ids[0]: float(u[0])}
