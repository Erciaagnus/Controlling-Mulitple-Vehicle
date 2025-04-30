from typing import List, Dict

from cpnav.hlcs import InterfaceHlc
from cpnav.utils import outer_lane_path, MeasurementTransformer
from cpnav.internal_types import PathPoint, VehicleStateList
from cpnav.models import CentralModel
import numpy as np
from cpnav.controllers import ModelPredictiveControl


def _get_reference_speed(t):
    if t < 15:
        return .5
    elif t < 25:
        return 1.4
    elif t < 35:
        return .8
    return 0


class CmpcHlc(InterfaceHlc):
    """
    High-Level Controller (HLC) for the Cooperative Model Predictive Control (CMP) system.
    This class implements the InterfaceHlc and provides functionality to handle vehicle state
    and control commands in a cooperative driving system.
    """

    def __init__(self, vehicle_ids: List[int], t_s: float):
        """
        Initializes the CMP HLC with vehicle IDs and sampling time.

        Args:
            vehicle_ids (List[int]): List of vehicle IDs for which the controller will be responsible.
            t_s (float): The sampling time, defining how frequently the controller updates.
        """
        super().__init__(vehicle_ids, t_s)
        
        print(vehicle_ids, flush=1)
        # Generate the path the vehicles will follow (e.g., outer lane path)
        self._path_points: List[PathPoint] = outer_lane_path()

        # Initialize the MeasurementTransformer to track vehicle measurements
        self._mt = MeasurementTransformer(self._path_points, vehicle_ids)

        self.n_vehicles = len(vehicle_ids)

        self.model = CentralModel(
            t_s, self.n_vehicles
        )

        self.t_s = t_s

        # Parameters for MPC prediction horizon and control horizon
        h_p: int = 20  # Prediction horizon
        self.h_p = h_p
        h_u: int = 10 # Control horizon

        # Spacing, speed and acceleration limits
        d_min: np.ndarray = np.array([0.3] * self.n_vehicles)
        self.d_ref: np.array = np.array([.5] * self.n_vehicles)
        self.d_min = d_min
        v_min: np.ndarray = np.array([0.0] * self.n_vehicles)
        v_max: np.ndarray = np.array([1.5] * self.n_vehicles)
        
        a_min: np.ndarray = np.array([-1.0] * self.n_vehicles)
        a_max: np.ndarray = np.array([0.5] * self.n_vehicles)

        self.v_min = v_min
        self.v_max = v_max

        # Control input bounds (change in velocity)
        dv_min: np.ndarray = a_min * t_s
        dv_max: np.ndarray = a_max * t_s

        q: np.ndarray = np.eye(self.model.model.C.shape[0])
        r: np.ndarray = np.eye(self.model.model.B.shape[1])

        # Kalman filter weights for state estimation
        q_kalman: np.ndarray = np.eye(self.model.model.A.shape[0])
        r_kalman: np.ndarray = np.eye(self.model.model.C.shape[0])

        # Measurement bounds for prediction
        y_min: np.ndarray = d_min
        y_min[-1] = v_min[-1]
        y_max: np.ndarray = np.array([1000] * (self.n_vehicles - 1) + [v_max[-1]])
        self._y_min = y_min
        self._y_max = y_max

        self._y_last_step = None
        # Set up the Model Predictive Controller (MPC)
        self._mpc = ModelPredictiveControl(self.model.model, h_p, h_u, v_min, v_max, dv_min,
                                           dv_max, y_min, y_max, q, r, q_kalman, r_kalman)
        

    def on_first_timestep(self, vehicle_state_list: VehicleStateList) -> None:
        """
        Method to handle the logic for the first timestep in the simulation.
        This method should be implemented by subclasses to set up initial conditions.

        Args:
            vehicle_state_list (VehicleStateList): The list of current vehicle states at the first timestep.
        """
        # TODO Implement CmpcHlc
        x = np.zeros([self.model.model.A.shape[0]]) # initial state
        y: np.ndarray = self._mt.measure_longitudinal(vehicle_state_list)

        self._y_last_step = np.zeros(y.shape[0])
        self._y_last_step[-2] = y[-2]

        x = np.array([[y[2*i], y[2*i+1], 0] for i in range(self.n_vehicles)]).flatten()

        # Set up the MPC with the initial state
        self._mpc.setup(x)

    def _get_y_ref(self, t):
        return np.array([self.d_ref[i] if i < self.n_vehicles - 1 else _get_reference_speed(t) for i in range(self.n_vehicles)])

    def _get_y_min(self, y):
        y_min = np.array([self.d_min[i] if i < self.n_vehicles - 1 else self.v_min[i] for i in range(self.n_vehicles)])
        return np.array(list(y_min) * self.h_p)
    
    def _get_y_max(self, y):
        return np.array([1000] * (self.n_vehicles - 1) + [self.v_max[-1]])
        
    def on_each_timestep(self, vehicle_state_list: VehicleStateList, t_expired_s: float) -> Dict[
        int, float]:
        """
        Method to handle the logic at each timestep during the simulation.

        Args:
            vehicle_state_list (VehicleStateList): The list of current vehicle states.
            t_expired_s (float): The amount of time that has passed since the start of the simulation in seconds.

        Returns:
            Dict[int, float]: A dictionary where the key is the vehicle ID and the value is the control command
                              (e.g., acceleration or steering) for each vehicle.
        """

        if t_expired_s > 40: print("ABBRUCH")

        y: np.ndarray = self._mt.measure_longitudinal(vehicle_state_list)
        
        u: np.ndarray

        t1: float = t_expired_s + self.t_s
        t2: float = t_expired_s + self._mpc.h_p * self.t_s
        timesteps = np.linspace(t1, t2 + self.t_s, self._mpc.h_p, endpoint=False)

        y_ref = np.array([self._get_y_ref(t) for t in timesteps]).flatten()

        y_min = np.array([self.d_min[i] if i < self.n_vehicles - 1 else self.v_min[i] for i in range(self.n_vehicles)])
        y_min = np.array(list(y_min) * self.h_p)

        y_max = np.array([1000] * (self.n_vehicles - 1) + [self.v_max[-1]])
        y_measured = np.array([y[(i+1)*2] - y[i*2] for i in range(self.n_vehicles - 1)] + [y[-1]])

        u, self._y_last_step = self._mpc.step(y_reference=y_ref, y_measured=y_measured, y_min=y_min, y_max=y_max)

        return { self.vehicle_ids[i]: float(u[i]) for i in range(self.n_vehicles) }
