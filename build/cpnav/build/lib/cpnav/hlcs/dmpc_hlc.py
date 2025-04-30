from typing import List, Dict, Tuple
import numpy as np


from cpnav.hlcs import InterfaceHlc
from cpnav.utils import outer_lane_path, MeasurementTransformer
from cpnav.internal_types import PathPoint, VehicleStateList
from cpnav.models import DistributedModel
from cpnav.controllers import ModelPredictiveControl


class DmpcHlc(InterfaceHlc):
    """
    High-Level Controller (HLC) for the Distributed Model Predictive Control (DMPC) system.
    This class implements the InterfaceHlc and provides functionality to handle vehicle state
    and control commands in a distributed driving control system.
    """

    def __init__(self, vehicle_ids: List[int], t_s: float):
        """
        Initializes the DMPC HLC with vehicle IDs and sampling time.

        Args:
            vehicle_ids (List[int]): List of vehicle IDs for which the controller will be responsible.
            t_s (float): The sampling time, defining how frequently the controller updates.
        """
        print("IDs:", self.sort_vehicle_ids_by_priority(vehicle_ids))
        self.vehicle_ids = self.sort_vehicle_ids_by_priority(vehicle_ids)
        self.vehicle_hlc_list = [DmpcVehicleHlc([vehicle_id], vehicle_id == self.vehicle_ids[0], t_s) for vehicle_id in self.vehicle_ids]

    def sort_vehicle_ids_by_priority(self, vehicle_ids: List[int]):
        return -np.sort(-np.array(vehicle_ids))

    def on_first_timestep(self, vehicle_state_list: VehicleStateList) -> None:
        """
        Method to handle the logic for the first timestep in the simulation.
        This method should be implemented by subclasses to set up initial conditions.

        Args:
            vehicle_state_list (VehicleStateList): The list of current vehicle states at the first timestep.
        """
        for vehicle in self.vehicle_hlc_list:
            vehicle_state = VehicleStateList({vehicle.vehicle_id : vehicle_state_list.vehicle_states.get(vehicle.vehicle_id)}, vehicle_state_list.t_now, vehicle_state_list.period_ms)
            
            vehicle.on_first_timestep(vehicle_state)

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

        u_dict = {}
        predecessors = []
        print([v.vehicle_id for v in self.vehicle_hlc_list])
        #print([v.vehicle_states.vehicle_id for v in vehicle_state_list.vehicle_states.values()])
        for vehicle in self.vehicle_hlc_list:
            print(f"Vehicle: {vehicle.vehicle_id} - Control Type {vehicle.control_type}", flush=True)
            u, y = vehicle.on_each_timestep(vehicle_state_list, predecessors, t_expired_s)
            u_dict[vehicle.vehicle_id] = u
            predecessors = [y]

        return u_dict



class DmpcVehicleHlc:
    """
    High-Level Controller (HLC) for the Distributed Model Predictive Control (DMPC) system.
    This class implements the InterfaceHlc and provides functionality to handle vehicle state
    and control commands in a distributed driving control system.
    """

    def __init__(self, vehicle_ids: List[int], head: bool, t_s: float):
        """
        Initializes the DMPC HLC with vehicle IDs and sampling time.
        This controller is designed to use MPC for controlling a single vehicle.

        Args:
            vehicle_ids (List[int]): List of vehicle IDs for which the controller will be responsible.
            t_s (float): The sampling time, defining how frequently the controller updates.
        """

        # Ensure only one vehicle is used for DMPC
        assert len(vehicle_ids) == 1, "Only one vehicle is allowed for DMPC."

        self.vehicle_id = vehicle_ids[0]

        # Parameters for the controller
        self._path_points: List[PathPoint] = outer_lane_path()
        self._mt = MeasurementTransformer(self._path_points, vehicle_ids)
        self.t_s = t_s

        # Setup the Longitudinal Model for the vehicle
        print("head" if head else "tail")
        vehicle_model: DistributedModel = DistributedModel("head" if head else "tail", self.t_s)
        self.control_type = "head" if head else "tail"

        # TODO Setup the MPC

        # Parameters for MPC prediction horizon and control horizon
        h_p: int = 20 # TODO Prediction horizon
        self.h_p = h_p
        h_u: int = 10 # TODO Control horizon

        # Speed and acceleration limits
        v_min: np.array = np.array([0.0])  # TODO
        v_max: np.array = np.array([1.5])  # TODO
        
        self.v_min = v_min
        self.v_max = v_max
        self.a_min: np.array = np.array([-1.0])  # TODO
        self.a_max: np.array = np.array([.5])  # TODO

        self.d_ref = .5
        self.d_min = .3

        # Control input bounds (change in velocity)
        dv_min: np.array = self.a_min * t_s  # TODO
        dv_max: np.array = self.a_max * t_s  # TODO

        # Cost function weights
        q: np.array = np.eye(2)  # TODO
        q[0, 0] = 0 if head else 1 # no penalty for distance error for head vehicle
        q[1, 1] = 1 # penalty for velocity error
        r: np.array = np.eye(1)  # TODO

        # Kalman filter weights for state estimation
        # Set based on noise of the system, increase q for larger process noise, increase r for larger measurement noise
        q_kalman: np.ndarray = np.eye(3)  # TODO
        r_kalman: np.ndarray = np.eye(2)  # TODO

        # Measurement bounds for prediction

        self.y_min = np.array([v_min[0] * t_s * h_p, v_min[0]])
        self.y_max = np.array([v_max[0] * t_s * h_p, v_max[0]])

        # Set up the Model Predictive Controller (MPC)
        self._mpc = ModelPredictiveControl(vehicle_model.model, h_p, h_u, v_min, v_max, dv_min,
                                           dv_max, self.y_min, self.y_max, q, r, q_kalman, r_kalman)

        # Initial position
        self._y_zero: float | None = None

    def _y_ref(self, h_p_predecessor: np.array, y_current, t) -> np.ndarray:
        """
        Compute y_ref based on y_predecessors.

        Args:
            h_p_predecessors: Control Horizons of predecessing Vehicles
            t: current time
        """

        y_ref = np.empty(2 * self.h_p)

        v_ref = []
        for t_ in range(self.h_p):
            t_step = t + t_ * self.t_s
            if t_step < 15.:
                v_ref += [0.5]
            elif 15. <= t_step < 25.:
                v_ref += [1.4]
            elif 25. <= t_step < 35.:
                v_ref += [0.8]
            else:
                v_ref += [0.0]

        y_ref[1::2] = v_ref

        if h_p_predecessor is not None:
            y_ref[0::2] = h_p_predecessor[0::2] - self.d_ref
        else:
            y_ref[0::2] = 0
            print("arange", zip(np.arange(0, self.t_s, self.h_p), v_ref))

        return y_ref

    def _y_max(self, h_p_predecessor, y_current):
        y_max = np.zeros(2 * self.h_p, dtype=np.float64)

        if h_p_predecessor is not None:
            y_max[0::2] = h_p_predecessor[0::2] - self.d_min
        else:
            y_max[0::2] = y_current[0] + 1000

        y_max[1::2] += self.v_max[0]
        return y_max

    def _y_min(self, y_current):
        y_min = np.zeros(2 * self.h_p, dtype=np.float64)
        y_min[0::2] = 0
        y_min[1::2] = self.v_min.repeat(self.h_p)
        return y_min

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

    def on_each_timestep(self, vehicle_state_list: VehicleStateList, predecessor_control_horizons: List[List[float]], t_expired_s: float) -> Tuple[float, np.array]:
        """
        Method to handle the logic at each timestep during the simulation, specifically using Model Predictive Control.

        Args:
            vehicle_state_list (VehicleStateList): The list of current vehicle states.
            predecessor_control_horizons (List[List[float]]): The list of predecessor control horizons.
            t_expired_s (float): The amount of time that has passed since the start of the simulation in seconds.

        Returns:
            Dict[int, float]: A dictionary where the key is the vehicle ID and the value is the control action (acceleration or steering) for the vehicle at this timestep.
        """
        # Get distance and speed of the vehicle on the path
        y: np.ndarray = self._mt.measure_longitudinal(vehicle_state_list)
        print("y_curr", y)

        u: np.ndarray

        pred = None if not len(predecessor_control_horizons) else predecessor_control_horizons[0]
        y_ref = self._y_ref(pred, y, t_expired_s)
        y_min = self._y_min(y)
        y_max = self._y_max(pred, y)

        u, y = self._mpc.step(
            y_measured=y,
            y_reference=y_ref,
            y_min=y_min,
            y_max=y_max)

        assert len(u) == 1, "Received control for more than one vehicle."

        # Return the control action for the vehicle
        return float(u[0]), y # TODO add control horizon
