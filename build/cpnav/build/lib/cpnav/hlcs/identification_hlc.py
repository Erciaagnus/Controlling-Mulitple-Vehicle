import os
import pickle
from typing import List, Dict

import numpy as np

from cpnav.hlcs import InterfaceHlc
from cpnav.utils import outer_lane_path, MeasurementTransformer
from cpnav.internal_types import PathPoint, VehicleStateList
import time


class IdentificationHlc(InterfaceHlc):
    """
    High-Level Controller (HLC) for model identification in a cooperative driving system.
    This class implements the InterfaceHlc and provides functionality to handle vehicle state
    and control commands for the purpose of identifying a vehicle model in a simulation.
    """

    def __init__(self, vehicle_ids: List[int], t_s: float):
        """
        Initializes the Identification HLC with vehicle IDs and sampling time.
        This controller is specifically designed for model identification and supports only
        a single vehicle for this task.

        Args:
            vehicle_ids (List[int]): List of vehicle IDs for which the controller will be responsible.
            t_s (float): The sampling time, defining how frequently the controller updates.
        
        Hints:
            You may want to initialize something for pickle.
        """
        super().__init__(vehicle_ids, t_s)

        # Generate the path the vehicle will follow (e.g., outer lane path)
        self._path_points: List[PathPoint] = outer_lane_path()

        # Initialize the MeasurementTransformer to track vehicle measurements
        self._mt = MeasurementTransformer(self._path_points, vehicle_ids)

        self._u_k_minus_one: float | None = None

        if os.path.exists("time-series-data.pkl"):
            os.remove("time-series-data.pkl")  # Clear the file if it exists
        with open("time-series-data.pkl", "wb") as f:
            pickle.dump({"u_in": [], "u_out": []}, f)

        # Ensure only one vehicle is used for model identification
        assert len(vehicle_ids) == 1, "Only one vehicle is allowed for model identification."

    def on_first_timestep(self, vehicle_state_list: VehicleStateList) -> None:
        """
        Method to handle the logic for the first timestep in the simulation.
        This method can be used to set up initial conditions if necessary.

        Args:
            vehicle_state_list (VehicleStateList): The list of current vehicle states at the first timestep.
        """
        pass

    def on_each_timestep(self, vehicle_state_list: VehicleStateList, t_expired_s: float) -> Dict[
        int, float]:
        """
        Method to handle the logic at each timestep during the simulation, specifically for model identification.

        Args:
            vehicle_state_list (VehicleStateList): The list of current vehicle states.
            t_expired_s (float): The amount of time that has passed since the start of the simulation in seconds.

        Returns:
            Dict[int, float]: A dictionary where the key is the vehicle ID and the value is the control action
                              (e.g., acceleration or steering) for the vehicle at this timestep.
        Hints:
            - You only need to simulate one vehicle for system identification.
            - Data structure for the pickle: {'u_in': [], 'u_out': []}.
            - Use self._mt.measure_longitudinal(vehicle_state_list) directly as u_out.
            - The assertions in utils/system_identification.py may give some pieces of information.
            - Vary your u_in to comprehensively test the system dynamics.
            - Run the simulation for at least one minue to collect enough data points.
        """
        # Measure distance and speed
        y: np.ndarray = self._mt.measure_longitudinal(vehicle_state_list)
        assert y.shape[0] == 2, "Received measurements for more than one vehicle."


        vehicle_id = self.vehicle_ids[0]
        vehicle_state = vehicle_state_list.vehicle_states.get(vehicle_id)

        # u = abs(0.5 * np.sin(2 * np.pi * 0.1 * t))
        # if(t_expired_s < 5):
        #     u = 0.0
        # else:
        #     u = 1.0
        
        # if(t_expired_s > 20):
        #     u = abs(np.sin(t_expired_s))

        # if(t_expired_s > 30):
        #     u = 0.0
        # if(t_expired_s > 31):
        #     time.sleep(9999)

        u = .0
        if t_expired_s < 15:
            u = 0.1 * t_expired_s * np.sin(2 * np.pi * 0.1 * t_expired_s) + 0.02 * np.random.normal() # Sine wave + noise
        if t_expired_s >= 15 and t_expired_s < 30:
            if np.floor(t_expired_s) % 5 == 0:
                u = 1.
            else:
                u = 0.
        if t_expired_s >= 30 and t_expired_s < 55:
            u = 1.

        if t_expired_s >= 60:
            time.sleep(9999)

        with open("time-series-data.pkl", "rb") as f:
            data = pickle.load(f)

        

        data["u_in"].append(u)
        data["u_out"].append([vehicle_state.speed, y[0]])
       # data["u_out"].append(np.array([x, y, speed]))

        with open("time-series-data.pkl", "wb") as f:
            pickle.dump(data, f)
        

        # Return the control action for the vehicle
        return {self.vehicle_ids[0]: u}
