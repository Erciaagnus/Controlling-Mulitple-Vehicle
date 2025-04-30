from typing import List, Tuple, Dict
import numpy as np
from cpnav.utils import compute_distance_on_path, compute_relative_distance_on_path
from cpnav.internal_types import PathPoint, VehicleStateList, VehicleState


class MeasurementTransformer:
    def __init__(self, path_points: List[PathPoint], vehicle_ids: List[int]):
        """
        Initialize the MeasurementTransformer instance with a given path and a list of vehicle IDs.

        Args:
            path_points (List[PathPoint]): List of PathPoint objects defining the path.
            vehicle_ids (List[int]): List of vehicle IDs to track.

        """
        self._path_points = path_points
        self._vehicle_ids = vehicle_ids
        self._n_vehicles: int = len(vehicle_ids)

        # Initialize dictionaries to store the relative position and total distance travelled for each vehicle
        self._s_on_loop: Dict[int, float] = dict.fromkeys(self._vehicle_ids, 0)
        self._s: Dict[int, float] = dict.fromkeys(self._vehicle_ids, 0)

    def measure_longitudinal(self, vehicle_state_list: VehicleStateList) -> np.ndarray:
        """
        Measure the longitudinal state (distance and speed) of each vehicle in the vehicle state list.

        Args:
            vehicle_state_list (VehicleStateList): List of vehicle states containing the positions and speeds
                                                   of all tracked vehicles.

        Returns:
            np.ndarray: An array containing the total distance and speed for each vehicle, where each vehicle
                        contributes two consecutive entries: [distance, speed].
        """
        # Initialize the result array with space for both distance and speed for each vehicle
        y: np.ndarray = np.zeros(2 * self._n_vehicles)

        vehicle_index: int = 0  # Index to track the vehicle's position in the array

        # Loop through each vehicle and compute its longitudinal measurements
        for vehicle_id in self._vehicle_ids:
            if vehicle_id not in vehicle_state_list.vehicle_states.keys():
                raise ValueError(f'Vehicle {vehicle_id} not found in vehicle state list')

            vehicle_state: VehicleState = vehicle_state_list.vehicle_states[vehicle_id]

            # Get the vehicle's position as a tuple (x, y)
            position: Tuple[float, float] = (vehicle_state.x, vehicle_state.y)

            # Compute the relative position ('s') on the path for this vehicle
            s_new: float = compute_distance_on_path(position, self._path_points)

            # Calculate the difference in 's' values (relative distance) from the previous measurement
            ds: float = compute_relative_distance_on_path(self._path_points,
                                                          self._s_on_loop[vehicle_id], s_new)

            # Update the vehicle's stored 's' values
            self._s_on_loop[vehicle_id] = s_new
            self._s[vehicle_id] += ds

            # Update the output array: each vehicle contributes two elements
            distance_index: int = vehicle_index * 2
            speed_index: int = vehicle_index * 2 + 1
            y[distance_index] = self._s[vehicle_id]  # Total distance for the vehicle
            y[speed_index] = vehicle_state.speed  # Speed of the vehicle

            vehicle_index += 1  # Move to the next vehicle

        return y

    def measure_central(self, vehicle_state_list: VehicleStateList) -> np.ndarray:
        """
        Measure the central state (longitudinal measurements and inter-vehicle distances) for each vehicle.

        This method builds on the `measure_longitudinal` method to also compute the inter-vehicle distance.
        The inter-vehicle distance is calculated by subtracting the total distance of one vehicle from the
        next one, in a cyclic order. The first vehicle's data is adjusted accordingly to ensure the output
        remains consistent.

        Args:
            vehicle_state_list (VehicleStateList): List of vehicle states containing the positions and speeds
                                                   of all tracked vehicles.

        Returns:
            np.ndarray: An array containing the total distance and speed for each vehicle along with
                        inter-vehicle distances.
        """
        # TODO Implement measure_central
        raise NotImplementedError('measure_central not implemented!')
