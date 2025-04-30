from abc import ABC, abstractmethod
from typing import List, Dict

from cpnav.internal_types import VehicleStateList


class InterfaceHlc(ABC):
    """
    Abstract base class for High-Level Controller (HLC). Subclasses must implement the logic
    for handling vehicle states and control commands at different timesteps in the simulation.
    """

    def __init__(self, vehicle_ids: List[int], t_s: float):
        """
        Initializes the HLC interface with the given vehicle IDs.

        Args:
            vehicle_ids (List[int]): List of vehicle IDs for which the HLC is responsible.
            t_s (float): The sampling time, defining how frequently the controller updates.
        """
        self.vehicle_ids: List[int] = vehicle_ids
        self.t_s: float = t_s

    @abstractmethod
    def on_first_timestep(self, vehicle_state_list: VehicleStateList) -> None:
        """
        Abstract method to be implemented by subclasses to handle the logic when the system is in
        the first timestep.

        Args:
            vehicle_state_list (VehicleStateList): The current list of vehicle states.
        """
        pass

    @abstractmethod
    def on_each_timestep(self, vehicle_state_list: VehicleStateList, t_expired_s: float) -> Dict[
        int, float]:
        """
        Abstract method to be implemented by subclasses to handle the logic for each timestep
        during the simulation.

        Args:
            t_expired_s (float): The time that has expired since the start of the simulation, in seconds.
            vehicle_state_list (VehicleStateList): The current list of vehicle states.

        Returns:
            Dict[int, float]: A dictionary where the key is the vehicle ID and the value is the
            control command (or other data) associated with the vehicle.
        """
        pass
