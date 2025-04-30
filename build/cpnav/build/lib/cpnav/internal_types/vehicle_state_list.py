from dataclasses import dataclass

from cpnav.internal_types import VehicleState

from typing import Dict


@dataclass
class VehicleStateList:
    """
    Represents a collection of vehicle states along with the current timestamp and the time period
    between updates.

    Attributes:
        vehicle_states (Dict[int, VehicleState]): A dictionary mapping vehicle IDs to their
        corresponding VehicleState objects.
        t_now (int): The current timestamp (e.g., time in milliseconds).
        period_ms (float): The time period between each update in milliseconds.
    """
    vehicle_states: Dict[int, VehicleState]
    t_now: int
    period_ms: float
