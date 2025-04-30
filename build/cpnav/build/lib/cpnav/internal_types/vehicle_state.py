from dataclasses import dataclass


@dataclass
class VehicleState:
    """
    Represents the state of a vehicle, including position, orientation, speed, and vehicle ID.

    Attributes:
        vehicle_id (int): The unique identifier for the vehicle.
        x (float): The x-coordinate of the vehicle's position.
        y (float): The y-coordinate of the vehicle's position.
        yaw (float): The orientation angle of the vehicle.
        speed (float): The current speed of the vehicle.
    """
    vehicle_id: int
    x: float
    y: float
    yaw: float
    speed: float
