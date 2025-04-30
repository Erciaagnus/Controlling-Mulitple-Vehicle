from dataclasses import dataclass


@dataclass
class PathPoint:
    """
    Represents an oriented point on a path in 2D space.

    Attributes:
        x (float): The x-coordinate of the point.
        y (float): The y-coordinate of the point.
        yaw (float): The orientation angle of the point (in radians).
        s (float): The distance of the point on the path.
    """
    x: float
    y: float
    yaw: float
    s: float
