from typing import List
from cpnav.internal_types import PathPoint


def compute_relative_distance_on_path(path_points: List[PathPoint], s_1: float,
                                      s_2: float) -> float:
    """
    Computes the relative distance between two arc-lengths (s_1 and s_2) along a path, considering the
    periodicity of the path.

    The function calculates the distance from `s_1` to `s_2` while taking into account that the path may be
    periodic (i.e., looping back to the start after reaching the end). If the difference between `s_2` and
    `s_1` exceeds half the total path length, the path is treated as circular, and the function adjusts the
    distance accordingly.

    Args:
        path_points (List[PathPoint]): A list of `PathPoint` objects that define the path. The last point in
                                       this list is used to get the total length of the path (`s_max`).
        s_1 (float): The arc-length value at the starting point.
        s_2 (float): The arc-length value at the ending point.

    Returns:
        float: The relative distance between `s_1` and `s_2` on the path, considering periodicity.
    """

    # Total arc-length of the path is taken from the last PathPoint in the list
    s_max: float = path_points[-1].s

    # Calculate the raw difference between s_2 and s_1
    ds: float = s_2 - s_1

    # If the distance is greater than half the total path length, adjust to account for periodicity
    if ds < -s_max / 2:
        ds += s_max

    # Return the adjusted relative distance
    return ds
