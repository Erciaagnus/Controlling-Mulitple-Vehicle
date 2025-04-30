import numpy as np
from typing import List
from cpnav.internal_types import PathPoint


def outer_lane_path() -> List[PathPoint]:
    """Function to generate the outer lane path points.

    This function constructs the path points for the outer lane of a defined path,
    consisting of straight and circular segments. The path is represented in 2D space
    with x, y coordinates, and each point has an associated yaw and cumulative distance 's'.

    Returns:
        List[PathPoint]: A list of PathPoint objects representing the path.
    """
    # Define the center coordinates and radius for the outer lane
    center_left_x: float = 1.25
    center_right_x: float = 3.25
    center_top_y: float = 2.75
    center_bottom_y: float = 1.25
    radius: float = 1.0

    # Calculate segment distances
    ds_lr: float = center_right_x - center_left_x  # Distance between left and right sides
    ds_tb: float = center_top_y - center_bottom_y  # Distance between top and bottom sides
    ds_qc: float = radius * np.pi / 2  # Quarter-circle arc length

    # Define yaw angles for each segment
    yaw: np.ndarray = np.array([
        0,  # 1st horizontal segment (right)
        0,  # 2nd horizontal segment (right)
        3 * np.pi / 2,  # 1st vertical segment (down)
        3 * np.pi / 2,  # 2nd vertical segment (down)
        np.pi,  # 1st horizontal segment (left)
        np.pi,  # 2nd horizontal segment (left)
        np.pi / 2,  # 1st vertical segment (up)
        np.pi / 2,  # 2nd vertical segment (up)
        0  # Closing the loop (right)
    ])

    # Define x coordinates for each path segment
    x: np.ndarray = np.array([
        center_left_x,  # Start at left side
        center_right_x,  # Move right
        center_right_x,  # Move right vertically
        center_right_x,  # Continue downward
        center_right_x,  # Continue downward horizontally
        center_left_x,  # Move left horizontally
        center_left_x,  # Continue left vertically
        center_left_x,  # Continue upward
        center_left_x  # Close the loop back to the left side
    ])

    # Adjust x coordinates to include the radius offset based on yaw angles
    x -= radius * np.sin(yaw)

    # Define y coordinates for each path segment
    y: np.ndarray = np.array([
        center_top_y,  # Top segment
        center_top_y,  # Top segment
        center_top_y,  # Moving down
        center_bottom_y,  # Continue down
        center_bottom_y,  # Continue down horizontally
        center_bottom_y,  # Continue bottom
        center_bottom_y,  # Moving up
        center_top_y,  # Continue up
        center_top_y  # Close the loop back to top
    ])

    # Adjust y coordinates to include the radius offset based on yaw angles
    y += radius * np.cos(yaw)

    # Define cumulative distance 's' along the path for each point
    s: np.ndarray = np.array([
        0 * ds_lr + 0 * ds_qc + 0 * ds_tb,  # Point 1
        1 * ds_lr + 0 * ds_qc + 0 * ds_tb,  # Point 2
        1 * ds_lr + 1 * ds_qc + 0 * ds_tb,  # Point 3
        1 * ds_lr + 1 * ds_qc + 1 * ds_tb,  # Point 4
        1 * ds_lr + 2 * ds_qc + 1 * ds_tb,  # Point 5
        2 * ds_lr + 2 * ds_qc + 1 * ds_tb,  # Point 6
        2 * ds_lr + 3 * ds_qc + 1 * ds_tb,  # Point 7
        2 * ds_lr + 3 * ds_qc + 2 * ds_tb,  # Point 8
        2 * ds_lr + 4 * ds_qc + 2 * ds_tb  # Point 9
    ])

    # Validate that all arrays have the same length
    assert len(s) == len(x) == len(y) == len(yaw), "Array lengths do not match."

    # Create a list of PathPoint objects
    path_points: List[PathPoint] = [
        PathPoint(float(x[i]), float(y[i]), float(yaw[i]), float(s[i]))
        for i in range(len(s))
    ]

    return path_points
