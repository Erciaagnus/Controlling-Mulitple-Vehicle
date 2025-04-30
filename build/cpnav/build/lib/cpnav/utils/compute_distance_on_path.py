import numpy as np
from typing import List, Tuple

from cpnav.internal_types import PathPoint


def compute_distance_on_path(position: Tuple[float, float], path_points: List[PathPoint]) -> float:
    """
    Computes the minimum distance between a given position and the path defined by a sequence of path points.

    The function subdivides the path segments into smaller segments and iteratively refines the path to find the
    closest point on the path to the given position. This is done by computing the squared Euclidean distance
    between the queried position and the interpolated points on the path, and selecting the smallest distance.

    Args:
        position (Tuple[float, float]): A tuple representing the coordinates (x, y) of the point to compute
                                       the minimum distance to.
        path_points (List[PathPoint]): A list of `PathPoint` objects, each containing position (x, y) and yaw
                                       values along the path.

    Returns:
        float: The minimum distance from the queried position to the path. This distance is computed by
               subdividing the path and finding the point closest to the given position.
    """
    # Ensure that there are at least two points in the path
    assert (len(path_points) > 1)

    # Number of subdivisions per path segment (refining the resolution)
    num_subdivs: int = 4
    n_points: int = len(path_points)

    # Indices of the path segments (between consecutive points)
    is_: np.ndarray = np.arange(0, n_points - 1)

    # Start and end arc-lengths for each segment
    s_a: np.ndarray = np.array([path_points[i].s for i in is_])
    s_b: np.ndarray = np.array([path_points[i + 1].s for i in is_])

    # Iterative process to refine the path and find the closest point
    localminidx: np.ndarray
    distance_squared: np.ndarray
    localmin: np.ndarray

    # Perform several iterations to refine the subdivision of the path
    for _ in range(5):
        # Subdivide each segment into smaller subsegments
        new_is_: np.ndarray = np.ones(len(is_) * num_subdivs, dtype=np.int64)
        new_s_a: np.ndarray = np.ones(len(is_) * num_subdivs)
        new_s_b: np.ndarray = np.ones(len(is_) * num_subdivs)

        # Generate the new subdivided path segments
        for i in range(len(is_)):
            delta_s: np.ndarray = (s_b[i] - s_a[i]) / num_subdivs  # Length of each subsegment
            s_base: float = float(s_a[i])
            for j in range(num_subdivs):
                new_i: int = i * num_subdivs + j
                new_is_[new_i] = is_[i]  # Assign the segment index
                new_s_a[new_i] = s_base  # Start of the subsegment
                s_base += delta_s  # Increment start of the next subsegment
                new_s_b[new_i] = s_base  # End of the subsegment

        # Update segment indices and arc-length arrays
        is_ = new_is_
        s_a = new_s_a
        s_b = new_s_b

        # Midpoint of each subsegment (query points for distance calculation)
        s_query: np.ndarray = (s_b + s_a) / 2

        # Compute distances from the queried position to each segment of the path
        dx: np.ndarray
        dy: np.ndarray
        dx, dy = compute_distance_on_segment(position, s_query,
                                             [path_points[i] for i in is_],
                                             [path_points[i + 1] for i in is_])
        # Squared Euclidean distance
        distance_squared: np.ndarray = dx * dx + dy * dy
        num_rows: int = len(distance_squared)

        # Padding with infinity to identify local minima
        padded_distances = np.concatenate(([np.inf], distance_squared, [np.inf]))

        # Identify local minima in the distance squared array
        localminidx = np.where((padded_distances[1:-1] < padded_distances[:-2]) &
                               (padded_distances[1:-1] < padded_distances[2:]))[0]

        # Mark the local minima
        localmin = np.zeros(num_rows, dtype=bool)
        localmin[localminidx] = True

        # Update the path segments to retain only those corresponding to local minima
        is_ = is_[localmin]
        s_a = s_a[localmin]
        s_b = s_b[localmin]

    # Select the minimum distance from the refined distances
    distance_squared = distance_squared[localmin]

    # Get the final query points corresponding to the local minima
    s_query = (s_a + s_b) / 2

    # Stack the arc-length and the corresponding squared distances
    query_dist = np.vstack([s_query, distance_squared])

    # Sort based on distance squared to find the minimum distance
    sorted_idx = np.argsort(query_dist[1, :])
    sorted_query_dist = query_dist[:, sorted_idx]

    # The minimum distance is the one corresponding to the smallest squared distance
    s_k_on_loop = sorted_query_dist[0, 0]

    return float(s_k_on_loop)


def compute_distance_on_segment(position: Tuple[float, float],
                                s_query: np.ndarray,
                                points_a: List[PathPoint],
                                points_b: List[PathPoint]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the Euclidean distance from a given position to each interpolated point on a path segment.

    This function calculates the distance between a specified position `(x, y)` and a set of interpolated points
    on the path. The path is defined by the start (`points_a`) and end (`points_b`) points, and the interpolation
    is performed at queried arc-length values `s_query` using the `path_interpolation` function.

    Args:
        position (Tuple[float, float]): A tuple representing the coordinates (x, y) of the position to compute distances to.
        s_query (np.ndarray): An array of queried arc-length values at which to compute the interpolated path.
        points_a (List[PathPoint]): A list of start points, each containing position (x, y) and yaw values.
        points_b (List[PathPoint]): A list of end points, each containing position (x, y) and yaw values.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Two numpy arrays containing the x and y components of the distance
                                        between the queried position and each interpolated point along the path.
    """
    # Interpolate the x and y positions along the path at the queried arc-length values
    interpolation_x: np.ndarray
    interpolation_y: np.ndarray
    interpolation_x, interpolation_y = path_interpolation(s_query, points_a, points_b)

    # Compute the differences between the queried position and the interpolated positions along the path
    dx: np.ndarray = position[0] * np.ones(len(s_query)) - interpolation_x
    dy: np.ndarray = position[1] * np.ones(len(s_query)) - interpolation_y

    # Return the x and y components of the distance (dx, dy) between the queried position and the interpolated points
    return dx, dy


def path_interpolation(s_queried: np.ndarray, start_points: List[PathPoint],
                       end_points: List[PathPoint]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolates between a set of start and end points along a path using a cubic Hermite spline.

    This function calculates intermediate positions (x, y) along the path between the start and end points.
    The interpolation uses the start and end positions, along with the start and end velocities, to compute
    the smooth path at given queried arc-length values.

    Args:
        s_queried (np.ndarray): Array of queried arc-length values at which to compute interpolated positions.
        start_points (List[PathPoint]): List of start points, each containing position (x, y) and yaw values.
        end_points (List[PathPoint]): List of end points, each containing position (x, y) and yaw values.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Interpolated x and y positions along the path at the queried arc-length values.
    """

    # Extract arc-length (s) values from the start and end points
    s_start: np.ndarray = np.array([start_point.s for start_point in start_points])
    s_end: np.ndarray = np.array([end_point.s for end_point in end_points])

    # Extract the x, y positions, and yaw (orientation) values for start and end points
    position_start_x: np.ndarray = np.array([start_point.x for start_point in start_points])
    position_end_x: np.ndarray = np.array([end_point.x for end_point in end_points])
    position_start_y: np.ndarray = np.array([start_point.y for start_point in start_points])
    position_end_y: np.ndarray = np.array([end_point.y for end_point in end_points])
    position_start_yaw: np.ndarray = np.array([start_point.yaw for start_point in start_points])
    position_end_yaw: np.ndarray = np.array([end_point.yaw for end_point in end_points])

    # Compute the change in arc-length (delta_s) between the start and end points
    delta_s: np.ndarray = s_end - s_start

    # Normalize the queried arc-length values to a range [0, 1] between start and end points
    tau: np.ndarray = (s_queried - s_start) / delta_s

    # Compute tau^2 and tau^3 for use in cubic spline interpolation
    tau2: np.ndarray = tau ** 2
    tau3: np.ndarray = tau ** 3

    # Calculate the velocity components (in the x and y directions) based on the yaw angles
    velocity_start_x: np.ndarray = np.cos(position_start_yaw) * delta_s
    velocity_start_y: np.ndarray = np.sin(position_start_yaw) * delta_s
    velocity_end_x: np.ndarray = np.cos(position_end_yaw) * delta_s
    velocity_end_y: np.ndarray = np.sin(position_end_yaw) * delta_s

    # Compute the cubic Hermite spline basis functions
    p0: np.ndarray = 2 * tau3 - 3 * tau2 + 1  # Position weight for the start point
    m0: np.ndarray = tau3 - 2 * tau2 + tau  # Velocity weight for the start point
    p1: np.ndarray = -2 * tau3 + 3 * tau2  # Position weight for the end point
    m1: np.ndarray = tau3 - tau2  # Velocity weight for the end point

    # Interpolate x and y positions along the path using the cubic spline formula
    interpolation_x: np.ndarray = (position_start_x * p0 + velocity_start_x * m0 +
                                   position_end_x * p1 + velocity_end_x * m1)
    interpolation_y: np.ndarray = (position_start_y * p0 + velocity_start_y * m0 +
                                   position_end_y * p1 + velocity_end_y * m1)

    # Return the interpolated x and y positions as a tuple
    return interpolation_x, interpolation_y
