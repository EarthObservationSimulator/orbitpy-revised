"""
.. module:: orbitpy.utils
    :synopsis: Utilities to help with various calculations.

Collection of utility functions for various calculations. The API of the functions
in this module should only use standard and widely used data structures such as lists
and numpy arrays. It should not use custom data structures or types.
This ensures compatibility and ease of use across different parts
of the codebase.
"""

from typing import Union
import numpy as np


def normalize(v: Union[list[float], np.ndarray]) -> list[float]:
    """Normalize an input vector.
    Args:
        v (Union[list[float], np.ndarray]): Input vector to be normalized.
    Returns:
        list[float]: Normalized vector.
    Raises:
        Exception: If the input vector has zero magnitude, resulting in division by zero.
    """
    v = np.array(v)
    norm = np.linalg.norm(v)
    if norm == 0:
        raise ZeroDivisionError(
            "Encountered division by zero in vector normalization function."
        )
    return v.tolist() / norm


def check_line_of_sight(
    object1_pos: np.ndarray, object2_pos: np.ndarray, obstacle_radius: float
) -> bool:
    """
    Determine if line-of-sight exists between two objects with a spherical obstacle in between.

    This method uses the algorithm described on Page 198 of "Fundamentals of Astrodynamics
    and Applications" by David A. Vallado (first algorithm).

    Args:
        object1_pos (np.ndarray): Position vector of the first object.
        object2_pos (np.ndarray): Position vector of the second object.
        obstacle_radius (float): Radius of the spherical obstacle.

    Returns:
        bool: True if line-of-sight exists, False otherwise.

    Note:
        The frame of reference for describing the object positions must be centered at
        the spherical obstacle.
    """
    obj1_unit_vec = normalize(object1_pos)
    obj2_unit_vec = normalize(object2_pos)

    # This condition tends to give a numerical error, so solve for it independently.
    eps = 1e-9
    x = np.dot(obj1_unit_vec, obj2_unit_vec)

    if (x > -1 - eps) and (x < -1 + eps):
        return False
    else:
        if x > 1:
            x = 1
        theta = np.arccos(x)

    obj1_r = np.linalg.norm(object1_pos)
    if obj1_r - obstacle_radius > 1e-5:
        theta1 = np.arccos(obstacle_radius / obj1_r)
    elif abs(obj1_r - obstacle_radius) < 1e-5:
        theta1 = 0.0
    else:
        return False  # object1 is inside the obstacle

    obj2_r = np.linalg.norm(object2_pos)
    if obj2_r - obstacle_radius > 1e-5:
        theta2 = np.arccos(obstacle_radius / obj2_r)
    elif abs(obj2_r - obstacle_radius) < 1e-5:
        theta2 = 0.0
    else:
        return False  # object2 is inside the obstacle

    if theta1 + theta2 < theta:
        return False
    else:
        return True


def calculate_elevation_angle(
    observer_position: np.ndarray, target_position: np.ndarray
) -> float:
    """
    Calculate the elevation angle between an observer and a target.

    Args:
        observer_position (np.ndarray): A 3D numpy array representing the observer's position
                                         in Cartesian coordinates (x, y, z).
        target_position (np.ndarray): A 3D numpy array representing the target's position
                                       in Cartesian coordinates (x, y, z).

    Returns:
        float: The elevation angle in degrees.
    """
    # Vector from observer to target
    relative_vector = target_position - observer_position

    # Normalize the observer's position to get the local zenith direction
    observer_norm = np.linalg.norm(observer_position)
    if observer_norm == 0:
        raise ValueError("Observer position vector cannot be zero.")
    zenith_direction = observer_position / observer_norm

    # Normalize the relative vector
    relative_norm = np.linalg.norm(relative_vector)
    if relative_norm == 0:
        raise ValueError(
            "Relative vector between observer and target cannot be zero."
        )
    relative_unit_vector = relative_vector / relative_norm

    # Compute the dot product between the zenith direction and the relative vector
    dot_product = np.dot(zenith_direction, relative_unit_vector)

    # Clamp the dot product to avoid numerical issues with arccos
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # Calculate the elevation angle in radians
    elevation_angle_rad = np.arcsin(dot_product)

    # Convert the elevation angle to degrees
    elevation_angle_deg = np.degrees(elevation_angle_rad)

    return elevation_angle_deg
