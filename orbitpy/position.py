"""
.. module:: orbitpy.position
   :synopsis: Position information.

Collection of classes and functions for handling position information.
"""

import numpy as np
from typing import Dict, Any, List, Optional

from skyfield.api import wgs84 as skyfield_wgs84

from .base import ReferenceFrame


class Cartesian3DPosition:
    """Handles 3D position information."""

    def __init__(
        self, x: float, y: float, z: float, frame: Optional[ReferenceFrame]
    ) -> None:
        if not all(isinstance(coord, (int, float)) for coord in [x, y, z]):
            raise ValueError("x, y, and z must be numeric values.")
        if frame is not None and not isinstance(frame, ReferenceFrame):
            raise ValueError("frame must be a ReferenceFrame object or None.")
        self.coords = np.array([x, y, z])
        self.frame = frame

    @staticmethod
    def from_list(
        list_in: List[float], frame: Optional[ReferenceFrame]
    ) -> "Cartesian3DPosition":
        """Construct a Cartesian3DPosition object from a list.

        Args:
            list_in (List[float]): Position coordinates in kilometers.
            frame (ReferenceFrame): The reference-frame.

        Returns:
            Cartesian3DPosition: Cartesian3DPosition object.
        """
        if len(list_in) != 3:
            raise ValueError("The list must contain exactly 3 elements.")
        if not all(isinstance(coord, (int, float)) for coord in list_in):
            raise ValueError("All elements in list_in must be numeric values.")
        return Cartesian3DPosition(list_in[0], list_in[1], list_in[2], frame)

    def to_list(self) -> List[float]:
        """Convert the Cartesian3DPosition object to a list.

        Returns:
            List[float]: List with the position coordinates in kilometers.
        """
        return self.coords.tolist()

    @staticmethod
    def from_dict(dict_in: Dict[str, Any]) -> "Cartesian3DPosition":
        """Construct a Cartesian3DPosition object from a dictionary.

        Args:
            dict_in (dict): Dictionary with the position information.
                The dictionary should contain the following key-value
                pairs:
                - "x" (float): The x-coordinate in kilometers.
                - "y" (float): The y-coordinate in kilometers.
                - "z" (float): The z-coordinate in kilometers.
                - "frame" (str): The reference-frame,
                                see :class:`orbitpy.util.ReferenceFrame`.

        Returns:
            Cartesian3DPosition: Cartesian3DPosition object.
        """
        frame = ReferenceFrame.get(dict_in["frame"])
        return Cartesian3DPosition(dict_in["x"], dict_in["y"], dict_in["z"], frame)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the Cartesian3DPosition object to a dictionary.

        Returns:
            dict: Dictionary with the position information.
        """
        return {
            "x": self.coords[0],
            "y": self.coords[1],
            "z": self.coords[2],
            "frame": self.frame.value if self.frame else None,
        }


class Cartesian3DVelocity:
    """Handles 3D velocity information."""

    def __init__(
        self, vx: float, vy: float, vz: float, frame: Optional[ReferenceFrame]
    ) -> None:
        if not all(isinstance(coord, (int, float)) for coord in [vx, vy, vz]):
            raise ValueError("vx, vy, and vz must be numeric values.")
        if frame is not None and not isinstance(frame, ReferenceFrame):
            raise ValueError("frame must be a ReferenceFrame object or None.")
        self.coords = np.array([vx, vy, vz])
        self.frame = frame

    @staticmethod
    def from_list(
        list_in: List[float], frame: Optional[ReferenceFrame]
    ) -> "Cartesian3DVelocity":
        """Construct a Cartesian3DVelocity object from a list.

        Args:
            list_in (List[float]): Velocity in km-per-s.
            frame (ReferenceFrame): The reference-frame.

        Returns:
            Cartesian3DVelocity: Cartesian3DVelocity object.
        """
        if len(list_in) != 3:
            raise ValueError("The list must contain exactly 3 elements.")
        if not all(isinstance(coord, (int, float)) for coord in list_in):
            raise ValueError("All elements in list_in must be numeric values.")
        return Cartesian3DVelocity(list_in[0], list_in[1], list_in[2], frame)

    def to_list(self) -> List[float]:
        """Convert the Cartesian3DVelocity object to a list.

        Returns:
            List[float]: Velocity coordinates in kilometers-per-second.
        """
        return self.coords.tolist()

    @staticmethod
    def from_dict(dict_in: Dict[str, Any]) -> "Cartesian3DVelocity":
        """Construct a Cartesian3DVelocity object from a dictionary.

        Args:
            dict_in (dict): Dictionary with the velocity information.
                The dictionary should contain the following key-value
                pairs:
                - "vx" (float): The x-coordinate in km-per-s.
                - "vy" (float): The y-coordinate in km-per-s.
                - "vz" (float): The z-coordinate in km-per-s.
                - "frame" (str): The reference-frame,
                    see :class:`orbitpy.util.ReferenceFrame`.

        Returns:
            Cartesian3DVelocity: Cartesian3DVelocity object.
        """
        frame = ReferenceFrame.get(dict_in["frame"])
        return Cartesian3DVelocity(dict_in["vx"], dict_in["vy"], dict_in["vz"], frame)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the Cartesian3DVelocity object to a dictionary.

        Returns:
            dict: Dictionary with the velocity information.
        """
        return {
            "vx": self.coords[0],
            "vy": self.coords[1],
            "vz": self.coords[2],
            "frame": self.frame.value if self.frame else None,
        }


class GeographicPosition:
    """Handles geographic position information using Skyfield.
    The geographic position is managed internally using the Skyfield
    GeographicPosition object and is referenced to the WGS84 ellipsoid."""

    def __init__(
        self,
        latitude_degrees: float,
        longitude_degrees: float,
        elevation_m: float,
    ):
        """
        Args:
            latitude_degrees (float): Latitude in degrees.
            longitude_degrees (float): Longitude in degrees.
            elevation_m (float): Elevation in meters.
        """
        self.skyfield_geo_position = skyfield_wgs84.latlon(
            latitude_degrees=latitude_degrees,
            longitude_degrees=longitude_degrees,
            elevation_m=elevation_m,
        )

    @classmethod
    def from_dict(cls, dict_in: Dict[str, Any]) -> "GeographicPosition":
        """Construct a GeographicPosition object from a dictionary.

        Args:
            dict_in (dict): Dictionary with the geographic position information.
                The dictionary should contain the following key-value pairs:
                - "latitude" (float): Latitude in degrees.
                - "longitude" (float): Longitude in degrees.
                - "elevation" (float): Elevation in meters.

        Returns:
            GeographicPosition: GeographicPosition object.
        """
        latitude_degrees = dict_in["latitude"]
        longitude_degrees = dict_in["longitude"]
        elevation_m = dict_in["elevation"]
        return cls(latitude_degrees, longitude_degrees, elevation_m)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the GeographicPosition object to a dictionary.

        Returns:
            dict: Dictionary with the geographic position information.
        """
        return {
            "latitude": self.skyfield_geo_position.latitude.degrees,
            "longitude": self.skyfield_geo_position.longitude.degrees,
            "elevation": self.skyfield_geo_position.elevation.m,
        }

    @property
    def latitude(self):
        """Get the latitude in degrees."""
        return self.skyfield_geo_position.latitude.degrees

    @property
    def longitude(self):
        """Get the longitude in degrees."""
        return self.skyfield_geo_position.longitude.degrees

    @property
    def elevation(self):
        """Get the elevation in meters."""
        return self.skyfield_geo_position.elevation.m

    @property
    def itrs_xyz(self):
        """Get the ITRS XYZ position in kilometers."""
        return self.skyfield_geo_position.itrs_xyz.km
