"""
.. module:: orbitpy.resources
   :synopsis: Models of space mission resources.

Collection of classes modeling space mission resources.
"""

from typing import Dict, Any, Optional
import uuid
from uuid import uuid4

from eosimutils.state import GeographicPosition


class GroundStation:
    """Handles ground station location and properties."""

    def __init__(
        self,
        identifier: Optional[str],
        name: Optional[str],
        geographic_position: GeographicPosition,
        min_elevation_angle_deg: float,
    ):
        """
        Args:
            identifier (str or None): Unique identifier for the ground station.
                               Must be a valid UUID.
                              If None, a new UUID is generated.
            name (str or None): Name of the ground station.
            geographic_position (:class:`orbitpy.util.GeographicPosition`):
                                    Geographic position of the ground station.
            min_elevation_angle_deg (float): Minimum elevation angle in degrees.
        """
        if identifier is not None:
            try:
                uuid.UUID(identifier)
            except ValueError as exc:
                raise ValueError("identifier must be a valid UUID.") from exc
        else:
            identifier = str(
                uuid4()
            )  # Generate a new UUID if identifier is None
        if not isinstance(geographic_position, GeographicPosition):
            raise TypeError(
                "geographic_position must be a GeographicPosition object."
            )
        if not isinstance(min_elevation_angle_deg, (int, float)):
            raise TypeError("min_elevation_angle_deg must be a numeric value.")
        if name is not None and not isinstance(name, str):
            raise TypeError("name must be a string or None.")

        self.identifier = identifier
        self.name = name
        self.geographic_position = geographic_position
        self.min_elevation_angle_deg = min_elevation_angle_deg

    @classmethod
    def from_dict(cls, dict_in: Dict[str, Any]) -> "GroundStation":
        """Construct a GroundStation object from a dictionary.

        Args:
            dict_in (dict): Dictionary with the ground station information.
                The dictionary should contain the following key-value pairs:
                - "id" (str): (Optional) Unique identifier.
                - "name" (str): (Optional) Name of the ground station.
                - "latitude" (float): Latitude in degrees.
                - "longitude" (float): Longitude in degrees.
                - "height" (float): WGS84 geodetic height in meters.
                - "min_elevation_angle" (float): Minimum elevation angle
                                                 in degrees.

        Returns:
            GroundStation: GroundStation object.
        """
        identifier = dict_in.get("id")  # allow id to be None
        name = dict_in["name"]
        latitude = dict_in["latitude"]
        longitude = dict_in["longitude"]
        height = dict_in["height"]
        min_elevation_angle_deg = dict_in["min_elevation_angle"]
        geographic_position = GeographicPosition.from_dict(
            {"latitude": latitude, "longitude": longitude, "elevation": height}
        )
        return cls(
            identifier, name, geographic_position, min_elevation_angle_deg
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the GroundStation object to a dictionary.

        Returns:
            dict: Dictionary with the ground station information.
        """
        return {
            "id": self.identifier,
            "name": self.name,
            "latitude": self.geographic_position.latitude,
            "longitude": self.geographic_position.longitude,
            "height": self.geographic_position.elevation,
            "min_elevation_angle": self.min_elevation_angle_deg,
        }
