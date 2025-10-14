"""
.. module:: orbitpy.resources
   :synopsis: Models of space mission resources.

Collection of classes modeling space mission resources.
"""

from typing import Dict, Any, Optional, Union, List
import uuid
from uuid import uuid4

from eosimutils.state import GeographicPosition
from eosimutils.fieldofview import (
    FieldOfViewFactory,
    CircularFieldOfView,
    RectangularFieldOfView,
    PolygonFieldOfView,
)
from eosimutils.standardframes import StandardFrameHandlerFactory, LVLHType1FrameHandler

from .orbits import OrbitFactory, TwoLineElementSet, OrbitalMeanElementsMessage, OsculatingElements


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
        identifier = dict_in.get("id", None)  # allow id to be None
        name = dict_in.get("name", None)
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


class Sensor:
    """Handles sensor properties."""

    def __init__(
        self,
        identifier: Optional[str],
        name: Optional[str],
        fov: Union[
            CircularFieldOfView, RectangularFieldOfView, PolygonFieldOfView
        ],
    ):
        """
        Args:
            identifier (str or None): Unique identifier for the ground station.
                               Must be a valid UUID.
                              If None, a new UUID is generated.
            name (str or None): Name of the sensor.
            fov (FieldOfView): Field of view object.
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
        self.identifier = identifier
        self.name = name
        self.fov = fov
    
    @classmethod
    def from_dict(cls, dict_in: Dict[str, Any]) -> "Sensor":
        """Construct a Sensor object from a dictionary.

        Args:
            dict_in (dict): Dictionary with the sensor information.
                The dictionary should contain the following key-value pairs:
                - "id" (str): (Optional) Unique identifier.
                - "name" (str): (Optional) Name of the sensor.
                - "fov" (FieldOfView): Field of view object.

        Returns:
            Sensor: Sensor object.
        """
        identifier = dict_in.get("id")
        name = dict_in.get("name")
        fov = FieldOfViewFactory.from_dict(dict_in["fov"])
        return cls(identifier, name, fov)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the Sensor object to a dictionary.

        Returns:
            dict: Dictionary with the sensor information.
        """
        return {
            "id": self.identifier,
            "name": self.name,
            "fov": self.fov.to_dict(),
        }

class Spacecraft:
    """Handles spacecraft properties."""

    def __init__(
        self,
        identifier: Optional[str],
        name: Optional[str],
        norad_id: Optional[int] = None,
        orbit: Optional[Union[TwoLineElementSet, OrbitalMeanElementsMessage, OsculatingElements]] = None,
        local_orbital_frame_handler: Optional[Union[LVLHType1FrameHandler]] = None,
        sensor: Optional[List[Sensor]] = None
    ):
        """
        Args:
            identifier (str or None): Unique identifier for the ground station.
                               Must be a valid UUID.
                              If None, a new UUID is generated.
            name (str or None): (Optional) Name of the spacecraft.
            norad_id (int or None): (Optional) NORAD ID of the spacecraft. Defaults to None.
            orbit (Union[TwoLineElementSet, OrbitalMeanElementsMessage, OsculatingElements] or None): (Optional) Orbit information. Defaults to None.
            local_orbital_frame_handler (Union[LVLHType1FrameHandler]): (Optional) Local orbital frame information (e.g., LVLH Type-1 frame handler).
            sensor (List[Sensor] or Sensor): (Optional) List of Sensor objects or a single Sensor object.
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
        self.identifier = identifier

        self.name = name

        if norad_id is not None and not isinstance(norad_id, int):
            raise TypeError("norad_id must be an integer or None.")
        self.norad_id = norad_id
        
        if orbit is not None and not isinstance(
            orbit, (TwoLineElementSet, OrbitalMeanElementsMessage, OsculatingElements)
        ):
            raise TypeError(
                "orbit must be a TwoLineElementSet, OrbitalMeanElementsMessage, or OsculatingElements object."
            )
        self.orbit = orbit

        if local_orbital_frame_handler is not None and not isinstance(local_orbital_frame_handler, LVLHType1FrameHandler):
            raise TypeError("local_orbital_frame_handler must be a frame handler object.")
        self.local_orbital_frame_handler = local_orbital_frame_handler

        if sensor is not None:
            if not isinstance(sensor, list):
                sensor = [sensor]  # Convert single sensor object to a list
            if not all(isinstance(s, Sensor) for s in sensor):
                raise TypeError("sensor must be a list of Sensor objects.")
        self.sensor = sensor

    @classmethod
    def from_dict(cls, dict_in: Dict[str, Any]) -> "Spacecraft":
        """Construct a Spacecraft object from a dictionary.

        Args:
            dict_in (dict): Dictionary with the spacecraft information.
                The dictionary should contain the following key-value pairs:
                - "id" (str): (Optional) Unique identifier.
                - "name" (str): (Optional) Name of the spacecraft.
                - "norad_id" (int): (Optional) NORAD ID of the spacecraft.
                - "orbit" (dict): (Optional) Orbit information. See `orbitpy.orbits.OrbitFactory.from_dict`.
                - "local_orbital_frame_handler" (dict): (Optional) Local orbital frame information (handler). See `eosimutils.standardframes.StandardFrameHandlerFactory.from_dict`.
                - "sensor" (List[dict] or dict): (Optional) List of Sensors or a single Sensor. See `orbitpy.resources.Sensor.from_dict`.

        Returns:
            Spacecraft: Spacecraft object.
        """
        identifier = dict_in.get("id")
        name = dict_in.get("name")
        orbit_data = dict_in.get("orbit", None)
        orbit = OrbitFactory.from_dict(orbit_data) if orbit_data else None
        local_orbital_frame_handler = dict_in.get("local_orbital_frame_handler", None)
        if local_orbital_frame_handler is not None:
            local_orbital_frame_handler = StandardFrameHandlerFactory.from_dict(local_orbital_frame_handler)
        sensor_data = dict_in.get("sensor", None)
        if isinstance(sensor_data, dict):  # Single sensor object
            sensor = [Sensor.from_dict(sensor_data)]
        elif isinstance(sensor_data, list):  # List of sensor objects
            sensor = [Sensor.from_dict(s) for s in sensor_data]
        else:
            sensor = None
        norad_id = dict_in.get("norad_id", None)
        return cls(identifier, name, norad_id, orbit, local_orbital_frame_handler, sensor)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the Spacecraft object to a dictionary.
        
        Returns:
            dict: Dictionary with the spacecraft information.
        """
        return {
            "id": self.identifier,
            "name": self.name,
            "norad_id": self.norad_id,
            "orbit": self.orbit.to_dict() if self.orbit else None,
            "local_orbital_frame_handler": self.local_orbital_frame_handler.to_dict() if self.local_orbital_frame_handler else None,
            "sensor": [s.to_dict() for s in self.sensor] if self.sensor else None,
        }
