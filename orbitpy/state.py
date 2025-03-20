"""
.. module:: orbitpy.state
   :synopsis: State vector information.

Collection of classes and functions for handling state vector information.
"""

from typing import Dict, Any, Optional
import numpy as np

from skyfield.positionlib import build_position as skyfield_build_position
from skyfield.constants import AU_KM as Skyfield_AU_KM

from .base import ReferenceFrame
from .time import AbsoluteDate
from .position import Cartesian3DPosition, Cartesian3DVelocity


class CartesianState:
    """Handles Cartesian state information."""

    def __init__(
        self,
        time: AbsoluteDate,
        position: Cartesian3DPosition,
        velocity: Cartesian3DVelocity,
        frame: Optional[ReferenceFrame],
    ) -> None:
        if frame is None:
            frame = (
                position.frame if position.frame is not None else velocity.frame
            )
        if position.frame is not None and position.frame != frame:
            raise ValueError(
                "Position frame does not match the provided frame."
            )
        if velocity.frame is not None and velocity.frame != frame:
            raise ValueError(
                "Velocity frame does not match the provided frame."
            )
        self.time: AbsoluteDate = time
        self.position: Cartesian3DPosition = position
        self.velocity: Cartesian3DVelocity = velocity
        self.frame: ReferenceFrame = frame

    @staticmethod
    def from_dict(dict_in: Dict[str, Any]) -> "CartesianState":
        """Construct a CartesianState object from a dictionary.

        Args:
            dict_in (dict): Dictionary with the Cartesian state information.
                The dictionary should contain the following key-value pairs:
                - "time" (dict): Dictionary with the date-time information.
                        See  :class:`orbitpy.util.AbsoluteDate.from_dict()`.
                - "frame" (str): The reference-frame
                                 See :class:`orbitpy.util.ReferenceFrame`.
                - "position" (List[float]): Position vector in kilometers.
                - "velocity" (List[float]): Velocity vector in km-per-s.

        Returns:
            CartesianState: CartesianState object.
        """
        time = AbsoluteDate.from_dict(dict_in["time"])
        frame = ReferenceFrame.get(dict_in["frame"])
        position = Cartesian3DPosition.from_list(dict_in["position"], frame)
        velocity = Cartesian3DVelocity.from_list(dict_in["velocity"], frame)
        return CartesianState(time, position, velocity, frame)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the CartesianState object to a dictionary.

        Returns:
            dict: Dictionary with the Cartesian state information.
        """
        return {
            "time": self.time.to_dict(),
            "position": self.position.to_list(),
            "velocity": self.velocity.to_list(),
            "frame": self.frame.value,
        }

    def to_skyfield_gcrf_position(self):
        """Convert the CartesianState object to a Skyfield position object.
        The Skyfield "position" object contains the position, velocity, time
        information, and is referenced in GCRF.

        Returns:
            Skyfield position (state) object.

        Raises:
            ValueError: If the frame is not GCRF.
        """
        if self.frame != ReferenceFrame.GCRF:
            raise ValueError(
                "Only CartesianState object in GCRF frame is supported for "
                "conversion to Skyfield GCRF position."
            )

        skyfield_time = self.time.to_skyfield_time()
        position_au = (
            np.array(self.position.to_list()) / Skyfield_AU_KM
        )  # convert to AU
        velocity_au_per_d = (
            np.array(self.velocity.to_list()) / Skyfield_AU_KM
        ) * 86400.0  # convert to AU/day
        return skyfield_build_position(
            position_au=position_au,
            velocity_au_per_d=velocity_au_per_d,
            t=skyfield_time,
            center=399,  # Geocentric
            target=None,
        )
