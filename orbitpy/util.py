"""
.. module:: orbitpy.util
   :synopsis: Collection of utility classes and functions.

Collection of utility classes and functions used by the `orbitpy` package.
Note some utility classes/functions are imported from the `instrupy` package.
"""

from typing import Dict, Any, List, Optional
from enum import Enum

from astropy.time import Time as Astropy_Time


class EnumBase(str, Enum):
    """Enumeration of recognized types.
    All enum values defined by the inheriting class are
    expected to be in uppercase."""

    @classmethod
    def get(cls, key):
        """Attempts to parse a type from a string, otherwise returns None."""
        if isinstance(key, cls):
            return key
        elif isinstance(key, list):
            return [cls.get(e) for e in key]
        else:
            try:
                return cls(key.upper())
            except:  # pylint: disable=bare-except
                return None


class TimeFormat(EnumBase):
    """
    Enumeration of recognized time formats.
    """

    GREGORIAN_DATE = "GREGORIAN_DATE"
    JULIAN_DATE = "JULIAN_DATE"


class TimeScale(EnumBase):
    """
    Enumeration of recognized time scales.
    """

    UT1 = "UT1"
    UTC = "UTC"


class AbsoluteDate:
    """Handles date-time information with support to Julian and Gregorian
    date-time formats and UT1 and UTC time scales. Date-time is managed
    internally using the Astropy Time object.

    .. note:: Skyfield Time was not used since it appears not to 
    support handling of JD-UTC."""

    def __init__(self, astropy_time: Astropy_Time) -> None:
        """Constructor for the AbsoluteDate class.

        Args:
            astropy_time (astropy.time.Time): Astropy Time object.
        """
        self.astropy_time = astropy_time

    @staticmethod
    def from_dict(dict_in: Dict[str, Any]) -> "AbsoluteDate":
        """Construct an AbsoluteDate object from a dictionary.

        Args:
            dict_in (dict): Dictionary with the date-time information.
                The dictionary should contain the following key-value pairs:
                - "time_format" (str): The date-time format, either
                                       "Gregorian_Date" or "Julian_Date"
                                       (case-insensitive).
                - "time_scale" (str): The time scale, e.g., "UTC" or "UT1"
                                      (case-insensitive).

                For "Gregorian_Date" format:
                - "year" (int): The year component of the date.
                - "month" (int): The month component of the date.
                - "day" (int): The day component of the date.
                - "hour" (int): The hour component of the time.
                - "minute" (int): The minute component of the time.
                - "second" (float): The second component of the time.

                For "Julian_Date" format:
                - "jd" (float): The Julian Date.

        Returns:
            AbsoluteDate: AbsoluteDate object.
        """
        time_scale: TimeScale = TimeScale.get(dict_in["time_scale"])
        time_format: TimeFormat = TimeFormat.get(dict_in["time_format"])
        if time_format == TimeFormat.GREGORIAN_DATE:
            year: int = dict_in["year"]
            month: int = dict_in["month"]
            day: int = dict_in["day"]
            hour: int = dict_in["hour"]
            minute: int = dict_in["minute"]
            second: float = dict_in["second"]
            astropy_time = Astropy_Time(
                f"{year}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:06.3f}",  # pylint: disable=line-too-long
                format="isot",
                scale=time_scale.value.lower(),
            )
        elif time_format == TimeFormat.JULIAN_DATE:
            jd: float = dict_in["jd"]
            astropy_time = Astropy_Time(
                jd, format="jd", scale=time_scale.value.lower()
            )
        else:
            raise ValueError(f"Unsupported date-time format: {time_format}")
        return AbsoluteDate(astropy_time=astropy_time)

    def to_dict(
        self, time_format_str: str = "GREGORIAN_DATE"
    ) -> Dict[str, Any]:
        """Convert the AbsoluteDate object to a dictionary.

        Args:
            time_format (str): The type of date-time format to use
                                ("GREGORIAN_DATE" or "JULIAN_DATE").

        Returns:
            dict: Dictionary with the date-time information.
        """
        time_format = TimeFormat.get(time_format_str)
        if time_format == TimeFormat.GREGORIAN_DATE:
            return {
                "time_format": "GREGORIAN_DATE",
                "year": self.astropy_time.datetime.year,
                "month": self.astropy_time.datetime.month,
                "day": self.astropy_time.datetime.day,
                "hour": self.astropy_time.datetime.hour,
                "minute": self.astropy_time.datetime.minute,
                "second": self.astropy_time.datetime.second,
                "time_scale": self.astropy_time.scale,
            }
        elif time_format == TimeFormat.JULIAN_DATE:
            return {
                "time_format": "JULIAN_DATE",
                "jd": self.astropy_time.jd,
                "time_scale": self.astropy_time.scale,
            }
        else:
            raise ValueError(f"Unsupported date-time format: {time_format}")


class ReferenceFrame(EnumBase):
    """
    Enumeration of recognized Reference frames.
    """

    ICRF = "ICRF"  # International Celestial Reference Frame
    ITRF = "ITRF"  # International Terrestrial Reference Frame
    # TEME = "TEME"  # True Equator Mean Equinox


class Cartesian3DPosition:
    """Handles 3D position information."""

    def __init__(
        self, x: float, y: float, z: float, frame: Optional[ReferenceFrame]
    ) -> None:
        if not all(isinstance(coord, (int, float)) for coord in [x, y, z]):
            raise ValueError("x, y, and z must be numeric values.")
        if frame is not None and not isinstance(frame, ReferenceFrame):
            raise ValueError("frame must be a ReferenceFrame object or None.")
        self.x = x
        self.y = y
        self.z = z
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
        return [self.x, self.y, self.z]

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
        x = dict_in["x"]
        y = dict_in["y"]
        z = dict_in["z"]
        frame = ReferenceFrame.get(dict_in["frame"])
        return Cartesian3DPosition(x, y, z, frame)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the Cartesian3DPosition object to a dictionary.

        Returns:
            dict: Dictionary with the position information.
        """
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
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
        self.vx = vx
        self.vy = vy
        self.vz = vz
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
        return [self.vx, self.vy, self.vz]

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
        vx = dict_in["vx"]
        vy = dict_in["vy"]
        vz = dict_in["vz"]
        frame = ReferenceFrame.get(dict_in["frame"])
        return Cartesian3DVelocity(vx, vy, vz, frame)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the Cartesian3DVelocity object to a dictionary.

        Returns:
            dict: Dictionary with the velocity information.
        """
        return {
            "vx": self.vx,
            "vy": self.vy,
            "vz": self.vz,
            "frame": self.frame.value if self.frame else None,
        }


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
