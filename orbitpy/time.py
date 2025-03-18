"""
.. module:: orbitpy.time
   :synopsis: Time information.

Collection of classes and functions for handling time information.
"""

from typing import Dict, Any, List, Optional

from astropy.time import Time as Astropy_Time
from skyfield.api import wgs84 as skyfield_wgs84

from .base import EnumBase, ReferenceFrame


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

    def __eq__(self, other: object) -> bool:
        """
        Check if two AbsoluteDate objects are equal.

        Args:
            other (object): The object to compare with.

        Returns:
            bool: True if the two AbsoluteDate objects represent the 
                  same date and time, False otherwise.
        """
        if not isinstance(other, AbsoluteDate):
            return False
        return self.astropy_time == other.astropy_time
