"""
.. module:: orbitpy.util
   :synopsis: Collection of utility classes and functions.

Collection of utility classes and functions used by the `orbitpy` package.
Note some utility classes/functions are imported from the `instrupy` package.
"""

from typing import Dict, Any
from astropy.time import Time as Astropy_Time


class AbsoluteDate:
    """Handles date-time information with support to Julian and Gregorian
    date-time formats and UT1 and UTC time scales."""

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
                                       "GregorianDate" or "JulianDate" 
                                       (case-insensitive).
                - "time_scale" (str): The time scale, e.g., "utc" or "ut1"
                                      (case-insensitive).

                For "GregorianDate" format:
                - "year" (int): The year component of the date.
                - "month" (int): The month component of the date.
                - "day" (int): The day component of the date.
                - "hour" (int): The hour component of the time.
                - "minute" (int): The minute component of the time.
                - "second" (float): The second component of the time.

                For "JulianDate" format:
                - "jd" (float): The Julian Date.

        Returns:
            AbsoluteDate: AbsoluteDate object.
        """
        time_scale: str = dict_in["time_scale"].lower()
        time_format: str = dict_in["time_format"].lower()
        if time_format == "gregoriandate":
            year: int = dict_in["year"]
            month: int = dict_in["month"]
            day: int = dict_in["day"]
            hour: int = dict_in["hour"]
            minute: int = dict_in["minute"]
            second: float = dict_in["second"]
            astropy_time = Astropy_Time(
                f"{year}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:06.3f}",  # pylint: disable=line-too-long
                format="isot",
                scale=time_scale,
            )
        elif time_format == "juliandate":
            jd: float = dict_in["jd"]
            astropy_time = Astropy_Time(jd, format="jd", scale=time_scale)
        else:
            raise ValueError(f"Unsupported date-time format: {time_format}")
        return AbsoluteDate(astropy_time=astropy_time)

    def to_dict(self, time_format: str = "GregorianDate") -> Dict[str, Any]:
        """Convert the AbsoluteDate object to a dictionary.

        Args:
            time_format (str): The type of date-time format to use
                                ("GregorianDate" or "JulianDate").

        Returns:
            dict: Dictionary with the date-time information.
        """
        time_format = time_format.lower()
        if time_format == "gregoriandate":
            return {
                "time_format": "GregorianDate",
                "year": self.astropy_time.datetime.year,
                "month": self.astropy_time.datetime.month,
                "day": self.astropy_time.datetime.day,
                "hour": self.astropy_time.datetime.hour,
                "minute": self.astropy_time.datetime.minute,
                "second": self.astropy_time.datetime.second,
                "time_scale": self.astropy_time.scale,
            }
        elif time_format == "juliandate":
            return {
                "time_format": "JulianDate",
                "jd": self.astropy_time.jd,
                "time_scale": self.astropy_time.scale,
            }
        else:
            raise ValueError(f"Unsupported date-time format: {time_format}")
