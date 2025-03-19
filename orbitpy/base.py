"""
.. module:: orbitpy.base
   :synopsis: Collection of basic utility classes and functions.

Collection of basic utility classes and functions.
"""

from enum import Enum


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


class ReferenceFrame(EnumBase):
    """
    Enumeration of recognized Reference frames.

    Attributes:
        GCRF (str): Geocentric Celestial Reference Frame. See:
                    https://rhodesmill.org/skyfield/api-position.html#geocentric-position-relative-to-the-earth

        ITRF (str): International Terrestrial Reference Frame. See:
                    https://rhodesmill.org/skyfield/api-framelib.html#skyfield.framelib.itrs

    """

    GCRF = "GCRF"  # Geocentric Celestial Reference Frame
    ITRF = "ITRF"  # International Terrestrial Reference Frame
    # TEME = "TEME"  # True Equator Mean Equinox
