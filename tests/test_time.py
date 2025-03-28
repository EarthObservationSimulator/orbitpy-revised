"""Unit tests for orbitpy.time module."""

import unittest

from astropy.time import Time as Astropy_Time

from orbitpy.time import AbsoluteDate


class TestAbsoluteDate(unittest.TestCase):
    """Test the AbsoluteDate class."""

    def test_from_dict_gregorian(self):
        dict_in = {
            "time_format": "Gregorian_Date",
            "year": 2025,
            "month": 3,
            "day": 10,
            "hour": 14,
            "minute": 30,
            "second": 0.0,
            "time_scale": "utc",
        }
        absolute_date = AbsoluteDate.from_dict(dict_in)
        self.assertEqual(
            absolute_date.astropy_time.iso, "2025-03-10 14:30:00.000"
        )

    def test_from_dict_julian(self):
        dict_in = {
            "time_format": "Julian_Date",
            "jd": 2457081.10417,
            "time_scale": "utc",
        }
        absolute_date = AbsoluteDate.from_dict(dict_in)
        self.assertAlmostEqual(
            absolute_date.astropy_time.jd, 2457081.10417, places=5
        )

    def test_to_dict_gregorian(self):
        astropy_time = Astropy_Time(
            "2025-03-10T14:30:00", format="isot", scale="utc"
        )
        absolute_date = AbsoluteDate(astropy_time)
        dict_out = absolute_date.to_dict("Gregorian_Date")
        expected_dict = {
            "time_format": "GREGORIAN_DATE",
            "year": 2025,
            "month": 3,
            "day": 10,
            "hour": 14,
            "minute": 30,
            "second": 0,
            "time_scale": "utc",
        }
        self.assertEqual(dict_out, expected_dict)

    def test_to_dict_julian(self):
        astropy_time = Astropy_Time(2457081.10417, format="jd", scale="utc")
        absolute_date = AbsoluteDate(astropy_time)
        dict_out = absolute_date.to_dict("Julian_Date")
        expected_dict = {
            "time_format": "JULIAN_DATE",
            "jd": 2457081.10417,
            "time_scale": "utc",
        }
        self.assertEqual(dict_out, expected_dict)

    def test_gregorian_to_julian(self):
        # Initialize with Gregorian date
        dict_in = {
            "time_format": "GREGORIAN_DATE",
            "year": 2025,
            "month": 3,
            "day": 11,
            "hour": 1,
            "minute": 23,
            "second": 37.0,
            "time_scale": "ut1",
        }
        absolute_date = AbsoluteDate.from_dict(dict_in)
        # Validation data from: https://aa.usno.navy.mil/data/JulianDate
        self.assertAlmostEqual(
            absolute_date.astropy_time.jd, 2460745.558067, places=5
        )

    def test_julian_to_gregorian(self):
        # Initialize with Julian Date
        dict_in = {
            "time_format": "Julian_Date",
            "jd": 2460325.145250,
            "time_scale": "ut1",
        }
        absolute_date = AbsoluteDate.from_dict(dict_in)
        # Validation data from: https://aa.usno.navy.mil/data/JulianDate
        self.assertEqual(
            absolute_date.astropy_time.iso, "2024-01-15 15:29:09.600"
        )

    def test_equality_operator(self):
        date1 = AbsoluteDate(
            Astropy_Time("2025-03-17T12:00:00", format="isot", scale="utc")
        )
        date2 = AbsoluteDate(
            Astropy_Time("2025-03-17T12:00:00", format="isot", scale="utc")
        )
        date3 = AbsoluteDate(
            Astropy_Time("2025-03-18T12:00:00", format="isot", scale="utc")
        )

        # Test equality
        self.assertTrue(date1 == date2)
        self.assertFalse(date1 == date3)

        # Test comparison with a non-AbsoluteDate object
        self.assertFalse(date1 == "not an AbsoluteDate")

    def test_to_astropy_time(self):
        astropy_time = Astropy_Time(
            "2025-03-17T12:00:00", format="isot", scale="utc"
        )
        absolute_date = AbsoluteDate(astropy_time)

        # Convert to Astropy Time
        converted_time = absolute_date.to_astropy_time()

        # Assert that the converted time matches the original Astropy Time
        self.assertEqual(converted_time, astropy_time)

    def test_to_skyfield_time(self):
        absolute_date = AbsoluteDate(
            Astropy_Time("2025-03-17T12:00:00", format="isot", scale="utc")
        )
        # Convert to Skyfield Time
        skyfield_time = absolute_date.to_skyfield_time()

        # Assert that the Skyfield time matches the AbsoluteTime in UTC
        self.assertEqual(
            skyfield_time.utc_strftime(format="%Y-%m-%d %H:%M:%S UTC"),
            "2025-03-17 12:00:00 UTC",
        )


if __name__ == "__main__":
    unittest.main()
