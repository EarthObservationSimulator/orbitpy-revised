"""Unit tests for orbitpy.util module."""

import unittest
from orbitpy.util import AbsoluteDate
from orbitpy.util import (
    Cartesian3DPosition,
    Cartesian3DVelocity,
    CartesianState,
    ReferenceFrame,
)
from astropy.time import Time as Astropy_Time
import random


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


class TestCartesian3DPosition(unittest.TestCase):
    """Test the Cartesian3DPosition class."""
    def setUp(self):
        self.x = round(random.uniform(-1e6, 1e6), 6)
        self.y = round(random.uniform(-1e6, 1e6), 6)
        self.z = round(random.uniform(-1e6, 1e6), 6)

    def test_initialization(self):
        pos = Cartesian3DPosition(self.x, self.y, self.z, ReferenceFrame.ICRF)
        self.assertEqual(pos.x, self.x)
        self.assertEqual(pos.y, self.y)
        self.assertEqual(pos.z, self.z)
        self.assertEqual(pos.frame, ReferenceFrame.ICRF)

    def test_from_list(self):
        pos = Cartesian3DPosition.from_list(
            [self.x, self.y, self.z], ReferenceFrame.ITRF
        )
        self.assertEqual(pos.x, self.x)
        self.assertEqual(pos.y, self.y)
        self.assertEqual(pos.z, self.z)
        self.assertEqual(pos.frame, ReferenceFrame.ITRF)

    def test_to_list(self):
        pos = Cartesian3DPosition(self.x, self.y, self.z, ReferenceFrame.ICRF)
        self.assertEqual(pos.to_list(), [self.x, self.y, self.z])

    def test_from_dict(self):
        dict_in = {"x": self.x, "y": self.y, "z": self.z, "frame": "ICRF"}
        pos = Cartesian3DPosition.from_dict(dict_in)
        self.assertEqual(pos.x, self.x)
        self.assertEqual(pos.y, self.y)
        self.assertEqual(pos.z, self.z)
        self.assertEqual(pos.frame, ReferenceFrame.ICRF)

    def test_to_dict(self):
        pos = Cartesian3DPosition(self.x, self.y, self.z, ReferenceFrame.ITRF)
        dict_out = pos.to_dict()
        self.assertEqual(dict_out["x"], self.x)
        self.assertEqual(dict_out["y"], self.y)
        self.assertEqual(dict_out["z"], self.z)
        self.assertEqual(dict_out["frame"], "ITRF")


class TestCartesian3DVelocity(unittest.TestCase):
    """Test the Cartesian3DVelocity class."""
    def setUp(self):
        self.vx = round(random.uniform(-1e6, 1e6), 6)
        self.vy = round(random.uniform(-1e6, 1e6), 6)
        self.vz = round(random.uniform(-1e6, 1e6), 6)

    def test_initialization(self):
        vel = Cartesian3DVelocity(
            self.vx, self.vy, self.vz, ReferenceFrame.ICRF
        )
        self.assertEqual(vel.vx, self.vx)
        self.assertEqual(vel.vy, self.vy)
        self.assertEqual(vel.vz, self.vz)
        self.assertEqual(vel.frame, ReferenceFrame.ICRF)

    def test_from_list(self):
        vel = Cartesian3DVelocity.from_list(
            [self.vx, self.vy, self.vz], ReferenceFrame.ITRF
        )
        self.assertEqual(vel.vx, self.vx)
        self.assertEqual(vel.vy, self.vy)
        self.assertEqual(vel.vz, self.vz)
        self.assertEqual(vel.frame, ReferenceFrame.ITRF)

    def test_to_list(self):
        vel = Cartesian3DVelocity(
            self.vx, self.vy, self.vz, ReferenceFrame.ICRF
        )
        self.assertEqual(vel.to_list(), [self.vx, self.vy, self.vz])

    def test_from_dict(self):
        dict_in = {"vx": self.vx, "vy": self.vy, "vz": self.vz, "frame": "ICRF"}
        vel = Cartesian3DVelocity.from_dict(dict_in)
        self.assertEqual(vel.vx, self.vx)
        self.assertEqual(vel.vy, self.vy)
        self.assertEqual(vel.vz, self.vz)
        self.assertEqual(vel.frame, ReferenceFrame.ICRF)

    def test_to_dict(self):
        vel = Cartesian3DVelocity(
            self.vx, self.vy, self.vz, ReferenceFrame.ITRF
        )
        dict_out = vel.to_dict()
        self.assertEqual(dict_out["vx"], self.vx)
        self.assertEqual(dict_out["vy"], self.vy)
        self.assertEqual(dict_out["vz"], self.vz)
        self.assertEqual(dict_out["frame"], "ITRF")


class TestCartesianState(unittest.TestCase):
    """Test the CartesianState class."""
    def setUp(self):
        self.time_dict = {
            "time_format": "Gregorian_Date",
            "year": 2025,
            "month": 3,
            "day": 10,
            "hour": 14,
            "minute": 30,
            "second": 0.0,
            "time_scale": "utc",
        }
        self.time = AbsoluteDate.from_dict(self.time_dict)

        self.position_dict = {
            "x": round(random.uniform(-1e6, 1e6), 6),
            "y": round(random.uniform(-1e6, 1e6), 6),
            "z": round(random.uniform(-1e6, 1e6), 6),
            "frame": "ICRF",
        }
        self.position = Cartesian3DPosition.from_dict(self.position_dict)

        self.velocity_dict = {
            "vx": round(random.uniform(-1e6, 1e6), 6),
            "vy": round(random.uniform(-1e6, 1e6), 6),
            "vz": round(random.uniform(-1e6, 1e6), 6),
            "frame": "ICRF",
        }
        self.velocity = Cartesian3DVelocity.from_dict(self.velocity_dict)

        self.frame = ReferenceFrame.ICRF

    def test_initialization(self):
        state = CartesianState(
            self.time, self.position, self.velocity, self.frame
        )
        self.assertEqual(state.time, self.time)
        self.assertEqual(state.position, self.position)
        self.assertEqual(state.velocity, self.velocity)
        self.assertEqual(state.frame, self.frame)

    def test_mismatched_frames(self):
        position_dict = {
            "x": round(random.uniform(-1e6, 1e6), 6),
            "y": round(random.uniform(-1e6, 1e6), 6),
            "z": round(random.uniform(-1e6, 1e6), 6),
            "frame": "ICRF",
        }
        position = Cartesian3DPosition.from_dict(position_dict)

        velocity_dict = {
            "vx": round(random.uniform(-1e6, 1e6), 6),
            "vy": round(random.uniform(-1e6, 1e6), 6),
            "vz": round(random.uniform(-1e6, 1e6), 6),
            "frame": "ITRF",
        }
        velocity = Cartesian3DVelocity.from_dict(velocity_dict)

        with self.assertRaises(ValueError) as context:
            CartesianState(self.time, position, velocity, ReferenceFrame.ICRF)

        self.assertTrue(
            "Velocity frame does not match the provided frame."
            in str(context.exception)
        )

    def test_from_dict(self):
        dict_in = {
            "time": self.time_dict,
            "position": self.position.to_list(),
            "velocity": self.velocity.to_list(),
            "frame": "ICRF",
        }
        state = CartesianState.from_dict(dict_in)
        self.assertEqual(
            state.time.astropy_time.iso, self.time.astropy_time.iso
        )
        self.assertEqual(state.position.x, self.position.x)
        self.assertEqual(state.position.y, self.position.y)
        self.assertEqual(state.position.z, self.position.z)
        self.assertEqual(state.velocity.vx, self.velocity.vx)
        self.assertEqual(state.velocity.vy, self.velocity.vy)
        self.assertEqual(state.velocity.vz, self.velocity.vz)
        self.assertEqual(state.frame, self.frame)

    def test_to_dict(self):
        state = CartesianState(
            self.time, self.position, self.velocity, self.frame
        )
        dict_out = state.to_dict()
        self.assertEqual(dict_out["time"], self.time.to_dict())
        self.assertEqual(dict_out["position"], self.position.to_list())
        self.assertEqual(dict_out["velocity"], self.velocity.to_list())
        self.assertEqual(dict_out["frame"], "ICRF")


if __name__ == "__main__":
    unittest.main()
