"""Unit tests for orbitpy.position module."""

import unittest
import random

from astropy.coordinates import EarthLocation as Astropy_EarthLocation
import astropy.units as astropy_u

from orbitpy.time import AbsoluteDate
from orbitpy.position import (
    Cartesian3DPosition,
    Cartesian3DVelocity,
    CartesianState,
    ReferenceFrame,
    GeographicPosition,
)


class TestCartesian3DPosition(unittest.TestCase):
    """Test the Cartesian3DPosition class."""

    def setUp(self):
        self.x = round(random.uniform(-1e6, 1e6), 6)
        self.y = round(random.uniform(-1e6, 1e6), 6)
        self.z = round(random.uniform(-1e6, 1e6), 6)

    def test_initialization(self):
        pos = Cartesian3DPosition(self.x, self.y, self.z, ReferenceFrame.GCRF)
        self.assertEqual(pos.x, self.x)
        self.assertEqual(pos.y, self.y)
        self.assertEqual(pos.z, self.z)
        self.assertEqual(pos.frame, ReferenceFrame.GCRF)

    def test_from_list(self):
        pos = Cartesian3DPosition.from_list(
            [self.x, self.y, self.z], ReferenceFrame.ITRF
        )
        self.assertEqual(pos.x, self.x)
        self.assertEqual(pos.y, self.y)
        self.assertEqual(pos.z, self.z)
        self.assertEqual(pos.frame, ReferenceFrame.ITRF)

    def test_to_list(self):
        pos = Cartesian3DPosition(self.x, self.y, self.z, ReferenceFrame.GCRF)
        self.assertEqual(pos.to_list(), [self.x, self.y, self.z])

    def test_from_dict(self):
        dict_in = {"x": self.x, "y": self.y, "z": self.z, "frame": "GCRF"}
        pos = Cartesian3DPosition.from_dict(dict_in)
        self.assertEqual(pos.x, self.x)
        self.assertEqual(pos.y, self.y)
        self.assertEqual(pos.z, self.z)
        self.assertEqual(pos.frame, ReferenceFrame.GCRF)

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
            self.vx, self.vy, self.vz, ReferenceFrame.GCRF
        )
        self.assertEqual(vel.vx, self.vx)
        self.assertEqual(vel.vy, self.vy)
        self.assertEqual(vel.vz, self.vz)
        self.assertEqual(vel.frame, ReferenceFrame.GCRF)

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
            self.vx, self.vy, self.vz, ReferenceFrame.GCRF
        )
        self.assertEqual(vel.to_list(), [self.vx, self.vy, self.vz])

    def test_from_dict(self):
        dict_in = {"vx": self.vx, "vy": self.vy, "vz": self.vz, "frame": "GCRF"}
        vel = Cartesian3DVelocity.from_dict(dict_in)
        self.assertEqual(vel.vx, self.vx)
        self.assertEqual(vel.vy, self.vy)
        self.assertEqual(vel.vz, self.vz)
        self.assertEqual(vel.frame, ReferenceFrame.GCRF)

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
            "frame": "GCRF",
        }
        self.position = Cartesian3DPosition.from_dict(self.position_dict)

        self.velocity_dict = {
            "vx": round(random.uniform(-1e6, 1e6), 6),
            "vy": round(random.uniform(-1e6, 1e6), 6),
            "vz": round(random.uniform(-1e6, 1e6), 6),
            "frame": "GCRF",
        }
        self.velocity = Cartesian3DVelocity.from_dict(self.velocity_dict)

        self.frame = ReferenceFrame.GCRF

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
            "frame": "GCRF",
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
            CartesianState(self.time, position, velocity, ReferenceFrame.GCRF)

        self.assertTrue(
            "Velocity frame does not match the provided frame."
            in str(context.exception)
        )

    def test_from_dict(self):
        dict_in = {
            "time": self.time_dict,
            "position": self.position.to_list(),
            "velocity": self.velocity.to_list(),
            "frame": "GCRF",
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
        self.assertEqual(dict_out["frame"], "GCRF")


class TestGeographicPosition(unittest.TestCase):
    """Test the GeographicPosition class."""

    def setUp(self):
        self.latitude_degrees = round(random.uniform(-90, 90), 6)
        self.longitude_degrees = round(random.uniform(-180, 180), 6)
        self.elevation_m = round(random.uniform(0, 10000), 6)

    def test_initialization(self):
        geo_pos = GeographicPosition(
            self.latitude_degrees, self.longitude_degrees, self.elevation_m
        )
        self.assertAlmostEqual(
            geo_pos.latitude, self.latitude_degrees, places=6
        )
        self.assertAlmostEqual(
            geo_pos.longitude, self.longitude_degrees, places=6
        )
        self.assertAlmostEqual(geo_pos.elevation, self.elevation_m, places=6)

    def test_from_dict(self):
        dict_in = {
            "latitude": self.latitude_degrees,
            "longitude": self.longitude_degrees,
            "elevation": self.elevation_m,
        }
        geo_pos = GeographicPosition.from_dict(dict_in)
        self.assertAlmostEqual(
            geo_pos.latitude, self.latitude_degrees, places=6
        )
        self.assertAlmostEqual(
            geo_pos.longitude, self.longitude_degrees, places=6
        )
        self.assertAlmostEqual(geo_pos.elevation, self.elevation_m, places=6)

    def test_to_dict(self):
        geo_pos = GeographicPosition(
            self.latitude_degrees, self.longitude_degrees, self.elevation_m
        )
        dict_out = geo_pos.to_dict()
        self.assertAlmostEqual(
            dict_out["latitude"], self.latitude_degrees, places=6
        )
        self.assertAlmostEqual(
            dict_out["longitude"], self.longitude_degrees, places=6
        )
        self.assertAlmostEqual(
            dict_out["elevation"], self.elevation_m, places=6
        )

    def test_itrs_xyz(self):
        geo_pos = GeographicPosition(37.7749, -122.4194, 10)
        itrs_xyz = geo_pos.itrs_xyz
        self.assertEqual(len(itrs_xyz), 3)
        self.assertTrue(all(isinstance(coord, float) for coord in itrs_xyz))
        # validation data from Astropy EarthLocation class
        expected_xyz = [-2706179.084e-3, -4261066.162e-3, 3885731.616e-3]
        for coord, expected in zip(itrs_xyz, expected_xyz):
            self.assertAlmostEqual(coord, expected, places=3)

    def test_itrs_xyz_astropy_validation(self):
        def geodetic_to_itrf(lat_deg: float, lon_deg: float, height_m: float):
            """
            Astropy function to convert WGS84 geodetic coordinates to
            ITRF (ECEF) Cartesian coordinates.

            Args:
                lat_deg (float): Latitude in degrees.
                lon_deg (float): Longitude in degrees.
                height_m (float): Height above WGS84 ellipsoid in meters.

            Returns:
                tuple: (x, y, z) coordinates in meters.
            """
            location = Astropy_EarthLocation.from_geodetic(
                lon=lon_deg * astropy_u.deg,
                lat=lat_deg * astropy_u.deg,
                height=height_m * astropy_u.m,
            )
            return (
                location.x.value,
                location.y.value,
                location.z.value,
            )  # Return as a tuple of floats

        geo_pos = GeographicPosition(
            self.latitude_degrees, self.longitude_degrees, self.elevation_m
        )
        itrs_xyz = geo_pos.itrs_xyz * 1e3  # convert to meters
        # validation data from Astropy EarthLocation class
        expected_xyz = geodetic_to_itrf(
            self.latitude_degrees, self.longitude_degrees, self.elevation_m
        )
        for coord, expected in zip(itrs_xyz, expected_xyz):
            self.assertAlmostEqual(coord, expected, places=3)


if __name__ == "__main__":
    unittest.main()
