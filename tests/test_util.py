"""Unit tests for orbitpy.util module."""

import unittest
from orbitpy.util import AbsoluteDate
from orbitpy.util import (
    Cartesian3DPosition,
    Cartesian3DVelocity,
    CartesianState,
    ReferenceFrame,
    GeographicPosition,
    GroundStation,
)
from astropy.time import Time as Astropy_Time
import random
import uuid

from astropy.coordinates import EarthLocation as Astropy_EarthLocation
import astropy.units as astropy_u


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


class TestGroundStation(unittest.TestCase):
    """Test the GroundStation class."""

    def setUp(self):
        self.identifier = str(uuid.uuid4())
        self.name = "Test Station"
        self.latitude_deg = round(random.uniform(-90, 90), 6)
        self.longitude_deg = round(random.uniform(-180, 180), 6)
        self.height_m = round(random.uniform(0, 1000), 6)
        self.min_elevation_angle_deg = round(random.uniform(0, 90), 6)
        self.geographic_position = GeographicPosition(
            self.latitude_deg, self.longitude_deg, self.height_m
        )

    def test_initialization(self):
        gs = GroundStation(
            self.identifier,
            self.name,
            self.geographic_position,
            self.min_elevation_angle_deg,
        )
        self.assertEqual(gs.identifier, self.identifier)
        self.assertEqual(gs.name, self.name)
        self.assertEqual(gs.geographic_position.latitude, self.latitude_deg)
        self.assertEqual(gs.geographic_position.longitude, self.longitude_deg)
        self.assertEqual(gs.geographic_position.elevation, self.height_m)
        self.assertEqual(
            gs.min_elevation_angle_deg, self.min_elevation_angle_deg
        )

    def test_from_dict(self):
        dict_in = {
            "id": self.identifier,
            "name": self.name,
            "latitude": self.latitude_deg,
            "longitude": self.longitude_deg,
            "height": self.height_m,
            "min_elevation_angle": self.min_elevation_angle_deg,
        }
        gs = GroundStation.from_dict(dict_in)
        self.assertEqual(gs.identifier, self.identifier)
        self.assertEqual(gs.name, self.name)
        self.assertEqual(gs.geographic_position.latitude, self.latitude_deg)
        self.assertEqual(gs.geographic_position.longitude, self.longitude_deg)
        self.assertEqual(gs.geographic_position.elevation, self.height_m)
        self.assertEqual(
            gs.min_elevation_angle_deg, self.min_elevation_angle_deg
        )

    def test_to_dict(self):
        gs = GroundStation(
            self.identifier,
            self.name,
            self.geographic_position,
            self.min_elevation_angle_deg,
        )
        dict_out = gs.to_dict()
        self.assertEqual(dict_out["id"], self.identifier)
        self.assertEqual(dict_out["name"], self.name)
        self.assertEqual(dict_out["latitude"], self.latitude_deg)
        self.assertEqual(dict_out["longitude"], self.longitude_deg)
        self.assertEqual(dict_out["height"], self.height_m)
        self.assertEqual(
            dict_out["min_elevation_angle"], self.min_elevation_angle_deg
        )

    def test_default_id(self):
        gs = GroundStation(
            None,
            self.name,
            self.geographic_position,
            self.min_elevation_angle_deg,
        )
        self.assertIsNotNone(gs.identifier)
        try:
            uuid.UUID(gs.identifier)
        except ValueError:
            self.fail("id is not a valid UUID")

    def test_invalid_identifier(self):
        invalid_identifier = "invalid-uuid"
        with self.assertRaises(ValueError) as context:
            GroundStation(
                invalid_identifier,
                self.name,
                self.geographic_position,
                self.min_elevation_angle_deg,
            )
        self.assertTrue("identifier must be a valid UUID."
                        in str(context.exception))

if __name__ == "__main__":
    unittest.main()
