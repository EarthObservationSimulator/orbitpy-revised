"""Unit tests for orbitpy.position module."""

import unittest
import random
import numpy as np

from astropy.coordinates import EarthLocation as Astropy_EarthLocation
import astropy.units as astropy_u

from orbitpy.position import (
    Cartesian3DPosition,
    Cartesian3DVelocity,
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
        np.testing.assert_array_equal(pos.coords, [self.x, self.y, self.z])
        self.assertEqual(pos.frame, ReferenceFrame.GCRF)

    def test_from_list(self):
        pos = Cartesian3DPosition.from_list(
            [self.x, self.y, self.z], ReferenceFrame.ITRF
        )
        np.testing.assert_array_equal(pos.coords, [self.x, self.y, self.z])
        self.assertEqual(pos.frame, ReferenceFrame.ITRF)

    def test_to_list(self):
        pos = Cartesian3DPosition(self.x, self.y, self.z, ReferenceFrame.GCRF)
        self.assertEqual(pos.to_list(), [self.x, self.y, self.z])

    def test_from_dict(self):
        dict_in = {"x": self.x, "y": self.y, "z": self.z, "frame": "GCRF"}
        pos = Cartesian3DPosition.from_dict(dict_in)
        np.testing.assert_array_equal(pos.coords, [self.x, self.y, self.z])
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
        np.testing.assert_array_equal(vel.coords, [self.vx, self.vy, self.vz])
        self.assertEqual(vel.frame, ReferenceFrame.GCRF)

    def test_from_list(self):
        vel = Cartesian3DVelocity.from_list(
            [self.vx, self.vy, self.vz], ReferenceFrame.ITRF
        )
        np.testing.assert_array_equal(vel.coords, [self.vx, self.vy, self.vz])
        self.assertEqual(vel.frame, ReferenceFrame.ITRF)

    def test_to_list(self):
        vel = Cartesian3DVelocity(
            self.vx, self.vy, self.vz, ReferenceFrame.GCRF
        )
        self.assertEqual(vel.to_list(), [self.vx, self.vy, self.vz])

    def test_from_dict(self):
        dict_in = {"vx": self.vx, "vy": self.vy, "vz": self.vz, "frame": "GCRF"}
        vel = Cartesian3DVelocity.from_dict(dict_in)
        np.testing.assert_array_equal(vel.coords, [self.vx, self.vy, self.vz])
        self.assertEqual(vel.frame, ReferenceFrame.GCRF)

    def test_to_dict(self):
        vel = Cartesian3DVelocity(
            self.vx, self.vy, self.vz, ReferenceFrame.ITRF
        )
        dict_out = vel.to_dict()
        np.testing.assert_array_equal(vel.coords, [self.vx, self.vy, self.vz])
        self.assertEqual(dict_out["frame"], "ITRF")


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
