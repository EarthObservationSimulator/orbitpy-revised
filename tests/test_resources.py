"""Unit tests for orbitpy.resources module."""

import unittest
import random
import uuid

from eosimutils.state import GeographicPosition

from orbitpy.resources import GroundStation


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
        self.assertTrue(
            "identifier must be a valid UUID." in str(context.exception)
        )


if __name__ == "__main__":
    unittest.main()
