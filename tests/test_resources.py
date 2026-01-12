"""Unit tests for orbitpy.resources module."""

import unittest
import random
import uuid

from eosimutils.state import GeographicPosition
from eosimutils.fieldofview import FieldOfViewFactory, CircularFieldOfView
from eosimutils.standardframes import LVLHType1FrameHandler

from orbitpy.orbits import TwoLineElementSet
from orbitpy.resources import GroundStation, Sensor, Spacecraft


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
        dict_out: dict = gs.to_dict()
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


class TestSensor(unittest.TestCase):
    """Unit tests for the Sensor class."""

    def setUp(self):
        """Set up test data for Sensor."""
        self.identifier = "865ddce7-4ade-44d9-8c22-2b1e6f65e830"
        self.name = "Test Sensor"
        self.fov_specs = {
            "fov_type": "CIRCULAR",
            "diameter": 60.0,
            "frame": "ICRF_EC",
            "boresight": [0.0, 0.0, 1.0],
        }
        self.fov = FieldOfViewFactory.from_dict(self.fov_specs)
        self.sensor_dict = {
            "id": "865ddce7-4ade-44d9-8c22-2b1e6f65e830",
            "name": "Test Sensor",
            "fov": {
                "fov_type": "CIRCULAR",
                "diameter": 60.0,
                "frame": "ICRF_EC",
                "boresight": [0.0, 0.0, 1.0],
            },
        }

    def test_initialization(self):
        """Test initialization of Sensor."""
        sensor = Sensor(self.identifier, self.name, self.fov)
        self.assertEqual(sensor.identifier, self.identifier)
        self.assertEqual(sensor.name, self.name)
        self.assertEqual(sensor.fov.diameter, self.fov.diameter)
        self.assertEqual(sensor.fov.frame, self.fov.frame)
        self.assertTrue((sensor.fov.boresight == self.fov.boresight).all())

    def test_from_dict(self):
        """Test creating a Sensor object from a dictionary."""
        sensor = Sensor.from_dict(self.sensor_dict)
        self.assertEqual(sensor.identifier, self.identifier)
        self.assertEqual(sensor.name, self.name)
        self.assertEqual(sensor.fov.diameter, self.fov.diameter)
        self.assertEqual(sensor.fov.frame, self.fov.frame)
        self.assertTrue((sensor.fov.boresight == self.fov.boresight).all())

    def test_to_dict(self):
        """Test converting a Sensor object to a dictionary."""
        sensor = Sensor(self.identifier, self.name, self.fov)
        dict_out = Sensor.to_dict(sensor)
        self.assertEqual(dict_out["id"], self.identifier)
        self.assertEqual(dict_out["name"], self.name)
        self.assertEqual(
            dict_out["fov"]["diameter"], self.fov_specs["diameter"]
        )
        self.assertEqual(dict_out["fov"]["frame"], self.fov_specs["frame"])
        self.assertEqual(
            dict_out["fov"]["boresight"], self.fov_specs["boresight"]
        )

    def test_default_id(self):
        """Test that a default UUID is generated if no identifier is provided."""
        sensor = Sensor(None, self.name, self.fov)
        self.assertIsNotNone(sensor.identifier)
        try:
            uuid.UUID(sensor.identifier)
        except ValueError:
            self.fail("Generated identifier is not a valid UUID.")

    def test_invalid_identifier(self):
        """Test that an invalid UUID raises a ValueError."""
        invalid_identifier = "invalid-uuid"
        with self.assertRaises(ValueError) as context:
            Sensor(invalid_identifier, self.name, self.fov)
        self.assertIn(
            "identifier must be a valid UUID.", str(context.exception)
        )


class TestSpacecraft(unittest.TestCase):
    """Unit tests for the Spacecraft class."""

    def setUp(self):
        """Set up test data for Spacecraft."""
        self.identifier = "ddd716b0-443b-4141-a413-19b14260db9a"
        self.name = "Test Spacecraft"
        self.norad_id = 49260
        self.orbit = TwoLineElementSet(
            line0="0 LANDSAT 9",
            line1="1 49260U 21088A   25106.07240456  .00000957  00000-0  22241-3 0  9997",
            line2="2 49260  98.1921 177.4890 0001161  87.5064 272.6267 14.57121096188801",
        )
        self.sensor_fov = CircularFieldOfView(
            diameter=60.0, frame="ICRF_EC", boresight=[0.0, 0.0, 1.0]
        )
        self.sensor = Sensor(
            identifier="699881aa-5e50-4186-ba03-1eafaa4e6f62",
            name="Test Sensor",
            fov=self.sensor_fov,
        )
        self.local_orbital_frame_handler = LVLHType1FrameHandler(
            "Test_LVLH_Frame"
        )

        self.spacecraft_dict = {
            "id": "ddd716b0-443b-4141-a413-19b14260db9a",
            "name": "Test Spacecraft",
            "norad_id": 49260,
            "orbit": {
                "orbit_type": "TWO_LINE_ELEMENT_SET",
                "TLE_LINE0": "0 LANDSAT 9",
                "TLE_LINE1": "1 49260U 21088A   25106.07240456  .00000957  00000-0  22241-3 0  9997",  # pylint: disable=line-too-long
                "TLE_LINE2": "2 49260  98.1921 177.4890 0001161  87.5064 272.6267 14.57121096188801",  # pylint: disable=line-too-long
            },
            "local_orbital_frame_handler": {
                "frame_type": "LVLH_TYPE_1",
                "name": "Test_LVLH_Frame",
            },
            "sensor": [
                {
                    "id": "699881aa-5e50-4186-ba03-1eafaa4e6f62",
                    "name": "Test Sensor",
                    "fov": {
                        "fov_type": "CIRCULAR",
                        "diameter": 60.0,
                        "frame": "ICRF_EC",
                        "boresight": [0.0, 0.0, 1.0],
                    },
                }
            ],
        }

    def test_initialization(self):
        """Test initialization of Spacecraft."""
        spc = Spacecraft(
            self.identifier,
            self.name,
            self.norad_id,
            self.orbit,
            self.local_orbital_frame_handler,
            [self.sensor],
        )
        self.assertEqual(spc.identifier, self.identifier)
        self.assertEqual(spc.name, self.name)
        self.assertEqual(spc.norad_id, self.norad_id)
        self.assertEqual(spc.orbit.line0, self.orbit.line0)
        self.assertEqual(spc.orbit.line1, self.orbit.line1)
        self.assertEqual(spc.orbit.line2, self.orbit.line2)
        self.assertEqual(len(spc.sensor), 1)
        self.assertEqual(spc.sensor[0].name, self.sensor.name)
        self.assertEqual(
            spc.local_orbital_frame_handler,
            self.local_orbital_frame_handler,
        )

    def test_from_dict_single_sensor(self):
        """Test creating a Spacecraft object from a dictionary."""
        spc = Spacecraft.from_dict(self.spacecraft_dict)
        self.assertEqual(spc.identifier, self.identifier)
        self.assertEqual(spc.name, self.name)
        self.assertEqual(spc.norad_id, self.norad_id)
        self.assertEqual(spc.orbit.line0, self.orbit.line0)
        self.assertEqual(spc.orbit.line1, self.orbit.line1)
        self.assertEqual(spc.orbit.line2, self.orbit.line2)
        self.assertEqual(len(spc.sensor), 1)
        self.assertEqual(spc.sensor[0].name, self.sensor.name)
        self.assertEqual(
            spc.local_orbital_frame_handler,
            self.local_orbital_frame_handler,
        )

    def test_from_dict_no_sensor(self):
        """Test creating a Spacecraft object from a dictionary without sensor."""
        dict_no_sensor = self.spacecraft_dict.copy()
        dict_no_sensor.pop("sensor")
        spc = Spacecraft.from_dict(dict_no_sensor)
        self.assertEqual(spc.identifier, self.identifier)
        self.assertEqual(spc.name, self.name)
        self.assertEqual(spc.norad_id, self.norad_id)
        self.assertEqual(spc.orbit.line0, self.orbit.line0)
        self.assertEqual(spc.orbit.line1, self.orbit.line1)
        self.assertEqual(spc.orbit.line2, self.orbit.line2)
        self.assertEqual(
            spc.local_orbital_frame_handler,
            self.local_orbital_frame_handler,
        )
        self.assertIsNone(spc.sensor)

    def test_from_dict_multiple_sensors(self):
        """Test creating a Spacecraft object from a dictionary with multiple sensors."""
        sensor2_dict = {
            "id": "3aecaa97-048a-4dd6-8b23-88342520836f",
            "name": "Test Sensor",
            "fov": {
                "fov_type": "RECTANGULAR",
                "frame": "ICRF_EC",
                "boresight": [0.0, 0.0, 1.0],
                "cross_angle": 90.0,
                "ref_angle": 30.0,
            },
        }
        spacecraft_dict_multiple_sensors = self.spacecraft_dict.copy()
        spacecraft_dict_multiple_sensors["sensor"] = [
            self.spacecraft_dict["sensor"][0],
            sensor2_dict,
        ]

        spc = Spacecraft.from_dict(spacecraft_dict_multiple_sensors)
        self.assertEqual(spc.identifier, self.identifier)
        self.assertEqual(spc.name, self.name)
        self.assertEqual(spc.norad_id, self.norad_id)
        self.assertEqual(spc.orbit.line0, self.orbit.line0)
        self.assertEqual(spc.orbit.line1, self.orbit.line1)
        self.assertEqual(spc.orbit.line2, self.orbit.line2)
        self.assertEqual(len(spc.sensor), 2)
        self.assertEqual(spc.sensor[0].name, self.sensor.name)
        self.assertEqual(spc.sensor[1].name, self.sensor.name)
        self.assertEqual(
            spc.local_orbital_frame_handler,
            self.local_orbital_frame_handler,
        )

    def test_to_dict(self):
        """Test converting a Spacecraft object to a dictionary."""
        spc = Spacecraft(
            self.identifier,
            self.name,
            self.norad_id,
            self.orbit,
            self.local_orbital_frame_handler,
            [self.sensor],
        )
        dict_out = spc.to_dict()
        self.assertEqual(dict_out["id"], self.identifier)
        self.assertEqual(dict_out["name"], self.name)
        self.assertEqual(dict_out["norad_id"], self.norad_id)
        orbit_dict = dict_out["orbit"]
        self.assertIsInstance(
            orbit_dict, dict, "Expected 'orbit' to be a dictionary."
        )
        self.assertEqual(orbit_dict.get("TLE_LINE0"), self.orbit.line0)
        self.assertEqual(orbit_dict.get("TLE_LINE1"), self.orbit.line1)
        self.assertEqual(orbit_dict.get("TLE_LINE2"), self.orbit.line2)
        local_orbital_frame_handler_dict = dict_out[
            "local_orbital_frame_handler"
        ]
        self.assertIsInstance(local_orbital_frame_handler_dict, dict)
        self.assertEqual(
            local_orbital_frame_handler_dict.get("frame_type").upper(),
            "LVLH_TYPE_1",
        )
        self.assertEqual(
            local_orbital_frame_handler_dict.get("name"),
            self.local_orbital_frame_handler.frame.to_string(),
        )
        sensors = dict_out.get("sensor")
        self.assertIsInstance(sensors, list)
        self.assertEqual(len(dict_out["sensor"]), 1)
        sensor0_dict: dict = sensors[0]
        self.assertIsInstance(sensor0_dict, dict)
        self.assertEqual(sensor0_dict.get("name"), self.sensor.name)

    def test_default_id(self):
        """Test that a default UUID is generated if no identifier is provided."""
        spc = Spacecraft(
            None,
            self.name,
            self.norad_id,
            self.orbit,
            self.local_orbital_frame_handler,
            [self.sensor],
        )
        self.assertIsNotNone(spc.identifier)
        try:
            uuid.UUID(spc.identifier)
        except ValueError:
            self.fail("Generated identifier is not a valid UUID.")

    def test_invalid_identifier(self):
        """Test that an invalid UUID raises a ValueError."""
        invalid_identifier = "invalid-uuid"
        with self.assertRaises(ValueError) as context:
            Spacecraft(
                invalid_identifier,
                self.name,
                self.norad_id,
                self.orbit,
                self.local_orbital_frame_handler,
                [self.sensor],
            )
        self.assertIn(
            "identifier must be a valid UUID.", str(context.exception)
        )

    def test_invalid_sensor_list(self):
        """Test that an invalid sensor list raises a TypeError."""
        with self.assertRaises(TypeError) as context:
            Spacecraft(
                self.identifier,
                self.name,
                self.norad_id,
                self.orbit,
                self.local_orbital_frame_handler,
                ["invalid_sensor"],
            )
        self.assertIn(
            "sensor must be a list of Sensor objects.", str(context.exception)
        )

    def test_invalid_orbit_type(self):
        """Test that an invalid orbit type raises a TypeError."""
        with self.assertRaises(TypeError) as context:
            Spacecraft(
                self.identifier,
                self.name,
                self.norad_id,
                "invalid_orbit",
                self.local_orbital_frame_handler,
                [self.sensor],
            )
        self.assertIn(
            "orbit must be a TwoLineElementSet, OrbitalMeanElementsMessage, "
            "or OsculatingElements object.",
            str(context.exception),
        )

    def test_invalid_frame_handler_type(self):
        """Test that an invalid frame handler type raises a TypeError."""
        with self.assertRaises(TypeError) as context:
            Spacecraft(
                self.identifier,
                self.name,
                self.norad_id,
                self.orbit,
                "invalid_frame_handler",
                [self.sensor],
            )
        self.assertIn(
            "local_orbital_frame_handler must be a frame handler object.",
            str(context.exception),
        )

    def test_default_local_orbital_frame_handler_initialized(self):
        """Test that a default LVLH Type-1 frame handler is created when not provided."""
        spc = Spacecraft(
            self.identifier,
            self.name,
            self.norad_id,
            self.orbit,
            None,  # no local_orbital_frame_handler provided
            [self.sensor],
        )
        self.assertIsNotNone(spc.local_orbital_frame_handler)
        self.assertIsInstance(
            spc.local_orbital_frame_handler, LVLHType1FrameHandler
        )
        expected_name = f"LVLH_{self.identifier}".upper()
        self.assertEqual(
            spc.local_orbital_frame_handler.frame.to_string(), expected_name
        )


if __name__ == "__main__":
    unittest.main()
