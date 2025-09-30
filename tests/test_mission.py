"""Unit tests for orbitpy.orbits module."""
import unittest
import random
import numpy as np

from eosimutils.base import JsonSerializer
from eosimutils.trajectory import StateSeries
from eosimutils.state import Cartesian3DPositionArray, GeographicPosition

from orbitpy.mission import Mission
from orbitpy.resources import Spacecraft, GroundStation
from orbitpy.propagator import SGP4Propagator
from orbitpy.eclipsefinder import EclipseInfo
from orbitpy.contactfinder import ContactInfo
from orbitpy.coverage import DiscreteCoverageTP

class TestMission_1(unittest.TestCase):
    """Tests for the Mission class with single spacecraft and single ground station."""

    def setUp(self):
        
        self.start_time_dict = {
            "time_format": "Gregorian_Date",
            "calendar_date": "2025-04-17T12:00:00",
            "time_scale": "utc",
        }
        self.duration_days = 1.0  # 1 day duration
        self.sc_dict = {
                    "id": 'ddd716b0-443b-4141-a413-19b14260db9a',
                    "orbit": {
                        "orbit_type": "TWO_LINE_ELEMENT_SET",
                        "TLE_LINE0": "0 LANDSAT 9",
                        "TLE_LINE1": "1 49260U 21088A   25106.07240456  .00000957  00000-0  22241-3 0  9997",
                        "TLE_LINE2": "2 49260  98.1921 177.4890 0001161  87.5064 272.6267 14.57121096188801"
                    },
                    "sensor": [{
                        "id": '699881aa-5e50-4186-ba03-1eafaa4e6f62',
                        "fov": {
                            "fov_type": "CIRCULAR",
                            "diameter": 60.0,
                            "frame": "ICRF_EC",
                            "boresight": [0.0, 0.0, 1.0],
                        },
                    }],
                }
        self.ground_stn_dict = {
                    "id": '69e3233c-50ce-4aeb-bd40-74d74537f0ed',
                    "latitude": 28.5383,
                    "longitude": -81.3792,
                    "elevation": 300.0,
                    "min_elevation_angle": 10.0,
                }
        self.propagator_dict = {
                "propagator_type": "SGP4_PROPAGATOR",
                "step_size": 60,  # seconds
            }
        self.point_array = Cartesian3DPositionArray.from_geographic_positions([
            GeographicPosition(latitude_degrees=45.0, longitude_degrees=45.0, elevation_m=0.0),
            GeographicPosition(latitude_degrees=-45.0, longitude_degrees=-45.0, elevation_m=0.0),
            GeographicPosition(latitude_degrees=0.0, longitude_degrees=0.0, elevation_m=0.0),
        ])
        
        # Below dictionary has to be in the exact same format which would be produced by to_dict() method
        # E.g. time_scale 'UTC' should be in upper case, having the (single) spacecraft in a list.
        self.mission_dict = {
            "start_time": {
                "time_format": "GREGORIAN_DATE",
                "calendar_date": "2025-04-17T12:00:00.000",
                "time_scale": "UTC",
            },
            "duration_days": 1.0,
            "spacecrafts": [{
                    "id": 'ddd716b0-443b-4141-a413-19b14260db9a',
                    "name": None,
                    "orbit": {
                        "orbit_type": "TWO_LINE_ELEMENT_SET",
                        "TLE_LINE0": "0 LANDSAT 9",
                        "TLE_LINE1": "1 49260U 21088A   25106.07240456  .00000957  00000-0  22241-3 0  9997",
                        "TLE_LINE2": "2 49260  98.1921 177.4890 0001161  87.5064 272.6267 14.57121096188801"
                    },
                    "sensor": [{
                        "id": '699881aa-5e50-4186-ba03-1eafaa4e6f62',
                        "name": None,
                        "fov": {
                            "fov_type": "CIRCULAR",
                            "diameter": 60.0,
                            "frame": "SENSOR_BODY_FIXED",
                            "boresight": [0.0, 0.0, 1.0],
                        },
                    }],
            }],
            "frame_transforms": {
                "orientation_transforms": [
                    {
                    "orientation_type": "constant", "rotations": [0.0, 0.0, 0.0],
                    "rotations_type": "EULER", "from": "SENSOR_BODY_FIXED", "to": "ORBITPY_LVLH", "euler_order": "xyz"
                    }
                ],
                "position_transforms": [
                    {
                        "from_frame": "SENSOR_BODY_FIXED", "to_frame": "ORBITPY_LVLH",
                        "position": { "x": 0.0,  "y": 0.0, "z": 0.0,
                            "frame": "SENSOR_BODY_FIXED", "type": "Cartesian3DPosition"
                        }
                    }
                ]
                
            },
            "ground_stations": [{
                    "id": '69e3233c-50ce-4aeb-bd40-74d74537f0ed',
                    "name": None,
                    "latitude": 28.5383,
                    "longitude": -81.3792,
                    "height": 200.0,
                    "min_elevation_angle": 10.0,
                }],
            "propagator": {
                "propagator_type": "SGP4_PROPAGATOR",
                "step_size": 60,  # seconds
            },
            "grid_points": self.point_array.to_dict(),
        }
        
    
    def test_from_dict_and_to_dict_roundtrip(self):
        m = Mission.from_dict(self.mission_dict)
        m_dict = m.to_dict()
        self.assertEqual(m_dict.get("start_time"), self.mission_dict.get("start_time"))
        self.assertEqual(m_dict.get("duration_days"), self.mission_dict.get("duration_days"))
        self.assertEqual(m_dict.get("spacecrafts"), self.mission_dict.get("spacecrafts"))  # single spacecraft
        self.assertEqual(m_dict.get("ground_stations"), self.mission_dict.get("ground_stations"))
        self.assertEqual(m_dict.get("propagator"), self.mission_dict.get("propagator"))
        #self.assertEqual(m_dict.get("grid_points"), self.mission_dict.get("grid_points")) TBD

    def test_execute_propagation(self):
        m = Mission.from_dict(self.mission_dict)
        propagated_trajectories = m.execute_propagation()
        self.assertIsInstance(propagated_trajectories, list)
        self.assertEqual(len(propagated_trajectories), len(m.spacecrafts)) # match the number of spacecrafts

        entry = propagated_trajectories[0]
        # check dict structure and types
        self.assertIsInstance(entry, dict)
        self.assertIsInstance(entry.get("spacecraft_id"), str)
        self.assertIn(entry.get("spacecraft_id"), m.spacecrafts[0].identifier)
        self.assertIn("trajectory", entry)
        self.assertIsInstance(entry["trajectory"], StateSeries)
        #JsonSerializer.save_to_json(propagated_trajectories, 'test_mission_propagation_output.json')

    def test_execute_eclipse_finder(self):
        m = Mission.from_dict(self.mission_dict)
        propagated_trajectories = m.execute_propagation()
        eclipse_results = m.execute_eclipse_finder(propagated_trajectories)
        self.assertIsInstance(eclipse_results, list)
        self.assertEqual(len(eclipse_results), len(m.spacecrafts)) # match the number of spacecrafts

        entry = eclipse_results[0]
        self.assertIsInstance(entry, dict)
        self.assertIsInstance(entry.get("spacecraft_id"), str)
        self.assertIn(entry.get("spacecraft_id"), m.spacecrafts[0].identifier)
        self.assertIn("eclipse_info", entry)
        self.assertIsInstance(entry["eclipse_info"], EclipseInfo)
        #JsonSerializer.save_to_json(eclipse_results, 'test_mission_eclipse_output.json')

    def test_execute_gs_contact_finder(self):
        m = Mission.from_dict(self.mission_dict)
        propagated_trajectories = m.execute_propagation()
        contact_info = m.execute_gs_contact_finder(propagated_trajectories)

        # contact_info is a list of dicts, each with "spacecraft_id" and "contacts" (a list of {ground_station_id, contact_info})
        self.assertIsInstance(contact_info, list)
        self.assertEqual(len(contact_info), len(m.spacecrafts))  # match the number of spacecrafts

        sc_id = m.spacecrafts[0].identifier
        gs_id = m.ground_stations[0].identifier

        # find the entry for our spacecraft
        sc_entry = contact_info[0]
        self.assertEqual(sc_entry.get("spacecraft_id"), sc_id)
        self.assertIn("contacts", sc_entry)
        self.assertIsInstance(sc_entry["contacts"], list)
        self.assertEqual(len(sc_entry["contacts"]), len(m.ground_stations))  # match the number of ground-stations

        # find the contact entry for the ground station and check its type
        gs_entry = contact_info[0]["contacts"][0]
        self.assertEqual(gs_entry.get("ground_station_id"), gs_id)
        self.assertIsNotNone(gs_entry)
        self.assertIn("contact_info", gs_entry)
        self.assertIsInstance(gs_entry["contact_info"], ContactInfo)
        #JsonSerializer.save_to_json(contact_info, 'test_mission_contact_output.json')

    def test_execute_coverage_calculator(self):
        m = Mission.from_dict(self.mission_dict)
        propagated_trajectories = m.execute_propagation()
        coverage_info = m.execute_coverage_calculator(propagated_trajectories)

        # Updated expectations to match Mission.execute_coverage_calculator output structure:
        # a list of dicts, each with "spacecraft_id" and "coverage" (a list of {sensor_id, coverage_info})
        self.assertIsInstance(coverage_info, list)
        self.assertEqual(len(coverage_info), len(m.spacecrafts))  # match the number of spacecrafts

        sc_id = m.spacecrafts[0].identifier
        sensor_id = m.spacecrafts[0].sensor[0].identifier

        # find the entry for our spacecraft
        sc_entry = coverage_info[0]
        self.assertEqual(sc_entry.get("spacecraft_id"), sc_id)
        self.assertIn("coverage", sc_entry)
        self.assertIsInstance(sc_entry["coverage"], list)
        self.assertEqual(len(sc_entry["coverage"]), len(m.spacecrafts[0].sensor))  # match the number of sensors

        # find the sensor coverage entry and verify its type
        sensor_entry = coverage_info[0]["coverage"][0]
        self.assertEqual(sensor_entry.get("sensor_id"), sensor_id)
        self.assertIn("coverage_info", sensor_entry)
        self.assertIsInstance(sensor_entry["coverage_info"], DiscreteCoverageTP)

        #JsonSerializer.save_to_json(coverage_info, 'test_mission_coverage_output.json')


if __name__ == "__main__":
    unittest.main()
