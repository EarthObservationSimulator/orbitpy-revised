"""Unit tests for orbitpy.orbits module."""
import unittest
import random
import numpy as np

from eosimutils.base import JsonSerializer
from eosimutils.time import AbsoluteDate
from eosimutils.trajectory import StateSeries

from orbitpy.mission import Mission, propagate_spacecraft
from orbitpy.resources import Spacecraft, GroundStation
from orbitpy.propagator import SGP4Propagator
from orbitpy.eclipsefinder import EclipseInfo
from orbitpy.contactfinder import ContactInfo

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
                            "frame": "ICRF_EC",
                            "boresight": [0.0, 0.0, 1.0],
                        },
                    }],
            }],
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
            }
        }
        
    
    def test_from_dict_and_to_dict_roundtrip(self):
        m = Mission.from_dict(self.mission_dict)
        m_dict = m.to_dict()
        self.assertEqual(m_dict.get("start_time"), self.mission_dict.get("start_time"))
        self.assertEqual(m_dict.get("duration_days"), self.mission_dict.get("duration_days"))
        self.assertEqual(m_dict.get("spacecrafts"), self.mission_dict.get("spacecrafts"))  # single spacecraft
        self.assertEqual(m_dict.get("ground_stations"), self.mission_dict.get("ground_stations"))
        self.assertEqual(m_dict.get("propagator"), self.mission_dict.get("propagator"))
        self.assertIsNone(m_dict.get("grid_points"))  # grid_points is None

    def test_execute_propagation(self):
        m = Mission.from_dict(self.mission_dict)
        propagated_trajectories = m.execute_propagation()
        self.assertIsInstance(propagated_trajectories, dict)
        self.assertIn(m.spacecrafts[0].identifier, propagated_trajectories)
        self.assertIsInstance(propagated_trajectories[m.spacecrafts[0].identifier], StateSeries)
        #JsonSerializer.save_to_json(propagated_trajectories[m.spacecrafts[0].identifier], 'test_mission_propagation_output.json')

    def test_execute_eclipse_finder(self):
        m = Mission.from_dict(self.mission_dict)
        propagated_trajectories = m.execute_propagation()
        eclipse_info = m.execute_eclipse_finder(propagated_trajectories)
        self.assertIsInstance(eclipse_info, dict)
        self.assertIn(m.spacecrafts[0].identifier, eclipse_info)
        self.assertIsInstance(eclipse_info[m.spacecrafts[0].identifier], EclipseInfo)
        #JsonSerializer.save_to_json(eclipse_info[m.spacecrafts[0].identifier], 'test_mission_eclipse_output.json')
    
    def test_execute_gs_contact_finder(self):
        m = Mission.from_dict(self.mission_dict)
        propagated_trajectories = m.execute_propagation()
        contact_info = m.execute_gs_contact_finder(propagated_trajectories)
        self.assertIsInstance(contact_info, dict)
        contact_info_key = f"{m.spacecrafts[0].identifier}_to_{m.ground_stations[0].identifier}"
        self.assertIn(contact_info_key, contact_info)
        self.assertIsInstance(contact_info[contact_info_key], ContactInfo)
        #JsonSerializer.save_to_json(contact_info[contact_info_key], 'test_mission_contact_output.json')

if __name__ == "__main__":
    unittest.main()
