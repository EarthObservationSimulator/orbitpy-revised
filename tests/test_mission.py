"""Unit tests for orbitpy.orbits module."""
import unittest
import random
import numpy as np

from eosimutils.base import JsonSerializer
from eosimutils.trajectory import StateSeries
from eosimutils.state import Cartesian3DPositionArray, GeographicPosition

from orbitpy.mission import Mission, PropagationResults, EclipseFinderResults
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
        self.assertIsInstance(propagated_trajectories, PropagationResults)
        self.assertEqual(len(propagated_trajectories), len(m.spacecrafts)) # match the number of spacecrafts
        self.assertIn(m.spacecrafts[0].identifier, propagated_trajectories.spacecraft_id)
        self.assertIsInstance(propagated_trajectories.trajectory[0], StateSeries)
        #JsonSerializer.save_to_json(propagated_trajectories.to_dict(), 'test_mission_propagation_output.json')

    def test_execute_eclipse_finder(self):
        m = Mission.from_dict(self.mission_dict)
        propagated_trajectories = m.execute_propagation()
        eclipse_results = m.execute_eclipse_finder(propagated_trajectories)
        self.assertIsInstance(eclipse_results, EclipseFinderResults)
        self.assertEqual(len(eclipse_results.spacecraft_id), len(m.spacecrafts)) # match the number of spacecrafts
        self.assertIn(m.spacecrafts[0].identifier, eclipse_results.spacecraft_id)
        self.assertIsInstance(eclipse_results.eclipse_info[0], EclipseInfo)
        #JsonSerializer.save_to_json(eclipse_info[m.spacecrafts[0].identifier], 'test_mission_eclipse_output.json')
    
    def test_execute_gs_contact_finder(self):
        m = Mission.from_dict(self.mission_dict)
        propagated_trajectories = m.execute_propagation()
        contact_info = m.execute_gs_contact_finder(propagated_trajectories)
        self.assertIsInstance(contact_info, dict)
        self.assertEqual(len(contact_info.keys()), len(m.spacecrafts)) # match the number of spacecrafts

        sc_id = m.spacecrafts[0].identifier
        gs_id = m.ground_stations[0].identifier

        # top-level key is spacecraft id
        self.assertIn(sc_id, contact_info)
        self.assertIsInstance(contact_info[sc_id], dict)

        # nested key is ground-station id and maps to ContactInfo
        self.assertIn(gs_id, contact_info[sc_id])
        self.assertEqual(len(contact_info[sc_id].keys()), len(m.ground_stations)) # match the number of ground-stations
        self.assertIsInstance(contact_info[sc_id][gs_id], ContactInfo)

        #JsonSerializer.save_to_json(contact_info[sc_id][gs_id], 'test_mission_contact_output.json')
    
    def test_execute_coverage_calculator(self):
        m = Mission.from_dict(self.mission_dict)
        propagated_trajectories = m.execute_propagation()
        coverage_info = m.execute_coverage_calculator(propagated_trajectories)
        self.assertIsInstance(coverage_info, dict)
        self.assertEqual(len(coverage_info.keys()), len(m.spacecrafts)) # match the number of spacecrafts

        sc_id = m.spacecrafts[0].identifier
        sensor_id = m.spacecrafts[0].sensor[0].identifier

        # top-level key is spacecraft id
        self.assertIn(sc_id, coverage_info)
        self.assertIsInstance(coverage_info[sc_id], dict)
        self.assertEqual(len(coverage_info[sc_id].keys()), len(m.spacecrafts[0].sensor)) # match the number of sensors

        # nested key is sensor id and maps to DiscreteCoverageTP
        self.assertIn(sensor_id, coverage_info[sc_id])
        self.assertIsInstance(coverage_info[sc_id][sensor_id], DiscreteCoverageTP)

        #JsonSerializer.save_to_json(coverage_info[sc_id][sensor_id], 'test_mission_coverage_output.json')


if __name__ == "__main__":
    unittest.main()
