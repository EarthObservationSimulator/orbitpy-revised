"""Unit tests for orbitpy.orbits module."""
import unittest
import random
import numpy as np

from eosimutils.base import JsonSerializer, ReferenceFrame, SPHERICAL_EARTH_MEAN_RADIUS
from eosimutils.trajectory import StateSeries
from eosimutils.state import Cartesian3DPositionArray, GeographicPosition, Cartesian3DPosition

from orbitpy.mission import Mission
from orbitpy.resources import Spacecraft, GroundStation
from orbitpy.propagator import SGP4Propagator
from orbitpy.eclipsefinder import EclipseInfo
from orbitpy.contactfinder import ContactInfo
from orbitpy.coverage import DiscreteCoverageTP


def uniform_lat_lon_spacing_grid(lat_lower_bound: float, lat_upper_bound: float, lat_step_deg: float, 
                                 lon_lower_bound: float, lon_upper_bound: float, lon_step_deg: float) -> Cartesian3DPositionArray:
    """Return uniformly spaced GeographicPosition points on a latitude/longitude grid given the bounds."""
    latitudes = np.arange(lat_lower_bound, lat_upper_bound + lat_step_deg, lat_step_deg)
    longitudes = np.arange(lon_lower_bound, lon_upper_bound + lon_step_deg, lon_step_deg)

    return ([GeographicPosition(latitude_degrees=lat, longitude_degrees=lon, elevation_m=0.0)
                for lat in latitudes for lon in longitudes
            ])

class TestMission_1(unittest.TestCase):
    """Tests for the Mission class with single spacecraft and single ground station."""
    def setUp(self):
        
        self.points_array = Cartesian3DPositionArray.from_geographic_positions([
            GeographicPosition(latitude_degrees=45.0, longitude_degrees=45.0, elevation_m=0.0),
            GeographicPosition(latitude_degrees=-45.0, longitude_degrees=-45.0, elevation_m=0.0),
            GeographicPosition(latitude_degrees=0.0, longitude_degrees=0.0, elevation_m=0.0),
        ])
        
        # For purposes of the `test_from_dict_and_to_dict_roundtrip` test, below dictionary
        # has to be in the exact same format which would be produced by to_dict() method.
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
                    "name": "SC_A",
                    "orbit": {
                        "orbit_type": "TWO_LINE_ELEMENT_SET",
                        "TLE_LINE0": "0 LANDSAT 9",
                        "TLE_LINE1": "1 49260U 21088A   25106.07240456  .00000957  00000-0  22241-3 0  9997",
                        "TLE_LINE2": "2 49260  98.1921 177.4890 0001161  87.5064 272.6267 14.57121096188801"
                    },
                    "local_orbital_frame_handler": {
                        "frame_type": "LVLH_TYPE_1",
                        "name": "LVLH_SC_A",
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
                    }
                    ],
            }],
            "frame_transforms": {
                "orientation_transforms": [
                    {
                    "orientation_type": "constant", "rotations": [0.0, 0.0, 0.0],
                    "rotations_type": "EULER", "from": "SENSOR_BODY_FIXED", "to": "LVLH_SC_A", "euler_order": "xyz"
                    }
                ],
                "position_transforms": [
                    {
                        "from_frame": "SENSOR_BODY_FIXED", "to_frame": "LVLH_SC_A",
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
                }
            ],
            "propagator": {
                "propagator_type": "SGP4_PROPAGATOR",
                "step_size": 60,  # seconds
            },
            "grid_points": self.points_array.to_dict(),
        }
        

    def test_from_dict_and_to_dict_roundtrip(self):
        m = Mission.from_dict(self.mission_dict)
        m_dict = m.to_dict()
        self.assertEqual(m_dict.get("start_time"), self.mission_dict.get("start_time"))
        self.assertEqual(m_dict.get("duration_days"), self.mission_dict.get("duration_days"))
        self.assertEqual(m_dict.get("spacecrafts"), self.mission_dict.get("spacecrafts"))  # single spacecraft
        self.assertEqual(m_dict.get("ground_stations"), self.mission_dict.get("ground_stations"))
        self.assertEqual(m_dict.get("propagator"), self.mission_dict.get("propagator"))
        self.assertEqual(m_dict.get("grid_points"), self.mission_dict.get("grid_points"))

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

        # find the contact entry for the 1st ground station and check its type
        gs_entry = contact_info[0]["contacts"][0]
        self.assertEqual(gs_entry.get("ground_station_id"), gs_id)
        self.assertIn("contact_info", gs_entry)
        self.assertIsInstance(gs_entry["contact_info"], ContactInfo)
        #JsonSerializer.save_to_json(contact_info, 'test_mission_contact_output.json')

    def test_execute_coverage_calculator(self):
        m = Mission.from_dict(self.mission_dict)
        propagated_trajectories = m.execute_propagation()
        coverage_info = m.execute_coverage_calculator(propagated_trajectories)

        # coverage_info is a list of dicts, each with "spacecraft_id" and "coverage"
        self.assertIsInstance(coverage_info, list)
        self.assertEqual(len(coverage_info), len(m.spacecrafts))

        for sc_idx, sc_entry in enumerate(coverage_info):
            # check spacecraft entry
            self.assertIsInstance(sc_entry, dict)
            expected_sc_id = m.spacecrafts[sc_idx].identifier
            self.assertEqual(sc_entry.get("spacecraft_id"), expected_sc_id)
            self.assertIn("coverage", sc_entry)
            self.assertIsInstance(sc_entry["coverage"], list)
            self.assertEqual(len(sc_entry["coverage"]), len(m.spacecrafts[sc_idx].sensor))

            # check each sensor coverage
            for s_idx, sensor_entry in enumerate(sc_entry["coverage"]):
                expected_sensor_id = m.spacecrafts[sc_idx].sensor[s_idx].identifier
                self.assertEqual(sensor_entry.get("sensor_id"), expected_sensor_id)
                self.assertIn("coverage_info", sensor_entry)
                self.assertIsInstance(sensor_entry["coverage_info"], DiscreteCoverageTP)

        # optionally dump to JSON for manual inspection
        # JsonSerializer.save_to_json(coverage_info, 'test_mission_coverage_output.json')

class TestMission_2(unittest.TestCase):
    """Tests for the Mission class with multiple spacecraft, sensors, and ground stations.
    
    - The first spacecraft 'SCA_A' is in a polar orbit and has two sensors with non-overlapping fields of view.
      Thus, the covered points at a given time should not overlap.
    
    - The second spacecraft 'SC_B' is in a low-inclination orbit and has one sensor. The covered points should be roughly
      within the maximum latitude range of the orbit (roughly, since the field of view of the sensor will cover slightly
      above the maximum covered latitude)."""

    def setUp(self):
        
        # Generate uniformly spaced points on a latitude/longitude grid
        self.all_geo_points_array = uniform_lat_lon_spacing_grid(-90, 90, 1, -180, 180, 1)
        points_array = Cartesian3DPositionArray.from_geographic_positions(self.all_geo_points_array)

        self.mission_dict = {
            "start_time": {
                "time_format": "GREGORIAN_DATE",
                "calendar_date": "2025-09-30T13 :00:00.000",
                "time_scale": "UTC",
            },
            "duration_days": 1.0,
            "spacecrafts": [{
                    "id": 'cf4cea95-e935-4314-a235-02ea87040b11',
                    "name": "SC_A",
                    "orbit": {
                        "orbit_type": "TWO_LINE_ELEMENT_SET",
                        "TLE_LINE0": "0 LANDSAT 8",
                        "TLE_LINE1": "1 39084U 13008A   25273.59195902  .00000981  00000-0  22767-3 0  9990",
                        "TLE_LINE2": "2 39084  98.2157 342.7113 0001218  95.7876 264.3462 14.57123342660093"
                    },
                    "local_orbital_frame_handler": {
                        "frame_type": "LVLH_TYPE_1",
                        "name": "LVLH_SC_A",
                    },
                    "sensor": [{
                        "id": '64bc90ed-a4d3-476c-a8ea-bf2f51e97272',
                        "name": None,
                        "fov": {
                            "fov_type": "RECTANGULAR",
                            "ref_angle": 15.0,
                            "cross_angle": 60.0,
                            "frame": "SENSOR_A1_BODY_FIXED",
                            "boresight": [0.0, 0.0, 1.0],
                        },
                    },
                    {
                        "id": 'ba40657a-291c-4797-becb-b82c55f4eb0b',
                        "name": None,
                        "fov": {
                            "fov_type": "RECTANGULAR",
                            "ref_angle": 15.0,
                            "cross_angle": 60.0,
                            "frame": "SENSOR_A2_BODY_FIXED",
                            "boresight": [0.0, 0.0, 1.0],
                        }
                    }],
            },
            {
                    "id": 'e37db1f2-4dc3-4655-8ffc-50d471c349b9',
                    "name": "SC_B",
                    "orbit": {
                        "orbit_type": "TWO_LINE_ELEMENT_SET",
                        "TLE_LINE0": "0 ISS (ZARYA)",
                        "TLE_LINE1": "1 25544U 98067A   25273.46788961  .00020592  00000-0  37370-3 0  9999",
                        "TLE_LINE2": "2 25544  51.6322 145.2523 0001079 178.6313 181.4679 15.49549771531499"
                    },
                    "local_orbital_frame_handler": {
                        "frame_type": "LVLH_TYPE_1",
                        "name": "LVLH_SC_B",
                    },
                    "sensor": [{
                        "id": '24f6c248-a4d9-4cc4-873a-dd590be4b5dd',
                        "name": None,
                        "fov": {
                            "fov_type": "CIRCULAR",
                            "diameter": 60.0,
                            "frame": "SENSOR_B_BODY_FIXED",
                            "boresight": [0.0, 0.0, 1.0],
                        },
                    }],
            }],
            "frame_transforms": {
                "orientation_transforms": [
                    {
                    "orientation_type": "constant", "rotations": [65.0, 0.0, 0.0],
                    "rotations_type": "EULER", "from": "SENSOR_A1_BODY_FIXED", "to": "LVLH_SC_A", "euler_order": "xyz"
                    },
                    {
                    "orientation_type": "constant", "rotations": [-65.0, 0.0, 0.0],
                    "rotations_type": "EULER", "from": "SENSOR_A2_BODY_FIXED", "to": "LVLH_SC_A", "euler_order": "xyz"
                    },
                    {
                    "orientation_type": "constant", "rotations": [0.0, 0.0, 0.0],
                    "rotations_type": "EULER", "from": "SENSOR_B_BODY_FIXED", "to": "LVLH_SC_B", "euler_order": "xyz"
                    }
                ],
                "position_transforms": [
                    {
                        "from_frame": "SENSOR_A1_BODY_FIXED", "to_frame": "LVLH_SC_A",
                        "position": { "x": 0.0,  "y": 0.0, "z": 0.0,
                            "frame": "SENSOR_A1_BODY_FIXED", "type": "Cartesian3DPosition"
                        }
                    },
                    {
                        "from_frame": "SENSOR_A2_BODY_FIXED", "to_frame": "LVLH_SC_A",
                        "position": { "x": 0.0,  "y": 0.0, "z": 0.0,
                            "frame": "SENSOR_A2_BODY_FIXED", "type": "Cartesian3DPosition"
                        }
                    },
                    {
                        "from_frame": "SENSOR_B_BODY_FIXED", "to_frame": "LVLH_SC_B",
                        "position": { "x": 0.0,  "y": 0.0, "z": 0.0,
                            "frame": "SENSOR_B_BODY_FIXED", "type": "Cartesian3DPosition"
                        }
                    }
                ]
                
            },
            "ground_stations": [{
                    "id": '4f1cadbe-5b96-431d-913f-70845aceff9d',
                    "name": None,
                    "latitude": 28.5383,
                    "longitude": -81.3792,
                    "height": 200.0,
                    "min_elevation_angle": 10.0,
                },
                {
                    "id": 'ec18fecb-4925-4ca6-ba49-7591a5f135cb',
                    "name": None,
                    "latitude": -25.0,
                    "longitude": 63,
                    "height": 350.0,
                    "min_elevation_angle": 25.0,
                },
            ],
            "propagator": {
                "propagator_type": "SGP4_PROPAGATOR",
                "step_size": 60,  # seconds
            },
            "grid_points": points_array.to_dict(),
        }
        self.m = Mission.from_dict(self.mission_dict)
        self.propagated_trajectories = self.m.execute_propagation()
        self.eclipse_results = self.m.execute_eclipse_finder(self.propagated_trajectories)
        self.contact_info = self.m.execute_gs_contact_finder(self.propagated_trajectories)
        self.coverage_info = self.m.execute_coverage_calculator(self.propagated_trajectories)


    def tearDown(self):
        return super().tearDown()
    
    def test_execute_propagation(self):
        self.assertIsInstance(self.propagated_trajectories, list)
        self.assertEqual(len(self.propagated_trajectories), len(self.m.spacecrafts)) # match the number of spacecrafts

        # check dict structure and types
        for idx, entry in enumerate(self.propagated_trajectories):
            self.assertIsInstance(entry, dict)
            self.assertIsInstance(entry.get("spacecraft_id"), str)
            self.assertIn(entry.get("spacecraft_id"), self.m.spacecrafts[idx].identifier)
            self.assertIn("trajectory", entry)
            self.assertIsInstance(entry["trajectory"], StateSeries)

    def test_execute_eclipse_finder(self):
        self.assertIsInstance(self.eclipse_results, list)
        self.assertEqual(len(self.eclipse_results), len(self.m.spacecrafts)) # match the number of spacecrafts

        for idx, entry in enumerate(self.eclipse_results):
            self.assertIsInstance(entry, dict)
            self.assertIsInstance(entry.get("spacecraft_id"), str)
            self.assertIn(entry.get("spacecraft_id"), self.m.spacecrafts[idx].identifier)
            self.assertIn("eclipse_info", entry)
            self.assertIsInstance(entry["eclipse_info"], EclipseInfo)

    def test_execute_gs_contact_finder(self):

        # contact_info is a list of dicts, each with "spacecraft_id" and "contacts" (a list of {ground_station_id, contact_info})
        self.assertIsInstance(self.contact_info, list)
        self.assertEqual(len(self.contact_info), len(self.m.spacecrafts))  # match the number of spacecrafts
        for spc_idx, entry in enumerate(self.contact_info):
            self.assertIsInstance(entry, dict)
            self.assertIsInstance(entry.get("spacecraft_id"), str)
            self.assertIn(entry.get("spacecraft_id"), self.m.spacecrafts[spc_idx].identifier)
            self.assertIn("contacts", entry)
            self.assertIsInstance(entry["contacts"], list)
            self.assertEqual(len(entry["contacts"]), len(self.m.ground_stations))  # match the number of ground-stations

            for gs_idx, gs_entry in enumerate(entry["contacts"]):
                self.assertIsInstance(gs_entry, dict)
                self.assertIsInstance(gs_entry.get("ground_station_id"), str)
                self.assertIn(gs_entry.get("ground_station_id"), self.m.ground_stations[gs_idx].identifier)
                self.assertIn("contact_info", gs_entry)
                self.assertIsInstance(gs_entry["contact_info"], ContactInfo)

    def test_execute_coverage_calculator(self):
    
        # coverage_info is a list of dicts, each with "spacecraft_id" and "coverage"
        self.assertIsInstance(self.coverage_info, list)
        self.assertEqual(len(self.coverage_info), len(self.m.spacecrafts))

        for sc_idx, sc_entry in enumerate(self.coverage_info):
            # check spacecraft entry
            self.assertIsInstance(sc_entry, dict)
            expected_sc_id = self.m.spacecrafts[sc_idx].identifier
            self.assertEqual(sc_entry.get("spacecraft_id"), expected_sc_id)
            self.assertIn("coverage", sc_entry)
            self.assertIsInstance(sc_entry["coverage"], list)
            self.assertEqual(len(sc_entry["coverage"]), len(self.m.spacecrafts[sc_idx].sensor))
            # check each sensor coverage
            for s_idx, sensor_entry in enumerate(sc_entry["coverage"]):
                expected_sensor_id = self.m.spacecrafts[sc_idx].sensor[s_idx].identifier
                self.assertEqual(sensor_entry.get("sensor_id"), expected_sensor_id)
                self.assertIn("coverage_info", sensor_entry)
                self.assertIsInstance(sensor_entry["coverage_info"], DiscreteCoverageTP)

        # optionally dump to JSON for manual inspection
        # JsonSerializer.save_to_json(coverage_info, 'test_mission_coverage_output.json')
    
    
    def test_polar_spacecraft_sensor_coverages_do_not_overlap(self):
        """The two sensors on the polar-orbiting spacecraft should have non-overlapping coverage at each time point."""
        polar_entry = next(item for item in self.coverage_info if item["spacecraft_id"] == "cf4cea95-e935-4314-a235-02ea87040b11") # SC_A
        self.assertEqual(len(polar_entry["coverage"]), 2)
        sensor1_cov = polar_entry["coverage"][0]["coverage_info"]
        sensor2_cov = polar_entry["coverage"][1]["coverage_info"]
        self.assertIsInstance(sensor1_cov, DiscreteCoverageTP)
        self.assertIsInstance(sensor2_cov, DiscreteCoverageTP)
        
        # Compare indices at each time point
        for time_idx in range(len(sensor1_cov.coverage)):
            sensor1_indices = set(sensor1_cov.coverage[time_idx])
            sensor2_indices = set(sensor2_cov.coverage[time_idx])
            self.assertEqual(sensor1_indices & sensor2_indices, set()) # check that the result is an empty set (no overlap)
    
    def test_low_inclination_spacecraft_coverage_within_latitude_bounds(self):
        """The covered points by the low-inclination spacecraft should be roughly within the maximum latitude range of the orbit."""
        iss_entry = next(item for item in self.coverage_info if item["spacecraft_name"] == "SC_B")
        self.assertEqual(len(iss_entry["coverage"]), 1)
        iss_cov = iss_entry["coverage"][0]["coverage_info"]
        self.assertIsInstance(iss_cov, DiscreteCoverageTP)
        
        # Get the Cartesian coordinates of the covered points
        covered_point_indices = []
        for sublist in iss_cov.coverage:
            for point_index in sublist:
                covered_point_indices.append(point_index)
        covered_point_indices = list(set(covered_point_indices))  # unique indices
        #print(covered_point_indices)

        # Find indices of the points in the original self.all_geo_points_array with latitude less than 60 degrees
        lat_bound_indices = [i for i, geo_pos in enumerate(self.all_geo_points_array) if geo_pos.latitude > -60 and geo_pos.latitude < 60]

        # Convert both lists to sets
        covered_point_indices_set = set(covered_point_indices)
        lat_bound_indices_set = set(lat_bound_indices)

        # Check if covered_point_indices is a subset of lat_bound_indices
        assert covered_point_indices_set.issubset(lat_bound_indices_set), "Some covered points are outside the latitude bounds."




if __name__ == "__main__":
    unittest.main()
