""" Validation tests for the contactfinder module using GMAT-generated data.
Maximum deviation in contact interval start and stop times were found to be within 14 seconds.
This is likely due to the different methods used by GMAT (continuous domain) 
and OrbitPy's ElevationAwareContactFinder (computation at discrete times).
"""

import os
import unittest
import pickle

from eosimutils.framegraph import FrameGraph

from orbitpy.resources import GroundStation
from orbitpy.contactfinder import (
    ContactInfo,
    ElevationAwareContactFinder,
)

from gmatutil import parse_gmat_state_file, parse_gmat_contact_file


class TestElevationAwareContactFinder(unittest.TestCase):

    def setUp(self):

        self.elevation_contact_finder = ElevationAwareContactFinder()
        self.frame_graph = FrameGraph()
        
        if "noaa_trajectory.pkl" in os.listdir():
            print("Loading cached noaa_trajectory.pkl")
            with open("noaa_trajectory.pkl", "rb") as f:
                self.noaa_trajectory = pickle.load(f)
        else:
            self.noaa_trajectory = parse_gmat_state_file("gmat/noaa20_viirs/Cartesian.txt")
            with open("noaa_trajectory.pkl", "wb") as f:
                pickle.dump(self.noaa_trajectory, f, protocol=pickle.HIGHEST_PROTOCOL)

        # define the ground-stations
        self.station_1 = GroundStation.from_dict({
            "name": "StationID1",
            "latitude": 0.0,
            "longitude": 0.0,
            "height": 0.0,
            "min_elevation_angle": 7
        })
        self.station_2 = GroundStation.from_dict({
            "name": "StationID2",
            "latitude": 45.0,
            "longitude": 45.0,
            "height": 350,
            "min_elevation_angle": 15
        })
        self.station_3 = GroundStation.from_dict({
            "name": "StationID3",
            "latitude": 45.0,
            "longitude": 315.0,
            "height": 100,
            "min_elevation_angle": 30
        })
        self.station_4 = GroundStation.from_dict({
            "name": "StationID4",
            "latitude": -45.0,
            "longitude": 45.0,
            "height": 1000,
            "min_elevation_angle": 0
        })
        self.station_5 = GroundStation.from_dict({
            "name": "StationID5",
            "latitude": -45.0,
            "longitude": 315.0,
            "height": 10000,
            "min_elevation_angle": 10
        })
        self.station_6 = GroundStation.from_dict({
            "name": "StationID6",
            "latitude": 90.0,
            "longitude": 90.0,
            "height": 500,
            "min_elevation_angle": 7
        })
        self.station_7 = GroundStation.from_dict({
            "name": "StationID7",
            "latitude": -90.0,
            "longitude": 180.0,
            "height": 500,
            "min_elevation_angle": 7
        })

    def test_elevation_contact_finder_with_station_1(self):
        """ Compare GMAT ContactLocator results with ElevationAwareContactFinder results.
        Errors are expected since the ElevationAwareContactFinder calculated the contacts at the
        trajectory time-steps only and evaluates the contact interval, whereas GMAT possibly 
        uses a continuous technique to find more accurate contact times.
        """
        gmat_contacts = parse_gmat_contact_file("gmat/noaa20_viirs/ContactLocator1.txt")

        result = self.elevation_contact_finder.execute(
            frame_graph=self.frame_graph,
            observer_state=self.station_1.geographic_position,
            target_state=self.noaa_trajectory,
            min_elevation_angle=self.station_1.min_elevation_angle_deg,
        )

        self.assertIsInstance(result, ContactInfo)

        self.assertEqual(len(gmat_contacts), len(result.contact_intervals())) # match number of contact intervals

        for gmat_contact_item, orbitpy_contact_item in zip(gmat_contacts, result.contact_intervals()):
            self.assertAlmostEqual(
                gmat_contact_item[0].to_spice_ephemeris_time(),
                orbitpy_contact_item[0].to_spice_ephemeris_time(),
                delta=10  # allow up to 10 seconds difference
            )
            self.assertAlmostEqual(
                gmat_contact_item[1].to_spice_ephemeris_time(),
                orbitpy_contact_item[1].to_spice_ephemeris_time(),
                delta=10  # allow up to 10 seconds difference
            )

    def test_elevation_contact_finder_with_other_stations(self):
            """ Compare GMAT ContactLocator results with ElevationAwareContactFinder results.
            Errors are expected since the ElevationAwareContactFinder calculated the contacts at the
            trajectory time-steps only and evaluates the contact interval, whereas GMAT possibly 
            uses a continuous technique to find more accurate contact times.
            """
            
            for station in [
                self.station_2,
                self.station_3,
                self.station_4,
                self.station_5,
                self.station_6,
                self.station_7,
            ]:
                with self.subTest(station=station.name):
                    gmat_contacts = parse_gmat_contact_file(
                        f"gmat/noaa20_viirs/ContactLocator{station.name[-1]}.txt"
                    )

                    result = self.elevation_contact_finder.execute(
                        frame_graph=self.frame_graph,
                        observer_state=station.geographic_position,
                        target_state=self.noaa_trajectory,
                        min_elevation_angle=station.min_elevation_angle_deg,
                    )

                    self.assertIsInstance(result, ContactInfo)

                    self.assertEqual(len(gmat_contacts), len(result.contact_intervals())) # match number of contact intervals

                    for gmat_contact_item, orbitpy_contact_item in zip(gmat_contacts, result.contact_intervals()):
                        self.assertAlmostEqual(
                            gmat_contact_item[0].to_spice_ephemeris_time(),
                            orbitpy_contact_item[0].to_spice_ephemeris_time(),
                            delta=14  # allow up to 14 seconds difference
                        )
                        self.assertAlmostEqual(
                            gmat_contact_item[1].to_spice_ephemeris_time(),
                            orbitpy_contact_item[1].to_spice_ephemeris_time(),
                            delta=14  # allow up to 14 seconds difference
                        )

if __name__ == "__main__":
    unittest.main()