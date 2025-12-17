""" Validation tests for the eclipsefinder module using GMAT-generated data.
Maximum deviation in eclipse interval start and stop times were found to be within 6 seconds for the NOAA-20 satellite
and 13.5 seconds for the CYGFM-01 satellite.
This is likely due to the different methods used by GMAT (continuous domain, more accurate computation of umbra, penumbra)
and OrbitPy's EclipseFinder (computation at discrete times, and point model of Sun).
"""

import os
import unittest
import pickle

from eosimutils.framegraph import FrameGraph

from orbitpy.eclipsefinder import EclipseFinder
from orbitpy.eclipsefinder import EclipseInfo

from gmatutil import parse_gmat_state_file, parse_gmat_eclipse_file

class TestEclipseFinder(unittest.TestCase):

    def setUp(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.gmat_data_dir = os.path.join(self.script_dir, "gmat")
        self.noaa_dir = os.path.join(self.gmat_data_dir, "noaa20_viirs")
        self.cygfm01_dir = os.path.join(self.gmat_data_dir, "cygfm01")

        self.eclipse_finder = EclipseFinder()
        self.frame_graph = FrameGraph()

        # Load NOAA-20 trajectory
        if "noaa_trajectory.pkl" in os.listdir(self.noaa_dir):
            print("Loading cached noaa_trajectory.pkl")
            with open(os.path.join(self.noaa_dir, "noaa_trajectory.pkl"), "rb") as f:
                self.noaa_trajectory = pickle.load(f)
        else:
            self.noaa_trajectory = parse_gmat_state_file(os.path.join(self.noaa_dir, "Cartesian.txt"))
            with open(os.path.join(self.noaa_dir, "noaa_trajectory.pkl"), "wb") as f:
                pickle.dump(self.noaa_trajectory, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Load CYGFM-01 trajectory
        if "cygfm01_trajectory.pkl" in os.listdir(self.cygfm01_dir):
            print("Loading cached cygfm01_trajectory.pkl")
            with open(os.path.join(self.cygfm01_dir, "cygfm01_trajectory.pkl"), "rb") as f:
                self.cygfm01_trajectory = pickle.load(f)
        else:
            self.cygfm01_trajectory = parse_gmat_state_file(os.path.join(self.cygfm01_dir, "Cartesian.txt"))
            with open(os.path.join(self.cygfm01_dir, "cygfm01_trajectory.pkl"), "wb") as f:
                pickle.dump(self.cygfm01_trajectory, f, protocol=pickle.HIGHEST_PROTOCOL)

    def test_eclipse_intervals_noaa20(self):
        """Compare GMAT EclipseLocator results with EclipseFinder results for the NOAA-20 satellite.
        Small differences are expected due to discrete time steps in OrbitPy versus GMAT which possibly uses continuous search.
        """
        gmat_eclipses = parse_gmat_eclipse_file(os.path.join(self.noaa_dir, "EclipseLocator1.txt"))

        result: EclipseInfo = self.eclipse_finder.execute(
            frame_graph=self.frame_graph,
            state=self.noaa_trajectory
        )

        # Only compare umbra intervals
        gmat_umbra_intervals = [
            (event["start"], event["stop"])
            for event in gmat_eclipses if event["type"].lower() == "umbra"
        ]
        orbitpy_eclipse_intervals = result.eclipse_intervals()

        self.assertEqual(len(gmat_umbra_intervals), len(orbitpy_eclipse_intervals))

        for gmat_event, orbitpy_event in zip(gmat_umbra_intervals, orbitpy_eclipse_intervals):
            self.assertAlmostEqual(
                gmat_event[0].to_spice_ephemeris_time(),
                orbitpy_event[0].to_spice_ephemeris_time(),
                delta=6  # allow up to 6 seconds difference
            )
            self.assertAlmostEqual(
                gmat_event[1].to_spice_ephemeris_time(),
                orbitpy_event[1].to_spice_ephemeris_time(),
                delta=6  # allow up to 6 seconds difference
            )
    
    def test_eclipse_intervals_cygfm01(self):
        """Compare GMAT EclipseLocator results with EclipseFinder results for the CYGFM-01 satellite.
        Small differences are expected due to discrete time steps in OrbitPy versus GMAT which possibly uses continuous search.
        """
        gmat_eclipses = parse_gmat_eclipse_file(os.path.join(self.cygfm01_dir, "EclipseLocator1.txt"))

        result: EclipseInfo = self.eclipse_finder.execute(
            frame_graph=self.frame_graph,
            state=self.cygfm01_trajectory
        )

        # Only compare umbra intervals
        gmat_umbra_intervals = [
            (event["start"], event["stop"])
            for event in gmat_eclipses if event["type"].lower() == "umbra"
        ]
        orbitpy_eclipse_intervals = result.eclipse_intervals()

        self.assertEqual(len(gmat_umbra_intervals), len(orbitpy_eclipse_intervals))

        for gmat_event, orbitpy_event in zip(gmat_umbra_intervals, orbitpy_eclipse_intervals):
            self.assertAlmostEqual(
                gmat_event[0].to_spice_ephemeris_time(),
                orbitpy_event[0].to_spice_ephemeris_time(),
                delta=13.5  # allow up to 13.5 seconds difference
            )
            self.assertAlmostEqual(
                gmat_event[1].to_spice_ephemeris_time(),
                orbitpy_event[1].to_spice_ephemeris_time(),
                delta=13.5  # allow up to 13.5 seconds difference
            )

if __name__ == "__main__":
    unittest.main()