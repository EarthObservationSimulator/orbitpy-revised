""" Validation tests for the eclipsefinder module using GMAT-generated data.
Maximum deviation in eclipse interval start and stop times were found to be within 6 seconds.
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

        self.eclipse_finder = EclipseFinder()
        self.frame_graph = FrameGraph()

        if "noaa_trajectory.pkl" in os.listdir():
            print("Loading cached noaa_trajectory.pkl")
            with open("noaa_trajectory.pkl", "rb") as f:
                self.noaa_trajectory = pickle.load(f)
        else:
            self.noaa_trajectory = parse_gmat_state_file(os.path.join(self.script_dir, "gmat/noaa20_viirs/Cartesian.txt"))
            with open("noaa_trajectory.pkl", "wb") as f:
                pickle.dump(self.noaa_trajectory, f, protocol=pickle.HIGHEST_PROTOCOL)

    def test_eclipse_intervals(self):
        """Compare GMAT EclipseLocator results with EclipseFinder results.
        Small differences are expected due to discrete time steps in OrbitPy versus GMAT which possibly uses continuous search.
        """
        gmat_eclipses = parse_gmat_eclipse_file(os.path.join(self.script_dir, "gmat/noaa20_viirs/EclipseLocator1.txt"))

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

if __name__ == "__main__":
    unittest.main()