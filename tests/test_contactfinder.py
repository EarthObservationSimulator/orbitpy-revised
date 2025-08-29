"""Unit tests for orbitpy.contactfinder module."""

import unittest
import random
from scipy.spatial.transform import Rotation as R
import numpy as np

from eosimutils.base import ReferenceFrame, WGS84_EARTH_POLAR_RADIUS
from eosimutils.time import AbsoluteDate, AbsoluteDateArray
from eosimutils.state import (
    Cartesian3DPosition,
    GeographicPosition,
    CartesianState,
    Cartesian3DPositionArray,
)
from eosimutils.trajectory import PositionSeries, StateSeries
from eosimutils.framegraph import FrameGraph
from eosimutils.orientation import ConstantOrientation

from orbitpy.contactfinder import ContactFinderFactory, ContactFinderType, LineOfSightContactFinder

class TestLineOfSightContactFinder(unittest.TestCase):

    def setUp(self):
        """Set up contact finder object."""
        self.los_contact_finder = LineOfSightContactFinder()
        self.registry = (
            FrameGraph()
        )  # ICRF_EC and ITRF frames are automatically registered.

        # register a new frame at 90 deg offset from ICRF_EC
        ReferenceFrame.add("ICRF_EC_90") 
        rot90 = R.from_euler("xyz", [0, 0, 90], degrees=True) # Define 90 deg rotations about Z
        constant_orien_90 = ConstantOrientation(
            rot90, ReferenceFrame.get("ICRF_EC"), ReferenceFrame.get("ICRF_EC_90")
        )
        self.registry.add_orientation_transform(constant_orien_90)
    
    def tearDown(self):
        """Clean up after each test."""
        del self.los_contact_finder
        del self.registry
        ReferenceFrame.delete("ICRF_EC_90")

    
    def test_fixed_geo_location_same_ref_frame(self):
        """Test line-of-sight detection between two fixed geographic positions in the same reference frame.

        The positions are fixed at the same latitude but at opposite longitudes, with random elevations 
        within a specified range. The expected result is False.
        """
        random_latitude = random.uniform(-60, 60)
        random_elevation = random.uniform(0, 10000.0)
        geo_position1 = GeographicPosition.from_dict(
            {
                "latitude": random_latitude,
                "longitude": 0.0,
                "elevation": random_elevation,
            }
        )
        geo_position2 = GeographicPosition.from_dict(
            {
                "latitude": random_latitude,
                "longitude": 180.0,
                "elevation": random_elevation,
            }
        )
        result = self.los_contact_finder.execute(
            self.registry,
            entity1_state=geo_position1,
            entity2_state=geo_position2,
        )
        expected_result = False
        self.assertEqual(result,
                         expected_result,
                         msg=f"Latitude: {random_latitude} and Elevation: {random_elevation} was used.")

    def test_fixed_cartesian_location_same_ref_frame(self):
        """Test line-of-sight detection between two fixed geographic positions in the same reference frame.

        The positions are fixed at the same latitude but at opposite longitudes, with random elevations 
        within a specified range. The expected result is False.
        """
        position1 = Cartesian3DPosition.from_dict(
            {
                "x": 7000.0,
                "y": 0.0,
                "z": 0.0,
                "frame": "ICRF_EC_90",
            }
        )
        position2 = Cartesian3DPosition.from_dict(
            {
                "x": -7000.0,
                "y": 0.0,
                "z": 0.0,
                "frame": "ICRF_EC_90",
            }
        )
        result = self.los_contact_finder.execute(
            self.registry,
            entity1_state=position1,
            entity2_state=position2,
        )
        expected_result = False
        self.assertEqual(result, expected_result,
                         msg=f"Position1: {position1} and Position2: {position2} was used.")

    def test_execute_with_position_series(self):
        """Test LineOfSightContactFinder.execute with a PositionSeries object."""
        frame = ReferenceFrame.get("ITRF")
        times = AbsoluteDateArray.from_dict(
                    {
                        "time_format": "Gregorian_Date",
                        "time_scale": "UTC",
                        "calendar_date": [
                            "2025-03-17T00:00:00.000",
                            "2025-03-17T12:00:00.000",
                            "2025-03-17T16:00:00.000",
                        ]
                    }
                )
        positions = Cartesian3DPositionArray.from_geographic_positions([GeographicPosition(0,0,10000), 
                                                                        GeographicPosition(0, 180, 10000), 
                                                                        GeographicPosition(90, 0, 10000)]).to_numpy()
        position_series = PositionSeries(times, positions, frame)

        fixed_position = GeographicPosition(0,0,0)

        # Execute the contact finder
        los_results = self.los_contact_finder.execute(
            self.registry, position_series, fixed_position
        )

        # Assert the results
        self.assertIsInstance(los_results, list)
        self.assertEqual(len(los_results), len(positions))
        for result in los_results:
            self.assertIsInstance(result, bool)
        expected_result = [True, False, False]
        self.assertEqual(los_results, expected_result,
                         msg=f"Position: {position_series} and Fixed Position: {fixed_position} was used.")