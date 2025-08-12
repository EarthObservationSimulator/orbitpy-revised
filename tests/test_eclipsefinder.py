"""Unit tests for orbitpy.eclipsefinder module."""

import unittest
import random
import numpy as np

from eosimutils.base import ReferenceFrame, SurfaceType
from eosimutils.time import AbsoluteDate, AbsoluteDateArray
from eosimutils.state import (
    Cartesian3DPosition,
    CartesianState,
)
from eosimutils.trajectory import PositionSeries, StateSeries
from eosimutils.framegraph import FrameGraph

from orbitpy.eclipsefinder import EclipseFinder

class TestEclipseFinder(unittest.TestCase):
    """Unit tests for the EclipseFinder class."""

    def setUp(self):
        """Set up common test data."""
        self.eclipse_finder = EclipseFinder()

        self.registry = FrameGraph()

        # Single time and position
        self.single_time = AbsoluteDate.from_dict({
            "time_format": "GREGORIAN_DATE",
            "calendar_date": "2025-03-17T13:00:00.000",
            "time_scale": "UTC",
        })
        self.single_position = Cartesian3DPosition.from_dict({
            "x": 7000.0,
            "y": 0.0,
            "z": 7000.0,
            "frame": "ITRF",
        })

        # Multiple times and single position
        self.multiple_times = AbsoluteDateArray.from_dict({
            "time_format": "GREGORIAN_DATE",
            "calendar_date": [
                "2025-03-17T13:00:00.000",
                "2025-03-17T14:00:00.000",
                "2025-03-17T15:00:00.000"
            ],
            "time_scale": "UTC",
        })

        # CartesianState
        self.cartesian_state = CartesianState.from_dict({
            "time": {
                "time_format": "GREGORIAN_DATE",
                "calendar_date": "2025-03-17T13:00:00.000",
                "time_scale": "UTC",
            },
            "position": [7000.0, 0.0, 7000.0],
            "velocity": [0.0, 7.5, 0.0],
            "frame": "ITRF",
        })

        # PositionSeries
        self.position_series = PositionSeries.from_dict({
            "time": {
                "time_format": "GREGORIAN_DATE",
                "calendar_date": [
                    "2025-03-17T13:00:00.000",
                    "2025-03-17T14:00:00.000",
                    "2025-03-17T15:00:00.000"
                ],
                "time_scale": "UTC",
            },
            "data": [
                [7000.0, 0.0, 7000.0],
                [7100.0, 0.0, 7100.0],
                [7200.0, 0.0, 7200.0],
            ],
            "frame": "ITRF",
        })

        # StateSeries
        self.state_series = StateSeries.from_dict({
            "time": {
                "time_format": "GREGORIAN_DATE",
                "calendar_date": [
                    "2025-03-17T13:00:00.000",
                    "2025-03-17T14:00:00.000"
                ],
                "time_scale": "UTC",
            },
            "data": [
                [[7000.0, 0.0, 7000.0], [7000.0, 0.0, 7000.0]],  # Positions
                [[1, 7, 0], [0, 7, 1]]  # Velocities
            ],
            "frame": "ITRF",
            "headers": [["pos_x", "pos_y", "pos_z"], ["vel_x", "vel_y", "vel_z"]]
        })

    def test_eclipse_single_position_single_time(self):
        """Test eclipse detection for a single position and single time."""
        result = self.eclipse_finder.execute(
            frame_graph=self.registry,
            time=self.single_time,
            position=self.single_position
        )
        expected_result = False  # Replace with the correct expected value
        self.assertEqual(result, expected_result)

    def test_eclipse_single_position_multiple_time(self):
        """Test eclipse detection for a single position and multiple times."""
        result = self.eclipse_finder.execute(
            frame_graph=self.registry,
            time=self.multiple_times,
            position=self.single_position
        )
        expected_results = [False, False, False]  # Replace with the correct expected values
        self.assertTrue(np.array_equal(result, expected_results))

    def test_eclipse_with_cartesian_state(self):
        """Test eclipse detection with a CartesianState object."""
        result = self.eclipse_finder.execute(
            frame_graph=self.registry,
            state=self.cartesian_state
        )
        expected_result = False  # Replace with the correct expected value
        self.assertEqual(result, expected_result)

    def test_eclipse_with_position_series(self):
        """Test eclipse detection with a PositionSeries object."""
        result = self.eclipse_finder.execute(
            frame_graph=self.registry,
            state=self.position_series
        )
        expected_results = [False, False, False]  # Replace with the correct expected values
        self.assertTrue(np.array_equal(result, expected_results))

    def test_eclipse_with_state_series(self):
        """Test eclipse detection with a StateSeries object."""
        result = self.eclipse_finder.execute(
            frame_graph=self.registry,
            state=self.state_series
        )
        expected_results = [False, False]  # Replace with the correct expected values
        self.assertTrue(np.array_equal(result, expected_results))

if __name__ == "__main__":
    unittest.main()