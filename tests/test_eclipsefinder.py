"""Unit tests for orbitpy.eclipsefinder module."""

import unittest
import numpy as np
import random

from eosimutils.base import WGS84_EARTH_POLAR_RADIUS
from eosimutils.time import AbsoluteDate, AbsoluteDateArray
from eosimutils.state import (
    Cartesian3DPosition,
    GeographicPosition,
    CartesianState,
)
from eosimutils.trajectory import PositionSeries, StateSeries
from eosimutils.framegraph import FrameGraph

from orbitpy.eclipsefinder import EclipseFinder, EclipseInfo


class TestEclipseInfo(unittest.TestCase):
    """Unit tests for the EclipseInfo class."""

    def setUp(self):
        """Set up test data for EclipseInfo."""
        self.times = AbsoluteDateArray.from_dict(
            {
                "time_format": "Gregorian_Date",
                "time_scale": "UTC",
                "calendar_date": [
                    "2025-03-17T00:00:00.000",
                    "2025-03-17T12:00:00.000",
                    "2025-03-17T16:00:00.000",
                    "2025-03-17T16:30:00.000",
                ],
            }
        )
        self.data = np.array([True, False, True, True], dtype=bool)
        self.eclipse_info = EclipseInfo(self.times, self.data)

    def test_init_valid_data(self):
        """Test initialization with valid data."""
        self.assertIsInstance(self.eclipse_info, EclipseInfo)
        self.assertTrue(np.array_equal(self.eclipse_info.data[0], self.data))
        self.assertEqual(self.eclipse_info.time, self.times)

    def test_init_invalid_data(self):
        """Test initialization with invalid data."""
        with self.assertRaises(TypeError):
            EclipseInfo(self.times, np.array([1, 0, 1, 1]))  # Non-boolean data

    def test_is_eclipsed(self):
        """Test the is_eclipsed method."""
        self.assertTrue(
            self.eclipse_info.is_eclipsed()
        )  # Check for any eclipse
        self.assertTrue(self.eclipse_info.is_eclipsed(0))  # Eclipse at index 0
        self.assertFalse(
            self.eclipse_info.is_eclipsed(1)
        )  # No eclipse at index 1
        self.assertTrue(self.eclipse_info.is_eclipsed(2))  # Eclipse at index 2
        self.assertTrue(self.eclipse_info.is_eclipsed(3))  # Eclipse at index 3
        self.assertIsNone(self.eclipse_info.is_eclipsed(4))  # Out of bounds

    def test_eclipse_intervals(self):
        """Test the eclipse_intervals method."""
        intervals = self.eclipse_info.eclipse_intervals()
        self.assertEqual(len(intervals), 2)
        self.assertEqual(intervals[0], (self.times[0], self.times[0]))
        self.assertEqual(intervals[1], (self.times[2], self.times[3]))


class TestEclipseFinder(unittest.TestCase):
    """Unit tests for the EclipseFinder class."""

    def setUp(self):
        """Set up eclipse finder object."""
        self.eclipse_finder = EclipseFinder()
        self.registry = (
            FrameGraph()
        )  # ICRF_EC and ITRF frames are automatically registered.

    def test_eclipse_single_position_single_time(self):
        """Test eclipse detection for a single position and single time."""
        time = AbsoluteDate.from_dict(
            {
                "time_format": "GREGORIAN_DATE",
                "calendar_date": "2025-03-25T09:00:00.000",
                "time_scale": "UTC",
            }
        )
        position = Cartesian3DPosition.from_dict(
            {
                "x": 7000.0,
                "y": 0.0,
                "z": 0.0,
                "frame": "ITRF",
            }
        )
        result = self.eclipse_finder.execute(
            frame_graph=self.registry, time=time, position=position
        )
        self.assertIsInstance(result, EclipseInfo)
        self.assertFalse(result.is_eclipsed())

    def test_eclipse_geographic_position(self):
        """Test eclipse detection for a single geographic position and single time.
        Below Geographic position is on the 0 deg longitude (UTC timezone)
        and at noon time in the summer (Northern Hemisphere).
        Therefore the positions along a wide range of latitudes are
        expected to be in sunlight."""

        axis_tilt = 23.5  # degrees
        time = AbsoluteDate.from_dict(
            {
                "time_format": "GREGORIAN_DATE",
                "calendar_date": "2025-06-25T12:00:00.000",
                "time_scale": "UTC",
            }
        )
        random_latitude = random.uniform(-90.0 + axis_tilt, 90.0 + axis_tilt)
        random_elevation = random.uniform(0, 10000.0)
        geo_position = GeographicPosition.from_dict(
            {
                "latitude": random_latitude,
                "longitude": 0.0,
                "elevation": random_elevation,
            }
        )
        result = self.eclipse_finder.execute(
            frame_graph=self.registry, time=time, position=geo_position
        )
        self.assertIsInstance(result, EclipseInfo)
        self.assertFalse(
            result.is_eclipsed(),
            msg=f"Latitude: {random_latitude} and Elevation: {random_elevation} was used.",
        )

    def test_eclipse_single_position_multiple_time(self):
        """Test eclipse detection for a single position and multiple times.
        3 different times corresponding to day, day and night times
        (corresponding to a position) are chosen for the test."""
        # Multiple times
        multiple_times = AbsoluteDateArray.from_dict(
            {
                "time_format": "GREGORIAN_DATE",
                "calendar_date": [
                    "2025-03-17T09:00:00.000",  # day
                    "2025-03-17T15:00:00.000",  # day
                    "2025-03-17T21:00:00.000",  # night
                ],
                "time_scale": "UTC",
            }
        )
        random_latitude = random.uniform(-30.0, 30.0)
        random_longitude = random.uniform(-25.0, 25.0)
        single_position = GeographicPosition.from_dict(
            {
                "latitude": random_latitude,
                "longitude": random_longitude,
                "elevation": 100.0,
            }
        ).to_cartesian3d_position()

        result = self.eclipse_finder.execute(
            frame_graph=self.registry,
            time=multiple_times,
            position=single_position,
        )
        self.assertIsInstance(result, EclipseInfo)
        self.assertTrue(
            np.array_equal(result.data[0], [False, False, True]),
            msg=f"Latitude: {random_latitude}, Longitude: {random_longitude} was used.",
        )

    def test_eclipse_with_cartesian_state(self):
        """Test eclipse detection with a CartesianState object.
        The position is chosen such that it is "above" Earth and not in eclipse
        for any time.
        """
        random_date = f"2025-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}T \
            {random.randint(0, 23):02d}:{random.randint(0, 59):02d}:{random.randint(0, 59):02d}.000"
        random_position = [0.0, 0.0, random.uniform(6800.0, 10000.0)]
        cartesian_state = CartesianState.from_dict(
            {
                "time": {
                    "time_format": "GREGORIAN_DATE",
                    "calendar_date": random_date,
                    "time_scale": "UTC",
                },
                "position": random_position,
                "velocity": [0.0, 7.5, 0.0],
                "frame": "ICRF_EC",
            }
        )
        result = self.eclipse_finder.execute(
            frame_graph=self.registry, state=cartesian_state
        )
        self.assertIsInstance(result, EclipseInfo)
        self.assertFalse(
            result.is_eclipsed(),
            msg=f"Date: {random_date} and Position: {random_position} was used.",
        )

    def test_eclipse_with_state_series(self):
        """Test eclipse detection with a StateSeries object."""
        state_series = StateSeries.from_dict(
            {
                "time": {
                    "time_format": "GREGORIAN_DATE",
                    "calendar_date": [
                        "2025-03-17T03:00:00.000",  # Early morning
                        "2025-03-17T15:00:00.000",  # Afternoon
                    ],
                    "time_scale": "UTC",
                },
                "data": [
                    [
                        [7000.0, 250.0, 0.0],
                        [7000.0, 250.0, 0.0],
                    ],  # Same Position, close to the 0 deg longitude.
                    [[1, 7, 0], [0, 7, 1]],  # Velocities
                ],
                "frame": "ITRF",
                "headers": [
                    ["pos_x", "pos_y", "pos_z"],
                    ["vel_x", "vel_y", "vel_z"],
                ],
            }
        )
        result = self.eclipse_finder.execute(
            frame_graph=self.registry, state=state_series
        )
        self.assertIsInstance(result, EclipseInfo)
        self.assertTrue(np.array_equal(result.data[0], [True, False]))

    def test_eclipse_with_position_series(self):
        """Test eclipse detection with a PositionSeries object."""
        position_series = PositionSeries.from_dict(
            {
                "time": {
                    "time_format": "GREGORIAN_DATE",
                    "calendar_date": [
                        "2025-03-17T03:00:00.000",  # early morning
                        "2025-03-17T03:00:00.000",
                        "2025-03-17T03:00:00.000",
                    ],
                    "time_scale": "UTC",
                },
                "data": [
                    [7000.0, 0.0, 0.0],  # 0 deg longitude
                    [-7000.0, 0.0, 0.0],  # 180 deg longitude
                    [0.0, 0.0, 7000.0],  # "above" Earth
                ],
                "frame": "ITRF",
            }
        )
        result = self.eclipse_finder.execute(
            frame_graph=self.registry, state=position_series
        )
        self.assertIsInstance(result, EclipseInfo)
        self.assertTrue(np.array_equal(result.data[0], [True, False, False]))

    def test_invalid_inputs(self):
        """Test invalid input combinations."""
        with self.assertRaises(ValueError):
            self.eclipse_finder.execute(
                frame_graph=self.registry, time=None, position=None, state=None
            )
        time = AbsoluteDate.from_dict(
            {
                "time_format": "GREGORIAN_DATE",
                "calendar_date": "2025-06-25T12:00:00.000",
                "time_scale": "UTC",
            }
        )
        position = Cartesian3DPosition.from_dict(
            {
                "x": 7000.0,
                "y": 0.0,
                "z": 7000.0,
                "frame": "ITRF",
            }
        )
        cartesian_state = CartesianState.from_dict(
            {
                "time": {
                    "time_format": "GREGORIAN_DATE",
                    "calendar_date": "2025-03-17T13:00:00.000",
                    "time_scale": "UTC",
                },
                "position": [7000.0, 0.0, 7000.0],
                "velocity": [0.0, 7.5, 0.0],
                "frame": "ITRF",
            }
        )
        with self.assertRaises(ValueError):
            self.eclipse_finder.execute(
                frame_graph=self.registry,
                time=time,
                position=position,
                state=cartesian_state,
            )

        with self.assertRaises(ValueError):
            self.eclipse_finder.execute(
                frame_graph=self.registry,
                time=None,
                position=position,
                state=cartesian_state,
            )

        with self.assertRaises(ValueError):
            self.eclipse_finder.execute(
                frame_graph=self.registry,
                time=time,
                position=None,
                state=cartesian_state,
            )

    def test_object_inside_earth(self):
        """Test eclipse detection for an object inside the Earth."""
        inside_earth_position = Cartesian3DPosition.from_dict(
            {
                "x": 1000.0,
                "y": 1000.0,
                "z": 1000.0,
                "frame": "ITRF",
            }
        )
        time = AbsoluteDate.from_dict(
            {
                "time_format": "GREGORIAN_DATE",
                "calendar_date": "2025-06-25T12:00:00.000",
                "time_scale": "UTC",
            }
        )
        result = self.eclipse_finder.execute(
            frame_graph=self.registry, time=time, position=inside_earth_position
        ).is_eclipsed()
        expected_result = True  # Object inside Earth is considered eclipsed
        self.assertEqual(result, expected_result)

    def test_object_on_earth_surface(self):
        """Test eclipse detection for an object on the Earth's surface."""
        on_surface_position = Cartesian3DPosition.from_dict(
            {
                "x": -(WGS84_EARTH_POLAR_RADIUS + 1e-6),
                "y": 0.0,
                "z": 0.0,
                "frame": "ICRF_EC",
            }
        )
        time = AbsoluteDate.from_dict(
            {
                "time_format": "GREGORIAN_DATE",
                "calendar_date": "2025-06-25T12:00:00.000",
                "time_scale": "UTC",
            }
        )

        result = self.eclipse_finder.execute(
            frame_graph=self.registry, time=time, position=on_surface_position
        ).is_eclipsed()
        expected_result = False
        self.assertEqual(result, expected_result)


if __name__ == "__main__":
    unittest.main()
