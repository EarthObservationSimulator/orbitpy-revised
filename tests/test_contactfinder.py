"""Unit tests for orbitpy.contactfinder module.
TODO: Make tests for the ContactFinderFactory class."""

import unittest
import random
from scipy.spatial.transform import Rotation as R
import numpy as np

from eosimutils.base import ReferenceFrame
from eosimutils.time import AbsoluteDateArray
from eosimutils.state import (
    Cartesian3DPosition,
    GeographicPosition,
    Cartesian3DPositionArray,
)
from eosimutils.trajectory import StateSeries, PositionSeries
from eosimutils.framegraph import FrameGraph
from eosimutils.orientation import ConstantOrientation

from orbitpy.contactfinder import ContactInfo, LineOfSightContactFinder, ElevationAwareContactFinder


class TestContactInfo(unittest.TestCase):
    """Unit tests for the ContactInfo class."""

    def setUp(self):
        """Set up test data for ContactInfo."""
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
        self.contact_info = ContactInfo(self.times, self.data)

    def test_init_valid_data(self):
        """Test initialization with valid data."""
        self.assertIsInstance(self.contact_info, ContactInfo)
        self.assertTrue(np.array_equal(self.contact_info.data[0], self.data))
        self.assertEqual(self.contact_info.time, self.times)

    def test_init_invalid_data(self):
        """Test initialization with invalid data."""
        with self.assertRaises(TypeError):
            ContactInfo(self.times, np.array([1, 0, 1, 1]))  # Non-boolean data

    def test_has_contact(self):
        """Test the has_contact method."""
        self.assertTrue(
            self.contact_info.has_contact()
        )  # Check for any contact
        self.assertTrue(self.contact_info.has_contact(0))  # Contact at index 0
        self.assertFalse(
            self.contact_info.has_contact(1)
        )  # No contact at index 1
        self.assertTrue(self.contact_info.has_contact(2))  # Contact at index 2
        self.assertTrue(self.contact_info.has_contact(3))  # Contact at index 3
        self.assertIsNone(self.contact_info.has_contact(4))  # Out of bounds

    def test_contact_intervals(self):
        """Test the contact_intervals method."""
        intervals = self.contact_info.contact_intervals()
        self.assertEqual(len(intervals), 2)
        self.assertEqual(intervals[0], (self.times[0], self.times[0]))
        self.assertEqual(intervals[1], (self.times[2], self.times[3]))


class TestLineOfSightContactFinder(unittest.TestCase):
    """Unit tests for the LineOfSightContactFinder class."""

    def setUp(self):
        """Set up contact finder object."""
        self.los_contact_finder = LineOfSightContactFinder()
        self.registry = (
            FrameGraph()
        )  # ICRF_EC and ITRF frames are automatically registered.

        # Register a new frame at 90 deg offset from ICRF_EC
        ReferenceFrame.add("ICRF_EC_90")
        rot90 = R.from_euler(
            "xyz", [0, 0, 90], degrees=True
        )  # Define 90 deg rotations about Z
        constant_orien_90 = ConstantOrientation(
            rot90,
            ReferenceFrame.get("ICRF_EC"),
            ReferenceFrame.get("ICRF_EC_90"),
        )
        self.registry.add_orientation_transform(constant_orien_90)

    def tearDown(self):
        """Clean up after each test."""
        del self.los_contact_finder
        del self.registry
        ReferenceFrame.delete("ICRF_EC_90")

    def test_fixed_geo_location_same_ref_frame(self):
        """Test line-of-sight detection between two fixed geographic
        positions in the same reference frame.

        The positions are fixed at the same latitude but at opposite longitudes,
        with random elevations within a specified range.
        The expected result is False.
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
        self.assertIsInstance(result, ContactInfo)
        self.assertFalse(
            result.has_contact(),
            msg=f"Latitude: {random_latitude} and \
                Elevation: {random_elevation} was used.",
        )

    def test_fixed_cartesian_location_same_ref_frame(self):
        """Test line-of-sight detection between two fixed Cartesian positions
        in the same (custom) reference frame."""
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
        self.assertIsInstance(result, ContactInfo)
        self.assertFalse(result.has_contact())

    def test_execute_with_position_series_and_geo_position_same_ref_frame(self):
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
                ],
            }
        )
        positions = Cartesian3DPositionArray.from_geographic_positions(
            [
                GeographicPosition(0, 0, 10000),
                GeographicPosition(0, 180, 10000),
                GeographicPosition(90, 0, 10000),
            ]
        ).to_numpy()
        position_series = PositionSeries(times, positions, frame)

        fixed_position = GeographicPosition(0, 0, 0)

        # Execute the contact finder
        result = self.los_contact_finder.execute(
            self.registry, position_series, fixed_position
        )

        # Assert the results
        self.assertIsInstance(result, ContactInfo)
        self.assertEqual(len(result.data[0]), len(positions))
        self.assertTrue(
            np.array_equal(result.data[0], [True, False, False]),
            msg=f"Position: {position_series} and Fixed Position: {fixed_position} was used.",
        )

    def test_execute_with_state_series_and_cartesian_position_different_ref_frame(
        self,
    ):
        """Test LineOfSightContactFinder.execute with a StateSeries object."""
        frame = ReferenceFrame.get("ICRF_EC")
        times = AbsoluteDateArray.from_dict(
            {
                "time_format": "Gregorian_Date",
                "time_scale": "UTC",
                "calendar_date": [
                    "2025-03-17T00:00:00.000",
                    "2025-03-17T12:00:00.000",
                    "2025-03-17T16:00:00.000",
                ],
            }
        )
        # Define position and velocity arrays
        positions = np.array(
            [[7000.0, 0.0, 0.0], [-7000.0, 0.0, 0.0], [0.0, 0.0, 7000.0]]
        )
        velocities = np.array(
            [[0.0, 7.5, 0.0], [0.0, 7.5, 0.0], [0.0, 7.5, 0.0]]
        )

        # Create the StateSeries object with both positions and velocities
        state_series = StateSeries(times, [positions, velocities], frame)

        fixed_position = GeographicPosition(0, 180, 0)

        # Execute the contact finder
        result = self.los_contact_finder.execute(
            self.registry, state_series, fixed_position
        )

        # Assert the results
        self.assertIsInstance(result, ContactInfo)
        self.assertEqual(len(result.data[0]), len(positions))
        self.assertTrue(
            np.array_equal(result.data[0], [True, True, False]),
            msg=(
                f"StateSeries: {state_series.position.to_numpy()} "
                f"and Fixed Position: {fixed_position.itrs_xyz} was used."
            ),
        )

class TestElevationAwareContactFinder(unittest.TestCase):
    """Unit tests for the ElevationAwareContactFinder class."""

    def setUp(self):
        """Set up the ElevationAwareContactFinder object."""
        self.elevation_contact_finder = ElevationAwareContactFinder()
        self.registry = FrameGraph()

    def tearDown(self):
        """Clean up after each test."""
        del self.elevation_contact_finder
        del self.registry

    def test_fixed_positions_with_elevation_constraint(self):
        """Test elevation-aware contact detection between two fixed positions.
        Note that the positions are chosen such that the line-of-sight
        always exists, and the contact is determined solely by the elevation constraint.
        """
        
        min_elevation_angle = 30  # degrees

        observer = Cartesian3DPosition.from_dict(
            {
                "x": 0.0,
                "y": 0.0,
                "z": 7000.0,
                "frame": "ICRF_EC",
            }
        )

        target_x = random.uniform(0, np.tan((90 - min_elevation_angle) * np.pi / 180) * (8000.0 - 7000.0))
        target = Cartesian3DPosition.from_dict(
            {
                "x": target_x,
                "y": 0.0,
                "z": 8000.0,
                "frame": "ICRF_EC",
            }
        )

        result = self.elevation_contact_finder.execute(
            self.registry, observer, target, min_elevation_angle=min_elevation_angle
        )

        self.assertIsInstance(result, ContactInfo)
        self.assertTrue(result.has_contact(), msg=f"target_x: {target_x} was used.")

    def test_no_contact_due_to_elevation_constraint(self):
        """Test when elevation constraint prevents contact.
        Note that the positions are chosen such that the line-of-sight
        always exists, and the lack of contact is determined solely by the elevation constraint.
        """

        min_elevation_angle = 30  # degrees

        observer = Cartesian3DPosition.from_dict(
            {
                "x": 0.0,
                "y": 0.0,
                "z": 7000.0,
                "frame": "ICRF_EC",
            }
        )
        target_y = random.uniform(np.tan((90 - min_elevation_angle) * np.pi / 180) * (8000.0 - 7000.0), 20000.0)
        target = Cartesian3DPosition.from_dict(
            {
                "x": 0.0,
                "y": target_y,
                "z": 8000.0,
                "frame": "ICRF_EC",
            }
        )

        result = self.elevation_contact_finder.execute(
            self.registry, observer, target, min_elevation_angle=min_elevation_angle
        )

        self.assertIsInstance(result, ContactInfo)
        self.assertFalse(result.has_contact(), msg=f"target_y: {target_y} was used.")

    def test_moving_target_with_elevation_constraint(self):
        """Test elevation-aware contact detection with a moving target."""
        min_elevation_angle = random.uniform(0, 90)  # degrees
        observer = Cartesian3DPosition.from_dict(
            {
                "x": 0.0,
                "y": 0.0,
                "z": 7000.0,
                "frame": "ITRF",
            }
        )
        times = AbsoluteDateArray.from_dict(
            {
                "time_format": "Gregorian_Date",
                "time_scale": "UTC",
                "calendar_date": [
                    "2025-03-17T00:00:00.000",
                    "2025-03-17T12:00:00.000",
                    "2025-03-17T16:00:00.000",
                ],
            }
        )
        target_x1 = random.uniform(0, np.tan((90 - min_elevation_angle) * np.pi / 180) * (8000.0 - 7000.0))
        target_x2 = random.uniform(0, np.tan((90 - min_elevation_angle) * np.pi / 180) * (8000.0 - 7000.0))
        target_x3 = random.uniform(np.tan((90 - min_elevation_angle) * np.pi / 180) * (8000.0 - 7000.0), 25000.0)
        positions = np.array(
            [[target_x1, 0.0, 8000.0], 
             [target_x2, 0.0, 8000.0], 
             [target_x3, 0.0, 8000.0]]
        )
        target = PositionSeries(times, positions, ReferenceFrame.get("ITRF"))

        # Minimum elevation angle
        result = self.elevation_contact_finder.execute(
            self.registry, observer, target, min_elevation_angle=min_elevation_angle
        )

        self.assertIsInstance(result, ContactInfo)
        self.assertEqual(len(result.data[0]), len(positions))
        self.assertTrue(
            np.array_equal(result.data[0], [True, True, False]),
            msg=f"min_elevation_angle: {min_elevation_angle}, target_x1: {target_x1}, target_x2: {target_x2}, target_x3: {target_x3} were used.",
        )
