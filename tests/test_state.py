"""Unit tests for orbitpy.state module."""

import unittest
import random
import numpy as np

from orbitpy.position import Cartesian3DPosition, Cartesian3DVelocity
from orbitpy.time import AbsoluteDate
from orbitpy.base import ReferenceFrame
from orbitpy.state import CartesianState


class TestCartesianState(unittest.TestCase):
    """Test the CartesianState class."""

    def setUp(self):
        self.time_dict = {
            "time_format": "Gregorian_Date",
            "year": 2025,
            "month": 3,
            "day": 10,
            "hour": 14,
            "minute": 30,
            "second": 0.0,
            "time_scale": "utc",
        }
        self.time = AbsoluteDate.from_dict(self.time_dict)

        self.position_dict = {
            "x": round(random.uniform(-1e6, 1e6), 6),
            "y": round(random.uniform(-1e6, 1e6), 6),
            "z": round(random.uniform(-1e6, 1e6), 6),
            "frame": "GCRF",
        }
        self.position = Cartesian3DPosition.from_dict(self.position_dict)

        self.velocity_dict = {
            "vx": round(random.uniform(-1e6, 1e6), 6),
            "vy": round(random.uniform(-1e6, 1e6), 6),
            "vz": round(random.uniform(-1e6, 1e6), 6),
            "frame": "GCRF",
        }
        self.velocity = Cartesian3DVelocity.from_dict(self.velocity_dict)

        self.frame = ReferenceFrame.GCRF

    def test_initialization(self):
        state = CartesianState(
            self.time, self.position, self.velocity, self.frame
        )
        self.assertEqual(state.time, self.time)
        np.testing.assert_array_equal(state.position.coords, self.position.coords)
        np.testing.assert_array_equal(state.velocity.coords, self.velocity.coords)
        self.assertEqual(state.frame, self.frame)

    def test_mismatched_frames(self):
        position_dict = {
            "x": round(random.uniform(-1e6, 1e6), 6),
            "y": round(random.uniform(-1e6, 1e6), 6),
            "z": round(random.uniform(-1e6, 1e6), 6),
            "frame": "GCRF",
        }
        position = Cartesian3DPosition.from_dict(position_dict)

        velocity_dict = {
            "vx": round(random.uniform(-1e6, 1e6), 6),
            "vy": round(random.uniform(-1e6, 1e6), 6),
            "vz": round(random.uniform(-1e6, 1e6), 6),
            "frame": "ITRF",
        }
        velocity = Cartesian3DVelocity.from_dict(velocity_dict)

        with self.assertRaises(ValueError) as context:
            CartesianState(self.time, position, velocity, ReferenceFrame.GCRF)

        self.assertTrue(
            "Velocity frame does not match the provided frame."
            in str(context.exception)
        )

    def test_from_dict(self):
        dict_in = {
            "time": self.time_dict,
            "position": self.position.to_list(),
            "velocity": self.velocity.to_list(),
            "frame": "GCRF",
        }
        state = CartesianState.from_dict(dict_in)
        self.assertEqual(
            state.time.astropy_time.iso, self.time.astropy_time.iso
        )
        np.testing.assert_array_equal(state.position.coords, self.position.coords)
        np.testing.assert_array_equal(state.velocity.coords, self.velocity.coords)
        self.assertEqual(state.frame, self.frame)

    def test_to_dict(self):
        state = CartesianState(
            self.time, self.position, self.velocity, self.frame
        )
        dict_out = state.to_dict()
        self.assertEqual(dict_out["time"], self.time.to_dict())
        self.assertEqual(dict_out["position"], self.position.to_list())
        self.assertEqual(dict_out["velocity"], self.velocity.to_list())
        self.assertEqual(dict_out["frame"], "GCRF")

    def test_to_skyfield_gcrf_position(self):
        # Create a CartesianState object
        state = CartesianState(
            self.time, self.position, self.velocity, self.frame
        )

        # Convert to Skyfield GCRS position
        skyfield_position = state.to_skyfield_gcrf_position()

        # Validate the Skyfield position object
        # Check that the position matches the CartesianState position
        self.assertAlmostEqual(
            skyfield_position.position.km[0], self.position.coords[0], places=6
        )
        self.assertAlmostEqual(
            skyfield_position.position.km[1], self.position.coords[1], places=6
        )
        self.assertAlmostEqual(
            skyfield_position.position.km[2], self.position.coords[2], places=6
        )

        # Check that the velocity matches the CartesianState velocity
        self.assertAlmostEqual(
            skyfield_position.velocity.km_per_s[0], self.velocity.coords[0], places=6
        )
        self.assertAlmostEqual(
            skyfield_position.velocity.km_per_s[1], self.velocity.coords[1], places=6
        )
        self.assertAlmostEqual(
            skyfield_position.velocity.km_per_s[2], self.velocity.coords[2], places=6
        )

        # Check that the time matches the CartesianState time
        self.assertEqual(
            skyfield_position.t.utc_iso(),
            "2025-03-10T14:30:00Z",
        )

    def test_to_skyfield_gcrf_position_invalid_frame(self):
        """Test that ValueError is raised when frame is not GCRF."""
        # Create a CartesianState object with a non-GCRF frame
        position = Cartesian3DPosition(
            self.position.coords[0],
            self.position.coords[1],
            self.position.coords[2],
            ReferenceFrame.ITRF,
        )
        velocity = Cartesian3DVelocity(
            self.velocity.coords[0],
            self.velocity.coords[1],
            self.velocity.coords[2],
            ReferenceFrame.ITRF,
        )
        state = CartesianState(
            self.time, position, velocity, ReferenceFrame.ITRF
        )

        # Check that ValueError is raised
        with self.assertRaises(ValueError) as context:
            state.to_skyfield_gcrf_position()

        self.assertTrue(
            "Only CartesianState object in GCRF frame is supported for "
            "conversion to Skyfield GCRF position." in str(context.exception)
        )
