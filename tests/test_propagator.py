"""Unit tests for orbitpy.propagator module."""

import unittest
import math
import numpy as np

from eosimutils.time import AbsoluteDate
from eosimutils.trajectory import StateSeries

from orbitpy.propagator import PropagatorFactory, SGP4Propagator
from orbitpy.orbits import TwoLineElementSet, Sgp4SatrecOrbitalParameters


class TestPropagatorFactory(unittest.TestCase):
    """Tests for the PropagatorFactory class."""

    class DummyNewPropagator:
        """A dummy propagator class for testing purposes."""

        def __init__(self, specs):
            self.specs = specs

        @classmethod
        def from_dict(cls, specs):
            return TestPropagatorFactory.DummyNewPropagator(specs)

    def setUp(self):
        # Clear registry before each test to avoid side effects
        PropagatorFactory._registry.clear()  # pylint: disable=protected-access
        # Register SGP4Propagator for tests
        PropagatorFactory.register_type("SGP4_PROPAGATOR")(SGP4Propagator)

    def test___init__(self):
        """Test initialization of PropagatorFactory."""
        self.assertIn(
            "SGP4_PROPAGATOR",
            PropagatorFactory._registry,  # pylint: disable=protected-access
        )
        self.assertEqual(
            PropagatorFactory._registry[  # pylint: disable=protected-access
                "SGP4_PROPAGATOR"
            ],
            SGP4Propagator,
        )

    def test_register_type(self):
        """Test registering a new propagator in the factory using register_type."""
        PropagatorFactory.register_type("New_Propagator")(
            TestPropagatorFactory.DummyNewPropagator
        )
        self.assertIn(
            "New_Propagator",
            PropagatorFactory._registry,  # pylint: disable=protected-access
        )
        self.assertEqual(
            PropagatorFactory._registry[  # pylint: disable=protected-access
                "New_Propagator"
            ],
            TestPropagatorFactory.DummyNewPropagator,
        )
        # Built-in propagator remains
        self.assertIn(
            "SGP4_PROPAGATOR",
            PropagatorFactory._registry,  # pylint: disable=protected-access
        )
        self.assertEqual(
            PropagatorFactory._registry[  # pylint: disable=protected-access
                "SGP4_PROPAGATOR"
            ],
            SGP4Propagator,
        )

    def test_from_dict(self):
        """Test retrieving propagators based on specifications."""
        PropagatorFactory.register_type("New_Propagator")(
            TestPropagatorFactory.DummyNewPropagator
        )
        specs = {"propagator_type": "SGP4_PROPAGATOR", "step_size": 60}
        sgp4_prop = PropagatorFactory.from_dict(specs)
        self.assertIsInstance(sgp4_prop, SGP4Propagator)
        specs = {"propagator_type": "New_Propagator"}
        new_prop = PropagatorFactory.from_dict(specs)
        self.assertIsInstance(
            new_prop, TestPropagatorFactory.DummyNewPropagator
        )

    def test_from_dict_invalid_type(self):
        """Test error handling for invalid propagator type."""
        with self.assertRaises(KeyError):
            PropagatorFactory.from_dict({})
        with self.assertRaises(ValueError):
            PropagatorFactory.from_dict(
                {"propagator_type": "Invalid_Propagator"}
            )


class TestSGP4Propagator(unittest.TestCase):
    """Unit tests for the SGP4Propagator class."""

    def setUp(self):
        """Set up test data for SGP4Propagator. """

        self.step_size = 60  # seconds
        self.propagator = SGP4Propagator(step_size=self.step_size)

        #Test satellite is Sentinel-1D (norad id: 66315)
        self.tle = TwoLineElementSet(
            line0="0 SENTINEL-1D",
            line1="1 66315U 25251A   25319.50889988  .00000433  00000-0  10101-3 0  9997",
            line2="2 66315  98.1943 324.8830 0000636 321.5378  38.5776 14.59503087  1569",
        )
        self.sgp4_satrec_params = Sgp4SatrecOrbitalParameters.from_dict({
            "epoch": {
                "time_format": "Gregorian_Date",
                "calendar_date": "2025-11-15T12:12:48.949632",
                "time_scale": "utc",
            },
            "inclo": 98.1943,
            "nodeo": 324.883,
            "ecco": 0.0000636,
            "argpo": 321.5378,
            "mo": 38.5776,
            "no_kozai": 14.59503087,
            "bstar": 0.00010101,
            "ndot": 0.00000433,
            "nddot": 0.0,
        })
        self.start_time = AbsoluteDate.from_dict(
            {
                "time_format": "Gregorian_Date",
                "calendar_date": "2025-11-16T12:00:00",
                "time_scale": "utc",
            }
        )
        self.duration_days = 1  # Propagate for 1 days (24 hours)

    def test_execute_with_tle(self):
        """Test the execute method with a TwoLineElementSet orbit."""
        result = self.propagator.execute(
            t0=self.start_time, duration_days=self.duration_days, orbit=self.tle
        )

        # Verify the result is a StateSeries object
        self.assertIsInstance(result, StateSeries)

        # Verify the time array in the StateSeries
        self.assertEqual(
            len(result.time),
            math.floor(self.duration_days * 86400.0 / self.step_size) + 1,
        )
        self.assertEqual(result.time[0], self.start_time)

        # below test has almost equal and not "equal" because of the way the
        # time steps over the specified duration are calculated
        self.assertAlmostEqual(
            result.time[-1].to_spice_ephemeris_time(),
            (
                self.start_time + (self.duration_days * 86400.0)
            ).to_spice_ephemeris_time(),
            places=3,
        )

        # Verify the position and velocity arrays
        self.assertEqual(
            result.data[0].shape, (len(result.time), 3)
        )  # Position array
        self.assertEqual(
            result.data[1].shape, (len(result.time), 3)
        )  # Velocity array

        # Verify that the reference frame is set to ICRF_EC
        self.assertEqual(result.frame, "ICRF_EC")

    def test_execute_invalid_orbit(self):
        """Test the execute method with an invalid orbit type."""
        with self.assertRaises(ValueError) as context:
            self.propagator.execute(
                t0=self.start_time,
                duration_days=self.duration_days,
                orbit="InvalidOrbit",
            )
        self.assertIn(
            "Invalid orbit type for SGP4 propagation", str(context.exception)
        )

    def test_execute_with_sgp4satrecorbitalparameters(self):
        """Test the execute method with a Sgp4SatrecOrbitalParameters object.
        """

        # run propagation with Sgp4SatrecOrbitalParameters
        result = self.propagator.execute(
            t0=self.start_time,
            duration_days=self.duration_days,
            orbit=self.sgp4_satrec_params,
        )

        # Verify the result is a StateSeries object
        self.assertIsInstance(result, StateSeries)

        # Verify the time array in the StateSeries
        self.assertEqual(
            len(result.time),
            math.floor(self.duration_days * 86400.0 / self.step_size) + 1,
        )
        self.assertEqual(result.time[0], self.start_time)

        # below test has almost equal and not "equal" because of the way the
        # time steps over the specified duration are calculated
        self.assertAlmostEqual(
            result.time[-1].to_spice_ephemeris_time(),
            (
                self.start_time + (self.duration_days * 86400.0)
            ).to_spice_ephemeris_time(),
            places=3,
        )

        # Verify the position and velocity arrays
        self.assertEqual(
            result.data[0].shape, (len(result.time), 3)
        )  # Position array
        self.assertEqual(
            result.data[1].shape, (len(result.time), 3)
        )  # Velocity array

        # Verify that the reference frame is set to ICRF_EC
        self.assertEqual(result.frame, "ICRF_EC")

    def test_compare_tle_and_sgp4satrecorbitalparameters_propagation(self):
        """Compare propagation results between TLE and Sgp4SatrecOrbitalParameters
        inputs.
        """

        # run propagation with TLE
        result_tle = self.propagator.execute(
            t0=self.start_time, duration_days=self.duration_days, orbit=self.tle
        )

        # run propagation with Sgp4SatrecOrbitalParameters
        result_sgp4 = self.propagator.execute(
            t0=self.start_time,
            duration_days=self.duration_days,
            orbit=self.sgp4_satrec_params,
        )

        # Verify that both results have the same time array
        self.assertEqual(len(result_tle.time), len(result_sgp4.time))
        for t1, t2 in zip(result_tle.time, result_sgp4.time):
            self.assertEqual(t1, t2)

        # Verify that position and velocity data are nearly equal
        np.testing.assert_allclose(
            result_tle.data[0], result_sgp4.data[0], rtol=1e-3, atol=1e-2
        )  # Position
        np.testing.assert_allclose(
            result_tle.data[1], result_sgp4.data[1], rtol=1e-3, atol=1e-2
        )  # Velocity

if __name__ == "__main__":
    unittest.main()
