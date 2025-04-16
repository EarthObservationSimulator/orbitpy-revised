"""Unit tests for orbitpy.propagator module."""

import unittest
from orbitpy.propagator import PropagatorFactory, SGP4Propagator


class TestPropagatorFactory(unittest.TestCase):
    """Tests for the PropagatorFactory class."""
    class DummyNewPropagator:
        """A dummy propagator class for testing purposes."""

        def __init__(self, specs):
            self.specs = specs

        @classmethod
        def from_dict(cls, specs):
            return TestPropagatorFactory.DummyNewPropagator(specs)

    def test___init__(self):
        """Test initialization of PropagatorFactory."""
        factory = PropagatorFactory()

        # Test the built-in propagators are registered
        self.assertIn("SGP4_PROPAGATOR", factory._creators) # pylint: disable=protected-access
        self.assertEqual(factory._creators["SGP4_PROPAGATOR"], SGP4Propagator) # pylint: disable=protected-access

    def test_register_propagator(self):
        """Test registering a new propagator in the factory."""
        factory = PropagatorFactory()
        factory.register_propagator(
            "New_Propagator", TestPropagatorFactory.DummyNewPropagator
        )

        # Verify the new propagator is registered
        self.assertIn("New_Propagator", factory._creators) # pylint: disable=protected-access
        self.assertEqual(
            factory._creators["New_Propagator"], # pylint: disable=protected-access
            TestPropagatorFactory.DummyNewPropagator,
        )

        # Verify the built-in propagators remain registered
        self.assertIn("SGP4_PROPAGATOR", factory._creators) # pylint: disable=protected-access
        self.assertEqual(factory._creators["SGP4_PROPAGATOR"], SGP4Propagator) # pylint: disable=protected-access

    def test_get_propagator(self):
        """Test retrieving propagators based on specifications."""
        factory = PropagatorFactory()

        # Register dummy propagator
        factory.register_propagator(
            "New_Propagator", TestPropagatorFactory.DummyNewPropagator
        )

        # Test SGP4 PROPAGATOR retrieval
        specs = {"propagator_type": "SGP4_PROPAGATOR", "step_size": 60}
        sgp4_prop = factory.get_propagator(specs)
        self.assertIsInstance(sgp4_prop, SGP4Propagator)

        # Test DummyNewPropagator retrieval
        specs = {
            "propagator_type": "New_Propagator"
        }  # in practice additional propagator specs shall be present in the dictionary
        new_prop = factory.get_propagator(specs)
        self.assertIsInstance(
            new_prop, TestPropagatorFactory.DummyNewPropagator
        )

    def test_get_propagator_invalid_type(self):
        """Test error handling for invalid propagator type."""
        factory = PropagatorFactory()

        # Test missing propagator_type key
        with self.assertRaises(KeyError):
            factory.get_propagator({})

        # Test unregistered propagator type
        with self.assertRaises(ValueError):
            factory.get_propagator({"propagator_type": "Invalid_Propagator"})
