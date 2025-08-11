"""Unit tests for orbitpy.orbits module."""

import unittest
import random
import numpy as np

from eosimutils.base import ReferenceFrame
from eosimutils.time import AbsoluteDate
from eosimutils.state import (
    Cartesian3DPosition,
    Cartesian3DVelocity,
    CartesianState,
)

from orbitpy.orbits import (
    TwoLineElementSet,
    OrbitalMeanElementsMessage,
    OsculatingElements,
    OrbitFactory,
    OrbitType,
)


class TestOrbitFactory(unittest.TestCase):
    """Unit tests for the OrbitFactory class."""

    class DummyOrbit:
        def __init__(self, specs):
            self.specs = specs

        @classmethod
        def from_dict(cls, specs):
            return TestOrbitFactory.DummyOrbit(specs)

    def setUp(self):
        # Clear registry before each test to avoid side effects
        OrbitFactory._registry.clear()  # pylint: disable=protected-access
        # Register built-in orbits for tests
        OrbitFactory.register_type(OrbitType.TWO_LINE_ELEMENT_SET.value)(
            TwoLineElementSet
        )
        OrbitFactory.register_type(
            OrbitType.ORBITAL_MEAN_ELEMENTS_MESSAGE.value
        )(
            OrbitalMeanElementsMessage
        )  # pylint: disable=line-too-long
        OrbitFactory.register_type(OrbitType.OSCULATING_ELEMENTS.value)(
            OsculatingElements
        )
        OrbitFactory.register_type(OrbitType.CARTESIAN_STATE.value)(
            CartesianState
        )
        self.tle_dict = {
            "orbit_type": OrbitType.TWO_LINE_ELEMENT_SET.value,
            "TLE_LINE0": "0 LANDSAT 9",
            "TLE_LINE1": "1 49260U 21088A   25106.07240456  .00000957  00000-0  22241-3 0  9997",
            "TLE_LINE2": "2 49260  98.1921 177.4890 0001161  87.5064 272.6267 14.57121096188801",
        }

    def test_from_dict_tle(self):
        """Test retrieving a TwoLineElementSet orbit object using from_dict."""
        orbit = OrbitFactory.from_dict(self.tle_dict)
        self.assertIsInstance(orbit, TwoLineElementSet)
        self.assertEqual(orbit.line0, self.tle_dict["TLE_LINE0"])
        self.assertEqual(orbit.line1, self.tle_dict["TLE_LINE1"])
        self.assertEqual(orbit.line2, self.tle_dict["TLE_LINE2"])

    def test_from_dict_invalid_type(self):
        """Test error handling for an invalid orbit type using from_dict."""
        invalid_dict = {"orbit_type": "INVALID_TYPE"}
        with self.assertRaises(ValueError) as context:
            OrbitFactory.from_dict(invalid_dict)
        self.assertIn(
            'Orbit type "INVALID_TYPE" is not registered.',
            str(context.exception),
        )

    def test_from_dict_missing_type(self):
        """Test error handling for a missing orbit type key using from_dict."""
        missing_type_dict = {"TLE_LINE0": "0 TEST SATELLITE"}
        with self.assertRaises(KeyError) as context:
            OrbitFactory.from_dict(missing_type_dict)
        self.assertIn(
            'Orbit type key "orbit_type" not found in specifications dictionary.',
            str(context.exception),
        )

    def test_from_dict_cartesian_state(self):
        """Test retrieving a CartesianState orbit object using from_dict."""
        cartesian_state_dict = {
            "orbit_type": OrbitType.CARTESIAN_STATE.value,
            "time": {
                "time_format": "Gregorian_Date",
                "calendar_date": "2025-04-16T12:00:00",
                "time_scale": "utc",
            },
            "position": [7000.0, 0.0, 0.0],
            "velocity": [0.0, 7.546, 0.0],
            "frame": "ICRF_EC",
        }
        orbit = OrbitFactory.from_dict(cartesian_state_dict)
        self.assertIsInstance(orbit, CartesianState)
        self.assertTrue(
            (orbit.to_numpy() == [7000.0, 0.0, 0.0, 0.0, 7.546, 0.0]).all()
        )
        self.assertEqual(orbit.frame.to_string(), "ICRF_EC")

    def test_register_type(self):
        """Test registering a new orbit type using register_type."""
        OrbitFactory.register_type("Dummy_Orbit")(TestOrbitFactory.DummyOrbit)
        dummy_dict = {"orbit_type": "Dummy_Orbit", "foo": "bar"}
        orbit = OrbitFactory.from_dict(dummy_dict)
        self.assertIsInstance(orbit, TestOrbitFactory.DummyOrbit)
        self.assertEqual(orbit.specs["foo"], "bar")


class TestTwoLineElementSet(unittest.TestCase):
    """Unit tests for the TwoLineElementSet class."""

    def setUp(self):
        """Set up test data for TwoLineElementSet."""
        self.line0 = "0 LANDSAT 9"
        self.line1 = "1 49260U 21088A   25106.07240456  .00000957  00000-0  22241-3 0  9997"
        self.line2 = "2 49260  98.1921 177.4890 0001161  87.5064 272.6267 14.57121096188801"
        self.tle_dict = {
            "TLE_LINE0": self.line0,
            "TLE_LINE1": self.line1,
            "TLE_LINE2": self.line2,
        }

    def test_initialization(self):
        """Test initialization of TwoLineElementSet."""
        tle = TwoLineElementSet(self.line0, self.line1, self.line2)
        self.assertEqual(tle.line0, self.line0)
        self.assertEqual(tle.line1, self.line1)
        self.assertEqual(tle.line2, self.line2)

    def test_from_dict(self):
        """Test constructing TwoLineElementSet from a dictionary."""
        tle = TwoLineElementSet.from_dict(self.tle_dict)
        self.assertEqual(tle.line0, self.line0)
        self.assertEqual(tle.line1, self.line1)
        self.assertEqual(tle.line2, self.line2)

    def test_to_dict(self):
        """Test converting TwoLineElementSet to a dictionary."""
        tle = TwoLineElementSet(self.line0, self.line1, self.line2)
        tle_dict = tle.to_dict()
        self.assertEqual(tle_dict["TLE_LINE0"], self.line0)
        self.assertEqual(tle_dict["TLE_LINE1"], self.line1)
        self.assertEqual(tle_dict["TLE_LINE2"], self.line2)

    def test_get_tle_as_tuple(self):
        """Test retrieving the TLE as a tuple of strings."""
        tle = TwoLineElementSet.from_dict(self.tle_dict)
        tle_tuple = tle.get_tle_as_tuple()
        self.assertEqual(tle_tuple, (self.line1, self.line2))

    def test_get_tle_as_tuple_missing_lines(self):
        """Test error handling when TLE lines are missing."""
        tle = TwoLineElementSet(self.line0, None, self.line2)
        with self.assertRaises(ValueError) as context:
            tle.get_tle_as_tuple()
        self.assertIn("TLE lines are missing.", str(context.exception))


class TestOrbitalMeanElementsMessage(unittest.TestCase):
    """Unit tests for the OrbitalMeanElementsMessage class."""

    def setUp(self):
        """Set up a valid OMM JSON string for testing."""
        self.valid_omm_json = """
        {
            "CCSDS_OMM_VERS": "2.0",
            "COMMENT": "GENERATED VIA SPACE-TRACK.ORG API",
            "CREATION_DATE": "2024-01-15T14:16:17",
            "ORIGINATOR": "18 SPCS",
            "OBJECT_NAME": "CYGFM03",
            "OBJECT_ID": "2016-078H",
            "CENTER_NAME": "EARTH",
            "REF_FRAME": "TEME",
            "TIME_SYSTEM": "UTC",
            "MEAN_ELEMENT_THEORY": "SGP4",
            "EPOCH": "2024-01-15T11:50:47.395968",
            "MEAN_MOTION": "15.24443449",
            "ECCENTRICITY": "0.0010251",
            "INCLINATION": "34.9521",
            "RA_OF_ASC_NODE": "177.5021",
            "ARG_OF_PERICENTER": "257.9235",
            "MEAN_ANOMALY": "102.033",
            "EPHEMERIS_TYPE": "0",
            "CLASSIFICATION_TYPE": "U",
            "NORAD_CAT_ID": "41891",
            "ELEMENT_SET_NO": "999",
            "REV_AT_EPOCH": "39282",
            "BSTAR": "0.00051433",
            "MEAN_MOTION_DOT": "0.00012941",
            "MEAN_MOTION_DDOT": "0",
            "TLE_LINE0": "0 CYGFM03",
            "TLE_LINE1": "1 41891U 16078H   24015.49360412  .00012941  00000-0  51433-3 0  9992",
            "TLE_LINE2": "2 41891  34.9521 177.5021 0010251 257.9235 102.0330 15.24443449392827",
            "SEMIMAJOR_AXIS": "6870.594",
            "PERIOD": "94.461",
            "APOAPSIS": "499.502",
            "PERIAPSIS": "485.416",
            "OBJECT_TYPE": "PAYLOAD",
            "DECAYED": "0"
        }
        """
        self.invalid_omm_json = "{INVALID_JSON}"

    def test_initialization_valid_json(self):
        """Test initialization with valid JSON."""
        omm = OrbitalMeanElementsMessage(self.valid_omm_json)
        self.assertIsInstance(omm, OrbitalMeanElementsMessage)
        self.assertEqual(omm.get_field_as_str("OBJECT_NAME"), "CYGFM03")

    def test_initialization_invalid_json(self):
        """Test initialization with invalid JSON."""
        with self.assertRaises(ValueError) as context:
            OrbitalMeanElementsMessage(self.invalid_omm_json)
        self.assertIn("Invalid JSON format for OMM", str(context.exception))

    def test_get_field(self):
        """Test retrieving a specific field."""
        omm = OrbitalMeanElementsMessage(self.valid_omm_json)
        self.assertEqual(omm.get_field_as_str("OBJECT_NAME"), "CYGFM03")
        self.assertEqual(
            omm.get_field_as_str("EPOCH"), "2024-01-15T11:50:47.395968"
        )
        self.assertIsNone(omm.get_field_as_str("NON_EXISTENT_FIELD"))

    def test_to_dict(self):
        """Test converting the OMM to a dictionary."""
        omm = OrbitalMeanElementsMessage(self.valid_omm_json)
        omm_dict = omm.to_dict()
        self.assertIsInstance(omm_dict, dict)
        self.assertEqual(omm_dict["OBJECT_NAME"], "CYGFM03")
        self.assertEqual(omm_dict["EPOCH"], "2024-01-15T11:50:47.395968")

    def test_to_json(self):
        """Test converting the OMM to a JSON string."""
        omm = OrbitalMeanElementsMessage(self.valid_omm_json)
        omm_json = omm.to_json()
        self.assertIsInstance(omm_json, str)
        self.assertIn('"OBJECT_NAME": "CYGFM03"', omm_json)
        self.assertIn('"EPOCH": "2024-01-15T11:50:47.395968"', omm_json)

    def test_get_tle_as_tuple_valid(self):
        """Test retrieving the TLE as a tuple of strings."""
        omm = OrbitalMeanElementsMessage(self.valid_omm_json)
        tle_tuple = omm.get_tle_as_tuple()
        expected_tle = (
            "1 41891U 16078H   24015.49360412  .00012941  00000-0  51433-3 0  9992",  # pylint: disable=line-too-long
            "2 41891  34.9521 177.5021 0010251 257.9235 102.0330 15.24443449392827",  # pylint: disable=line-too-long
        )
        self.assertEqual(tle_tuple, expected_tle)

    def test_get_tle_as_tuple_missing_line(self):
        """Test error handling when one of the TLE lines is missing."""
        missing_tle_omm_json = """
{
            "TLE_LINE1": "1 41891U 16078H   24015.49360412  .00012941  00000-0  51433-3 0  9992"  
        }
        """
        omm = OrbitalMeanElementsMessage(missing_tle_omm_json)
        with self.assertRaises(KeyError) as context:
            omm.get_tle_as_tuple()
        self.assertIn(
            "TLE lines are missing in the OMM", str(context.exception)
        )

    def test_from_dict(self):
        """Test constructing OrbitalMeanElementsMessage from a dictionary."""
        omm_dict = {
            "OBJECT_NAME": "CYGFM03",
            "EPOCH": "2024-01-15T11:50:47.395968",
            "TLE_LINE1": "1 41891U 16078H   24015.49360412  .00012941  00000-0  51433-3 0  9992",  # pylint: disable=line-too-long
            "TLE_LINE2": "2 41891  34.9521 177.5021 0010251 257.9235 102.0330 15.24443449392827",  # pylint: disable=line-too-long
        }

        # Create an OrbitalMeanElementsMessage object from the dictionary
        omm = OrbitalMeanElementsMessage.from_dict(omm_dict)

        # Validate the object
        self.assertIsInstance(omm, OrbitalMeanElementsMessage)
        self.assertEqual(omm.get_field_as_str("OBJECT_NAME"), "CYGFM03")
        self.assertEqual(
            omm.get_field_as_str("TLE_LINE1"),
            "1 41891U 16078H   24015.49360412  .00012941  00000-0  51433-3 0  9992",  # pylint: disable=line-too-long
        )
        self.assertEqual(
            omm.get_field_as_str("TLE_LINE2"),
            "2 41891  34.9521 177.5021 0010251 257.9235 102.0330 15.24443449392827",  # pylint: disable=line-too-long
        )

    def test_from_json(self):
        """Test constructing OrbitalMeanElementsMessage from a JSON string."""
        omm_json = """
        {
            "OBJECT_NAME": "CYGFM03",
            "EPOCH": "2024-01-15T11:50:47.395968",
            "TLE_LINE1": "1 41891U 16078H   24015.49360412  .00012941  00000-0  51433-3 0  9992",
            "TLE_LINE2": "2 41891  34.9521 177.5021 0010251 257.9235 102.0330 15.24443449392827"
        }
        """

        # Create an OrbitalMeanElementsMessage object from the JSON string
        omm = OrbitalMeanElementsMessage.from_json(omm_json)

        # Validate the object
        self.assertIsInstance(omm, OrbitalMeanElementsMessage)
        self.assertEqual(omm.get_field_as_str("OBJECT_NAME"), "CYGFM03")
        self.assertEqual(
            omm.get_field_as_str("EPOCH"), "2024-01-15T11:50:47.395968"
        )
        self.assertEqual(
            omm.get_field_as_str("TLE_LINE1"),
            "1 41891U 16078H   24015.49360412  .00012941  00000-0  51433-3 0  9992",  # pylint: disable=line-too-long
        )
        self.assertEqual(
            omm.get_field_as_str("TLE_LINE2"),
            "2 41891  34.9521 177.5021 0010251 257.9235 102.0330 15.24443449392827",  # pylint: disable=line-too-long
        )


class TestOsculatingElements(unittest.TestCase):
    """Unit tests for the OsculatingElements class."""

    def setUp(self):
        """Set up test data for OsculatingElements."""
        self.time_dict = {
            "time_format": "Gregorian_Date",
            "calendar_date": "2025-03-10T14:30:00",
            "time_scale": "utc",
        }
        self.time = AbsoluteDate.from_dict(self.time_dict)
        self.semi_major_axis = round(
            random.uniform(7000, 9000), 6
        )  # in kilometers
        self.eccentricity = round(random.uniform(0, 1), 6)
        self.inclination = round(random.uniform(0, 180), 6)  # in degrees
        self.raan = round(random.uniform(0, 360), 6)  # in degrees
        self.arg_of_perigee = round(random.uniform(0, 360), 6)  # in degrees
        self.true_anomaly = round(random.uniform(0, 360), 6)  # in degrees
        self.inertial_frame = ReferenceFrame.get("ICRF_EC")

    def test_initialization(self):
        """Test initialization of OsculatingElements."""
        state = OsculatingElements(
            self.time,
            self.semi_major_axis,
            self.eccentricity,
            self.inclination,
            self.raan,
            self.arg_of_perigee,
            self.true_anomaly,
            self.inertial_frame,
        )
        self.assertEqual(state.time, self.time)
        self.assertEqual(state.semi_major_axis, self.semi_major_axis)
        self.assertEqual(state.eccentricity, self.eccentricity)
        self.assertEqual(state.inclination, self.inclination)
        self.assertEqual(state.raan, self.raan)
        self.assertEqual(state.arg_of_perigee, self.arg_of_perigee)
        self.assertEqual(state.true_anomaly, self.true_anomaly)
        self.assertEqual(state.inertial_frame, self.inertial_frame)

    def test_invalid_inertial_frame(self):
        """Test initialization with an invalid inertial frame."""
        with self.assertRaises(ValueError) as context:
            OsculatingElements(
                self.time,
                self.semi_major_axis,
                self.eccentricity,
                self.inclination,
                self.raan,
                self.arg_of_perigee,
                self.true_anomaly,
                ReferenceFrame.get("ITRF"),  # Invalid frame
            )
        self.assertTrue(
            "Only ICRF_EC inertial reference frame is supported."
            in str(context.exception)
        )

    def test_from_dict(self):
        """Test constructing OsculatingElements from a dictionary."""
        dict_in = {
            "time": self.time_dict,
            "semi_major_axis": self.semi_major_axis,
            "eccentricity": self.eccentricity,
            "inclination": self.inclination,
            "raan": self.raan,
            "arg_of_perigee": self.arg_of_perigee,
            "true_anomaly": self.true_anomaly,
            "inertial_frame": "ICRF_EC",
        }
        state = OsculatingElements.from_dict(dict_in)
        self.assertEqual(state.time, self.time)
        self.assertEqual(state.semi_major_axis, self.semi_major_axis)
        self.assertEqual(state.eccentricity, self.eccentricity)
        self.assertEqual(state.inclination, self.inclination)
        self.assertEqual(state.raan, self.raan)
        self.assertEqual(state.arg_of_perigee, self.arg_of_perigee)
        self.assertEqual(state.true_anomaly, self.true_anomaly)
        self.assertEqual(state.inertial_frame, self.inertial_frame)

    def test_from_dict_invalid_frame(self):
        """Test constructing OsculatingElements from a
        dictionary with an invalid frame."""
        dict_in = {
            "time": self.time_dict,
            "semi_major_axis": self.semi_major_axis,
            "eccentricity": self.eccentricity,
            "inclination": self.inclination,
            "raan": self.raan,
            "arg_of_perigee": self.arg_of_perigee,
            "true_anomaly": self.true_anomaly,
            "inertial_frame": "ITRF",  # Invalid frame
        }
        with self.assertRaises(ValueError) as context:
            OsculatingElements.from_dict(dict_in)
        self.assertTrue(
            "Only ICRF_EC inertial reference frame is supported."
            in str(context.exception)
        )

    def test_to_dict(self):
        """Test converting OsculatingElements to a dictionary."""
        state = OsculatingElements(
            self.time,
            self.semi_major_axis,
            self.eccentricity,
            self.inclination,
            self.raan,
            self.arg_of_perigee,
            self.true_anomaly,
            self.inertial_frame,
        )
        dict_out = state.to_dict()
        self.assertEqual(dict_out["time"], self.time.to_dict())
        self.assertEqual(dict_out["semi_major_axis"], self.semi_major_axis)
        self.assertEqual(dict_out["eccentricity"], self.eccentricity)
        self.assertEqual(dict_out["inclination"], self.inclination)
        self.assertEqual(dict_out["raan"], self.raan)
        self.assertEqual(dict_out["arg_of_perigee"], self.arg_of_perigee)
        self.assertEqual(dict_out["true_anomaly"], self.true_anomaly)
        self.assertEqual(
            dict_out["inertial_frame"], self.inertial_frame.to_string()
        )

    def test_from_cartesian_state(self):
        """Test constructing OsculatingElements from a CartesianState object."""
        # Create a CartesianState object
        # velocity is chosen to make it a circular orbit, 90 deg inclination
        position = Cartesian3DPosition(
            7000.0, 0.0, 0.0, ReferenceFrame.get("ICRF_EC")
        )
        velocity = Cartesian3DVelocity(
            0.0, 0.0, 7.54605329011, ReferenceFrame.get("ICRF_EC")
        )
        cartesian_state = CartesianState(
            self.time, position, velocity, ReferenceFrame.get("ICRF_EC")
        )

        # Convert to OsculatingElements
        osculating_elements = OsculatingElements.from_cartesian_state(
            cartesian_state
        )

        # Validate the OsculatingElements object
        self.assertEqual(osculating_elements.time, cartesian_state.time)
        self.assertEqual(
            osculating_elements.inertial_frame, ReferenceFrame.get("ICRF_EC")
        )
        self.assertAlmostEqual(osculating_elements.eccentricity, 0.0, places=5)
        self.assertAlmostEqual(osculating_elements.inclination, 90.0)
        self.assertAlmostEqual(osculating_elements.raan, 0.0)
        self.assertAlmostEqual(osculating_elements.arg_of_perigee, 0.0)
        self.assertAlmostEqual(osculating_elements.true_anomaly, 0.0)

    def test_to_cartesian_state(self):
        """Test converting OsculatingElements to a CartesianState object."""
        # Create an OsculatingElements object
        osculating_elements = OsculatingElements(
            time=self.time,
            semi_major_axis=7000.0,  # in kilometers
            eccentricity=0.0,  # Circular orbit
            inclination=90.0,  # Polar orbit
            raan=0.0,  # Right Ascension of Ascending Node
            arg_of_perigee=0.0,  # Argument of Perigee
            true_anomaly=0.0,  # True Anomaly
            inertial_frame=ReferenceFrame.get("ICRF_EC"),
        )

        # Convert to CartesianState
        cartesian_state = osculating_elements.to_cartesian_state()

        # Validate the CartesianState object
        self.assertIsInstance(cartesian_state, CartesianState)
        self.assertEqual(cartesian_state.time, osculating_elements.time)
        self.assertEqual(cartesian_state.frame, ReferenceFrame.get("ICRF_EC"))

        # Validate position and velocity
        position = cartesian_state.position.to_numpy()
        velocity = cartesian_state.velocity.to_numpy()

        # Expected position and velocity for a circular polar orbit
        expected_position = [7000.0, 0.0, 0.0]  # in kilometers
        expected_velocity = [0.0, 0.0, 7.546]  # in kilometers per second

        # Assert position and velocity values
        np.testing.assert_almost_equal(position, expected_position, decimal=3)
        np.testing.assert_almost_equal(velocity, expected_velocity, decimal=3)

    def test_cartesian_to_osculating_and_back(self):
        """Test converting CartesianState to OsculatingElements and back to CartesianState."""
        # Generate random CartesianState
        position = Cartesian3DPosition(
            *np.random.uniform(-10000, 10000, size=3),
            ReferenceFrame.get("ICRF_EC")
        )
        velocity = Cartesian3DVelocity(
            *np.random.uniform(-10, 10, size=3), ReferenceFrame.get("ICRF_EC")
        )
        cartesian_state_original = CartesianState(
            self.time, position, velocity, ReferenceFrame.get("ICRF_EC")
        )

        # Convert to OsculatingElements
        osculating_elements = OsculatingElements.from_cartesian_state(
            cartesian_state_original
        )

        # Convert back to CartesianState
        cartesian_state_converted = osculating_elements.to_cartesian_state()

        # Validate that the original and converted CartesianState are approximately equal
        np.testing.assert_almost_equal(
            cartesian_state_original.position.to_numpy(),
            cartesian_state_converted.position.to_numpy(),
            decimal=3,
        )
        np.testing.assert_almost_equal(
            cartesian_state_original.velocity.to_numpy(),
            cartesian_state_converted.velocity.to_numpy(),
            decimal=3,
        )
        self.assertEqual(
            cartesian_state_original.frame, cartesian_state_converted.frame
        )
        self.assertEqual(
            cartesian_state_original.time, cartesian_state_converted.time
        )


if __name__ == "__main__":
    unittest.main()
