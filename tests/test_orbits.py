"""Unit tests for orbitpy.orbits module."""

import unittest
from orbitpy.orbits import OrbitalMeanElementsMessage


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
            "1 41891U 16078H   24015.49360412  .00012941  00000-0  51433-3 0  9992", # pylint: disable=line-too-long
            "2 41891  34.9521 177.5021 0010251 257.9235 102.0330 15.24443449392827", # pylint: disable=line-too-long
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


if __name__ == "__main__":
    unittest.main()
