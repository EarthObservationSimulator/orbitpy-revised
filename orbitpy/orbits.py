"""
.. module:: orbitpy.orbits
   :synopsis: Represention of spacecraft orbits.

Collection of classes and functions relating to
represention of spacecraft orbits.
Spacecraft state may also be represented in terms of Cartesian state
by using the :class:orbitpy.position.CartesianState class.
"""

import json
import requests
from typing import Dict, Tuple, Any, Optional

from skyfield.elementslib import (
    osculating_elements_of as skyfield_osculating_elements_of,
)

from astropy.constants import GM_earth as astropy_GM_earth

from orbitpy.position import ReferenceFrame, CartesianState
from orbitpy.time import AbsoluteDate


class OrbitalMeanElementsMessage:
    """Handles an Orbital Mean-Elements Message (OMM)."""

    def __init__(self, omm_json: str):
        """
        Initialize the OMM object with a JSON string.

        Args:
            omm_json (str): The OMM in JSON format.

        Raises:
            ValueError: If the input JSON is invalid or cannot be parsed.
        """
        try:
            self.omm_dict: Dict[str, Any] = json.loads(omm_json)
        except json.JSONDecodeError as e:
            raise ValueError("Invalid JSON format for OMM.") from e

    def get_field_as_str(self, field_name: str) -> str:
        """
        Retrieve the value of a specific field in the OMM as a string.

        Args:
            field_name (str): The name of the field to retrieve.

        Returns:
            str: The value of the field as a string, or None if
                 the field does not exist.
        """
        value = self.omm_dict.get(field_name)
        return str(value) if value is not None else None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the OMM object to a dictionary.

        Returns:
            Dict[str, Any]: The OMM as a dictionary.
        """
        return self.omm_dict

    def to_json(self) -> str:
        """
        Convert the OMM object to a JSON string.

        Returns:
            str: The OMM in JSON format.
        """
        return json.dumps(self.omm_dict, indent=4)

    def get_tle_as_tuple(self) -> Tuple[str, str]:
        """
        Retrieve the two lines of the TLE as a tuple of strings.

        Returns:
            Tuple[str, str]: The two lines of the TLE.

        Raises:
            KeyError: If the TLE lines are not present in the OMM.
        """
        try:
            tle_line1 = self.omm_dict["TLE_LINE1"]
            tle_line2 = self.omm_dict["TLE_LINE2"]
            return tle_line1, tle_line2
        except KeyError as e:
            raise KeyError("TLE lines are missing in the OMM.") from e


class SpaceTrackAPI:
    """
    A class to interface with Space-Track.org and retrieve satellite orbit data
    *created* before and closest to a specified target date. Note that the data
    has some latency from the time of the satellite's state measurement.

    *CREATION_DATE* is not the same as *EPOCH* in the OMM.

    Initialize SpaceTrackAPI instance with credentials from a JSON file
    in the following format:
    {
        "username": "xxxx",
        "password": "xxxx"
    }
    """

    BASE_URL = "https://www.space-track.org"

    def __init__(self, credentials_filepath: str):
        """
        Initialize the SpaceTrackAPI instance with credentials.

        Args:
            credentials_filepath (str): Path to the JSON file containing
                                        Space-Track credentials.

        Raises:
            ValueError: If the credentials file is missing required fields.
        """
        try:
            with open(credentials_filepath, "r", encoding="utf8") as file:
                credentials = json.load(file)
            self.username = credentials.get("username")
            self.password = credentials.get("password")

            if not self.username or not self.password:
                raise ValueError(
                    "Credentials file must contain 'username' and "
                    "'password' fields."
                )

            self.session: Optional[requests.Session] = None
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Credentials file not found: {credentials_filepath}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON format in credentials file:"
                f"{credentials_filepath}"
            ) from exc

    def login(self) -> None:
        """
        Log in to Space-Track.org using the provided credentials.

        Raises:
            RuntimeError: If the login request fails.
        """
        login_url = f"{self.BASE_URL}/ajaxauth/login"
        payload = {"identity": self.username, "password": self.password}

        self.session = requests.Session()
        response = self.session.post(login_url, data=payload)

        if response.status_code == 200:
            print("Login successful.")
        else:
            raise RuntimeError(
                f"Login failed with status code {response.status_code}:"
                f"{response.text}"
            )

    def get_closest_omm(
        self, norad_id: int, target_datetime: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve the closest available OMM data *created* before the
        specified target datetime for the given satellite.

        Args:
            norad_id (int): NORAD catalog ID of the satellite.
            target_datetime (str): Target datetime in ISO 8601 format
                                   (e.g., "2024-04-08T19:28:18").

        Returns:
            Optional[Dict[str, Any]]: The OMM data as a dictionary,
                                      or None if no data is found.

        Raises:
            RuntimeError: If the request fails or the
                          session is not initialized.
        """
        if not self.session:
            raise RuntimeError("Session not initialized. Please login first.")

        omm_url = (
            f"{self.BASE_URL}/basicspacedata/query/class/omm/"
            + f"NORAD_CAT_ID/{norad_id}/CREATION_DATE/"
            + f"<{target_datetime}/sorderby/EPOCH%20desc/"
            + "limit/1/format/json"
        )

        response = self.session.get(omm_url)

        if response.status_code == 200:
            closest_omm = response.json()
            if closest_omm:
                return closest_omm[0]  # Return the first OMM in the list
            else:
                print(
                    f"OMM not found for NORAD ID {norad_id}"
                    f"at {target_datetime}."
                    "It is possible the satellite has been launched after the "
                    "specified target date-time."
                )
                return None
        else:
            raise RuntimeError(
                f"Failed to retrieve OMM data for satellite with NORAD ID"
                f"{norad_id}: {response.status_code} - {response.text}"
            )

    def logout(self) -> None:
        """
        Log out and clear the session.

        Raises:
            RuntimeError: If the session is not initialized.
        """
        if not self.session:
            raise RuntimeError("Session not initialized.")
        self.session.cookies.clear()
        self.session = None
        print("Logged out successfully.")


class OsculatingElements:
    """
    Represents the state in terms of osculating (instantaneous)
    Keplerian elements in a specified inertial frame.

    - Time
    - Semi-major axis
    - Eccentricity
    - Inclination
    - Right Ascension of the Ascending Node
    - Argument of Perigee
    - True Anomaly
    - Inertial Frame
    """

    def __init__(
        self,
        time: AbsoluteDate,
        semi_major_axis: float,
        eccentricity: float,
        inclination: float,
        raan: float,
        arg_of_perigee: float,
        true_anomaly: float,
        inertial_frame: ReferenceFrame,
    ) -> None:
        """
        Initialize the `OsculatingElements` object.

        Args:
            time (AbsoluteDate): The epoch of the state.
            semi_major_axis (float): Semi-major axis in kilometers.
            eccentricity (float): Eccentricity (dimensionless).
            inclination (float): Inclination in degrees.
            raan (float): Right Ascension of the Ascending Node
                          (RAAN) in degrees.
            arg_of_perigee (float): Argument of Perigee in degrees.
            true_anomaly (float): True Anomaly in degrees.
            inertial_frame (ReferenceFrame): The inertial reference frame.

        Raises:
            ValueError: If the inertial_frame is not ReferenceFrame.GCRF.
        """
        if inertial_frame != ReferenceFrame.GCRF:
            raise ValueError("Only GCRF inertial reference frame is supported.")

        self.time = time
        self.semi_major_axis = semi_major_axis
        self.eccentricity = eccentricity
        self.inclination = inclination
        self.raan = raan
        self.arg_of_perigee = arg_of_perigee
        self.true_anomaly = true_anomaly
        self.inertial_frame = inertial_frame

    @staticmethod
    def from_dict(dict_in: Dict[str, Any]) -> "OsculatingElements":
        """
        Construct a `OsculatingElements` object from a dictionary.

        Args:
            dict_in (dict): Dictionary containing the state information.
                The dictionary should contain the following key-value pairs:
                - "time" (dict): Dictionary with the date-time information.
                        See :class:`orbitpy.util.AbsoluteDate.from_dict()`.
                - "semi_major_axis" (float): Semi-major axis in kilometers.
                - "eccentricity" (float): Eccentricity (dimensionless).
                - "inclination" (float): Inclination in degrees.
                - "raan" (float): Right Ascension of the Ascending Node 
                                  (RAAN) in degrees.
                - "arg_of_perigee" (float): Argument of Perigee in degrees.
                - "true_anomaly" (float): True Anomaly in degrees.
                - "inertial_frame" (str): The inertial reference frame.

        Returns:
            OsculatingElements: The `OsculatingElements` state object.

        Raises:
            ValueError: If the inertial_frame isn't :class:`ReferenceFrame.GCRF`
        """
        time = AbsoluteDate.from_dict(dict_in["time"])
        inertial_frame = ReferenceFrame.get(dict_in["inertial_frame"])
        if inertial_frame != ReferenceFrame.GCRF:
            raise ValueError("Only GCRF inertial reference frame is supported.")
        return OsculatingElements(
            time=time,
            semi_major_axis=dict_in["semi_major_axis"],
            eccentricity=dict_in["eccentricity"],
            inclination=dict_in["inclination"],
            raan=dict_in["raan"],
            arg_of_perigee=dict_in["arg_of_perigee"],
            true_anomaly=dict_in["true_anomaly"],
            inertial_frame=inertial_frame,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the `OsculatingElements` object to a dictionary.

        Returns:
            dict: Dictionary containing the state information.
        """
        return {
            "time": self.time.to_dict(),
            "semi_major_axis": self.semi_major_axis,
            "eccentricity": self.eccentricity,
            "inclination": self.inclination,
            "raan": self.raan,
            "arg_of_perigee": self.arg_of_perigee,
            "true_anomaly": self.true_anomaly,
            "inertial_frame": self.inertial_frame.value,
        }

    @staticmethod
    def from_cartesian_state(
        cartesian_state: CartesianState,
        gm_body_km3_s2: Optional[float] = astropy_GM_earth.value * 1e-9,
    ) -> "OsculatingElements":
        """
        Initialize an OsculatingElements object from a CartesianState object.

        Args:
            cartesian_state (CartesianState): Cartesian state of the spacecraft.
            gm_body_km3_s2 (float, optional): Gravitational parameter 
                                              in km^3/s^2.
                                              Defaults to GM_earth.

        Returns:
            OsculatingElements: The osculating elements derived from 
                                the Cartesian state.

        Raises:
            ValueError: If the CartesianState frame is not 
                        :class:`ReferenceFrame.GCRF`.
        """
        if cartesian_state.frame != ReferenceFrame.GCRF:
            raise ValueError("Only GCRF is supported.")

        skyfield_position = cartesian_state.to_skyfield_gcrf_position()

        #  The reference frame by default is the ICRF.
        # And GCRF is not rotated with respect to ICRF.
        elements = skyfield_osculating_elements_of(
            skyfield_position, reference_frame=None, gm_km3_s2=gm_body_km3_s2
        )

        # Create and return the OsculatingElements object
        return OsculatingElements(
            time=cartesian_state.time,
            semi_major_axis=elements.semi_major_axis.km,
            eccentricity=elements.eccentricity,
            inclination=elements.inclination.degrees,
            raan=elements.longitude_of_ascending_node.degrees,
            arg_of_perigee=elements.argument_of_periapsis.degrees,
            true_anomaly=elements.true_anomaly.degrees,
            inertial_frame=ReferenceFrame.GCRF,
        )
