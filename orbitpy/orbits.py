"""
.. module:: orbitpy.orbits
   :synopsis: Represention of spacecraft orbits.

Collection of classes and functions relating to
represention of spacecraft orbits.
"""

import json
import requests
from typing import Dict, Tuple, Any, Optional


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
                f'Credentials file not found: {credentials_filepath}'
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

        omm_url = f"{self.BASE_URL}/basicspacedata/query/class/omm/" +\
                  f"NORAD_CAT_ID/{norad_id}/CREATION_DATE/" +\
                  f"<{target_datetime}/sorderby/EPOCH%20desc/" +\
                  "limit/1/format/json"

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
