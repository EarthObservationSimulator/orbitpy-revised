"""
.. module:: orbitpy.orbits
   :synopsis: Representation of spacecraft orbits.

Collection of classes and functions relating to
represention of spacecraft orbit state.
Spacecraft state may also be represented in terms of Cartesian state
by using the :class:eosimutils.state.CartesianState class.
"""

import json
import requests
from typing import Dict, Tuple, Any, Optional, Type, Callable, Union
import numpy as np
from datetime import datetime

from skyfield.elementslib import (
    osculating_elements_of as skyfield_osculating_elements_of,
    eccentric_anomaly as skyfield_eccentric_anomaly,
    mean_anomaly as skyfield_mean_anomaly,
)
import spiceypy as spice
from sgp4.api import Satrec as Sgp4_Satrec
from sgp4.api import WGS72 as Sgp4_WGS72
from sgp4.conveniences import check_satrec as Sgp4_check_satrec

from eosimutils.base import ReferenceFrame, EnumBase
from eosimutils.state import CartesianState
from eosimutils.time import AbsoluteDate

GM_EARTH = 398600.435507  # km^3/s^2


class OrbitType(EnumBase):
    """Enumeration of supported orbit types."""

    TWO_LINE_ELEMENT_SET = "TWO_LINE_ELEMENT_SET"
    ORBITAL_MEAN_ELEMENTS_MESSAGE = "ORBITAL_MEAN_ELEMENTS_MESSAGE"
    OSCULATING_ELEMENTS = "OSCULATING_ELEMENTS"
    CARTESIAN_STATE = "CARTESIAN_STATE"


class OrbitFactory:
    """Factory class to register and create orbit objects."""

    # Class-level registry for orbit types
    _registry: Dict[str, Type] = {}

    @classmethod
    def register_type(cls, type_name: str) -> Callable[[Type], Type]:
        """
        Decorator to register an orbit class under a type name.
        """

        def decorator(orbit_class: Type) -> Type:
            cls._registry[type_name] = orbit_class
            return orbit_class

        return decorator

    @classmethod
    def from_dict(cls, specs: Dict[str, Any]) -> object:
        """
        Retrieves an instance of the appropriate orbit class based on specifications.

        Args:
            specs (Dict[str, Any]): A dictionary containing orbit specifications.
                Must include a valid orbit type in the "orbit_type" key.

        Returns:
            object: An instance of the appropriate orbit class initialized with
                    the given specifications.

        Raises:
            KeyError: If the "orbit_type" key is missing in the specifications dictionary.
            ValueError: If the specified orbit type is not registered.
        """
        orbit_type_str = specs.get("orbit_type")
        if orbit_type_str is None:
            raise KeyError(
                'Orbit type key "orbit_type" not found in specifications dictionary.'
            )
        orbit_class = cls._registry.get(orbit_type_str)
        if not orbit_class:
            raise ValueError(
                f'Orbit type "{orbit_type_str}" is not registered.'
            )
        return orbit_class.from_dict(specs)


@OrbitFactory.register_type(OrbitType.TWO_LINE_ELEMENT_SET.value)
class TwoLineElementSet:
    """Handles a Two-Line Element Set (TLE).

    Attributes:
        line0 (Optional[str]): The zeroth line of the TLE (optional).
        line1 (str): The first line of the TLE.
        line2 (str): The second line of the TLE.
    """

    def __init__(self, line0: Optional[str], line1: str, line2: str) -> None:
        """
        Initialize the TLE object with two lines.

        Args:
            line0 (Optional[str]): The zeroth line of the TLE (optional).
            line1 (str): The first line of the TLE.
            line2 (str): The second line of the TLE.
        """
        self.line0 = line0
        self.line1 = line1
        self.line2 = line2

    @classmethod
    def from_dict(cls, dict_in: Dict[str, str]) -> "TwoLineElementSet":
        """
        Construct a TLE object from a dictionary.

        Args:
            dict_in (Dict[str, str]): Dictionary containing the TLE lines.
                Expected keys:
                - "TLE_LINE0" (Optional[str]): The zeroth line of the TLE.
                - "TLE_LINE1" (str): The first line of the TLE.
                - "TLE_LINE2" (str): The second line of the TLE.

        Returns:
            TwoLineElementSet: The TLE object.
        """
        line0 = dict_in.get("TLE_LINE0")
        line1 = dict_in["TLE_LINE1"]
        line2 = dict_in["TLE_LINE2"]
        return cls(line0, line1, line2)

    def to_dict(self) -> Dict[str, str]:
        """
        Convert the TLE object to a dictionary.

        Returns:
            Dict[str, str]: The TLE as a dictionary with the following keys:
                - "TLE_LINE0" (Optional[str]): The zeroth line of the TLE.
                - "TLE_LINE1" (str): The first line of the TLE.
                - "TLE_LINE2" (str): The second line of the TLE.
        """
        return {
            "orbit_type": OrbitType.TWO_LINE_ELEMENT_SET.value,
            "TLE_LINE0": self.line0,
            "TLE_LINE1": self.line1,
            "TLE_LINE2": self.line2,
        }

    def get_tle_as_tuple(self) -> Tuple[str, str]:
        """
        Retrieve the two lines of the TLE as a tuple of strings.

        Returns:
            Tuple[str, str]: The two lines of the TLE.

        Raises:
            ValueError: If either TLE line1 or line2 is missing.
        """
        if self.line1 is None or self.line2 is None:
            raise ValueError("TLE lines are missing.")
        return self.line1, self.line2


@OrbitFactory.register_type(OrbitType.ORBITAL_MEAN_ELEMENTS_MESSAGE.value)
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

    @classmethod
    def from_dict(
        cls, omm_dict: Dict[str, Any]
    ) -> "OrbitalMeanElementsMessage":
        """
        Initialize the OMM object with a dictionary.

        Args:
            omm_dict (Dict[str, Any]): The OMM as a dictionary.

        Returns:
            OrbitalMeanElementsMessage: The OMM object.
        """
        # convert the dictionary to a JSON string
        omm_json = json.dumps(omm_dict)
        return cls(omm_json)

    @classmethod
    def from_json(cls, omm_json: str) -> "OrbitalMeanElementsMessage":
        """
        Initialize the OMM object with a JSON string.

        Args:
            omm_json (str): The OMM in JSON format.

        Returns:
            OrbitalMeanElementsMessage: The OMM object.
        """
        return cls(omm_json)

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
        out_dict = {
            "orbit_type": OrbitType.ORBITAL_MEAN_ELEMENTS_MESSAGE.value,
            **self.omm_dict,
        }
        return out_dict

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
            print("Spacetrack login successful.")
        else:
            raise RuntimeError(
                f"Spacetrack login failed with status code {response.status_code}:"
                f"{response.text}"
            )

    def get_closest_omm(
        self, norad_id: int, target_date_time: str, within_days: int = 1
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve the closest available OMM data *created* before the
        specified target datetime for the given satellite.

        Args:
            norad_id (int): NORAD catalog ID of the satellite.
            target_datetime (str): Target datetime in ISO 8601 format
                                   (e.g., "2024-04-08T19:28:18").
            within_days (int): Maximum number of days before the target
                               datetime to consider for the OMM data.
                               Default is 1 day.

        Returns:
            Optional[Dict[str, Any]]: The OMM data as a dictionary,
                                      or None if no data is found.

        Raises:
            RuntimeError: If the request fails or the
                          session is not initialized.
        """
        if not self.session:
            raise RuntimeError("Session not initialized. Please login first.")

        # Validate that target_date_time is a string in the
        # format %Y-%m-%dT%H:%M:%S.%f or %Y-%m-%dT%H:%M:%S
        try:
            # Try parsing with fractional seconds
            tdt_datetime = datetime.strptime(
                target_date_time, "%Y-%m-%dT%H:%M:%S.%f"
            )
        except ValueError:
            try:
                # Fallback to parsing without fractional seconds
                tdt_datetime = datetime.strptime(
                    target_date_time, "%Y-%m-%dT%H:%M:%S"
                )
            except ValueError:
                print(
                    "SpaceTrack: Invalid target_date_time format. "
                    "It should be a string in the format"
                    "'%Y-%m-%dT%H:%M:%S'. E.g., 2024-04-09T01:00:00"
                ) # Can fractional seconds be included?
                return None

        tdt = tdt_datetime.strftime(
            "%Y-%m-%dT%H:%M:%S"
        )  # ensure the format is correct
        omm_url = (
            f"{self.BASE_URL}/basicspacedata/query/class/omm/"
            + f"NORAD_CAT_ID/{norad_id}/CREATION_DATE/"
            + f"<{tdt}/orderby/EPOCH%20desc/"
            + "limit/1/format/json"
        )
        response = self.session.get(omm_url)

        if response.status_code == 200:
            omm_list = response.json()

            if not omm_list:
                print(
                    f"OMM not found for NORAD ID {norad_id}"
                    f" at {target_date_time}."
                    " It is possible the satellite id is wrong"
                    " or has been launched after the "
                    "specified target date-time."
                )
                return None

            closest_omm = omm_list[0]  # The first OMM in the list
            if closest_omm:
                # print(closest_omm)
                retrieved_cd = closest_omm["CREATION_DATE"]
                retrieved_cd_datetime = datetime.strptime(
                    retrieved_cd, "%Y-%m-%dT%H:%M:%S"
                )  # Convert to datetime object

                # Ensure the retrieved CREATION_DATE is before the target date-time
                if retrieved_cd_datetime > tdt_datetime:
                    raise ValueError(
                        f"The retrieved OMM CREATION_DATE {retrieved_cd} is after the "
                        f"target date-time {tdt}. Something is wrong."
                    )

                # Check if the retrieved CREATION_DATE is more than 1 day before the target
                # date-time
                if (tdt_datetime - retrieved_cd_datetime).days > within_days:
                    raise ValueError(
                        f"Retrieved OMM CREATION_DATE {retrieved_cd} is more than {within_days} "
                        f"days before the target date-time {tdt}. Something is wrong."
                    )

                return closest_omm
            else:
                print(
                    f"OMM not found for NORAD ID {norad_id}"
                    f"at {target_date_time}."
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


@OrbitFactory.register_type(OrbitType.OSCULATING_ELEMENTS.value)
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
            ValueError: If the inertial_frame is not ReferenceFrame.get("ICRF_EC").
        """
        if inertial_frame != ReferenceFrame.get("ICRF_EC"):
            raise ValueError(
                "Only ICRF_EC inertial reference frame is supported."
            )

        self.time = time
        self.semi_major_axis = semi_major_axis
        self.eccentricity = eccentricity
        self.inclination = inclination
        self.raan = raan
        self.arg_of_perigee = arg_of_perigee
        self.true_anomaly = true_anomaly
        self.inertial_frame = inertial_frame

    @classmethod
    def from_dict(cls, dict_in: Dict[str, Any]) -> "OsculatingElements":
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
            ValueError: If the inertial_frame isn't :class:`ReferenceFrame.get("ICRF_EC")`
        """
        time = AbsoluteDate.from_dict(dict_in["time"])
        inertial_frame = ReferenceFrame.get(dict_in["inertial_frame"])
        if inertial_frame != ReferenceFrame.get("ICRF_EC"):
            raise ValueError(
                "Only ICRF_EC inertial reference frame is supported."
            )
        return cls(
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
            "orbit_type": OrbitType.OSCULATING_ELEMENTS.value,
            "time": self.time.to_dict(),
            "semi_major_axis": self.semi_major_axis,
            "eccentricity": self.eccentricity,
            "inclination": self.inclination,
            "raan": self.raan,
            "arg_of_perigee": self.arg_of_perigee,
            "true_anomaly": self.true_anomaly,
            "inertial_frame": self.inertial_frame.to_string(),
        }

    @classmethod
    def from_cartesian_state(
        cls,
        cartesian_state: CartesianState,
        gm_body_km3_s2: Optional[float] = GM_EARTH,
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
                        :class:`ReferenceFrame.get("ICRF_EC")`.
        """
        if cartesian_state.frame != ReferenceFrame.get("ICRF_EC"):
            raise ValueError("Only ICRF_EC is supported.")

        skyfield_position = cartesian_state.to_skyfield_gcrf_position()

        # Orientation of ICRF_EC ~ GCRF (of Skyfield).
        # (The reference frame by default in the skyfield_osculating_elements_of
        #  function is the ICRF.)
        elements = skyfield_osculating_elements_of(
            skyfield_position, reference_frame=None, gm_km3_s2=gm_body_km3_s2
        )

        # Spice equivalent. Skyfield is preferred since SPICE does not directly
        # support true anomaly.
        # elements = spice.oscelt(
        #    cartesian_state.to_numpy(),
        #    cartesian_state.time.to_spice_ephemeris_time(),
        #    GM_EARTH
        # )
        # perifocal_distance, eccentricity, inclination, longitude_of_ascending_node, \
        # argument_of_periapsis, mean_anomaly, epoch, gravitational_parameter = elements  # pylint: disable=line-too-long
        # semi_major_axis = perifocal_distance / (1 - eccentricity)
        # true_anomaly = function(mean_anomaly, eccentricity)

        # Create and return the OsculatingElements object
        return cls(
            time=cartesian_state.time,
            semi_major_axis=elements.semi_major_axis.km,
            eccentricity=elements.eccentricity,
            inclination=elements.inclination.degrees,
            raan=elements.longitude_of_ascending_node.degrees,
            arg_of_perigee=elements.argument_of_periapsis.degrees,
            true_anomaly=elements.true_anomaly.degrees,
            inertial_frame=ReferenceFrame.get("ICRF_EC"),
        )

    def to_cartesian_state(
        self, gm_body_km3_s2: Optional[float] = GM_EARTH
    ) -> CartesianState:
        """
        Convert the osculating elements to a CartesianState object.

        Args:
            gm_body_km3_s2 (float, optional): Gravitational parameter
                                              in km^3/s^2.
                                              Defaults to GM_EARTH.

        Returns:
            CartesianState: The Cartesian state derived from the
                             osculating elements.
        """
        # Get the mean anomaly from the true anomaly
        semi_latus_rectum_km = self.semi_major_axis * (1 - self.eccentricity**2)
        eccentric_anomaly_rad = skyfield_eccentric_anomaly(
            np.deg2rad(self.true_anomaly),
            np.asarray(self.eccentricity),
            semi_latus_rectum_km,
        )
        mean_anomaly_rad = skyfield_mean_anomaly(
            eccentric_anomaly_rad, np.asarray(self.eccentricity)
        )

        # Using SPICE since Skyfield does not support conversion to
        # Cartesian coordinates.
        et = self.time.to_spice_ephemeris_time()
        elements = [
            self.semi_major_axis * (1 - self.eccentricity),
            self.eccentricity,
            np.deg2rad(self.inclination),
            np.deg2rad(self.raan),
            np.deg2rad(self.arg_of_perigee),
            mean_anomaly_rad,
            et,
            gm_body_km3_s2,
        ]
        state_vec = spice.conics(elements, et)

        return CartesianState.from_array(
            array_in=state_vec,
            time=self.time,
            frame=self.inertial_frame,
        )

class Sgp4SatrecOrbitalParameters:
    """Container for the orbital elements required by ``sgp4.Satrec``.

    All angular values (``inclo``, ``nodeo``, ``argpo``, ``mo``) are stored
    in **degrees**. ``no_kozai`` is in **revolutions per day**.
    The units are chosen to match those used in the OMM format.

    Note that these are mean Keplerian elements in the True Equator, Mean Equinox (TEME) 
    earth centered inertial frame.

    The epoch is stored as an :class:`~eosimutils.time.AbsoluteDate` object.

    The epoch required for the Satrec object is computed using the function
    :meth:`_derive_epoch_satrec_repr`.

    References:
        - https://pypi.org/project/sgp4/#providing-your-own-elements
    """

    def __init__(
        self,
        epoch: AbsoluteDate,
        ndot: float,
        nddot: float,
        bstar: float,
        inclo: float,
        nodeo: float,
        ecco: float,
        argpo: float,
        mo: float,
        no_kozai: float,
    ) -> None:
        if not isinstance(epoch, AbsoluteDate):
            raise TypeError("epoch must be an AbsoluteDate instance")

        self.epoch = epoch
        self.ndot = float(ndot)
        self.nddot = float(nddot)
        self.bstar = float(bstar)        
        self.inclo = inclo
        self.nodeo = self._wrap_angle_0_360_deg(nodeo)
        self.ecco = float(ecco)
        self.argpo = self._wrap_angle_0_360_deg(argpo)
        self.mo = self._wrap_angle_0_360_deg(mo)
        self.no_kozai = float(no_kozai)
        self.epoch_satrec_repr = self._derive_epoch_satrec_repr(epoch)

    @staticmethod
    def _wrap_angle_0_360_deg(angle: float) -> float:
        """Wrap an angle to the range [0, 360) degrees."""
        try:
            v = float(angle) % 360.0
        except Exception:
            raise TypeError("Angle must be a number")
        # modulo can return negative for some floats in Python; ensure non-negative
        if v < 0.0:
            v += 360.0
        # treat 360 as 0
        if v == 360.0:
            v = 0.0
        return v

    @staticmethod
    def compute_no_kozai_from_semi_major_axis(
        semi_major_axis_km: float, gm_km3_s2: float = GM_EARTH
    ) -> float:
        """Compute the Kozai mean motion (`no_kozai`) from semi-major axis.

        The classical Keplerian mean motion is n = sqrt(mu / a^3) [rad/s].
        This helper returns the value converted to **degrees per minute** to
        match the `Sgp4SatrecOrbitalParameters.no_kozai` convention used in
        this module.

        TODO: Verify that the gravity parameter used in SGP4 can be that corresponding
                to WGS84 and not WGS72. The SGP4 model uses WGS72.

        Args:
            semi_major_axis_km: Semi-major axis in kilometers.
            gm_km3_s2: Gravitational parameter in km^3/s^2 (defaults to Earth).

        Returns:
            float: Mean motion in revolutions per day.
        """
        if semi_major_axis_km <= 0.0:
            raise ValueError("semi_major_axis_km must be positive")

        n_rad_per_s = np.sqrt(gm_km3_s2 / (semi_major_axis_km ** 3))
        # convert rad/s -> deg/min
        n_deg_per_min = np.rad2deg(n_rad_per_s) * 60.0
        # convert deg/min to revolutions-per-day
        n_rev_per_day = n_deg_per_min * 1440.0 / 360.0
        return n_rev_per_day

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "Sgp4SatrecOrbitalParameters":
        """Instantiate the parameter set from a dictionary.

        Args:
            values (Dict[str, Any]): Dictionary containing the orbital parameters.
                Expected keys:
                - "epoch" (Dict[str, Any]): The epoch as an dictionary for AbsoluteDate.from_dict.
                - "ndot" (float): Optional. First time derivative of the mean motion (loaded from the TLE, but otherwise ignored).
                - "nddot" (float): Optional. Second time derivative of the mean motion  (loaded from the TLE, but otherwise ignored).
                - "bstar" (float): Optional. Ballistic drag coefficient B* (1/earth radii).
                - "inclo" (float): Inclination in degrees (0 to 180 degrees).
                - "nodeo" (float): Right Ascension of the Ascending Node in degrees ( 0 to 360 degrees).
                - "ecco" (float): Eccentricity (dimensionless).
                - "argpo" (float): Argument of Perigee in degrees (0 to 360 degrees).
                - "mo" (float): Mean Anomaly in degrees (0 to 360 degrees).
                - Either "no_kozai" (float): Mean Motion in revolutions per day,
                  or "sma" (float): semi-major axis in kilometers. 
                                    If `sma` is provided `no_kozai` will be computed.

        """
        # Determine no_kozai: accept explicit value or compute from semi-major axis
        if "no_kozai" in values and values["no_kozai"] is not None:
            no_kozai_val = float(values["no_kozai"])
        elif "sma" in values and values["sma"] is not None:
            sma = float(values["sma"])
            no_kozai_val = cls.compute_no_kozai_from_semi_major_axis(sma)
        else:
            raise KeyError("Either 'no_kozai' or 'sma' (semi-major axis in km) must be provided")

        return cls(
            epoch=AbsoluteDate.from_dict(values["epoch"]),
            ndot=float(values.get("ndot", 0.0)),
            nddot=float(values.get("nddot", 0.0)),
            bstar=float(values.get("bstar", 0.0)),
            inclo=float(values["inclo"]),
            nodeo=float(values["nodeo"]),
            ecco=float(values["ecco"]),
            argpo=float(values["argpo"]),
            mo=float(values["mo"]),
            no_kozai=no_kozai_val,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the parameters."""
        return {
            "epoch": self.epoch.to_dict(),
            "ndot": self.ndot,
            "nddot": self.nddot,
            "bstar": self.bstar,
            "inclo": self.inclo,
            "nodeo": self.nodeo,
            "ecco": self.ecco,
            "argpo": self.argpo,
            "mo": self.mo,
            "no_kozai": self.no_kozai,
        }

    @staticmethod
    def _derive_epoch_satrec_repr(epoch: AbsoluteDate) ->  float:
        """
        Compute the epoch representation required by the SGP4 `Satrec` class.

        This method calculates the epoch as the number of days since 1949 December 31 00:00 UT.
        The calculation is based on the Julian date of the provided epoch, adjusted by subtracting
        2433281.5 days, as specified in the below webpage.
            https://pypi.org/project/sgp4/#providing-your-own-elements:

        Args:
            epoch (AbsoluteDate): The epoch for which the representation is to be computed.
                It should be an instance of `AbsoluteDate`.

        Returns:
            float: A float representing the epoch in the format required by the SGP4 `Satrec` class.
        """
        julian_date = epoch.to_dict(time_format="JULIAN_DATE", time_scale="UTC")["jd"]
        return julian_date - 2433281.5
    
    def get_sgp4_satrec(self) -> Sgp4_Satrec:
        """Create and return an `sgp4.api.Satrec` object initialized with
        the stored orbital parameters.

        'WGS72' gravity model and 'i' (improved) mode are used by default.
        THe reason for using WGS72 is outlined in:
            https://pypi.org/project/sgp4/#providing-your-own-elements
        
        Returns:
            Sgp4_Satrec: An instance of `sgp4.api.Satrec` initialized with
                         the stored orbital parameters.
        """
        satrec = Sgp4_Satrec()
        no_kozai_rad_per_min = np.deg2rad(self.no_kozai * 360.0) / 1440.0  # rev/day -> rad/min
        satrec.sgp4init(
            Sgp4_WGS72,  # gravity model WGS-72
            'i',         # 'a' = old AFSPC mode, 'i' = improved mode
            0,      # satnum: Satellite number (not used here)
            self.epoch_satrec_repr, # epoch: days since 1949 December 31 00:00 UT
            self.bstar, # bstar: drag coefficient (1/earth radii)
            self.ndot, # ndot: ballistic coefficient (radians/minute^2)
            self.nddot,  # nddot: mean motion 2nd derivative (radians/minute^3)
            self.ecco, # ecco: eccentricity
            np.deg2rad(self.argpo), # argpo: argument of perigee (radians 0..2pi)
            np.deg2rad(self.inclo), # inclo: inclination (radians 0..pi)
            np.deg2rad(self.mo), # mo: mean anomaly (radians 0..2pi)
            no_kozai_rad_per_min,  # no_kozai: mean motion (radians/minute)
            np.deg2rad(self.nodeo), # nodeo: R.A. of ascending node (radians 0..2pi)
        )
        Sgp4_check_satrec(satrec) # validate that the elements are acceptable
        return satrec

    @classmethod
    def tle_lines_to_satrec_orbital_parameters(cls, tle_line1: str, tle_line2: str) -> "Sgp4SatrecOrbitalParameters":
        """Create an instance of `Sgp4SatrecOrbitalParameters` from TLE lines.

        Args:
            tle_line1 (str): The first line of the TLE.
            tle_line2 (str): The second line of the TLE.
        Returns:
            Sgp4SatrecOrbitalParameters: An instance initialized with
                                         parameters extracted from the TLE.
        """
        satrec = Sgp4_Satrec.twoline2rv(tle_line1, tle_line2)
        # Extract epoch as AbsoluteDate
        epoch = AbsoluteDate.from_dict({
            "time_format": "JULIAN_DATE",
            "time_scale": "UTC",
            "jd": satrec.jdsatepoch  + satrec.jdsatepochF,
        })
        return cls(
            epoch=epoch,
            ndot=satrec.ndot,
            nddot=satrec.nddot,
            bstar=satrec.bstar,
            inclo=np.rad2deg(satrec.inclo),
            nodeo=np.rad2deg(satrec.nodeo),
            ecco=satrec.ecco,
            argpo=np.rad2deg(satrec.argpo),
            mo=np.rad2deg(satrec.mo),
            no_kozai=np.rad2deg(satrec.no_kozai)*1440.0/360.0,  # rad/min -> rev/day
        )