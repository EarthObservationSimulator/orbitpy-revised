"""
.. module:: orbitpy.mission
   :synopsis: Module to handle mission initialization and execution.

Collection of classes and functions relating to represention of
mission parameters and execution. The main class is `Mission` which
holds all mission parameters and has functions to execute various
mission analyses such as orbit propagation, eclipse finding, contact
finding, and coverage calculation. A "Mission" object can be
initialized from a dictionary (e.g. loaded from a JSON file) using
the `from_dict` function. The mission analyses can be executed using
the respective `execute_...` functions.

This module is utilized by the `bin/run_mission.py` script to
execute a mission defined in a JSON file.

Base reference frames with which the module works are:
1/ ICRF_EC (Earth-Centered Inertial Frame) as defined in eosimutils.base.ReferenceFrame
2/ ITRF (Earth-Centered Earth-Fixed Frame) as defined in eosimutils.base.ReferenceFrame

The default surface type used for Earth surface representation is WGS84 ellipsoid.
**To do**: Update about the caveats of using WGS84 vs Sphere. For example, if WGS84 is used,
currently cell-based preprocessing in coverage calculations is skipped.

The features currently supported are:
- Mission with one or more spacecraft, each with one or more sensors.
- Orbit propagation using SGP4 propagator.
- Eclipse finding.
- Contact finding with one or more ground stations.
- Automatic retrieval of orbit data (OMM) retrieved from Space-Track.org
  using NORAD ID (at the specified start time of the mission).
- Coverage calculation for point and specular coverage. In case of specular
  coverage, multiple GNSS satellites can be specified as transmitters
  (in addition to the receiver spacecraft with one or more GNSS-R sensors).

The coding style is such that each of the dictionaries within the mission dictionary 
is formatted so that it can be directly converted to the corresponding object using the
`from_dict` function of the respective class. For example, the dictionary for a spacecraft
is such that it can be directly converted to a `Spacecraft` object using the
`Spacecraft.from_dict` function. (An exception is the `Settings` class which contains misscellaneous
mission settings.)
"""

import os
from typing import Dict, Any, Union, List, Optional, Tuple
import numpy as np

from eosimutils.base import ReferenceFrame, SurfaceType, JsonSerializer
from eosimutils.time import AbsoluteDate
from eosimutils.state import Cartesian3DPositionArray, GeographicPositionArray
from eosimutils.framegraph import FrameGraph
from eosimutils.trajectory import StateSeries

from .propagator import PropagatorFactory, SGP4Propagator
from .resources import Spacecraft, GroundStation
from .eclipsefinder import EclipseFinder, EclipseInfo
from .contactfinder import ElevationAwareContactFinder, ContactInfo
from .coveragecalculator import PointCoverage, SpecularCoverage, CoverageType
from .orbits import SpaceTrackAPI, OrbitalMeanElementsMessage
from .specular import get_specular_trajectory


def auto_retrieve_orbit(
    norad_id: int,
    target_date_time: AbsoluteDate,
    space_track_credentials_fp: str,
) -> OrbitalMeanElementsMessage:
    """Automatically retrieve orbit data (OMM) for a satellite using Space-Track.org.

    Args:
        norad_id (int): NORAD ID of the satellite for which to retrieve data.
        target_date_time (AbsoluteDate): Target date and time to find the closest OMM.
        space_track_credentials_fp (str): File path to SpaceTrackAPI credentials.

    Returns:
        OrbitalMeanElementsMessage: The closest available Orbital Mean Elements Message (OMM).
    """
    api = SpaceTrackAPI(space_track_credentials_fp)

    # Log in to Space-Track.org
    api.login()

    # Retrieve the *closest* available OMM data *created* before the
    # specified target datetime for the given satellite.

    target_date_time_str = target_date_time.to_dict(
        time_format="GREGORIAN_DATE", time_scale="UTC"
    )["calendar_date"]

    omm_data = api.get_closest_omm(
        norad_id=norad_id, target_date_time=target_date_time_str, within_days=5
    )

    # Log out from Space-Track.org
    api.logout()

    if omm_data is None:
        raise ValueError(
            f"OMM retrieval failed for NORAD ID {norad_id}"
            f" before {target_date_time_str}."
        )

    orbit = OrbitalMeanElementsMessage.from_dict(omm_data)

    return orbit


def propagate_spacecraft(
    spacecraft: Spacecraft,
    propagator: SGP4Propagator,
    t0: AbsoluteDate,
    duration_days: float,
    space_track_credentials_fp: Optional[str],
) -> StateSeries:
    """
    Propagate a spacecraft's orbit using a specified propagator.
    If the orbit specifications are not provided, the OMM is automatically retrieved
    from Space-Track.org using the provided NORAD ID. The OMM closest to the start time (t0)
    and *created* before the start time is retrieved.

    Args:
        spacecraft (Spacecraft): The spacecraft object to be propagated.
        propagator (SGP4Propagator): The propagator to use for orbit propagation.
        t0 (AbsoluteDate): Start time for propagation.
        duration_days (float): Duration of propagation in days.
        space_track_credentials_fp (Optional[str]): File path to Space-Track.org credentials.
                                    Required if the spacecraft does not have orbit specifications
                                    and the OMM needs to be retrieved using the NORAD ID.
    Returns:
        trajectory (StateSeries): StateSeries object containing the propagated states.
    """
    # determine the orbit to use for propagation
    if spacecraft is None:
        raise ValueError("Spacecraft cannot be None.")
    if spacecraft.orbit is not None:
        orbit = spacecraft.orbit
    elif spacecraft.norad_id is not None:
        orbit = auto_retrieve_orbit(
            spacecraft.norad_id, t0, space_track_credentials_fp
        )
    else:
        raise ValueError(
            "Spacecraft must have either orbit specifications or a NORAD ID for OMM retrieval."
        )

    if not isinstance(propagator, SGP4Propagator):
        raise ValueError("Unsupported propagator type.")

    propagated_states = propagator.execute(
        t0=t0, duration_days=duration_days, orbit=orbit
    )
    return propagated_states


def calculate_gs_contact(
    trajectory: StateSeries,
    ground_station: GroundStation,
    frame_graph: FrameGraph,
) -> ContactInfo:
    """
    Calculate contact periods between a spacecraft and a ground station.

    Args:
        trajectory (StateSeries): Propagated trajectory of the spacecraft.
        ground_station (GroundStation): Ground station object.
        frame_graph (FrameGraph): Frame graph for coordinate transformations.

    Returns:
        contact_info (ContactInfo): ContactInfo object containing contact periods and details.
    """
    contact_finder = ElevationAwareContactFinder()
    contact_info = contact_finder.execute(
        frame_graph=frame_graph,
        observer_state=ground_station.geographic_position.to_cartesian3d_position(),
        target_state=trajectory,
        min_elevation_angle=ground_station.min_elevation_angle_deg,
    )
    return contact_info


class Settings:
    """Class to hold miscellaneous mission settings."""

    def __init__(
        self,
        user_dir: Optional[str] = None,
        coverage_type: Optional[Union[CoverageType, str]] = CoverageType.POINT_COVERAGE,
        specular_radius_km: Optional[float] = None,
        spacetrack_credentials_relative_path: Optional[str] = None,
        surface_type: Optional[SurfaceType] = SurfaceType.WGS84,
    ):
        """
        Args:
            user_dir (Optional[str]): Absolute user directory path.
            coverage_type (Optional[Union[CoverageType, str]]): Type of coverage.
                                                See `orbitpy.coveragecalculator.CoverageType`.
                                                Defaults to POINT_COVERAGE.
            specular_radius_km (Optional[float]): Specular radius in kilometers
                                                (applicable for GNSSR coverage).
            spacetrack_credentials_relative_path (Optional[str]): Relative file path
                                    (relative to the `user_dir`) for Space-Track.org credentials.
                                    The credentials file is a json file with the following format:
                                    {
                                        "username": "xxxx",
                                        "password": "xxxx"
                                    }
            surface_type (Optional[SurfaceType]): Type of Earth surface model to use.
                                                See `eosimutils.base.SurfaceType`.
                                                Defaults to WGS84.

        """
        self.user_dir: str = user_dir
        self.coverage_type = coverage_type
        self.specular_radius_km = specular_radius_km
        self.spacetrack_credentials_relative_path = (
            spacetrack_credentials_relative_path
        )
        self.surface_type = surface_type

    @classmethod
    def from_dict(cls, dict_in: Dict[str, Any]) -> "Settings":
        """Create a Settings object from a dictionary.

        Args:
            dict_in (Dict[str, Any]): Dictionary containing miscellaneous mission settings.
                        Expected keys are:
                - user_dir (Optional[str]): Absolute user directory path.
                - coverage_type (Optional[Union[CoverageType, str]]): Type of coverage. If None, POINT_COVERAGE is used.
                - specular_radius_km (Optional[float]): Specular radius in kilometers.
                - spacetrack_credentials_relative_path (Optional[str]): Relative file path for Space-Track.org credentials.
                - surface_type (Optional[SurfaceType]): Type of Earth surface model to use. If None, WGS84 is used.

        Returns:
            Settings: An instance of the Settings class.
        """
        user_dir = dict_in.get("user_dir", None)
        coverage_type = dict_in.get("coverage_type", None)
        coverage_type = (
            CoverageType.get(coverage_type)
            if coverage_type
            else CoverageType.POINT_COVERAGE
        )
        specular_radius_km = dict_in.get("specular_radius_km", None)
        spacetrack_credentials_relative_path = dict_in.get(
            "spacetrack_credentials_relative_path", None
        )
        surface_type = dict_in.get("surface_type", None)
        surface_type = (
            SurfaceType.get(surface_type)
            if surface_type
            else SurfaceType.WGS84
        )
        return cls(
            user_dir=user_dir,
            coverage_type=coverage_type,
            specular_radius_km=specular_radius_km,
            spacetrack_credentials_relative_path=spacetrack_credentials_relative_path,
            surface_type=surface_type
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the Settings object to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of the Settings object.
        """
        return {
            "user_dir": self.user_dir,
            "coverage_type": (
                self.coverage_type.to_string() if self.coverage_type else None
            ),
            "specular_radius_km": self.specular_radius_km,
            "spacetrack_credentials_relative_path": self.spacetrack_credentials_relative_path,
            "surface_type": (
                self.surface_type.to_string() if self.surface_type else None
            ),
        }
    
class Mission:
    """Class to represent a space mission with spacecraft, sensors, and ground stations."""

    def __init__(
        self,
        start_time: AbsoluteDate,
        duration_days: float,
        spacecrafts: Union[Spacecraft, List[Spacecraft]],
        ground_stations: Optional[Union[GroundStation, List[GroundStation]]],
        propagator: Optional[SGP4Propagator],
        cartesian_spatial_points: Optional[Cartesian3DPositionArray] = None,
        gnss_spacecrafts: Optional[List[Spacecraft]] = None,
        frame_graph: FrameGraph = None,
        settings: Optional[Settings] = None,
    ):
        """
        Args:
            start_time (AbsoluteDate): Start time of the mission.
            duration_days (float): Duration of the mission in days.
            spacecrafts (Union[Spacecraft, List[Spacecraft]]): List of spacecraft in the mission.
            ground_stations (Optional[GroundStation, List[GroundStation]]): List of ground stations.
            propagator (Optional[SGP4Propagator]): Propagator to use for orbit propagation.
            cartesian_spatial_points (Optional[Cartesian3DPositionArray]): Cartesian spatial points 
                                                                            for coverage calculation.
            gnss_spacecrafts (Optional[List[Spacecraft]]): List of GNSS satellites (transmitters)
                                                            (applicable for GNSSR coverage).
            frame_graph (FrameGraph): Frame graph for coordinate transformations.
            settings (Optional[Settings]): Miscellaneous mission settings.
        """
        self.start_time = start_time
        self.duration_days = duration_days
        if not isinstance(spacecrafts, list):
            spacecrafts = [spacecrafts]
        self.spacecrafts = spacecrafts
        if ground_stations is not None and not isinstance(
            ground_stations, list
        ):
            ground_stations = [ground_stations]
        self.ground_stations = ground_stations
        self.propagator = propagator
        self.cartesian_spatial_points = cartesian_spatial_points
        if gnss_spacecrafts is not None:
            if not isinstance(gnss_spacecrafts, list):
                gnss_spacecrafts = [gnss_spacecrafts]
        self.gnss_spacecrafts = gnss_spacecrafts if gnss_spacecrafts else []
        self.frame_graph = (
            frame_graph if frame_graph is not None else FrameGraph()
        )
        self.settings = settings if settings else Settings()

    @staticmethod
    def load_object(other_cls, dict_in: dict, ref_dir: str) -> Optional[object]:
        """Load an object from a dictionary or a JSON file.

        Args:
            other_cls (Type[Any]): The class type to instantiate the object. 
                                    The class should support a 'from_dict' function.
            dict_in (dict): Dictionary containing object data or a reference to a JSON file.
            ref_dir (str): Directory path to resolve relative file paths.

        Returns:
            Any or None: An instance of the specified class, or None if dict_in is None.
        """
        if dict_in is None:
            return None

        if "relative_file_path" in dict_in:
            # Load from JSON file
            file_path = os.path.join(ref_dir, dict_in["relative_file_path"])
            loaded_object = JsonSerializer.load_from_json(other_cls, file_path)
            return loaded_object

        if isinstance(dict_in, list):
            # If dict_in is a list, return a list of objects
            return [other_cls.from_dict(item) for item in dict_in]
        else:
            # Otherwise, return a single object
            return other_cls.from_dict(dict_in)

    @classmethod
    def from_dict(cls, dict_in: Dict[str, Any]) -> "Mission":
        """Create a Mission object from a dictionary.

        Args:
            dict_in (Dict[str, Any]): Dictionary containing mission specifications.
        Returns:
            Mission: An instance of the Mission class.
        """
        # setup mission settings
        settings_dict = dict_in.get("settings", None)
        settings = (
            Settings.from_dict(settings_dict) if settings_dict else Settings()
        )

        # setup mission parameters
        start_time = AbsoluteDate.from_dict(dict_in["start_time"])
        duration_days = dict_in["duration_days"]

        # setup spacecrafts
        spacecrafts_dict_val = dict_in.get("spacecrafts", None)
        spacecrafts = Mission.load_object(
            Spacecraft, spacecrafts_dict_val, settings.user_dir
        )
        
        # setup GNSS (transmitter) spacecrafts (applicable for GNSSR coverage)
        gnss_spacecrafts_dict_val = dict_in.get("gnss_spacecrafts", None)
        gnss_spacecrafts = Mission.load_object(
            Spacecraft, gnss_spacecrafts_dict_val, settings.user_dir
        )

        # setup ground-stations
        ground_stations_dict_val = dict_in.get("ground_stations", None)
        ground_stations = Mission.load_object(
            GroundStation, ground_stations_dict_val, settings.user_dir
        )

        # setup propagator
        propagator_dict = dict_in.get("propagator", None)
        propagator = Mission.load_object(
            PropagatorFactory, propagator_dict, settings.user_dir
        )

        # setup spatial points as a Cartesian3DPositionArray.
        # The input could be either GeographicPositionArray or Cartesian3DPositionArray.
        spatial_points_dict = dict_in.get("spatial_points", None)
        if spatial_points_dict is not None:
            # determine the type of spatial points specified based on the dictionary keys
            if "geographic_array" in spatial_points_dict:
                geographic_points_dict = spatial_points_dict["geographic_array"]
                geographic_points = Mission.load_object(
                    GeographicPositionArray, geographic_points_dict, settings.user_dir
                )
                cartesian_spatial_points = geographic_points.to_cartesian3d_position_array()
            elif "cartesian_3d_array" in spatial_points_dict:
                cartesian_points_dict = spatial_points_dict["cartesian_3d_array"]
                cartesian_spatial_points = Mission.load_object(
                    Cartesian3DPositionArray, cartesian_points_dict, settings.user_dir
                )
            else:
                raise ValueError(
                    "Unsupported spatial points specification."
                )
        else:
            cartesian_spatial_points = None

        # setup the frames and the transformations
        # The spacecrafts need to be setup before this so that their
        # local orbit frames are registered.
        _transform_dict = dict_in.get("frame_transforms", None)
        transform_dict = Mission.load_object(
            dict, _transform_dict, settings.user_dir
        )

        # add reference frames from the user-specified transforms if not already present
        if transform_dict is not None:
            orientation_transforms = transform_dict.get(
                "orientation_transforms", None
            )
            position_transforms = transform_dict.get(
                "position_transforms", None
            )
            if orientation_transforms is not None:
                for ot in orientation_transforms:
                    if ReferenceFrame.get(ot["from"]) is None:
                        ReferenceFrame.add(ot["from"])
                    if ReferenceFrame.get(ot["to"]) is None:
                        ReferenceFrame.add(ot["to"])
            if position_transforms is not None:
                for pt in position_transforms:
                    if ReferenceFrame.get(pt["from_frame"]) is None:
                        ReferenceFrame.add(pt["from_frame"])
                    if ReferenceFrame.get(pt["to_frame"]) is None:
                        ReferenceFrame.add(pt["to_frame"])
        # create the frame graph using the user-specified transforms
        if transform_dict is not None:
            frame_graph = FrameGraph.from_dict(transform_dict, set_inverse=True)
        else:
            frame_graph = FrameGraph()

        return cls(
            start_time=start_time,
            duration_days=duration_days,
            spacecrafts=spacecrafts,
            gnss_spacecrafts=gnss_spacecrafts,
            ground_stations=ground_stations,
            propagator=propagator,
            cartesian_spatial_points=cartesian_spatial_points,
            frame_graph=frame_graph,
            settings=settings,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the Mission object to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of the Mission object.
        """
        return {
            "start_time": self.start_time.to_dict(),
            "duration_days": self.duration_days,
            "spacecrafts": [sc.to_dict() for sc in self.spacecrafts],
            "gnss_spacecrafts": (
                [gnss_sc.to_dict() for gnss_sc in self.gnss_spacecrafts]
                if self.gnss_spacecrafts
                else None
            ),
            "ground_stations": (
                [gs.to_dict() for gs in self.ground_stations]
                if self.ground_stations
                else None
            ),
            "propagator": (
                self.propagator.to_dict() if self.propagator else None
            ),
            "spatial_points": (
                { "cartesian_3d_array": self.cartesian_spatial_points.to_dict()} if self.cartesian_spatial_points else None
            ),
            "settings": self.settings.to_dict() if self.settings else None,
        }

    def execute_propagation(self) -> Tuple[List[Dict[str, Union[str, StateSeries]]], Optional[List[Dict[str, Union[str, StateSeries]]]]]:
        """Propagate all spacecrafts in the mission using the specified propagator.

        Returns:
            Tuple[List[Dict[str, Union[str, StateSeries]]], Optional[List[Dict[str, Union[str, StateSeries]]]]]:
                A tuple (results, tx_results) where:
                -    results: List of propagation results for the spacecrafts.
                - tx_results: Optional list of propagation results for GNSS (transmitter) spacecraft,
                              or None if no GNSS spacecraft were provided. Applcable only for GNSS-R coverage.

                Each list item is a dict with:
                - "spacecraft_id": str
                - "spacecraft_name": str
                - "trajectory": StateSeries

        Example:
            ([
                {
                    "spacecraft_id": "04a388ad-...",
                    "spacecraft_name": "sc1",
                    "trajectory": StateSeries(...)
                },
                {
                    "spacecraft_id": "44966609-...",
                    "spacecraft_name": "sc2",
                    "trajectory": StateSeries(...)
                }
            ],
            None)
        """
        if self.propagator is None:
            raise ValueError("No propagator specified for the mission.")

        space_track_credentials_fp = None
        if self.settings.spacetrack_credentials_relative_path is not None:
            space_track_credentials_fp = os.path.join(
                self.settings.user_dir,
                self.settings.spacetrack_credentials_relative_path,
            )

        results: List[Dict[str, Union[str, StateSeries]]] = []
        for sc in self.spacecrafts:
            trajectory = propagate_spacecraft(
                spacecraft=sc,
                propagator=self.propagator,
                t0=self.start_time,
                duration_days=self.duration_days,
                space_track_credentials_fp=space_track_credentials_fp,
            )
            results.append(
                {
                    "spacecraft_id": sc.identifier,
                    "spacecraft_name": sc.name,
                    "trajectory": trajectory,
                }
            )

        if self.gnss_spacecrafts:
            gnss_results: List[Dict[str, Union[str, StateSeries]]] = []
            for sc in self.gnss_spacecrafts:
                trajectory = propagate_spacecraft(
                    spacecraft=sc,
                    propagator=self.propagator,
                    t0=self.start_time,
                    duration_days=self.duration_days,
                    space_track_credentials_fp=space_track_credentials_fp,
                )
                gnss_results.append(
                    {
                        "spacecraft_id": sc.identifier,
                        "spacecraft_name": sc.name,
                        "trajectory": trajectory,
                    }
                )
            return (results, gnss_results)
        else:
            return (results, None)

    def execute_eclipse_finder(
        self, propagated_trajectories: List[Dict[str, Union[str, StateSeries]]]
    ) -> List[Dict[str, Union[str, EclipseInfo]]]:
        """Calculate eclipse periods for all spacecrafts in the mission.

        Args:
            propagated_trajectories (List[Dict[str, Union[str, StateSeries]]]):
                Propagation results containing spacecraft IDs/names and their corresponding StateSeries.

        Returns:
            List[Dict[str, Union[str, EclipseInfo]]]:
                One item per spacecraft with:
                - "spacecraft_id": str
                - "spacecraft_name": str
                - "eclipse_info": A list of tuples representing the start and end times of eclipse intervals.

        Example:
            [
                {   "spacecraft_id": "04a388ad-...",
                    "spacecraft_name": "sc1",
                    "eclipse_info": [ [start, end], [start, end], ... ]
                },
                {   "spacecraft_id": "44966609-...",
                    "spacecraft_name": "sc2",
                    "eclipse_info": [ [start, end], [start, end], ... ]
                }
            ]
        """
        eclipse_finder = EclipseFinder()
        results: List[Dict[str, Union[str, EclipseInfo]]] = []
        for item in propagated_trajectories:
            spc_id = item["spacecraft_id"]
            spc_name = item.get("spacecraft_name")
            trajectory = item["trajectory"]
            eclipses = eclipse_finder.execute(
                frame_graph=self.frame_graph, state=trajectory
            )
            results.append(
                {
                    "spacecraft_id": spc_id,
                    "spacecraft_name": spc_name,
                    "eclipse_info": eclipses.eclipse_intervals(),
                }
            )

        return results

    def execute_gs_contact_finder(
        self, propagated_trajectories: List[Dict[str, Union[str, StateSeries]]]
    ) -> List[Dict[str, Any]]:
        """Compute ground-station contacts for each spacecraft–station pair.

        Args:
            propagated_trajectories:
                Output of execute_propagation()[0] (or any list in the same format),
                i.e., a list of dicts with "spacecraft_id", "spacecraft_name", and "trajectory".

        Returns:
            List[Dict[str, Any]]: List of nested dictionaries mapping spacecraft IDs/names to a dict
                that maps ground-station IDs/names to ContactInfo.
                Each item has:
                - "spacecraft_id": str
                - "spacecraft_name": str
                - "contacts": List[Dict[str, Any]] where each dict has:
                        - "ground_station_id": str
                        - "ground_station_name": str
                        - "contact_info":  A list of tuples representing the start and end times of contact intervals.
            Example:
            [
                {
                    "spacecraft_id": "04a388ad-...",
                    "spacecraft_name": "sc1",
                    "contacts": [
                        {
                            "ground_station_id": "69e3233c-...",
                            "ground_station_name": "gs1",
                            "contact_info": [ [start, end], [start, end], ... ]
                        },
                        {
                            "ground_station_id": "c3ece70c-...",
                            "ground_station_name": "gs2",
                            "contact_info": [ [start, end], [start, end], ... ]
                        }
                    ]
                },
                {
                    "spacecraft_id": "44966609-...",
                    "spacecraft_name": "sc2",
                    "contacts": [
                        {
                            "ground_station_id": "69e3233c-...",
                            "ground_station_name": "gs1",
                            "contact_info": [ [start, end], [start, end], ... ]
                        },
                        {
                            "ground_station_id": "c3ece70c-...",
                            "ground_station_name": "gs2",
                            "contact_info": [ [start, end], [start, end], ... ]
                        }
                    ]
                }
            ]
        """
        if self.ground_stations is None:
            raise ValueError(
                "No ground stations specified for contact finding."
            )

        all_contact_info: List[Dict[str, Any]] = []
        for item in propagated_trajectories:
            spc_id = item["spacecraft_id"]
            spc_name = item.get("spacecraft_name")
            trajectory = item["trajectory"]
            ground_stn_contact_info = {
                "spacecraft_id": spc_id,
                "spacecraft_name": spc_name,
                "contacts": list(),
            }
            for gs in self.ground_stations:
                result = calculate_gs_contact(
                    trajectory=trajectory,
                    ground_station=gs,
                    frame_graph=self.frame_graph,
                )
                ground_stn_contact_info["contacts"].append(
                    {
                        "ground_station_id": gs.identifier,
                        "ground_station_name": gs.name,
                        "contact_info": result.contact_intervals(), # interval format
                    }
                )
            all_contact_info.append(ground_stn_contact_info)
        return all_contact_info

    def execute_coverage_calculator(
        self, propagated_trajectories: List[Dict[str, Union[str, StateSeries]]]
    ) -> List[Dict[str, Any]]:
        """Compute coverage over configured spatial points for each spacecraft sensor.
        Executed the PointCoverage calculator.

        Args:
            propagated_trajectories:
                Output of execute_propagation()[0].

        Returns:
            List[Dict[str, Any]]: Nested dictionary mapping spacecraft IDs/names to a dict that maps
                sensor IDs/names to their coverage information.
                Each item has:
                    One item per (spacecraft, sensor) with:
                    - "spacecraft_id": str
                    - "spacecraft_name": str
                    - "total_spacecraft_coverage": List[Dict[str, Any]] where each dict has:
                        - "sensor_id": str
                        - "sensor_name": str
                        - "coverage": PointCoverage

            Example:
            [
                {
                    "spacecraft_id": "04a388ad-...",
                    "spacecraft_name": "sc1",
                    "total_spacecraft_coverage": [
                        {
                            "sensor_id": "bb171ddb-...",
                            "sensor_name": "sensorA",
                            "coverage_info": DiscreteCoverageTP(...)
                        },
                        {
                            "sensor_id": "917953d3-...",
                            "sensor_name": "sensorB",
                            "coverage_info": DiscreteCoverageTP(...)
                        }
                    ]
                },
                {
                    "spacecraft_id": "44966609-...",
                    "spacecraft_name": "sc2",
                    "total_spacecraft_coverage": [
                        {
                            "sensor_id": "354acb08-...",
                            "sensor_name": "sensorC",
                            "coverage_info": DiscreteCoverageTP(...)
                        },
                        {
                            "sensor_id": "6ce6981d-...",
                            "sensor_name": "sensorD",
                            "coverage_info": DiscreteCoverageTP(...)
                        }
                    ]
                }
            ]
        
        Raises:
            ValueError: If no spatial points are specified for coverage calculation.
            ValueError: If a spacecraft does not have a local orbital frame handler specified.
        """
        if self.cartesian_spatial_points is None:
            raise ValueError(
                "No spatial points specified for coverage calculation."
            )

        coverage_calculator = PointCoverage()
        eci_frame = ReferenceFrame.get("ICRF_EC")

        all_coverage_info: List[Dict[str, Any]] = []
        for item in propagated_trajectories:
            spc_id = item["spacecraft_id"]
            spc_name = item.get("spacecraft_name")
            trajectory = item["trajectory"]
            times = (
                trajectory.time
            )  # times at which coverage is to be calculated

            # Associate trajectory with the spacecraft
            spacecraft = next(
                (sc for sc in self.spacecrafts if sc.identifier == spc_id), None
            )
            if spacecraft is None:
                continue

            # get the local frame details for the spacecraft
            if spacecraft.local_orbital_frame_handler is None:
                raise ValueError(
                    f"No local orbital frame handler specified for spacecraft {spc_id}."
                )
            local_orbital_frame_handler = spacecraft.local_orbital_frame_handler
            local_orbital_frame = local_orbital_frame_handler.get_frame()
            att_lvlh, pos_lvlh = local_orbital_frame_handler.get_transform(
                trajectory
            )

            sensor_cov: List[Dict[str, Any]] = []
            for sensor in spacecraft.sensor:
                # calculate coverage for each sensor
                self.frame_graph.add_orientation_transform(att_lvlh)
                self.frame_graph.add_pos_transform(
                    from_frame=eci_frame,
                    to_frame=local_orbital_frame,
                    position=pos_lvlh,
                )
                result = coverage_calculator.calculate_coverage(
                    target_point_array=self.cartesian_spatial_points,
                    fov=sensor.fov,
                    frame_graph=self.frame_graph,
                    times=times,
                    surface=self.settings.surface_type,
                )
                sensor_cov.append(
                    {
                        "sensor_id": sensor.identifier,
                        "sensor_name": sensor.name,
                        "coverage_info": result,
                    }
                )

            all_coverage_info.append(
                {
                    "spacecraft_id": spc_id,
                    "spacecraft_name": spc_name,
                    "total_spacecraft_coverage": sensor_cov,
                }
            )  # append results for this spacecraft
        return all_coverage_info

    def execute_gnssr_coverage_calculator(
        self,
        propagated_rx_trajectories: List[Dict[str, Union[str, StateSeries]]],
        propagated_tx_trajectories: List[Dict[str, Union[str, StateSeries]]],
    ) -> List[Dict[str, Any]]:
        """Calculate GNSS-R coverage for receiver spacecraft using GNSS satellites.
        The GNSS satellites are the transmitter satellites, and the receiver
        spacecraft(s) is the one with the GNSS-R sensor(s).

        Specular radius is taken from Settings.

        Args:
            propagated_rx_trajectories:
                Output of execute_propagation()[0].
            propagated_tx_trajectories:
                Output of execute_propagation()[1].

        Returns:
            List[Dict[str, Any]]: List of nested (3 levels) dictionary mapping spacecraft IDs/names to a dict
                that maps sensors IDs/names to their coverage information.
                Each sensor has a list of dictionaries that maps the
                GNSS spacecraft IDs/names to their coverage information.

                Each item has:
                - "spacecraft_id": str
                - "spacecraft_name": str
                - "total_spacecraft_coverage": List[Dict[str, Any]] where each dict has:
                    - "sensor_id": str
                    - "sensor_name": str
                    - "total_sensor_coverage": List[Dict[str, Any]] where each dict has:
                        - "gnss_spacecraft_id": str
                        - "gnss_spacecraft_name": str
                        - "coverage_info": DiscreteCoverageTP
                        - "rcg_factor": List[float]


            Example:
            [
                {
                    "spacecraft_id": "04a388ad-...",
                    "spacecraft_name": "receiver1",
                    "total_spacecraft_coverage": [
                        {
                            "sensor_id": "bfc33b46-...",
                            "sensor_name": "gnssrA",
                            "total_sensor_coverage": [
                                {
                                    "gnss_spacecraft_id": "daea41f9-...",
                                    "gnss_spacecraft_name": "GNSS_01",
                                    "coverage_info": DiscreteCoverageTP(...),
                                    "rcg_factor": List[float]
                                },
                                {
                                    "gnss_spacecraft_id": "46c79a9f-...",
                                    "gnss_spacecraft_name": "GNSS_02",
                                    "coverage_info": DiscreteCoverageTP(...),
                                    "rcg_factor": List[float]
                                }
                            ]
                        },
                        {
                            "sensor_id": "99a79793-...",
                            "sensor_name": "gnssrB",
                            "total_sensor_coverage": [
                                {
                                    ...
                                },
                                ...
                            ]
                        },
                        ...
                    ]
                },
                {
                    "spacecraft_id": "3381b372-...",
                    ...
                },
                ...
            ]
        """
        if not self.gnss_spacecrafts:
            raise ValueError(
                "No GNSS spacecrafts specified for GNSS-R coverage calculation."
            )
        if not self.cartesian_spatial_points:
            raise ValueError(
                "No spatial points specified for coverage calculation."
            )
        if not self.settings.specular_radius_km:
            raise ValueError(
                "Specular radius not specified for coverage calculation."
            )

        coverage_calculator = SpecularCoverage()
        eci_frame = ReferenceFrame.get("ICRF_EC")

        # load the GNSS info (names, id and frames)
        gnss_frames = []
        tx_spacecraft_id = []
        tx_spacecraft_name = []
        for tx_item in propagated_tx_trajectories:

            tx_spc_id = tx_item["spacecraft_id"]
            tx_spc_name = tx_item.get("spacecraft_name")
            tx_trajectory = tx_item["trajectory"]

            tx_spacecraft_id.append(tx_spc_id)
            tx_spacecraft_name.append(tx_spc_name)

            # Compute transformations and add the GNSS LVLH frames
            # Associate trajectory with the spacecraft
            tx_spacecraft = next(
                (
                    sc
                    for sc in self.gnss_spacecrafts
                    if sc.identifier == tx_spc_id
                ),
                None,
            )
            if tx_spacecraft is None:
                continue
            # get the local frame details for the spacecraft
            if tx_spacecraft.local_orbital_frame_handler is None:
                raise ValueError(
                    f"No local orbital frame handler specified for spacecraft {tx_spc_id}."
                )
            tx_local_orbital_frame_handler = (
                tx_spacecraft.local_orbital_frame_handler
            )
            tx_local_orbital_frame = tx_local_orbital_frame_handler.get_frame()
            tx_att_lvlh, tx_pos_lvlh = (
                tx_local_orbital_frame_handler.get_transform(tx_trajectory)
            )

            self.frame_graph.add_orientation_transform(tx_att_lvlh)
            self.frame_graph.add_pos_transform(
                from_frame=eci_frame,
                to_frame=tx_local_orbital_frame,
                position=tx_pos_lvlh,
            )

            gnss_frames.append(tx_local_orbital_frame)

        # initialize the results dictionary
        all_coverage_info: List[Dict[str, Any]] = []

        for rx_item in propagated_rx_trajectories:

            rx_spc_id = rx_item["spacecraft_id"]
            rx_spc_name = rx_item.get("spacecraft_name")
            rx_trajectory = rx_item["trajectory"]
            rx_times = (
                rx_trajectory.time
            )  # times at which coverage is to be calculated

            # Associate trajectory with the spacecraft
            rx_spacecraft = next(
                (sc for sc in self.spacecrafts if sc.identifier == rx_spc_id),
                None,
            )
            if rx_spacecraft is None:
                continue

            # get the local frame details for the spacecraft
            if rx_spacecraft.local_orbital_frame_handler is None:
                raise ValueError(
                    f"No local orbital frame handler specified for spacecraft {rx_spc_id}."
                )
            rx_local_orbital_frame_handler = (
                rx_spacecraft.local_orbital_frame_handler
            )
            rx_local_orbital_frame = rx_local_orbital_frame_handler.get_frame()
            rx_att_lvlh, rx_pos_lvlh = (
                rx_local_orbital_frame_handler.get_transform(rx_trajectory)
            )

            rx_sensor_cov: List[Dict[str, Any]] = []
            for rx_sensor in rx_spacecraft.sensor:
                # calculate coverage for each sensor

                # add rx spacecraft local orbital frame transforms to the frame graph
                self.frame_graph.add_orientation_transform(rx_att_lvlh)
                self.frame_graph.add_pos_transform(
                    from_frame=eci_frame,
                    to_frame=rx_local_orbital_frame,
                    position=rx_pos_lvlh,
                )

                result = coverage_calculator.calculate_coverage(
                    target_point_array=self.cartesian_spatial_points,
                    fov=rx_sensor.fov,
                    frame_graph=self.frame_graph,
                    times=rx_times,
                    transmitter_frames=gnss_frames,
                    specular_radius=self.settings.specular_radius_km,
                    surface=self.settings.surface_type,
                )

                # Structure the result as a list of dictionaries for clarity
                rsc_x = [
                    {
                        "gnss_spacecraft_id": tx_spacecraft_id[
                            i
                        ],  # GNSS spacecraft ID
                        "gnss_spacecraft_name": tx_spacecraft_name[
                            i
                        ],  # GNSS spacecraft name
                        "coverage_info": coverage[
                            0
                        ],  # DiscreteCoverageTP object
                        "rcg_factor": coverage[
                            1
                        ],  # RCG proportionality factor
                    }
                    for i, coverage in enumerate(result)
                ]
                rx_sensor_cov.append(
                    {
                        "sensor_id": rx_sensor.identifier,
                        "sensor_name": rx_sensor.name,
                        "total_sensor_coverage": rsc_x,
                    }
                )

            rx_spc_coverage = {
                "spacecraft_id": rx_spc_id,
                "spacecraft_name": rx_spc_name,
                "total_spacecraft_coverage": rx_sensor_cov,
            }  # append results for this spacecraft

            all_coverage_info.append(rx_spc_coverage)

        return all_coverage_info

    def execute_specular_trajectory_calculator(
        self,
        propagated_rx_trajectories: List[Dict[str, Union[str, StateSeries]]],
        propagated_tx_trajectories: List[Dict[str, Union[str, StateSeries]]],
    ) -> List[Dict[str, Any]]:
        """Calculate specular points for pairs of (receiver: GNSSR) spacecrafts 
            and (transmitter) GNSS spacecrafts.
            This routine calculates **all possible** specular points between each, meaning the the field of view
            of the GNSSR or the GNSS spacecrafts are not considered.

        Args:
            propagated_rx_trajectories:
                Output of execute_propagation()[0].
            propagated_tx_trajectories:
                Output of execute_propagation()[1].

        Returns:
            List[Dict[str, Any]]: List of nested dictionary mapping (receiver) spacecraft IDs/names to
                 a list of dictionaries that maps the (transmitter) GNSS spacecraft IDs/names 
                 to their calculated specular point information.

                Each item has:
                - "spacecraft_id": str
                - "spacecraft_name": str
                - "all_specular_trajectories": List[Dict[str, Union[str, PositionSeries]]] where each dict has:
                    - "gnss_spacecraft_id": str
                    - "gnss_spacecraft_name": str
                    - "specular_trajectory": PositionSeries


            Example:
            [
                {
                    "spacecraft_id": "04a388ad-...",
                    "spacecraft_name": "receiver1",
                    "all_spacecraft_specular_info": [
                            {
                                "gnss_spacecraft_id": "daea41f9-...",
                                "gnss_spacecraft_name": "GNSS_01",
                                "specular_info": PositionSeries(...)
                            },
                            {
                                "gnss_spacecraft_id": "46c79a9f-...",
                                "gnss_spacecraft_name": "GNSS_02",
                                "specular_info": PositionSeries(...),
                            },
                            ...
                    ]
                },
                {
                    "spacecraft_id": "3381b372-...",
                    ...
                },
                ...
            ]

        """
        if not self.gnss_spacecrafts:
            raise ValueError(
                "No GNSS spacecrafts specified for GNSS-R coverage calculation."
            )

        # initialize the results dictionary
        all_specular_info: List[Dict[str, Any]] = [] # accumulate results over all receiver spacecrafts
        for rx_item in propagated_rx_trajectories:

            rx_spc_id = rx_item["spacecraft_id"]
            rx_spc_name = rx_item.get("spacecraft_name")
            rx_trajectory_icrf = rx_item["trajectory"]
            rx_times = (
                rx_trajectory_icrf.time
            )  # times at which coverage is to be calculated

            # Convert the propgated trajectories from ICRF_EC frame to ITRF frame since the specular point calculation
            # assumes the input trajectories are in the ITRF frame.
            rx_rot_array, rx_w_array = self.frame_graph.get_orientation_transform(
                from_frame = ReferenceFrame.get("ICRF_EC"),
                to_frame = ReferenceFrame.get("ITRF"),
                t = rx_times
            )

            # Convert the propgated trajectories from ICRF_EC frame to ITRF frame since the specular point calculation
            # assumes the input trajectories are in the ITRF frame.
            rx_pos_icrf = rx_trajectory_icrf.data[0]
            rx_vel_icrf = rx_trajectory_icrf.data[1]
            rx_rot_array, rx_w_array = self.frame_graph.get_orientation_transform(
                from_frame = ReferenceFrame.get("ICRF_EC"),
                to_frame = ReferenceFrame.get("ITRF"),
                t = rx_times
            )
            pos_itrf = rx_rot_array.apply(rx_pos_icrf)
            vel_itrf = rx_rot_array.apply(rx_vel_icrf) + np.cross(rx_w_array, rx_pos_icrf)
            rx_trajectory_itrf = StateSeries(
                time=rx_trajectory_icrf.time,
                data=[pos_itrf, vel_itrf],
                frame=ReferenceFrame.get("ITRF")
            )

            all_spacecraft_specular_info: List[Dict[str, Any]] = [] # accumulate results for this receiver spacecrafts for all the GNSS transmitters
            for tx_item in propagated_tx_trajectories:

                tx_spc_id = tx_item["spacecraft_id"]
                tx_spc_name = tx_item.get("spacecraft_name")
                tx_trajectory_icrf = tx_item["trajectory"]
                
                # Convert the propagated trajectories from ICRF_EC frame to ITRF frame since the specular point calculation
                # assumes the input trajectories are in the ITRF frame.
                tx_pos_icrf = tx_trajectory_icrf.data[0]
                tx_vel_icrf = tx_trajectory_icrf.data[1]
                tx_rot_array, tx_w_array = self.frame_graph.get_orientation_transform(
                    from_frame = ReferenceFrame.get("ICRF_EC"),
                    to_frame = ReferenceFrame.get("ITRF"),
                    t = rx_times
                )
                pos_itrf = tx_rot_array.apply(tx_pos_icrf)
                vel_itrf = tx_rot_array.apply(tx_vel_icrf) + np.cross(tx_w_array, tx_pos_icrf)
                tx_trajectory_itrf = StateSeries(
                    time=tx_trajectory_icrf.time,
                    data=[pos_itrf, vel_itrf],
                    frame=ReferenceFrame.get("ITRF")
                )

                # Compute specular points
                sp_posseries = get_specular_trajectory(
                    transmitter=tx_trajectory_itrf,
                    receiver=rx_trajectory_itrf,
                    times=rx_times,
                    surface=self.settings.surface_type
                )

                all_spacecraft_specular_info.append(
                    {
                        "gnss_spacecraft_id": tx_spc_id,
                        "gnss_spacecraft_name": tx_spc_name,
                        "specular_info": sp_posseries,
                    }
                )
            
            all_specular_info.append(
                {
                    "spacecraft_id": rx_spc_id,
                    "spacecraft_name": rx_spc_name,
                    "all_spacecraft_specular_info": all_spacecraft_specular_info,
                }
            )  # append results for this spacecraft

        return all_specular_info


    def execute_all(self) -> Dict[str, Any]:
        """Run all configured mission analyses and return a structured result bundle.

        Returns:
            Dict[str, Any]:
                A dictionary with the following keys (present only if the respective analysis runs):
                - "propagator_results": List[Dict[str, Union[str, StateSeries]]]
                                  Primary spacecraft propagation results (see execute_propagation()).
                - "propagation_gnss": Optional[List[Dict[str, Union[str, StateSeries]]]]
                                      GNSS spacecraft propagation results (see execute_propagation()).
                                      Applies for the case of gnssr missions.
                - "eclipse_finder_results": List[Dict[str, Union[str, EclipseInfo]]]
                             Eclipse results per spacecraft (see execute_eclipse_finder()).
                - "contact_finder_results": List[Dict[str, Any]]
                              Ground-station contact results (see execute_gs_contact_finder()).
                - "coverage_calculator_results": List[Dict[str, Any]]
                              Coverage calculator results (see execute_coverage_calculator()).

        """
        # propagation
        propagated_trajectories, gnss_trajectories = self.execute_propagation()

        # eclipse calculation
        eclipse = self.execute_eclipse_finder(propagated_trajectories)

        contacts = None
        coverage = None

        # ground-station contacts
        if self.ground_stations:
            contacts = self.execute_gs_contact_finder(propagated_trajectories)

        # coverage calculation
        if self.cartesian_spatial_points is not None:
            if self.settings.coverage_type == CoverageType.POINT_COVERAGE:
                coverage = self.execute_coverage_calculator(
                    propagated_trajectories
                )
                # form the mission results dictionary
                mission_results = {
                    "propagator_results": propagated_trajectories,
                    "eclipse_finder_results": eclipse,
                    "contact_finder_results": contacts,
                    "coverage_calculator_results": coverage,
                }
                
            elif self.settings.coverage_type == CoverageType.SPECULAR_COVERAGE:
                if not gnss_trajectories:
                    raise ValueError(
                        "No GNSS spacecraft trajectories available for GNSS-R coverage calculation."
                    )
                specular_trajectories = self.execute_specular_trajectory_calculator(
                    propagated_rx_trajectories=propagated_trajectories,
                    propagated_tx_trajectories=gnss_trajectories,
                )
                coverage = self.execute_gnssr_coverage_calculator(
                    propagated_rx_trajectories=propagated_trajectories,
                    propagated_tx_trajectories=gnss_trajectories,
                )
                # form the mission results dictionary
                mission_results = {
                    "propagator_results": propagated_trajectories,
                    "gnssr_propagator_results": gnss_trajectories,
                    "eclipse_finder_results": eclipse,
                    "contact_finder_results": contacts,
                    "coverage_calculator_results": coverage,
                    "specular_trajectory_results": specular_trajectories,
                }
                
            else:
                raise ValueError(
                    f"Unsupported coverage type {self.settings.coverage_type}."
                )

       
        return mission_results
