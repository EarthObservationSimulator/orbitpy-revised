"""
.. module:: orbitpy.mission
   :synopsis: Module to handle mission initialization and execution.

Collection of classes and functions relating to represention of 
mission parameters and execution.

Base and standard reference frames with which the module works are:
1/ ICRF_EC (Earth-Centered Inertial Frame) as defined in eosimutils.base.ReferenceFrame
2/ ITRF (Earth-Centered Earth-Fixed Frame) as defined in eosimutils.base.ReferenceFrame
3/ ORBITPY_LVLH (Local Vertical Local Horizontal Frame) as defined in eosimutils.standardframes.get_lvlh
"""
from typing import Dict, Any, Union, List, Optional

from eosimutils.base import ReferenceFrame, SurfaceType
from eosimutils.time import AbsoluteDate
from eosimutils.state import Cartesian3DPositionArray
from eosimutils.trajectory import StateSeries
from eosimutils.framegraph import FrameGraph
from eosimutils.standardframes import get_lvlh

from .propagator import PropagatorFactory, SGP4Propagator
from .resources import Spacecraft, GroundStation, Sensor
from .eclipsefinder import EclipseFinder, EclipseInfo
from .contactfinder import ElevationAwareContactFinder, ContactInfo
from .coveragecalculator import PointCoverage, SpecularCoverage, CoverageType
from .coverage import DiscreteCoverageTP

def propagate_spacecraft(spacecraft: Spacecraft, propagator: SGP4Propagator,
                        t0: AbsoluteDate, duration_days: float) -> StateSeries:
    """
    Propagate a spacecraft's orbit using a specified propagator.

    Args:
        spacecraft (Spacecraft): The spacecraft object to be propagated.
        propagator (SGP4Propagator): The propagator to use for orbit propagation.
        t0 (AbsoluteDate): Start time for propagation.
        duration_days (float): Duration of propagation in days.
        
    Returns:
        trajectory (StateSeries): StateSeries object containing the propagated states.
    """
    if not isinstance(propagator, SGP4Propagator):
        raise ValueError("Unsupported propagator type.")
    orbit = spacecraft.orbit
    propagated_states = propagator.execute(
            t0=t0, duration_days=duration_days, orbit=orbit
        )
    return propagated_states

def calculate_gs_contact(trajectory: StateSeries, ground_station: GroundStation, frame_graph: FrameGraph) -> ContactInfo:
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

class CoverageSettings:
    """Class to hold coverage calculation settings."""
    def __init__(self, coverage_type: Optional[Union[CoverageType, str]] = None, specular_radius_km: Optional[float] = None):
        """
        Args:
            coverage_type (Optional[Union[CoverageType, str]]): Type of coverage. See `orbitpy.coveragecalculator.CoverageType`. 
                                                                Defaults to POINT_COVERAGE if None.
            specular_radius_km (Optional[float]): Specular radius in kilometers (applicable for GNSSR coverage).
        """
        self.coverage_type = coverage_type if coverage_type else CoverageType.POINT_COVERAGE
        self.specular_radius_km = specular_radius_km

    @classmethod
    def from_dict(cls, dict_in: Dict[str, Any]) -> "CoverageSettings":
        """Create a CoverageSettings object from a dictionary.

        Args:
            dict_in (Dict[str, Any]): Dictionary containing coverage settings.
        Returns:
            CoverageSettings: An instance of the CoverageSettings class.
        """
        coverage_type = dict_in.get("coverage_type", None)
        coverage_type = CoverageType.get(coverage_type) if coverage_type else CoverageType.POINT_COVERAGE
        specular_radius_km = dict_in.get("specular_radius_km", None)
        return cls(coverage_type=coverage_type, specular_radius_km=specular_radius_km)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the CoverageSettings object to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of the CoverageSettings object.
        """
        return {
            "coverage_type": self.coverage_type.to_string() if self.coverage_type else None,
            "specular_radius_km": self.specular_radius_km,
        }

class Mission:
    """Class to represent a space mission with spacecraft, sensors, and ground stations."""

    def __init__(self, 
                start_time: AbsoluteDate,
                duration_days: float,
                spacecrafts: Union[Spacecraft, List[Spacecraft]],
                ground_stations: Union[GroundStation, List[GroundStation], None], 
                propagator: Union[SGP4Propagator, None],
                grid_points: Union[Cartesian3DPositionArray, None],
                gnss_spacecrafts: Optional[List[Spacecraft]] = None,
                frame_graph: FrameGraph = None,
                coverage_settings: Optional[CoverageSettings] = None,
            ):
        """
        Args:
            start_time (AbsoluteDate): Start time of the mission.
            duration_days (float): Duration of the mission in days.
            spacecrafts (Union[Spacecraft, List[Spacecraft]]): List of spacecraft in the mission.
            ground_stations (Union[GroundStation, List[GroundStation], None]): List of ground stations.
            propagator (Union[SGP4Propagator, None]): Propagator to use for orbit propagation.
            grid_points (Union[Cartesian3DPositionArray, None]): Grid points for coverage analysis.
            gnss_spacecrafts (Optional[List[Spacecraft]]): List of GNSS satellites (source satellites) (applicable for GNSSR coverage).
            frame_graph (FrameGraph): Frame graph for coordinate transformations.
            coverage_settings (Optional[CoverageSettings]): Coverage calculation settings.
        """
        self.start_time = start_time
        self.duration_days = duration_days
        if not isinstance(spacecrafts, list):
            spacecrafts = [spacecrafts]
        self.spacecrafts = spacecrafts
        if ground_stations is not None and not isinstance(ground_stations, list):
            ground_stations = [ground_stations]
        self.ground_stations = ground_stations
        self.propagator = propagator
        self.grid_points = grid_points
        if gnss_spacecrafts is not None:
            if not isinstance(gnss_spacecrafts, list):
                gnss_spacecrafts = [gnss_spacecrafts]
        self.gnss_spacecrafts = gnss_spacecrafts if gnss_spacecrafts else []
        self.frame_graph = frame_graph if frame_graph is not None else FrameGraph()
        self.coverage_settings = coverage_settings if coverage_settings else CoverageSettings()

    @classmethod
    def from_dict(cls, dict_in: Dict[str, Any]) -> "Mission":
        """Create a Mission object from a dictionary.

        Args:
            dict_in (Dict[str, Any]): Dictionary containing mission specifications.
        Returns:
            Mission: An instance of the Mission class.
        """
        # setup mission parameters
        start_time = AbsoluteDate.from_dict(dict_in["start_time"])
        duration_days = dict_in["duration_days"]
        
        # setup spacecrafts
        spacecrafts = dict_in["spacecrafts"]
        if not isinstance(spacecrafts, list):
            spacecrafts = [spacecrafts]
        spacecrafts = [Spacecraft.from_dict(sc) for sc in spacecrafts]

        # setup GNSS (source) spacecrafts (applicable for GNSSR coverage)
        gnss_spacecrafts = dict_in.get("gnss_spacecrafts", [])
        if not isinstance(gnss_spacecrafts, list):
            gnss_spacecrafts = [gnss_spacecrafts]
        gnss_spacecrafts = [Spacecraft.from_dict(sc) for sc in gnss_spacecrafts]

        # setup ground-stations
        ground_stations = dict_in.get("ground_stations", None)
        if ground_stations is not None:
            if not isinstance(ground_stations, list):
                ground_stations = [ground_stations]
            ground_stations = [GroundStation.from_dict(gs) for gs in ground_stations]
        
        # setup propagator
        propagator_specs = dict_in.get("propagator", None)
        propagator = None
        if propagator_specs is not None:
            propagator = PropagatorFactory.from_dict(propagator_specs)
        
        # setup grid points
        grid_points_specs = dict_in.get("grid_points", None)
        grid_points = None
        if grid_points_specs is not None:
            grid_points = Cartesian3DPositionArray.from_dict(grid_points_specs)
        
        # setup the frames and the transformations
        # The spacecrafts need to be setup before this so that their local orbit frames are registered.
        transform_dict = dict_in.get("frame_transforms", None)
        # add reference frames from the user-specified transforms if not already present
        if transform_dict is not None:
            orientation_transforms = transform_dict.get("orientation_transforms", None)
            position_transforms = transform_dict.get("position_transforms", None)
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

        # setup coverage settings
        coverage_settings_dict = dict_in.get("coverage_settings", None)
        coverage_settings = CoverageSettings.from_dict(coverage_settings_dict) if coverage_settings_dict else CoverageSettings()

        return cls(
            start_time=start_time,
            duration_days=duration_days,
            spacecrafts=spacecrafts,
            gnss_spacecrafts=gnss_spacecrafts,
            ground_stations=ground_stations,
            propagator=propagator,
            grid_points=grid_points,
            frame_graph=frame_graph,
            coverage_settings=coverage_settings,
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
            "gnss_spacecrafts": [gnss_sc.to_dict() for gnss_sc in self.gnss_spacecrafts] if self.gnss_spacecrafts else None,
            "ground_stations": [gs.to_dict() for gs in self.ground_stations] if self.ground_stations else None,
            "propagator": self.propagator.to_dict() if self.propagator else None,
            "grid_points": self.grid_points.to_dict() if self.grid_points else None,
            "coverage_settings": self.coverage_settings.to_dict() if self.coverage_settings else None,
        }

    def execute_propagation(self) -> List[Dict[str, Union[str, StateSeries]]]:
        """Propagate all spacecrafts in the mission using the specified propagator.

        Returns:
            List[Dict[str, Any]]: Propagation results containing spacecraft_ids and corresponding trajectories (StateSeries objects).
        Example:
            [
                {"spacecraft_id": "04a388ad-...", "spacecraft_name": "sc1", "trajectory": StateSeries(...)},
                {"spacecraft_id": "44966609-...", "spacecraft_name": "sc2", "trajectory": StateSeries(...)},
                ...
            ]
        """
        if self.propagator is None:
            raise ValueError("No propagator specified for the mission.")

        results: List[Dict[str, Union[str, StateSeries]]] = []
        for sc in self.spacecrafts:
            trajectory = propagate_spacecraft(
                spacecraft=sc,
                propagator=self.propagator,
                t0=self.start_time,
                duration_days=self.duration_days,
            )
            results.append({"spacecraft_id": sc.identifier, "spacecraft_name": sc.name, "trajectory": trajectory})
        
        if self.gnss_spacecrafts:
            gnss_results: List[Dict[str, Union[str, StateSeries]]] = []
            for sc in self.gnss_spacecrafts:
                trajectory = propagate_spacecraft(
                    spacecraft=sc,
                    propagator=self.propagator,
                    t0=self.start_time,
                    duration_days=self.duration_days,
                )
                gnss_results.append({"spacecraft_id": sc.identifier, "spacecraft_name": sc.name, "trajectory": trajectory})
            return results, gnss_results
        else:
            return results

    def execute_eclipse_finder(self, propagated_trajectories: List[Dict[str, Union[str, StateSeries]]]) -> List[Dict[str, Union[str, EclipseInfo]]]:
        """Calculate eclipse periods for all spacecrafts in the mission.

        Args:
            propagated_trajectories (List[Dict[str, Union[str, StateSeries]]]):
                Propagation results containing spacecraft IDs and their corresponding StateSeries.

        Returns:
            List[Dict[str, Union[str, EclipseInfo]]]: Eclipse finding results.
            Example:
            [
                {"spacecraft_id": "04a388ad-...", "spacecraft_name": "sc1", "eclipse_info": EclipseInfo(...)},
                {"spacecraft_id": "44966609-...", "spacecraft_name": "sc2", "eclipse_info": EclipseInfo(...)},
                ...
            ]
        """
        eclipse_finder = EclipseFinder()
        results: List[Dict[str, Union[str, EclipseInfo]]] = []
        for item in propagated_trajectories:
            spc_id = item["spacecraft_id"]
            spc_name = item.get("spacecraft_name")
            trajectory = item["trajectory"]
            eclipses = eclipse_finder.execute(frame_graph=self.frame_graph, state=trajectory)
            results.append({"spacecraft_id": spc_id, "spacecraft_name": spc_name, "eclipse_info": eclipses})

        return results

    def execute_gs_contact_finder(self, propagated_trajectories: List[Dict[str, Union[str, StateSeries]]]) -> List[Dict[str, Any]]:
        """Calculate contact periods between spacecrafts and ground stations.

        Args:
            propagated_trajectories (List[Dict[str, Union[str, StateSeries]]]): Propagation results containing spacecraft IDs and their corresponding StateSeries.

        Returns:
            List[Dict[str, Any]]: List of nested dictionaries mapping spacecraft IDs to a dict that maps ground-station IDs to ContactInfo.
            Example:
            [
                {"spacecraft_id": "04a388ad-...", "spacecraft_name": "sc1", "contacts": [
                    {"ground_station_id": "69e3233c-...", "ground_station_name": "gs1", "contact_info": ContactInfo(...)},
                    {"ground_station_id": "c3ece70c-...", "ground_station_name": "gs2", "contact_info": ContactInfo(...)},
                    ...
                ]},
                {"spacecraft_id": "44966609-...", "spacecraft_name": "sc2", "contacts": [
                    {"ground_station_id": "69e3233c-...", "ground_station_name": "gs1", "contact_info": ContactInfo(...)},
                    {"ground_station_id": "c3ece70c-...", "ground_station_name": "gs2", "contact_info": ContactInfo(...)},
                    ...
                ]},
                ...
            ]
        """
        if self.ground_stations is None:
            raise ValueError("No ground stations specified for contact finding.")

        all_contact_info: List[Dict[str, Any]] = []
        for item in propagated_trajectories:
            spc_id = item["spacecraft_id"]
            spc_name = item.get("spacecraft_name")
            trajectory = item["trajectory"]
            ground_stn_contact_info = {"spacecraft_id": spc_id, "spacecraft_name": spc_name, "contacts": list()}
            for gs in self.ground_stations:
                result = calculate_gs_contact(
                    trajectory=trajectory,
                    ground_station=gs,
                    frame_graph=self.frame_graph,
                )
                ground_stn_contact_info["contacts"].append({"ground_station_id": gs.identifier, "ground_station_name": gs.name, "contact_info": result})
            all_contact_info.append(ground_stn_contact_info)
        return all_contact_info

    def execute_coverage_calculator(self, propagated_trajectories: List[Dict[str, Union[str, StateSeries]]]) -> List[Dict[str, Any]]:
        """Calculate coverage for spacecraft sensors over specified grid points.

        Args:
            propagated_trajectories (List[Dict[str, Union[str, StateSeries]]]): Propagation results containing spacecraft IDs and their corresponding StateSeries.

        Returns:
            List[Dict[str, Any]]: Nested dictionary mapping spacecraft IDs to a dict that maps sensor IDs to their coverage information.
            Example:
            [
                {"spacecraft_id": "04a388ad-...", "spacecraft_name": "sc1", "coverage": [
                    {"sensor_id": "bb171ddb-...", "sensor_name": "sensorA", "coverage_info": DiscreteCoverageTP(...)},
                    {"sensor_id": "917953d3-...", "sensor_name": "sensorB", "coverage_info": DiscreteCoverageTP(...)},
                    ...
                ]},
                {"spacecraft_id": "44966609-...", "spacecraft_name": "sc2", "coverage": [
                    {"sensor_id": "354acb08-...", "sensor_name": "sensorC", "coverage_info": DiscreteCoverageTP(...)},
                    {"sensor_id": "6ce6981d-...", "sensor_name": "sensorD", "coverage_info": DiscreteCoverageTP(...)}
                ]},
                ...
            ]
        """
        if self.grid_points is None:
            raise ValueError("No grid points specified for coverage calculation.")
        
        coverage_calculator = PointCoverage()
        eci_frame = ReferenceFrame.get("ICRF_EC")
        
        all_coverage_info: List[Dict[str, Any]] = []
        for item in propagated_trajectories:
            spc_id = item["spacecraft_id"]
            spc_name = item.get("spacecraft_name")
            trajectory = item["trajectory"]
            times = trajectory.time # times at which coverage is to be calculated

            # Associate trajectory with the spacecraft
            spacecraft = next((sc for sc in self.spacecrafts if sc.identifier == spc_id), None)
            if spacecraft is None:
                continue
            
            # get the local frame details for the spacecraft
            if spacecraft.local_orbital_frame_handler is None:
                raise ValueError(f"No local orbital frame handler specified for spacecraft {spc_id}.")
            local_orbital_frame_handler = spacecraft.local_orbital_frame_handler
            local_orbital_frame = local_orbital_frame_handler.get_frame()
            att_lvlh, pos_lvlh = local_orbital_frame_handler.get_transform(trajectory)

            sensor_cov: List[Dict[str, Any]] = []
            for sensor in spacecraft.sensor:
                # calculate coverage for each sensor
                self.frame_graph.add_orientation_transform(att_lvlh)
                self.frame_graph.add_pos_transform(from_frame=eci_frame, to_frame=local_orbital_frame, position=pos_lvlh)
                result = coverage_calculator.calculate_coverage(
                                    target_point_array=self.grid_points,                      
                                    fov=sensor.fov,
                                    frame_graph=self.frame_graph,
                                    times=times,
                                    surface=SurfaceType.SPHERE,
                            )
                sensor_cov.append({"sensor_id": sensor.identifier, "sensor_name": sensor.name, "coverage_info": result})

            all_coverage_info.append({"spacecraft_id": spc_id, "spacecraft_name": spc_name, "coverage": sensor_cov}) # append results for this spacecraft
        return all_coverage_info

    def execute_gnssr_coverage_calculator(self, 
                               propagated_rx_trajectories: List[Dict[str, Union[str, StateSeries]]], 
                               propagated_tx_trajectories: List[Dict[str, Union[str, StateSeries]]]) -> List[Dict[str, Any]]:
        """Calculate GNSS-R coverage for receiver spacecraft using GNSS satellites.
        The GNSS satellites are the transmitter satellites, and the receiver spacecraft(s) is the one with the GNSS-R sensor(s).

        Args:
            propagated_rx_trajectories (List[Dict[str, Union[str, StateSeries]]]): Propagation results containing receiver spacecraft IDs and their corresponding StateSeries.
            propagated_tx_trajectories (List[Dict[str, Union[str, StateSeries]]]): Propagation results containing GNSS satellite IDs and their corresponding StateSeries.

        Returns:
            Dict[str, Any]: Dictionary with two items:
                - tx_spacecrafts: Dictionary with lists of GNSS spacecraft IDs and their names.
                - rx_spacecrafts: Nested dictionary mapping spacecraft IDs to a dict that maps sensor IDs to their coverage information.
            Example:
            {   "tx_spacecrafts": { "spacecraft_id": ["441ab3ed-...", "4577c4c5--...", ], 
                                        "spacecraft_name": ["gnss1", "gnss2", ...]
                                        },
                "rx_spacecrafts": [{"spacecraft_id": "04a388ad-...", 
                                          "spacecraft_name": "receiver1", 
                                          "coverage": [
                                                        {"sensor_id": "bfc33b46-...", "sensor_name": "gnssrA", 
                                                        "coverage_info": List[Tuple[DiscreteCoverageTP(...), List[float]]
                                                    ]
                                         },
                                        ...
                                        ]
            }
        """
        if not self.gnss_spacecrafts:
            raise ValueError("No GNSS spacecrafts specified for GNSS-R coverage calculation.")
        if not self.grid_points:
            raise ValueError("No grid points specified for coverage calculation.")
        if not self.coverage_settings.specular_radius_km:
            raise ValueError("Specular radius not specified for coverage calculation.")

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
            tx_spacecraft = next((sc for sc in self.spacecrafts if sc.identifier == tx_item["spacecraft_id"]), None)
            if tx_spacecraft is None:
                continue
            # get the local frame details for the spacecraft
            if tx_spacecraft.local_orbital_frame_handler is None:
                raise ValueError(f"No local orbital frame handler specified for spacecraft {tx_spc_id}.")
            tx_local_orbital_frame_handler = tx_spacecraft.local_orbital_frame_handler
            tx_local_orbital_frame = tx_local_orbital_frame_handler.get_frame()
            tx_att_lvlh, tx_pos_lvlh = tx_local_orbital_frame_handler.get_transform(tx_trajectory)

            self.frame_graph.add_orientation_transform(tx_att_lvlh)
            self.frame_graph.add_pos_transform(from_frame=eci_frame, to_frame=tx_local_orbital_frame, position=tx_pos_lvlh)

            gnss_frames.append(tx_local_orbital_frame)
        
        # initialize the results dictionary
        all_coverage_info = {"tx_spacecrafts": {"spacecraft_id": tx_spacecraft_id,
                                                    "spacecraft_name": tx_spacecraft_name},
                             "rx_spacecrafts": []}

        for rx_item in propagated_rx_trajectories:

            rx_spc_id = rx_item["spacecraft_id"]
            rx_spc_name = rx_item.get("spacecraft_name")
            rx_trajectory = rx_item["trajectory"]
            rx_times = rx_trajectory.time # times at which coverage is to be calculated

            # Associate trajectory with the spacecraft
            rx_spacecraft = next((sc for sc in self.spacecrafts if sc.identifier == rx_spc_id), None)
            if rx_spacecraft is None:
                continue

            # get the local frame details for the spacecraft
            if rx_spacecraft.local_orbital_frame_handler is None:
                raise ValueError(f"No local orbital frame handler specified for spacecraft {rx_spc_id}.")
            rx_local_orbital_frame_handler = rx_spacecraft.local_orbital_frame_handler
            rx_local_orbital_frame = rx_local_orbital_frame_handler.get_frame()
            rx_att_lvlh, rx_pos_lvlh = rx_local_orbital_frame_handler.get_transform(rx_trajectory)

            rx_sensor_cov: List[Dict[str, Any]] = []
            for rx_sensor in rx_spacecraft.sensor:
                # calculate coverage for each sensor

                # add rx spacecraft local orbital frame transforms to the frame graph
                self.frame_graph.add_orientation_transform(rx_att_lvlh)
                self.frame_graph.add_pos_transform(from_frame=eci_frame, to_frame=rx_local_orbital_frame, position=rx_pos_lvlh)
                
                result = coverage_calculator.calculate_coverage(
                                    target_point_array=self.grid_points,                      
                                    fov=rx_sensor.fov,
                                    frame_graph=self.frame_graph,
                                    times=rx_times,
                                    transmitters=gnss_frames,
                                    specular_radius=self.coverage_settings.specular_radius_km,
                                    surface=SurfaceType.SPHERE,
                            )
                print(result)
                rx_sensor_cov.append({"sensor_id": rx_sensor.identifier, "sensor_name": rx_sensor.name, "coverage_info": result})


            rx_spc_coverage = {"spacecraft_id": rx_spc_id, "spacecraft_name": rx_spc_name, "coverage": rx_sensor_cov}

            all_coverage_info["receiver_spacecrafts"].append(rx_spc_coverage)

        return all_coverage_info
    '''
    def execute_all(self) -> Dict[str, Any]:
        """Run propagation, eclipse, contact and coverage and return a dictionary of results.
        Does not modify Mission instance state; all results are returned in the dict.
        """
        propagate = self.execute_propagation()
        eclipse = self.execute_eclipse_finder(propagate)

        contacts = None
        coverage = None

        if self.ground_stations:
            contacts = self.execute_gs_contact_finder(propagate)
        if self.grid_points is not None:
            coverage = self.execute_coverage_calculator(propagate)

        mission_results = {
            "propagate": propagate,
            "eclipse": eclipse,
            "contacts": contacts,
            "coverage": coverage,
        }
        return mission_results
    '''