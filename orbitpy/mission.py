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

from eosimutils.base import ReferenceFrame, SurfaceType, JsonSerializer
from eosimutils.time import AbsoluteDate
from eosimutils.state import Cartesian3DPositionArray
from eosimutils.trajectory import StateSeries
from eosimutils.framegraph import FrameGraph
from eosimutils.standardframes import get_lvlh

from .propagator import PropagatorFactory, SGP4Propagator
from .resources import Spacecraft, GroundStation, Sensor
from .eclipsefinder import EclipseFinder, EclipseInfo
from .contactfinder import ElevationAwareContactFinder, ContactInfo
from .coveragecalculator import PointCoverage
from .coverage import DiscreteCoverageTP

def propagate_spacecraft(spacecraft: Spacecraft, propagator: Union[SGP4Propagator],
                        t0: AbsoluteDate, duration_days: float):
    """
    Propagate a spacecraft's orbit using a specified propagator.

    Args:
        spacecraft (Spacecraft): The spacecraft object to be propagated.
        propagator (Union[SGP4Propagator]): The propagator to use for orbit propagation.
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

class PropagationResults:
    """Data structure holding the propagation results mapping spacecraft IDs to StateSeries."""

    def __init__(self, spacecraft_id: List[str], trajectory: List[StateSeries]):
        """
        Args:
            spacecraft_id (List[str]): List of spacecraft identifiers (UUIDs).
            trajectory (List[StateSeries]): List of StateSeries objects corresponding to each spacecraft.
        """
        self.spacecraft_id = spacecraft_id if isinstance(spacecraft_id, list) else [spacecraft_id]
        self.trajectory = trajectory if isinstance(trajectory, list) else [trajectory]
        if len(self.spacecraft_id) != len(self.trajectory):
            raise ValueError("Length of spacecraft_id and trajectory lists must be the same.")

    def from_dict(cls, dict_in: Dict[str, Any]) -> "PropagationResults":
        """Create a PropagationResults object from a dictionary.

        Args:
            dict_in (Dict[str, Any]): Dictionary containing (list of) spacecraft-identifiers and (list of) trajectories.
                                        Expected keys are "spacecraft_id" and "trajectory".
        Returns:
            PropagationResults: An instance of the PropagationResults class.
        """
        spacecraft_id = dict_in["spacecraft_id"] if isinstance(dict_in["spacecraft_id"], list) else [dict_in["spacecraft_id"]]
        trajectory = [StateSeries.from_dict(ts) for ts in dict_in["trajectory"]] if isinstance(dict_in["trajectory"], list) else [StateSeries.from_dict(dict_in["trajectory"])]
        return cls(spacecraft_id=spacecraft_id, trajectory=trajectory)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the PropagationResults object to a list of dictionaries.

        Returns:
            Dict[str, StateSeries]: List of dictionaries mapping spacecraft IDs to their propagated StateSeries.
        """
        return {"spacecraft_id": self.spacecraft_id,
                 "trajectory": [traj.to_dict() for traj in self.trajectory]
                }

    def __len__(self) -> int:
        """Return the number of spacecraft in the PropagationResults."""
        return len(self.spacecraft_id)
    
    def __iter__(self):
        """Iterator over (spacecraft_id, trajectory) pairs."""
        return zip(self.spacecraft_id, self.trajectory).__iter__()

    def __getitem__(self, index: int) -> tuple[str, StateSeries]:
        """Get the (spacecraft_id, trajectory) pair at the specified index.

        Args:
            index (int): Index of the desired pair.

        Returns:
            (str, StateSeries): Tuple containing the spacecraft ID and its corresponding trajectory.
        """
        return self.spacecraft_id[index], self.trajectory[index]

class EclipseFinderResults:
    """Data structure holding the eclipse finding results mapping spacecraft IDs to StateSeries."""

    def __init__(self, spacecraft_id: List[str], eclipse_info: List[StateSeries]):
        """
        Args:
            spacecraft_id (List[str]): List of spacecraft identifiers (UUIDs).
            eclipse_info (List[EclipseInfo]): List of EclipseInfo objects corresponding to each spacecraft.
        """
        self.spacecraft_id = spacecraft_id if isinstance(spacecraft_id, list) else [spacecraft_id]
        self.eclipse_info = eclipse_info if isinstance(eclipse_info, list) else [eclipse_info]
        if len(self.spacecraft_id) != len(self.eclipse_info):
            raise ValueError("Length of spacecraft_id and eclipse_info lists must be the same.")

    def from_dict(cls, dict_in: Dict[str, Any]) -> "EclipseFinderResults":
        """Create a EclipseFinderResults object from a dictionary.

        Args:
            dict_in (Dict[str, Any]): Dictionary containing (list of) spacecraft-identifiers and (list of) eclipse-info.
                                        Expected keys are "spacecraft_id" and "eclipse_info".
        Returns:
            EclipseFinderResults: An instance of the EclipseFinderResults class.
        """
        spacecraft_id = dict_in["spacecraft_id"] if isinstance(dict_in["spacecraft_id"], list) else [dict_in["spacecraft_id"]]
        eclipse_info = [EclipseInfo.from_dict(ts) for ts in dict_in["eclipse_info"]] if isinstance(dict_in["eclipse_info"], list) else [EclipseInfo.from_dict(dict_in["eclipse_info"])]
        return cls(spacecraft_id=spacecraft_id, eclipse_info=eclipse_info)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the EclipseFinderResults object to a list of dictionaries.

        Returns:
            Dict[str, EclipseInfo]: List of dictionaries mapping spacecraft IDs to their EclipseInfo.
        """
        return {"spacecraft_id": self.spacecraft_id,
                 "eclipse_info": [info.to_dict() for info in self.eclipse_info]
                }

    def __len__(self) -> int:
        """Return the number of spacecraft in the EclipseFinderResults."""
        return len(self.spacecraft_id)
    
    def __iter__(self):
        """Iterator over (spacecraft_id, eclipse_info) pairs."""
        return zip(self.spacecraft_id, self.eclipse_info).__iter__()

    def __getitem__(self, index: int) -> tuple[str, EclipseInfo]:
        """Get the (spacecraft_id, eclipse_info) pair at the specified index.

        Args:
            index (int): Index of the desired pair.

        Returns:
            (str, EclipseInfo): Tuple containing the spacecraft ID and its corresponding eclipse information.
        """
        return self.spacecraft_id[index], self.eclipse_info[index]

class Mission:
    """Class to represent a space mission with spacecraft, sensors, and ground stations."""

    def __init__(self, 
                start_time: AbsoluteDate,
                duration_days: float,
                spacecrafts: Union[Spacecraft, List[Spacecraft]],
                ground_stations: Union[GroundStation, List[GroundStation], None], 
                propagator: Union[SGP4Propagator, None],
                grid_points:Union[Cartesian3DPositionArray, None],
                frame_graph: FrameGraph = None
            ):
        """
        Args:
            start_time (AbsoluteDate): Start time of the mission.
            duration_days (float): Duration of the mission in days.
            spacecrafts (Union[Spacecraft, List[Spacecraft]]): List of spacecraft in the mission.
            ground_stations (Union[GroundStation, List[GroundStation], None]): List of ground stations.
            propagator (Union[SGP4Propagator, None]): Propagator to use for orbit propagation.
            grid_points (Union[Cartesian3DPositionArray, None]): Grid points for coverage analysis.
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

        self.frame_graph = frame_graph if frame_graph is not None else FrameGraph()

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
        
        # setup spacecrafts and ground stations
        spacecrafts = dict_in["spacecrafts"]
        if not isinstance(spacecrafts, list):
            spacecrafts = [spacecrafts]
        spacecrafts = [Spacecraft.from_dict(sc) for sc in spacecrafts]
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
        # register the the OrbitPy LVLH frame if not already present
        if ReferenceFrame.get("ORBITPY_LVLH") is None:
            ReferenceFrame.add("ORBITPY_LVLH")
        transform_dict = dict_in.get("frame_transforms", None)
        # add reference frames from the user-specified transforms if not already present
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

        return cls(
            start_time=start_time,
            duration_days=duration_days,
            spacecrafts=spacecrafts,
            ground_stations=ground_stations,
            propagator=propagator,
            grid_points=grid_points,
            frame_graph=frame_graph,
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
            "ground_stations": [gs.to_dict() for gs in self.ground_stations] if self.ground_stations else None,
            "propagator": self.propagator.to_dict() if self.propagator else None,
            "grid_points": self.grid_points.to_dict() if self.grid_points else None,
        }

    def execute_propagation(self) -> PropagationResults:
        """Propagate all spacecrafts in the mission using the specified propagator.

        Returns:
            PropagationResults: Propagation results containing spacecraft_ids and corresponding StateSeries.
        """
        if self.propagator is None:
            raise ValueError("No propagator specified for the mission.")

        spacecraft_ids: List[str] = []
        trajectories: List[StateSeries] = []

        for sc in self.spacecrafts:
            trajectory = propagate_spacecraft(
                spacecraft=sc,
                propagator=self.propagator,
                t0=self.start_time,
                duration_days=self.duration_days,
            )
            spacecraft_ids.append(sc.identifier)
            trajectories.append(trajectory)

        return PropagationResults(spacecraft_id=spacecraft_ids, trajectory=trajectories)
    
    def execute_eclipse_finder(self, propagated_trajectories: PropagationResults) -> EclipseFinderResults:
        """Calculate eclipse periods for all spacecrafts in the mission.

        Args:
            propagated_trajectories (PropagationResults):
                Propagation results containing spacecraft IDs and their corresponding StateSeries.

        Returns:
            EclipseFinderResults: Eclipse finding results.
        """
        eclipse_finder = EclipseFinder()
        spacecraft_ids: List[str] = []
        eclipse_info: List[StateSeries] = []
        for spc_id, trajectory in propagated_trajectories:
            result = eclipse_finder.execute(frame_graph=self.frame_graph, state=trajectory)
            spacecraft_ids.append(spc_id)
            eclipse_info.append(result)
        return EclipseFinderResults(spacecraft_id=spacecraft_ids, eclipse_info=eclipse_info)

    def execute_gs_contact_finder(self, propagated_trajectories: PropagationResults) -> Dict[str, Dict[str, ContactInfo]]:
        """Calculate contact periods between spacecrafts and ground stations.

        Args:
            propagated_trajectories (PropagationResults): PropagationResults instance.
        Returns:
            Dict[str, Dict[str, ContactInfo]]: Nested dictionary mapping spacecraft IDs to a dict that maps ground-station IDs to ContactInfo.
        """
        if self.ground_stations is None:
            raise ValueError("No ground stations specified for contact finding.")

        contact_info: Dict[str, Dict[str, ContactInfo]] = {}
        for spc_id, trajectory in propagated_trajectories:
            contact_info[spc_id] = {}
            for gs in self.ground_stations:
                result = calculate_gs_contact(
                    trajectory=trajectory,
                    ground_station=gs,
                    frame_graph=self.frame_graph,
                )
                contact_info[spc_id][gs.identifier] = result
        return contact_info

    def execute_coverage_calculator(self, propagated_trajectories: PropagationResults) -> Dict[str, Dict[str, DiscreteCoverageTP]]:
        """Calculate coverage for spacecraft sensors over specified grid points.

        Args:
            propagated_trajectories (PropagationResults): PropagationResults instance.
        Returns:
            Dict[str, Dict[str, DiscreteCoverageTP]]: Nested dictionary mapping spacecraft IDs to a dict that maps sensor IDs to their coverage information.
        """
        coverage_calculator = PointCoverage()

        # Add OrbitPy LVLH frame transformation based on the first spacecraft's trajectory
        eci_frame = ReferenceFrame.get("ICRF_EC")
        lvlh_frame = ReferenceFrame.get("ORBITPY_LVLH")
        
        if self.grid_points is None:
            raise ValueError("No grid points specified for coverage calculation.")
        
        coverage_info: Dict[str, Dict[str, DiscreteCoverageTP]] = {}
        for spc_id, trajectory in propagated_trajectories:
            # Associate sensors with the spacecraft
            spacecraft = next((sc for sc in self.spacecrafts if sc.identifier == spc_id), None)
            if spacecraft is None:
                continue
            times = trajectory.time
            coverage_info[spc_id] = {}
            for sensor in spacecraft.sensor:
                att_lvlh, pos_lvlh = get_lvlh(trajectory, lvlh_frame)
                self.frame_graph.add_orientation_transform(att_lvlh)
                self.frame_graph.add_pos_transform(from_frame=eci_frame, to_frame=lvlh_frame, position=pos_lvlh)
                result = coverage_calculator.calculate_coverage(
                                    target_point_array=self.grid_points,                      
                                    fov=sensor.fov,
                                    frame_graph=self.frame_graph,
                                    times=times,
                                    surface=SurfaceType.SPHERE,
                            )
                coverage_info[spc_id][sensor.identifier] = result
        return coverage_info

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