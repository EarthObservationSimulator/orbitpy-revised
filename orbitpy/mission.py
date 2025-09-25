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
from typing import Dict, Any, Union, List

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

# parse mission json file
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
        transform_dict = dict_in.get("transforms", None)
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
            frame_graph = FrameGraph.from_dict(transform_dict)
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

    def execute_propagation(self) -> Dict[str, StateSeries]:
        """Propagate all spacecrafts in the mission using the specified propagator.

        Returns:
            Dict[str, StateSeries]: Dictionary mapping spacecraft IDs to their propagated StateSeries.
        """
        if self.propagator is None:
            raise ValueError("No propagator specified for the mission.")
        propagated_trajectories = {}
        for sc in self.spacecrafts:
            trajectory = propagate_spacecraft(
                spacecraft=sc,
                propagator=self.propagator,
                t0=self.start_time,
                duration_days=self.duration_days,
            )
            propagated_trajectories[sc.identifier] = trajectory
        return propagated_trajectories
    
    def execute_eclipse_finder(self, propagated_trajectories: Dict[str, StateSeries]) -> Dict[str, EclipseInfo]:
        """Calculate eclipse periods for all spacecrafts in the mission.

        Args:
            propagated_trajectories (Dict[str, StateSeries]): Dictionary mapping spacecraft IDs to their propagated StateSeries.

        Returns:
            Dict[str, EclipseInfo]: Dictionary mapping spacecraft IDs to their eclipse information.
        """
        eclipse_finder = EclipseFinder()
        eclipse_info = {}
        for spc_id, trajectory in propagated_trajectories.items():
            result = eclipse_finder.execute(
                frame_graph=self.frame_graph, state=trajectory
            )
            eclipse_info[spc_id] = result
        return eclipse_info
    
    def execute_gs_contact_finder(self, propagated_trajectories: Dict[str, StateSeries]) -> Dict[str, Dict[str, ContactInfo]]:
        """Calculate contact periods between spacecrafts and ground stations.

        Args:
            propagated_trajectories (Dict[str, StateSeries]): Dictionary mapping spacecraft IDs to their propagated StateSeries.
        Returns:
            Dict[str, Dict[str, ContactInfo]]: Nested dictionary mapping spacecraft IDs to a dict that maps ground-station IDs to ContactInfo.
        """
        if self.ground_stations is None:
            raise ValueError("No ground stations specified for contact finding.")

        contact_info = {}
        for spc_id, trajectory in propagated_trajectories.items():
            contact_info[spc_id] = {}
            for gs in self.ground_stations:
                result = calculate_gs_contact(
                    trajectory=trajectory,
                    ground_station=gs,
                    frame_graph=self.frame_graph,
                )
                contact_info[spc_id][gs.identifier] = result
        return contact_info

    def execute_coverage_calculator(self, propagated_trajectories: Dict[str, StateSeries]) -> Dict[str, Dict[str, DiscreteCoverageTP]]:
        """Calculate coverage for spacecraft sensors over specified grid points.

        Args:
            propagated_trajectories (Dict[str, StateSeries]): Dictionary mapping spacecraft IDs to their propagated StateSeries.

        Returns:
            Dict[str, Dict[str, DiscreteCoverageTP]]: Nested dictionary mapping spacecraft IDs to a dict that maps sensor IDs to their coverage information.
        """
        coverage_calculator = PointCoverage()

        # Add OrbitPy LVLH frame transformation based on the first spacecraft's trajectory
        eci_frame = ReferenceFrame.get("ICRF_EC")
        lvlh_frame = ReferenceFrame.get("ORBITPY_LVLH")
        
        if self.grid_points is None:
            raise ValueError("No grid points specified for coverage calculation.")
        
        coverage_info = {}
        for spc_id, trajectory in propagated_trajectories.items():
            # Associate sensors with the spacecraft
            spacecraft = next((sc for sc in self.spacecrafts if sc.identifier == spc_id), None)
            times = trajectory.time
            if spacecraft is None:
                continue
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