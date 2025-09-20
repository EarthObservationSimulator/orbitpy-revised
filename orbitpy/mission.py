"""
.. module:: orbitpy.mission
   :synopsis: Module to handle mission initialization and execution.

Collection of classes and functions relating to represention of 
mission parameters and execution.
"""
from typing import Dict, Any, Union, List

from eosimutils.time import AbsoluteDate
from eosimutils.state import Cartesian3DPositionArray
from eosimutils.trajectory import StateSeries
from eosimutils.framegraph import FrameGraph

from .propagator import PropagatorFactory, SGP4Propagator
from .resources import Spacecraft, GroundStation, Sensor
from .eclipsefinder import EclipseFinder, EclipseInfo

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

# def calculate eclipses for a spacecraft


# def calculate contact with ground-station


# def calculate coverage spacecraft + sensor id, and a grid
# 


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

        self.frame_graph = FrameGraph()

    @classmethod
    def from_dict(cls, dict_in: Dict[str, Any]) -> "Mission":
        """Create a Mission object from a dictionary.

        Args:
            dict_in (Dict[str, Any]): Dictionary containing mission specifications.
        Returns:
            Mission: An instance of the Mission class.
        """
        start_time = AbsoluteDate.from_dict(dict_in["start_time"])
        duration_days = dict_in["duration_days"]
        spacecrafts = dict_in["spacecrafts"]
        if not isinstance(spacecrafts, list):
            spacecrafts = [spacecrafts]
        spacecrafts = [Spacecraft.from_dict(sc) for sc in spacecrafts]
        ground_stations = dict_in.get("ground_stations", None)
        if ground_stations is not None:
            if not isinstance(ground_stations, list):
                ground_stations = [ground_stations]
            ground_stations = [GroundStation.from_dict(gs) for gs in ground_stations]
        propagator_specs = dict_in.get("propagator", None)
        propagator = None
        if propagator_specs is not None:
            propagator = PropagatorFactory.from_dict(propagator_specs)
        grid_points_specs = dict_in.get("grid_points", None)
        grid_points = None
        if grid_points_specs is not None:
            grid_points = Cartesian3DPositionArray.from_dict(grid_points_specs)
        return cls(
            start_time=start_time,
            duration_days=duration_days,
            spacecrafts=spacecrafts,
            ground_stations=ground_stations,
            propagator=propagator,
            grid_points=grid_points,
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