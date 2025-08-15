"""
.. module:: orbitpy.eclipsefinder
    :synopsis: Module to find eclipse times, i.e. the times at which a location
                (corresponding to a satellite or ground station)
                is in the shadow of Earth.
"""

from typing import Union, Optional
import spiceypy as spice
import numpy as np

from eosimutils.base import ReferenceFrame, EARTH_POLAR_RADIUS
from eosimutils.spicekernels import load_spice_kernels
from eosimutils.framegraph import FrameGraph
from eosimutils.time import AbsoluteDate, AbsoluteDateArray
from eosimutils.state import (
    CartesianState,
    GeographicPosition,
    Cartesian3DPosition,
)
from eosimutils.trajectory import StateSeries, PositionSeries


class EclipseFinder:
    """Class to calculate eclipse times for a location."""

    def __init__(self):
        """Initialize the EclipseFinder class."""
        pass

    @staticmethod
    def normalize(v: Union[list[float], np.ndarray]) -> np.ndarray:
        """Normalize an input vector.
        Args:
            v (Union[list[float], np.ndarray]): Input vector to be normalized.
        Returns:
            list[float]: Normalized vector.
        Raises:
            Exception: If the input vector has zero magnitude, resulting in division by zero.
        """
        v = np.array(v)
        norm = np.linalg.norm(v)
        if norm == 0:
            raise Exception(
                "Encountered division by zero in vector normalization function."
            )
        return v.tolist() / norm

    @staticmethod
    def check_line_of_sight(
        object1_pos: np.ndarray, object2_pos: np.ndarray, obstacle_radius: float
    ) -> bool:
        """
        Determine if line-of-sight exists between two objects with a spherical obstacle in between.

        This method uses the algorithm described on Page 198 of "Fundamentals of Astrodynamics 
        and Applications" by David A. Vallado (first algorithm).

        Args:
            object1_pos (np.ndarray): Position vector of the first object.
            object2_pos (np.ndarray): Position vector of the second object.
            obstacle_radius (float): Radius of the spherical obstacle.

        Returns:
            bool: True if line-of-sight exists, False otherwise.

        Note:
            The frame of reference for describing the object positions must be centered at 
            the spherical obstacle.
        """
        obj1_unitVec = EclipseFinder.normalize(object1_pos)
        obj2_unitVec = EclipseFinder.normalize(object2_pos)

        # This condition tends to give a numerical error, so solve for it independently.
        eps = 1e-9
        x = np.dot(obj1_unitVec, obj2_unitVec)

        if (x > -1 - eps) and (x < -1 + eps):
            return False
        else:
            if x > 1:
                x = 1
            theta = np.arccos(x)

        obj1_r = np.linalg.norm(object1_pos)
        if obj1_r - obstacle_radius > 1e-5:
            theta1 = np.arccos(obstacle_radius / obj1_r)
        elif abs(obj1_r - obstacle_radius) < 1e-5:
            theta1 = 0.0
        else:
            return False  # object1 is inside the obstacle

        obj2_r = np.linalg.norm(object2_pos)
        if obj2_r - obstacle_radius > 1e-5:
            theta2 = np.arccos(obstacle_radius / obj2_r)
        elif abs(obj2_r - obstacle_radius) < 1e-5:
            theta2 = 0.0
        else:
            return False  # object2 is inside the obstacle

        if theta1 + theta2 < theta:
            return False
        else:
            return True

    @staticmethod
    def _in_eclipse_single_item(et: float, e2o: np.ndarray) -> bool:
        """Check if the object is in eclipse at a single time point.

        All evaluations are done in the ICRF_EC frame.

        Args:
            et (float): Ephemeris Time (ET) as defined by SPICE.
            e2o (np.ndarray): Position of the object in the Earth-centered inertial frame (ICRF_EC).

        Returns:
            bool: True if the object is in eclipse, False otherwise.
        """

        # Check if the object is inside Earth
        if e2o[0] ** 2 + e2o[1] ** 2 + e2o[2] ** 2 < EARTH_POLAR_RADIUS**2:
            # The object is inside the Earth, so it is eclipsed by the surface of Earth
            return True

        # Get Sun position relative to Earth (in the J2000 frame ~ ICRF frame in SPICE) 
        # at the given time.
        # See `eosimutils.base.ReferenceFrame` for more details on the reference frame.
        earth_to_sun, _ = spice.spkpos("SUN", et, "J2000", "LT+S", "EARTH")

        return not EclipseFinder.check_line_of_sight(
            earth_to_sun, e2o, EARTH_POLAR_RADIUS
        )

    @staticmethod
    def _in_eclipse_array(
        et_array: np.ndarray, earth_to_obj_array: np.ndarray
    ) -> np.ndarray:
        """
        Wrapper for the _in_eclipse function which acts on array inputs.

        Args:
            et_array (np.ndarray): Array of ephemeris times (1D).
            earth_to_obj_array (np.ndarray): Array of positions relative to Earth (Nx3).

        Returns:
            np.ndarray: Boolean array indicating whether each object is in eclipse.
        """
        # Validate input dimensions
        if len(et_array) != earth_to_obj_array.shape[0]:
            raise ValueError(
                "et_array and earth_to_obj_array must have the same length."
            )

        # Apply the existing _in_eclipse function to each pair of (et, earth_to_obj)
        in_eclipse_results = np.array(
            [
                EclipseFinder._in_eclipse_single_item(et, earth_to_obj)
                for et, earth_to_obj in zip(et_array, earth_to_obj_array)
            ]
        )
        return in_eclipse_results

    def execute(
        self,
        frame_graph: FrameGraph,
        time: Optional[Union[AbsoluteDate, AbsoluteDateArray]] = None,
        position: Optional[
            Union[GeographicPosition, Cartesian3DPosition]
        ] = None,
        state: Optional[
            Union[StateSeries, PositionSeries, CartesianState]
        ] = None,
    ) -> Union[bool, list[bool]]:
        """Run the eclipse detection algorithm with Earth (spherical model with the
        radius as the Polar radius) as the occluding body.

        The method is simplistic, and evaluates the line-of-sight from the Sun (as a point-source) to
        the object, with a spherical Earth as occluder.
        (The method does not evaluate the umbra, penumbra or antiumbra conditions.)
        The polar-radius of Earth is used, instead of the equatorial or the mean radius to prevent errors
        in the checks involving testing presence of objects inside the Earth.

        Either 'time' and 'position' or 'state' must be provided.

        References:
            https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/spkpos_c.html
            https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/FORTRAN/spicelib/npedln.html

        Args:
            frame_graph (FrameGraph): The frame graph containing transformations between reference frames.
            time (Optional[Union[AbsoluteDate, AbsoluteDateArray]]): Time point or array of time points to 
                                                            evaluate the eclipse condition (default is None).
            position (Optional[Union[GeographicPosition, Cartesian3DPosition]]): The location's geographic or 
                                                                        Cartesian position (default is None).
            state (Optional[Union[StateSeries, PositionSeries, CartesianState]]): The state containing time(s) 
                                                    and position(s) for eclipse evaluation (default is None).

        Returns:
            Union[bool, list[bool]]: (List of) True or False indicating whether the location is in eclipse 
                                    at the specified times.
        """
        # Validate input arguments
        if (time is None or position is None) and state is None:
            raise ValueError(
                "Either 'time' and 'position' or 'state' must be provided."
            )
        if (
            (time is not None and position is not None)
            and state is not None
            or (time is not None and position is None)
            and state is not None
            or (time is None and position is not None)
            and state is not None
        ):
            raise ValueError(
                "Only one of 'time' and 'position' or 'state' should be provided."
            )

        # Load SPICE kernel files
        load_spice_kernels()

        # Helper function to transform position vectors to the ICRF_EC frame
        def in_ICRF_EC_frame(frame_graph, from_frame, input_pos_vector, times):
            """Get the input position vector in the ICRF_EC frame."""
            to_frame = ReferenceFrame.get("ICRF_EC")
            rot_array, _ = frame_graph.get_orientation_transform(
                from_frame, to_frame, times
            )
            return rot_array.apply(input_pos_vector)

        e2o_frame = None
        # Handle time and position inputs
        if (time is not None) and (position is not None):
            if isinstance(position, GeographicPosition):
                cartesian_position = position.to_cartesian3d_position()
                e2o_vector = cartesian_position.to_numpy()
                e2o_frame = cartesian_position.frame
            elif isinstance(position, Cartesian3DPosition):
                e2o_vector = position.to_numpy()
                e2o_frame = position.frame
            else:
                raise TypeError(
                    "Position must be of type GeographicPosition or Cartesian3DPosition."
                )

            if isinstance(time, AbsoluteDate):
                pass
            elif isinstance(time, AbsoluteDateArray):
                e2o_vector = np.tile(e2o_vector, (time.length, 1))

        # Handle state input
        if state is not None:
            if isinstance(state, StateSeries):
                time = state.time
                e2o_vector = state.position.to_numpy()
                e2o_frame = state.frame
            elif isinstance(state, PositionSeries):
                time = state.time
                e2o_vector = state.position.to_numpy()
                e2o_frame = state.frame
            elif isinstance(state, CartesianState):
                time = state.time
                e2o_vector = state.position.to_numpy()
                e2o_frame = state.frame
            else:
                raise TypeError(
                    "State must be of type StateSeries, PositionSeries, or CartesianState."
                )

        # Transform position vectors to the ICRF_EC frame
        e2o_in_icrf_ec = in_ICRF_EC_frame(
            frame_graph, e2o_frame, e2o_vector, time
        )
        e2o_et = time.to_spice_ephemeris_time()

        # Determine eclipse condition
        if isinstance(e2o_et, np.ndarray):
            # If et is an array, call the array version of the function
            return EclipseFinder._in_eclipse_array(e2o_et, e2o_in_icrf_ec)
        else:
            # If et is a single value, call the single item version of the function
            return EclipseFinder._in_eclipse_single_item(e2o_et, e2o_in_icrf_ec)
