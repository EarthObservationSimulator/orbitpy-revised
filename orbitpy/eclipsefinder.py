"""
.. module:: orbitpy.eclipsefinder
    :synopsis: Module to find eclipse times, i.e. the times at which a location 
                (corresponding to a satellite or ground station)
                is in the shadow of Earth.


"""
from typing import Type, Dict, Any, Union, Optional
import spiceypy as spice
import numpy as np

from eosimutils.base import ReferenceFrame, EARTH_RADIUS, EARTH_POLAR_RADIUS
from eosimutils.spicekernels import load_spice_kernels
from eosimutils.framegraph import FrameGraph
from eosimutils.time import AbsoluteDate,AbsoluteDateArray, AbsoluteDateIntervalArray
from eosimutils.state import CartesianState, GeographicPosition, Cartesian3DPosition
from eosimutils.trajectory import StateSeries, PositionSeries

class EclipseFinder:
    """ Class to calculate eclipse times for a location."""

    def __init__(self):
        pass

    @staticmethod
    def _in_eclipse_single_item(et, earth_to_obj) -> bool:
        """Check if the object is in eclipse at a single time point.

        Args:
            et (float): Ephemeris Time (ET) as defined by SPICE.
            earth_to_obj (np.ndarray): Position of the object in the Earth-centered inertial frame (ICRF_EC).

        Returns:
            bool: True if the object is in eclipse, False otherwise.
        """

        # Define the ellipsoid parameters for Earth
        A = EARTH_RADIUS
        B = EARTH_RADIUS
        C = EARTH_POLAR_RADIUS

        # check if object is inside Earth
        if (earth_to_obj[0]/A)**2 + (earth_to_obj[1]/B)**2 + (earth_to_obj[2]/C)**2 < 1.0:
            #print("The object is inside the Earth.")
            return False

        # Get Sun position relative to Earth (in the J2000 frame ~ ICRF in SPICE) at given time
        # See `eosimutils.base.ReferenceFrame` for more details on the reference frame.
        earth_to_sun, _ = spice.spkpos("SUN", et, "J2000", "LT+S", "EARTH")
        #print(f"Sun position wrt Earth at {et}: {earth_to_sun}")

        sun_to_obj = np.subtract(earth_to_obj, earth_to_sun)

        # NPEDLN ( Nearest point on ellipsoid to line )
        pnear, dist = spice.npedln(A, B, C, earth_to_obj, sun_to_obj/np.linalg.norm(sun_to_obj))

        #print(pnear, dist)

        if dist == 0.0:
            #print("The object is eclipsed by the Earth.")
            return True
        else:
            #print("The object is not eclipsed by the Earth.")
            return False
    
    @staticmethod
    def _in_eclipse_array(et_array, earth_to_obj_array):
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
            raise ValueError("et_array and earth_to_obj_array must have the same length.")

        # Apply the existing _in_eclipse function to each pair of (et, earth_to_obj)
        in_eclipse_results = np.array([
            EclipseFinder._in_eclipse_single_item(et, earth_to_obj)
            for et, earth_to_obj in zip(et_array, earth_to_obj_array)
        ])
        return in_eclipse_results
        
    @staticmethod
    def _in_eclipse(et, earth_to_obj) -> bool:
        if isinstance(et, np.ndarray):
            # If et is an array, call the array version of the function
            return EclipseFinder._in_eclipse_array(et, earth_to_obj)
        else:
            # If et is a single value, call the single item version of the function
            return EclipseFinder._in_eclipse_single_item(et, earth_to_obj)

    def execute(self, frame_graph: FrameGraph,
                time:Optional[Union[AbsoluteDate, AbsoluteDateArray]]=None, position:Optional[Union[GeographicPosition, Cartesian3DPosition]]=None, 
                state:Optional[Union[StateSeries, PositionSeries, CartesianState]]=None) -> Union[bool, list[bool]]:
        """Run the eclipse detection algorithm with Earth as the occluding body.
        The method is simplistic, and evaluates the line-of-sight from the Sun (as a point-source) to 
        the object, with Earth as occluder. 
        (The method does not evaluate the umbra, penumbra or antiumbra conditions.) 

        Either 'time' and 'position' or 'state' must be provided.

        References:
            https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/spkpos_c.html
            https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/FORTRAN/spicelib/npedln.html
                

        Args:
            frame_graph (FrameGraph): The frame graph containing transformations between reference frames.
            time (Optional[Union[AbsoluteDate, AbsoluteDateArray]]): Time point or array of time points to evaluate the eclipse condition (default is None).
            position (Optional[Union[GeographicPosition, Cartesian3DPosition]]): The location's geographic or Cartesian position (default is None).
            state (Optional[Union[StateSeries, PositionSeries, CartesianState]]): The state containing time(s) and position(s) for eclipse evaluation (default is None).

        Returns:
            Union[bool, list[bool]]: (List of) True or False indicating whether the location is in eclipse at the specified times.
        """
        if (time is None or position is None) and state is None:
            raise ValueError("Either 'time' and 'position' or 'state' must be provided.")
        if (time is not None and position is not None) and state is not None:
            raise ValueError("Only one of 'time' and 'position' or 'state' should be provided.")

        # Load SPICE kernel files
        load_spice_kernels()
        
        # Get the position vector in the ICRF_EC frame
        def in_ICRF_EC_frame(frame_graph, from_frame, pos_vector, times):
            """Get the vector in the ICRF_EC frame"""
            to_frame = ReferenceFrame.get("ICRF_EC")
            rot_array, w_array = frame_graph.get_orientation_transform(
                from_frame,
                to_frame,
                times
            )
            return rot_array.apply(pos_vector)

        if (time is not None) and (position is not None):
            if isinstance(position, GeographicPosition):
                _position = position.to_cartesian3d_position()
                e2o_vector = _position.to_numpy()
                e2o_frame = _position.frame
            elif isinstance(position, Cartesian3DPosition):
                e2o_vector = position.to_numpy()
                e2o_frame = position.frame
            else:
                raise TypeError("Position must be of type GeographicPosition or Cartesian3DPosition.")

            if isinstance(time, AbsoluteDate):
                pass
            elif isinstance(time, AbsoluteDateArray):
                e2o_vector = np.tile(e2o_vector, (time.length, 1))

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
                raise TypeError("State must be of type StateSeries, PositionSeries, or CartesianState.")

        e2o_in_icrf_ec = in_ICRF_EC_frame(frame_graph, e2o_frame, e2o_vector, time)
        e2o_et = time.to_spice_ephemeris_time()
        in_eclipse = EclipseFinder._in_eclipse(e2o_et, e2o_in_icrf_ec)
        return in_eclipse
