"""
.. module:: orbitpy.eclipsefinder
    :synopsis: Module to find eclipse times, i.e. the times at which a location
                (corresponding to a satellite or a ground location)
                is in the shadow of Earth.
"""

from typing import Union, Optional
import spiceypy as spice
import numpy as np


from eosimutils.base import ReferenceFrame, WGS84_EARTH_POLAR_RADIUS
from eosimutils.spicekernels import load_spice_kernels
from eosimutils.framegraph import FrameGraph
from eosimutils.time import AbsoluteDate, AbsoluteDateArray
from eosimutils.state import (
    CartesianState,
    GeographicPosition,
    Cartesian3DPosition,
)
from eosimutils.timeseries import Timeseries, _group_contiguous
from eosimutils.trajectory import StateSeries, PositionSeries

from .utils import check_line_of_sight


class EclipseInfo(Timeseries):
    """
    A class to store the results of the eclipse-finder execute function.

    This class inherits from the Timeseries class and ensures that the data stored
    is an array of booleans representing times of eclipse.

    Attributes:
        time (AbsoluteDateArray): Time values provided as an AbsoluteDateArray object.
        data (list): A list containing a single numpy array of booleans. 'T' indicates eclipse.
                     'F' indicates no eclipse.
        headers (list): A list containing headers for the data array.
    """

    def __init__(self, time: AbsoluteDateArray, data: np.ndarray):
        """
        Initialize a EclipseInfo instance.

        Args:
            time (AbsoluteDateArray): Time values provided as an AbsoluteDateArray object.
            data (np.ndarray): A numpy array of booleans representing eclipse.

        Raises:
            TypeError: If `data` is not a numpy array of booleans.
        """
        headers = [["eclipse"]]
        if not isinstance(data, np.ndarray) or not np.issubdtype(
            data.dtype, np.bool_
        ):
            raise TypeError("data must be a numpy array of booleans.")
        super().__init__(time, [data], headers)

    def is_eclipsed(self, index: int = None) -> Union[bool, None]:
        """
        Check if there is eclipse in the data, or at a specific index.

        Args:
            index (int, optional): The index corresponding to a specific time.
                                    If None, checks for any eclipse.

        Returns:
            bool: True if there is eclipse at the specified index or at least one eclipse exists.
            None: If the index is out of bounds.
        """
        if index is None:
            return np.any(self.data[0])
        if 0 <= index < len(self.data[0]):
            return self.data[0][index]
        return None

    def eclipse_intervals(self) -> list:
        """
        Get the time intervals where eclipse exist.

        Returns:
            list: A list of tuples representing the start and end times of eclipse intervals.
        """
        eclipse_indices = np.where(self.data[0])[0]
        groups = _group_contiguous(eclipse_indices)
        intervals = [
            (self.time[i[0]], self.time[i[-1]]) for i in groups if len(i) > 0
        ]
        return intervals


class EclipseFinder:
    """Class to calculate eclipse times for a location."""

    def __init__(self):
        """Initialize the EclipseFinder class."""
        pass

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
        # Get Sun position relative to Earth (in the J2000 frame ~ ICRF frame in SPICE)
        # at the given time.
        # See `eosimutils.base.ReferenceFrame` for more details on the reference frame.
        earth_to_sun, _ = spice.spkpos("SUN", et, "J2000", "LT+S", "EARTH")

        return not check_line_of_sight(
            earth_to_sun, e2o, WGS84_EARTH_POLAR_RADIUS
        )

    @staticmethod
    def _in_eclipse_array(
        et_array: np.ndarray, earth_to_obj_array: np.ndarray
    ) -> np.ndarray:
        """
        Wrapper for the _in_eclipse function which acts on array inputs.

        Args:
            et_array (np.ndarray): Array of ephemeris times (1D).
            earth_to_obj_array (np.ndarray): Array of 'N' positions relative to Earth (Nx3).

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
        interpolator: str = "linear",
    ) -> EclipseInfo:
        """Run the eclipse detection algorithm with Earth (spherical model with the
        radius as the Polar radius) as the occluding body.

        The method is simplistic, and evaluates the line-of-sight from the Sun
        (as a point-source) to the object, with a spherical Earth as occluder.
        (The method does not evaluate the umbra, penumbra or antiumbra conditions.)
        The polar-radius of Earth is used (largest dimension), instead of the equatorial
        or the mean radius to prevent errors in the checks involving testing presence of
        objects inside the Earth.

        The following combinations of inputs are supported:
            1. 'time' and 'position' provided as inputs. Eclipse is evaluated at the given time(s) for the input position.
            2. 'state' provided as input. Eclipse is evaluated at the time(s) and position(s) in the state/ state-series.
            3. 'time' and 'state' provided as inputs. Eclipse is evaluated at the given time(s) for the position(s) in the state(series).
                                                        The time(s) in the state(series) are ignored. 

        References:
            https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/spkpos_c.html
            https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/FORTRAN/spicelib/npedln.html

        Args:
            frame_graph (FrameGraph): The frame graph containing transformations between
                                        reference frames.
            time (Optional[Union[AbsoluteDate, AbsoluteDateArray]]): Time point or array
                        of time points to evaluate the eclipse condition (default is None).
            position (Optional[Union[GeographicPosition, Cartesian3DPosition]]): The location's
                                    geographic or Cartesian position (default is None).
            state (Optional[Union[StateSeries, PositionSeries, CartesianState]]): The state
                    containing time(s) and position(s) for eclipse evaluation (default is None).
            interpolator (str, optional): Interpolation method used when resampling state data
                    onto the requested time grid (default is "linear").

        Returns:
            Union[bool, list[bool]]: (List of) True or False indicating whether the location
                                                        is in eclipse at the specified times.
        """
        # Validate input arguments
        # Supported:
        # 1. 'time' and 'position' provided (state is None)
        # 2. 'state' provided (time and position are None)
        # 3. 'time' and 'state' provided (position is None)
        if (
            (time is not None and position is not None and state is None)
            or (time is None and position is None and state is not None)
            or (time is not None and position is None and state is not None)
        ):
            pass  # valid combinations
        else:
            raise ValueError(
                "Supported input combinations are: "
                "1) 'time' and 'position', "
                "2) 'state', "
                "3) 'time' and 'state'."
            )

        # Load SPICE kernel files
        load_spice_kernels()

        # Helper function to normalize time input as AbsoluteDateArray
        def _as_absolute_date_array(
            time_value: Union[AbsoluteDate, AbsoluteDateArray]
        ) -> AbsoluteDateArray:
            """Normalize a time input to an AbsoluteDateArray."""
            if isinstance(time_value, AbsoluteDateArray):
                return time_value
            if isinstance(time_value, AbsoluteDate):
                return AbsoluteDateArray(
                    np.array([time_value.to_spice_ephemeris_time()], dtype=float)
                )
            raise TypeError(
                "time must be an AbsoluteDate or AbsoluteDateArray when provided."
            )

        # Helper function to transform position vectors to the ICRF_EC frame
        def in_icrf_ec_frame(frame_graph, from_frame, input_pos_vector, times):
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
                e2o_frame = state.frame
                if time is None:
                    time = state.time
                    e2o_vector = state.position.to_numpy()
                else:
                    eval_time = _as_absolute_date_array(time)
                    resampled_state = state.resample(
                        eval_time, method=interpolator
                    )
                    time = resampled_state.time
                    e2o_vector = resampled_state.position.to_numpy()
            elif isinstance(state, PositionSeries):
                e2o_frame = state.frame
                if time is None:
                    time = state.time
                    e2o_vector = state.position.to_numpy()
                else:
                    eval_time = _as_absolute_date_array(time)
                    resampled_series = state.resample(
                        eval_time, method=interpolator
                    )
                    time = resampled_series.time
                    e2o_vector = resampled_series.position.to_numpy()
            elif isinstance(state, CartesianState):
                e2o_frame = state.frame
                state_position = state.position.to_numpy()
                if time is None:
                    time = state.time
                    e2o_vector = state_position
                else:
                    eval_time = _as_absolute_date_array(time)
                    time = eval_time
                    e2o_vector = np.tile(state_position, (eval_time.length, 1)) # the position is assumed constant over the eval times
            else:
                raise TypeError(
                    "State must be of type StateSeries, PositionSeries, or CartesianState."
                )

        # Transform position vectors to the ICRF_EC frame
        e2o_in_icrf_ec = in_icrf_ec_frame(
            frame_graph, e2o_frame, e2o_vector, time
        )
        e2o_et = time.to_spice_ephemeris_time()

        # Determine eclipse condition
        if isinstance(e2o_et, np.ndarray):
            # If et is an array, call the array version of the function
            eclipses = EclipseFinder._in_eclipse_array(e2o_et, e2o_in_icrf_ec)
            return EclipseInfo(AbsoluteDateArray(e2o_et), eclipses)
        else:
            # If et is a single value, call the single item version of the function
            eclipses = EclipseFinder._in_eclipse_single_item(
                e2o_et, e2o_in_icrf_ec
            )
            return EclipseInfo(
                AbsoluteDateArray(np.array([e2o_et])), np.array([eclipses])
            )
