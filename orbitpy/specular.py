"""
.. module:: orbitpy.specular
   :synopsis: Module providing code for specular point calculation.

"""

from os import times
from typing import Type, Dict, Any, Union, Optional, Callable
import math
import numpy as np

from eosimutils.trajectory import StateSeries, PositionSeries
from eosimutils.framegraph import FrameGraph
from eosimutils.base import (
    ReferenceFrame,
    SurfaceType,
    WGS84_EARTH_EQUATORIAL_RADIUS,
    SPHERICAL_EARTH_MEAN_RADIUS,
    WGS84_EARTH_POLAR_RADIUS,
)
from eosimutils.time import AbsoluteDateArray

import kcl
import GeometricTools as gte

# -----------------------------------------------
# Some interfaces for specular point calculation:
# -----------------------------------------------

# # Take list of GPS trajectories as input, and compute specular point for strongest pair at each step
# def get_best_specular_trajectory(transmitter: StateSeries, receivers: List[StateSeries],
#                             times: AbsoluteDateArray, surface: SurfaceType = SurfaceType.WGS84) -> PositionSeries:


def get_specular_trajectory(
    transmitter: StateSeries,
    receiver: StateSeries,
    times: AbsoluteDateArray,
    surface: SurfaceType = SurfaceType.WGS84,
) -> PositionSeries:
    """
    Compute the specular point for the given transmitter/receiver pair.

    The specular point will be computed for each timepoint in the input AbsoluteDateArray and
    returned as a PositionSeries. Time points for which there is no specular point (due to lack of
    line of sight) will be stored as NaN vectors in the PositionSeries.

    Args:
        transmitter (StateSeries): The GNSS satellite trajectory in ITRF frame.
        receiver (StateSeries): The Earth observation satellite trajectory in ITRF frame.
        times (AbsoluteDateArray): Times for which to compute specular point.

    Returns:
        PositionSeries: Object which stores computed specular point at each time point.
    """

    # Store all output in memory
    buff_size = len(times)

    # These are control parameters for Newton's method, used to control specular point calculation.
    # They are not very important, since I have never encountered a scenario where the algorithm
    # doesn't converge.
    tol = 1e-10
    max_iter = 20

    transmitter_pos = transmitter.at(times)[0]
    receiver_pos = receiver.at(times)[0]

    transmitter_pos_list = [gte.Vector3d(p) for p in transmitter_pos]
    receiver_pos_list = [gte.Vector3d(p) for p in receiver_pos]

    transmitter_pos_source = kcl.ListSourceVector3d(transmitter_pos_list)
    receiver_pos_source = kcl.ListSourceVector3d(receiver_pos_list)

    if surface == SurfaceType.SPHERE:
        extents = gte.Vector3d(
            [
                SPHERICAL_EARTH_MEAN_RADIUS,
                SPHERICAL_EARTH_MEAN_RADIUS,
                SPHERICAL_EARTH_MEAN_RADIUS,
            ]
        )

        # TODO: Remove this. Used to compare directly with old tests.
        # extents = gte.Vector3d(
        #     [WGS84_EARTH_EQUATORIAL_RADIUS, WGS84_EARTH_EQUATORIAL_RADIUS,WGS84_EARTH_EQUATORIAL_RADIUS]
        # )

        earth_ellipsoid = gte.Ellipsoid3d(gte.Vector3d.Zero(), extents)
    elif surface == SurfaceType.WGS84:
        extents = gte.Vector3d(
            [
                WGS84_EARTH_EQUATORIAL_RADIUS,
                WGS84_EARTH_EQUATORIAL_RADIUS,
                WGS84_EARTH_POLAR_RADIUS,
            ]
        )
        earth_ellipsoid = gte.Ellipsoid3d(gte.Vector3d.Zero(), extents)
    else:
        raise ValueError(f"Unsupported surface type: {surface}")

    earth_source = kcl.ConstantSourceEllipsoid3d(earth_ellipsoid)

    los_source = kcl.LOSEventSourceEllipsoid3d(
        receiver_pos_source, transmitter_pos_source, earth_source, buff_size
    )

    specular_source = kcl.SpecularPointSource(
        transmitter_pos_source,
        receiver_pos_source,
        earth_source,
        los_source,
        buff_size,
        tol,
        max_iter,
    )

    variables = [los_source, specular_source]

    driver = kcl.VarDriver(variables)
    start_time = 0
    stop_time = len(times) - 1
    writers = []
    kcl.driveCoverage(buff_size, start_time, stop_time, driver, writers)

    specular_positions = np.zeros((buff_size, 3))
    for i in range(buff_size):
        specular_point = specular_source.get(i)
        specular_positions[i, :] = [
            specular_point[0],
            specular_point[1],
            specular_point[2],
        ]

    result = PositionSeries(
        data=specular_positions, time=times, frame=transmitter.frame
    )

    return result
