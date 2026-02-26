"""
.. module:: orbitpy.specular
   :synopsis: Module providing code for specular point calculation.

"""

import numpy as np
import copy
from typing import List, Tuple, Optional

from eosimutils.trajectory import StateSeries, PositionSeries
from eosimutils.base import (
    SurfaceType,
    WGS84_EARTH_EQUATORIAL_RADIUS,
    SPHERICAL_EARTH_MEAN_RADIUS,
    WGS84_EARTH_POLAR_RADIUS,
)
from eosimutils.time import AbsoluteDateArray

import kcl
import GeometricTools as gte

def get_best_trajectory(
    traj_list: List[Tuple[PositionSeries, List[float]]],
) -> Tuple[PositionSeries, List[float]]:
    """
    Takes a list of tuples (specular_trajectory, rcg) and returns a single tuple whose
    trajectory/rcg at each time index comes from the entry with the highest RCG at that time.
    specular_trajectory is a timeseries of position data and rcg is a timeseries of range-corrected
    gain values at the same time points.

    Args:
        traj_list (List[Tuple[PositionSeries, List[float]]]): List of tuples containing
            specular trajectory and corresponding RCG list.
    Returns:
        Tuple[PositionSeries, List[float]]: A tuple containing the best specular trajectory
            and corresponding RCG list.
    """

    output_traj, output_rcg = copy.deepcopy(traj_list[0])
    num_timepoints = len(output_traj.time)
    num_lists = len(traj_list)

    for i in range(num_timepoints):
        max_rcg = -np.inf
        max_rcg_idx = 0
        for j in range(0, num_lists):
            rcg_j = traj_list[j][1][i]  # rcg at timepoint i for transmitter j
            if np.isfinite(rcg_j) and rcg_j > max_rcg:
                max_rcg = rcg_j
                max_rcg_idx = j

        output_traj.data[0][i] = traj_list[max_rcg_idx][0].data[0][i]
        output_rcg[i] = max_rcg

    return output_traj, output_rcg

def get_topk_trajectories(
    traj_list: List[Tuple[PositionSeries, List[float]]],
    k: int,
    ids: List[int],
) -> Tuple[List[Tuple[PositionSeries, List[float]]], List[List[int]]]:
    """
    Takes a list of tuples (specular_trajectory, rcg) and returns the top-k trajectories.
    For each rank r (0 <= r < k), the r-th output trajectory/rcg at each time index comes
    from the entry with the r-th highest RCG at that time.

    Also returns, for each timepoint, the k ids selected (in rank order: best -> k-th best).

    Args:
        traj_list: List of (specular_trajectory, rcg) tuples.
        k: Number of top trajectories to return.
        ids: Integer ids corresponding 1:1 with traj_list entries.

    Returns:
        outputs: List of k (PositionSeries, rcg_list) tuples
        selected_ids_by_time: List (length T) of lists (length k) of ids selected at each time.
    """
    if len(ids) != len(traj_list):
        raise ValueError(f"ids must have same length as traj_list (got {len(ids)} vs {len(traj_list)})")
    if len(traj_list) == 0:
        raise ValueError("traj_list must be non-empty")
    if k <= 0:
        raise ValueError("k must be >= 1")

    num_lists = len(traj_list)
    k = min(k, num_lists)

    # Initialize outputs as deep copies of the first k trajectories
    outputs = [copy.deepcopy(traj_list[i]) for i in range(k)]
    num_timepoints = len(outputs[0][0].time)

    selected_ids_by_time: List[List[int]] = [[-1] * k for _ in range(num_timepoints)]

    for i in range(num_timepoints):
        # Collect RCGs at this time index
        rcgs = np.array([traj_list[j][1][i] for j in range(num_lists)], dtype=float)

        # Treat non-finite values as -inf so they sort to the bottom
        rcgs = np.where(np.isfinite(rcgs), rcgs, -np.inf)

        # Indices sorted by descending RCG
        sorted_indices = np.argsort(rcgs)[::-1]

        for r in range(k):
            idx = int(sorted_indices[r])
            outputs[r][0].data[0][i] = traj_list[idx][0].data[0][i]
            outputs[r][1][i] = float(rcgs[idx])
            selected_ids_by_time[i][r] = ids[idx]

    return outputs, selected_ids_by_time

def get_specular_trajectory(
    transmitter_states_itrf: StateSeries,
    receiver_states_itrf: StateSeries,
    times: AbsoluteDateArray,
    surface: SurfaceType = SurfaceType.WGS84,
) -> PositionSeries:
    """
    Compute the specular point for the given transmitter/receiver pair.

    The specular point will be computed for each timepoint in the input AbsoluteDateArray and
    returned as a PositionSeries. Time points for which there is no specular point (due to lack of
    line of sight) will be stored as NaN vectors in the PositionSeries.

    Note that the input transmitter and receiver StateSeries must be in the **ITRF** frame.

    Args:
        transmitter_states_itrf (StateSeries): The GNSS satellite trajectory in ITRF frame.
        receiver_states_itrf (StateSeries): The Earth observation satellite trajectory in ITRF frame.
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

    transmitter_pos = transmitter_states_itrf.at(times)[0]
    receiver_pos = receiver_states_itrf.at(times)[0]

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

    # Returns the range-corrected gain (RCG) factor for the radar.
    radar_gain = 1.0  # Placeholder value
    rcg_source = kcl.RCGSource(
            transmitter_pos_source,
            receiver_pos_source,
            specular_source,
            radar_gain,
            buff_size,
        )


    variables = [los_source, specular_source, rcg_source]

    driver = kcl.VarDriver(variables)
    start_time = 0
    stop_time = len(times) - 1
    writers = []
    kcl.driveCoverage(buff_size, start_time, stop_time, driver, writers)

    rcg_factor = np.zeros(buff_size)
    specular_positions = np.zeros((buff_size, 3))
    for i in range(buff_size):
        specular_point = specular_source.get(i)
        specular_positions[i, :] = [
            specular_point[0],
            specular_point[1],
            specular_point[2],
        ]
        rcg_factor[i] = rcg_source.get(i)

    result = PositionSeries(
        data=specular_positions, time=times, frame=transmitter_states_itrf.frame
    )

    return result, rcg_factor
