"""Estimate CYGNSS drag coefficient with GLSDC nonlinear least squares.

Solves for the drag coefficient using an ~3 day measurement arc consisting
of CYGNSS reference position data, which is sampled every 300 seconds.
The initial state is initialized using a TLE from celestrak. 

After four iterations, this solves for a drag coefficient of -2.36. The negative
drag coefficient probably implies that the CYGNSS reference velocity data
is not very accurate, and this error is being absorbed into the drag coefficient.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional
import os

import numpy as np
import matplotlib.pyplot as plt

from eosimutils.time import AbsoluteDate, AbsoluteDateArray
from eosimutils.trajectory import StateSeries
from eosimutils.base import ReferenceFrame
from eosimutils.framegraph import FrameGraph
from eosimutils.state import CartesianState

import orbitpy.orekitpropagator  # triggers decorator registration
from orbitpy.propagator import PropagatorFactory
from orbitpy.orbits import SpaceTrackAPI, OrbitalMeanElementsMessage

from orbitpy.glsdc import GLSDC, FiniteDifferenceJacobian, GLSDCResults, VectorFunction

def convert_datetime(np_dt):
    arr = np.atleast_1d(np_dt)

    if not np.issubdtype(arr.dtype, np.datetime64):
        raise TypeError("np_dt must be a numpy.datetime64 scalar or array")

    arr_ms = arr.astype("datetime64[ms]")
    iso = np.datetime_as_string(arr_ms, unit="ms", timezone="naive")
    iso_list = [str(iso)] if np.isscalar(iso) else iso.astype(str).tolist()

    return AbsoluteDateArray.from_dict(
        {
            "time_format": "GREGORIAN_DATE",
            "calendar_date": iso_list,
            "time_scale": "UTC",
        }
    )

def main():

    M_TO_KM = 1.0 / 1000.0

    norad_id = 41891

    file_dir = os.path.dirname(os.path.abspath(__file__))

    cyg_file = os.path.join(
        file_dir,
        "cygnss_data",
        "cygnss_spacecraft_trajectory_combined.csv",
    )

    credentials_file = os.path.join(
        file_dir,
        "spacetrack",
        "credentials.json",
    )

    # -------------------------------------------------------------------------
    # Read CYGNSS truth trajectory
    # -------------------------------------------------------------------------

    cyg_pos = np.loadtxt(
        cyg_file,
        delimiter=",",
        skiprows=1,
        usecols=(1, 2, 3),
    ) * M_TO_KM

    cyg_vel = np.loadtxt(
        cyg_file,
        delimiter=",",
        skiprows=1,
        usecols=(4, 5, 6),
    ) * M_TO_KM

    ddm_timestamp_utc = np.loadtxt(
        cyg_file,
        delimiter=",",
        skiprows=1,
        usecols=(0,),
        dtype="datetime64[ns]",
    )

    original_datearray = convert_datetime(ddm_timestamp_utc)

    sc_stateseries = StateSeries(
        time=original_datearray,
        data=[cyg_pos, cyg_vel],
        frame=ReferenceFrame.get("ITRF"),
    )

    # Transform reference data to inertial frame
    registry = FrameGraph()
    to_frame = ReferenceFrame.get("ICRF_EC")
    sc_stateseries_icrf_full = registry.transform_series(sc_stateseries, to_frame)

    start_time = original_datearray[0]
    stop_time = original_datearray[-1]
    # 1/2 day from start point of reference data
    # This time point is used to retrieve the closest OMM from Space-Track
    mid_time = original_datearray[86400*1]

    # -------------------------------------------------------------------------
    # Build propagator
    # -------------------------------------------------------------------------

    specs = {
        "propagator_type": "OREKIT_PROPAGATOR",
        "stepSize": 10,
    }

    prop = PropagatorFactory.from_dict(specs)

    # -------------------------------------------------------------------------
    # Get closest OMM from Space-Track
    # -------------------------------------------------------------------------

    date_str_stop = mid_time.to_dict(time_format="Gregorian_Date")["calendar_date"]

    api = SpaceTrackAPI(credentials_file)
    api.login()

    omm_dict = api.get_closest_omm(
        norad_id=norad_id,
        target_date_time=date_str_stop,
        within_days=1,
    )

    if not omm_dict:
        raise RuntimeError("Could not retrieve OMM from Space-Track.")

    tle_epoch = AbsoluteDate.from_dict(
        {
            "time_scale": omm_dict["TIME_SYSTEM"],
            "time_format": "GREGORIAN_DATE",
            "calendar_date": omm_dict["EPOCH"],
        }
    )
    init_time = tle_epoch

    omm = OrbitalMeanElementsMessage.from_dict(omm_dict)
    init_state = omm

    # Uncomment to run code using initial state from cygnss data instead of SGP4
    # init_time = start_time
    # init_state = CartesianState.from_array(np.concatenate((cyg_pos[0],cyg_vel[0])),init_time,"ITRF")
    # init_state = registry.transform(init_state,sc_stateseries_icrf_full.frame,init_time)

    # -------------------------------------------------------------------------
    # Define measurement epochs
    # -------------------------------------------------------------------------

    # Downsample to 300 second timestep for measurements
    datearray = AbsoluteDateArray.linspace(init_time, stop_time, 300)

    # Resample reference trajectory onto propagation epochs
    sc_stateseries_icrf = sc_stateseries_icrf_full.resample(datearray)

    # Reference positions: shape (M, 3), where M is number of measurements
    y_reference_positions = sc_stateseries_icrf.data[0]

    # GLSDC code requires column vector: shape (3M, 1)
    y = y_reference_positions.reshape(-1, 1)

    # -------------------------------------------------------------------------
    # Define measurement model F(x)
    # -------------------------------------------------------------------------

    # Function which maps solved-for variable x to the measurements
    # which will result from that value of x
    def F(x: np.ndarray) -> np.ndarray:
        drag_coeff = float(np.asarray(x).reshape(-1)[0])

        prop.set_drag_coeff(drag_coeff)

        propagated_state_series = prop.execute_2(
            times=datearray,
            initial_state=init_state,
        )

        propagated_positions = propagated_state_series.data[0]

        return propagated_positions.reshape(-1, 1)

    # -------------------------------------------------------------------------
    # Solve for drag coefficient
    # -------------------------------------------------------------------------

    # Initial guess for drag coeff
    x0 = np.array([[300.0]])

    solver = GLSDC(
        max_iters=4,
        tol=1.0e-2,
        fd_eps=1.0e-4,
    )

    results = solver.solve(
        F=F,
        x0=x0,
        y=y,
        R=None,
        verbose=True,
    )

    cd_hat = float(results.x_hat[0, 0])
    cd_sigma = float(np.sqrt(results.covariance[0, 0]))

    print()
    print("=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Converged:                  {results.converged}")
    print(f"Iterations:                 {results.iterations}")
    print(f"Estimated drag coefficient: {cd_hat:.16e}")
    print(f"1-sigma uncertainty: {cd_sigma:.16e}")
    print(f"Final cost:                 {results.cost:.16e}")
    print(f"Final RMS position resid:   {np.sqrt(np.mean(results.residual ** 2)):.16e} km")

    # -------------------------------------------------------------------------
    # Plot final residuals
    # -------------------------------------------------------------------------

    final_prop_positions = results.y_hat.reshape((-1, 3))
    diff_pos = final_prop_positions - y_reference_positions
    error_norm = np.linalg.norm(diff_pos, axis=1)

    time_days = datearray.ephemeris_time * (1.0 / 86400.0)

    plt.figure()
    plt.plot(time_days, diff_pos[:, 0], label="X")
    plt.plot(time_days, diff_pos[:, 1], label="Y")
    plt.plot(time_days, diff_pos[:, 2], label="Z")
    plt.xlabel("Ephemeris Time (days)")
    plt.ylabel("Position Residual (km)")
    plt.title(f"Final Position Residuals, Cd = {cd_hat:.6f}")
    plt.legend()
    plt.grid(True)

    plt.figure()
    plt.plot(time_days, error_norm)
    plt.xlabel("Ephemeris Time (days)")
    plt.ylabel("Position Error Norm (km)")
    plt.title("Final Position Error Norm")
    plt.grid(True)

    plt.figure()
    plt.plot([float(x[0, 0]) for x in results.x_history], marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Drag Coefficient")
    plt.title("Drag Coefficient Convergence")
    plt.grid(True)

    plt.show()

if __name__ == "__main__":
    main()