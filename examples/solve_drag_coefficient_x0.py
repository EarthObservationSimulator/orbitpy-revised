"""Estimate CYGNSS drag coefficient and initial Cartesian state with GLSDC.

Similar to the example solve_drag_coefficient, except it initializes the 
state using the cygnss data rather than SGP4, and also includes the initial
state as a solved-for variable. Hence it achieves a very low position
error over the course of the 3-day arc (max ~.06 km) and a positive drag coefficient
(1.41). 

"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt

from eosimutils.time import AbsoluteDateArray
from eosimutils.trajectory import StateSeries
from eosimutils.base import ReferenceFrame
from eosimutils.framegraph import FrameGraph
from eosimutils.state import CartesianState

import orbitpy.orekitpropagator  # triggers decorator registration
from orbitpy.propagator import PropagatorFactory
from orbitpy.glsdc import GLSDC


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

    file_dir = os.path.dirname(os.path.abspath(__file__))

    cyg_file = os.path.join(
        file_dir,
        "cygnss_data",
        "cygnss_spacecraft_trajectory_combined.csv",
    )

    # -------------------------------------------------------------------------
    # Read CYGNSS reference trajectory
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

    sc_stateseries_itrf = StateSeries(
        time=original_datearray,
        data=[cyg_pos, cyg_vel],
        frame=ReferenceFrame.get("ITRF"),
    )

    # Transform reference trajectory to inertial frame
    registry = FrameGraph()
    inertial_frame = ReferenceFrame.get("ICRF_EC")

    sc_stateseries_icrf_full = registry.transform_series(
        sc_stateseries_itrf,
        inertial_frame,
    )

    init_time = original_datearray[0]
    stop_time = original_datearray[-1]

    # -------------------------------------------------------------------------
    # Build nominal initial state from the first reference state
    # -------------------------------------------------------------------------

    init_state_itrf = CartesianState.from_array(
        np.concatenate((cyg_pos[0], cyg_vel[0])),
        init_time,
        "ITRF",
    )

    init_state_icrf = registry.transform(
        init_state_itrf,
        inertial_frame,
        init_time,
    )

    r0_nominal = init_state_icrf.position.to_numpy()
    v0_nominal = init_state_icrf.velocity.to_numpy()

    # -------------------------------------------------------------------------
    # Build propagator
    # -------------------------------------------------------------------------

    specs = {
        "propagator_type": "OREKIT_PROPAGATOR",
        "step_size": 10,
        "drag_coeff": 2.2
    }

    prop = PropagatorFactory.from_dict(specs)

    # -------------------------------------------------------------------------
    # Measurement epochs and measurements
    # -------------------------------------------------------------------------

    datearray = AbsoluteDateArray.linspace(init_time, stop_time, 300)

    sc_stateseries_icrf = sc_stateseries_icrf_full.resample(datearray)

    y_reference_positions = sc_stateseries_icrf.data[0]
    y = y_reference_positions.reshape(-1, 1)

    def F(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)

        drag_coeff = float(x[0])
        r0 = x[1:4]
        v0 = x[4:7]

        prop.set_drag_coeff(drag_coeff)

        initial_state = CartesianState.from_array(
            np.concatenate((r0, v0)),
            init_time,
            "ICRF_EC",
        )

        propagated_state_series = prop.execute_2(
            times=datearray,
            initial_state=initial_state,
        )

        propagated_positions = propagated_state_series.data[0]

        return propagated_positions.reshape(-1, 1)

    # -------------------------------------------------------------------------
    # Initial guess
    # -------------------------------------------------------------------------

    cd0 = 2.2

    x0 = np.array(
        [
            cd0,
            r0_nominal[0],
            r0_nominal[1],
            r0_nominal[2],
            v0_nominal[0],
            v0_nominal[1],
            v0_nominal[2],
        ],
        dtype=float,
    ).reshape(-1, 1)

    # Scale used only internally by GLSDC.
    #
    # Characteristic magnitudes for each solve variable:
    #   Cd scale:       1
    #   position scale: 1 km
    #   velocity scale: 1e-3 km/s = 1 m/s
    # The measurement model still receives x in physical units.
    x_scale = np.array(
        [
            1.0,
            1.0,
            1.0,
            1.0,
            1.0e-3,
            1.0e-3,
            1.0e-3,
        ],
        dtype=float,
    ).reshape(-1, 1)

    solver = GLSDC(
        max_iters=5,
        tol=1.0e-3,
        fd_eps=1.0e-4,
        x_scale=x_scale,
    )

    results = solver.solve(
        F=F,
        x0=x0,
        y=y,
        R=None,
        verbose=True,
    )

    x_hat = results.x_hat.reshape(-1)

    cd_hat = x_hat[0]
    r0_hat = x_hat[1:4]
    v0_hat = x_hat[4:7]

    dr_hat = r0_hat - r0_nominal
    dv_hat = v0_hat - v0_nominal

    cd_sigma = np.sqrt(results.covariance[0, 0])
    state_sigma = np.sqrt(np.diag(results.covariance))

    print()
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Converged:                    {results.converged}")
    print(f"Iterations:                   {results.iterations}")
    print(f"Estimated drag coefficient:   {cd_hat:.16e}")
    print(f"Cd 1-sigma uncertainty:       {cd_sigma:.16e}")
    print(f"Final cost:                   {results.cost:.16e}")
    print(f"Final RMS position residual:  {np.sqrt(np.mean(results.residual ** 2)):.16e} km")

    print()
    print("Estimated initial state correction:")
    print(f"dr_x: {dr_hat[0]: .16e} km")
    print(f"dr_y: {dr_hat[1]: .16e} km")
    print(f"dr_z: {dr_hat[2]: .16e} km")
    print(f"dv_x: {dv_hat[0]: .16e} km/s")
    print(f"dv_y: {dv_hat[1]: .16e} km/s")
    print(f"dv_z: {dv_hat[2]: .16e} km/s")

    print()
    print("Estimated initial state:")
    print(f"r0_x: {r0_hat[0]: .16e} km")
    print(f"r0_y: {r0_hat[1]: .16e} km")
    print(f"r0_z: {r0_hat[2]: .16e} km")
    print(f"v0_x: {v0_hat[0]: .16e} km/s")
    print(f"v0_y: {v0_hat[1]: .16e} km/s")
    print(f"v0_z: {v0_hat[2]: .16e} km/s")

    print()
    print("1-sigma uncertainties:")
    print(f"sigma Cd:   {state_sigma[0]: .16e}")
    print(f"sigma r0_x: {state_sigma[1]: .16e} km")
    print(f"sigma r0_y: {state_sigma[2]: .16e} km")
    print(f"sigma r0_z: {state_sigma[3]: .16e} km")
    print(f"sigma v0_x: {state_sigma[4]: .16e} km/s")
    print(f"sigma v0_y: {state_sigma[5]: .16e} km/s")
    print(f"sigma v0_z: {state_sigma[6]: .16e} km/s")

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

    plt.figure()
    plt.plot([float(x[1, 0] - r0_nominal[0]) for x in results.x_history], label="dr_x")
    plt.plot([float(x[2, 0] - r0_nominal[1]) for x in results.x_history], label="dr_y")
    plt.plot([float(x[3, 0] - r0_nominal[2]) for x in results.x_history], label="dr_z")
    plt.xlabel("Iteration")
    plt.ylabel("Initial Position Correction (km)")
    plt.title("Initial Position Correction Convergence")
    plt.legend()
    plt.grid(True)

    plt.figure()
    plt.plot([float(x[4, 0] - v0_nominal[0]) for x in results.x_history], label="dv_x")
    plt.plot([float(x[5, 0] - v0_nominal[1]) for x in results.x_history], label="dv_y")
    plt.plot([float(x[6, 0] - v0_nominal[2]) for x in results.x_history], label="dv_z")
    plt.xlabel("Iteration")
    plt.ylabel("Initial Velocity Correction (km/s)")
    plt.title("Initial Velocity Correction Convergence")
    plt.legend()
    plt.grid(True)

    plt.show()

if __name__ == "__main__":
    main()