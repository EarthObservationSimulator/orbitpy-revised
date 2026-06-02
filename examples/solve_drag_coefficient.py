"""Estimate CYGNSS drag coefficient with GLSDC nonlinear least squares."""

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

import orbitpy.orekitpropagator  # triggers decorator registration
from orbitpy.propagator import PropagatorFactory
from orbitpy.orbits import SpaceTrackAPI, OrbitalMeanElementsMessage


# =============================================================================
# GLSDC IMPLEMENTATION
# =============================================================================

VectorFunction = Callable[[np.ndarray], np.ndarray]


@dataclass
class GLSDCResults:
    x_hat: np.ndarray
    y_hat: np.ndarray
    residual: np.ndarray
    jacobian: np.ndarray
    normal_matrix: np.ndarray
    covariance: np.ndarray
    dx: np.ndarray
    cost: float
    iterations: int
    converged: bool
    x_history: list[np.ndarray] = field(default_factory=list)
    cost_history: list[float] = field(default_factory=list)


class FiniteDifferenceJacobian:
    def __init__(self, function: VectorFunction, eps: float = 1.0e-6):
        self.function = function
        self.eps = float(eps)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = _as_column(x, "x")
        y0 = _as_column(self.function(x), "F(x)")

        ny = y0.shape[0]
        nx = x.shape[0]
        jac = np.empty((ny, nx), dtype=float)

        for j in range(nx):
            h = self.eps * max(1.0, abs(float(x[j, 0])))

            xp = x.copy()
            xm = x.copy()
            xp[j, 0] += h
            xm[j, 0] -= h

            yp = _as_column(self.function(xp), "F(x + h)")
            ym = _as_column(self.function(xm), "F(x - h)")

            jac[:, j] = ((yp - ym) / (2.0 * h)).ravel()

        return jac


class GLSDC:
    def __init__(self, max_iters: int = 50, tol: float = 1.0e-10, fd_eps: float = 1.0e-6):
        self.max_iters = int(max_iters)
        self.tol = float(tol)
        self.fd_eps = float(fd_eps)

    def solve(
        self,
        F: VectorFunction,
        x0: np.ndarray,
        y: np.ndarray,
        R: Optional[np.ndarray] = None,
        jacobian: Optional[VectorFunction] = None,
        verbose: bool = True,
    ) -> GLSDCResults:
        x = _as_column(x0, "x0")
        y = _as_column(y, "y")

        nx = x.shape[0]
        ny = y.shape[0]

        Rinv = _inverse_covariance(R, ny)
        df = jacobian if jacobian is not None else FiniteDifferenceJacobian(F, eps=self.fd_eps)

        x_history = [x.copy()]
        cost_history = []
        dx = np.zeros_like(x)
        converged = False

        for k in range(self.max_iters):
            y_model = _as_column(F(x), "F(x)")
            H = np.asarray(df(x), dtype=float)
            _validate_dimensions(y_model, H, nx, ny)

            residual = y - y_model
            cost = float((residual.T @ Rinv @ residual)[0, 0])

            normal_matrix = H.T @ Rinv @ H
            rhs = H.T @ Rinv @ residual

            dx = _solve_normal_equations(normal_matrix, rhs)
            x = x + dx

            x_history.append(x.copy())
            cost_history.append(cost)

            dx_norm = float(np.linalg.norm(dx))

            if verbose:
                print("=" * 72)
                print(f"GLSDC iteration {k + 1}")
                print(f"x         = {x.ravel()}")
                print(f"dx        = {dx.ravel()}")
                print(f"|dx|      = {dx_norm:.16e}")
                print(f"cost      = {cost:.16e}")
                print(f"rms resid = {np.sqrt(np.mean(residual ** 2)):.16e}")

            if dx_norm < self.tol:
                converged = True
                break

        y_hat = _as_column(F(x), "F(x_hat)")
        H = np.asarray(df(x), dtype=float)
        _validate_dimensions(y_hat, H, nx, ny)

        residual = y - y_hat
        normal_matrix = H.T @ Rinv @ H
        covariance = np.linalg.pinv(normal_matrix)
        final_cost = float((residual.T @ Rinv @ residual)[0, 0])

        return GLSDCResults(
            x_hat=x,
            y_hat=y_hat,
            residual=residual,
            jacobian=H,
            normal_matrix=normal_matrix,
            covariance=covariance,
            dx=dx,
            cost=final_cost,
            iterations=len(cost_history),
            converged=converged,
            x_history=x_history,
            cost_history=cost_history,
        )


def _as_column(value: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)

    if arr.ndim == 0:
        return arr.reshape(1, 1)

    if arr.ndim == 1:
        return arr.reshape(-1, 1)

    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr

    raise ValueError(f"{name} must be convertible to a column vector")


def _inverse_covariance(R: Optional[np.ndarray], ny: int) -> np.ndarray:
    if R is None:
        return np.eye(ny)

    R = np.asarray(R, dtype=float)

    try:
        chol = np.linalg.cholesky(R)
        return np.linalg.solve(chol.T, np.linalg.solve(chol, np.eye(ny)))
    except np.linalg.LinAlgError:
        return np.linalg.pinv(R)


def _validate_dimensions(y_model: np.ndarray, H: np.ndarray, nx: int, ny: int) -> None:
    if y_model.shape != (ny, 1):
        raise ValueError(f"F(x) must have shape ({ny}, 1), got {y_model.shape}")
    if H.shape != (ny, nx):
        raise ValueError(f"Jacobian must have shape ({ny}, {nx}), got {H.shape}")


def _solve_normal_equations(normal_matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(normal_matrix, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(normal_matrix) @ rhs


# =============================================================================
# ORBIT / DATA UTILITIES
# =============================================================================

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


# =============================================================================
# MAIN SCRIPT
# =============================================================================

def main():
    KM_TO_M = 1000.0
    M_TO_KM = 1.0 / 1000.0

    norad_id = 41891

    file_dir = os.path.dirname(os.path.abspath(__file__))

    cyg_file = os.path.join(
        file_dir,
        "cygnss_data",
        "cygnss_spacecraft_trajectory.csv",
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

    # Transform measured/reference data to inertial frame
    registry = FrameGraph()
    to_frame = ReferenceFrame.get("ICRF_EC")
    sc_stateseries_icrf_full = registry.transform_series(sc_stateseries, to_frame)

    start_time = original_datearray[0]
    stop_time = original_datearray[-1]

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

    date_str_stop = stop_time.to_dict(time_format="Gregorian_Date")["calendar_date"]

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

    omm = OrbitalMeanElementsMessage.from_dict(omm_dict)

    # -------------------------------------------------------------------------
    # Define measurement epochs
    # -------------------------------------------------------------------------

    datearray = AbsoluteDateArray.linspace(tle_epoch, stop_time, 300)

    # Resample reference trajectory onto propagation epochs
    sc_stateseries_icrf = sc_stateseries_icrf_full.resample(datearray)

    # Reference positions: shape (M, 3)
    y_reference_positions = sc_stateseries_icrf.data[0]

    # GLSDC requires column vector: shape (3M, 1)
    y = y_reference_positions.reshape(-1, 1)

    # -------------------------------------------------------------------------
    # Define measurement model F(x)
    # -------------------------------------------------------------------------

    def F(x: np.ndarray) -> np.ndarray:
        drag_coeff = float(np.asarray(x).reshape(-1)[0])

        prop.set_drag_coeff(drag_coeff)

        propagated_state_series = prop.execute_2(
            times=datearray,
            initial_state=omm,
        )

        propagated_positions = propagated_state_series.data[0]

        return propagated_positions.reshape(-1, 1)

    # -------------------------------------------------------------------------
    # Solve for drag coefficient
    # -------------------------------------------------------------------------

    x0 = np.array([[300.2]])

    solver = GLSDC(
        max_iters=3,
        tol=1.0e-8,
        fd_eps=1.0e-2,
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
    print(f"1-sigma formal uncertainty: {cd_sigma:.16e}")
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