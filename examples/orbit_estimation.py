import numpy as np
import os
import matplotlib.pyplot as plt

from eosimutils.time import AbsoluteDate, AbsoluteDateArray
from eosimutils.trajectory import StateSeries, PositionSeries
from eosimutils.base import ReferenceFrame, SurfaceType
from eosimutils.state import CartesianState
from eosimutils.framegraph import FrameGraph


from orbitpy.specular import get_specular_trajectory, get_topk_trajectories

import orbitpy.orekitpropagator  # triggers decorator registration
from orbitpy.propagator import PropagatorFactory
from orbitpy.orbits import TwoLineElementSet, SpaceTrackAPI, OrbitalMeanElementsMessage

def convert_datetime(np_dt):
    """Convert numpy datetime object to AbsoluteDateArray

    Args:
        np_dt (np.ndarray): Numpy datetime64 array.

    Returns:
        AbsoluteDateArray: Orbitpy AbsoluteDateArray object with the times from the datetime object
    """

    arr = np.atleast_1d(np_dt)

    if not np.issubdtype(arr.dtype, np.datetime64):
        raise TypeError("np_dt must be a numpy.datetime64 scalar or array")

    # Convert to millisecond precision ISO strings without timezone
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

# -------------
# READ IN DATA
# -------------

KM_TO_M = 1000.0
M_TO_KM = 1.0 / 1000.0

sat_number = "03"
ddm_num = 1
folder = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "cygnss_data", ""
)
ext = ".npy"

cyg_file = folder + "cygnss_spacecraft_trajectory.csv"
cyg_pos = data = np.loadtxt(
    cyg_file,
    delimiter=",",
    skiprows=1,
    usecols=(1, 2, 3)
)*M_TO_KM
cyg_vel = data = np.loadtxt(
    cyg_file,
    delimiter=",",
    skiprows=1,
    usecols=(4, 5, 6)
)*M_TO_KM

ddm_timestamp_utc = np.loadtxt(
    cyg_file,
    delimiter=",",
    skiprows=1,
    usecols=(0,),
    dtype="datetime64[ns]"
)

norad_id = 41891

# -----------------------------------
# Process input into orbitpy objects
# -----------------------------------

# Create AbsoluteDateArray from ddm timestamps
datearray = convert_datetime(ddm_timestamp_utc)

sc_stateseries = StateSeries(
    time=datearray, data=[cyg_pos, cyg_vel], frame=ReferenceFrame.get("ITRF")
)


# Transform reference data to inertial frame
registry   = FrameGraph()
to_frame = ReferenceFrame.get("ICRF_EC")
sc_stateseries_icrf = registry.transform_series(sc_stateseries, to_frame)

# Start and stop time of reference data
start_time = datearray[0]
stop_time = datearray[-1]

# Setup propagator
specs = {
    "propagator_type": "OREKIT_PROPAGATOR",
    "stepSize": 10,
}
prop = PropagatorFactory.from_dict(specs)
prop.set_drag_coeff(0.0)

# Use spacetrack to get initial tle
# Convert AbsoluteDate to ISO 8601 string format
date_str_start = start_time.to_dict(time_format="Gregorian_Date")["calendar_date"]
date_str_stop = stop_time.to_dict(time_format="Gregorian_Date")["calendar_date"]
file_dir = os.path.dirname(__file__)
api = SpaceTrackAPI(os.path.join(file_dir, "spacetrack/credentials.json"))
api.login()
omm_dict = api.get_closest_omm(
    norad_id=norad_id,  # NORAD catalog ID of your satellite
    target_date_time=date_str_stop,
    within_days=1  # Optional: max days before target to search
)

# Extract the TLE and time from the OMM
if omm_dict:
    tle_epoch = AbsoluteDate.from_dict({"time_scale" : omm_dict["TIME_SYSTEM"], "time_format" : "GREGORIAN_DATE", "calendar_date" : omm_dict["EPOCH"]})
    omm = OrbitalMeanElementsMessage.from_dict(omm_dict)

# Create array of dates starting at TLE epoch and continuing until end of reference trajectory,
# with a 300 second step size (5 min)
datearray = AbsoluteDateArray.linspace(tle_epoch, stop_time, 300)

omm = sc_stateseries_icrf.at(tle_epoch)
omm = np.hstack((omm[0][0], omm[1][0]))
omm = CartesianState.from_array(omm, frame=ReferenceFrame.get("ICRF_EC"), time=tle_epoch)
# Propagate starting from TLE to each of the time points in the date array
sat_state_series_sgp4 = prop.execute_2(times=datearray, initial_state=omm)

# Resample reference trajectory onto these time points.
sc_stateseries_icrf = sc_stateseries_icrf.resample(datearray)

# Take difference between propagated and reference states
diff_st = sat_state_series_sgp4 - sc_stateseries_icrf

# --------------------------
# Plot results
# --------------------------

# Plot individual error components
plt.figure()
plt.plot(sat_state_series_sgp4.time.ephemeris_time*(1/86400.0), diff_st.data[0][:, 0], label="Propagated X")
plt.plot(sat_state_series_sgp4.time.ephemeris_time*(1/86400.0), diff_st.data[0][:, 1], label="Propagated Y")
plt.plot(sat_state_series_sgp4.time.ephemeris_time*(1/86400.0), diff_st.data[0][:, 2], label="Propagated Z")
plt.xlabel("Ephemeris Time (days)")
plt.ylabel("Position (km)")
plt.title("Propagated Satellite Position Over Time")
plt.legend()

# Plot error norm
plt.figure()
error_norm = np.linalg.norm(diff_st.data[0], axis=1)
plt.plot(sat_state_series_sgp4.time.ephemeris_time*(1/86400.0), error_norm)
plt.xlabel("Ephemeris Time (days)")
plt.ylabel("Position Error Norm (km)")
plt.title("Norm of Position Error Between Propagated State and Reference State")
plt.show()