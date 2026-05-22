import numpy as np
import os
import matplotlib.pyplot as plt

from eosimutils.time import AbsoluteDateArray
from eosimutils.trajectory import StateSeries, PositionSeries
from eosimutils.base import ReferenceFrame, SurfaceType
from eosimutils.state import CartesianState
from eosimutils.framegraph import FrameGraph


from orbitpy.specular import get_specular_trajectory, get_topk_trajectories

import orbitpy.orekitpropagator  # triggers decorator registration
from orbitpy.propagator import PropagatorFactory
from orbitpy.orbits import TwoLineElementSet, SpaceTrackAPI, OrbitalMeanElementsMessage

def get_first_idx(arr):
    """Get index of first non-naan row in numpy array"""

    mask = ~np.isnan(arr).all(axis=1)
    first_idx = np.argmax(mask)
    return first_idx

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

id = "44"
norad_id = 26407
gps_file = folder + "prn_" + id + "_trajectory.csv"
gps_pos = np.genfromtxt(
    gps_file,
    delimiter=",",
    skip_header=1,
    usecols=(1, 2, 3),
    dtype=float,
    filling_values=np.nan
)*M_TO_KM
gps_vel = np.genfromtxt(
    gps_file,
    delimiter=",",
    skip_header=1,
    usecols=(4, 5, 6),
    dtype=float,
    filling_values=np.nan
)*M_TO_KM

# -----------------------------------
# Process input into orbitpy objects
# -----------------------------------

# Create AbsoluteDateArray from ddm timestamps
datearray = convert_datetime(ddm_timestamp_utc)

# Create Stateseries from CYGNSS positions/velocities
sc_stateseries = StateSeries(
    time=datearray, data=[gps_pos, gps_vel], frame=ReferenceFrame.get("ITRF")
)


# Get first available state for receiver (should always start at zero)
first_idx = get_first_idx(sc_stateseries.data[0])
first_pos = sc_stateseries.data[0][first_idx]
first_vel = sc_stateseries.data[1][first_idx]
first_state = np.hstack((first_pos, first_vel))
start_time = datearray[first_idx]
stop_time = datearray[-1]
duration_days = (stop_time.to_spice_ephemeris_time() - start_time.to_spice_ephemeris_time())/86400.0
first_state = CartesianState.from_array(first_state, frame=ReferenceFrame.get("ITRF"), time=datearray[first_idx])

registry   = FrameGraph()
to_frame = ReferenceFrame.get("ICRF_EC")
first_state_icrf = registry.transform(first_state, to_frame, start_time)
sc_stateseries_icrf = registry.transform_series(sc_stateseries, to_frame)

specs = {
    "propagator_type": "OREKIT_PROPAGATOR",
    "stepSize": 10,
}

duration_days = duration_days # Propagate for 1 day
prop = PropagatorFactory.from_dict(specs)
prop.set_drag_coeff(0.0)

sat_state_series = prop.execute(
    t0=start_time, duration_days=duration_days, initial_state=first_state_icrf
)

diff = sat_state_series - sc_stateseries_icrf

plt.figure()
plt.plot(sat_state_series.time.ephemeris_time*(1/86400.0), diff.data[0][:, 0], label="Propagated X")
plt.plot(sat_state_series.time.ephemeris_time*(1/86400.0), diff.data[0][:, 1], label="Propagated Y")
plt.plot(sat_state_series.time.ephemeris_time*(1/86400.0), diff.data[0][:, 2], label="Propagated Z")
plt.xlabel("Ephemeris Time (s)")
plt.ylabel("Position (km)")
plt.title("Propagated Satellite Position Over Time")
plt.legend()

# Plot error norm
plt.figure()
error_norm = np.linalg.norm(diff.data[0], axis=1)
plt.plot(sat_state_series.time.ephemeris_time*(1/86400.0), error_norm)
plt.xlabel("Ephemeris Time (s)")
plt.ylabel("Position Error Norm (km)")
plt.title("Norm of Position Error Between Propagated State and Reference State")

## Try again using SGP4 instead.
file_dir = os.path.dirname(__file__)
api = SpaceTrackAPI(os.path.join(file_dir, "../spacetrack/credentials.json"))
# api = SpaceTrackAPI("path/to/credentials.json")
api.login()

# Convert your AbsoluteDate to ISO 8601 string format
# If your AbsoluteDate object is 'my_date':
date_str = start_time.to_dict(time_format="Gregorian_Date")["calendar_date"]
omm_dict = api.get_closest_omm(
    norad_id=49260,  # NORAD catalog ID of your satellite
    target_date_time=date_str,
    within_days=1  # Optional: max days before target to search
)

# Extract the TLE from the OMM
if omm_dict:
    omm = OrbitalMeanElementsMessage.from_dict(omm_dict)

# --------------------------
# Plot orbital energy vs time
# --------------------------

MU_EARTH = 398600.4418  # km^3 / s^2

r = sc_stateseries_icrf.data[0]  # km
v = sc_stateseries_icrf.data[1]  # km/s
r_norm = np.linalg.norm(r, axis=1)
v_norm = np.linalg.norm(v, axis=1)
specific_energy = 0.5 * v_norm**2 - MU_EARTH / r_norm  # km^2/s^2
t = sc_stateseries_icrf.time.ephemeris_time
t_hours = (t - t[0]) / 3600.0

plt.figure()
plt.plot(t_hours, specific_energy)
plt.xlabel("Time since start (hours)")
plt.ylabel("Specific Orbital Energy (km$^2$/s$^2$)")
plt.title("Satellite Specific Orbital Energy Over Time")
plt.grid(True)
plt.legend()

r = sat_state_series.data[0]  # km
v = sat_state_series.data[1]  # km/s
r_norm = np.linalg.norm(r, axis=1)
v_norm = np.linalg.norm(v, axis=1)
specific_energy = 0.5 * v_norm**2 - MU_EARTH / r_norm  # km^2/s^2
t = sat_state_series.time.ephemeris_time
t_hours = (t - t[0]) / 3600.0

plt.figure()
plt.plot(t_hours, specific_energy)
plt.xlabel("Time since start (hours)")
plt.ylabel("Specific Orbital Energy (km$^2$/s$^2$)")
plt.title("Satellite Specific Orbital Energy Over Time")
plt.grid(True)
plt.legend()
plt.show()