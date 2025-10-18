"""
Example script demonstrating specular point computation. Uses saved data for CYGNSS position, GPS
transmitter position, and specular point position computed from the CYGNSS dataset.

Computes specular point using spherical Earth model and Ellipsoidal earth model, and plots results
for each.
"""

import numpy as np
import os
import matplotlib.pyplot as plt

from eosimutils.time import AbsoluteDateArray
from eosimutils.trajectory import StateSeries, PositionSeries
from eosimutils.base import ReferenceFrame, SurfaceType

from orbitpy.specular import get_specular_trajectory


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

KM_TO_M = 1.0 / 1000.0

sat_number = "03"
ddm_num = 1
folder = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "processed_data", ""
)
ext = ".npy"

prefix = folder + "CYG" + sat_number
ddm_timestamp_utc = np.load(prefix + "ddm_timestamp_utc" + ext)
pvt_timestamp_utc = np.load(prefix + "pvt_timestamp_utc" + ext)
sc_pos_x = np.load(prefix + "sc_pos_x" + ext) * KM_TO_M
sc_pos_y = np.load(prefix + "sc_pos_y" + ext) * KM_TO_M
sc_pos_z = np.load(prefix + "sc_pos_z" + ext) * KM_TO_M
sc_vel_x = np.load(prefix + "sc_vel_x" + ext) * KM_TO_M
sc_vel_y = np.load(prefix + "sc_vel_y" + ext) * KM_TO_M
sc_vel_z = np.load(prefix + "sc_vel_z" + ext) * KM_TO_M

prefix = prefix + "DDM" + str(ddm_num)
sp_pos_x = np.load(prefix + "sp_pos_x" + ext) * KM_TO_M
sp_pos_y = np.load(prefix + "sp_pos_y" + ext) * KM_TO_M
sp_pos_z = np.load(prefix + "sp_pos_z" + ext) * KM_TO_M
tx_pos_x = np.load(prefix + "tx_pos_x" + ext) * KM_TO_M
tx_pos_y = np.load(prefix + "tx_pos_y" + ext) * KM_TO_M
tx_pos_z = np.load(prefix + "tx_pos_z" + ext) * KM_TO_M
tx_vel_x = np.load(prefix + "tx_vel_x" + ext) * KM_TO_M
tx_vel_y = np.load(prefix + "tx_vel_y" + ext) * KM_TO_M
tx_vel_z = np.load(prefix + "tx_vel_z" + ext) * KM_TO_M

# -----------------------------------
# Process input into orbitpy objects
# -----------------------------------

# Create AbsoluteDateArray from ddm timestamps
datearray = convert_datetime(ddm_timestamp_utc)

# Create Stateseries from CYGNSS positions/velocities
sc_pos = np.vstack((sc_pos_x, sc_pos_y, sc_pos_z)).T
sc_vel = np.vstack((sc_vel_x, sc_vel_y, sc_vel_z)).T
sc_stateseries = StateSeries(
    time=datearray, data=[sc_pos, sc_vel], frame=ReferenceFrame.get("ITRF")
)

# Create Stateseries from GPS positions/velocities
tx_pos = np.vstack((tx_pos_x, tx_pos_y, tx_pos_z)).T
tx_vel = np.vstack((tx_vel_x, tx_vel_y, tx_vel_z)).T
tx_stateseries = StateSeries(
    time=datearray, data=[tx_pos, tx_vel], frame=ReferenceFrame.get("ITRF")
)

# Create PositionSeries from specular point positions
sp_pos = np.vstack((sp_pos_x, sp_pos_y, sp_pos_z)).T
sp_posseries_ref = PositionSeries(
    time=datearray, data=sp_pos, frame=ReferenceFrame.get("ITRF")
)

# ------------------------------
# Ellipsoidal Earth Computation
# ------------------------------

# Compute specular points using ellipsoidal Earth model
sp_posseries = get_specular_trajectory(
    transmitter=tx_stateseries,
    receiver=sc_stateseries,
    times=datearray,
    surface=SurfaceType.WGS84,
)

# Get the difference between the reference and computed data
diff = sp_posseries_ref - sp_posseries
norms = np.linalg.norm(diff.data[0], axis=1)

# Plot the norm of the difference
plt.figure()
plt.plot(datearray.ephemeris_time, norms)
plt.title("Specular Location Error Norm, Ellipsoidal Earth")

# -----------------------------------
# SPHERICAL EARTH COMPUTATION
# -----------------------------------

# Compute specular points using spherical Earth model
sp_posseries_spherical = get_specular_trajectory(
    transmitter=tx_stateseries,
    receiver=sc_stateseries,
    times=datearray,
    surface=SurfaceType.SPHERE,
)

# Get the difference between the reference and computed data
diff = sp_posseries_ref - sp_posseries_spherical
norms = np.linalg.norm(diff.data[0], axis=1)

# Plot the norm of the difference

plt.figure()
plt.plot(datearray.ephemeris_time, norms)
plt.title("Specular Location Error Norm, Spherical Earth")

# ---------------
# Display plots
# ---------------

plt.show()
