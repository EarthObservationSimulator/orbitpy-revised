"""
Example script demonstrating specular point and RCG factor computation for a single transmitter/receiver pair.

Computes specular point using Ellipsoidal earth model, and plots the RCG factor over time.
"""

import json
import numpy as np
import os
import matplotlib.pyplot as plt

from eosimutils.time import AbsoluteDateArray, AbsoluteDate
from eosimutils.trajectory import StateSeries, PositionSeries
from eosimutils.base import ReferenceFrame, SurfaceType

from orbitpy.specular import get_specular_trajectory
from orbitpy.orbits import OrbitalMeanElementsMessage
from orbitpy.propagator import SGP4Propagator

def get_stateseries(ids, starting_date, duration_days=1):
    """Load OMM from data directory by ID and create stateseries using SGP4 propagator

    Args:
        ids (List[str]): A list of satellite IDs to load.
        starting_date (AbsoluteDate): The start date.
        duration_days (int): The duration in days to propagate.

    Returns:
        List[StateSeries]: A list of state series objects for the specified satellites.
    """

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.normpath(os.path.join(script_dir, "specular_coverage", "data"))

    outputs = []
    for norad_id in ids:
        with open(
            os.path.join(input_dir, f"omm_{norad_id}.json"), encoding="utf-8"
        ) as f:
            omm_data = json.load(f)

        orbit_obj = OrbitalMeanElementsMessage.from_json(omm_data)

        prop = SGP4Propagator(step_size=60.0)

        output = prop.execute(starting_date, duration_days, orbit_obj)

        outputs.append(output)

    return outputs

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

# Specify the NORAD ID of the satellite for which to retrieve data
gps_id = "24876"

# Test with a single cygnss satellite
cygnss_id = "41884"

# Specify the target date and time to find the closest OMM
# (format: YYYY-MM-DDTHH:MM:SS)
target_date_time = "2024-04-09T01:00:00"
start_time = "2024-04-15T01:00:00"
start_date = AbsoluteDate.from_dict(
    {
        "time_format": "Gregorian_Date",
        "time_scale": "UTC",
        "calendar_date": start_time,
    }
)

# Get stateseries for GPS and CYGNSS satellites
tx_stateseries = get_stateseries([gps_id], start_date)[0]
sc_stateseries = get_stateseries([cygnss_id], start_date)[0]

# ------------------------------
# Ellipsoidal Earth Computation
# ------------------------------

# Compute specular points using ellipsoidal Earth model
sp_posseries, rcg_factor_ellipsoidal = get_specular_trajectory(
    transmitter_states_itrf=tx_stateseries,
    receiver_states_itrf=sc_stateseries,
    times=tx_stateseries.time,
    surface=SurfaceType.WGS84,
)

# Plot ellipsoidal RCG factor
plt.figure()
plt.plot(tx_stateseries.time.ephemeris_time, rcg_factor_ellipsoidal)
plt.xlabel("Ephemeris Time (s)")
plt.ylabel("Range-Corrected Gain Factor")
plt.title("Range-Corrected Gain Factor, Ellipsoidal Earth")
# ---------------
# Display plots
# ---------------

plt.show()