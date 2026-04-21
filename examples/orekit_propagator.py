"""Example for the orekit propagator with multiple drag coefficients."""

from time import perf_counter as timer

import orbitpy.orekitpropagator  # triggers decorator registration
from orbitpy.propagator import PropagatorFactory
from orbitpy.orbits import TwoLineElementSet

from eosimutils.time import AbsoluteDate
from eosimutils.base import WGS84_EARTH_EQUATORIAL_RADIUS

import numpy as np
import matplotlib.pyplot as plt

specs = {
    "propagator_type": "OREKIT_PROPAGATOR",
    "stepSize": 10,
}

start_time = AbsoluteDate.from_dict(
    {
        "time_format": "Gregorian_Date",
        "calendar_date": "2025-04-17T12:00:00",
        "time_scale": "utc",
    }
)

tle = TwoLineElementSet(
    line0="0 LANDSAT 9",
    line1="1 49260U 21088A   25106.07240456  .00000957  00000-0  22241-3 0  9997",
    line2="2 49260  98.1921 177.4890 0001161  87.5064 272.6267 14.57121096188801",
)

duration_days = 0.25  # Propagate for 0.25 days

# Drag coefficients to run
drag_coeffs = [10000.0, 20000.0, 40000.0]
colors = ["blue", "red", "green"]

results = []

print("Starting propagations...")
for cd in drag_coeffs:
    prop = PropagatorFactory.from_dict(specs)
    prop.set_drag_coeff(cd)

    sat_state_series = prop.execute(
        t0=start_time, duration_days=duration_days, initial_state=tle
    )

    results.append((cd, sat_state_series))

print("Finished propagation...")

# Plot altitude over time
fig, ax = plt.subplots(figsize=(10, 6))

for (cd, series), color in zip(results, colors):
    altitude = (
        np.linalg.norm(series.data[0], axis=1) - WGS84_EARTH_EQUATORIAL_RADIUS
    )

    ax.plot(
        series.time.ephemeris_time,
        altitude,
        color=color,
        label=f"Cd = {cd}",
    )

plt.xlabel("Ephemeris Time (s)")
plt.ylabel("Altitude (km)")
plt.title("Orbit Altitude vs Time for Different Drag Coefficients")
plt.legend()
plt.grid(True)

plt.show()