"""Example for the coveragecalculator module."""

from time import perf_counter as timer

import orbitpy.coveragecalculator
from orbitpy.propagator import PropagatorFactory
from orbitpy.orbits import TwoLineElementSet

from eosimutils.time import AbsoluteDate
from eosimutils.base import ReferenceFrame, WGS84_EARTH_RADIUS
from eosimutils.framegraph import FrameGraph
from eosimutils.standardframes import get_lvlh
from eosimutils.fieldofview import CircularFieldOfView
from eosimutils.state import Cartesian3DPositionArray

import numpy as np


def random_points_on_sphere(n, r=WGS84_EARTH_RADIUS):
    """
    Generate n random points uniformly on the surface of a sphere with radius R.

    Parameters:
        n (int): Number of points.
        R (float): Radius of the sphere.

    Returns:
        np.ndarray: An (n x 3) array of 3D points on the sphere.
    """
    phi = np.random.uniform(0, 2 * np.pi, n)
    cos_theta = np.random.uniform(-1, 1, n)
    theta = np.arccos(cos_theta)

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * cos_theta

    return np.stack((x, y, z), axis=-1)


# Create a coverage calculator
cov = orbitpy.coveragecalculator.CoverageFactory.from_dict(
    {
        "coverage_type": orbitpy.coveragecalculator.CoverageType.POINT_COVERAGE.to_string()
    }
)

# Create a propagator
factory = PropagatorFactory()
specs = {"propagator_type": "SGP4_PROPAGATOR", "step_size": 1}
sgp4_prop = factory.get_propagator(specs)

tle = TwoLineElementSet(
    line0="0 LANDSAT 9",
    line1="1 49260U 21088A   25106.07240456  .00000957  00000-0  22241-3 0  9997",
    line2="2 49260  98.1921 177.4890 0001161  87.5064 272.6267 14.57121096188801",
)

start_time = AbsoluteDate.from_dict(
    {
        "time_format": "Gregorian_Date",
        "calendar_date": "2025-04-17T12:00:00",
        "time_scale": "utc",
    }
)

duration_days = 1  # Propagate for 1 days (24 hours)

print("Starting propagation...")
sat_state_series = sgp4_prop.execute(
    t0=start_time, duration_days=duration_days, orbit=tle
)
print("Finished propagation...")

times = sat_state_series.time

# Create frame graph and add LVLH frame
lvlh_frame = ReferenceFrame.add("LVLH")
att_lvlh, pos_lvlh = get_lvlh(sat_state_series, lvlh_frame)
registry = FrameGraph()
registry.add_orientation_transform(att_lvlh)
from_frame = ReferenceFrame.get("ICRF_EC")
to_frame = ReferenceFrame.get("LVLH")
registry.add_pos_transform(from_frame, to_frame, pos_lvlh)

# Create a circular field of view
diameter = 30.0  # deg
fov = CircularFieldOfView(diameter=diameter, frame=lvlh_frame)

# Calculate point coverage
target_points = random_points_on_sphere(
    100000
)  # Generate 100000 random points on the sphere
target_point_array = Cartesian3DPositionArray(
    target_points, ReferenceFrame.get("ITRF")
)

print("Calculating coverage for target points using circular sensor...")
start_time = timer()
circ_coverage = cov.calculate_coverage(
    target_point_array, fov=fov, frame_graph=registry, times=times
)
end_time = timer()
print(f"Coverage calculation took {end_time - start_time:.2f} seconds.")
