"""
Example script demonstrating the use of SpecularCoverage class to calculate GNSS coverage.
"""

import json
import os
import numpy as np
from time import perf_counter as timer

from orbitpy.orbits import OrbitalMeanElementsMessage
from orbitpy.propagator import SGP4Propagator
from orbitpy.coveragecalculator import CoverageFactory, CoverageType

from eosimutils.time import AbsoluteDate
from eosimutils.base import ReferenceFrame, SPHERICAL_EARTH_MEAN_RADIUS
from eosimutils.standardframes import get_lvlh
from eosimutils.framegraph import FrameGraph
from eosimutils.fieldofview import CircularFieldOfView
from eosimutils.state import Cartesian3DPositionArray


def random_points_on_sphere(n, r=SPHERICAL_EARTH_MEAN_RADIUS):
    """
    Generate n random points uniformly on the surface of a sphere with radius R.

    Args:
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
    input_dir = os.path.normpath(os.path.join(script_dir, "data"))

    outputs = []
    for norad_id in ids:
        with open(os.path.join(input_dir, f"omm_{norad_id}.json"), encoding="utf-8") as f:
            omm_data = json.load(f)

        orbit_obj = OrbitalMeanElementsMessage.from_json(omm_data)

        prop = SGP4Propagator(step_size=60.0)

        output = prop.execute(starting_date, duration_days, orbit_obj)

        outputs.append(output)

    return outputs


# Specify the NORAD ID of the satellite for which to retrieve data
gps_ids = [
    "24876",
    "26360",
    "26605",
    "27663",
    "27704",
    "28190",
    "28474",
    "28874",
    "29486",
    "29601",
    "32260",
    "32384",
    "32711",
    "35752",
    "36287",
    "36585",
    "36828",
    "37210",
    "37256",
    "37384",
    "37753",
    "37763",
    "37846",
    "37847",
    "37948",
    "38091",
    "38250",
    "38251",
    "38775",
    "38833",
    "38857",
    "38858",
    "38953",
    "39166",
    "39533",
    "39741",
    "40105",
    "40128",
    "40129",
    "40294",
    "40534",
    "40544",
    "40545",
    "40549",
    "40730",
    "40748",
    "40749",
    "40889",
    "40890",
    "40938",
    "41019",
    "41174",
    "41175",
    "41328",
    "41434",
    "41549",
    "41550",
    "41586",
    "41859",
    "41860",
    "41861",
    "41862",
    "43001",
    "43002",
    "43055",
    "43056",
    "43057",
    "43058",
    "43107",
    "43108",
    "43207",
    "43208",
    "43245",
    "43246",
    "43539",
    "43564",
    "43565",
    "43566",
    "43567",
    "43581",
    "43582",
    "43602",
    "43603",
    "43622",
    "43623",
    "43647",
    "43648",
    "43683",
    "43706",
    "43707",
    "43873",
    "44204",
    "44231",
    "44337",
    "44506",
    "44542",
    "44543",
    "44709",
    "44793",
    "44794",
    "44864",
    "44865",
    "45344",
    "45807",
    "45854",
    "46826",
    "48859",
    "49809",
    "49810",
]

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
gps_stateseries = get_stateseries(gps_ids, start_date)
cygnss_stateseries = get_stateseries([cygnss_id], start_date)[0]

# Fill frame graph with transforms for GPS satellites
registry = FrameGraph()
gps_frames = []
for i in range(len(gps_stateseries)):
    # Create frame graph and add LVLH frame
    lvlh_frame = ReferenceFrame.add(f"LVLH_GPS_{gps_ids[i]}")
    att_lvlh, pos_lvlh = get_lvlh(gps_stateseries[i], lvlh_frame)
    registry.add_orientation_transform(att_lvlh)
    from_frame = ReferenceFrame.get("ICRF_EC")
    to_frame = lvlh_frame
    registry.add_pos_transform(from_frame, to_frame, pos_lvlh)
    gps_frames.append(lvlh_frame)

# Fill frame graph with transforms for CYGNSS satellite
lvlh_cygnss = ReferenceFrame.add("LVLH_CYGNSS")
att_lvlh_cygnss, pos_lvlh_cygnss = get_lvlh(cygnss_stateseries, lvlh_cygnss)
registry.add_orientation_transform(att_lvlh_cygnss)
from_frame = ReferenceFrame.get("ICRF_EC")
to_frame = lvlh_cygnss
registry.add_pos_transform(from_frame, to_frame, pos_lvlh_cygnss)

# Create a circular field of view
diameter = 70.0  # deg
fov = CircularFieldOfView(diameter=diameter, frame=lvlh_cygnss)

# Generate 50000 random points on the sphere
points = random_points_on_sphere(50000)
points_array = Cartesian3DPositionArray(points, ReferenceFrame.get("ITRF"))

# Create a coverage calculator
cov = CoverageFactory.from_dict(
    {"coverage_type": CoverageType.SPECULAR_COVERAGE.to_string()}
)

# Time points for coverage calculation
times = gps_stateseries[0].time

# Radius of glistening zone in km
specular_radius = 10

print("Calculating coverage for target points using circular sensor...")
start_time = timer()
coverage = cov.calculate_coverage(
    points_array, fov, registry, times, gps_frames, specular_radius
)

# Sum up total coverage time
total_time = 0
for cov in coverage:
    total_time += cov[0].coverage_time()

print(f"Total coverage time summed accross target grid points is {total_time}")

end_time = timer()
print(f"Coverage calculation took {end_time - start_time:.2f} seconds.")
