"""Script to get OMM files from spacetrack for specular coverage example."""

import time
import json
import os

from orbitpy.orbits import SpaceTrackAPI, OrbitalMeanElementsMessage


def get_gps_omms(spacetrack_api, ids):
    """Retrieve OMM files for a list of GPS satellite IDs.

    Args:
        spacetrack_api (SpaceTrackAPI): The SpaceTrackAPI instance to use for data retrieval.
        ids (list): A list of GPS satellite NORAD IDs.

    Returns:
        list: A list of OrbitalMeanElementsMessage objects containing the OMM data.
    """

    outputs = []
    for norad_id in ids:
        # Sleep to avoid triggering rate limit
        time.sleep(2.5)
        # Retrieve the closest available OMM data created before the
        # specified target datetime for the given satellite.
        omm_data = spacetrack_api.get_closest_omm(
            norad_id=norad_id,
            target_date_time=target_date_time,
            within_days=100,
        )

        orbit_obj = OrbitalMeanElementsMessage.from_dict(omm_data)

        outputs.append(orbit_obj)

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

# CYGFM06 (41889) is no longer on orbit, removed from ids list.
cygnss_ids = ["41884", "41885", "41886", "41887", "41888", "41890", "41891"]

norad_ids = gps_ids + cygnss_ids

# Specify the target date and time to find the closest OMM
# (format: YYYY-MM-DDTHH:MM:SS)
target_date_time = "2025-09-01T01:00:00"

script_dir = os.path.dirname(os.path.abspath(__file__))
cred_path = os.path.normpath(
    os.path.join(script_dir, "..", "spacetrack", "credentials.json")
)

# Log in to Space-Track.org
st_api = SpaceTrackAPI(cred_path)
st_api.login()

# Create an output directory
output_dir = os.path.normpath(os.path.join(script_dir, "data"))
os.makedirs(output_dir, exist_ok=True)

omms = get_gps_omms(st_api, norad_ids)

# Write each json file to the data folder
for i in range(len(omms)):

    output_filename = os.path.join(output_dir, f"omm_{norad_ids[i]}.json")
    output_json = omms[i].to_json()

    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=4)
