"""
Script to retrieve the *closest* available satellite orbit data (OMM)
from Space-Track.org *created* *before* a specified target date.
Note that the data has some latency from the time of measurement of
the satellite's state.

The SpaceTrackAPI instance is initialized with credentials from a JSON file
in the following format:
{
    "username": "your_username",
    "password": "your_password"
}
"""

import json
import os

from orbitpy.orbits import SpaceTrackAPI

# Specify the NORAD ID of the satellite for which to retrieve data
norad_id = "31698"  # Example: TerraSAR-X

# Specify the target date and time to find the closest OMM
# (format: YYYY-MM-DDTHH:MM:SS)
target_date_time = "2024-04-09T01:00:00"


if __name__ == "__main__":
    # Initialize the SpaceTrackAPI with credentials
    file_dir = os.path.dirname(__file__)
    api = SpaceTrackAPI(os.path.join(file_dir, "credentials.json"))

    # Log in to Space-Track.org
    api.login()

    # Retrieve the *closest* available OMM data *created* before the
    # specified target datetime for the given satellite.
    omm_data = api.get_closest_omm(
        norad_id=norad_id, target_date_time=target_date_time
    )

    # Print the retrieved OMM data in a formatted JSON structure
    if omm_data:
        print("OMM Data:")
        print(json.dumps(omm_data, indent=4))
    else:
        print(
            f"No OMM data found for NORAD ID {norad_id}"
            f"at {target_date_time}."
        )

    # Log out from Space-Track.org
    api.logout()
