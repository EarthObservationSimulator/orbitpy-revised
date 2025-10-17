""" Script to run a mission from a JSON configuration file.
See the `orbitpy.mission` module for details on the mission configuration schema.

The script expects a user directory as input, which should contain a `MissionSpecs.json`
file with the mission configuration. The script will execute the mission and write the results
to `MissionOutput.json` in the same directory.

Example usage:
    python bin/run_mission.py <path_to_user_directory>
"""
import os
import json
import argparse
import time
from typing import Any

from eosimutils.base import JsonSerializer

from orbitpy.mission import Mission

def main(user_dir: str) -> None:
    """
    Executes a mission according to an input JSON configuration file.

    Args:
        user_dir (str): Path to the user directory where it expects a `MissionSpecs.json`
                        configuration file and auxiliary files. Output files are written
                        in the same directory by default.

    Example:
        python bin/run_mission.py examples/mission_1/
    """
    start_time = time.process_time()

    mission_specs_path = os.path.join(user_dir, 'MissionSpecs.json')
    if not os.path.isfile(mission_specs_path):
        raise FileNotFoundError(f"MissionSpecs.json not found in {user_dir}")

    with open(mission_specs_path, 'r') as mission_specs:
        mission_dict: dict[str, Any] = json.load(mission_specs)

    mission_dict.setdefault("settings", {})["user_dir"] = user_dir  # Ensure settings and set user directory.

    mission = Mission.from_dict(mission_dict)

    print("Start mission.")
    results = mission.execute_all()

    elapsed_time = time.process_time() - start_time
    print(f"Mission complete. Time taken to execute in seconds: {elapsed_time:.2f}")

    print("Writing results to output directory.")
    start_time = time.process_time()
    results_fp = os.path.join(user_dir, 'MissionOutput.json')
    JsonSerializer.save_to_json(results, results_fp)
    elapsed_time = time.process_time() - start_time
    print(f"Results written to MissionOutput.json. Time taken: {elapsed_time:.2f} seconds")

class ReadableDir(argparse.Action):
    """Custom argparse Action to validate a readable directory."""

    def __call__(self, parser, namespace, values, option_string=None) -> None:
        prospective_dir = values
        if not os.path.isdir(prospective_dir):
            raise argparse.ArgumentTypeError(f"{prospective_dir} is not a valid path.")
        if not os.access(prospective_dir, os.R_OK):
            raise argparse.ArgumentTypeError(f"{prospective_dir} is not a readable directory.")
        setattr(namespace, self.dest, prospective_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Mission")
    parser.add_argument(
        'user_dir',
        action=ReadableDir,
        help="Directory with user config JSON file, and also to write the results."
    )
    args = parser.parse_args()

    try:
        main(args.user_dir)
    except Exception as e:
        print(f"Error: {e}")