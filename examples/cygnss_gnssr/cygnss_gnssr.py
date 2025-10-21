""" The below script calculates specular coverage of CYGNSS satellite(s) over the CONUS.

NORAD IDs of CYGNSS
====================
The NORAD IDs of the 8 CYGNSS satellites are:
- [CYGFM01: '41887', CYGFM02: '41886', CYGFM03: '41891', CYGFM04: '41885', CYGFM05: '41884', CYGFM06: '41889', CYGFM07: '41890', CYGFM08: '41888']

CYGFM06 becomes defunct post late-2022? The NORAD IDS of the 7 active CYGNSS satellites are:
- [CYGFM01: '41887', CYGFM02: '41886', CYGFM03: '41891', CYGFM04: '41885', CYGFM05: '41884', ----: '----', CYGFM07: '41890', CYGFM08: '41888']

NORAD IDs of GNSS Satellites
============================
NORAD IDs of GNSS satellites sourced from: https://celestrak.org/NORAD/elements/
Note: This list may exclude GNSS satellites that were active during the mission epoch but are currently inactive at the time of writing this comment (4 Oct 2024).
Also satellites not launched before the date of interest, and decayed satellites are excluded later in the script.
This list will not include GNSS satellites active after 4 Oct 2024.

gps_sat_norad_ids = ['55268', '48859', '46826', '45854', '44506', '43873', '41328', '41019', '40730', '40534', '40294', '40105', '39741', '39533', '39166', '38833', '36585', '35752', '32711', '32384', '32260', '29601', '29486', '28874', '28474', '28190', '27704', '27663', '26407', '26360', '24876']
galelio_sat_norad_ids = ['37846', '37847', '38857', '40128', '40129', '40544', '40545', '40889', '40890', '41174', '41175', '41549', '41550', '41859', '41860', '41861', '41862', '43055', '43056', '43057', '43058', '43564', '43565', '43566', '43567', '49809', '49810', '59598', '59600']


"""
import shutil
from typing import Any
import sys
import os
import time
import json
import pandas as pd

from eosimutils.base import JsonSerializer

from orbitpy.mission import Mission

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dshield_format_converter import write_dshield_format_of_propagator_results, write_dshield_format_of_contact_results, write_dshield_format_of_eclipse_results, write_dshield_format_of_gnssr_coverage_results

exec_start_time = time.process_time()

user_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(user_dir, "results")

# Create a fresh results folder.
if os.path.exists(results_dir) and os.path.isdir(results_dir):
    shutil.rmtree(results_dir)
os.makedirs(results_dir, exist_ok=True)

# Load mission specifications from JSON file
mission_specs_path = os.path.join(user_dir, 'MissionSpecs.json')
with open(mission_specs_path, 'r') as mission_specs:
    mission_dict: dict[str, Any] = json.load(mission_specs)

mission_dict.setdefault("settings", {})["user_dir"] = user_dir  # Ensure settings and set user directory.

# Create Mission object from dictionary
mission = Mission.from_dict(mission_dict)

# Execute the mission
print("Start mission.")
results = mission.execute_all()

propagator_results = results.get("propagator_results", [])
contact_finder_results = results.get("contact_finder_results", {})
eclipse_finder_results = results.get("eclipse_finder_results", {})
gnssr_coverage_calculator_results = results.get("coverage_calculator_results", {})

elapsed_time = time.process_time() - exec_start_time
print(f"Mission complete. Time taken to execute in seconds: {elapsed_time:.2f}")

##### Save results to the format expected in the D-SHIELD project #####
print("Writing results to output directory.")
exec_start_time = time.process_time()

epoch = mission.start_time.to_dict(time_format="GREGORIAN_DATE", time_scale="UTC")
step_size = mission.propagator.step_size

results_fp = os.path.join(results_dir, 'coverage_calculator_results.json')
JsonSerializer.save_to_json(gnssr_coverage_calculator_results, results_fp)


# Write propagation results using the in-memory propagator_results.
propagator_results_serializable = JsonSerializer.to_serializable(propagator_results)
write_dshield_format_of_propagator_results(propagator_results_serializable, results_dir)

contact_finder_results_serializable = JsonSerializer.to_serializable(contact_finder_results)
write_dshield_format_of_contact_results(contact_finder_results_serializable, results_dir, epoch, step_size_seconds=step_size)

eclipse_finder_results_serializable = JsonSerializer.to_serializable(eclipse_finder_results)
write_dshield_format_of_eclipse_results(eclipse_finder_results_serializable, results_dir, epoch, step_size_seconds=step_size)

gnssr_coverage_calculator_results_serializable = JsonSerializer.to_serializable(gnssr_coverage_calculator_results)
write_dshield_format_of_gnssr_coverage_results(gnssr_coverage_calculator_results_serializable, results_dir, epoch, step_size_seconds=step_size) 

elapsed_time = time.process_time() - exec_start_time
print(f"Results written to MissionOutput.json. Time taken: {elapsed_time:.2f} seconds")


