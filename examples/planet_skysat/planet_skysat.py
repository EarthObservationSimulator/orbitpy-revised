""" The below script calculates point coverage with Planet SkySat satellites over the CONUS region.

"""
import shutil
from typing import Any
import sys
import os
import time
import json

from eosimutils.base import JsonSerializer

from orbitpy.mission import Mission

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dshield_format_converter import write_dshield_format_of_propagator_results, write_dshield_format_of_contact_results, write_dshield_format_of_eclipse_results, write_dshield_format_of_gnssr_coverage_results, write_dshield_format_of_specular_trajectory_results, write_dshield_format_of_point_coverage_results

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
point_coverage_calculator_results = results.get("coverage_calculator_results", {})

elapsed_time = time.process_time() - exec_start_time
print(f"Mission complete. Time taken to execute in seconds: {elapsed_time:.2f}")

##### Save complete results to JSON file. #####
results_fp = os.path.join(results_dir, 'MissionOutput.json')
time_start = time.process_time()
JsonSerializer.save_to_json(results, results_fp)
elapsed_time = time.process_time() - time_start
print(f"Results written to MissionOutput.json. Time taken: {elapsed_time:.2f} seconds")


##### Save results to the format expected in the D-SHIELD project #####
print("Writing results in the format expected by the D-SHIELD project to output directory.")

epoch = mission.start_time.to_dict(time_format="GREGORIAN_DATE", time_scale="UTC")
step_size = mission.propagator.step_size

#### Write propagation results using the in-memory propagator_results. ####

exec_start_time = time.process_time()
propagator_results_serializable = JsonSerializer.to_serializable(propagator_results)
elapsed_time = time.process_time() - exec_start_time
print(f'elapsed time for propagator_results_serializable is {elapsed_time}')

exec_start_time = time.process_time()
write_dshield_format_of_propagator_results(propagator_results_serializable, epoch, results_dir)
elapsed_time = time.process_time() - exec_start_time
print(f'elapsed time for converting to csv format and writing to disk is {elapsed_time}')
#########################################################################################

#### Write contact finder results using the in-memory contact_finder_results. ####
exec_start_time = time.process_time()
contact_finder_results_serializable = JsonSerializer.to_serializable(contact_finder_results)
elapsed_time = time.process_time() - exec_start_time
print(f'elapsed time for contact_finder_results_serializable is {elapsed_time}')

exec_start_time = time.process_time()
write_dshield_format_of_contact_results(contact_finder_results_serializable, results_dir, epoch, step_size_seconds=step_size)
elapsed_time = time.process_time() - exec_start_time
print(f'elapsed time for converting to csv format and writing to disk is {elapsed_time}')

#########################################################################################

#### Write eclipse finder results using the in-memory eclipse_finder_results. ####
exec_start_time = time.process_time()
eclipse_finder_results_serializable = JsonSerializer.to_serializable(eclipse_finder_results)
elapsed_time = time.process_time() - exec_start_time
print(f'elapsed time for eclipse_finder_results_serializable is {elapsed_time}')

exec_start_time = time.process_time()
write_dshield_format_of_eclipse_results(eclipse_finder_results_serializable, results_dir, epoch, step_size_seconds=step_size)
elapsed_time = time.process_time() - exec_start_time
print(f'elapsed time for converting to csv format and writing to disk is {elapsed_time}')

#########################################################################################

#### Write coverage calculator results using the in-memory coverage_calculator_results. ####

exec_start_time = time.process_time()
point_coverage_calculator_results_serializable = JsonSerializer.to_serializable(point_coverage_calculator_results)
elapsed_time = time.process_time() - exec_start_time
print(f'elapsed time for point_coverage_calculator_results_serializable is {elapsed_time}')

exec_start_time = time.process_time()
epoch_ephemeris_seconds = mission.start_time.to_spice_ephemeris_time()
write_dshield_format_of_point_coverage_results(point_coverage_calculator_results_serializable, results_dir, epoch, epoch_ephemeris_seconds, step_size_seconds=step_size)
elapsed_time = time.process_time() - exec_start_time
print(f'elapsed time for converting to csv format and writing to disk is {elapsed_time}')

#########################################################################################