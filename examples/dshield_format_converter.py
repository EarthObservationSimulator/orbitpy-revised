import os
import shutil
import pandas as pd

from eosimutils.time import AbsoluteDate

def write_dshield_format_of_propagator_results(propagator_results: list[dict], out_dir: str) -> None:
    """
    Write one CSV file per spacecraft with the format requested by D-SHIELD project.
    Uses propagator_results in-memory structure (list of spacecraft dicts).
    There is an assumption that the propagated times have an uniform step size.

    The resulting files are produced within the following directory structure:
    
        <out_dir>/<spacecraft_name>/<propagation>/state.csv
    
    - Each spacrcraft produces one CSV file named state.csv.
    - The '<spacecraft_name>' folder is created if it does not exist. If it exists, it is reused.
    - The '<propagation>' folder is erased and recreated.

    Args:
        propagator_results (list[dict]): List of spacecraft propagation results.
                        See `orbitpy.mission.Mission.execute_propagation` for structure.
        out_dir (str): Output directory to save the CSV files.
    Returns:
        None
    """
    for sc in propagator_results:

        # create directory and file path
        sc_name = sc.get("spacecraft_name", sc.get("spacecraft_id", "spacecraft"))
        sc_folder = os.path.join(out_dir, sc_name)
        os.makedirs(sc_folder, exist_ok=True)
        # Create a fresh results folder.
        results_dir = os.path.join(sc_folder, "propagation")
        if os.path.exists(results_dir) and os.path.isdir(results_dir):
            shutil.rmtree(results_dir)
        os.makedirs(results_dir, exist_ok=True)
        out_fp = os.path.join(results_dir, f"state.csv")

        name = sc.get("spacecraft_name", sc.get("spacecraft_id", "spacecraft"))
        traj = sc.get("trajectory", {})
        frame = traj.get("frame", "UNKNOWN")
        time_info = traj.get("time", {})
        times = traj.get("time", {}).get("calendar_date", [])
        if not times:
            print(f"Skipping {name}: no time array")
            continue

        # obtain time format and scale from input (fallback to sensible defaults)
        time_format = time_info.get("time_format", "UNKNOWN_TIME_FORMAT")
        time_scale = time_info.get("time_scale", "")

        # parse times with pandas for robust ISO parsing
        times_dt = pd.to_datetime(times)
        if len(times_dt) >= 2:
            step_seconds = (times_dt[1] - times_dt[0]).total_seconds()
            mission_days = (times_dt[-1] - times_dt[0]).total_seconds() / 86400.0
        else:
            step_seconds = 0.0
            mission_days = 0.0

        pos_list = traj.get("data", [[], []])[0]
        vel_list = traj.get("data", [[], []])[1]

        
        with open(out_fp, "w", newline="") as f:
            # header lines (exact wording as requested by the D-SHIELD project)
            f.write(f"Satellite states are in {frame} frame\n")
            f.write(f"Epoch [Format: {time_format}. Scale: {time_scale}] is {times[0]}\n")
            f.write(f"Step size [s] is {float(step_seconds)}\n")
            f.write(f"Mission Duration [Days] is {mission_days}\n")
            # CSV header
            f.write("time index,x [km],y [km],z [km],vx [km/s],vy [km/s],vz [km/s]\n")

            # write rows
            for idx, (p, v) in enumerate(zip(pos_list, vel_list)):
                # ensure 3 components each
                px, py, pz = (p + [None]*3)[:3]
                vx, vy, vz = (v + [None]*3)[:3]
                row = f"{idx},{px},{py},{pz},{vx},{vy},{vz}\n"
                f.write(row)

        print(f"Wrote state CSV for {name}: {out_fp}")


def write_dshield_format_of_contact_results(contact_results: list[dict], out_dir: str, epoch_dict: dict, step_size_seconds: float = 60.0) -> None:
    """
    Write per-spacecraft / per-ground-station contact CSV files in the D-SHIELD style.

    The resulting files are produced within the following directory structure:
    
        <out_dir>/<spacecraft_name>/<ground_contact>/<ground_station_name>_contacts.csv
    
    - Each ground station will produce one CSV file named <ground_station_name>_contacts.csv.
    - The '<spacecraft_name>' folder is created if it does not exist. If it exists, it is reused.
    - The '<ground_contact>' folder is erased and recreated.

    The contact times indicated in the contact intervals need to be represented in Gregorian date UTC format.

    Args:
        contact_results (list[dict]): List of spacecraft to ground-station contact results.
                        See `orbitpy.mission.Mission.execute_gs_contact_finder` for structure.
        out_dir (str): Output directory to save the CSV files.
        epoch_dict (dict): Dictionary representing the epoch (start time) of the mission.
                           See `eosimutils.time.AbsoluteDate.to_dict` for structure.
        step_size_seconds (float): Step size in seconds used to compute time indices.

    """

    if step_size_seconds <= 0:
        # cannot compute indices; skip interval
        raise ValueError("step_size_seconds must be positive to compute time indices.")

    # parse epoch to string and datetime
    def _to_dt(t):
        if isinstance(t, dict):
            if t.get("time_format") != "GREGORIAN_DATE" or t.get("time_scale") != "UTC":
                raise ValueError(f"Unsupported time format: {t.get('time_format')} and/or time scale: {t.get('time_scale')}")
            
            return pd.to_datetime(t.get("calendar_date"))
        return pd.to_datetime(t)
    epoch_str = epoch_dict.get("calendar_date", None)
    if epoch_str is None:
        raise ValueError("epoch_dict must contain 'calendar_date' key")
    tf_label = epoch_dict.get("time_format", "UNKNOWN")
    ts_label = epoch_dict.get("time_scale", "UNKNOWN")
    epoch_dt = _to_dt(epoch_str)

    # iterate over spacecrafts
    for sc in contact_results:
        sc_name = sc.get("spacecraft_name", sc.get("spacecraft_id", "spacecraft"))
        sc_folder = os.path.join(out_dir, sc_name)
        os.makedirs(sc_folder, exist_ok=True)

        # Create a fresh results folder.
        results_dir = os.path.join(sc_folder, "ground_contact")
        if os.path.exists(results_dir) and os.path.isdir(results_dir):
            shutil.rmtree(results_dir)
        os.makedirs(results_dir, exist_ok=True)

        contacts = sc.get("contacts", [])
        if not contacts:
            print(f"Skipping {sc_name}: no contacts")
            continue

        for g_idx,gs in enumerate(contacts):
            gs_name = gs.get("ground_station_name", gs.get("ground_station_id", "groundstation"))
            contact_info = gs.get("contact_info", [])
            if len(contact_info) == 0:
                print(f"Skipping {sc_name} -> {gs_name}: no contact intervals")
                continue

            # write CSV
            out_fp = os.path.join(results_dir, f"gs{g_idx+1}_contacts.csv")
            with open(out_fp, "w", newline="") as f:
                header_title = f"Contacts between spacecraft {sc_name} with Ground station {gs_name}"
                f.write(header_title + "\n")
                f.write(f"Epoch [Format: {tf_label}. Scale: {ts_label}] is {epoch_str}\n")
                f.write(f"Step size [s] is {float(step_size_seconds)}\n")
                f.write("start index,end index\n")

                # write intervals as indices relative to epoch and step size
                for interval in contact_info:
                    if not interval or len(interval) < 2:
                        continue
                    # parse start/end times
                    try:
                        start_dt = _to_dt(interval[0])
                        end_dt = _to_dt(interval[1])
                    except Exception:
                        continue

                    start_idx = int(round((start_dt - epoch_dt).total_seconds() / step_size_seconds))
                    end_idx = int(round((end_dt - epoch_dt).total_seconds() / step_size_seconds))
                    f.write(f"{start_idx},{end_idx}\n")

            print(f"Wrote contact CSV for {sc_name} -> {gs_name}: {out_fp}")