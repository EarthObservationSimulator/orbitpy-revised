import os
import shutil
import pandas as pd

def write_dshield_format_of_propagator_results(propagator_results: list[dict], epoch_dict: dict, out_dir: str) -> None:
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
        epoch_dict (dict): Dictionary containing epoch information in Gregorian date UTC format.
                           See `eosimutils.time.AbsoluteDate.to_dict` for structure.
        out_dir (str): Output directory to save the CSV files.
    Returns:
        None
    """
    epoch_str = epoch_dict.get("calendar_date")
    if epoch_str is None:
        raise ValueError("epoch_dict must contain 'calendar_date' key")
    tf_label = epoch_dict.get("time_format", "UNKNOWN")
    ts_label = epoch_dict.get("time_scale", "UNKNOWN")
    
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
        ephemeris_seconds = traj.get("time", {}).get("ephemeris_time", [])
        if not ephemeris_seconds:
            print(f"Skipping {name}: no ephemeris time array")
            continue

        # parse times with pandas for robust ISO parsing
        if len(ephemeris_seconds) >= 2:
            step_seconds = (ephemeris_seconds[1] - ephemeris_seconds[0])
            mission_days = (ephemeris_seconds[-1] - ephemeris_seconds[0]) / 86400.0
        else:
            step_seconds = 0.0
            mission_days = 0.0

        pos_list = traj.get("data", [[], []])[0]
        vel_list = traj.get("data", [[], []])[1]

        
        with open(out_fp, "w", newline="") as f:
            # header lines (exact wording as requested by the D-SHIELD project)
            f.write(f"Satellite states are in {frame} frame\n")
            f.write(f"Epoch [Format: {tf_label}. Scale: {ts_label}] is {epoch_str}\n")
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


def write_dshield_format_of_contact_results(contact_results: list[dict], out_dir: str, epoch_dict: dict, step_size_seconds: float) -> None:
    """
    Write per-spacecraft / per-ground-station contact CSV files in the D-SHIELD style.

    The resulting files are produced within the following directory structure:
    
        <out_dir>/<spacecraft_name>/<ground_contact>/<ground_station_name>_contacts.csv
    
    - Each ground station will produce one CSV file named <ground_station_name>_contacts.csv.
    - The '<spacecraft_name>' folder is created if it does not exist. If it exists, it is reused.
    - The '<ground_contact>' folder is erased and recreated.

    The contact times indicated in the contact intervals need to be represented in Gregorian date UTC format.

    Args:
        contact_results (list[dict): List of spacecraft to ground-station contact results.
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

def write_dshield_format_of_eclipse_results(eclipse_results: list[dict], out_dir: str, epoch_dict: dict, step_size_seconds: float) -> None:
    """
    Write per-spacecraft eclipse interval CSV files in the D-SHIELD style.

    The resulting files are produced within the following directory structure:

        <out_dir>/<spacecraft_name>/eclipse/eclipse_intervals.csv

    - Each spacecraft produces one CSV file named eclipse_intervals.csv.
    - The '<spacecraft_name>' folder is created if it does not exist. If it exists, it is reused.
    - The '<eclipse>' folder is erased and recreated.

    The eclipse times indicated in the eclipse intervals need to be represented in Gregorian date UTC format.

    Args:
        eclipse_results (list[dict]): List of spacecraft eclipse results. Each entry should have
                                      "eclipse_info" as a list of [start, end] time pairs.
        out_dir (str): Output directory to save the CSV files.
        epoch_dict (dict): Dictionary representing the epoch (start time) of the mission.
                           See `eosimutils.time.AbsoluteDate.to_dict` for structure.
        step_size_seconds (float): Step size in seconds used to compute time indices.
    """
    if step_size_seconds <= 0:
        raise ValueError("step_size_seconds must be positive to compute time indices.")

    # helper to parse provided time representation to pandas.Timestamp
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

    for sc in eclipse_results:
        sc_name = sc.get("spacecraft_name", sc.get("spacecraft_id", "spacecraft"))
        sc_folder = os.path.join(out_dir, sc_name)
        os.makedirs(sc_folder, exist_ok=True)

        # Create a fresh results folder.
        results_dir = os.path.join(sc_folder, "eclipse")
        if os.path.exists(results_dir) and os.path.isdir(results_dir):
            shutil.rmtree(results_dir)
        os.makedirs(results_dir, exist_ok=True)

        eclipse_info = sc.get("eclipse_info", [])
        if not eclipse_info:
            print(f"Skipping {sc_name}: no eclipse intervals")
            continue

        out_fp = os.path.join(results_dir, "eclipse_intervals.csv")
        with open(out_fp, "w", newline="") as f:
            header_title = f"Eclipse intervals for spacecraft {sc_name}"
            f.write(header_title + "\n")
            f.write(f"Epoch [Format: {tf_label}, Scale: {ts_label}] is {epoch_str}\n")
            f.write(f"Step size [s] is {float(step_size_seconds)}\n")
            f.write("start index,end index\n")

            for interval in eclipse_info:
                if not interval or len(interval) < 2:
                    continue
                try:
                    start_dt = _to_dt(interval[0])
                    end_dt = _to_dt(interval[1])
                except Exception:
                    # skip malformed interval
                    continue

                start_idx = int(round((start_dt - epoch_dt).total_seconds() / step_size_seconds))
                end_idx = int(round((end_dt - epoch_dt).total_seconds() / step_size_seconds))
                f.write(f"{start_idx},{end_idx}\n")

        print(f"Wrote eclipse CSV for {sc_name}: {out_fp}")


def write_dshield_format_of_gnssr_coverage_results(coverage_results: list[dict], out_dir: str, epoch_dict: dict, epoch_ephemeris_seconds: float, step_size_seconds: float) -> None:
    """
    Write per-spacecraft, per-sensor coverage CSV files in the D-SHIELD style.
    """
    if step_size_seconds <= 0:
        raise ValueError("step_size_seconds must be positive to compute time indices.")

    epoch_str = epoch_dict.get("calendar_date")
    if epoch_str is None:
        raise ValueError("epoch_dict must contain 'calendar_date' key")
    tf_label = epoch_dict.get("time_format", "UNKNOWN")
    ts_label = epoch_dict.get("time_scale", "UNKNOWN")
    
    for sc in coverage_results:
        sc_name = sc.get("spacecraft_name", sc.get("spacecraft_id", "spacecraft"))
        sc_id = sc.get("spacecraft_id", "")
        sc_folder = os.path.join(out_dir, sc_name)
        os.makedirs(sc_folder, exist_ok=True)

        coverage_dir = os.path.join(sc_folder, "access")
        if os.path.exists(coverage_dir) and os.path.isdir(coverage_dir):
            shutil.rmtree(coverage_dir)
        os.makedirs(coverage_dir, exist_ok=True)

        sensors = sc.get("total_spacecraft_coverage", [])
        if not sensors:
            print(f"Skipping {sc_name}: no sensor coverage data")
            continue

        for sensor_idx, sensor in enumerate(sensors):
            sensor_name = sensor.get("sensor_name") or ""
            sensor_id = sensor.get("sensor_id")
            out_fp = os.path.join(coverage_dir, f"sensor{sensor_idx+1}_access.csv")

            sensor_coverages = sensor.get("total_sensor_coverage", [])
            if not sensor_coverages:
                print(f"Skipping {sc_name} sensor {sensor_name} id: {sensor_id}: no coverage data")
                continue

            with open(out_fp, "w", newline="") as f:
                f.write(f"Spacecraft with name {sc_name}, id {sc_id}. Sensor with name {sensor_name}, id {sensor_id} \n")
                f.write(f"Epoch [Format: {tf_label}. Scale: {ts_label}] is {epoch_str}\n")
                f.write(f"Step size [s] is {float(step_size_seconds)}\n")
                f.write("\"time index\" \"source name\" \"GP index\"\n")

                # iterate over gnss spacecrafts (sources) source => gnss spacecraft
                for source in sensor_coverages:
                    source_name = source.get("gnss_spacecraft_name") or "UNKNOWN SOURCE NAME"
                    source_id = source.get("gnss_spacecraft_id")
                    coverage_info = source.get("coverage_info", {})
                    time_info = coverage_info.get("time", {})
                    # expected time format is 'SPICE_ET'
                    ephemeris_seconds = time_info.get("ephemeris_time", [])
                    if not ephemeris_seconds:
                        continue

                    coverage_lists = coverage_info.get("coverage", [])
                    if len(ephemeris_seconds) != len(coverage_lists):
                        print(f"Warning: Mismatched lengths for coverage times and lists for {sc_name} sensor {sensor_name} source {source_name}")
                    for idx in range(len(ephemeris_seconds)):
                        gp_indices = coverage_lists[idx]
                        if not gp_indices:
                            continue
                        time_idx = int(round((ephemeris_seconds[idx] - epoch_ephemeris_seconds) / step_size_seconds))
                        gp_str = ",".join(str(int(gp)) for gp in gp_indices)
                        f.write(f"{time_idx} {source_name} {gp_str}\n")

            print(f"Wrote coverage CSV for {sc_name} sensor {sensor_name}: {out_fp}")

            aggregate_gnssr_coverage_by_time(out_fp, os.path.join(coverage_dir, f"sensor{sensor_idx+1}_access_aggregated.csv"))


def aggregate_gnssr_coverage_by_time(coverage_file: str, output_file: str) -> None:
    """
    Aggregate a GNSSR coverage CSV (write_dshield_format_of_gnssr_coverage_results output)
    by time index, listing all source names and GP indices observed at each time.
    """

    # Preserve the original three metadata header lines so the aggregated file mirrors them.
    with open(coverage_file, "r", encoding="utf-8") as f:
        header_lines = [next(f) for _ in range(4)]

    # Load the GNSSR coverage file while skipping the three metadata lines plus the column header.
    # The D-SHIELD output uses whitespace as the delimiter, so we read with regex-based splitting
    # and explicitly set the column names expected downstream.
    df = pd.read_csv(
        coverage_file,
        skiprows=4,
        sep=r"\s+",
        names=["time index", "source name", "GP index"],
        engine="python",
        dtype={"time index": "int64", "source name": "string", "GP index": "string"},
    )

    # Helper that merges all comma-separated GP index strings in a pandas Series into one
    # sorted, de-duplicated comma-separated string. Each GP index token is stripped of
    # surrounding whitespace before being converted into an integer.
    def _collect_gp(series: pd.Series) -> str:
        gp_set: set[int] = set()
        for value in series:
            for token in str(value).split(","):
                token = token.strip()
                if token:
                    gp_set.add(int(token))
        return ",".join(str(idx) for idx in sorted(gp_set))

    records: list[tuple[int, str, str]] = []
    # Group every row by its time index. Within each group we aggregate the distinct source names
    # and merge all GP indices using the helper above.
    for time_idx, group in df.groupby("time index"):
        # Returns a comma-separated, alphabetically sorted list of unique, non-empty source names found in the current time-index group
        sources = ",".join(sorted(set(name.strip() for name in group["source name"] if name.strip())))
        gp_indices = _collect_gp(group["GP index"])
        records.append((int(time_idx), sources, gp_indices))

    # Preserve chronological order
    records.sort(key=lambda item: item[0])

    # Write a space-delimited summary file that follows the D-SHIELD textual header style.
    with open(output_file, "w", encoding="utf-8") as f:
        # Reproduce the original metadata header lines.
        for meta_line in header_lines[:3]:
            f.write(meta_line)
        # Write a simple space-delimited column header appropriate for the aggregated content.
        f.write("\"time index\" \"source names\" \"GP index\"\n")
        # Write each aggregated row.
        for time_idx, sources, gp_indices in records:
            f.write(f"{time_idx} {sources} {gp_indices}\n")