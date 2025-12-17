"""File containing utility functions for working with GMAT produced data."""

import numpy as np
from datetime import datetime
from eosimutils.trajectory import StateSeries
from eosimutils.base import ReferenceFrame
from eosimutils.time import AbsoluteDateArray, AbsoluteDate

def parse_gmat_state_file(filepath):
    """
    Parse a GMAT state file and return a StateSeries object.

    Args:
        filepath (str): Path to the GMAT state file.

    Returns:
        StateSeries: Parsed trajectory.
    """
    times = []
    positions = []
    velocities = []
    with open(filepath, "r") as f:
        lines = f.readlines()
    # Skip header
    for line in lines[1:]:
        parts = line.strip().split()
        # Skip empty or non-data lines
        if len(parts) < 7:
            continue
        # Parse time string and state values
        time_str = " ".join(parts[:4])
        dt = datetime.strptime(time_str, "%d %b %Y %H:%M:%S.%f") # Parse to datetime object
        iso_str = dt.isoformat() # Get ISO formatted string
        try:
            # Try parsing the ISO formatted string to AbsoluteDate
            t = AbsoluteDate.from_dict({ "time_format": "GREGORIAN_DATE",
                "calendar_date": iso_str,
                "time_scale": "UTC"})
        except Exception:
            raise ValueError("Failed to parse time string: {}".format(time_str))
        state_vals = [float(x) for x in parts[4:10]]
        positions.append(state_vals[:3])
        velocities.append(state_vals[3:])
        times.append(t.ephemeris_time)
    # Convert to arrays
    times = np.array(times)
    positions = np.array(positions)
    velocities = np.array(velocities)
    # Build StateSeries
    time_obj = AbsoluteDateArray(times)
    frame = ReferenceFrame("ICRF_EC")  # MJ2000Eq is ICRF_EC in eosimutils (since eosimutils uses SPICE in which ICRF ~ J2000)
    return StateSeries(time_obj, [positions, velocities], frame)


def parse_gmat_contact_file(filepath):
    """
    Parse a GMAT ContactLocator file and return a list of (start, end) AbsoluteDate tuples.

    Args:
        filepath (str): Path to the ContactLocator file.

    Returns:
        List[Tuple[AbsoluteDate, AbsoluteDate]]: List of contact intervals.
    """
    contacts = []
    with open(filepath, "r") as f:
        lines = f.readlines()
    # Find the line index where the table starts (after header lines)
    table_start = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("Start Time"):
            table_start = i + 1
            break
    if table_start == 0:
        return []  # No contacts found

    # Parse each data line until an empty line or non-data line
    for line in lines[table_start:]:
        parts = line.strip().split()
        if len(parts) < 6:
            continue
        # Join date/time parts for start and stop
        start_str = " ".join(parts[0:4])
        stop_str = " ".join(parts[4:8])
        # Parse to AbsoluteDate using ISO string
        start_dt = datetime.strptime(start_str, "%d %b %Y %H:%M:%S.%f")
        stop_dt = datetime.strptime(stop_str, "%d %b %Y %H:%M:%S.%f")
        start_iso = start_dt.isoformat()
        stop_iso = stop_dt.isoformat()
        start_ad = AbsoluteDate.from_dict({
            "time_format": "GREGORIAN_DATE",
            "calendar_date": start_iso,
            "time_scale": "UTC"
        })
        stop_ad = AbsoluteDate.from_dict({
            "time_format": "GREGORIAN_DATE",
            "calendar_date": stop_iso,
            "time_scale": "UTC"
        })
        contacts.append((start_ad, stop_ad))
    return contacts

def parse_gmat_eclipse_file(filepath):
    """
    Parse a GMAT EclipseLocator file and return a list of eclipse event dictionaries.

    Args:
        filepath (str): Path to the EclipseLocator file.

    Returns:
        List[dict]: List of eclipse events, each as a dictionary with start, stop (AbsoluteDate), type, and event number.
    """
    events = []
    with open(filepath, "r") as f:
        lines = f.readlines()
    # Find the line index where the table starts (after header lines)
    for i, line in enumerate(lines):
        if line.strip().startswith("Start Time"):
            table_start = i + 1
            break
    # Parse each data line until an empty line or non-data line
    for line in lines[table_start:]:
        parts = line.strip().split()
        if len(parts) < 12:
            continue
        # Join date/time parts for start and stop
        start_str = " ".join(parts[0:4])
        stop_str = " ".join(parts[4:8])
        # Parse to AbsoluteDate using ISO string
        start_dt = datetime.strptime(start_str, "%d %b %Y %H:%M:%S.%f")
        stop_dt = datetime.strptime(stop_str, "%d %b %Y %H:%M:%S.%f")
        start_iso = start_dt.isoformat()
        stop_iso = stop_dt.isoformat()
        start_ad = AbsoluteDate.from_dict({
            "time_format": "GREGORIAN_DATE",
            "calendar_date": start_iso,
            "time_scale": "UTC"
        })
        stop_ad = AbsoluteDate.from_dict({
            "time_format": "GREGORIAN_DATE",
            "calendar_date": stop_iso,
            "time_scale": "UTC"
        })
        event_type = parts[10]
        event_number = int(parts[11])
        events.append({
            "start": start_ad,
            "stop": stop_ad,
            "type": event_type,
            "event_number": event_number
        })
    return events