"""Convert Grid.csv lat/lon coordinates into the spatial_points.json format.

Grid.csv columns:  GP index, lat [deg], lon [deg]
spatial_points.json:  {"geo_positions": [[lat, lon, 0], ...]}

Usage:
    python convert_grid_to_spatial_points.py [input.csv] [output.json]
"""
import csv
import json
import os
import sys


def convert(input_csv, output_json):
    geo_positions = []
    with open(input_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lat = float(row["lat [deg]"])
            lon = float(row["lon [deg]"])
            geo_positions.append([lat, lon, 0])

    with open(output_json, "w") as f:
        json.dump({"geo_positions": geo_positions}, f)

    print(f"Wrote {len(geo_positions)} points to {output_json}")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    input_csv = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "Grid.csv")
    output_json = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, "spatial_points.json")
    convert(input_csv, output_json)
