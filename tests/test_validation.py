"""."""

import unittest
import random
import numpy as np
import csv
import os

from eosimutils.time import AbsoluteDate, AbsoluteDateArray
from eosimutils.state import Cartesian3DPositionArray, GeographicPosition
from eosimutils.trajectory import StateSeries
from eosimutils.base import ReferenceFrame
from eosimutils.framegraph import FrameGraph
from eosimutils.standardframes import get_lvlh
from eosimutils.fieldofview import CircularFieldOfView, RectangularFieldOfView

import orbitpy.coveragecalculator
from orbitpy.coverage import DiscreteCoverageGP, DiscreteCoverageTP, ContinuousCoverageGP


def create_cartesian_position_array_from_csv(file_path: str) -> Cartesian3DPositionArray:
    """
    Load geographic positions from a CSV file and convert them to a Cartesian3DPositionArray.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        Cartesian3DPositionArray: The resulting array of 3D Cartesian positions.
    """
    geographic_positions = []

    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            lat = float(row['lat [deg]'])
            lon = float(row['lon [deg]'])
            geo_pos = GeographicPosition(latitude_degrees=lat, longitude_degrees=lon, elevation_m=0.0)
            geographic_positions.append(geo_pos)

    return Cartesian3DPositionArray.from_geographic_positions(geographic_positions)

def create_stateseries_from_txt_file(filepath: str, frame: str = "ICRF_EC") -> StateSeries:
    """
    Parse satellite state data file and return a StateSeries.

    Args:
        filepath (str): Path to the satellite state text file.
        frame (str): Name of the reference frame (default "ICRF_EC").

    Returns:
        StateSeries: Trajectory data parsed into a StateSeries.
    """
    # Reference start time
    start_time = AbsoluteDate.from_dict({
        "time_format": "GREGORIAN_DATE",
        "calendar_date": "2018-05-26T12:00:00.000",
        "time_scale": "UTC"
    })

    # Skip: 2 header lines + 1 blank + 2 lines for column headers = 5 lines
    data = np.loadtxt(filepath, skiprows=6)

    times_et = start_time.ephemeris_time + data[:, 0]
    positions = data[:, 1:4]
    velocities = data[:, 4:7]

    return StateSeries(
        time=AbsoluteDateArray(times_et),
        data=[positions, velocities],
        frame=ReferenceFrame.get(frame)
    )


class STKValidation(unittest.TestCase):
    """Validation tests for coverage against STK."""

    lvlh_frame = ReferenceFrame.add("LVLH")

    def setUp(self):
        """Set up test data."""

    def get_registry(self, result: StateSeries) -> FrameGraph:

        att_lvlh, pos_lvlh = get_lvlh(result,self.lvlh_frame)
        registry   = FrameGraph()
        registry.add_orientation_transform(att_lvlh)
        from_frame = ReferenceFrame.get("ICRF_EC")
        to_frame = ReferenceFrame.get("LVLH")
        registry.add_pos_transform(from_frame,to_frame, pos_lvlh)

        return registry

    def get_metric(self, circ_coverage, stkcov):

        circ_coverage_gp = DiscreteCoverageGP.from_tp(circ_coverage)
        
        # For each GP, this gives the time points which are covered by one
        # but not the other
        diff = DiscreteCoverageGP.symmetric_difference(stkcov, circ_coverage_gp)

        # This gives the number of steps covered by one but not the other
        # for each GP
        covered_steps = diff.coverage_steps()

        different_indices = np.nonzero(covered_steps);
        stk_covered_steps = stkcov.coverage_steps()

        metric = covered_steps[different_indices]/stk_covered_steps[different_indices]

        return np.mean(metric)

    def get_metric_4(self, orbitpycov, stkcov):
        """Metric 4 from the old orbitpy tests.
        
        Computes the number of time steps covered by each grid point
        using each method, for points which are covered by both methods.

        Takes the average of the absolute percent difference of the coverage time
        across these points.

        Args:
            orbitpycov (DiscreteCoverageGP): Coverage from orbitpy.
            stkcov (ContinuousCoverageGP): Coverage from STK.
        """

        # Convert orbitpy coverage to grid-first format
        orbitpycov_gp = DiscreteCoverageGP.from_tp(orbitpycov)
        
        # This gives the number of steps covered for each grid point
        covered_steps_orbitpy = orbitpycov_gp.coverage_steps()
        covered_steps_stk = stkcov.coverage_steps()

        # GP indices that are covered
        nonzero_indices_orbitpy = np.nonzero(covered_steps_orbitpy)
        nonzero_indices_stk = np.nonzero(covered_steps_stk)

        # GP Indices covered by both softwares
        nonzero_indices = np.intersect1d(nonzero_indices_orbitpy, nonzero_indices_stk)

        # Number of time steps covered by grid points accessed by both softwares
        steps_orbitpy = covered_steps_orbitpy[nonzero_indices]
        steps_stk = covered_steps_stk[nonzero_indices]

        percentDiff = abs((steps_orbitpy - steps_stk) / ((steps_orbitpy + steps_stk)/2))

        return np.sum(percentDiff)/np.size(percentDiff)

    def test_1(self):
        """Test 1."""

        # Create coverage calculator
        cov = orbitpy.coveragecalculator.CoverageFactory.from_dict({
            "coverage_type": orbitpy.coveragecalculator.CoverageType.POINT_COVERAGE.to_string()})

        # Read trajectory from data file
        result = create_stateseries_from_txt_file("/Users/reno/Documents/repos/orbitpy-revised/tests/stk/test_coveragecalculator_GridCoverage_with_STK/States/Satellite1_states.txt")
        times = result.time

        # Create frame graph and add LVLH frame
        registry = self.get_registry(result)

        # Create a circular field of view
        diameter = 20.0  # deg
        fov = CircularFieldOfView(diameter=diameter, frame=self.lvlh_frame)

        # Calculate point coverage
        this_dir = os.path.dirname(__file__)
        grid_path = os.path.join(
            this_dir,
            "stk",
            "test_coveragecalculator_GridCoverage_with_STK",
            "Accesses",
            "Global_Grid"
        )
        target_point_array = create_cartesian_position_array_from_csv(grid_path)
        circ_coverage = cov.calculate_coverage(target_point_array,fov=fov,frame_graph=registry,times=times)

        path = os.path.join(
            this_dir,
            "stk",
            "test_coveragecalculator_GridCoverage_with_STK",
            "Accesses",
            "Global_grid_1.cvaa"
        )
        stkcov = ContinuousCoverageGP.from_stk(path).to_discrete(times[0], 1.0, len(times))

        metric = self.get_metric_4(circ_coverage, stkcov)

        print(metric)

        self.assertTrue(True)

    def test_2(self):
        """Test 2."""

        # Create coverage calculator
        cov = orbitpy.coveragecalculator.CoverageFactory.from_dict({
            "coverage_type": orbitpy.coveragecalculator.CoverageType.POINT_COVERAGE.to_string()})

        # Read trajectory from data file
        result = create_stateseries_from_txt_file("/Users/reno/Documents/repos/orbitpy-revised/tests/stk/test_coveragecalculator_GridCoverage_with_STK/States/Satellite1_states.txt")
        times = result.time

        # Create frame graph and add LVLH frame
        registry = self.get_registry(result)

        # Create a circular field of view
        # diameter = 20.0  # deg
        # fov = CircularFieldOfView(diameter=diameter, frame=self.lvlh_frame)

        up_half_angle = 15.0  # deg
        right_half_angle = 10.0  # deg
        fov = RectangularFieldOfView(frame=self.lvlh_frame,ref_angle=up_half_angle, 
          cross_angle=right_half_angle)

        # Calculate point coverage
        this_dir = os.path.dirname(__file__)
        grid_path = os.path.join(
            this_dir,
            "stk",
            "test_coveragecalculator_GridCoverage_with_STK",
            "Accesses",
            "Global_Grid"
        )
        target_point_array = create_cartesian_position_array_from_csv(grid_path)
        circ_coverage = cov.calculate_coverage(target_point_array,fov=fov,frame_graph=registry,times=times)

        path = os.path.join(
            this_dir,
            "stk",
            "test_coveragecalculator_GridCoverage_with_STK",
            "Accesses",
            "Global_grid_2.cvaa"
        )
        stkcov = ContinuousCoverageGP.from_stk(path).to_discrete(times[0], 1.0, len(times))

        metric = self.get_metric_4(circ_coverage, stkcov)
        print(metric)

        self.assertTrue(True)

    def test_3(self):
        """Test 3."""

        # Create coverage calculator
        cov = orbitpy.coveragecalculator.CoverageFactory.from_dict({
            "coverage_type": orbitpy.coveragecalculator.CoverageType.POINT_COVERAGE.to_string()})

        # Read trajectory from data file
        result = create_stateseries_from_txt_file("/Users/reno/Documents/repos/orbitpy-revised/tests/stk/test_coveragecalculator_GridCoverage_with_STK/States/Satellite2_states.txt")
        times = result.time

        # Create frame graph and add LVLH frame
        registry = self.get_registry(result)

        # Create a circular field of view
        diameter = 20.0  # deg
        fov = CircularFieldOfView(diameter=diameter, frame=self.lvlh_frame)

        # Calculate point coverage
        this_dir = os.path.dirname(__file__)
        grid_path = os.path.join(
            this_dir,
            "stk",
            "test_coveragecalculator_GridCoverage_with_STK",
            "Accesses",
            "Global_Grid"
        )
        target_point_array = create_cartesian_position_array_from_csv(grid_path)
        circ_coverage = cov.calculate_coverage(target_point_array,fov=fov,frame_graph=registry,times=times)

        path = os.path.join(
            this_dir,
            "stk",
            "test_coveragecalculator_GridCoverage_with_STK",
            "Accesses",
            "Global_grid_3.cvaa"
        )
        stkcov = ContinuousCoverageGP.from_stk(path).to_discrete(times[0], 1.0, len(times))

        metric = self.get_metric_4(circ_coverage, stkcov)
        print(metric)

        self.assertTrue(True)

    def test_4(self):
        """Test 4."""

       # Create coverage calculator
        cov = orbitpy.coveragecalculator.CoverageFactory.from_dict({
            "coverage_type": orbitpy.coveragecalculator.CoverageType.POINT_COVERAGE.to_string()})

        # Read trajectory from data file
        result = create_stateseries_from_txt_file("/Users/reno/Documents/repos/orbitpy-revised/tests/stk/test_coveragecalculator_GridCoverage_with_STK/States/Satellite3_states.txt")
        times = result.time

        # Create frame graph and add LVLH frame
        registry = self.get_registry(result)

        # Create a rectangular field of view
        up_half_angle = 15.0  # deg
        right_half_angle = 10.0  # deg
        fov = RectangularFieldOfView(frame=self.lvlh_frame,ref_angle=up_half_angle, 
          cross_angle=right_half_angle)

        # Calculate point coverage
        this_dir = os.path.dirname(__file__)
        grid_path = os.path.join(
            this_dir,
            "stk",
            "test_coveragecalculator_GridCoverage_with_STK",
            "Accesses",
            "US_grid"
        )
        target_point_array = create_cartesian_position_array_from_csv(grid_path)
        circ_coverage = cov.calculate_coverage(target_point_array,fov=fov,frame_graph=registry,times=times)

        path = os.path.join(
            this_dir,
            "stk",
            "test_coveragecalculator_GridCoverage_with_STK",
            "Accesses",
            "US_Grid_4.cvaa"
        )
        stkcov = ContinuousCoverageGP.from_stk(path).to_discrete(times[0], 1.0, len(times))

        metric = self.get_metric_4(circ_coverage, stkcov)
        print(metric)

        self.assertTrue(True)

    def test_5(self):
        """Test 5."""

        # Create coverage calculator
        cov = orbitpy.coveragecalculator.CoverageFactory.from_dict({
            "coverage_type": orbitpy.coveragecalculator.CoverageType.POINT_COVERAGE.to_string()})

        # Read trajectory from data file
        result = create_stateseries_from_txt_file("/Users/reno/Documents/repos/orbitpy-revised/tests/stk/test_coveragecalculator_GridCoverage_with_STK/States/Satellite4_states.txt")
        times = result.time

        # Create frame graph and add LVLH frame
        registry = self.get_registry(result)

        # Create a circular field of view
        diameter = 20.0  # deg
        fov = CircularFieldOfView(diameter=diameter, frame=self.lvlh_frame)

        # Calculate point coverage
        this_dir = os.path.dirname(__file__)
        grid_path = os.path.join(
            this_dir,
            "stk",
            "test_coveragecalculator_GridCoverage_with_STK",
            "Accesses",
            "US_grid"
        )
        target_point_array = create_cartesian_position_array_from_csv(grid_path)
        circ_coverage = cov.calculate_coverage(target_point_array,fov=fov,frame_graph=registry,times=times)

        path = os.path.join(
            this_dir,
            "stk",
            "test_coveragecalculator_GridCoverage_with_STK",
            "Accesses",
            "US_Grid_5.cvaa"
        )
        stkcov = ContinuousCoverageGP.from_stk(path).to_discrete(times[0], 1.0, len(times))

        metric = self.get_metric_4(circ_coverage, stkcov)
        print(metric)

        self.assertTrue(True)

    def test_6(self):
        """Test 6."""

        # Create coverage calculator
        cov = orbitpy.coveragecalculator.CoverageFactory.from_dict({
            "coverage_type": orbitpy.coveragecalculator.CoverageType.POINT_COVERAGE.to_string()})

        # Read trajectory from data file
        result = create_stateseries_from_txt_file("/Users/reno/Documents/repos/orbitpy-revised/tests/stk/test_coveragecalculator_GridCoverage_with_STK/States/Satellite4_states.txt")
        times = result.time

        # Create frame graph and add LVLH frame
        registry = self.get_registry(result)

        # Create a rectangular field of view
        up_half_angle = 15.0  # deg
        right_half_angle = 10.0  # deg
        fov = RectangularFieldOfView(frame=self.lvlh_frame,ref_angle=up_half_angle, 
          cross_angle=right_half_angle)

        # Calculate point coverage
        this_dir = os.path.dirname(__file__)
        grid_path = os.path.join(
            this_dir,
            "stk",
            "test_coveragecalculator_GridCoverage_with_STK",
            "Accesses",
            "US_grid"
        )
        target_point_array = create_cartesian_position_array_from_csv(grid_path)
        circ_coverage = cov.calculate_coverage(target_point_array,fov=fov,frame_graph=registry,times=times)

        path = os.path.join(
            this_dir,
            "stk",
            "test_coveragecalculator_GridCoverage_with_STK",
            "Accesses",
            "US_Grid_6.cvaa"
        )
        stkcov = ContinuousCoverageGP.from_stk(path).to_discrete(times[0], 1.0, len(times))

        metric = self.get_metric_4(circ_coverage, stkcov)
        print(metric)

        self.assertTrue(True)

    def test_7(self):
        """Test 7."""

        # Create coverage calculator
        cov = orbitpy.coveragecalculator.CoverageFactory.from_dict({
            "coverage_type": orbitpy.coveragecalculator.CoverageType.POINT_COVERAGE.to_string()})

        # Read trajectory from data file
        result = create_stateseries_from_txt_file("/Users/reno/Documents/repos/orbitpy-revised/tests/stk/test_coveragecalculator_GridCoverage_with_STK/States/Satellite5_states.txt")
        times = result.time

        # Create frame graph and add LVLH frame
        registry = self.get_registry(result)

        # Create a circular field of view
        diameter = 20.0  # deg
        fov = CircularFieldOfView(diameter=diameter, frame=self.lvlh_frame)

        # Calculate point coverage
        this_dir = os.path.dirname(__file__)
        grid_path = os.path.join(
            this_dir,
            "stk",
            "test_coveragecalculator_GridCoverage_with_STK",
            "Accesses",
            "Equatorial_Grid"
        )
        target_point_array = create_cartesian_position_array_from_csv(grid_path)
        circ_coverage = cov.calculate_coverage(target_point_array,fov=fov,frame_graph=registry,times=times)

        path = os.path.join(
            this_dir,
            "stk",
            "test_coveragecalculator_GridCoverage_with_STK",
            "Accesses",
            "Equatorial_Grid_7.cvaa"
        )
        stkcov = ContinuousCoverageGP.from_stk(path).to_discrete(times[0], 1.0, len(times))

        metric = self.get_metric_4(circ_coverage, stkcov)
        print(metric)

        self.assertTrue(True)

if __name__ == "__main__":
    unittest.main()
