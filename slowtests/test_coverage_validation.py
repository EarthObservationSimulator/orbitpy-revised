"""Validation tests for orbitpy.coveragecalculator module.

Tests orbitpy coverage calculator against STK data produced from various mission scenarios, varying
orbit type (circular, elliptical)/(inclined, equatorial), sensor type (rectangular, conical),
pointing direction (nadir/off nadir), and grid (global, US, equatorial). Thetrajectory data is read
from the STK output file directly, so these tests cover the frame transformations and geometry
calculations (point in FOV checks). To compare coverage output to orbitpy, the STK result is
discretized. 

"""

import unittest
import random
import numpy as np
import csv
import os

from eosimutils.time import AbsoluteDate, AbsoluteDateArray
from eosimutils.state import Cartesian3DPosition, Cartesian3DPositionArray, GeographicPosition, GeographicPositionArray
from eosimutils.trajectory import StateSeries
from eosimutils.base import ReferenceFrame
from eosimutils.framegraph import FrameGraph
from eosimutils.standardframes import LVLHType1FrameHandler
from eosimutils.fieldofview import CircularFieldOfView, RectangularFieldOfView, PolygonFieldOfView
from eosimutils.orientation import Orientation, ConstantOrientation

import orbitpy.coveragecalculator
from orbitpy.coverage import DiscreteCoverageGP, DiscreteCoverageTP, ContinuousCoverageGP
from orbitpy.plotting import plot_covered_steps

import matplotlib.pyplot as plt

def plot_results(orbitpycov, stkcov, target_point_array):
    """Plot the results of the coverage comparison."""

    orbitpycov_gp = DiscreteCoverageGP.from_tp(orbitpycov)
    ax_stk = plot_covered_steps(stkcov, target_point_array)
    ax_orbitpy = plot_covered_steps(orbitpycov_gp, target_point_array)

    # View angle
    ax_stk.view_init(elev=15, azim=-95)
    ax_orbitpy.view_init(elev=15, azim=-95)

    plt.show()

def create_cartesian_position_array_from_csv(file_path: str) -> Cartesian3DPositionArray:
    """
    Load geographic positions from a CSV file and convert them to a Cartesian3DPositionArray.

    Used to read STK coverage output for tests.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        Cartesian3DPositionArray: The resulting array of 3D Cartesian positions.
    """
    geographic_position_list = []

    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            lat = float(row['lat [deg]'])
            lon = float(row['lon [deg]'])
            geo_pos = GeographicPosition(
                latitude_degrees=lat, longitude_degrees=lon, elevation_m=0.0)
            geographic_position_list.append(geo_pos)
    
    geographic_positions_array = GeographicPositionArray.from_geographic_position_list(geographic_position_list)

    return Cartesian3DPositionArray.from_geographic_position_array(geographic_positions_array)

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
    sensor_frame = ReferenceFrame.add("Sensor")
    plot_tests = False # Set to True to enable plotting in tests

    # File paths for test data
    this_dir = os.path.dirname(__file__)
    states_dir = os.path.join(
        this_dir, "stk", "test_coveragecalculator_GridCoverage_with_STK", "States")
    accesses_dir = os.path.join(
        this_dir, "stk", "test_coveragecalculator_GridCoverage_with_STK",
        "Accesses")
    
    def get_registry(self, result: StateSeries) -> FrameGraph:
        """
        Create a frame graph which stores LVLH to ITRF transformation.

        Args:
            result (StateSeries): A state series containing satellite (inertial) trajectory data.

        Returns:
            FrameGraph: The frame graph with the LVLH to ITRF transformation
        """

        result = result.to_frame(ReferenceFrame.get("ITRF"))
        handler = LVLHType1FrameHandler("LVLH")
        att_lvlh, pos_lvlh = handler.get_transform(result)
        registry   = FrameGraph()
        registry.add_orientation_transform(att_lvlh)
        from_frame = ReferenceFrame.get("ITRF")
        to_frame = ReferenceFrame.get("LVLH")
        registry.add_pos_transform(from_frame,to_frame, pos_lvlh)

        return registry

    def get_metric_0(self, orbitpycov, stkcov):
        """  
        For each grid point, calculates the number of time steps N classified differently
        by STK and Orbitpy. Returns the max of N across all grid points.
        """

        # Convert orbitpy coverage to grid-first format
        orbitpycov_gp = DiscreteCoverageGP.from_tp(orbitpycov)
        
        # For each GP, this gives the time points which are covered by one
        # but not the other
        diff = DiscreteCoverageGP.symmetric_difference(stkcov, orbitpycov_gp)

        # This gives the number of steps covered by one but not the other
        # for each GP
        covered_steps = diff.coverage_steps()

        return max(covered_steps)
    
    def get_metric_1(self, orbitpycov, stkcov):
        """Metric 1 from the old orbitpy tests.
        
        Computes the number of grid points covered by each method.
        Takes the absolute percent difference of these two numbers."""

        # Convert orbitpy coverage to grid-first format
        orbitpycov_gp = DiscreteCoverageGP.from_tp(orbitpycov)
        
        # This gives the number of steps covered for each grid point
        covered_steps_orbitpy = orbitpycov_gp.coverage_steps()
        covered_steps_stk = stkcov.coverage_steps()

        # Calculate total number of covered points for each method
        covered_points_orbitpy = len(covered_steps_orbitpy[covered_steps_orbitpy > 0])
        covered_points_stk = len(covered_steps_stk[covered_steps_stk > 0])

        percentDiff = abs((covered_points_orbitpy - covered_points_stk) / (
            (covered_points_orbitpy + covered_points_stk)/2))

        return percentDiff
    
    def get_metric_2(self, orbitpycov, stkcov):
        """Metric 2 from the old orbitpy tests.
        
        Computes the total number of time steps (across all grid points) covered by each method.
        Takes the absolute percent difference of these two numbers."""

        # Convert orbitpy coverage to grid-first format
        orbitpycov_gp = DiscreteCoverageGP.from_tp(orbitpycov)
        
        # This gives the number of steps covered for each grid point
        covered_steps_orbitpy = orbitpycov_gp.coverage_steps()
        covered_steps_stk = stkcov.coverage_steps()

        # This gives the total number of steps covered for each method
        num_orbitpy = sum(covered_steps_orbitpy)
        num_stk =  sum(covered_steps_stk)

        percentDiff = abs((num_orbitpy - num_stk) / ((num_orbitpy + num_stk)/2))

        return percentDiff

    def get_metric_4(self, orbitpycov, stkcov):
        """Metric 4 from the old orbitpy tests.
        
        Computes the number of time steps covered by each grid point
        using each method, for points which are covered by both methods.

        Takes the average of the absolute percent difference of the coverage time
        across these points."""

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
    
    def get_metrics(self, orbitpycov, stkcov, num):
        """Print and assert all metrics for the coverage comparison.""" 

        metric0 = self.get_metric_0(orbitpycov, stkcov)
        print("Test " + str(num) + ", Metric 0 : ", metric0)

        metric1 = self.get_metric_1(orbitpycov, stkcov)
        print("Test " + str(num) + " Metric 1: ", metric1)

        metric2 = self.get_metric_2(orbitpycov, stkcov)
        print("Test " + str(num) + " Metric 2: ", metric2)

        metric4 = self.get_metric_4(orbitpycov, stkcov)
        print("Test " + str(num) + " Metric 4: ", metric4)

        self.assertLessEqual(
            metric0, 2, 
            "Maximum number of misclassified time points for any gp should be no higher than 2.")
        self.assertLessEqual(metric1, 0.0,
            "Percent difference in number of covered points should be zero")
        self.assertLessEqual(metric2, 0.001,
            "Percent diff in total number of covered time steps should be no higher than .1%.")
        self.assertLessEqual(metric4, 0.0025,
            "Average percent diff in number of covered time steps should be no higher than .25%.")

        return
    
    def test_1(self):
        """Test an equatorial orbit on a global grid with a 20 degree diameter conical sensor."""

        # Create coverage calculator
        cov = orbitpy.coveragecalculator.CoverageFactory.from_dict({
            "coverage_type": orbitpy.coveragecalculator.CoverageType.POINT_COVERAGE.to_string()})

        # Get file paths
        state_path = os.path.join(self.states_dir, "Satellite1_states.txt")
        accesses_path = os.path.join(self.accesses_dir,"Global_grid_1.cvaa")
        grid_path = os.path.join(self.accesses_dir,"Global_Grid")

        # Read trajectory from data file
        result = create_stateseries_from_txt_file(state_path)
        times = result.time

        # Create frame graph and add LVLH frame
        registry = self.get_registry(result)

        # Create a circular field of view
        diameter = 20.0  # deg
        fov = CircularFieldOfView(diameter=diameter, frame=self.lvlh_frame)

        # Calculate point coverage
        target_point_array = create_cartesian_position_array_from_csv(grid_path)
        orbitpycov = cov.calculate_coverage(
            target_point_array,fov=fov,frame_graph=registry,times=times)
        stkcov = ContinuousCoverageGP.from_stk(
            accesses_path, target_point_array).to_discrete(times[0], 1.0, len(times))

        if self.plot_tests:
            plot_results(orbitpycov, stkcov, target_point_array)

        self.get_metrics(orbitpycov, stkcov, 1)

    def test_2(self):
        """Test an equatorial orbit on a global grid with a 30 deg AT, 20 deg CT sensor."""

        # Create coverage calculator
        cov = orbitpy.coveragecalculator.CoverageFactory.from_dict({
            "coverage_type": orbitpy.coveragecalculator.CoverageType.POINT_COVERAGE.to_string()})

        # Get file paths
        state_path = os.path.join(self.states_dir, "Satellite1_states.txt")
        accesses_path = os.path.join(self.accesses_dir, "Global_grid_2.cvaa")
        grid_path = os.path.join(self.accesses_dir, "Global_Grid")

        # Read trajectory from data file
        result = create_stateseries_from_txt_file(state_path)
        times = result.time

        # Create frame graph and add LVLH frame
        registry = self.get_registry(result)

        # Create a rectangular field of view
        up_half_angle = 15.0  # deg
        right_half_angle = 10.0  # deg
        fov = RectangularFieldOfView(frame=self.lvlh_frame, ref_angle=up_half_angle,
                                     cross_angle=right_half_angle)

        # Calculate point coverage
        target_point_array = create_cartesian_position_array_from_csv(grid_path)
        orbitpycov = cov.calculate_coverage(
            target_point_array, fov=fov, frame_graph=registry, times=times)
        stkcov = ContinuousCoverageGP.from_stk(
            accesses_path, target_point_array).to_discrete(times[0], 1.0, len(times))
        
        fov = PolygonFieldOfView.from_rectangular(fov)
        orbitpycov_poly = cov.calculate_coverage(
            target_point_array, fov=fov, frame_graph=registry, times=times)

        if self.plot_tests:
            plot_results(orbitpycov, stkcov, target_point_array)

        self.get_metrics(orbitpycov, stkcov, 2)
        self.get_metrics(orbitpycov_poly, stkcov, 2)

    def test_3(self):
        """Test a near-equatorial orbit on a global grid with a 20 degree diameter conical 
        sensor."""

        # Create coverage calculator
        cov = orbitpy.coveragecalculator.CoverageFactory.from_dict({
            "coverage_type": orbitpy.coveragecalculator.CoverageType.POINT_COVERAGE.to_string()})

        # Get file paths
        state_path = os.path.join(self.states_dir, "Satellite2_states.txt")
        accesses_path = os.path.join(self.accesses_dir, "Global_grid_3.cvaa")
        grid_path = os.path.join(self.accesses_dir, "Global_Grid")

        # Read trajectory from data file
        result = create_stateseries_from_txt_file(state_path)
        times = result.time

        # Create frame graph and add LVLH frame
        registry = self.get_registry(result)

        # Create a circular field of view
        diameter = 20.0  # deg
        fov = CircularFieldOfView(diameter=diameter, frame=self.lvlh_frame)

        # Calculate point coverage
        target_point_array = create_cartesian_position_array_from_csv(grid_path)
        orbitpycov = cov.calculate_coverage(
            target_point_array, fov=fov, frame_graph=registry, times=times)
        stkcov = ContinuousCoverageGP.from_stk(
            accesses_path, target_point_array).to_discrete(times[0], 1.0, len(times))

        if self.plot_tests:
            plot_results(orbitpycov, stkcov, target_point_array)

        self.get_metrics(orbitpycov, stkcov, 3)

    def test_4(self):
        """Test a polar orbit on a US grid with a 30 deg AT, 20 deg CT sensor."""

        # Create coverage calculator
        cov = orbitpy.coveragecalculator.CoverageFactory.from_dict({
            "coverage_type": orbitpy.coveragecalculator.CoverageType.POINT_COVERAGE.to_string()})

        # Get file paths
        state_path = os.path.join(self.states_dir, "Satellite3_states.txt")
        accesses_path = os.path.join(self.accesses_dir, "US_Grid_4.cvaa")
        grid_path = os.path.join(self.accesses_dir, "US_grid")

        # Read trajectory from data file
        result = create_stateseries_from_txt_file(state_path)
        times = result.time

        # Create frame graph and add LVLH frame
        registry = self.get_registry(result)

        # Create a rectangular field of view
        up_half_angle = 15.0  # deg
        right_half_angle = 10.0  # deg
        fov = RectangularFieldOfView(frame=self.lvlh_frame, ref_angle=up_half_angle,
                                     cross_angle=right_half_angle)

        # Calculate point coverage
        target_point_array = create_cartesian_position_array_from_csv(grid_path)
        orbitpycov = cov.calculate_coverage(
            target_point_array, fov=fov, frame_graph=registry, times=times)
        stkcov = ContinuousCoverageGP.from_stk(
            accesses_path, target_point_array).to_discrete(times[0], 1.0, len(times))

        fov = PolygonFieldOfView.from_rectangular(fov)
        orbitpycov_poly = cov.calculate_coverage(
            target_point_array, fov=fov, frame_graph=registry, times=times)

        if self.plot_tests:
            plot_results(orbitpycov, stkcov, target_point_array)

        self.get_metrics(orbitpycov, stkcov, 4)
        self.get_metrics(orbitpycov_poly, stkcov, 4)

    def test_5(self):
        """Test an inclined orbit on a US grid with a 20 degree diameter conical sensor."""

        # Create coverage calculator
        cov = orbitpy.coveragecalculator.CoverageFactory.from_dict({
            "coverage_type": orbitpy.coveragecalculator.CoverageType.POINT_COVERAGE.to_string()})

        # Get file paths
        state_path = os.path.join(self.states_dir, "Satellite4_states.txt")
        accesses_path = os.path.join(self.accesses_dir, "US_Grid_5.cvaa")
        grid_path = os.path.join(self.accesses_dir, "US_grid")

        # Read trajectory from data file
        result = create_stateseries_from_txt_file(state_path)
        times = result.time

        # Create frame graph and add LVLH frame
        registry = self.get_registry(result)

        # Create a circular field of view
        diameter = 20.0  # deg
        fov = CircularFieldOfView(diameter=diameter, frame=self.lvlh_frame)

        # Calculate point coverage
        target_point_array = create_cartesian_position_array_from_csv(grid_path)
        orbitpycov = cov.calculate_coverage(
            target_point_array, fov=fov, frame_graph=registry, times=times)
        stkcov = ContinuousCoverageGP.from_stk(
            accesses_path, target_point_array).to_discrete(times[0], 1.0, len(times))

        if self.plot_tests:
            plot_results(orbitpycov, stkcov, target_point_array)

        self.get_metrics(orbitpycov, stkcov, 5)

    def test_6(self):
        """Test an inclined orbit on a US grid with a 30 deg AT, 20 deg CT sensor."""

        # Create coverage calculator
        cov = orbitpy.coveragecalculator.CoverageFactory.from_dict({
            "coverage_type": orbitpy.coveragecalculator.CoverageType.POINT_COVERAGE.to_string()})

        # Get file paths
        state_path = os.path.join(self.states_dir, "Satellite4_states.txt")
        accesses_path = os.path.join(self.accesses_dir, "US_Grid_6.cvaa")
        grid_path = os.path.join(self.accesses_dir, "US_grid")

        # Read trajectory from data file
        result = create_stateseries_from_txt_file(state_path)
        times = result.time

        # Create frame graph and add LVLH frame
        registry = self.get_registry(result)

        # Create a rectangular field of view
        up_half_angle = 15.0  # deg
        right_half_angle = 10.0  # deg
        fov = RectangularFieldOfView(frame=self.lvlh_frame, ref_angle=up_half_angle,
                                     cross_angle=right_half_angle)

        # Calculate point coverage
        target_point_array = create_cartesian_position_array_from_csv(grid_path)
        orbitpycov = cov.calculate_coverage(
            target_point_array, fov=fov, frame_graph=registry, times=times)
        stkcov = ContinuousCoverageGP.from_stk(
            accesses_path, target_point_array).to_discrete(times[0], 1.0, len(times))
        
        fov = PolygonFieldOfView.from_rectangular(fov)
        orbitpycov_poly = cov.calculate_coverage(
            target_point_array, fov=fov, frame_graph=registry, times=times)

        if self.plot_tests:
            plot_results(orbitpycov, stkcov, target_point_array)

        self.get_metrics(orbitpycov, stkcov, 6)
        self.get_metrics(orbitpycov_poly, stkcov, 6)

    def test_7(self):
        """Test a sun-sync orbit on an equatorial grid with a 20 degree diameter conical sensor."""

        # Create coverage calculator
        cov = orbitpy.coveragecalculator.CoverageFactory.from_dict({
            "coverage_type": orbitpy.coveragecalculator.CoverageType.POINT_COVERAGE.to_string()})

        # Get file paths
        state_path = os.path.join(self.states_dir, "Satellite5_states.txt")
        accesses_path = os.path.join(self.accesses_dir, "Equatorial_Grid_7.cvaa")
        grid_path = os.path.join(self.accesses_dir, "Equatorial_Grid")

        # Read trajectory from data file
        result = create_stateseries_from_txt_file(state_path)
        times = result.time

        # Create frame graph and add LVLH frame
        registry = self.get_registry(result)

        # Create a circular field of view
        diameter = 20.0  # deg
        fov = CircularFieldOfView(diameter=diameter, frame=self.lvlh_frame)

        # Calculate point coverage
        target_point_array = create_cartesian_position_array_from_csv(grid_path)
        circ_coverage = cov.calculate_coverage(
            target_point_array, fov=fov, frame_graph=registry, times=times)
        stkcov = ContinuousCoverageGP.from_stk(
            accesses_path, target_point_array).to_discrete(times[0], 1.0, len(times))

        if self.plot_tests:
            plot_results(circ_coverage, stkcov, target_point_array)

        self.get_metrics(circ_coverage, stkcov, 7)

    def test_8(self):
        """Test a sun-sync orbit on an equatorial grid with a 20 degree diameter pointed conical
        sensor."""

        # Create coverage calculator
        cov = orbitpy.coveragecalculator.CoverageFactory.from_dict({
            "coverage_type": orbitpy.coveragecalculator.CoverageType.POINT_COVERAGE.to_string()})

        # Get file paths
        state_path = os.path.join(self.states_dir, "Satellite5_states.txt")
        accesses_path = os.path.join(self.accesses_dir, "Equatorial_Grid_8.cvaa")
        grid_path = os.path.join(self.accesses_dir, "Equatorial_Grid")

        # Read trajectory from data file
        result = create_stateseries_from_txt_file(state_path)
        times = result.time

        # Create frame graph and add LVLH frame
        registry = self.get_registry(result)

        # Create orientation
        deg2rad = np.pi / 180.0
        beta = 25*deg2rad
        alpha = -30*deg2rad
        gamma = 5*deg2rad

        sensor_orientation = Orientation.from_dict({
            "orientation_type": "constant",
            "rotations_type": "euler",
            "from": "Sensor",
            "to": "LVLH",
            "euler_order": "XYZ",
            "rotations": [alpha,beta,gamma]
        })

        # Create a circular field of view
        diameter = 20.0  # deg
        fov = CircularFieldOfView(diameter=diameter, frame=self.sensor_frame)

        registry.add_orientation_transform(sensor_orientation)
        registry.add_pos_transform(
            ReferenceFrame.get("LVLH"),
            ReferenceFrame.get("Sensor"),
            Cartesian3DPosition(0.0, 0.0, 0.0, ReferenceFrame.get("LVLH")),
            True,
        )

        # Calculate point coverage
        target_point_array = create_cartesian_position_array_from_csv(grid_path)
        orbitpycov = cov.calculate_coverage(
            target_point_array, fov=fov, frame_graph=registry, times=times)
        stkcov = ContinuousCoverageGP.from_stk(
            accesses_path, target_point_array).to_discrete(times[0], 1.0, len(times))

        if self.plot_tests:
            plot_results(orbitpycov, stkcov, target_point_array)

        self.get_metrics(orbitpycov, stkcov, 8)

    def test_9(self):
        """Test a sun-sync orbit on an equatorial grid with a with a 20 deg AT, 30 deg CT pointed
        sensor."""

        # Create coverage calculator
        cov = orbitpy.coveragecalculator.CoverageFactory.from_dict({
            "coverage_type": orbitpy.coveragecalculator.CoverageType.POINT_COVERAGE.to_string()})

        # Get file paths
        state_path = os.path.join(self.states_dir, "Satellite5_states.txt")
        accesses_path = os.path.join(self.accesses_dir, "Equatorial_Grid_9.cvaa")
        grid_path = os.path.join(self.accesses_dir, "Equatorial_Grid")

        # Read trajectory from data file
        result = create_stateseries_from_txt_file(state_path)
        times = result.time

        # Create frame graph and add LVLH frame
        registry = self.get_registry(result)

        # Create orientation
        deg2rad = np.pi / 180.0
        beta = -24.0*deg2rad
        alpha = 30.0*deg2rad
        gamma = -6.0*deg2rad

        sensor_orientation = Orientation.from_dict({
            "orientation_type": "constant",
            "rotations_type": "euler",
            "from": "Sensor",
            "to": "LVLH",
            "euler_order": "XYZ",
            "rotations": [alpha,beta,gamma]
        })

        # Create a rectangular field of view
        up_half_angle = 10.0  # deg
        right_half_angle = 15.0  # deg
        fov = RectangularFieldOfView(frame=self.sensor_frame,ref_angle=up_half_angle, 
        cross_angle=right_half_angle)

        registry.add_orientation_transform(sensor_orientation)
        registry.add_pos_transform(
            ReferenceFrame.get("LVLH"),
            ReferenceFrame.get("Sensor"),
            Cartesian3DPosition(0.0, 0.0, 0.0, ReferenceFrame.get("LVLH")),
            True,
        )

        fov = PolygonFieldOfView.from_rectangular(fov)

        # Calculate point coverage
        target_point_array = create_cartesian_position_array_from_csv(grid_path)
        orbitpycov = cov.calculate_coverage(
            target_point_array, fov=fov, frame_graph=registry, times=times)
        stkcov = ContinuousCoverageGP.from_stk(
            accesses_path, target_point_array).to_discrete(times[0], 1.0, len(times))
        
        orbitpycov_poly = cov.calculate_coverage(
            target_point_array, fov=fov, frame_graph=registry, times=times)

        if self.plot_tests:
            plot_results(orbitpycov, stkcov, target_point_array)

        self.get_metrics(orbitpycov, stkcov, 9)
        self.get_metrics(orbitpycov_poly, stkcov, 9)

    def test_10(self):
        """Test a sun-sync orbit on a US grid with a 30 deg AT, 20 deg CT sensor."""

        # Create coverage calculator
        cov = orbitpy.coveragecalculator.CoverageFactory.from_dict({
            "coverage_type": orbitpy.coveragecalculator.CoverageType.POINT_COVERAGE.to_string()})

        # Get file paths
        state_path = os.path.join(self.states_dir, "Satellite5_states.txt")
        accesses_path = os.path.join(self.accesses_dir, "US_Grid_10.cvaa")
        grid_path = os.path.join(self.accesses_dir, "US_grid")

        # Read trajectory from data file
        result = create_stateseries_from_txt_file(state_path)
        times = result.time

        # Create frame graph and add LVLH frame
        registry = self.get_registry(result)

        # Create a rectangular field of view
        up_half_angle = 15.0  # deg
        right_half_angle = 10.0  # deg
        fov = RectangularFieldOfView(frame=self.lvlh_frame, ref_angle=up_half_angle,
                                     cross_angle=right_half_angle)

        # Calculate point coverage
        target_point_array = create_cartesian_position_array_from_csv(grid_path)
        orbitpycov = cov.calculate_coverage(
            target_point_array, fov=fov, frame_graph=registry, times=times)
        stkcov = ContinuousCoverageGP.from_stk(
            accesses_path, target_point_array).to_discrete(times[0], 1.0, len(times))
        
        fov = PolygonFieldOfView.from_rectangular(fov)
        orbitpycov_poly = cov.calculate_coverage(
            target_point_array, fov=fov, frame_graph=registry, times=times)

        if self.plot_tests:
            plot_results(orbitpycov, stkcov, target_point_array)

        self.get_metrics(orbitpycov, stkcov, 10)
        self.get_metrics(orbitpycov_poly, stkcov, 10)

    def test_11(self):
        """Test a sun-sync orbit on a US grid with a 20 deg AT, 30 deg CT pointed sensor."""

        # Create coverage calculator
        cov = orbitpy.coveragecalculator.CoverageFactory.from_dict({
            "coverage_type": orbitpy.coveragecalculator.CoverageType.POINT_COVERAGE.to_string()})

        # Get file paths
        state_path = os.path.join(self.states_dir, "Satellite5_states.txt")
        accesses_path = os.path.join(self.accesses_dir, "US_Grid_11.cvaa")
        grid_path = os.path.join(self.accesses_dir, "US_grid")

        # Read trajectory from data file
        result = create_stateseries_from_txt_file(state_path)
        times = result.time

        # Create frame graph and add LVLH frame
        registry = self.get_registry(result)

        # Create orientation
        deg2rad = np.pi / 180.0
        beta = 25*deg2rad
        alpha = -30*deg2rad
        gamma = 5*deg2rad

        sensor_orientation = Orientation.from_dict({
            "orientation_type": "constant",
            "rotations_type": "euler",
            "from": "Sensor",
            "to": "LVLH",
            "euler_order": "XYZ",
            "rotations": [alpha,beta,gamma]
        })

        # Create a rectangular field of view
        up_half_angle = 10.0  # deg
        right_half_angle = 15.0  # deg
        fov = RectangularFieldOfView(frame=self.sensor_frame,ref_angle=up_half_angle, 
        cross_angle=right_half_angle)
        
        registry.add_orientation_transform(sensor_orientation)
        registry.add_pos_transform(
            ReferenceFrame.get("LVLH"),
            ReferenceFrame.get("Sensor"),
            Cartesian3DPosition(0.0, 0.0, 0.0, ReferenceFrame.get("LVLH")),
            True,
        )

        # Calculate point coverage
        target_point_array = create_cartesian_position_array_from_csv(grid_path)
        orbitpycov = cov.calculate_coverage(
            target_point_array, fov=fov, frame_graph=registry, times=times)
        stkcov = ContinuousCoverageGP.from_stk(
            accesses_path, target_point_array).to_discrete(times[0], 1.0, len(times))
        
        fov = PolygonFieldOfView.from_rectangular(fov)
        orbitpycov_poly = cov.calculate_coverage(
            target_point_array, fov=fov, frame_graph=registry, times=times)

        if self.plot_tests:
            plot_results(orbitpycov, stkcov, target_point_array)

        self.get_metrics(orbitpycov, stkcov, 11)
        self.get_metrics(orbitpycov_poly, stkcov, 11)

    def test_12(self):
        """Test a sun-sync orbit on a US grid with a 30 deg AT, 20 deg CT pointed sensor."""

        # Create coverage calculator
        cov = orbitpy.coveragecalculator.CoverageFactory.from_dict({
            "coverage_type": orbitpy.coveragecalculator.CoverageType.POINT_COVERAGE.to_string()})

        # Get file paths
        state_path = os.path.join(self.states_dir, "Satellite5_states.txt")
        accesses_path = os.path.join(self.accesses_dir, "US_Grid_12.cvaa")
        grid_path = os.path.join(self.accesses_dir, "US_grid")

        # Read trajectory from data file
        result = create_stateseries_from_txt_file(state_path)
        times = result.time

        # Create frame graph and add LVLH frame
        registry = self.get_registry(result)

        # Create orientation
        deg2rad = np.pi / 180.0
        beta = -24.0*deg2rad
        alpha = 30.0*deg2rad
        gamma = -6.0*deg2rad

        sensor_orientation = Orientation.from_dict({
            "orientation_type": "constant",
            "rotations_type": "euler",
            "from": "Sensor",
            "to": "LVLH",
            "euler_order": "XYZ",
            "rotations": [alpha,beta,gamma]
        })

        # Create a rectangular field of view
        up_half_angle = 15.0  # deg
        right_half_angle = 10.0  # deg
        fov = RectangularFieldOfView(frame=self.sensor_frame,ref_angle=up_half_angle, 
        cross_angle=right_half_angle)
        
        registry.add_orientation_transform(sensor_orientation)
        registry.add_pos_transform(
            ReferenceFrame.get("LVLH"),
            ReferenceFrame.get("Sensor"),
            Cartesian3DPosition(0.0, 0.0, 0.0, ReferenceFrame.get("LVLH")),
            True,
        )

        # Calculate point coverage
        target_point_array = create_cartesian_position_array_from_csv(grid_path)
        orbitpycov = cov.calculate_coverage(
            target_point_array, fov=fov, frame_graph=registry, times=times)
        stkcov = ContinuousCoverageGP.from_stk(
            accesses_path, target_point_array).to_discrete(times[0], 1.0, len(times))
        
        fov = PolygonFieldOfView.from_rectangular(fov)
        orbitpycov_poly = cov.calculate_coverage(
            target_point_array, fov=fov, frame_graph=registry, times=times)

        if self.plot_tests:
            plot_results(orbitpycov, stkcov, target_point_array)

        self.get_metrics(orbitpycov, stkcov, 12)
        self.get_metrics(orbitpycov_poly, stkcov, 12)

if __name__ == "__main__":
    unittest.main()