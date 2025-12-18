"""
.. module:: orbitpy.coveragecalculator
   :synopsis: Module providing classes and functions to handle coverage related calculations.

"""

from typing import Type, Dict, Any, Union, Callable, List, Tuple
import math
import numpy as np

from .coverage import DiscreteCoverageTP

from eosimutils.base import (
    EnumBase,
    SurfaceType,
    ReferenceFrame,
    WGS84_EARTH_EQUATORIAL_RADIUS,
    WGS84_EARTH_POLAR_RADIUS,
    SPHERICAL_EARTH_MEAN_RADIUS,
)
from eosimutils.state import Cartesian3DPositionArray
from eosimutils.fieldofview import (
    CircularFieldOfView,
    RectangularFieldOfView,
    PolygonFieldOfView,
    OmnidirectionalFieldOfView,
    cone_footprint_area
)
from eosimutils.framegraph import FrameGraph
from eosimutils.time import AbsoluteDateArray

import kcl
import GeometricTools as gte


class CoverageType(EnumBase):
    """Enumeration of supported coverage types.

    Attributes:
        POINT_COVERAGE (str): Point coverage type.
    """

    POINT_COVERAGE = "POINT_COVERAGE"
    SPECULAR_COVERAGE = "SPECULAR_COVERAGE"


class CoverageFactory:
    """Factory class to register and invoke the appropriate coverage calculator class.

    This class allows registering coverage calculator classes and retrieving instances
    of the appropriate coverage calculator based on specifications.

    Example:
        class CustomCoverage:
            @classmethod
            def from_dict(cls, specs):
                return cls()
        CoverageFactory.register_type("CustomCoverage")(CustomCoverage)
        specs = {"coverage_type": "CustomCoverage", ...}
        coverage_calc = CoverageFactory.from_dict(specs)

    Attributes:
        _registry (Dict[str, Type]): A dictionary mapping coverage type
                                     labels to their respective classes.
    """

    # Registry for factory pattern
    _registry: Dict[str, Type] = {}

    @classmethod
    def register_type(cls, type_name: str) -> Callable[[Type], Type]:
        """
        Decorator to register a coverage calculator class under a type name.
        """

        def decorator(coverage_class: Type) -> Type:
            cls._registry[type_name] = coverage_class
            return coverage_class

        return decorator

    @classmethod
    def from_dict(cls, specs: Dict[str, Any]) -> object:
        """Retrieves an instance of the appropriate coverage calculator based on type key.

        Args:
            specs (Dict[str, Any]): A dictionary containing coverage calculator specifications.
                Must include a valid coverage calculator type in the "coverage_type" key.

        Returns:
            object: An instance of the appropriate coverage calculator class initialized
                 with the given specifications.

        Raises:
            KeyError: If the "coverage_type" key is missing in the dictionary.
            ValueError: If the specified coverage calculator type is not registered.
        """
        coverage_type_str = specs.get("coverage_type")
        if coverage_type_str is None:
            raise KeyError(
                'Coverage calculator type key "coverage_type" not found in'
                "specifications dictionary."
            )
        coverage_class = cls._registry.get(coverage_type_str)
        if not coverage_class:
            raise ValueError(
                f'Coverage calculator type "{coverage_type_str}" is not registered.'
            )
        return coverage_class.from_dict(specs)


@CoverageFactory.register_type(CoverageType.SPECULAR_COVERAGE.to_string())
class SpecularCoverage:
    """A coverage calculator class which handles GNSSR coverage calculation for an array of target
    points."""

    @classmethod
    def from_dict(
        cls, specs: Dict[str, Any]  # pylint: disable=unused-argument
    ) -> "SpecularCoverage":
        # Empty since class does not require any initialization parameters
        return cls()

    def calculate_coverage(
        self,
        target_point_array: Cartesian3DPositionArray,
        fov: CircularFieldOfView,
        frame_graph: FrameGraph,
        times: AbsoluteDateArray,
        transmitters: List[OmnidirectionalFieldOfView],
        specular_radius: float,
        surface: SurfaceType = SurfaceType.SPHERE,
        buff_size=100,
    ) -> List[Tuple[DiscreteCoverageTP, List[float]]]:
        """Calculates the specular coverage over an array of target points given a field of view.

        Uses the coverage kinematics library (C++) for coverage calculation. The following
        concepts are relevant to understanding the implementation:

        Source<T>: An object which satisfies the Source interface defines a "get" method,
        which takes the simulation time index as input and returns a reference to an object
        of type T.

        Variable: An object which satisfies the Variable interface represents a quantity which
        must be updated over the course of the simulation. Variable objects define an "update"
        method, which takes the simulation time index as input and updates the object for that
        time step.

        Args:
            target_point_array (Cartesian3DPositionArray): An array of target points in a Cartesian
                coordinate system.
            fov (CircularFieldOfView): The field of view to use for
                coverage calculation.
            frame_graph (FrameGraph): The frame graph containing the necessary transformations
                between frames.
            times (AbsoluteDateArray): An array of time points for which the coverage is to be
                calculated at discrete instances.
            transmitters (List[OmnidirectionalFieldOfView]): A list of OmnidirectionalFieldOfView objects used to lookup
                trajectories of the GNSS transmitters to be used for specular coverage calculation.
                Transmitter trajectories must first be added to the frame graph for this to work.
            specular_radius (float): The radius of the specular reflection area ("glistening zone")
                around each target point. Corresponds to a 3D sphere around the specular point.
            surface (SurfaceType): The type of surface to consider for coverage calculation.
                Defaults to SurfaceType.SPHERE. Surface type will affect the horizon check, which
                checks that the points are not occluded by the surface. Points are not necessarily
                required to be on the surface. Surface is assumed to be fixed to the frame of the
                target points. This is also the surface type used to calculate the specular point.
            buff_size (int): Specifies the buffer size for parallel computation. For optimal
                performance, it should be at least equal to the number of CPU cores available
                on the executing machine.
        Returns:
            List[Tuple[DiscreteCoverageTP, List[float]]]: A list of tuples containing (1) the
                coverage for each GPS transmitter and (2) the radar's range-corrected
                gain (RCG), assuming unity gain, at the input time points.

        For reference on RCG formula, see https://ieeexplore.ieee.org/abstract/document/8370081.
        Formula for RCG is G/(R_t^2 * R_r^2), where R_t is the distance between the transmitter
        and the specular point, and R_r is the distance between the receiver and the specular
        point. Here G is assumed to be 1.0 (unity gain).

        Coverage is calculated by the following steps at each time step:

        1. Get the position/orientation of all relevant objects in the
        target frame (receiver, transmitters, target points)
        2. Check line-of-sight (LOS) between each transmitter and the receiver
        3. Calculate the specular point between each transmitter and the
        receiver, if LOS is available. Also calculate the RCG factor at the specular point.
        4. Create a sphere around the specular point with radius specular_radius.
        5. Check which target points are in the sphere, the receiver FOV, and have LOS
        to the receiver (i.e. are in the horizon). Points which satisfy all three conditions are
        considered covered.

        """

        # Store all coverage output in memory
        buff_size_full = len(times)

        fov_frame = fov.frame
        target_frame = target_point_array.frame

        # Get the position of the fov frame origin in the target frame
        # expressed in target frame coordinates
        pos_fov_target = frame_graph.get_pos_transform(
            from_frame=target_frame, to_frame=fov_frame, t=times
        )

        # Create list source for the (target frame) fov position
        pos_fov_target_gte = [gte.Vector3d(p) for p in pos_fov_target]
        pos_fov_target_source = kcl.ListSourceVector3d(pos_fov_target_gte)

        # Get the rotation from the fov frame to the target frame
        rot_fov_to_target, _ = frame_graph.get_orientation_transform(
            fov_frame, target_frame, times
        )

        # Get the transformation from fov to target frame as a Listsource
        # A Listsource stores the data in a list for each simulation time step
        # (it is not a variable and does not need to be added to the list of variables)
        fov_to_target_gte = [
            gte.Matrix3x3d(r.flatten()) for r in rot_fov_to_target.as_matrix()
        ]
        fov_to_target_source = kcl.ListSourceMatrix3x3d(fov_to_target_gte)

        # Setup horizon source. A point and an ellipsoid/sphere are sufficient to define a polar
        # plane, which divides the ellipsoid/sphere into a region visible from that point
        # and a region not visible (no LOS). The halfspace supported by that plane can be used
        # to check for visibility.
        if surface == SurfaceType.SPHERE:
            extents = gte.Vector3d(
                [
                    SPHERICAL_EARTH_MEAN_RADIUS,
                    SPHERICAL_EARTH_MEAN_RADIUS,
                    SPHERICAL_EARTH_MEAN_RADIUS,
                ]
            )
        elif surface == SurfaceType.WGS84:
            extents = gte.Vector3d(
                [
                    WGS84_EARTH_EQUATORIAL_RADIUS,
                    WGS84_EARTH_EQUATORIAL_RADIUS,
                    WGS84_EARTH_POLAR_RADIUS,
                ]
            )
        else:
            raise ValueError(f"Unsupported surface type: {surface}")

        earth_shape = gte.Ellipsoid3d(gte.Vector3d.Zero(), extents)

        # The ellipsoid source is a constant source, since the ellipsoid's
        # parameters remain the same for all time steps
        earth_source = kcl.ConstantSourceEllipsoid3d(earth_shape)

        # The horizon source is a Variable object, so it must be added to the
        # list of variables to be updated throughout the simulation.
        horizon_source = kcl.PolarHalfspaceSourceEllipsoid3d(
            earth_source, pos_fov_target_source, buff_size
        )
        variables = [horizon_source]

        horizon_viewer = kcl.ViewerHalfspace3d(horizon_source)

        if isinstance(fov, CircularFieldOfView):
            deg2rad = math.pi / 180.0
            half_angle_rad = fov.diameter * 0.5 * deg2rad
            boresight_fov_gte = gte.Vector3d(fov.boresight)

            # First, define the boresight vector and cone angle in the FOV frame as a constant
            # source.
            boresight_fov_source = kcl.ConstantSourceVector3d(boresight_fov_gte)
            angle_source = kcl.ConstantSourced(half_angle_rad)

            # Define a source which transforms the boresight from the FOV frame to the targe frame.
            # This is a Variable object so it must be added to the variables list to be updated.
            boresight_target_source = kcl.TransformedVector3dSource(
                boresight_fov_source, fov_to_target_source, buff_size
            )

            # Define a source which builds a cone object using the FOV position, boresight
            # unit vector, and cone half-angle.
            # This is a Variable object so it must be added to the variables list to be updated.
            fov_source = kcl.PosDirConeSource(
                pos_fov_target_source,
                boresight_target_source,
                angle_source,
                buff_size,
            )

            # Define a Viewer object for the cone shape. Viewer objects are used to make the shape
            # compatible with constructive solid geometry (CSG) operations.
            fov_viewer = kcl.ViewerCone3d(fov_source)

            # Add variables to list
            variables.append(boresight_target_source)
            variables.append(fov_source)
        else:
            raise ValueError("Unsupported field of view type.")

        # Get the position of the grid points in the target frame
        target_points_list_gte = [
            gte.Vector3d(row) for row in target_point_array.to_numpy()
        ]
        # List of target points stored as a constant-source in the target frame
        grid_source = kcl.ConstantSourceListVector3d(target_points_list_gte)

        # The output list of tuples
        output_list = []

        # A list of coverage sources: there will be one for each GPS transmitter
        cov_sources = []
        # A list of distance sources, which will output the gps-specular point-receiver distances.
        # There will be one for each GPS transmitter.
        rcg_sources = []

        # This list is used to store the source objects for each looop iteration, otherwise,
        # the pointers will be deleted after each loop iteration due to default Pybind11 behavior.
        # Pybind11 will create a unique pointer for dynamically created bound objects, so that when
        # the object goes out of scope in Python, it is deleted, even if it is still referenced in
        # C++.
        source_refs = []
        for transmitter in transmitters:

            tx_frame = transmitter.frame

            # Get the position of the tx frame origin in the target frame
            # expressed in target frame coordinates
            pos_tx_target = frame_graph.get_pos_transform(
                from_frame=target_frame, to_frame=tx_frame, t=times
            )

            # Create list source for the (target frame) tx position
            pos_tx_target_gte = [gte.Vector3d(p) for p in pos_tx_target]
            pos_tx_target_source = kcl.ListSourceVector3d(pos_tx_target_gte)

            # This will return true/false indicating whether there is LOS between the transmitter
            # and receiver satellite
            los_source = kcl.LOSEventSourceEllipsoid3d(
                pos_fov_target_source,
                pos_tx_target_source,
                earth_source,
                buff_size,
            )

            # Whenever LOS is available, the specular source will calculate the specular point
            specular_source = kcl.SpecularPointSource(
                pos_tx_target_source,
                pos_fov_target_source,
                earth_source,
                los_source,
                buff_size,
            )

            # Returns the range-corrected gain (RCG) factor for the radar.
            radar_gain = 1.0  # Placeholder value
            rcg_source = kcl.RCGSource(
                pos_tx_target_source,
                pos_fov_target_source,
                specular_source,
                radar_gain,
                buff_size_full,
            )

            radius_source = kcl.ConstantSourced(specular_radius)
            sphere_source = kcl.DefaultSphereSource(
                specular_source, radius_source, buff_size
            )

            variables.append(los_source)
            variables.append(specular_source)
            variables.append(rcg_source)
            variables.append(sphere_source)
            rcg_sources.append(rcg_source)

            specular_viewer = kcl.ViewerSphere3d(sphere_source)

            # Points are considered to be covered if they are in the horizon AND the FOV
            # AND the glistening zone
            total_view = kcl.Intersection(
                [specular_viewer, horizon_viewer, fov_viewer]
            )

            # Setup Coverage source (no preprocessor)
            preprocessor = None
            cov_source = kcl.CovSource(
                grid_source, preprocessor, total_view, buff_size_full
            )
            variables.append(cov_source)
            cov_sources.append(cov_source)

            source_refs.append(
                (
                    pos_tx_target_source,
                    los_source,
                    specular_source,
                    radius_source,
                    rcg_source,
                    sphere_source,
                    cov_source,
                    specular_viewer,
                    total_view,
                )
            )

        # --- Calculate coverage

        # Add variables list to the VarDriver class, which is responsible for updating them
        # during the simulation
        driver = kcl.VarDriver(variables)
        start_time = 0
        stop_time = len(times) - 1

        # Run the simulation
        kcl.driveCoverage(buff_size, start_time, stop_time, driver, [])

        # Prepare coverage output
        for idx in range(len(cov_sources)):
            coverage_kcl = [cov_sources[idx].get(i) for i in range(len(times))]
            rcg_factor = [rcg_sources[idx].get(i) for i in range(len(times))]
            coverage = DiscreteCoverageTP(
                times, coverage_kcl, target_point_array
            )
            output_list.append((coverage, rcg_factor))

        return output_list


@CoverageFactory.register_type(CoverageType.POINT_COVERAGE.to_string())
class PointCoverage:
    """A coverage calculator class which handles coverage calculation for an array of target
    points."""

    @classmethod
    def from_dict(
        cls, specs: Dict[str, Any]  # pylint: disable=unused-argument
    ) -> "PointCoverage":
        # Empty since class does not require any initialization parameters
        return cls()
    
    @staticmethod
    def scale_cbpa_cells(area: float, num_pts: int,
                           ref_area: float, ref_num_pts: int, ref_s: float) -> int:
        """Scales the number of CBPA cells using the scaling laws from the CBPA paper. 
        That is, assuming that ref_s is the optimal cell parameter found for a coverage simulation using a
        sensor whose footprint area is "ref_area" on the unit sphere and "ref_num_pts" target points,
        then the optimal number of cells for a sensor with footprint area "area" and "num_pts"
        target points is calculated using this function.

        Args:
            area (float): The area of the sensor footprint on the unit sphere in steradians.
            num_pts (int): The number of target points.
            ref_area (float): The area of the sensor footprint on the unit sphere
                in steradians for the reference case.
            ref_num_pts (int): The number of target points for the reference case.
            ref_s (int): The optimal cell parameter s for the reference case.

        Returns:
            int: The optimal number of CBPA cells.
        """
        
        # Cell grid parameter s, see paper for details
        s = ref_s*(num_pts/ref_num_pts)**(2/3)
        cbpa_cells = int(4.0 * np.pi * s / (area / ref_area))
        return cbpa_cells
    
    @staticmethod
    def compute_cbpa_cells(fov: Union[CircularFieldOfView, RectangularFieldOfView], distance: float, num_pts: int) -> int:
        """Computes an optimal number of CBPA cells based on the FOV area, fov vertex distance
         from surface of the Earth, and number of target points.

        Calculates an approximation of the sensor footprint area by assuming:
            (1) A conical field of view (if the field of view is rectangular, the maximum cone angle is used).
            (2) A spherical earth.
            (3) Nadir pointing.

        Assumes the given distance is representative of the distance over the course of the simulation. Ideally,
        provide an average distance for best results with all orbit types.
                    
        Args:
            fov (Union[CircularFieldOfView, RectangularFieldOfView]): The field of view to use for coverage calculation.
            distance (float): The distance of the sensor from the Earth's center.
            num_pts (int): The number of target points.

        Returns:
            int: The computed number of CBPA cells.
        """

        if (isinstance(fov, CircularFieldOfView)):
            radius = fov.diameter / 2.0
        elif (isinstance(fov, RectangularFieldOfView)):
            ref_angle = np.deg2rad(fov.ref_angle)
            cross_angle = np.deg2rad(fov.cross_angle)
            # Compute maximum half-cone angle for rectangular sensor
            radius = np.arctan(np.sqrt(np.tan(ref_angle)**2 + np.tan(cross_angle)**2))
            radius = np.rad2deg(radius)

        area = cone_footprint_area(
            theta=radius,
            d=distance,
            Re=SPHERICAL_EARTH_MEAN_RADIUS,
        )

        # Parameters from IEEE paper tuning
        # In the IEEE paper, it was found that for a circular FOV with
        # half-angle of 22.5 deg, orbital radius of 7080.48, a coverage
        # simulation with 100,000 points had optimal cell parameter s=20.
        radius_ref = 22.5
        distance_ref = 7080.48
        area_ref = cone_footprint_area(
            theta=radius_ref,
            d=distance_ref,
            Re=SPHERICAL_EARTH_MEAN_RADIUS,
        )
        ref_num_pts = 100000
        ref_s = 20
        
        return PointCoverage.scale_cbpa_cells(
            area, num_pts, area_ref, ref_num_pts, ref_s
        ) 

    def calculate_coverage(
        self,
        target_point_array: Cartesian3DPositionArray,
        fov: Union[CircularFieldOfView, RectangularFieldOfView, OmnidirectionalFieldOfView, PolygonFieldOfView],
        frame_graph: FrameGraph,
        times: AbsoluteDateArray,
        surface: SurfaceType = SurfaceType.SPHERE,
        use_cbpa: bool = False,
        cbpa_cells=None,
        buff_size=None,
    ) -> DiscreteCoverageTP:
        """Calculates the coverage over an array of target points given a field of view.

        Uses the coverage kinematics library (C++) for coverage calculation. The following
        concepts are relevant to understanding the implementation:

        Source<T>: An object which satisfies the Source interface defines a "get" method,
        which takes the simulation time index as input and returns a reference to an object
        of type T.

        Variable: An object which satisfies the Variable interface represents a quantity which
        must be updated over the course of the simulation. Variable objects define an "update"
        method, which takes the simulation time index as input and updates the object for that
        time step.

        Notes on CBPA: The cell-based preprocessing algorithm (CBPA) is used to speed up coverage
        calculation, especially for small-FOV sensors. It works by pre-loading all of the target 
        points into a spherical grid with equal latitude spacing and equal longitude spacing withing
        each latitude band. During each time step, the sensor profile is projected onto the sphere 
        and a tight spherical bounding box is created, which is used to quickly lookup the target
        points which fall in the box in sublinear time. The algorithm has been tested extensively
        and is exact for a spherical earth. On the ellipsoidal earth, the algorithm is run using
        a mean-radius approximation of the ellipsoid, which can lead to small errors which are
        corrected by adding a 25% increase in the box length in both dimensions. The 25% growth
        was selected using probabilistic unit tests as follows. (1) Generate 10 million random
        points on WGS-84; (2) Generate a satellite at a random direction and altitude between
        100 and 35000 km; (3) Generate random pointing angles between 0.001 and pi-0.001; (4)
        Generate random FOV angles between 0.001 and pi-0.001; (5) Check that every point
        inside the FOV is inside the bounding box.

        CBPA is described in detail in the publication https://doi.org/10.1109/AERO58975.2024.10521431

        Args:
            target_point_array (Cartesian3DPositionArray): An array of target points in a Cartesian
                coordinate system.
            fov (Union[CircularFieldOfView, RectangularFieldOfView, OmnidirectionalFieldOfView,
                PolygonFieldOfView]): The field of view to use for coverage calculation.
            frame_graph (FrameGraph): The frame graph containing the necessary transformations
                between frames.
            times (AbsoluteDateArray): An array of time points for which the coverage is to be
                calculated at discrete instances.
            surface (SurfaceType): The type of surface to consider for coverage calculation.
                Defaults to SurfaceType.SPHERE. Surface type will affect the horizon check, which
                checks that the points aren't occluded by the surface. Points in target_point_array
                are not necessarily required to be on the surface. Surface is assumed to be fixed
                to the frame of the target points.
            use_cbpa (bool): If True, use the cell-based preprocessing algorithm (CBPA).
            cbpa_cells (int): Only supported for circular and rectangular FOVs. If set,
                the program will use CBPA with the number of cells in the grid approximately
                equal to the specified value. If this value is not set, a default optimal cell
                count will be calculated based on the number of target points and the FOV 
                footprint area.
            buff_size (int): Specifies the buffer size for parallel computation. For optimal
                performance, it should be at least equal to the number of CPU cores available
                on the executing machine. Defaults to len(times).
        Returns:
            DiscreteCoverageTP: An object reporting the grid points covered at each time point.
        """

        if buff_size is None:
            buff_size = len(times)

        fov_frame = fov.frame
        target_frame = target_point_array.frame

        # Get the position of the fov frame origin in the target frame
        # expressed in target frame coordinates
        pos_fov_target = frame_graph.get_pos_transform(
            from_frame=target_frame, to_frame=fov_frame, t=times
        )

        # Create list source for the (target frame) fov position
        pos_fov_target_gte = [gte.Vector3d(p) for p in pos_fov_target]
        pos_fov_target_source = kcl.ListSourceVector3d(pos_fov_target_gte)

        # Get the rotation from the fov frame to the target frame
        rot_fov_to_target, _ = frame_graph.get_orientation_transform(
            fov_frame, target_frame, times
        )

        # Get the transformation from fov to target frame as a Listsource
        # A Listsource stores the data in a list for each simulation time step
        # (it is not a variable and does not need to be added to the list of variables)
        fov_to_target_gte = [
            gte.Matrix3x3d(r.flatten()) for r in rot_fov_to_target.as_matrix()
        ]
        fov_to_target_source = kcl.ListSourceMatrix3x3d(fov_to_target_gte)

        # If number of cells is not provided
        # calculate it optimially using the formulas from IEEE Aero paper.
        if use_cbpa and cbpa_cells is None:
            # Average distance of satellite from Earth's center
            distance = np.linalg.norm(pos_fov_target, axis=1).mean()
            cbpa_cells = PointCoverage.compute_cbpa_cells(fov, distance, len(target_point_array))

        sphere_source = None
        # Setup horizon source. A point and a ellipsoid/sphere are sufficient to define a polar
        # plane, which divides the ellipsoid/sphere into a region visible from that point
        # and a region not visible (no LOS). The halfspace supported by that plane can be used
        # to check for visibility.
        if surface == SurfaceType.SPHERE:
            earth_sphere = gte.Sphere3d(
                gte.Vector3d.Zero(), WGS84_EARTH_EQUATORIAL_RADIUS
            )
            # The sphere source is a constant source, since the sphere's
            # parameters remain the same for all time steps
            sphere_source = kcl.ConstantSourceSphere3d(earth_sphere)

            # The horizon source is a Variable object, so it must be added to the
            # list of variables to be updated throughout the simulation.
            horizon_source = kcl.PolarHalfspaceSourceSphere3d(
                sphere_source, pos_fov_target_source, buff_size
            )
            variables = [horizon_source]
        elif surface == SurfaceType.WGS84:
            extents = gte.Vector3d(
                [
                    WGS84_EARTH_EQUATORIAL_RADIUS,
                    WGS84_EARTH_EQUATORIAL_RADIUS,
                    WGS84_EARTH_POLAR_RADIUS,
                ]
            )
            earth_ellipsoid = gte.Ellipsoid3d(gte.Vector3d.Zero(), extents)

            # The ellipsoid source is a constant source, since the ellipsoid's
            # parameters remain the same for all time steps
            ellipsoid_source = kcl.ConstantSourceEllipsoid3d(earth_ellipsoid)

            # The horizon source is a Variable object, so it must be added to the
            # list of variables to be updated throughout the simulation.
            horizon_source = kcl.PolarHalfspaceSourceEllipsoid3d(
                ellipsoid_source, pos_fov_target_source, buff_size
            )
            variables = [horizon_source]
        elif surface == SurfaceType.NONE:
            variables = []
        else:
            raise ValueError(f"Unsupported surface type: {surface}")

        box_source = None
        if isinstance(fov, CircularFieldOfView):
            deg2rad = math.pi / 180.0
            half_angle_rad = fov.diameter * 0.5 * deg2rad
            boresight_fov_gte = gte.Vector3d(fov.boresight)

            # First, define the boresight vector and cone angle in the FOV frame as a constant
            # source.
            boresight_fov_source = kcl.ConstantSourceVector3d(boresight_fov_gte)
            angle_source = kcl.ConstantSourced(half_angle_rad)

            # Define a source which transforms the boresight from the FOV frame to the targe frame.
            # This is a Variable object so it must be added to the variables list to be updated.
            boresight_target_source = kcl.TransformedVector3dSource(
                boresight_fov_source, fov_to_target_source, buff_size
            )

            # Define a source which builds a cone object using the FOV position, boresight
            # unit vector, and cone half-angle.
            # This is a Variable object so it must be added to the variables list to be updated.
            fov_source = kcl.PosDirConeSource(
                pos_fov_target_source,
                boresight_target_source,
                angle_source,
                buff_size,
            )

            # Define a Viewer object for the cone shape. Viewer objects are used to make the shape
            # compatible with constructive solid geometry (CSG) operations.
            fov_viewer = kcl.ViewerCone3d(fov_source)

            # Add variables to list
            variables.append(boresight_target_source)
            variables.append(fov_source)
            if use_cbpa:
                
                earth_sphere_cbpa = gte.Sphere3d(
                gte.Vector3d.Zero(), SPHERICAL_EARTH_MEAN_RADIUS)
                # The sphere source is a constant source, since the sphere's
                # parameters remain the same for all time steps
                sphere_source_cbpa = kcl.ConstantSourceSphere3d(earth_sphere_cbpa)

                box_source = kcl.Cone3dSourceAlignedBoxS2d(
                    fov_source, sphere_source_cbpa, buff_size
                )

                if (surface == SurfaceType.WGS84):
                    # Increase box size by 25% to account for ellipsoidal earth approximation
                    box_source.setFactor(.25)

                variables.append(box_source)

        elif isinstance(fov, RectangularFieldOfView):
            deg2rad = math.pi / 180.0
            up_angle_rad = (
                fov.ref_angle * 2.0 * deg2rad
            )  # convert half angle to full to match kcl
            right_angle_rad = (
                fov.cross_angle * 2.0 * deg2rad
            )  # convert half angle to full to match kcl
            right_fov = np.cross(fov.boresight, fov.ref_vector)
            right_fov_gte = gte.Vector3d(right_fov)
            up_fov_gte = gte.Vector3d(fov.ref_vector)

            # First, define the "up" and "right" vectors in the FOV frame as constant sources.
            # These orthogonal vectors define the first and second basis vectors for the FOV image
            # plane.
            up_fov_source = kcl.ConstantSourceVector3d(up_fov_gte)
            right_fov_source = kcl.ConstantSourceVector3d(right_fov_gte)

            # Define sources which transform up and right vectors from fov frame to target frame
            # These are Variable objects so they must be added to the variables list to be updated.
            up_target_source = kcl.TransformedVector3dSource(
                up_fov_source, fov_to_target_source, buff_size
            )
            right_target_source = kcl.TransformedVector3dSource(
                right_fov_source, fov_to_target_source, buff_size
            )

            # Define constant sources for the FOV angles about the up and right vectors
            right_angle_source = kcl.ConstantSourced(right_angle_rad)
            up_angle_source = kcl.ConstantSourced(up_angle_rad)

            # Define a source which builds a rectangle object using the FOV position, up and right
            # unit vectors, and FOV angles.
            # This is a Variable object so it must be added to the variables list to be updated.
            fov_source = kcl.VectorAngleRectViewSource(
                pos_fov_target_source,
                up_target_source,
                right_target_source,
                up_angle_source,
                right_angle_source,
                buff_size,
            )

            # Define a Viewer object for the FOV shape. Viewer objects are used to make the shape
            # compatible with constructive solid geometry (CSG) operations.
            fov_viewer = kcl.ViewerRectView3d(fov_source)

            # Add variables to list
            variables.append(up_target_source)
            variables.append(right_target_source)
            variables.append(fov_source)
            if use_cbpa:

                earth_sphere_cbpa = gte.Sphere3d(
                gte.Vector3d.Zero(), SPHERICAL_EARTH_MEAN_RADIUS)
                # The sphere source is a constant source, since the sphere's
                # parameters remain the same for all time steps
                sphere_source_cbpa = kcl.ConstantSourceSphere3d(earth_sphere_cbpa)

                box_source = kcl.RectView3dSourceAlignedBoxS2d(
                    fov_source, sphere_source_cbpa, buff_size
                )

                if (surface == SurfaceType.WGS84):
                    # Increase box size by 25% to account for ellipsoidal earth approximation
                    box_source.setFactor(.25)

                variables.append(box_source)

        elif isinstance(fov, PolygonFieldOfView):

            # Define the polygon vertices and interior point in the FOV frame
            vertices_fov = [gte.Vector3d(v) for v in fov.boundary_corners]
            interior_fov = gte.Vector3d(fov.boresight)

            # Define a source which builds a spherical polygon object using the FOV position,
            # polygon vertices and interior point.
            # This is a Variable object so it must be added to the variables list to be updated.
            fov_source = kcl.SphericalPolySource(
                vertices_fov, interior_fov, pos_fov_target_source,fov_to_target_source, buff_size
            )

            # Define a Viewer object for the polygon shape. Viewer objects are used to make the shape
            # compatible with constructive solid geometry (CSG) operations.
            fov_viewer = kcl.ViewerSphericalPolygond(fov_source)

            # Add variables to list
            variables.append(fov_source)

            if (use_cbpa):
                raise ValueError(
                    "CBPA preprocessor is only supported for circular or rectangular FOVs."
                )

        elif isinstance(fov, OmnidirectionalFieldOfView):
            fov_viewer = None
        else:
            raise ValueError("Unsupported field of view type.")

        # Setup Viewers. Viewers define a function which determines whether a given point is
        # inside the viewer. Complex shapes can be constructed through CSG operations
        # (union, intersection, compliment)
        horizon_viewer = kcl.ViewerHalfspace3d(horizon_source)
        # Points are considered to be covered if they are in the horizon AND the FOV
        # If there is no FOV, only check horizon
        if fov_viewer is None:
            total_view = horizon_viewer
        else:
            total_view = kcl.Intersection([horizon_viewer, fov_viewer])

        # Get the position of the grid points in the target frame
        target_points_list_gte = [
            gte.Vector3d(row) for row in target_point_array.to_numpy()
        ]
        # List of target points stored as a constant-source in the target frame
        grid_source = kcl.ConstantSourceListVector3d(target_points_list_gte)

        # Setup Coverage source (no preprocessor)
        # Store all coverage output in memory
        buff_size_coverage = len(times)

        preprocessor = None
        if use_cbpa:
            if box_source is None:
                raise ValueError(
                    "CBPA preprocessor is only supported for circular or rectangular FOVs."
                )
            cellgrid = kcl.CellGrid(cbpa_cells)
            cellgrid.AddPoints(target_points_list_gte)
            preprocessor = kcl.RangeCovSource(cellgrid, box_source, buff_size)
            variables.append(preprocessor)

        cov_source = kcl.CovSource(
            grid_source, preprocessor, total_view, buff_size_coverage
        )
        variables.append(cov_source)

        # Add variables list to the VarDriver class, which is responsible for updating them
        # during the simulation
        driver = kcl.VarDriver(variables)
        start_time = 0
        stop_time = len(times) - 1

        # Run the simulation
        kcl.driveCoverage(buff_size, start_time, stop_time, driver, [])

        # Prepare coverage output
        coverage = [cov_source.get(i) for i in range(len(times))]

        return DiscreteCoverageTP(times, coverage, target_point_array)
