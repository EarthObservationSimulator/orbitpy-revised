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
    def from_dict(cls, specs: Dict[str, Any]) -> "SpecularCoverage":
        # Empty since class does not require any initialization parameters
        return cls()

    def calculate_coverage(
        self,
        target_point_array: Cartesian3DPositionArray,
        fov: CircularFieldOfView,
        frame_graph: FrameGraph,
        times: AbsoluteDateArray,
        transmitters: List[ReferenceFrame],
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
            transmitters (List[ReferenceFrame]): A list of ReferenceFrame objects used to lookup
                trajectories of the GNSS transmitters to be used for specular coverage calculation.
                Transmitter trajectories must first be added to the frame graph for this to work.
            specular_radius (float): The radius of the specular reflection area ("glistening zone")
                around each target point.
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
                gain (RCG) at the input time points.

        For reference on RCG calculation, see https://ieeexplore.ieee.org/abstract/document/8370081
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
        for tx_frame in transmitters:

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

            # Returns the range-corrected gain (RCG) for the radar.
            radar_gain = 1.0  # Placeholder value
            rcg_source = kcl.RCGSource(
                pos_tx_target_source,
                pos_fov_target_source,
                specular_source,
                radar_gain,
                buff_size
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
            distance = [rcg_sources[idx].get(i) for i in range(len(times))]
            coverage = DiscreteCoverageTP(
                times, coverage_kcl, target_point_array
            )
            output_list.append((coverage, distance))

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

    def calculate_coverage(
        self,
        target_point_array: Cartesian3DPositionArray,
        fov: Union[CircularFieldOfView, RectangularFieldOfView],
        frame_graph: FrameGraph,
        times: AbsoluteDateArray,
        surface: SurfaceType = SurfaceType.SPHERE,
        buff_size=86401,
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

        Args:
            target_point_array (Cartesian3DPositionArray): An array of target points in a Cartesian
                coordinate system.
            fov (Union[CircularFieldOfView, RectangularFieldOfView]): The field of view to use for
                coverage calculation.
            frame_graph (FrameGraph): The frame graph containing the necessary transformations
                between frames.
            times (AbsoluteDateArray): An array of time points for which the coverage is to be
                calculated at discrete instances.
            surface (SurfaceType): The type of surface to consider for coverage calculation.
                Defaults to SurfaceType.SPHERE. Surface type will affect the horizon check, which
                checks that the points are not occluded by the surface. Points are not necessarily
                required to be on the surface. Surface is assumed to be fixed to the frame of the
                target points.
            buff_size (int): Specifies the buffer size for parallel computation. For optimal
                performance, it should be at least equal to the number of CPU cores available
                on the executing machine.
        Returns:
            DiscreteCoverageTP: An object reporting the grid points covered at each time point.
        """

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

        elif isinstance(fov, PolygonFieldOfView):
            raise NotImplementedError(
                "PolygonFieldOfView is not yet implemented."
            )
        else:
            raise ValueError("Unsupported field of view type.")

        # Setup Viewers. Viewers define a function which determines whether a given point is
        # inside the viewer. Complex shapes can be constructed through CSG operations
        # (union, intersection, compliment)
        horizon_viewer = kcl.ViewerHalfspace3d(horizon_source)
        # Points are considered to be covered if they are in the horizon AND the FOV
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
