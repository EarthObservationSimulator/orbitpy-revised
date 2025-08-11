"""
.. module:: orbitpy.coveragecalculator
   :synopsis: Module providing classes and functions to handle coverage related calculations.

"""

from typing import Type, Dict, Any, Union, Optional, Callable
import math
import numpy as np

from .coverage import DiscreteCoverageTP

from eosimutils.base import EnumBase, SurfaceType, EARTH_RADIUS, EARTH_POLAR_RADIUS
from eosimutils.state import Cartesian3DPosition, Cartesian3DPositionArray
from eosimutils.fieldofview import CircularFieldOfView, RectangularFieldOfView, PolygonFieldOfView
from eosimutils.framegraph import FrameGraph
from eosimutils.time import AbsoluteDate, AbsoluteDateArray

import kcl
import GeometricTools as gte

class CoverageType(EnumBase):
    """Enumeration of supported coverage types.

    Attributes:
        POINT_COVERAGE (str): Point coverage type.
    """
    POINT_COVERAGE = "POINT_COVERAGE"

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
            raise KeyError('Coverage calculator type key "coverage_type" not found in' \
            'specifications dictionary.')
        coverage_class = cls._registry.get(coverage_type_str)
        if not coverage_class:
            raise ValueError(
                f'Coverage calculator type "{coverage_type_str}" is not registered.'
            )
        return coverage_class.from_dict(specs)

@CoverageFactory.register_type(CoverageType.POINT_COVERAGE.to_string())
class PointCoverage:
    """A coverage calculator class which handles coverage calculation for an array of target
    points."""

    @classmethod
    def from_dict(cls, specs: Dict[str, Any]) -> "PointCoverage":
        # Empty since class does not require any initialization parameters
        return cls()

    def calculate_coverage(self, 
                           target_point_array: Cartesian3DPositionArray, 
                           fov: Union[CircularFieldOfView, RectangularFieldOfView],
                           frame_graph: FrameGraph,
                           times: AbsoluteDateArray,
                           surface: SurfaceType = SurfaceType.SPHERE,
                           buff_size=86401) -> DiscreteCoverageTP:
        """Calculates the coverage over an array of target points given a field of view.

        Args:
            target_point_array (Cartesian3DPositionArray): An array of target points in a Cartesian
                coordinate system.
            fov (Union[CircularFieldOfView, RectangularFieldOfView]): The field of view to use for
                coverage calculation.
            frame_graph (FrameGraph): The frame graph containing the necessary transformations
                between frames.
            times (AbsoluteDateArray): An array of time points for which the coverage is calculated
                at discrete instances.
            surface (SurfaceType): The type of surface to consider for coverage calculation.
                Defaults to SurfaceType.SPHERE. Surface type will effect the horizon check.
            buff_size (int): The size of the buffer for parallel computation. Should be at least
                the number of cores for optimal parallel performance.
        Returns:
            DiscreteCoverageTP: An object reporting the grid points covered at each time point.
        """

        num_pts = target_point_array.positions.shape[0]
        
        fov_frame = fov.frame
        target_frame = target_point_array.frame

        # Get the position of the fov frame origin in the target frame
        # expressed in target frame coordinates
        pos_fov_TARGET = frame_graph.get_pos_transform(from_frame=target_frame,
            to_frame=fov_frame, t=times)
        
        # Create list source for the (target frame) fov position
        pos_fov_TARGET_gte = [gte.Vector3d(p) for p in pos_fov_TARGET]
        pos_fov_TARGET_source = kcl.ListSourceVector3d(pos_fov_TARGET_gte)

        # Get the rotation from the fov frame to the target frame
        rot_fov_to_target, _ = frame_graph.get_orientation_transform(fov_frame,target_frame,times)

        # Get the transformation from fov to target frame as a ListSource
        fov_to_target_gte = [gte.Matrix3x3d(r.flatten()) for r in rot_fov_to_target.as_matrix()]
        FOV_to_TARGET_source = kcl.ListSourceMatrix3x3d(fov_to_target_gte)

        # Setup horizon source
        if surface == SurfaceType.SPHERE:
            earth_sphere = gte.Sphere3d(gte.Vector3d.Zero(), EARTH_RADIUS)
            sphere_source = kcl.ConstantSourceSphere3d(earth_sphere)
            horizon_source = \
                kcl.PolarHalfspaceSourceSphere3d(sphere_source, pos_fov_TARGET_source, buff_size)
            variables = [horizon_source]
        elif surface == SurfaceType.WGS84:
            extents = gte.Vector3d([EARTH_RADIUS,EARTH_RADIUS,EARTH_POLAR_RADIUS])
            earth_ellipsoid = gte.Ellipsoid3d(gte.Vector3d.Zero(), extents)
            ellipsoid_source = kcl.ConstantSourceEllipsoid3d(earth_ellipsoid)
            horizon_source = kcl.PolarHalfspaceSourceEllipsoid3d(
                    ellipsoid_source, pos_fov_TARGET_source, buff_size)
            variables = [horizon_source]
        elif surface == SurfaceType.NONE:
            variables = []
        else:
            raise ValueError(f"Unsupported surface type: {surface}")

        if isinstance(fov, CircularFieldOfView):
            deg2rad = math.pi / 180.0
            half_angle_rad = fov.diameter * 0.5 * deg2rad
            boresight_FOV_gte = gte.Vector3d(fov.boresight);
            boresight_FOV_source = kcl.ConstantSourceVector3d(boresight_FOV_gte)
            boresight_TARGET_source = \
                kcl.TransformedVector3dSource(boresight_FOV_source,FOV_to_TARGET_source,buff_size)
            angle_source = kcl.ConstantSourced(half_angle_rad)

            fov_source = kcl.PosDirConeSource(
                pos_fov_TARGET_source,boresight_TARGET_source,angle_source,buff_size)
            fov_viewer = kcl.ViewerCone3d(fov_source)                

            variables.append(boresight_TARGET_source)
            variables.append(fov_source)

        elif isinstance(fov, RectangularFieldOfView):
            deg2rad = math.pi / 180.0
            up_angle_rad = fov.ref_angle*2.0*deg2rad # convert half angle to full to match kcl
            right_angle_rad = fov.cross_angle*2.0*deg2rad # convert half angle to full to match kcl
            right_FOV = np.cross(fov.boresight, fov.ref_vector)
            right_FOV_gte = gte.Vector3d(right_FOV)
            up_FOV_gte = gte.Vector3d(fov.ref_vector)
            up_FOV_source = kcl.ConstantSourceVector3d(up_FOV_gte)
            right_FOV_source = kcl.ConstantSourceVector3d(right_FOV_gte)

            # Transform up and right vectors from FOV frame to TARGET frame
            up_TARGET_source = kcl.TransformedVector3dSource(
                up_FOV_source,FOV_to_TARGET_source,buff_size)
            right_TARGET_source = kcl.TransformedVector3dSource(
                right_FOV_source,FOV_to_TARGET_source,buff_size)

            right_angle_source = kcl.ConstantSourced(right_angle_rad)
            up_angle_source = kcl.ConstantSourced(up_angle_rad)
            fov_source = kcl.VectorAngleRectViewSource(pos_fov_TARGET_source,
                up_TARGET_source, right_TARGET_source, up_angle_source, right_angle_source,
                buff_size)
            
            fov_viewer = kcl.ViewerRectView3d(fov_source)

            # Add variables
            variables.append(up_TARGET_source)
            variables.append(right_TARGET_source)
            variables.append(fov_source)

        elif isinstance(fov, PolygonFieldOfView):
            raise NotImplementedError("PolygonFieldOfView is not yet implemented.")
        else:
            raise ValueError("Unsupported field of view type.")
        
        # Setup Viewers
        horizon_viewer = kcl.ViewerHalfspace3d(horizon_source)
        total_view = kcl.Intersection([horizon_viewer,fov_viewer])

        # Get the position of the grid points in the target frame
        target_points_list_gte = [gte.Vector3d(row) for row in target_point_array.to_numpy()]
        grid_source = kcl.ConstantSourceListVector3d(target_points_list_gte)

        # Setup Coverage source (no preprocessor)
        # Store all coverage output in memory
        buff_size_coverage = len(times)

        preprocessor = None
        cov_source = kcl.CovSource(grid_source, preprocessor, total_view, buff_size_coverage)
        variables.append(cov_source)

        # Drive coverage
        driver = kcl.VarDriver(variables)
        start_time = 0
        stop_time = len(times) - 1

        kcl.driveCoverage(buff_size,start_time,stop_time,driver,[])
        
        # Prepare coverage output
        coverage = [cov_source.get(i) for i in range(len(times))]

        return DiscreteCoverageTP(times,coverage,num_pts)