
"""Orekit-based orbit propagator."""

from typing import Any, Dict, Optional

from os import chdir, getcwd, path
from pathlib import Path
import urllib.request
import numpy as np

import orekit
vm = orekit.initVM()

from orekit.pyhelpers import (
    setup_orekit_curdir,
    datetime_to_absolutedate,
    download_orekit_data_curdir,
)
from org.orekit.data import DataSource
from org.hipparchus.util import FastMath
from org.orekit.orbits import PositionAngleType, KeplerianOrbit, OrbitType
from org.orekit.propagation.numerical import NumericalPropagator
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
from org.orekit.models.earth.atmosphere import JB2008, Atmosphere
from org.orekit.forces.drag import DragForce, IsotropicDrag
from org.orekit.propagation import SpacecraftState
from org.orekit.bodies import CelestialBodyFactory, OneAxisEllipsoid
from org.orekit.frames import FramesFactory
from org.orekit.utils import Constants, IERSConventions, PVCoordinates
from org.orekit.time import AbsoluteDate as OrekitAbsoluteDate, TimeScalesFactory
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel, ThirdBodyAttraction, OceanTides
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.forces.radiation import IsotropicRadiationSingleCoefficient, SolarRadiationPressure
from org.orekit.models.earth.atmosphere.data import JB2008SpaceEnvironmentData
from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.orekit.propagation.analytical.tle import TLE, TLEPropagator

from eosimutils.trajectory import StateSeries
from eosimutils.state import CartesianState
from eosimutils.time import AbsoluteDate, AbsoluteDateArray
from eosimutils.base import ReferenceFrame

from .propagator import PropagatorFactory, PropagatorType

def setup_data_directory() -> Path:
    """Ensure the local ../data directory exists and contains required files."""
    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    orekit_zip = data_dir / "orekit-data.zip"
    solfsmy_txt = data_dir / "SOLFSMY.TXT"
    dtcfile_txt = data_dir / "DTCFILE.TXT"

    if not orekit_zip.exists():
        try:
            old_cwd = getcwd()
            chdir(data_dir)
            download_orekit_data_curdir()
            chdir(old_cwd)

            alt_zip = data_dir / "orekit-data-main.zip"
            if not orekit_zip.exists() and alt_zip.exists():
                alt_zip.rename(orekit_zip)
        except Exception:
            chdir(old_cwd)
            urllib.request.urlretrieve(
                "https://gitlab.orekit.org/orekit/orekit-data/-/archive/main/orekit-data-main.zip",
                orekit_zip,
            )

    if not solfsmy_txt.exists():
        urllib.request.urlretrieve(
            "https://sol.spacenvironment.net/JB2008/indices/SOLFSMY.TXT",
            solfsmy_txt,
        )

    if not dtcfile_txt.exists():
        urllib.request.urlretrieve(
            "https://sol.spacenvironment.net/JB2008/indices/DTCFILE.TXT",
            dtcfile_txt,
        )

    return data_dir

@PropagatorFactory.register_type(PropagatorType.OREKIT_PROPAGATOR.value)
class OrekitPropagator:
    """Numerical orbit propagator using Orekit.

    This class wraps an Orekit `NumericalPropagator` configured with common
    Earth-orbit force models (gravity, tides, drag, SRP, third bodies). Uses
    a Dormand-Prince integrator, with user-configurable
    absolute and relative tolerances.

    Args:
        stepSize: Propagation step size in seconds.
        mass: Spacecraft mass in kilograms.
        cross_section: Effective cross-sectional area for drag/SRP, in square meters.
        drag_coeff: Drag coefficient.
        srp_coeff: Solar radiation pressure coefficient.
        gravity_degree: Spherical harmonics degree for the Earth gravity model.
        gravity_order: Spherical harmonics order for the Earth gravity model.
        tide_degree: Spherical harmonics degree for the ocean tides model.
        tide_order: Spherical harmonics order for the ocean tides model.
        use_sun_third_body: If True, include Sun third-body perturbation.
        use_moon_third_body: If True, include Moon third-body perturbation.
        abs_tol: Absolute tolerance for the numerical integrator.
        rel_tol: Relative tolerance for the numerical integrator.
    """
    def __init__(
        self,
        stepSize: Optional[float] = None,
        mass: Optional[float] = None,
        cross_section: Optional[float] = None,
        drag_coeff: Optional[float] = None,
        srp_coeff: Optional[float] = None,
        gravity_degree: Optional[int] = None,
        gravity_order: Optional[int] = None,
        tide_degree: Optional[int] = None,
        tide_order: Optional[int] = None,
        use_sun_third_body: bool = True,
        use_moon_third_body: bool = True,
        abs_tol: Optional[float] = None,
        rel_tol: Optional[float] = None
    ) -> None:
        """Initialize an OrekitPropagator instance."""

        # Use provided values or defaults
        self.stepSize = float(stepSize) if stepSize is not None else 60.0
        self.mass = mass if mass is not None else 100.0
        self.cross_section = cross_section if cross_section is not None else 1.0
        self.drag_coeff = drag_coeff if drag_coeff is not None else 1.0
        self.srp_coeff = srp_coeff if srp_coeff is not None else 1.0
        self.abs_tol = abs_tol if abs_tol is not None else 1e-08
        self.rel_tol = rel_tol if rel_tol is not None else 1e-10
        self.gravity_degree = gravity_degree if gravity_degree is not None else 70
        self.gravity_order = gravity_order if gravity_order is not None else 70
        self.tide_degree = tide_degree if tide_degree is not None else 10
        self.tide_order = tide_order if tide_order is not None else 10
        self.use_sun_third_body = use_sun_third_body
        self.use_moon_third_body = use_moon_third_body

        # Initialize
        self.begin()

    def begin(self) -> None:
        """Set up frames, bodies, Earth model, force models, and propagator."""

        # Set up Orekit data
        data_dir = setup_data_directory()
        orekit_data_dir = str(data_dir / "orekit-data.zip")
        solfsmy_data_dir = str(data_dir / "SOLFSMY.TXT")
        dtc_data_dir = str(data_dir / "DTCFILE.TXT")

        setup_orekit_curdir(orekit_data_dir)
       
        # Create DataSource instances for the JB2008 data
        solfsmy_data = DataSource(solfsmy_data_dir)
        dtc_data = DataSource(dtc_data_dir)
        env_data = JB2008SpaceEnvironmentData(solfsmy_data, dtc_data)

        # Define celestial bodies, frames, and initial date
        self.sun = CelestialBodyFactory.getSun()
        self.moon = CelestialBodyFactory.getMoon()
        self.utc = TimeScalesFactory.getUTC()
        self.eci = FramesFactory.getGCRF()
        self.ecef = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
        self.earth = OneAxisEllipsoid(Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
                                      Constants.WGS84_EARTH_FLATTENING,
                                      self.ecef)

        # Setup solver parameters
        # For now, min step and max step are hardcoded
        # but it may be of interest in the future to make these user-configurable as well.
        integrator = DormandPrince853Integrator(1.0e-3, 300.0, self.abs_tol, self.rel_tol)
        self.propagator = NumericalPropagator(integrator)

        # Atmospheric Model
        self.atmosphere = JB2008(env_data, self.sun, self.earth)

        # Earth's Gravity Field
        provider = GravityFieldFactory.getNormalizedProvider(self.gravity_degree, self.gravity_order)
        itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
        self.gravity_model = HolmesFeatherstoneAttractionModel(itrf, provider)

        # Ocean Tides
        conventions = IERSConventions.IERS_2010
        ut1 = TimeScalesFactory.getUT1(conventions, True)
        self.tides_model = OceanTides(itrf,
                                Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
                                Constants.EIGEN5C_EARTH_MU,
                                self.tide_degree,
                                self.tide_order,
                                conventions,
                                ut1)

        # Third Body Attraction
        if self.use_sun_third_body:
            self.sun_model = ThirdBodyAttraction(self.sun)
        if self.use_moon_third_body:
            self.moon_model = ThirdBodyAttraction(self.moon)

        # Drag
        self.drag_model = DragForce(self.atmosphere, IsotropicDrag(self.cross_section, self.drag_coeff))

        # Solar Radiation Pressure
        self.srp_model = SolarRadiationPressure(self.sun, self.earth, IsotropicRadiationSingleCoefficient(self.cross_section, self.srp_coeff))

        self.update_force_models()

    def update_force_models(self) -> None:
        """Refresh the force models configured on the underlying propagator."""
        self.propagator.removeForceModels()
        self.propagator.addForceModel(self.gravity_model)
        self.propagator.addForceModel(self.tides_model)

        if self.use_sun_third_body:
            self.propagator.addForceModel(self.sun_model)
        if self.use_moon_third_body:
            self.propagator.addForceModel(self.moon_model)

        self.propagator.addForceModel(self.drag_model)
        self.propagator.addForceModel(self.srp_model)

    def set_drag_coeff(self, drag_coeff: float) -> None:
        """Update the drag coefficient and reconfigure force models.

        Args:
            drag_coeff: New drag coefficient to set.
        """
        self.drag_coeff = drag_coeff
        self.drag_model = DragForce(self.atmosphere, IsotropicDrag(self.cross_section, self.drag_coeff))
        self.update_force_models()

    def set_srp_coeff(self, srp_coeff: float) -> None:
        """Update the SRP coefficient and reconfigure force models.

        Args:
            srp_coeff: New SRP coefficient to set.
        """
        self.srp_coeff = srp_coeff
        self.srp_model = SolarRadiationPressure(self.sun, self.earth, IsotropicRadiationSingleCoefficient(self.cross_section, self.srp_coeff))
        self.update_force_models()

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "OrekitPropagator":
        """Create an OrekitPropagator from a specification dictionary.

        Args:
            d: Dictionary with Orekit propagator specifications. Supported keys:
                * "stepSize": Step size in seconds.
                * "mass": Spacecraft mass.
                * "cross_section": Cross-sectional area for drag/SRP.
                * "drag_coeff": Drag coefficient.
                * "srp_coeff": SRP coefficient.
                * "abs_tol": Absolute tolerance for the integrator.
                * "rel_tol": Relative tolerance for the integrator.
                * "gravity_degree": Earth gravity model degree.
                * "gravity_order": Earth gravity model order.
                * "tide_degree": Ocean tides model degree.
                * "tide_order": Ocean tides model order.
                * "use_sun_third_body": Whether to include Sun third-body perturbation.
                * "use_moon_third_body": Whether to include Moon third-body perturbation.

        Returns:
            OrekitPropagator: A configured OrekitPropagator instance.
        """

        return OrekitPropagator(
            stepSize=d.get('stepSize', None),
            mass=d.get('mass', None),
            cross_section=d.get('cross_section', None),
            drag_coeff=d.get('drag_coeff', None),
            srp_coeff=d.get('srp_coeff', None),
            abs_tol=d.get('abs_tol', None),
            rel_tol=d.get('rel_tol', None),
            gravity_degree=d.get('gravity_degree', None),
            gravity_order=d.get('gravity_order', None),
            tide_degree=d.get('tide_degree', None),
            tide_order=d.get('tide_order', None),
            use_sun_third_body=d.get('use_sun_third_body', True),
            use_moon_third_body=d.get('use_moon_third_body', True)
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the OrekitPropagator.

        Returns:
            dict: A dictionary that can be passed to `from_dict` to recreate
            an equivalent instance.
        """
        return {
            "@type": "OREKIT PROPAGATOR",
            "stepSize": self.stepSize,
            "mass": self.mass,
            "cross_section": self.cross_section,
            "drag_coeff": self.drag_coeff,
            "srp_coeff": self.srp_coeff,
            "abs_tol": self.abs_tol,
            "rel_tol": self.rel_tol,
            "gravity_degree": self.gravity_degree,
            "gravity_order": self.gravity_order,
            "tide_degree": self.tide_degree,
            "tide_order": self.tide_order,
            "use_sun_third_body": self.use_sun_third_body,
            "use_moon_third_body": self.use_moon_third_body
        }
    
    def execute(
        self,
        t0: AbsoluteDate,
        duration_days: float,
        initial_state: CartesianState,
    ) -> StateSeries:
        """Propagate an initial Cartesian state using the configured Orekit model.

        Args:
            t0: Start time for propagation.
            duration_days: Duration of propagation in days.
            initial_state: Initial state.

        Returns:
            StateSeries: Trajectory sampled at `self.stepSize`-second intervals.
            containing Cartesian position (km) and velocity (km/s) in the ICRF_EC frame.
        """
        # Convert start time to Orekit AbsoluteDate using astropy
        py_datetime = t0.to_astropy_time().to_datetime()
        orekit_start_date = datetime_to_absolutedate(py_datetime)

        # Build initial PVCoordinates in SI (meters, m/s)
        pv0 = None

        # Case 1: CartesianState initial condition
        if isinstance(initial_state, CartesianState):
            # Optional sanity check: require initial_state.time == t0
            try:
                if abs(initial_state.time.ephemeris_time - t0.ephemeris_time) > 1e-6:
                    raise ValueError(
                        "Start time t0 must match the time of the CartesianState for OrekitPropagator."
                    )
            except AttributeError:
                # If your AbsoluteDate doesn't expose ephemeris_time, skip check.
                pass

            pos_km = initial_state.position.to_numpy()
            vel_km_s = initial_state.velocity.to_numpy()

            pos_m = pos_km * 1000.0
            vel_m_s = vel_km_s * 1000.0

            pv0 = PVCoordinates(
                Vector3D(float(pos_m[0]), float(pos_m[1]), float(pos_m[2])),
                Vector3D(float(vel_m_s[0]), float(vel_m_s[1]), float(vel_m_s[2])),
            )
        # Case 2: TLE initial condition. Use orekit to propagate the TLE to t0 and extract 
        # the Cartesian state as the initial condition for the numerical propagator.
        elif hasattr(initial_state, "get_tle_as_tuple"):
            line1, line2 = initial_state.get_tle_as_tuple()
            tle_obj = TLE(line1, line2)
            tle_prop = TLEPropagator.selectExtrapolator(tle_obj)

            # Propagate TLE to the requested start time to get PV
            st = tle_prop.propagate(orekit_start_date)
            pv_tle = st.getPVCoordinates(self.eci)

            pv0 = PVCoordinates(
                Vector3D(
                    float(pv_tle.getPosition().getX()),
                    float(pv_tle.getPosition().getY()),
                    float(pv_tle.getPosition().getZ()),
                ),
                Vector3D(
                    float(pv_tle.getVelocity().getX()),
                    float(pv_tle.getVelocity().getY()),
                    float(pv_tle.getVelocity().getZ()),
                ),
            )
        else:
            raise ValueError("OrekitPropagator.execute expects a CartesianState or TLE")

        # Build Orekit orbit + spacecraft state
        orbit = KeplerianOrbit(pv0, self.eci, orekit_start_date, Constants.EIGEN5C_EARTH_MU)
        ic = SpacecraftState(orbit, self.mass)

        self.propagator.resetInitialState(ic)
        self.propagator.setOrbitType(OrbitType.CARTESIAN)

        # Setup time grid
        duration_sec = float(duration_days) * 86400.0
        step_time = float(self.stepSize)
        num_steps = int(np.floor(duration_sec / step_time)) + 1

        positions: list[list[float]] = []
        velocities: list[list[float]] = []

        # Build the time array for output
        t0_et = float(t0.ephemeris_time)

        et_array = np.empty(num_steps, dtype=float)
        et_array[0] = t0_et

        # Propagate
        for k in range(num_steps):
            dt = float(k) * step_time
            if k > 0:
                et_array[k] = t0_et + dt

            tt = orekit_start_date.shiftedBy(dt)
            st = self.propagator.propagate(tt)

            pv = st.getPVCoordinates(self.eci)
            p = pv.getPosition()
            v = pv.getVelocity()

            # Convert SI to km, km/s
            positions.append([p.getX() / 1000.0, p.getY() / 1000.0, p.getZ() / 1000.0])
            velocities.append([v.getX() / 1000.0, v.getY() / 1000.0, v.getZ() / 1000.0])

        positions_arr = np.asarray(positions, dtype=float)
        velocities_arr = np.asarray(velocities, dtype=float)

        absolute_date_array = AbsoluteDateArray(et_array)
        reference_frame = ReferenceFrame.get("ICRF_EC")

        return StateSeries(
            time=absolute_date_array,
            data=[positions_arr, velocities_arr],
            frame=reference_frame,
        )