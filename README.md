# orbitpy-revised
Python package (with C++ base) to compute satellite remote-sensing related orbit data. Revision of the `OrbitPy` package.

**Currently under active development.**

# Developer Notes

This repository is a revamp of the OrbitPy codebase, with the primary objective of removing the dependency on the `propcov` library. 
Instead, we will use the `CoverageKinematics` library for coverage calculations and integrate orbit propagators from third-party libraries, such as Skyfield and Orekit.
The `CoverageKinematics` library will be added as a submodule, and is based on C++, with python bindings using PyBind11.

One challenge in removing the `propcov` dependency lies in the use of (wrapped) propcov C++ objects by some core utility classes in OrbitPy. For example, 
classes like `propcov.OrbitState` and `propcov.AbsoluteDate` are integral to existing functionality. 
However, this transition offers an opportunity to redesign and improve these classes (their API), incorporating lessons learned from the development and use of the `OrbitPy` library.

## Guidelines:

We can use the following sources as reference for reformulating the API:

* [OrbitPy library](https://github.com/EarthObservationSimulator/orbitpy/wiki) 
     * Refer to the `docs/api_reference.rst` (and sub-links within the file) for high-level description of the modules and classes.
* [EOSE-API library](https://github.com/eose-tools-org/eose-api)
* [TAT-C schemas](https://github.com/code-lab-org/tatc/tree/main/src/tatc/schemas)
* [InstruPy](https://github.com/EarthObservationSimulator/instrupy/tree/master/instrupy): 
    Support import of instrument objects and datametrics calculation with the InstruPy library, while providing a default instrument object.
* [SPICE](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/index.html)
* [Cesium CZML](https://github.com/AnalyticalGraphicsInc/czml-writer/wiki)
* [STK](https://help.agi.com/stk/)

### Coding style

Each class/ function is supported with the following:

* Sphinx style inline commenting. Make the comments descriptive and provide examples of implementations. (Check out [SPICE](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/index.html) documentation.)

* Unit tests, validation.

* Descriptive documentation if required in Sphinx style documentation.

### List of major 3rd party dependencies under consideration:

* AstroPy, python datetime for handling time
* Skyfield for SGP4 propagation, coordinate transformations to the TEME frame, conversations of Inertial from/to Keplerian elements.
* SPICE (SpiceyPy) for time, coordinate transformations (favored for its computational speed over AstroPy)
* Orekit for numerical orbit propagation
* SciPy for rotations, interpolation.


### Other general guidelines

* We will rely on tested, and well maintained open-source 3rd party software whenever available, 
  instead of trying to recreate the functionality in-house. 
  For example, we can use `AstroPy` for time related operations, instead of coding in transformations from one time system to another.

* Use and support particular frame, time representations, which are commonly used another third party software, 
  real-world data representation (e.g., CYNGSS data). ITRF for ECEF, J2000 or ICRF for ECI. UTC and UT1 for time. 

* Use basic objects for API (python dicitonaries, numpy, and not niche objects such as AstroPy, SpicePY)

* Use the latest stable version of Python at the time of writing (Python 3.13.2).

* Sphinx will be used for documentation.

* Handle file writing by ....

* Use `pydantic`?

* Favor use of numpy (for representing trajectories). python dictionaries for data sets. numpy for file reading. xarray or pandas for data indexing.

## Revamping strategy

- List all the classes and functions in OrbitPy.
- Make a GitHub discussion post for each (revised/ revised) module, class, function.
- Make a prototype of the class/function with a unittest/ validation-test after the discussion and get it reviewed.
- Make a corresponding high-level Sphinx API doc for the class (along with inline Sphinx type documentation).
- Update examples with the revised codebase.

## List of classes, functions to be revamped/ developed

### `orbitpy.util`

*Classes*

* orbitpy.util.StateType
* orbitpy.util.DateType
* orbitpy.util.OrbitState
* orbitpy.util.SpacecraftBus
* orbitpy.util.Spacecraft
* orbitpy.util.GroundStation
* orbitpy.util.SpaceTrackAPI
* orbitpy.util.OrbitPyDefaults
* orbitpy.util.OutputInfoUtility

*Functions*

* orbitpy.util.helper_extract_spacecraft_params
* orbitpy.util.extract_auxillary_info_from_state_file
* orbitpy.util.dictionary_list_to_object_list
* orbitpy.util.object_list_to_dictionary_list
* orbitpy.util.initialize_object_list
* orbitpy.util.add_to_list
* orbitpy.util.calculate_inclination_circular_SSO

### `orbitpy.constellation`

*Classes*

* orbitpy.constellation.ConstellationFactory
* orbitpy.constellation.WalkerDeltaConstellation
* orbitpy.constellation.TrainConstellation

### `orbitpy.propagator`

*Classes*

* orbitpy.propagator.PropagatorFactory
* orbitpy.propagator.J2AnalyticalPropagator
* orbitpy.propagator.SGP4Propagator
* orbitpy.propagator.PropagatorOutputInfo

*Functions*

* orbitpy.propagator.compute_time_step

### `orbitpy.grid`

*Classes*

* orbitpy.grid.Grid
* orbitpy.grid.GridOutputInfo

*Functions*

* orbitpy.grid.GridPoint
* orbitpy.grid.compute_grid_res

### `orbitpy.coveragecalculator`

*Classes*

* orbitpy.coveragecalculator.CoverageCalculatorFactory
* orbitpy.coveragecalculator.GridCoverage
* orbitpy.coveragecalculator.PointingOptionsCoverage
* orbitpy.coveragecalculator.PointingOptionsWithGridCoverage
* orbitpy.coveragecalculator.SpecularCoverage
* orbitpy.coveragecalculator.CoverageOutputInfo

*Functions*

* orbitpy.coveragecalculator.helper_extract_coverage_parameters_of_spacecraft
* orbitpy.coveragecalculator.find_in_cov_params_list
* orbitpy.coveragecalculator.filter_mid_interval_access
* orbitpy.coveragecalculator.find_access_intervals

### `orbitpy.contactfinder`

*Classes*

* orbitpy.contactfinder.ContactFinder
* orbitpy.contactfinder.ContactFinderOutputInfo

*Functions*

* orbitpy.contactfinder.ContactPairs

### `orbitpy.eclipsefinder`

*Classes*

* orbitpy.eclipsefinder.EclipseFinder
* orbitpy.eclipsefinder.EclipseFinderOutputInfo

### `orbitpy.datametricscalculator`

*Classes*

* orbitpy.datametricscalculator.DataMetricsCalculator
* orbitpy.datametricscalculator.DataMetricsOutputInfo

*Functions*

* orbitpy.datametricscalculator.AccessFileInfo

### `orbitpy.mission`

*Classes*

* orbitpy.mission.Settings
* orbitpy.mission.Mission


### `orbitpy.sensorpixelprojection`

TBD if time permits.