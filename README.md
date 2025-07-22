# **orbitpy-revised**  
A Python package (with a C++ base) for computing satellite remote-sensing orbit data. (This is a revision of the package in the `orbitpy` repository.)

The purpose of `OrbitPy` is to offer a comprehensive set of classes and functions tailored for Earth Observation mission simulations. 
Wherever feasible, third-party library classes and functions are leveraged to avoid redundancy. In certain cases, `OrbitPy` classes act as wrappers around these third-party implementations.

**Currently under active development.**

## Installation

Requires: Unix-like operating system, `python 3.13`, `pip`

Create a conda environment and install the dependencies:
```
conda create -n eosim-revised python=3.13
conda activate eosim-revised
conda install sphinx
pip install sphinx-rtd-theme
pip install pylint
pip install black
pip install coverage
pip install skyfield
pip install astropy
```

Install the `eosimutils` package: first download the repository and follow the instructions provided in its README file. This package installs the core utilities required by `orbitpy`.

Once the repository is set up, run the following command in the terminal to complete the installation of `orbitpy`:
```
make install
```

## Developer Notes

This repository is a revamp of the `OrbitPy` codebase, primarily aimed at removing the dependency on the `propcov` library. Instead, we will:  
- Use the `CoverageKinematics` library for coverage calculations (added as a submodule).  
- Integrate orbit propagators from third-party libraries such as Skyfield and Orekit.  
- Use `PyBind11` to create Python bindings for C++ implementations in `CoverageKinematics`. 

### **Challenges & Opportunities**  

The transition away from `propcov` presents a challenge because several core `OrbitPy` utility classes rely on wrapped `propcov` C++ objects (e.g., `propcov.OrbitState`, `propcov.AbsoluteDate`). However, this provides an opportunity to:  
- Redesign these classes with a cleaner API.  
- Incorporate lessons learned from prior development and use of `OrbitPy`.  
- Improve efficiency and maintainability.  

## **API & Design References**  

For API design, we will reference the following sources:  

- [OrbitPy library](https://github.com/EarthObservationSimulator/orbitpy/wiki)  
  - Key reference: `docs/api_reference.rst` for high-level descriptions of modules and classes.  
- [EOSE-API library](https://github.com/eose-tools-org/eose-api)  
- [TAT-C schemas](https://github.com/code-lab-org/tatc/tree/main/src/tatc/schemas)  
- [InstruPy](https://github.com/EarthObservationSimulator/instrupy/tree/master/instrupy)  
  - Supports instrument object imports and data metric calculations, while providing default instrument objects.  
- [SPICE Toolkit](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/index.html)  
- [Cesium CZML](https://github.com/AnalyticalGraphicsInc/czml-writer/wiki)  
- [STK](https://help.agi.com/stk/)

## **Coding Guidelines**  

Each class/function must adhere to the following:  

1. **Documentation**  
   - Use **Sphinx-style inline comments** 
   - Provide descriptive documentation and examples. (similar to [SPICE](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/index.html)).  

2. **Testing**  
   - Write unit tests and validation tests.  
   - Validation data is sourced from other mission simulators (e.g., GMAT or STK) or real-world datasets (e.g., CYGNSS data).  

3. **Code Style & Standards**  
   - Follow a well-defined **Styling README**.  
   - Use Python **type hints** and follow PEP 8/PEP 257 where applicable.  

4. **File Handling & Data Structures**  
   - Favor **NumPy arrays** for representing trajectories.  
   - Use **Python dictionaries** for data structures instead of niche objects (AstroPy, SpiceyPy).  
   - Consider **Xarray** or **Pandas** for data indexing.  

## **Major Dependencies**  

- **AstroPy**, `datetime` → Handling time conversions.  
- **Skyfield** → SGP4 propagation, TEME frame transformations, Keplerian elements.  
- **SPICE (SpiceyPy)** → Time and coordinate transformations (preferred over AstroPy for speed).  
- **Orekit** → Numerical orbit propagation.  
- **SciPy** → Rotations and interpolation.  
- **PyBind11** → Python bindings for C++ code.  
- **Pydantic?** → For enforcing data validation in API.  

## **General Guidelines**  

- **Leverage existing open-source tools** whenever possible instead of reinventing functionality.  
- **Standardize frame and time representations**:  
  - **Frames:** ITRF (ECEF), GCRF (ECI).  
  - **Time systems:** UTC, UT1.  
- **Use the latest stable Python version** (currently Python 3.13.2).  
- **Sphinx will be used for documentation.**

## **Revamping Strategy**  

1. **Catalog Existing Codebase**  
   - List all classes, functions, and modules in `OrbitPy`.  
   - Identify `propcov` dependencies that need replacements.

  **Count:** 
  - Modules: 9
  - Classes: 9 + 3 + 4 + 2 + 6 + 2 + 2 + 2 + 2 = 33
  - Functions: 7 + 0 + 1 + 2 + 4 + 1 + 0 + 1 + 0 = 16
  - Examples: ~10

Total: 58 + 10

2. **Establish Development Standards**  
   - Create a **Styling README** for code and documentation.  

3. **Collaborative Review Process**  
   - Open **GitHub discussions** for each module/class/function revision.  

4. **Prototyping & Validation**  
   - Develop prototypes with unit tests and validation cases.  
   - Get reviews before finalizing implementations.  

5. **Documentation & Examples**  
   - Generate high-level **Sphinx API documentation** for each class.  
   - Update examples to reflect the revised codebase.  

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


<strike> ### `orbitpy.sensorpixelprojection` </strike>

TBD if time permits.