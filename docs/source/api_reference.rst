.. _api_reference:

API Reference
********************

.. todo:: Write about the commonality of the object structure, using from_dict, to_dict, OutputInfo objects, etc.

# Module Hierarchy

## 1. Core Utilities
- **base** - General-purpose utilities (logging, configuration, common mathematical functions).
- **time** - Time representations and conversions.  
  - Supported time scales: UTC, UT1  
  - Supported formats: Julian Date, Gregorian Calendar  
- **position** - Position reference frames and transformations.  
  - Reference Frames:  
    - ECI (ICRF)  
    - ECEF (ITRF)  
  - Representations:  
    - Cartesian  
    - WGS-84  

---

## 2. Orbital & Spacecraft Components
- **orbits** - Orbital state representations and transformations.  
  - Representations: TLE, Cartesian, Keplerian  
- **resources** - Spacecraft, Ground station, and Sensors.  
- **grid** - Spatial grids for coverage calculations.  

---

## 3. Dynamics & Event Modeling (Calculators)
- **propagation** - Orbit propagation methods.  
  - Supported models: J2, SGP4, Orekit numerical integrators  
- **coverage** - Observation/ground coverage computations.  
- **contact** - Contact time windows for ground stations and inter-satellite communications.  
- **eclipse** - Eclipse duration calculations.  
- **datametrics** - Metrics of (potential) observations based on instrument and viewing geometry.  

---

## 4. Mission-Level Analysis
- **mission** - Scenario setup, execution, and high-level mission representation.  
- **analysis** - Post-processing, data analysis, and visualization of results.  

                                ┌───────────────────────────┐
                                │      Core Utilities       │
                                └───────────────────────────┘
                                          │
        ┌────────────────────────────────┴────────────────────────────────┐
        │                         │                         │              │
      base                      time                   position           (…)
 (General utils)    (Time representations & conv.)  (Frames & conv.)  (Future)

─────────────────────────────────────────────────────────────────────────────────

                                ┌───────────────────────────┐
                                │ Orbital & Spacecraft Comp │
                                └───────────────────────────┘
                                          │
        ┌────────────────────────────────┴──────────────────────────────┐
        │                         │                        │            │
     orbits                  resources                   grid         (…)

 (Orbital states & trans.)  (Spacecraft, GS, sensors)  (Spatial grids)

─────────────────────────────────────────────────────────────────────────────────

                                ┌───────────────────────────┐
                                │  Dynamics & Event Models  │
                                └───────────────────────────┘
                                          │
   ┌────────────────────┬───────────────┬───────────────┬──────────────┬──────────────┐
   │                    │               │               │              │              │
 propagation        coverage         contact         eclipse      datametrics        (…)
(Orbit propagation)  (Obs./GS cov.)  (Comm. windows) (Eclipse calc) (Metrics & geom.)

─────────────────────────────────────────────────────────────────────────────────

                                ┌───────────────────────────┐
                                │  Mission-Level Analysis   │
                                └───────────────────────────┘
                                          │
                        ┌─────────────────┴─────────────────┐
                        │                                   │
                     mission                             analysis
         (Scenario setup & exec.)       (Post-processing, data analysis, viz.)

─────────────────────────────────────────────────────────────────────────────────


Please navigate below for detailed descriptions of OrbitPy's classes and functions. 

.. toctree::
   :maxdepth: 1
   
   module_base

.. comment   
   module_constellation
   module_propagator 
   module_grid
   module_coveragecalculator
   module_contactfinder
   module_eclipsefinder
   module_datametricscalculator
   module_mission
   module_sensorpixelprojection
   

   






