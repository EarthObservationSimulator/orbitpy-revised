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

---

## 3. Dynamics & Event Modeling (Calculators)
- **propagation** - Orbit propagation methods.  
  - Supported models: J2, SGP4, Orekit numerical integrators  
- **coveragecalculator** - Observation/ground coverage computations.  
- **contactfinder** - Contact time windows for ground stations and inter-satellite communications.  
- **eclipsefinder** - Eclipse duration calculations.  
- **datametrics** - Metrics of (potential) observations based on instrument and viewing geometry.  

---

## 4. Mission-Level Analysis
- **mission** - Scenario setup, execution, and high-level mission representation.

                                              ┌───────────────────────────────┐
                                              │         Core Utilities        │
                                              │          (eosimutils)         │
                                              └───────────────────────────────┘
                                                            │
        ┌───────────────────────────────────────────────────┴───────────────────────────────────────────────────────────────────────────────────────────┐
        │                 │                       │                       │                  │              │               │            │              │
      base              time                   state              timeseries         trajectory       orientation      framegraph    standardframes  fieldofview
(General utilities)  (Time scales,       (Cartesian positions)    (Time-varying     (path data)        (Attitude/        (frame       (LVLH frame)     (FOVs)  
                formats, conversions)                              data series)                          pointing)  transformations)      

miscellaneous: spicekernels, plotting, thirdpartyutils, utils
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

                                ┌────────────────────────────────────┐
                                │     Orbital & Spacecraft Comp.     │
                                └────────────────────────────────────┘
                                              │
                        ┌─────────────────────┴─────┐
                        │                           │
                     orbits                     resources
     (representations & conversions)     (Spacecraft, ground stations, sensors)

───────────────────────────────────────────────────────────────────────────────────────────────────────────────

                                ┌────────────────────────────────────┐
                                │     Dynamics & Event Modeling      │
                                └────────────────────────────────────┘
                                                │
             ┌──────────────────┬────────────────────────────┬──────────────┐
             │                  │                            │              │
        propagator        coveragecalculator        contactfinder        eclipsefinder
   (Orbit propagation)   (+ coverage, specular)    (Comms windows)       (Eclipse
                        (Access opportunities)                            predictions)

───────────────────────────────────────────────────────────────────────────────────────────────────────────────

                                ┌────────────────────────────────────┐
                                │     Mission-Level Simulation       │
                                └────────────────────────────────────┘
                                              │
                                            mission                             
                            (Scenario definition & execution)      

miscellaneous: utils, specular, coverage, plotting


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
   

   






