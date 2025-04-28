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
                                ┌───────────────────────────────┐
                                │         Core Utilities         │
                                │          (eosimutils)          │
                                └───────────────────────────────┘
                                          │
        ┌─────────────────────────────────┴─────────────────────────────────────────────────────────────────┐
        │                 │                       │                       │                  │              │
      base              time                   state              timeseries         trajectory     orientation
(General utilities)  (Time scales,       (Cartesian positions)    (Time-varying     (path data)      (Attitude/
                     formats, conversions)                        data series)                        pointing)

───────────────────────────────────────────────────────────────────────────────────────────────────────────────

                                ┌────────────────────────────────────┐
                                │     Orbital & Spacecraft Comp.     │
                                └────────────────────────────────────┘
                                              │
                        ┌─────────────────────┴────────────────────────────────────────┐
                        │                           │                                  │
                     orbits                     resources                            grid
     (representations & conversions)     (Spacecraft, ground stations, sensors)  (Spatial coverage grids)

───────────────────────────────────────────────────────────────────────────────────────────────────────────────

                                ┌────────────────────────────────────┐
                                │     Dynamics & Event Modeling      │
                                └────────────────────────────────────┘
                                              │
             ┌──────────────┬──────────────┬──────────────┬──────────────┐
             │              │              │              │              │
        propagation      coverage        contact        eclipse     datametrics
   (Orbit propagation) (Observation/  (Comms windows)  (Eclipse      (Instrument     
                        GS coverage)                   predictions)  metrics)

───────────────────────────────────────────────────────────────────────────────────────────────────────────────

                                ┌────────────────────────────────────┐
                                │     Mission-Level Analysis         │
                                └────────────────────────────────────┘
                                              │
                            ┌─────────────────┴──────────────────┐
                            │                                    │
                         mission                             analysis
        (Scenario definition & execution)   (Results post-processing, metrics, visualization)



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
   

   






