# CLAUDE.md — orbitpy-revised Architecture & Developer Reference

## Project Overview

**orbitpy-revised** is a Python package for computing satellite remote-sensing orbit data. It is a ground-up revision of the original OrbitPy, replacing the `propcov` C++ propagation dependency with a cleaner split:
- **CoverageKinematics** (C++ via PyBind11) — geometry and coverage calculation
- **Skyfield** — SGP4 orbit propagation
- **Orekit** — high-fidelity numerical orbit propagation
- **eosimutils** — shared time, state, frame, and FOV primitives

- **Version**: 0.1.0 (active development)
- **License**: MIT
- **Python requirement**: >= 3.13
- **Authors**: Vinay Ravindra (vravindra@baeri.org), Ryan Ketzner (ketzner@ucf.edu)

---

## Repository Layout

```
orbitpy-revised/
├── orbitpy/                  # Main package (14 modules)
│   ├── __init__.py           # Empty — no explicit re-exports
│   ├── mission.py            # Mission orchestrator (top-level façade)
│   ├── orbits.py             # Orbit representations + Space-Track API
│   ├── propagator.py         # Propagator factory + SGP4Propagator
│   ├── orekitpropagator.py   # High-fidelity Orekit numerical propagator
│   ├── coverage.py           # Coverage data structures (TP/GP/continuous)
│   ├── coveragecalculator.py # Coverage computation algorithms
│   ├── contactfinder.py      # Ground station / entity contact finding
│   ├── eclipsefinder.py      # Earth shadow eclipse detection
│   ├── specular.py           # GNSS-R specular point geometry
│   ├── glsdc.py              # Generalized least-squares differential correction solver
│   ├── resources.py          # Spacecraft, Sensor, GroundStation classes
│   ├── utils.py              # LOS check, elevation angle, vector utils
│   └── plotting.py           # matplotlib visualization of coverage
├── extern/
│   └── CoverageKinematics/   # C++ coverage/geometry submodule (PyBind11)
│       └── lib/
│           ├── GeometricTools/  # Geometric primitives fork (submodule)
│           ├── gmatutil/        # GMAT utility code
│           └── specular/        # Specular point C++ implementation
├── tests/                    # 6 unittest modules
├── slowtests/                # STK/GMAT validation tests
│   ├── stk/
│   └── gmat/
├── examples/                 # Mission example scripts + JSON configs
│   ├── mission_simple/       # MissionSpecs.json — single spacecraft
│   ├── mission_gnssr/        # GNSS-R mission config
│   ├── planet_skysat/        # Planet SkySat constellation
│   ├── cygnss_gnssr/         # CYGNSS constellation
│   ├── dshield-cygnss-demo/  # CYGNSS GNSS-R demo + D-SHIELD CSV export (see section below)
│   ├── specular_coverage/    # GPS OMM data + specular examples
│   ├── cygnss_data/          # CYGNSS + GPS PRN reference trajectory CSVs
│   ├── processed_data/       # Pre-computed .npy arrays for validation
│   └── *.py                  # Standalone scripts (solve_drag_coefficient*.py, specular_*.py, point_coverage.py, orekit_propagator.py, dshield_format_converter.py)
├── bin/
│   └── run_mission.py        # CLI entry point: execute mission from JSON
├── pyproject.toml            # Build config (scikit-build-core, pybind11)
├── environment.yml           # Conda env with C++ build deps
└── Makefile                  # install / test / lint / format / docs / coverage
```

---

## Module Dependency Graph

```
eosimutils  (external package — shared foundation)
│
orbits.py               (imports: eosimutils.base, time, state, trajectory; external: sgp4, skyfield, requests)
propagator.py           (imports: orbits, eosimutils.base, time, trajectory; external: skyfield, sgp4)
orekitpropagator.py     (imports: eosimutils.time, trajectory, orbits, propagator; external: orekit, skyfield)
resources.py            (imports: eosimutils.base, time, state, fieldofview, standardframes, framegraph)
coverage.py             (imports: eosimutils.time, state; external: numpy)
coveragecalculator.py   (imports: coverage, eosimutils.*; external: kcl C++ bindings)
contactfinder.py        (imports: eosimutils.*, utils; external: numpy)
eclipsefinder.py        (imports: eosimutils.*, spicekernels; external: spiceypy)
specular.py             (imports: eosimutils.*, coverage; external: kcl C++ bindings)
glsdc.py                (standalone — no orbitpy/eosimutils imports; external: numpy)
utils.py                (imports: eosimutils.base; external: numpy)
plotting.py             (imports: coverage, eosimutils.state; external: matplotlib)
mission.py              (imports: orbits, propagator, resources, coveragecalculator, contactfinder, eclipsefinder, specular, eosimutils.*)
```

---

## Module Reference

### `mission.py` — Mission Orchestrator

**Purpose**: Top-level façade. Holds all mission parameters and runs propagation, eclipse finding, contact finding, and coverage calculation in sequence.

#### Module-Level Functions

**`auto_retrieve_orbit(norad_id, target_date_time, space_track_credentials_fp) -> OrbitalMeanElementsMessage`**
Fetches the OMM from Space-Track.org created closest to (but before) `target_date_time`. Note: `CREATION_DATE` in the OMM differs from `EPOCH`.

**`propagate_spacecraft(spacecraft, propagator, t0, duration_days, space_track_credentials_fp=None) -> StateSeries`**
Propagates a single spacecraft. If the spacecraft has only a `norad_id` (no explicit orbit), fetches OMM automatically. Returns `StateSeries` in ICRF_EC.

**`calculate_gs_contact(trajectory, ground_station, frame_graph) -> ContactInfo`**
Computes contact windows between spacecraft trajectory and a single ground station using `ElevationAwareContactFinder`.

#### `Settings`

Miscellaneous mission settings.

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `user_dir` | `Optional[str]` | None | Base directory for resolving relative paths in JSON |
| `coverage_type` | `CoverageType` | `POINT_COVERAGE` | `POINT_COVERAGE` or `SPECULAR_COVERAGE` |
| `specular_radius_km` | `Optional[float]` | None | Radius around specular point (GNSS-R) |
| `spacetrack_credentials_relative_path` | `Optional[str]` | None | Relative path to credentials JSON |
| `surface_type` | `SurfaceType` | `WGS84` | Earth surface model |

Methods: `from_dict()`, `to_dict()`

#### `Mission`

The central execution object.

**Constructor Args**:

| Arg | Type | Description |
|-----|------|-------------|
| `start_time` | `AbsoluteDate` | Mission start |
| `duration_days` | `float` | Duration |
| `spacecrafts` | `Union[Spacecraft, List[Spacecraft]]` | Primary spacecraft (receiver) |
| `ground_stations` | `Optional[Union[GroundStation, List[GroundStation]]]` | Ground stations for contact finding (single value normalized to list) |
| `propagator` | `Optional[SGP4Propagator]` | Propagator (SGP4 default) |
| `cartesian_spatial_points` | `Optional[Cartesian3DPositionArray]` | Coverage target grid |
| `gnss_spacecrafts` | `Optional[List[Spacecraft]]` | GNSS transmitter spacecraft (GNSS-R only) |
| `frame_graph` | `FrameGraph` | Auto-created with SPICE transforms if None |
| `settings` | `Optional[Settings]` | |

**Key Methods**:

| Method | Returns | Description |
|--------|---------|-------------|
| `execute_propagation()` | `(List[Dict], Optional[List[Dict]])` | Propagates all rx + tx spacecraft |
| `execute_eclipse_finder(trajectories)` | `List[Dict]` | Eclipse intervals per spacecraft |
| `execute_gs_contact_finder(trajectories)` | `List[Dict]` | Contact intervals per spacecraft/GS pair |
| `execute_coverage_calculator(trajectories)` | `List[Dict]` | Point coverage per spacecraft/sensor |
| `execute_gnssr_coverage_calculator(rx_traj, tx_traj, topk=None, ranking_rcg=None)` | `List[Dict]` | GNSS-R coverage; `topk` keeps the top-k transmitters per time step. `ranking_rcg` overrides the ranking key (used by `execute_all` to share the trajectory's geometric RCG) |
| `execute_specular_trajectory_calculator(rx_traj, tx_traj, topk=None)` | `List[Dict]` | Specular point trajectories; `topk` keeps only the top-k highest-RCG trajectories per time step (ranked by the geometric LOS RCG). Thin wrapper over `_specular_data_by_rx` + `_build_specular_output` |
| `execute_all(topk=None)` | `Dict` | Runs all configured analyses; returns unified result bundle. `topk` forwarded to both GNSS-R coverage and specular trajectories, ranked by one shared geometric RCG so the two top-k outputs are aligned. Gotcha: returns an **empty dict** if no spatial points are configured (propagation/eclipse/contact still run but their results are dropped); `contact_finder_results` is `None` (not absent) when there are no ground stations |
| `from_dict(dict)` | `Mission` | Deserialize; supports references to external JSON files |
| `to_dict()` | `Dict` | Serialize |

**`from_dict` JSON format supports inline objects or relative-path file references for spacecraft, sensors, propagator, etc.**

---

### `orbits.py` — Orbit Representations

**Purpose**: Spacecraft orbit formats plus Space-Track.org API access.

**Constant**: `GM_EARTH = 398600.435507` km³/s²

#### `OrbitType(EnumBase)`
Values: `TWO_LINE_ELEMENT_SET`, `ORBITAL_MEAN_ELEMENTS_MESSAGE`, `OSCULATING_ELEMENTS`, `SGP4_SATREC_ORBITAL_PARAMETERS`, `CARTESIAN_STATE`

#### Factory

**`OrbitFactory`** — Registry/factory pattern. `from_dict()` dispatches on `"orbit_type"` key.

#### Orbit Classes

**`TwoLineElementSet`**
- Attributes: `line0` (optional name line), `line1`, `line2` (strings)
- `get_tle_as_tuple()` → `(line1, line2)`
- Dict keys: `TLE_LINE0` (opt), `TLE_LINE1`, `TLE_LINE2`

**`OrbitalMeanElementsMessage`**
- Wraps the OMM JSON dict per CCSDS standard
- `from_json(json_str)` — Parse from JSON string
- `get_field_as_str(field_name)` — Extract any OMM field
- `get_tle_as_tuple()` — Extract TLE lines embedded in OMM

**`OsculatingElements`** (Keplerian elements; ICRF_EC frame only)

| Attribute | Unit | Description |
|-----------|------|-------------|
| `time` | `AbsoluteDate` | Epoch |
| `semi_major_axis` | km | |
| `eccentricity` | — | |
| `inclination` | degrees | |
| `raan` | degrees | Right Ascension of Ascending Node |
| `arg_of_perigee` | degrees | |
| `true_anomaly` | degrees | |

- `from_cartesian_state(state, gm=GM_EARTH)` — Cartesian → Keplerian
- `to_cartesian_state(gm=GM_EARTH)` — Keplerian → Cartesian

**`Sgp4SatrecOrbitalParameters`** (SGP4 mean elements; TEME frame)
- Used to directly drive the `sgp4` library without TLE strings
- Key attributes: `epoch`, `inclo`, `nodeo`, `ecco`, `argpo`, `mo`, `no_kozai` (rev/day), `bstar`, `ndot`, `nddot`
- `epoch_satrec_repr` — Days since 1949 Dec 31 (sgp4 library internal format)
- `get_sgp4_satrec()` → sgp4 `Satrec` object (WGS72, improved mode)
- `compute_no_kozai_from_semi_major_axis(sma_km)` — Mean motion from SMA
- `tle_lines_to_satrec_orbital_parameters(line1, line2)` — Extract from TLE strings

#### `SpaceTrackAPI`

Interface to [space-track.org](https://www.space-track.org).

- Credentials loaded from a JSON file with `"username"` and `"password"` keys
- `login()` / `logout()` — Manage session
- `get_closest_omm(norad_id, target_date_time, within_days=1)` — Retrieve OMM created before and nearest to target date

---

### `propagator.py` — Propagator Factory + SGP4

**Purpose**: Factory for propagators and the SGP4 implementation (wraps Skyfield).

**`PropagatorType(EnumBase)`**: `SGP4_PROPAGATOR`, `OREKIT_PROPAGATOR`

**`PropagatorFactory`** — `from_dict()` dispatches on `"propagator_type"` key.

#### `SGP4Propagator`

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `step_size` | `float` | 60 | Time step in seconds |

**`execute(t0, duration_days, orbit) -> StateSeries`**
- `orbit`: `TwoLineElementSet`, `OrbitalMeanElementsMessage`, or `Sgp4SatrecOrbitalParameters`
- Returns `StateSeries` in ICRF_EC (Skyfield's GCRS equivalent)
- Uniform time grid from `t0` to `t0 + duration_days`

---

### `orekitpropagator.py` — Orekit Numerical Propagator

**Purpose**: High-fidelity numerical propagation including gravity harmonics, atmospheric drag (JB2008), SRP, and third-body perturbations.

**`setup_data_directory()`** — Ensures `../data/` directory with required files (auto-downloads `orekit-data.zip`, `SOLFSMY.TXT`, `DTCFILE.TXT`).

#### `OrekitPropagator`

Default constructor parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `step_size` | 60.0 s | Output time step |
| `mass` | 100.0 kg | Spacecraft mass |
| `cross_section` | 1.0 m² | Drag cross-section |
| `drag_coeff` | 1.0 | Drag coefficient Cd |
| `srp_coeff` | 1.0 | SRP coefficient Cr |
| `gravity_degree/order` | 70 | EGM gravity spherical harmonics |
| `tide_degree/order` | 10 | Ocean tides (EIGEN5C) |
| `use_sun_third_body` | True | |
| `use_moon_third_body` | True | |
| `abs_tol / rel_tol` | 1e-8 / 1e-10 | Dormand-Prince 853 integrator tolerances |

**`begin()`** — Initialize Orekit JVM, frames, and all force models. Must be called before `execute()`.

**`execute(t0, duration_days, initial_state) -> StateSeries`**
- `initial_state`: `CartesianState`, `TwoLineElementSet`, or `OrbitalMeanElementsMessage`
- For TLE input: propagates to `t0` via SGP4, then numerically integrates from there
- Returns `StateSeries` in ICRF_EC

---

### `coverage.py` — Coverage Data Structures

**Purpose**: Data model for discrete and continuous coverage results.

#### `DiscreteCoverageTP` (time-primary indexing)

Coverage stored as: for each time index → list of covered grid point indices.

| Attribute | Type | Description |
|-----------|------|-------------|
| `time` | `AbsoluteDateArray` | Time grid |
| `coverage` | `List[List[int]]` | `coverage[t_idx]` = list of zero-indexed grid point indices |
| `grid_points` | `Cartesian3DPositionArray` | Grid point positions |

- `coverage_time()` → total count of (time, grid-point) covered pairs
- `from_gp(DiscreteCoverageGP)` / `to_gp()` — Convert between TP and GP indexing
- `__getitem__(time_index)` → `List[int]` covered grid indices at that time

#### `DiscreteCoverageGP` (grid-primary indexing)

Coverage stored as: for each grid point → list of covered time indices.

| Attribute | Type | Description |
|-----------|------|-------------|
| `time` | `AbsoluteDateArray` | Time grid |
| `coverage` | `List[List[int]]` | `coverage[gp_idx]` = list of zero-indexed time indices |
| `grid_points` | `Cartesian3DPositionArray` | Grid point positions |

- `coverage_steps()` → `np.ndarray` of covered step count per grid point
- `symmetric_difference(a, b)` (static) — XOR coverage between two GP coverages
- `from_tp(DiscreteCoverageTP)` / `to_tp()` — Convert between indexing schemes

#### `ContinuousCoverageGP` (interval-based, grid-primary)

| Attribute | Type | Description |
|-----------|------|-------------|
| `coverage` | `List[AbsoluteDateIntervalArray]` | Coverage intervals per grid point |
| `grid_points` | `Cartesian3DPositionArray` | |

- `to_discrete(start, step, num_points)` → `DiscreteCoverageGP`
- `from_stk(file_path, grid_points)` → Parse STK `.cvaa` coverage report format

#### Helper

**`get_integer_intervals(sorted_array) -> List[Tuple[int,int]]`**
Groups a sorted integer array into contiguous run tuples. E.g. `[1,2,3,7]` → `[(1,3),(7,7)]`.

---

### `coveragecalculator.py` — Coverage Computation

**Purpose**: Algorithms for determining which target grid points a spacecraft sensor covers at each time step.

**`CoverageType(EnumBase)`**: `POINT_COVERAGE`, `SPECULAR_COVERAGE`

**`CoverageFactory`** — `from_dict()` dispatches on `"coverage_type"` key.

#### `PointCoverage`

**`calculate_coverage(target_point_array, fov, frame_graph, times, surface=SPHERE, use_cbpa=False, cbpa_cells=None, buff_size=None) -> DiscreteCoverageTP`**

Steps performed at each time step:
1. Horizon check — target point must have LOS to spacecraft through Earth surface
2. FOV containment — target point must lie within the sensor's field of view

- `surface`: `SPHERE`, `WGS84`, or `NONE` (controls horizon masking)
- `use_cbpa`: enable Cell-Based Preprocessing Algorithm for large grids
- `compute_cbpa_cells(fov, distance, num_pts)` — recommended cell count
- `scale_cbpa_cells(area, num_pts, ...)` — scale cells empirically

Leverages the C++ `kcl` (CoverageKinematics) library for geometric operations.

#### `SpecularCoverage`

For GNSS-R scenarios: checks that a specular reflection point between a transmitter and receiver falls within the receiver's FOV and covers a target point.

**`calculate_coverage(target_point_array, fov, frame_graph, times, transmitters, specular_radius, surface=SPHERE, buff_size=100) -> List[Tuple[DiscreteCoverageTP, List[float]]]`**

Per transmitter, at each time:
1. LOS check: transmitter ↔ receiver
2. Find specular point on Earth surface
3. Target point must lie within `specular_radius` km of specular point
4. Target point must be in receiver FOV

Returns list of `(coverage, rcg_factor_list)` tuples — one per transmitter.
- **RCG formula**: `G / (R_t² × R_r²)` where R_t and R_r are transmitter/receiver distances to the specular point; G=1.0 (unity gain).

- `get_best_coverage(coverage_list)` — Best RCG across transmitters at each time
- `get_topk_coverage(coverage_list, k)` — Top-k transmitters by RCG, returned as `k` `(DiscreteCoverageTP, rcg_list)` tuples ordered best-first (rank 0 = highest RCG). Selection is **per time step** (the transmitter behind a rank can change each step). Note: this method does **not** track which transmitter each rank came from — it drops identity. To recover identity, `mission.py` re-ranks the same RCGs with `specular.get_topk_trajectories` (identical ranking logic), so the selected ids line up exactly with the coverage ranks.

---

### `contactfinder.py` — Contact Finding

**Purpose**: Determine when line-of-sight contact exists between two entities (satellite-to-ground, satellite-to-satellite).

**`ContactFinderType(EnumBase)`**: `LOS_CONTACT_FINDER`

**`ContactFinderFactory`** — `from_dict()` dispatches on `"contact_finder_type"` key.

#### `ContactInfo(Timeseries)`

Boolean timeseries of contact state.

- `data[0]` — numpy bool array, length = number of time points
- `headers` — `["contact"]`
- `has_contact(index=None)` — Contact at specific index or any contact in series
- `contact_intervals()` — List of `(start_AbsoluteDate, end_AbsoluteDate)` tuples

#### `LineOfSightContactFinder`

**`execute(frame_graph, entity1_state, entity2_state) -> ContactInfo`**
- Entity state types accepted: `StateSeries`, `PositionSeries`, `CartesianState`, `GeographicPosition`, `Cartesian3DPosition`
- LOS uses spherical Earth with `WGS84_EARTH_POLAR_RADIUS`
- Limitation: requires at least one entity to be fixed

#### `ElevationAwareContactFinder(LineOfSightContactFinder)`

**`execute(frame_graph, observer_state, target_state, min_elevation_angle) -> ContactInfo`**
- First computes LOS, then filters by elevation angle at observer
- `min_elevation_angle`: degrees above horizon (0° = horizon, 90° = zenith)

#### Helper

**`get_entities_position_as_numpy(frame_graph, entity1_state, entity2_state) -> (bool, ndarray, bool, ndarray)`**
Extracts positions as numpy arrays in a common frame, applying frame transforms as needed.

---

### `eclipsefinder.py` — Eclipse Detection

**Purpose**: Determine when spacecraft (or ground stations) are in Earth's shadow.

#### `EclipseInfo(Timeseries)`

Boolean timeseries of eclipse state.

- `headers` — `["eclipse"]`
- `is_eclipsed(index=None)` — Eclipse at specific index or any eclipse
- `eclipse_intervals()` — List of `(start, end)` `AbsoluteDate` tuples

#### `EclipseFinder`

Uses SPICE for Sun position; simple spherical Earth shadow model (no umbra/penumbra distinction).

**`execute(frame_graph, time=None, position=None, state=None, interpolator="linear") -> EclipseInfo`**

Accepted call patterns:
- `time + position` — Fixed position at specified time(s)
- `state` — Moving object; time and position from `StateSeries`
- `time + state` — Evaluate at given times using state's positions

Uses `WGS84_EARTH_POLAR_RADIUS` to avoid false positives near limb.

---

### `specular.py` — GNSS-R Specular Geometry

**Purpose**: Compute specular reflection points on the Earth's surface between GNSS transmitters and receivers.

**`get_specular_trajectory(transmitter_states_itrf, receiver_states_itrf, times, surface=WGS84) -> (PositionSeries, ndarray)`**
- **Both state series must be in ITRF frame**
- Uses Newton's method (tol=1e-10, max 20 iterations) for ellipsoidal specular point
- Returns NaN positions for times with no LOS between tx and rx
- RCG factor returned as `ndarray` (G/(R_t² × R_r²), G=1.0)

**`get_best_trajectory(traj_list) -> (PositionSeries, List[float])`**
Select trajectory with highest RCG at each time.

**`get_topk_trajectories(traj_list, k, ids) -> (List[Tuple], List[List[int]])`**
Select top-k trajectories by RCG, returning trajectory tuples and the selected transmitter IDs per time.

---

### `glsdc.py` — Generalized Least-Squares Differential Correction

**Purpose**: Standalone nonlinear least-squares solver (batch orbit-determination style). Minimizes `(y − F(x))ᵀ R⁻¹ (y − F(x))`. Not imported by any other orbitpy module; used by `examples/solve_drag_coefficient.py` and `examples/solve_drag_coefficient_x0.py` to estimate the CYGNSS drag coefficient (and initial state) with the Orekit propagator as the measurement model.

#### `GLSDC`

| Constructor arg | Default | Description |
|-----------------|---------|-------------|
| `max_iters` | 50 | Maximum differential-correction iterations |
| `tol` | 1e-10 | Convergence tolerance on the parameter-correction norm |
| `fd_eps` | 1e-6 | Relative finite-difference step size |
| `x_scale` | None | Optional parameter scale vector (internal scaling only) |

**`solve(F, x0, y, R=None, jacobian=None, verbose=True) -> GLSDCResults`**
- `F`: measurement model (physical parameter vector → measurement vector)
- `R`: measurement covariance (identity if None)
- `jacobian`: optional analytic Jacobian; central finite differences if None
- Normal equations solved with `np.linalg.solve`, falling back to pseudo-inverse

#### `GLSDCResults` (dataclass)
Fields: `x_hat`, `y_hat`, `residual`, `jacobian`, `normal_matrix`, `covariance` (pinv of normal matrix), `dx`, `cost`, `iterations`, `converged`, `x_history`, `cost_history`.

#### `FiniteDifferenceJacobian`
Callable computing a central finite-difference Jacobian of a vector function; step per parameter is `eps * max(1, |x_j|)`.

---

### `resources.py` — Mission Resources

**Purpose**: Data classes for spacecraft, sensors, and ground stations.

#### `GroundStation`

| Attribute | Type | Description |
|-----------|------|-------------|
| `identifier` | `str` | UUID (auto-generated) |
| `name` | `Optional[str]` | |
| `geographic_position` | `GeographicPosition` | WGS84 lat/lon/height |
| `min_elevation_angle_deg` | `float` | Minimum contact elevation (degrees) |

Dict keys: `id`, `name`, `latitude`, `longitude`, `height`, `min_elevation_angle`

#### `Sensor`

| Attribute | Type | Description |
|-----------|------|-------------|
| `identifier` | `str` | UUID (auto-generated) |
| `name` | `Optional[str]` | |
| `fov` | `CircularFOV \| RectangularFOV \| PolygonFOV` | Field of view (via `FieldOfViewFactory`) |

Dict key `"fov"` dispatched through `FieldOfViewFactory.from_dict()`.

#### `Spacecraft`

| Attribute | Type | Description |
|-----------|------|-------------|
| `identifier` | `str` | UUID (auto-generated) |
| `name` | `Optional[str]` | |
| `norad_id` | `Optional[int\|str]` | For Space-Track auto-retrieval |
| `orbit` | `TwoLineElementSet \| OrbitalMeanElementsMessage \| OsculatingElements \| None` | |
| `local_orbital_frame_handler` | `LVLHType1FrameHandler` | Default: LVLH_TYPE_1 with name `"LVLH_<identifier>"` (identifier used as-is, no case change; it is a UUID only when not user-supplied) |
| `sensor` | `List[Sensor]` | Always a list; single sensor is normalized |

**Design**: Either `orbit` or `norad_id` must be provided. If `norad_id` only, orbit is fetched at runtime via Space-Track. Spacecraft always gets an LVLH frame registered in the frame graph during mission execution.

---

### `utils.py` — Utility Functions

**`normalize(v) -> list[float]`**
Normalize vector to unit length. Raises `ZeroDivisionError` if zero magnitude.

**`check_line_of_sight(object1_pos, object2_pos, obstacle_radius) -> bool`**
LOS check using spherical obstacle. Algorithm: Vallado "Fundamentals of Astrodynamics and Applications" p.198. Handles antipodal and near-surface edge cases (tolerances 1e-5 to 1e-9). Frame must be centered at obstacle center.

**`calculate_elevation_angle(observer_position, target_position) -> float`**
Elevation angle in degrees. Convention: 90° = zenith, 0° = horizon, -90° = nadir. Computed as `arcsin(dot(zenith_unit, relative_unit))`.

---

### `plotting.py` — Visualization

**`plot_covered_steps(coverage, positions, max_steps=None) -> Axes3D`**
3D scatter plot of grid points colored by coverage count using viridis colormap. Returns matplotlib 3D axes.

---

### `extern/CoverageKinematics/` — C++ Coverage Submodule

**Purpose**: A standalone C++ library (with Python bindings via PyBind11) for discrete coverage geometry. Imported as `kcl` in Python code.

**Key Capabilities**:
- Point-in-FOV containment tests
- Horizon masking against spherical and ellipsoidal Earth
- Specular point calculation (Newton's method on sphere and WGS84 ellipsoid)
- Cell-Based Preprocessing Algorithm (CBPA) for large grids
- Thread-parallel evaluation via TBB

**C++ Dependencies**: Eigen ≥3.4, TBB, fmt, GoogleTest, pybind11, GeometricTools (sub-submodule)

**Build**: CMake with Release/Debug modes. Conda environment provides all C++ deps (`eigen`, `tbb`, `fmt`, `gtest`, `cmake`, `ninja`).

**Python `kcl` objects** (used inside `coveragecalculator.py` and `specular.py`):
- Source/Variable pattern for efficient geometric updates
- Viewer objects with CSG operations (intersection, union, complement)

---

## `bin/run_mission.py` — CLI Entry Point

Execute a mission from a JSON config file:

```bash
python bin/run_mission.py <path_to_user_directory>
```

- Reads `MissionSpecs.json` from the user directory
- Calls `Mission.from_dict()` → `mission.execute_all()`
- Writes results to `MissionOutput.json` in the same directory
- Reports execution time

---

## Data Units & Conventions

| Quantity | Unit | Notes |
|----------|------|-------|
| Position | km | Cartesian XYZ |
| Velocity | km/s | Cartesian XYZ |
| Time (internal) | ET seconds past J2000 | From eosimutils (SPICE TDB) |
| Latitude/Longitude | degrees | Geodetic (WGS84) |
| Elevation/Height | meters | Above WGS84 ellipsoid |
| FOV angles | degrees | Diameter for circular; half-angles for rectangular |
| Elevation angle (contact) | degrees | 0° = horizon, 90° = zenith |
| Mean motion (SGP4) | rev/day | Kozai mean motion |
| GM_EARTH | 398600.435507 km³/s² | |
| RCG factor | 1/km⁴ | G/(R_t² × R_r²), unity gain G=1 |

---

## Design Patterns

### Factory + Registry
Used in: `OrbitFactory`, `PropagatorFactory`, `CoverageFactory`, `ContactFinderFactory`, and eosimutils `FieldOfViewFactory`, `StandardFrameHandlerFactory`.

Pattern: a `_registry` dict on the factory class; `@register_type("name")` decorator populates it; `from_dict(specs)` dispatches on a type key (`"orbit_type"`, `"propagator_type"`, `"coverage_type"`, etc.).

### Serialization (`from_dict` / `to_dict`)
Every major class implements both. `Mission.from_dict()` additionally supports loading sub-objects from relative JSON file paths when `user_dir` is set in `Settings`.

### Facade (Mission)
`Mission` composes all sub-systems. Users call `execute_all()` for a complete mission run, or call individual `execute_*()` methods for selective analysis.

### Timeseries Inheritance
`ContactInfo` and `EclipseInfo` extend `eosimutils.timeseries.Timeseries` with a single boolean data channel, gaining interpolation and serialization from the base class.

### Coverage Indexing Duality
`DiscreteCoverageTP` (time-primary) and `DiscreteCoverageGP` (grid-primary) are complementary views of the same data. Conversion methods `from_tp()` / `to_tp()` / `from_gp()` / `to_gp()` allow switching between them. Use TP for time-loop queries, GP for per-point statistics.

### C++/Python Split
The computational hot path (point-in-FOV, LOS checks over many time steps and grid points) runs in C++ (`kcl` / CoverageKinematics). Python orchestrates setup, frame transforms, and data I/O.

---

## Typical Mission Workflow

```
1. Define Mission (JSON → Mission.from_dict())
   - Spacecraft (orbit/norad_id, sensor FOV, LVLH frame)
   - Ground stations (lat/lon/elevation constraint)
   - Coverage grid (Cartesian3DPositionArray in ITRF)
   - Propagator (SGP4 or Orekit) + Settings

2. Propagate (orbit → StateSeries)
   - SGP4Propagator.execute() → StateSeries in ICRF_EC
   - OrekitPropagator.execute() for high-fidelity scenarios
   - Optional: auto-fetch OMM from Space-Track via norad_id

3. Eclipse Finding (StateSeries → EclipseInfo)
   - EclipseFinder uses SPICE Sun position
   - Spherical shadow model (no umbra/penumbra)

4. Contact Finding (StateSeries + GroundStation → ContactInfo)
   - ElevationAwareContactFinder: LOS + elevation constraint
   - Returns boolean intervals per spacecraft/GS pair

5. Coverage Calculation (StateSeries + FOV + grid → DiscreteCoverageTP)
   - PointCoverage: horizon + FOV containment per time step
   - SpecularCoverage: GNSS-R specular reflection geometry

6. Output (→ MissionOutput.json)
   - All results serialized via to_dict()
   - Optional D-SHIELD format export
```

---

## eosimutils Dependency Summary

orbitpy-revised uses eosimutils throughout. Key imports:

| eosimutils Type | Used In | Role |
|-----------------|---------|------|
| `AbsoluteDate` / `AbsoluteDateArray` | All modules | Epoch and time grid |
| `AbsoluteDateIntervalArray` | `coverage.py`, `contactfinder.py` | Coverage/contact intervals |
| `StateSeries` / `PositionSeries` | `propagator.py`, `mission.py`, `specular.py` | Propagated trajectories |
| `CartesianState` / `Cartesian3DPosition` | `orbits.py`, `contactfinder.py` | State/position types |
| `Cartesian3DPositionArray` | `coverage.py`, `coveragecalculator.py` | Coverage grid |
| `GeographicPosition` | `resources.py`, `contactfinder.py` | Ground station location |
| `FrameGraph` | `mission.py`, all calculators | Frame transform composition |
| `LVLHType1FrameHandler` | `resources.py` | Spacecraft body frame |
| `CircularFOV` / `RectangularFOV` / `PolygonFOV` | `resources.py`, calculators | Sensor FOV |
| `ReferenceFrame` (`ICRF_EC`, `ITRF`) | Throughout | Frame identification |
| `SurfaceType` | `mission.py`, `specular.py` | WGS84 vs SPHERE |
| `WGS84_EARTH_POLAR_RADIUS` | `contactfinder.py`, `eclipsefinder.py` | LOS/shadow geometry |
| `EnumBase` | All enumerations | Enum base class |

**Important**: eosimutils `__init__.py` is empty — import directly from submodules (e.g., `from eosimutils.time import AbsoluteDate`). Same pattern applies to orbitpy: `__init__.py` is empty.

---

## Build & Development

```bash
conda env create -f environment.yml   # create environment with C++ deps
conda activate orbitpy

make install    # pip install -e .   (builds C++ CoverageKinematics extension)
make test       # coverage run -m unittest discover -s tests
make slowtest   # coverage run -m unittest discover -s slowtests (STK/GMAT validation)
make lint       # pylint --rcfile=pylintrc orbitpy tests slowtests examples
make format     # black orbitpy tests slowtests examples
make docs       # sphinx-build -b html docs/source docs/build
make coverage   # coverage report -m + HTML
```

**Build system**: scikit-build-core + pybind11 (declared in `pyproject.toml`). The C++ submodule must be initialized (`git submodule update --init --recursive`) before building.

---

## Testing

**Fast tests** (`tests/`): 6 unittest modules. Run with `make test`.

| Test Module | Coverage |
|-------------|---------|
| `test_orbits.py` | TLE, OMM, OsculatingElements, Sgp4SatrecOrbitalParameters; Cartesian↔Keplerian round-trips |
| `test_propagator.py` | PropagatorFactory, SGP4Propagator with TLE and Satrec params |
| `test_mission.py` | Full mission pipelines: single spacecraft, constellation, GNSS-R |
| `test_contactfinder.py` | LOS and elevation-aware contact finding |
| `test_eclipsefinder.py` | Eclipse interval detection |
| `test_resources.py` | GroundStation, Sensor, Spacecraft serialization; UUID defaults |

**Slow tests** (`slowtests/`): High-fidelity validation against STK and GMAT. Run with `make slowtest`. Covers coverage, eclipse finding, and contact finding against reference simulators.

**Test data**: `tests/example_omm_list.json` — sample OMM records. `examples/processed_data/` — pre-computed `.npy` arrays for CYGNSS validation.

---

## Examples

| Directory / File | Scenario |
|-----------------|---------|
| `examples/mission_simple/` | Single Landsat 9 spacecraft, 1 circular sensor, 1 GS, 3 target points |
| `examples/planet_skysat/planet_skysat.py` | Planet SkySat constellation; CONUS coverage; D-SHIELD export |
| `examples/cygnss_gnssr/cygnss_gnssr.py` | CYGNSS + GPS/Galileo GNSS-R specular coverage constellation |
| `examples/specular_coverage/` | GPS constellation OMMs + specular coverage computation |
| `examples/solve_drag_coefficient.py` | CYGNSS drag-coefficient estimation via GLSDC + Orekit (TLE-initialized state, ~3-day arc of CYGNSS reference positions) |
| `examples/solve_drag_coefficient_x0.py` | Like the above, but also solves for the initial Cartesian state; fits on the first half of the arc, evaluates over the full arc |
| `examples/cygnss_data/` | CYGNSS + GPS PRN reference trajectory CSVs (input data for the drag-coefficient examples) |
| `examples/point_coverage.py`, `specular_example.py`, `specular_rcg_example.py`, `specular_validation.py`, `orekit_propagator.py` | Standalone demonstration/validation scripts |
| `examples/spacetrack/` | SpaceTrack credentials template |
| `bin/run_mission.py` | Generic CLI runner for any `MissionSpecs.json` |
| `orbitpy/orekitpropagator.py` | (module contains usage example in docstring) |

Mission JSON configs (e.g., `MissionSpecs.json`) use the `from_dict` schema for `Mission`, and may reference external files for sub-objects using relative paths under `user_dir`.

---

## D-SHIELD Demo Example (`examples/dshield-cygnss-demo/`)

`examples/dshield-cygnss-demo/` is a **GNSS-Reflectometry (GNSS-R) specular
coverage** mission: CYGNSS receiver spacecraft observe GPS transmitter signals reflected
off Earth, coverage is evaluated over a CONUS target grid, and results are additionally
exported in the CSV format expected by the **D-SHIELD** project. It started as a copy of
`examples/cygnss_gnssr/` but has since diverged (full constellation, 1 s step, a
Grid.csv→spatial_points helper, a dynamically-set start epoch).

### Entry point (`dshield-cygnss-demo.py`)

1. Resolves all paths relative to the script (`os.path.dirname(os.path.abspath(__file__))`) and recreates a fresh `output/` directory (any existing one is deleted first).
2. Loads `MissionSpecs.json`, injects `settings.user_dir`, and builds the mission via `Mission.from_dict(...)`.
3. Runs the full pipeline with `mission.execute_all(topk=None)` (the current setting), collecting `propagator_results`, `contact_finder_results`, `eclipse_finder_results`, `coverage_calculator_results` (the GNSS-R coverage), and `specular_trajectory_results`. With **`topk=None`** both the GNSS-R coverage and the specular trajectories contain **one entry per GPS transmitter**. Passing an integer **`topk=k`** instead reduces **both**, at each time step, to only the `k` highest-RCG entries across all GPS transmitters (see `Mission.execute_gnssr_coverage_calculator` and `Mission.execute_specular_trajectory_calculator`); both are then ranked by the **same** geometric line-of-sight RCG (computed once by `execute_all`), so rank *r* of the coverage names the same GPS transmitter as rank *r* of the trajectory at every time step (see Critical Note #11).
4. Serializes the complete bundle to `output/MissionOutput.json` via `eosimutils.base.JsonSerializer`.
5. Saves the OMM actually used to propagate each satellite (receivers **and** GNSS transmitters) as one JSON file per satellite under `output/omm/` — after `execute_all()`, each spacecraft's `orbit` attribute holds the OMM auto-retrieved from Space-Track (`propagate_spacecraft` stores it back on the `Spacecraft`).
6. Converts each result set to the D-SHIELD CSV format and writes it to `output/`, using helpers imported from a sibling `dshield_format_converter` module (the example's parent `examples/` dir is added to `sys.path`): `write_dshield_format_of_propagator_results`, `..._contact_results`, `..._eclipse_results`, `..._gnssr_coverage_results`, `..._specular_trajectory_results`.
7. Copies the entire `output/` folder to `/home/ubuntu/dshield-2026-demo/orbits/output/<YYYYMMDD>` (folder named after the epoch date; any existing copy for that date is replaced), then deletes `MissionOutput.json` from the copy — the D-SHIELD demo consumes only the CSV exports.

The module docstring records the NORAD IDs of the 8 CYGNSS satellites and the GPS transmitter NORAD IDs (reference data; Galileo IDs have been removed since this example is GPS-only). The docstring's `gps_sat_norad_ids` list mirrors `gnss_spacecrafts.json` — keep it in sync when editing that file.

> **GPS-only transmitters**: This example deliberately considers **only GPS** GNSS
> transmitters — **Galileo is excluded**. The active `gnss_spacecrafts.json` lists GPS
> satellites only; do not add Galileo entries here.

### Coverage output shape (topk=None vs topk=k)

With `topk=None` (the current setting) `coverage_calculator_results` uses the
all-transmitters shape: each sensor's `total_sensor_coverage` holds **one entry per GPS
transmitter**, with scalar `gnss_spacecraft_id` / `gnss_spacecraft_name`. If the demo is
changed to pass `topk=k`, each sensor's `total_sensor_coverage` instead holds **k rank
entries** (not one per transmitter). Each rank entry has `rank` (1 = highest RCG),
`coverage_info` (`DiscreteCoverageTP`), `rcg_factor`, and — because the selected
transmitter varies per time step — `gnss_spacecraft_id` / `gnss_spacecraft_name` as
**per-time-step lists**. `write_dshield_format_of_gnssr_coverage_results` handles both
shapes: it normalizes the source id/name to per-time-step lists (broadcasting the scalar
in the non-top-k case), so each CSV row's "source id" is the transmitter selected at that
specific time step.

`specular_trajectory_results` follows the same pattern: with `topk` set, each receiver's
`all_spacecraft_specular_info` holds **k rank entries** (not one per transmitter), each
with `rank`, `specular_info` (`PositionSeries`), `rcg_factor`, and per-time-step
`gnss_spacecraft_id` / `gnss_spacecraft_name` lists.
`write_dshield_format_of_specular_trajectory_results` normalizes the id/name the same way,
so each CSV row's "source id" names the transmitter whose specular point won that rank at
that time step.

### Mission spec (`MissionSpecs.json`)

- **start_time**: **not present in `MissionSpecs.json`** — set dynamically by `dshield-cygnss-demo.py` to *(current UTC date + 1 day)* at `00:00:00.00` UTC
- **duration_days**: 1.0 (→ 86,401 time steps at the 1 s step)
- **propagator**: `SGP4_PROPAGATOR`, **1 s** step size
- **spacecrafts** → `cygnss_spacecrafts.json` (receivers)
- **gnss_spacecrafts** → `gnss_spacecrafts.json` (GPS transmitters)
- **ground_stations** → `ground_stations.json`
- **spatial_points** → `geographic_array` → `spatial_points.json` (coverage grid)
- **settings**: `coverage_type = SPECULAR_COVERAGE`, `specular_radius_km = 25.0`, `spacetrack_credentials_relative_path = ../spacetrack/credentials.json`

Sub-objects are pulled from external JSON files via the **`relative_file_path`** key, resolved against `user_dir`.

### Supporting data files

- `cygnss_spacecrafts.json` — **7 CYGNSS receivers** (NORAD 41884, 41885, 41886, 41887, 41888, 41890, 41891 — i.e. the 8-sat constellation minus the defunct CYGFM06); each has a CIRCULAR FOV of 107.35° diameter in its `LVLH_TYPE_1` frame with nadir boresight `[0,0,1]`; orbits auto-fetched by `norad_id`.
- `gnss_spacecrafts.json` — **32 GPS transmitters** (by `norad_id`), no instrument. Tracks the CelesTrak `gps-ops` operational constellation, maintained by hand (the script docstring records the update history — e.g. 26360 and 24876 removed, 68791 added). **Galileo is intentionally excluded** from this example.
- `ground_stations.json` — 3 ground stations: `gs1` (id `HI`, Hawaii), `gs2` (id `CHI`, Chile), `gs3` (id `AUS`, Australia), each `min_elevation_angle` 7°.
- `spatial_points.json` — `{"geo_positions": [...]}` with **114,454** `[lat, lon, 0]` points forming a CONUS grid (first point `[39.989, −128.539]`). Verified to match `Grid.csv` row-for-row (same order, GP index = array position, height 0 added).
- `Grid.csv` — source grid (`GP index, lat [deg], lon [deg]`) from which `spatial_points.json` is generated. Currently **byte-identical** (same MD5) to the dated grid `Grid_WLFP_20260608.csv`, which lives outside the repo at `/home/ubuntu/dshield-2026-demo/pre-fire-priority/`.
- `convert_grid_to_spatial_points.py` — helper that converts `Grid.csv` → `spatial_points.json` (`{"geo_positions": [[lat, lon, 0], ...]}`). Run it (no args) after editing `Grid.csv` to regenerate the grid.

### Running

```bash
cd examples/dshield-cygnss-demo
python dshield-cygnss-demo.py
```

Outputs land in `examples/dshield-cygnss-demo/output/`: `MissionOutput.json`, one OMM
JSON per satellite under `omm/`, and one folder per receiver spacecraft holding the
D-SHIELD CSVs:

```
output/<receiver_name>/
├── propagation/state.csv
├── ground_contact/<ground_station_name>   # one file per station, no extension
├── eclipse/eclipse                        # no extension
├── access/DDMI.csv                        # single-sensor case; sensor<N>_access.csv if multiple
└── specular/specular.csv
```

The script then copies `output/` to `/home/ubuntu/dshield-2026-demo/orbits/output/<YYYYMMDD>`
(minus `MissionOutput.json`) — see entry-point step 7. Within the example dir the script
only writes to `output/`.

### Notes / dependencies

- Orbits are auto-fetched from Space-Track by `norad_id`, so valid credentials are needed at `examples/spacetrack/credentials.json` (per the spec's `spacetrack_credentials_relative_path`).
- The D-SHIELD CSV export depends on `dshield_format_converter`, which the script adds to `sys.path` as the **parent** of the example folder; the example must therefore live directly under `examples/` for that import to resolve.

---

## Critical Notes

1. **`__init__.py` is empty** — import directly from submodules (e.g., `from orbitpy.mission import Mission`).

2. **C++ submodule must be initialized**: Run `git submodule update --init --recursive` before `make install`. The `kcl` Python module is built from `extern/CoverageKinematics`.

3. **Specular coverage requires ITRF states**: `get_specular_trajectory()` and `SpecularCoverage.calculate_coverage()` require transmitter and receiver `StateSeries` in the **ITRF** frame. The `FrameGraph` can convert from ICRF_EC.

4. **Orekit requires `begin()` call**: `OrekitPropagator.begin()` initializes the Java VM and force models. It must be called before `execute()`.

5. **Space-Track credentials file**: JSON file with `"username"` and `"password"` keys. Path specified as `spacetrack_credentials_relative_path` in `Settings` (relative to `user_dir`).

6. **Coverage indexing**: `DiscreteCoverageTP.coverage[t]` → grid indices accessed at time `t`. `DiscreteCoverageGP.coverage[gp]` → time indices when grid point `gp` was accessed. Both are zero-indexed.

7. **SPICE kernels**: Loaded automatically when `StateSeries`/`PositionSeries` from eosimutils are constructed (downloads to `~/.eosimutils_spice/` on first use). `EclipseFinder` also loads kernels directly.

8. **LOS uses polar radius**: Both `check_line_of_sight()` in `utils.py` and `EclipseFinder` use `WGS84_EARTH_POLAR_RADIUS` (not equatorial) to conservatively avoid false positives near the Earth limb.

9. **SGP4 frame convention**: SGP4 propagation (via Skyfield) outputs positions in GCRS, which orbitpy maps to eosimutils `ICRF_EC`. This is an approximation — GCRS ≈ ICRF_EC for Earth satellites.

10. **RCG factor units**: The Range-Corrected Geometry factor has units 1/km⁴. When comparing across scenarios, ensure distances are in km (consistent with eosimutils position units).

11. **Top-k rankings are aligned by a shared geometric RCG (specular trajectory vs. coverage)**: Both `specular_trajectory_results` and `coverage_calculator_results` are ranked by the **same** physical quantity — the geometric, line-of-sight RCG at the specular point (`G/(R_t²·R_r²)`). The receiver FOV and specular radius shape only *which grid points count as covered* (the `coverage_info` payload); they do **not** enter the ranking RCG (`RCGSource` in `coveragecalculator.py` takes only tx/rx positions + the specular point). `execute_all(topk=k)` computes the geometric RCG once (`Mission._specular_data_by_rx`) and feeds it as the single ranking key to both reductions — to the trajectory directly, and to coverage via `execute_gnssr_coverage_calculator(..., ranking_rcg=...)`. As a result, rank *r* of one output names the same transmitter as rank *r* of the other at every time step. **Caveat:** calling the two `execute_...` methods individually (outside `execute_all`) ranks each by its own specular-point solver (Python Newton vs. C++ `SpecularPointSource`), so selections can still differ at a near-tie — use `execute_all` when aligned top-k outputs are required. (This shares one specular computation; coverage still computes its own specular points in C++ for the coverage geometry, so there is no redundant work.)

12. **C++ `Variable`/`Source` `update()` must be reentrant (no mutable scratch members)**: `driveCoverage` (`CoverageDriver.hpp`) calls each `Variable::update(time_idx)` concurrently across time indices via `tbb::parallel_for`, on the **same** source object. Any `update()` that writes to a **member** other than its own per-index `data[i]`/`status[i]` slot is a data race. This previously bit `LOSEventSource::update()`, which mutated member `ray`/`query` objects → corrupted line-of-sight results → NaN specular points → **missing rows** in the GNSS-R specular output (non-deterministic, ~18% of LOS points under load). Fixed by making `ray`/`query` **local** to `update()`. When adding a new source, keep all scratch state local. *(Requires rebuilding the `kcl` extension — `make install` — for changes to take effect; verify the build is newer than the edit before trusting any test.)*

13. **SGP4 propagation time grid needs sub-second precision for sub-second steps**: `SGP4Propagator.execute` formats its time grid with `utc_strftime("%Y-%m-%dT%H:%M:%S.%f")`. The `.%f` is required: at a non-whole-second `step_size` a second-precision format quantizes grid points onto the same second, so downstream `round((t-epoch)/step)` index mapping skips/duplicates indices. Whole-second steps (e.g. the demo's 1 s) are unaffected.
