# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-04-09 — PySALMO

### Added — inSALMO Parity (Milestone 1)
- LifeStage IntEnum with 7 stages (FRY through RETURNING_ADULT)
- Adult holding behavior — anadromous spawners skip feeding via `skip_indices`
- Two-piece condition-survival (broken-stick starvation function)
- Stochastic outmigration probability (inSALMO style, opt-in config flag)
- Outmigrant date recording for virtual screw-trap output
- Stochastic spawn-cell perturbation (noise term on suitability scores)
- Modified growth fitness term `ln(1+L/Lmax)` (inSALMO)
- Migration as 4th activity in habitat selection (competing with drift/search/hide)
- 4x habitat selection passes per day for anadromous species (dawn/day/dusk/night)
- Per-substep resource regeneration between habitat selection passes
- `survival^step_length` fitness exponentiation (correct for sub-daily)
- Forward condition-survival projection (`mean_condition_survival`) matching NetLogo
- Horizon-projected fitness for migration comparison
- inSALMO parity validation suite with NetLogo 7.4 reference data

### Added — Marine Extension (Milestone 2)
- Marine growth module (O'Neill 1986 temperature function, Hanson et al. CMax)
- 7 marine mortality sources (seal, cormorant, background, temperature, M74, fishing harvest, bycatch)
- Gear-specific fishing (logistic/normal selectivity, 4 gear types, seasonal/zone closures)
- Smoltification module (photoperiod + temperature triggers, seasonal window gating)
- Maturation check with sea-winter conditional probabilities (Baltic distribution)
- Marine zone migration with configurable zone graph
- MarineDomain orchestrating marine step (growth, survival, fishing, maturation)
- MarineSpace with zone connectivity and ZoneState
- StaticDriver for environmental data from YAML seasonal tables
- MarineConfig, ZoneConfig, GearConfig Pydantic models
- TroutState marine fields (zone_idx, sea_winters, smolt_date, natal_reach_idx, smolt_readiness)
- Smolt transition at river mouth (PARR -> marine instead of death)
- Full lifecycle integration test (smolt -> ocean -> return -> spawn)
- Performance parity test suite (Python vs NetLogo 7.4)

### Changed
- Project renamed from inSTREAM-py to PySALMO
- Version bumped to 1.0.0
- CLI entry point: `pysalmo` (+ `instream` alias)
- Emerged anadromous fish assigned PARR (not FRY) to enable migration
- `example_a.yaml` now includes `adult_arrival_file` and `population_file`
- Replaced all life_history magic numbers with LifeStage enum
- Extended `_LIFE_HISTORY_COLORS` for 7 stages in Shiny frontend

### Fixed
- Committed previously untracked `arrival_reader.py` and `help_panel.py`
- Added `pytest.importorskip("meshio")` for optional dependency tests
- Regenerated fitness-golden.csv reference data for 4x substep mode

## [0.11.0] - 2026-04-05

### Added
- Angler harvest module with size-selective mortality, bag limits, and CSV schedule (`modules/harvest.py`)
- Morris one-at-a-time sensitivity analysis framework (`modules/sensitivity.py`)
- Config-driven habitat restoration scenarios (cell property modification at scheduled dates)
- Fitness memory (exponential moving average) for smoother habitat selection decisions
- Drift regeneration distance blocking for cells near drift-feeding fish
- Spawn defense area exclusion for new redd placement
- YearShuffler wiring for stochastic multi-year time series remapping
- Anadromous adult life history transitions and post-spawn mortality
- Habitat summary and growth report output types (7 total output writers)
- SpeciesParams completed with all ~90 species parameter fields
- Cross-backend parity tests for survival, spawn_suitability, evaluate_logistic
- `InSTREAMModel` now accepts `ModelConfig` objects directly (enables programmatic config)

### Fixed
- Migration now uses per-species `migrate_fitness_L1/L9` instead of species_order[0]
- Solar irradiance uses daily-integral formula instead of overestimating noon-elevation
- Beer-Lambert light attenuation includes `light_turbid_const` additive term
- Superindividual split uses per-species `superind_max_length` threshold
- Numba `evaluate_logistic` supports array L1/L9 parameters

### Performance
- Vectorized survival computation in NumPy backend (replaces 80-line per-fish loop)
- Implemented `survival`, `growth_rate`, `spawn_suitability`, `deplete_resources` in all 3 backends
- Survival loop in model.py replaced with single `backend.survival()` dispatch

### Infrastructure
- 674 tests (was 499), 11/11 validation tests passing
- Gap-closure design spec and reviewed implementation plans

---

## [0.10.0] - 2026-03-23

### Added
- Add /deploy skill for laguna.ku.lt Shiny Server deployment
- Add main Shiny app entry point with sidebar, tabs, and extended_task
- Add spatial panel module (shiny_deckgl map + matplotlib fallback)
- Add population panel module (plotly line chart)
- Add simulation wrapper with config overrides and results collection

### Changed
- Update README with Shiny frontend, JAX backend, FEM mesh in completed features
- Add frontend optional-dependencies (shiny, plotly, shiny-deckgl)

---

## [0.9.0] - 2026-03-22

### Added
- Add automated release script, replace bump_version.py
- Implement JAX vectorized growth_rate and survival kernels

---

## [0.8.0] - 2026-03-22

### Added
- JAX compute backend: `update_hydraulics`, `compute_light`, `compute_cell_light`, `evaluate_logistic`, `interp1d` implemented with `jax.vmap` vectorization
- FEM mesh reader (`space/fem_mesh.py`): reads triangular meshes via meshio (River2D .2dm, GMSH .msh, and all meshio-supported formats)
- FEM mesh computes centroids, areas, and edge-based adjacency from element connectivity
- 7 new tests: JAX backend cross-validation against NumpyBackend, FEM mesh reading/area/adjacency

### Changed
- `get_backend("jax")` now returns a working JaxBackend (was NotImplementedError)
- FEM mesh areas automatically convert m^2 to cm^2

---

## [0.7.0] - 2026-03-22

### Added
- **InSTREAM-SD sub-daily scheduling**: multiple habitat selections per day with variable flow
- Auto-detection of input frequency (hourly, 6-hourly, daily) from time-series timestamps
- SubDailyScheduler with row-pointer advancement, is_day_boundary, substep_index
- Partial resource repletion between sub-steps (drift + search food regeneration)
- Growth accumulation in memory arrays, applied once at day boundary
- Solar irradiance cached per day, cell light recomputed each sub-step (depth-dependent)
- Synthetic hourly and peaking fixture data for testing
- 10 new sub-daily integration tests + 2 daily regression tests
- GitHub Actions CI pipeline

### Changed
- model.step() restructured into sub-step operations (every step) and day-boundary operations (end of day)
- TroutState.max_steps_per_day auto-sized from detected input frequency
- Survival applied each sub-step with `** step_length` scaling
- Spawning, redd development, census, age increment only at day boundaries

### Fixed
- Substep index off-by-one in TimeManager (solar cache population)
- Growth accumulation count at day boundary

---

## [0.6.0] - 2026-03-22

### Added
- All 11 validation tests now active (was 0 at v0.1.0, 5 at v0.3.0)
- 6 new golden-snapshot reference CSVs: CStepMax, growth report, trout survival, redd survival, spawn cell suitability, fitness snapshot
- Reference data generator covers all ecological modules (growth, survival, spawning, fitness)

### Validation
- 11/11 tests passing: GIS, depths, velocities, day length, CMax interpolation, CStepMax, growth report, trout survival, redd survival, spawn cell, fitness
- Golden snapshots from Python v0.5.0 — ready for cross-validation against NetLogo when available
- Note: NetLogo not installed; reference data computed by Python implementation itself (regression guard)

---

## [0.5.0] - 2026-03-22

### Added
- Output writer module (`io/output.py`) with 6 file types: census, fish/redd/cell snapshots, outmigrants, summary
- `write_outputs()` method on InSTREAMModel, called automatically at end of `run()`
- Redd superimposition: existing redd eggs reduced when new redd placed on same cell
- Working CLI: `instream config.yaml --output-dir results/ --end-date 2012-01-01`
- 8 new tests (output writers, superimposition)

### Changed
- `create_redd()` returns slot index (not bool) for superimposition support
- `run()` now calls `write_outputs()` at completion
- CLI uses argparse with config, --data-dir, --output-dir, --end-date, --quiet

---

## [0.4.0] - 2026-03-22

### Added
- Multi-reach support: per-reach hydraulic loading, light computation, resource reset
- Multi-species support: per-fish species parameter dispatch via pre-built arrays
- Per-fish reach-based temperature, turbidity, and intermediate lookups
- Multi-species initial population loading from CSV
- Per-reach per-species spawning and redd development
- Example B config (3 reaches x 3 species) generated from NLS
- Example B integration tests (init + 10-day + 30-day runs)
- Case-insensitive shapefile column name resolution

### Changed
- model.py no longer hardcodes reach_order[0] or species_order[0]
- Habitat selection uses per-fish species/reach parameters
- Survival loop uses per-fish species/reach mortality parameters
- Piscivore density computed with per-species length threshold

---

## [0.3.0] - 2026-03-22

### Added
- Numba brute-force candidate search replacing KD-tree queries (136ms -> 15ms)
- Sparse per-fish candidate lists replacing dense (2000, 1373) boolean mask
- 5 analytical validation tests activated (day length, CMax interpolation, GIS, depths, velocities)
- Reference data generator script (`scripts/generate_analytical_reference.py`)
- Migration wired into model.step() with reach graph construction
- Census day data collection wired into model.step()
- Adult arrivals stub (ready for multi-reach/species)

### Performance
- Full step: 179ms -> 98ms (1.8x faster, 633x vs original)
- Candidate mask build: 136ms -> 15ms (9x faster via Numba brute-force)
- Estimated full 912-day run: 2.2 min -> 1.4 min
- Now within 2-3x of NetLogo performance (was 130x slower at v0.1.0)

### Validation
- 5/11 validation tests now active (was 0/11)
- Tests use analytically computed reference data (no NetLogo dependency)

---

## [0.2.0] - 2026-03-22

### Added
- Survival-integrated fitness function (Phase 5)
- Piscivore density computed from fish distribution
- Annual age increment on January 1
- Numba @njit compiled fitness evaluation kernel (60x speedup)
- Hypothesis property-based tests
- Performance regression tests
- Population file configurable via YAML
- `netlogo-oracle` skill for validation data generation
- `validation-checker` agent for implementation completeness

### Fixed
- Random sex assignment for initial population and emerged fish
- `spawned_this_season` reset at spawn season start
- Growth stored during habitat selection (not recomputed from depleted resources)
- Deferred imports moved to module level
- Condition factor updated after spawning weight loss
- Division-by-zero guards on all logistic functions
- Clamped max_swim_temp_term >= 0, resp_temp_term overflow guard
- Zombie fish at weight=0 (condition=0 now lethal)
- Egg count rounding (np.round instead of truncation)
- Negative velocity and egg development clamped to non-negative
- Survival probability raised to step_length power
- Redd survival uses logistic^step_length formula
- Logistic exp() argument clipped to prevent overflow

### Changed
- `evaluate_logistic` uses math.exp instead of np.exp (2.7x faster)
- `survival_condition` uses min/max instead of np.clip (55x faster)
- `cmax_temp_function` uses bisect instead of np.interp (4.8x faster)
- Habitat selection inner loop pre-computes step/fish invariants
- Vectorized hydraulic interpolation using searchsorted + lerp
- Batch redd emergence (eliminates O(eggs*capacity) scan)
- Integer activity codes in growth and survival functions

### Performance
- Full step: 62s -> 179ms (346x faster)
- Full 912-day run: ~129 min -> ~2.1 min (61x faster)

## [0.1.0] - 2026-03-20

### Added
- Initial Python implementation of inSTREAM 7.4
- Mesa 3.x model orchestration
- SoA state containers (TroutState, CellState, ReddState, ReachState)
- FEMSpace with KD-tree spatial queries
- Wisconsin bioenergetics (growth, consumption, respiration)
- 5 survival sources (temperature, stranding, condition, fish predation, terrestrial predation)
- Spawning, egg development, redd emergence
- Migration framework
- YAML configuration with NLS converter
- NumPy and Numba compute backends
- 376 unit tests
