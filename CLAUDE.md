# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Oceanostics.jl is a companion Julia package for [Oceananigans.jl](https://github.com/CliMA/Oceananigans.jl) that provides diagnostic quantities for ocean simulations. It computes terms from governing equations (tracer, kinetic energy, turbulent kinetic energy, tracer variance, potential energy budgets) and flow diagnostics (Richardson number, Rossby number, Ertel potential vorticity, strain rate, etc.).

## Build and Test Commands

```bash
# Run all tests
julia --project -e 'using Pkg; Pkg.test()'

# Run a specific test group (much faster for development)
TEST_GROUP=vel_diagnostics julia --project -e 'using Pkg; Pkg.test()'
```

Available TEST_GROUP values: `vel_diagnostics`, `tracer_diagnostics`, `u_momentum_diagnostics`, `v_momentum_diagnostics`, `w_momentum_diagnostics`, `ke_diagnostics`, `filtered_ke_diagnostics`, `subfilter_ke_diagnostics`, `tke_diagnostics`, `pe_diagnostics`, `ape_diagnostics`, `filtered_ape_diagnostics`, `subfilter_ape_diagnostics`, `active_tracer_diagnostics`, `tracer_variance_diagnostics`, `general_flow_diagnostics`, `canonical_flows`, `progress_messengers`, `spatial_filters`, `perf_invariants`.

```bash
# Instantiate/build the package
julia --project -e 'using Pkg; Pkg.instantiate()'

# Build documentation
julia --project=docs docs/make.jl
```

## Architecture

### Core Pattern: KernelFunctionOperation Wrappers

Every diagnostic is built on Oceananigans' `KernelFunctionOperation` (KFO). The pattern is:

1. Define an `@inline` kernel function that computes values at grid point `(i, j, k)` using Oceananigans operators (interpolation `ℑ`, derivatives `∂`, etc.)
2. Define a `const` type alias via `CustomKFO{F}` (a parametric alias for `KernelFunctionOperation` parameterized on the kernel function type)
3. Provide constructor(s) that accept a `model` (or individual fields) and return a `KernelFunctionOperation`

All kernel functions use Oceananigans' staggered grid conventions with location triplets like `(Center, Center, Center)` or `(Face, Face, Face)`. Location suffixes on kernel function names (e.g., `_ccc`, `_ccf`, `_fff`) indicate the grid location where the result lives.

### Module Structure

- **`Oceanostics`** (main module in `src/Oceanostics.jl`): Shared utilities — `validate_location`, `validate_dissipative_closure`, `add_background_fields`, `perturbation_fields`, `get_coriolis_frequency_components`, viscosity helpers for closure tuples (`_νᶜᶜᶜ`)
- **`TracerEquation`**: Advection, Diffusion, ImmersedDiffusion, TotalDiffusion, Forcing terms
- **`UMomentumEquation` / `VMomentumEquation` / `WMomentumEquation`**: Per-component momentum-budget terms (advection, stress, pressure gradient, Coriolis, buoyancy, forcing). Tested as separate `*_momentum_diagnostics` groups.
- **`SpatialFilters`** (submodule): Spatial filters (`box_filter.jl`, `gaussian_filter.jl`) for diagnostics that need scale separation. Every 1D kernel sizes its in-range check with `stencil_length(grid, d, ψ)` (the operand's own extent, `N+1` for a `Face` location along a `Bounded` direction) rather than `size(grid, d)`; the recursive (fused) methods reach the operand through `fargs[end]`. Using the cell count there silently mistreats the last face of a `Face`-located operand (`:shrink` drops its own weight, `:edge` clamps to the face below), which surfaced as `NaN`s when filtering the model's `diffusive_flux_z` KFO with a degenerate identity-scale Gaussian. That extent must be measured **on the host** and carried into the kernel by the boundary policy (`SizedBoundary`), never recomputed inside it: on a GPU the kernel's `ψ` is the *adapted* operand, which has no location, so `location(ψ)[d]` is `Nothing` and Oceananigans' `length(::Nothing, topo, N)` collapses it to `1`. That made every filtered direction one cell long, and failed silently in two directions at once — `wrap_periodic_index(i, 1) == i-1` displaced the whole field by a cell per filtered dimension, while `ShrinkBoundary` kept only the tap at index 1. Constant fields are immune to both, which is why the identity-scale checks in the filtered/sub-filter suites never caught it and only real GPU hardware did.
- **`KineticEnergyEquation`**: KE, its tendency, advection, stress, forcing, pressure redistribution, buoyancy production, dissipation rate (general and isotropic)
- **`FilteredKineticEnergyEquation`**: Filtered KE budget terms — `FilteredKineticEnergy` (Kˡ = ½ūᵢūᵢ, KE of the filtered flow; reuses `KineticEnergyEquation`'s `kinetic_energy_ccc` kernel), `subfilter_stress_tensor` (τⁱʲ = filter(uⁱuʲ) − ūⁱūʲ), `KineticEnergyCrossScaleFlux` (Πₖ = −τⁱʲS̄ⁱʲ, Aluie et al. 2018), and `FilteredKineticEnergyDissipationRate` (εˡ, dissipation of the filtered flow; kernel `filtered_dissipation_rate_ccc`). Built on `FlowDiagnostics`' `StressTensor`/`StrainRateTensor` and the `Filters` submodule, so it is included after both.
- **`SubFilterKineticEnergyEquation`**: Sub-filter KE budget terms — `SubFilterKineticEnergy` (Kˢ = ½τⁱⁱ, computed as `filter(K) − Kˡ` from `KineticEnergy` and `FilteredKineticEnergy`, which share the same interpolate-the-square discretization, so the discrete decomposition `filter(K) = Kˡ + Kˢ` holds exactly by construction on any grid) and `SubFilterKineticEnergyDissipationRate` (εˢ = filter(ε) − εˡ). Both are `KernelFunctionOperation`s wrapping the underlying composite op (à la `KineticEnergyCrossScaleFlux`). Also re-exports `KineticEnergyCrossScaleFlux` (a source term of this budget). Built on `FilteredKineticEnergyEquation` and `KineticEnergyEquation`, so it is included after both.
- **`TurbulentKineticEnergyEquation`**: TKE, isotropic dissipation, shear production rates (X/Y/Z and total)
- **`TracerVarianceEquation`**: Tendency, dissipation rate, diffusion of tracer variance
- **`PotentialEnergyEquation`**: Potential energy for BuoyancyTracer, linear/nonlinear SeawaterBuoyancy
  (`PotentialEnergy`, Eₚ = -bz) plus the terms of its budget. The `-z ×` forms of the buoyancy
  equation's own terms are `Tendency` (-z∂ₜb, off Oceananigans' `tracer_tendency`),
  `BuoyancyAdvection` (z∂ⱼ(uⱼb), total velocities), `BuoyancyDiffusion` (z∂ⱼqⱼ) and `Forcing` (-zFᵇ);
  those three sum to `Tendency` *exactly*, since it is the model's own tendency taken apart. Pulling z
  inside each derivative splits them into a transport and a conversion: `Advection` (∂ⱼ(uⱼeₚ)) with
  `PotentialToKineticEnergyConversion` (wb), and `Diffusion` (∂ⱼ(zqⱼ)) with
  `DiffusiveVerticalBuoyancyFlux` (Φ = κ∂b/∂z = -q₃). The two transports are built as genuine flux
  divergences, so each integrates to zero to roundoff, which is what leaves `∫BuoyancyAdvection = -∫wb`
  and `∫BuoyancyDiffusion = ∫Φ` — those hold only to truncation error. The budget terms need
  `BuoyancyTracer` and a `NegativeZDirection` gravity, both checked at construction via
  `validate_buoyancy_is_a_tracer` and `validate_gravity_is_z_aligned`; the latter is shared with
  `BackgroundPotentialEnergyEquation` and `AvailablePotentialEnergyEquation`, whose model constructors
  inherit it through `reference_height` and whose `(model, z✶)` forms call it directly, since a `z✶`
  built from a bare `Field` carries no model to check. `DiffusiveVerticalBuoyancyFlux` and
  `PotentialToKineticEnergyConversion` are deliberately exempt: neither depends on the gravity
  direction, the first being a genuine vertical flux and the second the full uᵢbᵢ contraction. A
  `BackgroundField` buoyancy adds a term z∂ⱼ(uⱼB) that has no diagnostic yet, so the split closes only
  without one — `Tendency` still includes it, since it comes off the model's own kernel. The exact split
  and the two integral identities are covered only by `test_pe_diagnostics.jl`;
  `docs/examples/baroclinic_adjustment.jl` closes the *integrated* budget, which needs only the two
  conversions (`PotentialToKineticEnergyConversion` and `DiffusiveVerticalBuoyancyFlux`), so it does not
  exercise the `-z ×` terms. That example is a double-front run whose buoyancy is periodic and whose
  `∫eₚ dV` is therefore finite — a single Eady front is not a valid test, since its uniform background
  gradient makes the domain's potential energy infinite. It owns the buoyancy-formulation type
  aliases (`BuoyancyTracerModel`, `BuoyancyLinearEOSModel`, `BuoyancyBoussinesqEOSModel`, …) and
  `validate_gravity_unit_vector`, which `AvailablePotentialEnergyEquation` imports.
- **`BackgroundPotentialEnergyEquation`**: the reference state and `BackgroundPotentialEnergy`
  (E_b = -bz✶). The reference height z✶ comes from `reference_height`, which returns a `Field` whose
  operand is a `SortedReferenceState` rather than a `KernelFunctionOperation`: building the reference
  state is a whole-domain operation, so it hooks into `compute!` the way `Oceananigans.Fields.Scan`
  does for `Integral`/`Average`, and is rebuilt on every `compute!` so the diagnostic tracks the flow
  when written out during a simulation. The `method` keyword picks one of four
  `AbstractReferenceHeightMethod`s, which agree on every volume integral: `ThreeDimensionalSort`
  (default; each cell gets its own slot, so tied cells spread over a grid cell), `HeavisideIntegral`
  (Winters eq. 11, tied cells share their layer's mid-height, so z✶ is a function of buoyancy alone,
  local maps are clean, and it is the only method that accepts a stretched grid), `ProfileLookup`
  (matches each cell to a sorted profile by buoyancy, so the profile need not come from the field being
  diagnosed), and `VerticalSort` (the sorted column on its own `1×1×N` grid of equal-volume cells).
  Only the last three actually sort: `ProfileLookup` handed an external profile does not.
  `reference_buoyancy(z✶)` returns the buoyancy that pairs with z✶ cell by cell, which is the sorted
  profile b✶ under `VerticalSort` and the model's own buoyancy under the model-grid methods. It also
  owns `Ψ = ∫b✶dz̃` (`reference_potential`), a property of the reference profile that
  `AvailablePotentialEnergy` consumes. No method runs on an `ImmersedBoundaryGrid` yet. Built on
  `PotentialEnergyEquation`, so it is included after it
- **`AvailablePotentialEnergyEquation`**: the other half of the Winters et al. (1995) split,
  `AvailablePotentialEnergy` (Eₐ), computed in the local Holliday & McIntyre (1981) form
  `∫[b✶(z̃) - b]dz̃`, which is non-negative everywhere rather than only in the integral. Built on
  `BackgroundPotentialEnergyEquation`, so it is included after it, and it re-exports
  `reference_height`, `reference_buoyancy` and the four reference-height methods so either module can
  be used on its own
- **`FilteredAvailablePotentialEnergyEquation`**: the APE of the *filtered* buoyancy and its
  dissipation — `FilteredAvailablePotentialEnergy` (Eₐˡ = eₐ(b̄, z), the local APE kernel evaluated on
  the reference height of b̄ = filter(b); its kernel `filtered_ape_ccc` just forwards to
  `local_ape_ccc` so it gets its own type alias/display, the way `FilteredKineticEnergy` wraps
  `kinetic_energy_ccc`) and `FilteredAvailablePotentialEnergyDissipationRate` (εₐˡ = −q̄ᵢ∂ᵢΥˡ, the
  closure's diffusive flux *low-pass filtered* — the same filtered-flux choice
  `FilteredKineticEnergyDissipationRate` makes for the viscous flux, exact for constant κ — against the
  displacement potential Υˡ of b̄). Both take `(model, filter; method, geopotential_height)` and a
  low-level form on a prebuilt `z✶ˡ` (`FilteredAvailablePotentialEnergy(model, z✶ˡ)`,
  `FilteredAvailablePotentialEnergyDissipationRate(model, filter, z✶ˡ; upsilon)`) so one lookup / one Υˡ
  can be shared. b̄ is measured against a profile it did not produce (ordinarily the full field's), so
  `method` must be a `ProfileLookup`; `shared_profile_lookup` resolves the default `ProfileLookup()`
  into a `VerticalSort` column of the model's buoyancy (re-sorted every `compute!`), and
  `filtered_buoyancy_and_lookup` returns `(b, b̄, lookup)` for the sub-filter module to build the full
  field's `z✶` against the same lookup. Built on `AvailablePotentialEnergyEquation` and
  `SpatialFilters`, so it is included after both
- **`SubFilterAvailablePotentialEnergyEquation`**: sub-filter APE budget terms —
  `SubFilterAvailablePotentialEnergy` (Eₐˢ = filter(eₐ) − eₐˡ) and
  `SubFilterAvailablePotentialEnergyDissipationRate` (εₐˢ = filter(εₐ) − εₐˡ), each the filtered
  full-field quantity minus the `FilteredAvailablePotentialEnergyEquation` one on the same shared
  profile (`subfilter_reference_heights` builds `z✶` and `z✶ˡ` from that module's
  `filtered_buoyancy_and_lookup`, which is what makes the difference a decomposition). Both are
  `KernelFunctionOperation`s wrapping the underlying `BinaryOperation` (à la
  `SubFilterKineticEnergyDissipationRate`), and the module re-exports `FilteredAvailablePotentialEnergy`
  and `FilteredAvailablePotentialEnergyDissipationRate` (as `SubFilterKineticEnergyEquation` re-exports
  `KineticEnergyCrossScaleFlux`). `method` must be a `ProfileLookup`: the default builds a
  `VerticalSort` column of the model's buoyancy, `ProfileLookup(z✶_column)` shares an existing column,
  and `ProfileLookup(b✶, z✶)` with arrays freezes the reference and makes the diagnostics sort-free.
  eₐ is convex in b, so Eₐˢ ≥ 0 pointwise for filters with no vertical component (Jensen); vertical
  filtering (and the nearest-class fallback for buoyancies off the profile) can produce locally
  negative values. An identity-scale filter (σ ≪ Δx, N=3) makes the filtered diagnostics equal the
  full-field ones and the sub-filter ones vanish, all to the bit, which
  `test_filtered_ape_diagnostics.jl` / `test_subfilter_ape_diagnostics.jl` use to check the filtered
  kernels against the full-field ones without reimplementation. Built on
  `FilteredAvailablePotentialEnergyEquation`, so it is included after it (currently last of the
  equation modules)
- **`FlowDiagnostics`**: Richardson/Rossby numbers, Ertel/ThermalWind potential vorticity, strain rate & vorticity tensor moduli, Q-criterion, `subfilter_covariance` (generalized subfilter covariance `τ(a,b) = filter(a·b) − filter(a)·filter(b)`, unifying subfilter tracer flux and momentum stress), MixedLayerDepth, BottomCellValue
- **`ProgressMessengers`** (submodule): Composable simulation progress reporters using `+` (comma-separated) and `*` (concatenation) operators

### Key Dependencies

- **Oceananigans.jl**: The ocean simulation framework — provides grids, models, operators, closures, and `KernelFunctionOperation`. Model constructors (e.g. `NonhydrostaticModel`, `HydrostaticFreeSurfaceModel`) take the grid **positionally**: `NonhydrostaticModel(grid; closure=..., tracers=...)`, *not* `NonhydrostaticModel(; grid, ...)`
- **SeawaterPolynomials.jl**: Equation of state for density calculations (used in PotentialEnergy, MixedLayerDepth)
- **DocStringExtensions.jl**: `$(SIGNATURES)` and `$(TYPEDEF)` macros in docstrings
- **Crayons.jl**: ANSI terminal coloring used by `ProgressMessengers` for the `ColoredNumber` wrapper and the user-facing `set_number_color!` / `@crayon_str` / `Crayon` exports

### Testing

Tests in `test/` share setup via `test_utils.jl` which defines common grids (regular and stretched), closures, buoyancy/coriolis formulations, and model types. Tests typically create Oceananigans models, construct diagnostic KFOs, compute them on a `Field`, and verify values against known analytical solutions or budget closures.

Budget closure is checked by `@test` assertions embedded in `docs/examples/two_dimensional_turbulence.jl` (hidden from the rendered output via Literate `#hide`), so the docs build acts as the budget regression test.

The `perf_invariants` test group guards against performance regressions without encoding hardware-specific numbers: it asserts zero-allocation, type-stable per-cell evaluation on representative KFOs from every module (so accidental boxing or `Any`-typed dispatch fails immediately), plus same-runner ratio invariants on the separable filters (staged 3D wide-stencil path must beat the fused path by ≥2× — same hardware, ratio cancels noise).

GPU coverage comes from a Buildkite pipeline (`.buildkite/gpu-pipeline.yml`) rather than GitHub Actions,
which has no GPU: `test_utils.jl` picks `arch = has_cuda_gpu() ? GPU() : CPU()`, so *every* Actions run
is CPU-only regardless of group. The pipeline runs the same `Pkg.test()` path over the same
`TEST_GROUP`s on a self-hosted V100 agent (`queue=Oceanostics-nautilus`), minus `perf_invariants`, whose
grids are `CPU()` by construction and whose filter checks are wall-clock ratios that a CI host shared
with other projects would make flaky. No `CUDA_Runtime_jll` pin is needed — CUDA.jl already selects a
12.x runtime against the agent's 535 driver — unlike the Oceananigans and Chitin pipelines on the same
host, which pin explicitly. `JULIA_CUDA_USE_COMPAT=false` is set as a conservative default rather than a
fix: measured in the agent container, either setting works, since the container runtime exposes the
image's forward-compat driver (575, still sm_70-capable) regardless. It would start to matter if the
image were rebased onto CUDA 13, whose driver drops sm_70.

## Conventions

- **Naming**: functions and methods use `snake_case`; `CamelCase` is reserved for types, structs, and modules. A `const X = CustomKFO{<:typeof(...)}` parametric type alias *is* a type — methods like `function X(model, ...)` are constructor methods on it and stay CamelCase, since `X(args)` invokes the type's constructor. Genuine standalone helper functions (e.g. inline kernel helpers like `total_∂ⱼ_τ₁ⱼ`) are snake_case.
- Diagnostic constructors accept either a full `model` object or individual fields (velocities, tracers, etc.) for flexibility
- Many constructors use `validate_location` to enforce that diagnostics are only computed at their mathematically valid grid locations
- Dissipation rate diagnostics use `validate_dissipative_closure` to restrict to `AbstractScalarDiffusivity{<:Any, ThreeDimensionalFormulation}`
- Unicode identifiers are used extensively (ψ, ε, ν, ∂, ℑ, etc.) matching mathematical notation
- One-line code expressions are preferred when they fit within 130 columns; only break them across lines when they exceed that width
- Prose text (docstrings, comments, `.md` files) should wrap at around 100 columns
- **Julia examples** in docstrings and docs use ```` ```jldoctest ```` fenced blocks (not ```` ```julia ````) unless explicitly stated otherwise, so the examples are validated as doctests
- **`jldoctest` style**: prefer script style — the code lines followed by a `# output` marker and the expected output — over REPL style (`julia> ` prompts with interleaved results), unless explicitly stated otherwise
- When adding a new leaf progress messenger, wrap its formatted-number string (the result of `@sprintf` / `prettytime`) in `ColoredNumber(...)` so the value participates in the configurable `NUMBER_CRAYON` coloring; prefix and unit text stay as plain `String`
- **Code folding markers**: collapsible code sections are delimited by `#+++ <title>` to open (note the space after `#+++`) and `#---` to close — always exactly three `+`/`-`, never `#++`/`#--`. Nested sections use the same `#+++`/`#---` markers (each `#---` closes the most recent `#+++`)
