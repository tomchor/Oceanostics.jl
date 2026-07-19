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

Available TEST_GROUP values: `vel_diagnostics`, `tracer_diagnostics`, `u_momentum_diagnostics`, `v_momentum_diagnostics`, `w_momentum_diagnostics`, `ke_diagnostics`, `filtered_ke_diagnostics`, `subfilter_ke_diagnostics`, `tke_diagnostics`, `pe_diagnostics`, `active_tracer_diagnostics`, `tracer_variance_diagnostics`, `general_flow_diagnostics`, `canonical_flows`, `progress_messengers`, `spatial_filters`, `perf_invariants`.

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
- **`SpatialFilters`** (submodule): Spatial filters (`box_filter.jl`, `gaussian_filter.jl`) for diagnostics that need scale separation.
- **`KineticEnergyEquation`**: KE, its tendency, advection, stress, forcing, pressure redistribution, buoyancy production, dissipation rate (general and isotropic)
- **`FilteredKineticEnergyEquation`**: Filtered (coarse-grained) KE budget terms — `FilteredKineticEnergy` (Kˡ = ½ūᵢūᵢ, KE of the filtered flow; reuses `KineticEnergyEquation`'s `kinetic_energy_ccc` kernel), `subfilter_stress_tensor` (τⁱʲ = filter(uⁱuʲ) − ūⁱūʲ), `KineticEnergyCrossScaleFlux` (Πₖ = −τⁱʲS̄ⁱʲ, Aluie et al. 2018), and `FilteredKineticEnergyDissipationRate` (εˡ, dissipation of the filtered flow; kernel still named `coarse_grained_dissipation_rate_ccc`). Built on `FlowDiagnostics`' `StressTensor`/`StrainRateTensor` and the `Filters` submodule, so it is included after both.
- **`SubFilterKineticEnergyEquation`**: Sub-filter KE budget terms — `SubFilterKineticEnergy` (Kˢ = ½τⁱⁱ, computed as `filter(K) − Kˡ` from `KineticEnergy` and `FilteredKineticEnergy`, which share the same interpolate-the-square discretization, so the discrete decomposition `filter(K) = Kˡ + Kˢ` holds exactly by construction on any grid) and `SubFilterKineticEnergyDissipationRate` (εˢ = filter(ε) − εˡ). Both are `KernelFunctionOperation`s wrapping the underlying composite op (à la `KineticEnergyCrossScaleFlux`). Also re-exports `KineticEnergyCrossScaleFlux` (a source term of this budget). Built on `FilteredKineticEnergyEquation` and `KineticEnergyEquation`, so it is included after both.
- **`TurbulentKineticEnergyEquation`**: TKE, isotropic dissipation, shear production rates (X/Y/Z and total)
- **`TracerVarianceEquation`**: Tendency, dissipation rate, diffusion of tracer variance
- **`PotentialEnergyEquation`**: Potential energy for BuoyancyTracer, linear/nonlinear SeawaterBuoyancy
  (`PotentialEnergy`, Eₚ = -bz), plus the Winters et al. (1995) split into `BackgroundPotentialEnergy`
  (E_b = -bz✶) and `AvailablePotentialEnergy` (Eₐ = -b(z - z✶)). The reference height z✶ comes from
  `sorted_reference_height`, which returns a `Field` whose operand is a `SortedReferenceState` rather
  than a `KernelFunctionOperation`: sorting is a whole-domain operation, so it hooks into `compute!`
  the way `Oceananigans.Fields.Scan` does for `Integral`/`Average`, and is re-sorted on every
  `compute!` so the diagnostic tracks the flow when written out during a simulation. The `method`
  keyword picks one of three `AbstractSortingMethod`s, which agree on every volume integral:
  `ThreeDimensionalSort` (default; each cell gets its own slot, so tied cells spread over a grid cell),
  `HeavisideIntegral` (Winters eq. 11, tied cells share their layer's mid-height, so z✶ is a function
  of buoyancy alone and local maps are clean), and `OneDimensionalSort` (the sorted column on its own
  `1×1×N` grid of equal-volume cells; needs uniform cell volumes)
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
