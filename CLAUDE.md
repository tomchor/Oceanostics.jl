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

Available TEST_GROUP values: `vel_diagnostics`, `tracer_diagnostics`, `u_momentum_diagnostics`, `v_momentum_diagnostics`, `w_momentum_diagnostics`, `ke_diagnostics`, `filtered_ke_diagnostics`, `subfilter_ke_diagnostics`, `tke_diagnostics`, `pe_diagnostics`, `ape_diagnostics`, `subfilter_ape_diagnostics`, `active_tracer_diagnostics`, `tracer_variance_diagnostics`, `general_flow_diagnostics`, `canonical_flows`, `progress_messengers`, `spatial_filters`, `perf_invariants`.

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
- **`SpatialFilters`** (submodule): Spatial filters (`box_filter.jl`, `gaussian_filter.jl`) for diagnostics that need scale separation. Every 1D kernel sizes its in-range check with `stencil_length(grid, d, ψ)` (the operand's own extent, `N+1` for a `Face` location along a `Bounded` direction) rather than `size(grid, d)`; the recursive (fused) methods reach the operand through `fargs[end]`. Using the cell count there silently mistreats the last face of a `Face`-located operand (`:shrink` drops its own weight, `:edge` clamps to the face below), which surfaced as `NaN`s when filtering the model's `diffusive_flux_z` KFO with a degenerate identity-scale Gaussian.
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
- **`SubFilterAvailablePotentialEnergyEquation`**: sub-filter APE budget terms —
  `SubFilterAvailablePotentialEnergy` (Eₐˢ = filter(eₐ) − eₐ(b̄, z), the filtered full local APE minus
  the local APE of the filtered buoyancy b̄ = filter(b)) and
  `SubFilterAvailablePotentialEnergyDissipationRate` (εₐˢ = filter(εₐ) − εₐˡ, where
  εₐˡ = −q̄ᵢ∂ᵢΥˡ contracts the closure's diffusive flux *low-pass filtered* — the same filtered-flux
  choice `FilteredKineticEnergyDissipationRate` makes for the viscous flux, exact for constant κ —
  against the displacement potential Υˡ of the filtered buoyancy). Both quantities measure the full
  and the filtered buoyancy against *one shared reference profile*, so their `method` keyword must be
  a `ProfileLookup`: the default `ProfileLookup()` internally builds a `VerticalSort` column of the
  model's buoyancy (re-sorted every `compute!`, so the reference tracks the flow),
  `ProfileLookup(z✶_column)` shares an existing column across diagnostics, and `ProfileLookup(b✶, z✶)`
  with arrays freezes the reference in time and makes the diagnostics sort-free. eₐ is convex in b, so
  Eₐˢ ≥ 0 pointwise for filters with no vertical component (Jensen); vertical filtering (and the
  nearest-slot fallback for buoyancies outside the profile) can produce locally negative values. An
  identity-scale filter (σ ≪ Δx, N=3) makes both diagnostics vanish to the bit, which
  `test_subfilter_ape_diagnostics.jl` uses to check the filtered-state kernels against the full-state
  ones without reimplementation. Built on `AvailablePotentialEnergyEquation` and `SpatialFilters`, so
  it is included after both (currently last of the equation modules)
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
