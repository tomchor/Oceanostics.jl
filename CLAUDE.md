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

Available TEST_GROUP values: `vel_diagnostics`, `tracer_diagnostics`, `ke_diagnostics`, `tke_diagnostics`, `pe_diagnostics`, `active_tracer_diagnostics`, `tracer_variance_diagnostics`, `general_flow_diagnostics`, `canonical_flows`, `progress_messengers`, `budgets`.

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
- **`KineticEnergyEquation`**: KE, its tendency, advection, stress, forcing, pressure redistribution, buoyancy production, dissipation rate (general and isotropic)
- **`TurbulentKineticEnergyEquation`**: TKE, isotropic dissipation, shear production rates (X/Y/Z and total)
- **`TracerVarianceEquation`**: Tendency, dissipation rate, diffusion of tracer variance
- **`PotentialEnergyEquation`**: Potential energy for BuoyancyTracer, linear/nonlinear SeawaterBuoyancy
- **`FlowDiagnostics`**: Richardson/Rossby numbers, Ertel/ThermalWind potential vorticity, strain rate & vorticity tensor moduli, Q-criterion, MixedLayerDepth, BottomCellValue
- **`ProgressMessengers`** (submodule): Composable simulation progress reporters using `+` (comma-separated) and `*` (concatenation) operators

### Key Dependencies

- **Oceananigans.jl**: The ocean simulation framework — provides grids, models, operators, closures, and `KernelFunctionOperation`
- **SeawaterPolynomials.jl**: Equation of state for density calculations (used in PotentialEnergy, MixedLayerDepth)
- **DocStringExtensions.jl**: `$(SIGNATURES)` and `$(TYPEDEF)` macros in docstrings

### Testing

Tests in `test/` share setup via `test_utils.jl` which defines common grids (regular and stretched), closures, buoyancy/coriolis formulations, and model types. Tests typically create Oceananigans models, construct diagnostic KFOs, compute them on a `Field`, and verify values against known analytical solutions or budget closures.

The `budgets` test group is the most expensive (5-hour CI timeout) and validates that equation terms sum correctly.

## Conventions

- Diagnostic constructors accept either a full `model` object or individual fields (velocities, tracers, etc.) for flexibility
- Many constructors use `validate_location` to enforce that diagnostics are only computed at their mathematically valid grid locations
- Dissipation rate diagnostics use `validate_dissipative_closure` to restrict to `AbstractScalarDiffusivity{<:Any, ThreeDimensionalFormulation}`
- Unicode identifiers are used extensively (ψ, ε, ν, ∂, ℑ, etc.) matching mathematical notation
- One-line code expressions are preferred when they fit within 130 columns; only break them across lines when they exceed that width
- Prose text (docstrings, comments, `.md` files) should wrap at around 100 columns
