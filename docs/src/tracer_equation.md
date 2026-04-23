# Tracer equation

The `TracerEquation` module provides diagnostics corresponding to individual terms
in the tracer conservation equation. In Oceananigans, the prognostic equation for a
tracer ``c`` (e.g. temperature, salinity, or a passive scalar) is

```math
\partial_t c = -\partial_j (u_j c) + \partial_j q^c_j + F^c
```

where ``u_j`` is the resolved velocity field, ``q^c_j`` is the subgrid diffusive flux
of ``c`` in the ``j``-th direction (parameterized by the turbulence closure), and ``F^c``
represents any external forcing applied to the tracer.

This module decomposes the right-hand side into its constituent terms so that each
can be computed, output, and analyzed independently. This is useful for constructing
tracer budgets, diagnosing the relative importance of advection versus diffusion,
or verifying that the budget closes numerically (i.e., that the sum of all terms
equals the tendency).

Each diagnostic is built on Oceananigans' `KernelFunctionOperation` and is computed
at `(Center, Center, Center)`. Constructors accept either a full Oceananigans `model`
object (with a tracer name) or individual fields for maximum flexibility.

## Example

```jldoctest tracer_eq
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(grid; tracers=:b, closure=ScalarDiffusivity(ν=1e-4, κ=1e-4));

julia> adv  = TracerEquation.Advection(model, :b)
KernelFunctionOperation at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: div_Uc (generic function with 10 methods)
└── arguments: ("Centered", "NamedTuple", "Field")

julia> diff = TracerEquation.Diffusion(model, :b)
KernelFunctionOperation at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: ∇_dot_qᶜ (generic function with 10 methods)
└── arguments: ("ScalarDiffusivity", "Nothing", "Val", "Field", "Clock", "NamedTuple", "Nothing")

julia> forc = TracerEquation.Forcing(model, :b)
KernelFunctionOperation at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: Returns (generic function with 1 method)
└── arguments: ("Clock", "NamedTuple")
```

## Advection

```@docs
Oceanostics.TracerEquation.Advection
```

## Diffusion

```@docs
Oceanostics.TracerEquation.Diffusion
Oceanostics.TracerEquation.ImmersedDiffusion
Oceanostics.TracerEquation.TotalDiffusion
```

## Forcing

```@docs
Oceanostics.TracerEquation.Forcing
```
