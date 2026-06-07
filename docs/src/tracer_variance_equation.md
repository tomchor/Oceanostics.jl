# Tracer variance equation

The `TracerVarianceEquation` module provides diagnostics for the tracer variance
budget. Tracer variance is a second-order quantity that measures the intensity of
tracer fluctuations and is central to understanding mixing in ocean simulations.
For a tracer ``c``, the variance ``c^2`` evolves according to

```math
\partial_t c^2 = 2c\,\partial_t c = -2c\,\partial_j(u_j c) + 2c\,\partial_j F_j + 2c\,F^c
```

where ``F_j`` is the diffusive flux of ``c`` in the ``j``-th direction and ``F^c``
is any applied forcing. This budget can be further decomposed into a diffusive
transport term and a dissipation rate:

```math
2c\,\partial_j F_j = \partial_j(2c\,F_j) - 2\,\partial_j c \cdot F_j
```

The second term, ``\chi = 2\,\partial_j c \cdot F_j``, is the tracer variance
dissipation rate. For Fickian diffusion with diffusivity ``\kappa``, this reduces
to the familiar form ``\chi = 2\kappa |\nabla c|^2``. The dissipation rate is
always positive and represents the irreversible destruction of tracer variance
by molecular or subgrid diffusion -- it is a direct measure of mixing.

All diagnostics are computed at `(Center, Center, Center)`.

## Example

```jldoctest tvar_eq
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(grid; tracers=:b, closure=SmagorinskyLilly());

julia> χ = TracerVarianceEquation.TracerVarianceDissipationRate(model, :b)
TracerVarianceDissipationRate (KernelFunctionOperation) at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: tracer_variance_dissipation_rate_ccc (generic function with 1 method)
└── arguments: ("Oceananigans.TurbulenceClosures.Smagorinskys.Smagorinsky", "NamedTuple", "Val", "Field", "Clock", "NamedTuple", "Nothing")
└── computes: tracer variance dissipation rate  χ = 2 ∂ⱼc·Fⱼ

julia> diff = TracerVarianceEquation.TracerVarianceDiffusion(model, :b)
TracerVarianceDiffusion (KernelFunctionOperation) at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: c∇_dot_qᶜ (generic function with 1 method)
└── arguments: ("Oceananigans.TurbulenceClosures.Smagorinskys.Smagorinsky", "NamedTuple", "Val", "Field", "Clock", "NamedTuple", "Nothing")
└── computes: tracer variance diffusion  2c ∂ⱼFⱼ
```

## Tendency

```@docs
Oceanostics.TracerVarianceEquation.TracerVarianceTendency
```

## Diffusion

```@docs
Oceanostics.TracerVarianceEquation.TracerVarianceDiffusion
```

## Dissipation rate

```@docs
Oceanostics.TracerVarianceEquation.TracerVarianceDissipationRate
```
