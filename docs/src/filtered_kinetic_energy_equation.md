# Filtered kinetic energy equation

The `FilteredKineticEnergyEquation` module provides diagnostics for the kinetic energy budget of the
filtered flow, in which a low-pass spatial filter ``\overline{(\,\cdot\,)}`` separates a
filtered from a subfilter scale. The section below derives that budget.

## Deriving the filtered-flow kinetic energy budget

Oceananigans' [`NonhydrostaticModel`](https://clima.github.io/OceananigansDocumentation/stable/physics/nonhydrostatic_model/)
evolves the velocity ``u_i`` with the momentum equation (here without the background-flow, surface-wave,
and forcing terms)

```math
\partial_t u_i = - u_j \, \partial_j u_i
               - \epsilon_{ijk} \, f_j \, u_k
               - \partial_i p
               + b \, \hat g_i
               - \partial_j \tau_{ij} ,
```

with Coriolis parameter ``f_i``, kinematic pressure ``p``, buoyancy ``b`` along the vertical ``\hat g_i``,
the permutation symbol ``\epsilon_{ijk}``, and the diffusive momentum flux ``\tau_{ij}`` supplied by the
closure. The velocity components are ``(u_1, u_2, u_3) = (u, v, w)``, and the resolved viscous dissipation
``\varepsilon_k = -\partial_j u_i \, \tau_{ij}`` is what
[`KineticEnergyDissipationRate`](@ref Oceanostics.KineticEnergyEquation.DissipationRate) computes.
Incompressibility ``\partial_i u_i = 0`` lets us write advection in flux form,
``u_j \, \partial_j u_i = \partial_j (u_i u_j)``.

We can define the subfilter (residual) stress tensor
```math
\tau^r_{ij} = \overline{u_i u_j} - \bar u_i\,\bar u_j ,
```
and re-write the filtered momentum equation as:

```math
\partial_t \bar u_i = - \bar u_j \, \partial_j \bar u_i
                    - \partial_j \tau^r_{ij}
                    - \epsilon_{ijk} \, f_j \, \bar u_k
                    - \partial_i \bar p
                    + \bar b \, \hat g_i
                    - \partial_j \bar\tau_{ij} .
```

Multiplying by the filtered velocity ``\bar u_i`` gives the budget for the kinetic energy
of the filtered flow ``e_k^l = \tfrac{1}{2}\,\bar u_i\,\bar u_i`` ([`FilteredKineticEnergy`](@ref)).
Advection, pressure, and the two stress terms
each split into a transport divergence plus a local term; Coriolis does no work
(``\bar u_i\,\epsilon_{ijk}\,f_j\,\bar u_k = 0``), leaving

```math
\partial_t e_k^l + \partial_j J_j
    = \underbrace{\bar w\,\bar b}_{\text{buoyancy production}}
    - \underbrace{\Pi_k}_{\text{cross-scale flux}}
    - \underbrace{\varepsilon_k^l}_{\text{dissipation}} ,
```

where ``J_i`` collects the (advective, pressure, and stress) transport fluxes, which vanish when
integrated over a closed or periodic domain. The two local exchange terms are

```math
\Pi_k = -\tau^r_{ij}\,\bar S_{ij} ,
\qquad
\varepsilon_k^l = -\partial_j \bar u_i \, \bar\tau_{ij} ,
\qquad
\bar S_{ij} = \tfrac{1}{2}\left(\partial_j \bar u_i + \partial_i \bar u_j\right) .
```

The buoyancy production ``\bar w\,\bar b`` converts between filtered kinetic and potential energy.
``\Pi_k`` ([`KineticEnergyCrossScaleFlux`](@ref)) is the cross-scale kinetic energy flux:
the rate at which the filter transfers kinetic energy from the filtered to the subfilter scales, following
[Aluie et al. (2018)](https://doi.org/10.1175/JPO-D-17-0100.1). A positive ``\Pi_k`` denotes a forward
(downscale) transfer, and ``\bar S_{ij}`` is the strain rate tensor of the filtered velocity. The
subfilter stress ``\tau^r_{ij}`` itself is available as [`subfilter_stress_tensor`](@ref).

``\varepsilon_k^l`` ([`FilteredKineticEnergyDissipationRate`](@ref)) is the viscous dissipation
of the filtered flow: the filtered velocity gradient contracted with the *filtered* momentum flux
``\bar\tau_{ij}``. The flux is filtered, not recomputed from ``\bar u_i``:
``\overline{\tau_{ij}(u_i)}`` and ``\tau_{ij}(\bar u_i)`` agree only for a constant, uniform
viscosity. When ``\tau_{ij}`` is symmetric (as for an isotropic closure) the antisymmetric part of the gradient drops out
and the dissipation can be written with the strain rate, ``\varepsilon_k^l = -\bar\tau_{ij}\,\bar S_{ij}``,
reducing further to ``2\nu\,\bar S_{ij}\bar S_{ij}`` for a constant-viscosity closure.

These diagnostics take a `filter` argument: any callable mapping a field to its low-pass-filtered
counterpart, typically a [`GaussianFilter`](@ref) or [`BoxFilter`](@ref). The directions the
filter acts in (set inside `filter`) are independent of how each diagnostic contracts: the stress tensor
and cross-scale flux take a `dims` argument selecting the directions they contract over — so you can
filter horizontally yet contract the full 3D tensor — while the filtered dissipation always forms
the full viscous contraction.

## Example

```jldoctest
using Oceananigans, Oceanostics

grid = RectilinearGrid(size=(16, 16, 16), extent=(1, 1, 1), topology=(Periodic, Periodic, Bounded))
model = NonhydrostaticModel(grid; closure=ScalarDiffusivity(ν=1e-4))

ℓ = 0.2  # Gaussian filter scale (full width at half maximum) in all three directions
filter = GaussianFilter(; dims=(1, 2, 3), σ=ℓ / (2√(2log(2))), boundary=(left=0, right=0))

τ  = subfilter_stress_tensor(model, filter)                 # the subfilter stress tensor components
Πₖ  = KineticEnergyCrossScaleFlux(model, filter)             # the cross-scale KE flux, at (Center, Center, Center)
εₖˡ = FilteredKineticEnergyDissipationRate(model, filter)    # dissipation of the filtered flow

# equivalently, the convenience methods build the Gaussian filter from σ for you:
εₖˡ = FilteredKineticEnergyDissipationRate(model; σ=ℓ / (2√(2log(2))), boundary=(left=0, right=0))

# output

FilteredKineticEnergyDissipationRate KernelFunctionOperation at (Center, Center, Center)
├── grid: 16×16×16 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: filtered_dissipation_rate_ccc (generic function with 1 method)
└── arguments: ("NamedTuple", "NamedTuple")
└── computes: filtered kinetic energy dissipation rate  εₖˡ = -∂ⱼūᵢ·τ̄ᵢⱼ
```

## Filtered kinetic energy

```@docs
Oceanostics.FilteredKineticEnergyEquation.FilteredKineticEnergy
```

## Subfilter-scale stress tensor

```@docs
Oceanostics.FilteredKineticEnergyEquation.subfilter_stress_tensor
```

## Cross-scale kinetic energy flux

```@docs
Oceanostics.FilteredKineticEnergyEquation.KineticEnergyCrossScaleFlux
```

## Filtered kinetic energy dissipation

```@docs
Oceanostics.FilteredKineticEnergyEquation.FilteredKineticEnergyDissipationRate
```
