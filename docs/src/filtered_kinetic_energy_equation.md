# Filtered kinetic energy equation

The `FilteredKineticEnergyEquation` module provides diagnostics for the kinetic energy budget of the
coarse-grained (filtered) flow, in which a low-pass spatial filter ``\widetilde{(\,\cdot\,)}`` separates a
filtered from a subfilter scale. The section below derives that budget.

## Deriving the coarse-grained kinetic energy budget

Oceananigans' [`NonhydrostaticModel`](https://clima.github.io/OceananigansDocumentation/stable/physics/nonhydrostatic_model/)
evolves the velocity ``v_i`` with the momentum equation (here without the background-flow, surface-wave,
and forcing terms)

```math
\partial_t v_i = - v_j \, \partial_j v_i
               - \epsilon_{ijk} \, f_j \, v_k
               - \partial_i p
               + b \, \hat g_i
               - \partial_j \tau_{ij} ,
```

with Coriolis parameter ``f_i``, kinematic pressure ``p``, buoyancy ``b`` along the vertical ``\hat g_i``,
the permutation symbol ``\epsilon_{ijk}``, and the kinematic stress tensor ``\tau_{ij}`` supplied by the
closure. The velocity components are ``(v_1, v_2, v_3) = (u, v, w)``, and the resolved viscous dissipation
``\varepsilon = -\partial_j v_i \, \tau_{ij}`` is what
[`KineticEnergyDissipationRate`](@ref Oceanostics.KineticEnergyEquation.DissipationRate) computes.
Incompressibility ``\partial_i v_i = 0`` lets us write advection in flux form,
``v_j \, \partial_j v_i = \partial_j (v_i v_j)``.

We can define the residual (subfilter) stress tensor
```math
\tau^r_{ij} = \widetilde{v_i v_j} - \tilde v_i\,\tilde v_j ,
```
and re-write the filtered momentum equation as:

```math
\partial_t \tilde v_i = - \tilde v_j \, \partial_j \tilde v_i
                    - \partial_j \tau^r_{ij}
                    - \epsilon_{ijk} \, f_j \, \tilde v_k
                    - \partial_i \tilde p
                    + \tilde b \, \hat g_i
                    - \partial_j \tilde\tau_{ij} .
```

Multiplying by the filtered velocity ``\tilde v_i`` gives the budget for the kinetic energy
of the filtered flow ``K^l = \tfrac{1}{2}\,\tilde v_i\,\tilde v_i`` ([`FilteredKineticEnergy`](@ref)).
Advection, pressure, and the two stress terms
each split into a transport divergence plus a local term; Coriolis does no work
(``\tilde v_i\,\epsilon_{ijk}\,f_j\,\tilde v_k = 0``), leaving

```math
\partial_t K^l + \partial_j J_j
    = \underbrace{\tilde w\,\tilde b}_{\text{buoyancy production}}
    - \underbrace{\Pi_K}_{\text{cross-scale flux}}
    - \underbrace{\varepsilon^l}_{\text{dissipation}} ,
```

where ``J_i`` collects the (advective, pressure, and stress) transport fluxes, which vanish when
integrated over a closed or periodic domain. The two local exchange terms are

```math
\Pi_K = -\tau^r_{ij}\,\widetilde{S}_{ij} ,
\qquad
\varepsilon^l = -\partial_j \tilde v_i \, \tilde\tau_{ij} ,
\qquad
\widetilde{S}_{ij} = \tfrac{1}{2}\left(\partial_j \tilde v_i + \partial_i \tilde v_j\right) .
```

The buoyancy production ``\tilde w\,\tilde b`` converts between filtered kinetic and potential energy.
``\Pi_K`` ([`KineticEnergyCrossScaleFlux`](@ref)) is the cross-scale kinetic energy flux:
the rate at which the filter transfers kinetic energy from the filtered to the subfilter scales, following
[Aluie et al. (2018)](https://doi.org/10.1175/JPO-D-17-0100.1). A positive ``\Pi_K`` denotes a forward
(downscale) transfer, and ``\widetilde{S}_{ij}`` is the strain rate tensor of the filtered velocity. The
residual stress ``\tau^r_{ij}`` itself is available as [`subfilter_stress_tensor`](@ref).

``\varepsilon^l`` ([`FilteredKineticEnergyDissipationRate`](@ref)) is the viscous dissipation
of the filtered flow: the filtered velocity gradient contracted with the *filtered* stress
``\tilde\tau_{ij}``. The stress is filtered, not recomputed from ``\tilde v_i``:
``\widetilde{\tau_{ij}(v_i)}`` and ``\tau_{ij}(\tilde v_i)`` agree only for a constant, uniform
viscosity. When ``\tau_{ij}`` is symmetric (as for an isotropic closure) the antisymmetric part of the gradient drops out
and the dissipation can be written with the strain rate, ``\varepsilon^l = -\tilde\tau_{ij}\,\widetilde{S}_{ij}``,
reducing further to ``2\nu\,\widetilde{S}_{ij}\widetilde{S}_{ij}`` for a constant-viscosity closure.

These diagnostics take a `filter` argument: any callable mapping a field to its low-pass-filtered
counterpart, typically a [`GaussianFilter`](@ref) or [`BoxFilter`](@ref). The directions the
filter acts in (set inside `filter`) are independent of how each diagnostic contracts: the stress tensor
and cross-scale flux take a `dims` argument selecting the directions they contract over — so you can
filter horizontally yet contract the full 3D tensor — while the coarse-grained dissipation always forms
the full viscous contraction.

## Example

```jldoctest
using Oceananigans, Oceanostics

grid = RectilinearGrid(size=(16, 16, 16), extent=(1, 1, 1), topology=(Periodic, Periodic, Bounded))
model = NonhydrostaticModel(grid; closure=ScalarDiffusivity(ν=1e-4))

ℓ = 0.2  # Gaussian filter scale (full width at half maximum) in all three directions
filter = GaussianFilter(; dims=(1, 2, 3), σ=ℓ / (2√(2log(2))), boundary=(left=0, right=0))

τ  = subfilter_stress_tensor(model, filter)                  # the subfilter stress tensor components
Πₖ = KineticEnergyCrossScaleFlux(model, filter)              # the cross-scale KE flux, at (Center, Center, Center)
εˡ = FilteredKineticEnergyDissipationRate(model, filter) # dissipation of the filtered flow

# equivalently, the convenience methods build the Gaussian filter from σ for you:
εˡ = FilteredKineticEnergyDissipationRate(model; σ=ℓ / (2√(2log(2))), boundary=(left=0, right=0))

# output

FilteredKineticEnergyDissipationRate KernelFunctionOperation at (Center, Center, Center)
├── grid: 16×16×16 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: coarse_grained_dissipation_rate_ccc (generic function with 1 method)
└── arguments: ("NamedTuple", "NamedTuple")
└── computes: coarse-grained kinetic energy dissipation rate  εˡ = ∂ⱼūᵢ·F̄ᵢⱼ
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

## Coarse-grained kinetic energy dissipation

```@docs
Oceanostics.FilteredKineticEnergyEquation.FilteredKineticEnergyDissipationRate
```
