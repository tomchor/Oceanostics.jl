# Coarse-grained kinetic energy equation

The `CoarseGrainedKineticEnergyEquation` module provides diagnostics for the *filtered* (coarse-grained)
kinetic energy budget, in which a low-pass spatial filter ``\overline{(\,\cdot\,)}`` separates a filtered
(resolved) scale from a subfilter scale. The section below derives that budget; the two terms it adds to
the resolved balance are the diagnostics this module exposes.

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
closure (its components are the model's viscous fluxes, the ones
[`KineticEnergyDissipationRate`](@ref Oceanostics.KineticEnergyEquation.DissipationRate) contracts; the
velocity components are ``(v_1, v_2, v_3) = (u, v, w)``). Incompressibility ``\partial_i v_i = 0`` lets us
write advection in flux form, ``v_j \, \partial_j v_i = \partial_j (v_i v_j)``.

Filtering commutes with the derivatives, but not with the nonlinear product ``v_i v_j``. Writing

```math
\overline{v_i v_j} = \bar v_i\,\bar v_j + \tau^d_{ij} ,
\qquad
\tau^d_{ij} = \overline{v_i v_j} - \bar v_i\,\bar v_j ,
```

the filtered momentum equation picks up one genuinely new term, the **residual (subfilter) stress tensor**
``\tau^d_{ij}``:

```math
\partial_t \bar v_i = - \bar v_j \, \partial_j \bar v_i
                    - \partial_j \tau^d_{ij}
                    - \epsilon_{ijk} \, f_j \, \bar v_k
                    - \partial_i \bar p
                    + \bar b \, \hat g_i
                    - \partial_j \bar\tau_{ij} .
```

Multiplying by the filtered velocity ``\bar v_i`` gives the budget for the filtered kinetic
energy ``\overline{K} = \tfrac{1}{2}\,\bar v_i\,\bar v_i``. Advection, pressure, and the two stress terms
each split into a transport divergence plus a local term; Coriolis does no work
(``\bar v_i\,\epsilon_{ijk}\,f_j\,\bar v_k = 0``), leaving

```math
\partial_t \overline{K} + \partial_j J_j
    = \underbrace{\bar w\,\bar b}_{\text{buoyancy production}}
    - \underbrace{\Pi_K}_{\text{cross-scale flux}}
    - \underbrace{\overline{\varepsilon}}_{\text{dissipation}} ,
```

where ``J_i`` collects the (advective, pressure, and stress) transport fluxes, which vanish when
integrated over a closed or periodic domain. The two local exchange terms are

```math
\Pi_K = -\tau^d_{ij}\,\overline{S}_{ij} ,
\qquad
\overline{\varepsilon} = \partial_j \bar v_i \, \overline{F}_{ij} = -\bar\tau_{ij}\,\overline{S}_{ij} ,
\qquad
\overline{S}_{ij} = \tfrac{1}{2}\left(\partial_j \bar v_i + \partial_i \bar v_j\right) .
```

The buoyancy production ``\bar w\,\bar b`` converts between filtered kinetic and potential energy (the
Kelvin-Helmholtz example closes this budget term by term). The two exchange terms are the module's
diagnostics.

``\Pi_K`` ([`KineticEnergyCrossScaleFlux`](@ref)) is the cross-scale (scale-to-scale) kinetic energy flux:
the rate at which the filter transfers kinetic energy from the filtered to the subfilter scales, following
[Aluie et al. (2018)](https://doi.org/10.1175/JPO-D-17-0100.1). A positive ``\Pi_K`` denotes a forward
(downscale) transfer, and ``\overline{S}_{ij}`` is the strain rate tensor of the filtered velocity. The
residual stress ``\tau^d_{ij}`` itself is available as [`subfilter_stress_tensor`](@ref).

``\overline{\varepsilon}`` ([`CoarseGrainedKineticEnergyDissipationRate`](@ref)) is the viscous dissipation
of the filtered flow: the filtered velocity gradient contracted with the *filtered* viscous flux
``\overline{F}_{ij} = \overline{F_{ij}(v)}``. The flux is filtered, not recomputed from
``\bar v_i``: ``\overline{F_{ij}(v)}`` and ``F_{ij}(\bar v)`` agree only for a constant, uniform
viscosity (where the filter commutes with the flux) and differ once the viscosity varies in space. For a
constant-viscosity closure it reduces to ``\overline{\varepsilon} = 2\nu\,\overline{S}_{ij}\overline{S}_{ij}``,
the dissipation of the resolved strain.

Both ``\Pi_K`` and ``\overline{\varepsilon}`` are per unit mass (units ``\mathrm{m^2\,s^{-3}}``); multiply
by a reference density ``\rho_0`` for a volumetric power.

These diagnostics take a `filter` argument: any callable mapping a field to its low-pass-filtered
counterpart, typically a reusable [`GaussianFilter`](@ref) or [`BoxFilter`](@ref). The directions the
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
ε̄ = CoarseGrainedKineticEnergyDissipationRate(model, filter) # dissipation of the filtered flow

# equivalently, the convenience methods build the Gaussian filter from σ for you:
ε̄ = CoarseGrainedKineticEnergyDissipationRate(model; σ=ℓ / (2√(2log(2))), boundary=(left=0, right=0))

# output

CoarseGrainedKineticEnergyDissipationRate KernelFunctionOperation at (Center, Center, Center)
├── grid: 16×16×16 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: coarse_grained_dissipation_rate_ccc (generic function with 1 method)
└── arguments: ("NamedTuple", "NamedTuple")
└── computes: coarse-grained kinetic energy dissipation rate  ε̄ = ∂ⱼūᵢ·F̄ᵢⱼ
```

## Subfilter-scale stress tensor

```@docs
Oceanostics.CoarseGrainedKineticEnergyEquation.subfilter_stress_tensor
```

## Cross-scale kinetic energy flux

```@docs
Oceanostics.CoarseGrainedKineticEnergyEquation.KineticEnergyCrossScaleFlux
```

## Coarse-grained kinetic energy dissipation

```@docs
Oceanostics.CoarseGrainedKineticEnergyEquation.CoarseGrainedKineticEnergyDissipationRate
```
