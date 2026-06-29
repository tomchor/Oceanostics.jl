# Coarse-grained kinetic energy equation

The `CoarseGrainedKineticEnergyEquation` module provides diagnostics for the *filtered* (coarse-grained)
kinetic energy budget, in which a low-pass spatial filter `(\overline{\;\cdot\;})` separates a filtered
scale from a subfilter scale. Applying the filter to the momentum equation and contracting with the
filtered velocity gives an evolution equation for the filtered kinetic energy
``\overline{K} = \tfrac{1}{2}\,\overline{u}_i\,\overline{u}_i`` in which a new term appears that exchanges
energy between scales:

```math
\Pi_K = -\tau_{ij}\,\overline{S}_{ij},
\qquad
\tau_{ij} = \overline{u_i u_j} - \overline{u}_i\,\overline{u}_j,
\qquad
\overline{S}_{ij} = \tfrac{1}{2}\left(\frac{\partial \overline{u}_i}{\partial x_j} + \frac{\partial \overline{u}_j}{\partial x_i}\right)
```

Here ``\tau_{ij}`` is the subfilter-scale stress tensor and ``\overline{S}_{ij}`` is the strain rate
tensor of the filtered velocity. ``\Pi_K`` is the cross-scale (scale-to-scale) kinetic energy flux: the
rate at which the filter transfers kinetic energy from the filtered to the subfilter scales, following the
coarse-graining framework of [Aluie et al. (2018)](https://doi.org/10.1175/JPO-D-17-0100.1). A positive
``\Pi_K`` denotes a forward (downscale) transfer. It is computed per unit mass (units ``\mathrm{m^2\,s^{-3}}``);
multiply by a reference density ``\rho_0`` for a volumetric power.

The filtered kinetic energy ``\overline{K}`` also has a viscous sink: the dissipation acting on the
*filtered* flow. Mirroring the resolved-scale dissipation ``\varepsilon = \partial_j u_i\,F_{ij}``
([`KineticEnergyDissipationRate`](@ref)) but evaluated on the filtered velocities, it is

```math
\overline{\varepsilon} = \frac{\partial \overline{u}_i}{\partial x_j}\,F_{ij}(\overline{u})
```

where ``\overline{u}_i = \overline{u_i}`` is the filtered velocity and ``F_{ij}`` is the viscous stress
(flux) tensor supplied by the model's closure, evaluated on ``\overline{u}``. For a constant-viscosity
closure this reduces to ``\overline{\varepsilon} = 2\nu\,\overline{S}_{ij}\overline{S}_{ij}``, the
dissipation of the resolved strain. Like ``\Pi_K`` it is per unit mass (units ``\mathrm{m^2\,s^{-3}}``);
multiply by ``\rho_0`` for a volumetric power.

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
└── arguments: ("Nothing", "NamedTuple", "NamedTuple")
└── computes: coarse-grained kinetic energy dissipation rate  ε̄ = ∂ⱼūᵢ·Fᵢⱼ
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
