# Coarse-grained kinetic energy equation

The `CoarseGrainedKineticEnergyEquation` module provides diagnostics for the *filtered* (coarse-grained)
kinetic energy budget, in which a low-pass spatial filter `(\overline{\;\cdot\;})` separates a resolved
scale from a subfilter scale. Applying the filter to the momentum equation and contracting with the
filtered velocity gives an evolution equation for the resolved kinetic energy
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
rate at which the filter transfers kinetic energy from the resolved to the subfilter scales, following the
coarse-graining framework of [Aluie et al. (2018)](https://doi.org/10.1175/JPO-D-17-0100.1). A positive
``\Pi_K`` denotes a forward (downscale) transfer. It is computed per unit mass (units ``\mathrm{m^2\,s^{-3}}``);
multiply by a reference density ``\rho_0`` for a volumetric power.

Both diagnostics take a `filter` argument: a function mapping a field to its low-pass-filtered
counterpart, typically a closure over [`GaussianFilter`](@ref) or [`BoxFilter`](@ref). This decouples the
choice of filter (kernel, scale, boundary treatment, and the directions it acts in) from the directions
the tensors are contracted over (`dims`), so you can — for instance — filter horizontally yet contract the
full 3D tensor.

## Example

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(16, 16, 16), extent=(1, 1, 1), topology=(Periodic, Periodic, Bounded));

julia> model = NonhydrostaticModel(grid);

julia> ℓ = 0.2;  # Gaussian filter scale (full width at half maximum) in all three directions

julia> filter(ψ) = GaussianFilter(ψ; dims=(1, 2, 3), σ=ℓ / (2√(2log(2))), boundary=(left=0, right=0))

julia> CrossScaleKineticEnergyFlux(model, filter)   # the cross-scale KE flux, at (Center, Center, Center)
CrossScaleKineticEnergyFlux KernelFunctionOperation at (Center, Center, Center)
├── grid: 16×16×16 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: cross_scale_ke_flux_ccc (generic function with 1 method)
└── arguments: ("Oceananigans.AbstractOperations.UnaryOperation",)
└── computes: cross-scale kinetic energy flux  Πₖ = -τⁱʲS̄ⁱʲ

julia> keys(SubfilterStressTensor(model, filter))   # the subfilter stress tensor components
(:τ₁₁, :τ₂₂, :τ₃₃, :τ₁₂, :τ₁₃, :τ₂₃)

julia> CrossScaleKineticEnergyFlux(model; σ=ℓ / (2√(2log(2))), boundary=(left=0, right=0)) isa CrossScaleKineticEnergyFlux  # convenience method
true
```

## Subfilter-scale stress tensor

```@docs
Oceanostics.CoarseGrainedKineticEnergyEquation.SubfilterStressTensor
```

## Cross-scale kinetic energy flux

```@docs
Oceanostics.CoarseGrainedKineticEnergyEquation.CrossScaleKineticEnergyFlux
```
