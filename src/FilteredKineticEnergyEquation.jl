module FilteredKineticEnergyEquation

using DocStringExtensions

export FilteredKineticEnergy
export subfilter_stress_tensor, KineticEnergyCrossScaleFlux, CrossScaleFlux
export FilteredKineticEnergyDissipationRate, DissipationRate

using Oceananigans: fields
using Oceananigans.Grids: Center, Face
using Oceananigans.Fields: Field
using Oceananigans.Operators
using Oceananigans.AbstractOperations: @at, KernelFunctionOperation
using Oceananigans.TurbulenceClosures: viscous_flux_ux, viscous_flux_uy, viscous_flux_uz,
                                       viscous_flux_vx, viscous_flux_vy, viscous_flux_vz,
                                       viscous_flux_wx, viscous_flux_wy, viscous_flux_wz

using Oceanostics: CustomKFO, NamedKernel, index_operation_ccc
using ..FlowDiagnostics: StressTensor, StrainRateTensor
import ..FlowDiagnostics            # for the (unexported) `validate_dims`
using ..SpatialFilters: GaussianFilter, BoxFilter   # BoxFilter is imported so its docstring `@ref` resolves in-module
using ..KineticEnergyEquation: kinetic_energy_ccc   # reuse the ½uᵢuᵢ kernel, applied to the filtered velocities

#+++ Shared helpers
# Filter only the velocities that the requested `dims` actually use: component τᵢⱼ / S̄ᵢⱼ needs uᵢ and
# uⱼ, so velocity component d is filtered iff d ∈ dims; the others are passed through untouched (they
# never enter the kept tensor components). Each filtered velocity is materialized as a `Field` so the
# separable filter takes its fast staged path and is computed once and reused by both tensors.
function filtered_velocities(filter, dims, u, v, w)
    ū = (1 in dims) ? Field(filter(u)) : u
    v̄ = (2 in dims) ? Field(filter(v)) : v
    w̄ = (3 in dims) ? Field(filter(w)) : w
    return ū, v̄, w̄
end

# Low-level method of `subfilter_stress_tensor` taking already-filtered velocities `ū, v̄, w̄`, so the
# (expensive) velocity filtering can be done once and shared — `KineticEnergyCrossScaleFlux` reuses
# `ū, v̄, w̄` for both the strain rate and this stress. Computes τⁱʲ = filter(uⁱuʲ) - ūⁱūʲ component by
# component, reusing `StressTensor` to build the momentum-flux tensor uⁱuʲ of both the full and the
# filtered velocity; `filter` is applied to the (materialized) full flux and the filtered flux is
# subtracted. The result is a `NamedTuple` with the same keys/locations as `StressTensor`.
function subfilter_stress_tensor(filter, grid, u, v, w, ū, v̄, w̄; dims, collocate_diagonals=false)
    flux_full = StressTensor(grid, u, v, w; dims, collocate_diagonals)   # uⁱuʲ
    flux_filt = StressTensor(grid, ū, v̄, w̄; dims, collocate_diagonals)   # ūⁱūʲ
    subfilter(full, coarse) = Field(filter(Field(full))) - coarse        # filter(uⁱuʲ) - ūⁱūʲ
    ks = keys(flux_full)
    return NamedTuple{ks}(map(subfilter, values(flux_full), values(flux_filt)))
end
#---

#+++ Filtered kinetic energy
# Kˡ = ½ ūᵢūᵢ, the kinetic energy of the filtered velocity field ūᵢ = filter(uᵢ). It is exactly
# `KineticEnergyEquation`'s `kinetic_energy_ccc` kernel (½uᵢuᵢ interpolated to ccc) applied to the
# filtered velocities, so we reuse that kernel directly and only tag it with a `NamedKernel` label
# (`:Kˡ`) to give `FilteredKineticEnergy` its own type alias and `@diagnostic_show` display.
const FilteredKineticEnergy = CustomKFO{<:NamedKernel{:Kˡ}}

"""
    $(SIGNATURES)

Return the kinetic energy of the filtered flow `Kˡ`, the kinetic energy carried by the
scales that a low-pass `filter` keeps:

```
    Kˡ = ½ ūⁱ ūⁱ = ½ (ū² + v̄² + w̄²) ,   ūⁱ = filter(uⁱ)
```

It is the filtered counterpart of the sub-filter kinetic energy `Kˢ = ½τⁱⁱ` (`SubFilterKineticEnergy`):
the filter splits the flow's kinetic energy into the part it keeps (`Kˡ`) and the part it removes (`Kˢ`).

`filter` is any callable mapping a field to its low-pass-filtered counterpart, e.g. a reusable
[`GaussianFilter`](@ref) or [`BoxFilter`](@ref). The filtered velocities are materialized as `Field`s (so
the separable filter takes its fast staged path) and the result lives at `(Center, Center, Center)`, per
unit mass (units `m² s⁻²`):

```jldoctest
using Oceananigans, Oceanostics

grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1), topology=(Periodic, Periodic, Bounded))
model = NonhydrostaticModel(grid)

filter = GaussianFilter(; dims=(1, 2, 3), σ=0.1)
FilteredKineticEnergy(model, filter)

# output

FilteredKineticEnergy KernelFunctionOperation at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: kinetic_energy_ccc (generic function with 1 method)
└── arguments: ("Field", "Field", "Field")
└── computes: kinetic energy of the filtered flow  Kˡ = ½ūᵢūᵢ
```

A convenience method `FilteredKineticEnergy(model; σ, dims, boundary, N)` builds the Gaussian `filter`
for you from a standard deviation `σ` (with `σ = ℓ / (2√(2 ln 2))` for a FWHM `ℓ`).
"""
function FilteredKineticEnergy(model, filter)
    u, v, w = model.velocities
    ū, v̄, w̄ = filtered_velocities(filter, (1, 2, 3), u, v, w)   # materialize all three filtered velocities
    kernel = NamedKernel{:Kˡ}(kinetic_energy_ccc)
    return KernelFunctionOperation{Center, Center, Center}(kernel, model.grid, ū, v̄, w̄)
end

FilteredKineticEnergy(model; σ, dims = (1, 2, 3), boundary = :shrink, N = nothing) = FilteredKineticEnergy(model, GaussianFilter(; dims, σ, boundary, N))
#---

#+++ Subfilter stress tensor
"""
    $(SIGNATURES)

Return the components of the subfilter-scale (SFS) stress tensor `τ`, the residual momentum flux that
a low-pass `filter` removes from the filtered scales:

```
    τⁱʲ = filter(uⁱuʲ) - ūⁱ ūʲ ,   ūⁱ = filter(uⁱ)
```

(also called the sub-grid stress in LES, or the generalized central moment in the coarse-graining
framework of Aluie et al., 2018, *J. Phys. Oceanogr.*, doi:10.1175/JPO-D-17-0100.1). It is the
quantity contracted with the filtered strain rate to form the cross-scale kinetic-energy flux — see
[`KineticEnergyCrossScaleFlux`](@ref).

`filter` is any callable that maps a field to its low-pass-filtered counterpart, e.g. a reusable
[`GaussianFilter`](@ref) or [`BoxFilter`](@ref):

```jldoctest
using Oceananigans, Oceanostics

grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1), topology=(Periodic, Periodic, Bounded))
model = NonhydrostaticModel(grid)

filter = GaussianFilter(; dims=(1, 2, 3), σ=0.1)
τ = subfilter_stress_tensor(model, filter)

keys(τ)

# output

(:τ₁₁, :τ₂₂, :τ₃₃, :τ₁₂, :τ₁₃, :τ₂₃)
```

The result is a `NamedTuple` with the independent components, each living at the same staggered
location as the corresponding [`StressTensor`](@ref) component; `collocate_diagonals` has the same
meaning as there and is forwarded to it (use `collocate_diagonals = true` to put the diagonals at
`ccc`, e.g. to form the subfilter kinetic energy `½(τ₁₁ + τ₂₂ + τ₃₃)`). The filtered velocities `ūⁱ`
and the filtered momentum fluxes `filter(uⁱuʲ)` are materialized as `Field`s internally (the filter's
fast staged path only fires when wrapped directly in a `Field`), so each returned component is a lazy
operation over those computed fields and recomputes correctly when written by an `OutputWriter`.

`dims` selects which spatial directions enter the tensor, exactly as in [`StressTensor`](@ref):
component `τⁱʲ` is kept only when both `i` and `j` are in `dims`, and only the velocities used by the
kept components are filtered. The default `dims = (1, 2, 3)` returns the full tensor; `dims = (1, 3)`
returns the `x`–`z` subset (`τ₁₁`, `τ₃₃`, `τ₁₃`).

A convenience method `subfilter_stress_tensor(model; σ, dims, boundary, N, collocate_diagonals)` builds
the Gaussian `filter` for you from a standard deviation `σ` (a Gaussian of full width at half maximum
`ℓ` has `σ = ℓ / (2√(2 ln 2))`).
"""
function subfilter_stress_tensor(model, filter; dims = (1, 2, 3), collocate_diagonals = false)
    FlowDiagnostics.validate_dims(dims)
    grid = model.grid
    u, v, w = model.velocities
    ū, v̄, w̄ = filtered_velocities(filter, dims, u, v, w)
    return subfilter_stress_tensor(filter, grid, u, v, w, ū, v̄, w̄; dims, collocate_diagonals)
end

subfilter_stress_tensor(model; σ, dims = (1, 2, 3), boundary = :shrink, N = nothing, collocate_diagonals = false) =
    subfilter_stress_tensor(model, GaussianFilter(; dims, σ, boundary, N); dims, collocate_diagonals)
#---

#+++ Cross-scale kinetic-energy flux
# Πₖ = -τⁱʲ S̄ⁱʲ contracted at cell centers. τ and S̄ share keys/ordering (both built with the same
# `dims`), so we pair them component-by-component and weight the off-diagonals by 2 (tensor symmetry).
# Each component is interpolated to (Center, Center, Center) before multiplying, matching the offline
# postprocessing convention.
to_center(ψ) = @at (Center, Center, Center) ψ

const _CONTRACTION = ((:τ₁₁, :S₁₁, 1), (:τ₂₂, :S₂₂, 1), (:τ₃₃, :S₃₃, 1),
                      (:τ₁₂, :S₁₂, 2), (:τ₁₃, :S₁₃, 2), (:τ₂₃, :S₂₃, 2))

function _cross_scale_ke_flux(τ, S̄)
    terms = (weight * to_center(τ[kτ]) * to_center(S̄[kS]) for (kτ, kS, weight) in _CONTRACTION if haskey(τ, kτ))
    return -sum(terms)
end

# Expose the flux as a single `KernelFunctionOperation` so it displays like the other diagnostics (via
# `@diagnostic_show` in `Oceanostics`) and composes inside larger operation trees. The per-cell kernel
# just evaluates the contraction operation `Πᵏ` built above; `Πᵏ`'s leaves are the materialized filtered
# `Field`s, so it only reads those fields and does arithmetic, never re-filtering. That "index a
# pre-assembled ccc operation" body is the shared `index_operation_ccc`, tagged here with the `:Πₖ`
# `NamedKernel` label to give this diagnostic its own type alias and display.
const KineticEnergyCrossScaleFlux = CustomKFO{<:NamedKernel{:Πₖ}}
const CrossScaleFlux = KineticEnergyCrossScaleFlux

"""
    $(SIGNATURES)

Return the cross-scale (scale-to-scale) kinetic-energy flux `Πₖ`, the rate at which a low-pass
`filter` transfers kinetic energy from the filtered to the subfilter scales (Aluie et al., 2018,
*J. Phys. Oceanogr.*, doi:10.1175/JPO-D-17-0100.1):

```
    Πₖ = -τⁱʲ S̄ⁱʲ
```

where `τⁱʲ = filter(uⁱuʲ) - ūⁱūʲ` is the subfilter-scale stress tensor ([`subfilter_stress_tensor`](@ref))
and `S̄ⁱʲ = ½(∂ūⁱ/∂xʲ + ∂ūʲ/∂xⁱ)` is the strain rate tensor of the filtered velocity
([`StrainRateTensor`](@ref) applied to `ūⁱ`). The contraction is evaluated at `(Center, Center,
Center)`, with off-diagonal components counted twice by symmetry. `Πₖ > 0` is forward (downscale,
filtered → subfilter) transfer. The result is per unit mass (units `m² s⁻³`); multiply by a reference
density `ρ₀` for a volumetric power.

`filter` is any callable mapping a field to its filtered counterpart, e.g. a reusable
[`GaussianFilter`](@ref):

```jldoctest
using Oceananigans, Oceanostics

grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1), topology=(Periodic, Periodic, Bounded))
model = NonhydrostaticModel(grid)

ℓ = 0.2  # filter scale (full width at half maximum)
filter = GaussianFilter(; dims=(1, 2, 3), σ=ℓ / (2√(2log(2))))

KineticEnergyCrossScaleFlux(model, filter)

# output

KineticEnergyCrossScaleFlux KernelFunctionOperation at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: index_operation_ccc (generic function with 1 method)
└── arguments: ("Oceananigans.AbstractOperations.UnaryOperation",)
└── computes: cross-scale kinetic energy flux  Πₖ = -τⁱʲS̄ⁱʲ
```

The returned object is a lazy operation over internally materialized filtered `Field`s, so it is
ready for `Field`, `Integral`, and `OutputWriter`s and recomputes as the simulation evolves.

`dims` selects which directions enter the tensors (only their `i,j` components are summed, and only
the velocities they use are filtered): the default `dims = (1, 2, 3)` gives the full 3D flux, while
`dims = (1, 3)` gives the 2D `x`–`z` flux `Πₖ = -(τ₁₁S̄₁₁ + τ₃₃S̄₃₃ + 2τ₁₃S̄₁₃)`. The filter's own
directions are set independently inside `filter`, so you can filter horizontally yet contract the
full tensor.

A convenience method `KineticEnergyCrossScaleFlux(model; σ, dims, boundary, N)` builds the Gaussian
`filter` for you from a standard deviation `σ` (with `σ = ℓ / (2√(2 ln 2))` for a FWHM `ℓ`).
"""
function KineticEnergyCrossScaleFlux(model, filter; dims = (1, 2, 3))
    FlowDiagnostics.validate_dims(dims)
    grid = model.grid
    u, v, w = model.velocities
    ū, v̄, w̄ = filtered_velocities(filter, dims, u, v, w)

    # Strain S̄ⁱʲ of the filtered velocities, and the subfilter stress τⁱʲ. The
    # contraction interpolates every component to cell centers, so the components can stay at their
    # natural staggered locations here; the result is wrapped in a `KernelFunctionOperation`.
    S̄ = StrainRateTensor(grid, ū, v̄, w̄; dims)
    τ = subfilter_stress_tensor(filter, grid, u, v, w, ū, v̄, w̄; dims)
    kernel = NamedKernel{:Πₖ}(index_operation_ccc)
    return KernelFunctionOperation{Center, Center, Center}(kernel, grid, _cross_scale_ke_flux(τ, S̄))
end

KineticEnergyCrossScaleFlux(model; σ, dims = (1, 2, 3), boundary = :shrink, N = nothing) =
    KineticEnergyCrossScaleFlux(model, GaussianFilter(; dims, σ, boundary, N); dims)
#---

#+++ Filtered-flow kinetic-energy dissipation
# εˡ = ∂ⱼūᵢ·τ̄ᵢⱼ, the dissipation of the filtered flow: the filtered velocity gradient ∂ⱼūᵢ contracted with
# the *filtered* viscous flux τ̄ᵢⱼ = filter(τᵢⱼ(u)). τᵢⱼ(u) is the model's viscous momentum flux built from
# the FULL velocities and closure (the same `viscous_flux_uᵢxⱼ` that `KineticEnergyDissipationRate`
# contracts), and it is low-pass filtered. Filtering the flux — rather than recomputing it from ūᵢ — is
# what makes this correct when the viscosity is non-uniform; the two coincide only for a constant
# viscosity, where the filter commutes with the (then-linear) flux.
#
# Per-component -Aⱼ·δⱼūᵢ·τ̄ᵢⱼ at each flux location, mirroring the `KineticEnergyDissipationRate` helpers
# but with the filtered velocity ūᵢ in the gradient and the pre-filtered flux field τ̄ᵢⱼ (read directly)
# in place of the inline `viscous_flux_uᵢxⱼ` call.
δū_τ̄₁₁(i, j, k, grid, ū, τ̄) = -Axᶜᶜᶜ(i, j, k, grid) * δxᶜᵃᵃ(i, j, k, grid, ū) * @inbounds(τ̄[i, j, k])
δū_τ̄₁₂(i, j, k, grid, ū, τ̄) = -Ayᶠᶠᶜ(i, j, k, grid) * δyᵃᶠᵃ(i, j, k, grid, ū) * @inbounds(τ̄[i, j, k])
δū_τ̄₁₃(i, j, k, grid, ū, τ̄) = -Azᶠᶜᶠ(i, j, k, grid) * δzᵃᵃᶠ(i, j, k, grid, ū) * @inbounds(τ̄[i, j, k])
δv̄_τ̄₂₁(i, j, k, grid, v̄, τ̄) = -Axᶠᶠᶜ(i, j, k, grid) * δxᶠᵃᵃ(i, j, k, grid, v̄) * @inbounds(τ̄[i, j, k])
δv̄_τ̄₂₂(i, j, k, grid, v̄, τ̄) = -Ayᶜᶜᶜ(i, j, k, grid) * δyᵃᶜᵃ(i, j, k, grid, v̄) * @inbounds(τ̄[i, j, k])
δv̄_τ̄₂₃(i, j, k, grid, v̄, τ̄) = -Azᶜᶠᶠ(i, j, k, grid) * δzᵃᵃᶠ(i, j, k, grid, v̄) * @inbounds(τ̄[i, j, k])
δw̄_τ̄₃₁(i, j, k, grid, w̄, τ̄) = -Axᶠᶜᶠ(i, j, k, grid) * δxᶠᵃᵃ(i, j, k, grid, w̄) * @inbounds(τ̄[i, j, k])
δw̄_τ̄₃₂(i, j, k, grid, w̄, τ̄) = -Ayᶜᶠᶠ(i, j, k, grid) * δyᵃᶠᵃ(i, j, k, grid, w̄) * @inbounds(τ̄[i, j, k])
δw̄_τ̄₃₃(i, j, k, grid, w̄, τ̄) = -Azᶜᶜᶜ(i, j, k, grid) * δzᵃᵃᶜ(i, j, k, grid, w̄) * @inbounds(τ̄[i, j, k])

# fv = (u=ū, v=v̄, w=w̄) filtered velocities; ff = (τ̄₁₁, …, τ̄₃₃) pre-filtered full-flow viscous fluxes. Each
# off-diagonal term is interpolated from its flux location to ccc exactly as in
# `viscous_dissipation_rate_ccc`; the /V paired with the A·δ makes the gradient a proper derivative.
@inline coarse_grained_dissipation_rate_ccc(i, j, k, grid, fv, ff) =
    (δū_τ̄₁₁(i, j, k, grid, fv.u, ff.τ̄₁₁) +
     ℑxyᶜᶜᵃ(i, j, k, grid, δū_τ̄₁₂, fv.u, ff.τ̄₁₂) +
     ℑxzᶜᵃᶜ(i, j, k, grid, δū_τ̄₁₃, fv.u, ff.τ̄₁₃) +

     ℑxyᶜᶜᵃ(i, j, k, grid, δv̄_τ̄₂₁, fv.v, ff.τ̄₂₁) +
     δv̄_τ̄₂₂(i, j, k, grid, fv.v, ff.τ̄₂₂) +
     ℑyzᵃᶜᶜ(i, j, k, grid, δv̄_τ̄₂₃, fv.v, ff.τ̄₂₃) +

     ℑxzᶜᵃᶜ(i, j, k, grid, δw̄_τ̄₃₁, fv.w, ff.τ̄₃₁) +
     ℑyzᵃᶜᶜ(i, j, k, grid, δw̄_τ̄₃₂, fv.w, ff.τ̄₃₂) +
     δw̄_τ̄₃₃(i, j, k, grid, fv.w, ff.τ̄₃₃)) / Vᶜᶜᶜ(i, j, k, grid)

const FilteredKineticEnergyDissipationRate = CustomKFO{<:typeof(coarse_grained_dissipation_rate_ccc)}
const DissipationRate = FilteredKineticEnergyDissipationRate

"""
    $(SIGNATURES)

Return the filtered-flow kinetic-energy dissipation rate `εˡ`, the rate at which
viscosity removes kinetic energy from the *filtered* velocity field `ūᵢ = filter(uᵢ)`:

```
    εˡ = ∂ⱼūᵢ · τ̄ᵢⱼ ,   τ̄ᵢⱼ = filter(τᵢⱼ(u))
```

Here `τᵢⱼ(u)` is the model's viscous momentum-flux tensor built from the **full** velocities and closure
(the same fluxes [`KineticEnergyDissipationRate`](@ref Oceanostics.KineticEnergyEquation.DissipationRate)
contracts), and `τ̄ᵢⱼ = filter(τᵢⱼ(u))` is that flux low-pass filtered. Contracting the filtered flux with
the filtered velocity gradient gives the viscous sink in the budget of the filtered kinetic energy
`Kˡ = ½ūᵢūᵢ` (coarse-graining framework of Aluie et al., 2018, *J. Phys. Oceanogr.*,
doi:10.1175/JPO-D-17-0100.1).

Note the flux is filtered, `filter(τᵢⱼ(u))`, not recomputed from the filtered velocity, `τᵢⱼ(ū)`. The two
agree only for a constant, uniform viscosity, where the filter commutes with the flux; they differ once
the viscosity varies in space (e.g. an eddy viscosity), and only the filtered-flux form is the
dissipation that appears in the filtered KE budget. For a constant-viscosity `ScalarDiffusivity` it
reduces to `2ν S̄ᵢⱼS̄ᵢⱼ`, the dissipation of the resolved strain. It is evaluated at `(Center, Center,
Center)`, per unit mass (units `m² s⁻³`); multiply by a reference density `ρ₀` for a volumetric power.

`filter` is any callable mapping a field to its filtered counterpart, e.g. a reusable
[`GaussianFilter`](@ref):

```jldoctest
using Oceananigans, Oceanostics

grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1), topology=(Periodic, Periodic, Bounded))
model = NonhydrostaticModel(grid; closure=ScalarDiffusivity(ν=1e-4))

ℓ = 0.2  # filter scale (full width at half maximum)
filter = GaussianFilter(; dims=(1, 2, 3), σ=ℓ / (2√(2log(2))))

FilteredKineticEnergyDissipationRate(model, filter)

# output

FilteredKineticEnergyDissipationRate KernelFunctionOperation at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: coarse_grained_dissipation_rate_ccc (generic function with 1 method)
└── arguments: ("NamedTuple", "NamedTuple")
└── computes: filtered kinetic energy dissipation rate  εˡ = ∂ⱼūᵢ·τ̄ᵢⱼ
```

The viscosity and fluxes come from `model.closure`/`model.closure_fields`, exactly as in
[`KineticEnergyDissipationRate`](@ref Oceanostics.KineticEnergyEquation.DissipationRate), so the model
needs a closure whose viscous fluxes are defined. The filtered velocities and the filtered fluxes are
materialized as `Field`s internally (and refreshed on recompute), so the returned object is a lazy
operation ready for `Field`, `Integral`, and `OutputWriter`s and recomputes as the simulation evolves.

Unlike the cross-scale flux and the stress tensor, this diagnostic takes no `dims` argument: it always
forms the full viscous contraction (matching
[`KineticEnergyDissipationRate`](@ref Oceanostics.KineticEnergyEquation.DissipationRate)). The directions
the filter acts in are set inside `filter`.

A convenience method `FilteredKineticEnergyDissipationRate(model; σ, dims, boundary, N)` builds the
Gaussian `filter` for you from a standard deviation `σ` (with `σ = ℓ / (2√(2 ln 2))` for a FWHM `ℓ`);
`dims` here selects the directions the Gaussian filter acts in.
"""
function FilteredKineticEnergyDissipationRate(model, filter)
    grid = model.grid
    u, v, w = model.velocities

    # Filtered velocities ūᵢ for the gradient ∂ⱼūᵢ, materialized so the separable filter takes its fast
    # staged path.
    fv = (u = Field(filter(u)), v = Field(filter(v)), w = Field(filter(w)))

    # τ̄ᵢⱼ = filter(τᵢⱼ(u)): the model's full-flow viscous fluxes (from the FULL velocities and closure),
    # each low-pass filtered and materialized at its staggered location. The flux operation reads the live
    # model fields, so both `fv` and `ff` refresh when the diagnostic is recomputed.
    flux_args = (model.closure, model.closure_fields, model.clock, fields(model), model.buoyancy)
    filtered_flux(f, LX, LY, LZ) = Field(filter(KernelFunctionOperation{LX, LY, LZ}(f, grid, flux_args...)))
    ff = (τ̄₁₁ = filtered_flux(viscous_flux_ux, Center, Center, Center),
          τ̄₁₂ = filtered_flux(viscous_flux_uy, Face,   Face,   Center),
          τ̄₁₃ = filtered_flux(viscous_flux_uz, Face,   Center, Face),
          τ̄₂₁ = filtered_flux(viscous_flux_vx, Face,   Face,   Center),
          τ̄₂₂ = filtered_flux(viscous_flux_vy, Center, Center, Center),
          τ̄₂₃ = filtered_flux(viscous_flux_vz, Center, Face,   Face),
          τ̄₃₁ = filtered_flux(viscous_flux_wx, Face,   Center, Face),
          τ̄₃₂ = filtered_flux(viscous_flux_wy, Center, Face,   Face),
          τ̄₃₃ = filtered_flux(viscous_flux_wz, Center, Center, Center))
    return KernelFunctionOperation{Center, Center, Center}(coarse_grained_dissipation_rate_ccc, grid, fv, ff)
end

FilteredKineticEnergyDissipationRate(model; σ, dims = (1, 2, 3), boundary = :shrink, N = nothing) =
    FilteredKineticEnergyDissipationRate(model, GaussianFilter(; dims, σ, boundary, N))
#---

end # module
