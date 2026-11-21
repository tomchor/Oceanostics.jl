module FilteredKineticEnergyEquation

using DocStringExtensions
using DocStringExtensions: MethodSignatures

export FilteredKineticEnergy
export subfilter_stress_tensor, KineticEnergyCrossScaleFlux, CrossScaleFlux
export FilteredKineticEnergyDissipationRate, DissipationRate

using Oceananigans: fields
using Oceananigans.Grids: Center, Face
using Oceananigans.Fields: Field, ZeroField
using Oceananigans.Operators
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.TurbulenceClosures: viscous_flux_ux, viscous_flux_uy, viscous_flux_uz,
                                       viscous_flux_vx, viscous_flux_vy, viscous_flux_vz,
                                       viscous_flux_wx, viscous_flux_wy, viscous_flux_wz

using Oceanostics: CustomKFO, validate_closure_supplies_a_flux, prognostic_velocities, prognostic_w_row
using ..FlowDiagnostics: StressTensor, StrainRateTensor, to_center
import ..FlowDiagnostics            # for the (unexported) `validate_dims`
using ..SpatialFilters: GaussianFilter, BoxFilter   # BoxFilter is imported so its docstring `@ref` resolves in-module
using ..KineticEnergyEquation: kinetic_energy_ccc   # reuse the ½uᵢuᵢ kernel, applied to the filtered velocities

#+++ Shared helpers
# Filter only the velocities that the requested `dims` actually use: component τᵢⱼ / S̄ᵢⱼ needs uᵢ and
# uⱼ, so velocity component d is filtered iff d ∈ dims; the others are passed through untouched (they
# never enter the kept tensor components). Each filtered velocity is materialized as a `Field` so the
# separable filter takes its fast staged path and is computed once and reused by both tensors. A
# `ZeroField`, the `w` that a hydrostatic model's budgets drop (see `prognostic_velocities`), filters
# to itself.
filtered_velocity(filter, ψ) = Field(filter(ψ))
filtered_velocity(filter, ψ::ZeroField) = ψ

function filtered_velocities(filter, dims, u, v, w)
    ū = (1 in dims) ? filtered_velocity(filter, u) : u
    v̄ = (2 in dims) ? filtered_velocity(filter, v) : v
    w̄ = (3 in dims) ? filtered_velocity(filter, w) : w
    return ū, v̄, w̄
end

# Low-level method of `subfilter_stress_tensor` taking already-filtered velocities `ū, v̄, w̄`, so the
# (expensive) velocity filtering can be done once and shared — `KineticEnergyCrossScaleFlux` reuses
# `ū, v̄, w̄` for both the strain rate and this stress. Computes τᵢⱼ = filter(uᵢuⱼ) - ūᵢūⱼ component by
# component, reusing `StressTensor` to build the momentum-flux tensor uᵢuⱼ of both the full and the
# filtered velocity; `filter` is applied to the (materialized) full flux and the filtered flux is
# subtracted. The result is a `NamedTuple` with the same keys/locations as `StressTensor`, less the
# components named in `without`, which are never filtered.
function subfilter_stress_tensor(filter, grid, u, v, w, ū, v̄, w̄; dims, collocate_diagonals=false, without=())
    flux_full = StressTensor(grid, u, v, w; dims, collocate_diagonals)   # uᵢuⱼ
    flux_filt = StressTensor(grid, ū, v̄, w̄; dims, collocate_diagonals)   # ūᵢūⱼ
    subfilter(full, filt) = Field(filter(Field(full))) - filt            # filter(uᵢuⱼ) - ūᵢūⱼ
    ks = Tuple(k for k in keys(flux_full) if !(k in without))
    return NamedTuple{ks}(map(k -> subfilter(flux_full[k], flux_filt[k]), ks))
end
#---

#+++ Filtered kinetic energy
# eₖˡ = ½ ūᵢūᵢ, the kinetic energy of the filtered velocity field ūᵢ = filter(uᵢ). It reuses
# `KineticEnergyEquation`'s `kinetic_energy_ccc` kernel (½uᵢuᵢ interpolated to ccc); wrapping it under a
# distinct kernel name gives `FilteredKineticEnergy` its own type alias and `@diagnostic_show` display.
@inline filtered_kinetic_energy_ccc(i, j, k, grid, ū, v̄, w̄) = kinetic_energy_ccc(i, j, k, grid, ū, v̄, w̄)

const FilteredKineticEnergy = CustomKFO{<:typeof(filtered_kinetic_energy_ccc)}

"""
    $(SIGNATURES)

Return the kinetic energy of the filtered flow `eₖˡ`, the kinetic energy carried by the
scales that a low-pass `filter` keeps:

```
    eₖˡ = ½ ūᵢ ūᵢ = ½ (ū² + v̄² + w̄²) ,   ūᵢ = filter(uᵢ)
```

It is the filtered counterpart of the subfilter kinetic energy `eₖˢ = ½τᵢᵢ` (`SubFilterKineticEnergy`):
the filter splits the flow's kinetic energy into the part it keeps (`eₖˡ`) and the part it removes (`eₖˢ`).
On a `HydrostaticFreeSurfaceModel`, which has no `w`-momentum equation, the sum runs over `ū` and `v̄`
only, so `eₖˡ = ½(ū² + v̄²)`, as the kinetic energy is `½(u² + v²)` there.

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
├── kernel_function: filtered_kinetic_energy_ccc (generic function with 1 method)
└── arguments: ("Field", "Field", "Field")
└── computes: kinetic energy of the filtered flow  ½ūᵢūᵢ
```

A convenience method `FilteredKineticEnergy(model; σ, dims, boundary, N)` builds the Gaussian `filter`
for you from a standard deviation `σ` (with `σ = ℓ / (2√(2 ln 2))` for a FWHM `ℓ`).
"""
function FilteredKineticEnergy(model, filter)
    u, v, w = prognostic_velocities(model)                       # w is a `ZeroField` on a hydrostatic model
    ū, v̄, w̄ = filtered_velocities(filter, (1, 2, 3), u, v, w)   # materialize the filtered velocities
    return KernelFunctionOperation{Center, Center, Center}(filtered_kinetic_energy_ccc, model.grid, ū, v̄, w̄)
end

FilteredKineticEnergy(model; σ, dims = (1, 2, 3), boundary = :shrink, N = nothing) = FilteredKineticEnergy(model, GaussianFilter(; dims, σ, boundary, N))
#---

#+++ Subfilter stress tensor
"""
    $(MethodSignatures(; defaults = true))

Return the components of the subfilter-scale (SFS) stress tensor `τ`, the residual momentum flux that
a low-pass `filter` removes from the filtered scales:

```
    τᵢⱼ = filter(uᵢuⱼ) - ūᵢ ūⱼ ,   ūᵢ = filter(uᵢ)
```

(also called the sub-grid stress in LES, or the generalized central moment in the filtering
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
`ccc`). The filtered velocities `ūᵢ` and the filtered momentum fluxes `filter(uᵢuⱼ)` are materialized
as `Field`s internally (the filter's fast staged path only fires when wrapped directly in a `Field`), so
each returned component is a lazy operation over those computed fields and recomputes correctly when
written by an `OutputWriter`.

Half the trace of the collocated diagonals squares each velocity after interpolating it, so it differs
at the grid scale from
[`SubFilterKineticEnergy`](@ref Oceanostics.SubFilterKineticEnergyEquation.SubFilterKineticEnergy),
which squares each velocity on its own face as `KineticEnergy` does. Use the latter in a budget (#342).

`dims` selects which spatial directions enter the tensor, exactly as in [`StressTensor`](@ref):
component `τᵢⱼ` is kept only when both `i` and `j` are in `dims`, and only the velocities used by the
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
# Πₖ = -τᵢⱼ∂ⱼūᵢ contracted at cell centers, over the rows i the model has a momentum equation for and
# the columns j in `dims`. With every row kept, τ's symmetry makes it the strain contraction -τᵢⱼS̄ᵢⱼ: τ
# and S̄ share keys/ordering (both built with the same `dims`), so we pair them component-by-component and
# weight the off-diagonals by 2. Each component is interpolated to (Center, Center, Center) before
# multiplying (via `FlowDiagnostics`' shared `to_center`), matching the offline postprocessing convention.
const _CONTRACTION = ((:τ₁₁, :S₁₁, 1), (:τ₂₂, :S₂₂, 1), (:τ₃₃, :S₃₃, 1),
                      (:τ₁₂, :S₁₂, 2), (:τ₁₃, :S₁₃, 2), (:τ₂₃, :S₂₃, 2))

strain_contraction_terms(τ, S̄) =
    Tuple(weight * to_center(τ[kτ]) * to_center(S̄[kS]) for (kτ, kS, weight) in _CONTRACTION if haskey(S̄, kS))

cross_scale_ke_flux_terms(::Val{true}, filter, grid, dims, u, v, w, ū, v̄, w̄) =
    strain_contraction_terms(subfilter_stress_tensor(filter, grid, u, v, w, ū, v̄, w̄; dims),
                             StrainRateTensor(grid, ū, v̄, w̄; dims))

# A model with no `w` row (see `prognostic_velocities`) keeps the rows i ∈ dims ∩ (1, 2) and the
# columns j ∈ dims. Its i, j ≤ 2 block is the strain contraction. Its j = 3 column, -τᵢ₃∂zūᵢ, is the
# vertical advection of horizontal momentum across scales (#342); the S̄ᵢ₃ of the strain contraction
# would also bring in ∂ᵢw̄, from the `w` row. Each τᵢ₃∂zūᵢ is formed at τᵢ₃'s edge, where the model
# forms the vertical flux of uᵢ, and the product is interpolated to (Center, Center, Center). τ₃₃
# belongs to the `w` row alone, so it is never filtered.
function cross_scale_ke_flux_terms(::Val{false}, filter, grid, dims, u, v, w, ū, v̄, w̄)
    rows = Tuple(d for d in dims if d != 3)
    isempty(rows) && throw(ArgumentError("On a model without a `w`-momentum equation, such as a `HydrostaticFreeSurfaceModel`, \
                                          the cross-scale flux contracts the rows of `u` and `v` only, so `dims` \
                                          must include 1 or 2; got dims = $dims"))
    τ = subfilter_stress_tensor(filter, grid, u, v, w, ū, v̄, w̄; dims, without = (:τ₃₃,))
    terms = strain_contraction_terms(τ, StrainRateTensor(grid, ū, v̄, w̄; dims = rows))
    3 in dims || return terms
    1 in rows && (terms = (terms..., to_center(τ.τ₁₃ * KernelFunctionOperation{Face, Center, Face}(∂zᶠᶜᶠ, grid, ū))))
    2 in rows && (terms = (terms..., to_center(τ.τ₂₃ * KernelFunctionOperation{Center, Face, Face}(∂zᶜᶠᶠ, grid, v̄))))
    return terms
end

# Expose the flux as a single `KernelFunctionOperation` so it displays like the other diagnostics (via
# `@diagnostic_show` in `Oceanostics`) and composes inside larger operation trees. The kernel just
# evaluates the contraction operation `Πᵏ` built above; `Πᵏ`'s leaves are the materialized filtered
# `Field`s, so this per-cell evaluation only reads those fields and does arithmetic — it never re-filters.
@inline cross_scale_ke_flux_ccc(i, j, k, grid, Πᵏ) = @inbounds Πᵏ[i, j, k]

const KineticEnergyCrossScaleFlux = CustomKFO{<:typeof(cross_scale_ke_flux_ccc)}
const CrossScaleFlux = KineticEnergyCrossScaleFlux

"""
    $(MethodSignatures(; defaults = true))

Return the cross-scale (scale-to-scale) kinetic-energy flux `Πₖ`, the rate at which a low-pass
`filter` transfers kinetic energy from the filtered to the subfilter scales (Aluie et al., 2018,
*J. Phys. Oceanogr.*, doi:10.1175/JPO-D-17-0100.1):

```
    Πₖ = -τᵢⱼ S̄ᵢⱼ
```

where `τᵢⱼ = filter(uᵢuⱼ) - ūᵢūⱼ` is the subfilter-scale stress tensor ([`subfilter_stress_tensor`](@ref))
and `S̄ᵢⱼ = ½(∂ūᵢ/∂xⱼ + ∂ūⱼ/∂xᵢ)` is the strain rate tensor of the filtered velocity
([`StrainRateTensor`](@ref) applied to `ūᵢ`). The contraction is evaluated at `(Center, Center,
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
├── kernel_function: cross_scale_ke_flux_ccc (generic function with 1 method)
└── arguments: ("UnaryOperation",)
└── computes: cross-scale kinetic energy flux  -τᵢⱼS̄ᵢⱼ
```

The returned object is a lazy operation over internally materialized filtered `Field`s, so it is
ready for `Field`, `Integral`, and `OutputWriter`s and recomputes as the simulation evolves.

`dims` selects which directions enter the tensors (only their `i,j` components are summed, and only
the velocities they use are filtered): the default `dims = (1, 2, 3)` gives the full 3D flux, while
`dims = (1, 3)` gives the 2D `x`–`z` flux `Πₖ = -(τ₁₁S̄₁₁ + τ₃₃S̄₃₃ + 2τ₁₃S̄₁₃)`. The filter's own
directions are set independently inside `filter`, so you can filter horizontally yet contract the
full tensor.

On a `HydrostaticFreeSurfaceModel` the filtered kinetic energy is `½(ū² + v̄²)`: the model steps `u`
and `v` and diagnoses `w`, which has no momentum equation but still advects `u` and `v`. The flux is
then `Πₖ = -τᵢⱼ ∂ⱼūᵢ`, summed over the rows `i` in `dims` other than 3 and the columns `j` in `dims`, so
the default `dims = (1, 2, 3)` gives

```
    Πₖ = -(τ₁₁S̄₁₁ + τ₂₂S̄₂₂ + 2τ₁₂S̄₁₂) - (τ₁₃ ∂zū + τ₂₃ ∂zv̄)
```

The second group is the vertical advection of horizontal momentum across scales. It takes the place of
the `2τ₁₃S̄₁₃ + 2τ₂₃S̄₂₃` of the full flux, whose `∂xw̄` and `∂yw̄` parts, like `τ₃₃S̄₃₃`, belong to the
`w` row. Each of its products is formed at the edge where `τ₁₃` or `τ₂₃` lives, which is where the model
computes the vertical flux of `u` or `v`, and then interpolated to `(Center, Center, Center)`.
`dims = (1, 2)` gives the first group alone, which is not the flux of `½(ū² + v̄²)` (#342), and `dims`
must include 1 or 2.

The contraction is a second-order discretization that does not depend on the advection scheme, so it
does not reproduce the model's own discrete advective exchange between the filtered and subfilter
scales. The two differ at the grid scale. When the filter width is a few grid spacings, that difference
can be a sizable fraction of the domain integral of `Πₖ`, which is a small difference between forward
and backward transfer (#342).

A convenience method `KineticEnergyCrossScaleFlux(model; σ, dims, boundary, N)` builds the Gaussian
`filter` for you from a standard deviation `σ` (with `σ = ℓ / (2√(2 ln 2))` for a FWHM `ℓ`).
"""
function KineticEnergyCrossScaleFlux(model, filter; dims = (1, 2, 3))
    FlowDiagnostics.validate_dims(dims)
    grid = model.grid
    u, v, w = model.velocities
    ū, v̄, w̄ = filtered_velocities(filter, dims, u, v, w)

    # Each term pairs a component of the subfilter stress τᵢⱼ with the gradient of the filtered velocity
    # it is contracted with. The components stay at their natural staggered locations until the
    # contraction moves them to cell centers, and the sum is wrapped in a `KernelFunctionOperation`.
    terms = cross_scale_ke_flux_terms(prognostic_w_row(model), filter, grid, dims, u, v, w, ū, v̄, w̄)
    return KernelFunctionOperation{Center, Center, Center}(cross_scale_ke_flux_ccc, grid, -sum(terms))
end

KineticEnergyCrossScaleFlux(model; σ, dims = (1, 2, 3), boundary = :shrink, N = nothing) =
    KineticEnergyCrossScaleFlux(model, GaussianFilter(; dims, σ, boundary, N); dims)
#---

#+++ Filtered-flow kinetic-energy dissipation
# εₖˡ = -∂ⱼūᵢ·τ̄ᵢⱼ, the dissipation of the filtered flow: the filtered velocity gradient ∂ⱼūᵢ contracted with
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

# A hydrostatic model has no `w` row (see `prognostic_velocities`): its `w̄` and its fluxes τ̄₃ⱼ are
# `ZeroField`s, and the three terms are skipped without reading them
δw̄_τ̄₃₁(i, j, k, grid, ::ZeroField, τ̄) = zero(grid)
δw̄_τ̄₃₂(i, j, k, grid, ::ZeroField, τ̄) = zero(grid)
δw̄_τ̄₃₃(i, j, k, grid, ::ZeroField, τ̄) = zero(grid)

# fv = (u=ū, v=v̄, w=w̄) filtered velocities; ff = (τ̄₁₁, …, τ̄₃₃) pre-filtered full-flow viscous fluxes. Each
# off-diagonal term is interpolated from its flux location to ccc exactly as in
# `viscous_dissipation_rate_ccc`; the /V paired with the A·δ makes the gradient a proper derivative.
@inline filtered_dissipation_rate_ccc(i, j, k, grid, fv, ff) =
    (δū_τ̄₁₁(i, j, k, grid, fv.u, ff.τ̄₁₁) +
     ℑxyᶜᶜᵃ(i, j, k, grid, δū_τ̄₁₂, fv.u, ff.τ̄₁₂) +
     ℑxzᶜᵃᶜ(i, j, k, grid, δū_τ̄₁₃, fv.u, ff.τ̄₁₃) +

     ℑxyᶜᶜᵃ(i, j, k, grid, δv̄_τ̄₂₁, fv.v, ff.τ̄₂₁) +
     δv̄_τ̄₂₂(i, j, k, grid, fv.v, ff.τ̄₂₂) +
     ℑyzᵃᶜᶜ(i, j, k, grid, δv̄_τ̄₂₃, fv.v, ff.τ̄₂₃) +

     ℑxzᶜᵃᶜ(i, j, k, grid, δw̄_τ̄₃₁, fv.w, ff.τ̄₃₁) +
     ℑyzᵃᶜᶜ(i, j, k, grid, δw̄_τ̄₃₂, fv.w, ff.τ̄₃₂) +
     δw̄_τ̄₃₃(i, j, k, grid, fv.w, ff.τ̄₃₃)) / Vᶜᶜᶜ(i, j, k, grid)

const FilteredKineticEnergyDissipationRate = CustomKFO{<:typeof(filtered_dissipation_rate_ccc)}
const DissipationRate = FilteredKineticEnergyDissipationRate

"""
    $(SIGNATURES)

Return the filtered-flow kinetic-energy dissipation rate `εₖˡ`, the rate at which
viscosity removes kinetic energy from the *filtered* velocity field `ūᵢ = filter(uᵢ)`:

```
    εₖˡ = -∂ⱼūᵢ · τ̄ᵢⱼ ,   τ̄ᵢⱼ = filter(τᵢⱼ(u))
```

Here `τᵢⱼ(u)` is the model's viscous momentum flux (minus the stress tensor) built from the **full**
velocities and closure
(the same fluxes [`KineticEnergyDissipationRate`](@ref Oceanostics.KineticEnergyEquation.DissipationRate)
contracts), and `τ̄ᵢⱼ = filter(τᵢⱼ(u))` is that flux low-pass filtered. Contracting the filtered flux with
the filtered velocity gradient gives the viscous sink in the budget of the filtered kinetic energy
`eₖˡ = ½ūᵢūᵢ` (filtering framework of Aluie et al., 2018, *J. Phys. Oceanogr.*,
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
├── kernel_function: filtered_dissipation_rate_ccc (generic function with 1 method)
└── arguments: ("NamedTuple", "NamedTuple")
└── computes: filtered kinetic energy dissipation rate  -∂ⱼūᵢ·τ̄ᵢⱼ
```

The viscosity and fluxes come from `model.closure`/`model.closure_fields`, exactly as in
[`KineticEnergyDissipationRate`](@ref Oceanostics.KineticEnergyEquation.DissipationRate), so the model
needs a closure whose viscous fluxes are defined. The filtered velocities and the filtered fluxes are
materialized as `Field`s internally (and refreshed on recompute), so the returned object is a lazy
operation ready for `Field`, `Integral`, and `OutputWriter`s and recomputes as the simulation evolves.

Unlike the cross-scale flux and the stress tensor, this diagnostic takes no `dims` argument: it
contracts every row the model has a momentum equation for, matching
[`KineticEnergyDissipationRate`](@ref Oceanostics.KineticEnergyEquation.DissipationRate). On a
`HydrostaticFreeSurfaceModel` that leaves out the `w` row, so `εₖˡ` is the dissipation of `½(ū² + v̄²)`,
while `τ̄₁ⱼ` and `τ̄₂ⱼ` are still the model's fluxes, built with its diagnosed `w`. The directions the
filter acts in are set inside `filter`.

A convenience method `FilteredKineticEnergyDissipationRate(model; σ, dims, boundary, N)` builds the
Gaussian `filter` for you from a standard deviation `σ` (with `σ = ℓ / (2√(2 ln 2))` for a FWHM `ℓ`);
`dims` here selects the directions the Gaussian filter acts in.
"""
function FilteredKineticEnergyDissipationRate(model, filter)
    validate_closure_supplies_a_flux("FilteredKineticEnergyDissipationRate", model)
    grid = model.grid

    # Filtered velocities ūᵢ for the gradient ∂ⱼūᵢ of each row, materialized so the separable filter takes
    # its fast staged path. A hydrostatic model has no `w` row, so its `w̄` is the `ZeroField` that
    # `prognostic_velocities` puts in place of `w`.
    fv = map(ψ -> filtered_velocity(filter, ψ), prognostic_velocities(model))

    # τ̄ᵢⱼ = filter(τᵢⱼ(u)): the model's full-flow viscous fluxes (from the FULL velocities and closure),
    # each low-pass filtered and materialized at its staggered location. The flux operation reads the live
    # model fields, so both `fv` and `ff` refresh when the diagnostic is recomputed. The fluxes of a
    # dropped `w` row are not filtered at all.
    flux_args = (model.closure, model.closure_fields, model.clock, fields(model), model.buoyancy)
    filtered_flux(f, LX, LY, LZ) = Field(filter(KernelFunctionOperation{LX, LY, LZ}(f, grid, flux_args...)))
    w_row_flux(f, LX, LY, LZ) = fv.w isa ZeroField ? ZeroField() : filtered_flux(f, LX, LY, LZ)
    ff = (τ̄₁₁ = filtered_flux(viscous_flux_ux, Center, Center, Center),
          τ̄₁₂ = filtered_flux(viscous_flux_uy, Face,   Face,   Center),
          τ̄₁₃ = filtered_flux(viscous_flux_uz, Face,   Center, Face),
          τ̄₂₁ = filtered_flux(viscous_flux_vx, Face,   Face,   Center),
          τ̄₂₂ = filtered_flux(viscous_flux_vy, Center, Center, Center),
          τ̄₂₃ = filtered_flux(viscous_flux_vz, Center, Face,   Face),
          τ̄₃₁ = w_row_flux(viscous_flux_wx,    Face,   Center, Face),
          τ̄₃₂ = w_row_flux(viscous_flux_wy,    Center, Face,   Face),
          τ̄₃₃ = w_row_flux(viscous_flux_wz,    Center, Center, Center))
    return KernelFunctionOperation{Center, Center, Center}(filtered_dissipation_rate_ccc, grid, fv, ff)
end

FilteredKineticEnergyDissipationRate(model; σ, dims = (1, 2, 3), boundary = :shrink, N = nothing) =
    FilteredKineticEnergyDissipationRate(model, GaussianFilter(; dims, σ, boundary, N))
#---

end # module
