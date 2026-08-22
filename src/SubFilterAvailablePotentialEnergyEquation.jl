module SubFilterAvailablePotentialEnergyEquation

using DocStringExtensions

export SubFilterAvailablePotentialEnergy
export SubFilterAvailablePotentialEnergyDissipationRate, DissipationRate
# The filtered-flow APE and its dissipation are the "ˡ" halves of the two splits below (and the sinks of
# the filtered budget), so they are re-exported here from `FilteredAvailablePotentialEnergyEquation`,
# where they are defined — as `SubFilterKineticEnergyEquation` re-exports `KineticEnergyCrossScaleFlux`.
export FilteredAvailablePotentialEnergy, FilteredAvailablePotentialEnergyDissipationRate
# Πₐ is a source term of the sub-filter APE budget (and a sink of the filtered one), so it is
# re-exported from the same module.
export AvailablePotentialEnergyCrossScaleFlux
# The shared reference profile both states are measured against is built with
# `BackgroundPotentialEnergyEquation`'s machinery, so the pieces needed to construct and share one are
# re-exported here and this module can be used on its own.
export reference_height, reference_buoyancy, VerticalSort, ProfileLookup

using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Fields: Field
using Oceananigans.Grids: Center
using Oceananigans.Models: model_geopotential_height

using Oceanostics: CustomKFO

using ..PotentialEnergyEquation: validate_buoyancy_is_a_diffused_tracer, validate_closure_supplies_a_flux,
                                 validate_gravity_is_z_aligned
using ..BackgroundPotentialEnergyEquation: reference_height, reference_buoyancy, VerticalSort, ProfileLookup
using ..AvailablePotentialEnergyEquation: AvailablePotentialEnergy, AvailablePotentialEnergyDissipationRate
using ..FilteredAvailablePotentialEnergyEquation: FilteredAvailablePotentialEnergy,
                                                  FilteredAvailablePotentialEnergyDissipationRate,
                                                  AvailablePotentialEnergyCrossScaleFlux,
                                                  filtered_buoyancy_and_lookup
# `GaussianFilter` builds the convenience methods' filter; `BoxFilter` is imported only so its docstring
# `@ref` resolves in-module.
using ..SpatialFilters: GaussianFilter, BoxFilter

#+++ Shared reference heights
# The two reference heights every sub-filter APE diagnostic needs: the full buoyancy `b` and the
# filtered buoyancy `b̄ = filter(b)`, each looked up in the same shared profile. The buoyancies and the
# lookup come from `FilteredAvailablePotentialEnergyEquation`, so the filtered-flow diagnostics built on
# `z✶ˡ` here measure against exactly the profile the full-field ones on `z✶` do — that shared profile is
# what makes `filter(eₐ) - eₐˡ` a decomposition rather than a difference of two unrelated quantities.
function subfilter_reference_heights(diagnostic, model, filter, method, geopotential_height)
    b, b̄, lookup = filtered_buoyancy_and_lookup(diagnostic, model, filter, method, geopotential_height)
    z✶  = reference_height(b;  method = lookup)
    z✶ˡ = reference_height(b̄; method = lookup)
    return z✶, z✶ˡ
end
#---

#+++ Sub-filter available potential energy
# eₐˢ = filter(eₐ) - eₐˡ, exposed with the same wrapper trick as `SubFilterKineticEnergyDissipationRate`:
# the kernel just indexes the pre-assembled operation, whose leaves are materialized `Field`s, so per-cell
# evaluation only reads those fields and subtracts — it never re-sorts or re-filters.
@inline subfilter_ape_ccc(i, j, k, grid, eₐˢ) = @inbounds eₐˢ[i, j, k]

const SubFilterAvailablePotentialEnergy = CustomKFO{<:typeof(subfilter_ape_ccc)}

"""
    $(SIGNATURES)

Return the sub-filter-scale (SFS) available potential energy `eₐˢ`, the available potential energy
carried by the scales that a low-pass `filter` removes from the buoyancy field — the filtered full APE
minus the APE of the filtered buoyancy `b̄ = filter(b)`:

```
    eₐˢ = filter(eₐ(b, z)) - eₐ(b̄, z) ,   eₐ(b, z) = ∫_{z✶(b)}^{z} [b✶(z̃) - b] dz̃
```

where `eₐ` is the local available potential energy density ([`AvailablePotentialEnergy`](@ref)),
`eₐ(b̄, z)` is the APE of the filtered buoyancy ([`FilteredAvailablePotentialEnergy`](@ref)), and both
terms are measured against **one shared reference profile** `(b✶, z✶)`, following the filtered APE
framework of [Wenegrat, Chor & Barkan (2026)](https://arxiv.org/abs/2605.15879): the filtered buoyancy
is looked up in the same profile the full field is measured against, which is exactly what
[`ProfileLookup`](@ref) was built for. It is the potential-energy counterpart of the sub-filter
kinetic energy `Kˢ` ([`SubFilterKineticEnergy`](@ref Oceanostics.SubFilterKineticEnergyEquation.SubFilterKineticEnergy)),
just as [`FilteredAvailablePotentialEnergy`](@ref) is that of the filtered kinetic energy `Kˡ`.

Because the two states have to share one profile, `method` must be a [`ProfileLookup`](@ref):

  - `ProfileLookup()` (the default) sorts the profile from the model's own buoyancy — a
    [`VerticalSort`](@ref) column built internally and re-sorted on every `compute!`, so the reference
    state tracks the flow.
  - `ProfileLookup(z✶_column)` borrows a column you already built with
    `reference_height(model, method=VerticalSort())`, sharing its sort across several diagnostics.
  - `ProfileLookup(b✶, z✶)` with plain arrays holds the reference profile fixed in time, which also
    makes the diagnostic sort-free: each `compute!` is then a filter plus two binary-search lookups.

`eₐ` is convex in buoyancy (`∂²eₐ/∂b² = ∂z✶/∂b ≥ 0` on a stable profile), so when the filter has no
vertical component `eₐˢ ≥ 0` pointwise, by Jensen's inequality. A filter that acts vertically mixes
heights as well as buoyancies and can produce locally negative values, as can, marginally, filtered
buoyancies that fall between the profile's entries (the lookup then takes the nearest class).

`filter` is any callable mapping a field to its low-pass-filtered counterpart, e.g. a reusable
[`GaussianFilter`](@ref) or [`BoxFilter`](@ref). The filtered buoyancy and the filtered APE are
materialized as `Field`s internally (so the separable filter takes its fast staged path), and the
returned object is a lazy operation over them, ready for `Field`, `Integral` and `OutputWriter`s. It
lives at `(Center, Center, Center)`, per unit mass (units `m² s⁻²`):

```jldoctest
using Oceananigans, Oceanostics

grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1), topology=(Periodic, Periodic, Bounded))
model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b)

filter = GaussianFilter(; dims=(1, 2, 3), σ=0.1)
SubFilterAvailablePotentialEnergy(model, filter)

# output

SubFilterAvailablePotentialEnergy KernelFunctionOperation at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: subfilter_ape_ccc (generic function with 1 method)
└── arguments: ("Oceananigans.AbstractOperations.BinaryOperation",)
└── computes: sub-filter available potential energy  eₐˢ = filter(eₐ) - eₐˡ
```

A convenience method `SubFilterAvailablePotentialEnergy(model; σ, dims, boundary, N)` builds the
Gaussian `filter` for you from a standard deviation `σ` (with `σ = ℓ / (2√(2 ln 2))` for a FWHM `ℓ`).
`geopotential_height` enters the buoyancy construction exactly as in [`reference_height`](@ref).
"""
function SubFilterAvailablePotentialEnergy(model, filter; method = ProfileLookup(),
                                           geopotential_height = model_geopotential_height(model))
    validate_gravity_is_z_aligned("SubFilterAvailablePotentialEnergy", model)
    z✶, z✶ˡ = subfilter_reference_heights("SubFilterAvailablePotentialEnergy", model, filter, method, geopotential_height)

    eₐ  = AvailablePotentialEnergy(model, z✶)           # eₐ(b, z), the full buoyancy against the shared profile
    eₐˡ = FilteredAvailablePotentialEnergy(model, z✶ˡ)  # eₐ(b̄, z), the filtered buoyancy against the same one
    eₐˢ = Field(filter(Field(eₐ))) - eₐˡ

    return KernelFunctionOperation{Center, Center, Center}(subfilter_ape_ccc, model.grid, eₐˢ)
end

SubFilterAvailablePotentialEnergy(model; σ, dims = (1, 2, 3), boundary = :shrink, N = nothing, kwargs...) =
    SubFilterAvailablePotentialEnergy(model, GaussianFilter(; dims, σ, boundary, N); kwargs...)
#---

#+++ Sub-filter available potential energy dissipation rate
# εₐˢ = filter(εₐ) - εₐˡ, the same wrapper trick as `SubFilterKineticEnergyDissipationRate`.
@inline subfilter_ape_dissipation_rate_ccc(i, j, k, grid, εₐˢ) = @inbounds εₐˢ[i, j, k]

const SubFilterAvailablePotentialEnergyDissipationRate = CustomKFO{<:typeof(subfilter_ape_dissipation_rate_ccc)}
const DissipationRate = SubFilterAvailablePotentialEnergyDissipationRate

"""
    $(SIGNATURES)

Return the sub-filter-scale (SFS) available potential energy dissipation rate `εₐˢ`, the APE
destruction by diffusion carried by the scales that a low-pass `filter` removes:

```
    εₐˢ = filter(εₐ) - εₐˡ ,   εₐˡ = -q̄ᵢ ∂ᵢΥˡ ,   q̄ᵢ = filter(qᵢ) ,   Υˡ = z✶(b̄) - z
```

where `εₐ = -qᵢ∂ᵢΥ` is the dissipation rate of the full field
([`AvailablePotentialEnergyDissipationRate`](@ref)), `qᵢ` is the closure's own diffusive buoyancy
flux, and `εₐˡ` is the same contraction evaluated on the filtered state
([`FilteredAvailablePotentialEnergyDissipationRate`](@ref)): the *filtered* flux `q̄ᵢ` against the
displacement potential `Υˡ` of the filtered buoyancy `b̄ = filter(b)`. Filtering the flux rather than
recomputing it from `b̄` is what makes `εₐˡ` the sink of the filtered-state budget when `κ` varies in
space; that docstring has the details. Both states are measured against one shared reference profile,
exactly as in [`SubFilterAvailablePotentialEnergy`](@ref), whose budget this is the diffusive sink of
([Wenegrat, Chor & Barkan, 2026](https://arxiv.org/abs/2605.15879)); it mirrors what
[`SubFilterKineticEnergyDissipationRate`](@ref Oceanostics.SubFilterKineticEnergyEquation.SubFilterKineticEnergyDissipationRate)
is to the sub-filter kinetic energy.

`method` has to be a [`ProfileLookup`](@ref), for the reason
[`SubFilterAvailablePotentialEnergy`](@ref) gives, and the lookup also makes each `z✶` a function of
buoyancy alone — the property that differentiating `Υ` and `Υˡ` needs (see
[`DisplacementPotential`](@ref Oceanostics.AvailablePotentialEnergyEquation.DisplacementPotential)).
Like [`AvailablePotentialEnergyDissipationRate`](@ref), this diagnostic needs the buoyancy to be a
tracer the closure diffuses (`BuoyancyTracer` only) and a closure that supplies a diffusive flux.

`filter` is any callable mapping a field to its low-pass-filtered counterpart, e.g. a reusable
[`GaussianFilter`](@ref) or [`BoxFilter`](@ref). The filtered fluxes, the filtered buoyancy, `Υˡ` and
the filtered full-field dissipation are materialized as `Field`s internally, so the returned object is
a lazy operation ready for `Field`, `Integral` and `OutputWriter`s, recomputing (re-sorting included)
as the simulation evolves. It lives at `(Center, Center, Center)`, per unit mass (units `m² s⁻³`):

```jldoctest
using Oceananigans, Oceanostics

grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1), topology=(Periodic, Periodic, Bounded))
model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b, closure=ScalarDiffusivity(κ=1e-4))

filter = GaussianFilter(; dims=(1, 2, 3), σ=0.1)
SubFilterAvailablePotentialEnergyDissipationRate(model, filter)

# output

SubFilterAvailablePotentialEnergyDissipationRate KernelFunctionOperation at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: subfilter_ape_dissipation_rate_ccc (generic function with 1 method)
└── arguments: ("Oceananigans.AbstractOperations.BinaryOperation",)
└── computes: sub-filter available potential energy dissipation rate  εₐˢ = filter(εₐ) - εₐˡ
```

A convenience method `SubFilterAvailablePotentialEnergyDissipationRate(model; σ, dims, boundary, N)`
builds the Gaussian `filter` for you from a standard deviation `σ` (with `σ = ℓ / (2√(2 ln 2))` for a
FWHM `ℓ`).
"""
function SubFilterAvailablePotentialEnergyDissipationRate(model, filter; method = ProfileLookup(),
                                                          geopotential_height = model_geopotential_height(model))
    validate_buoyancy_is_a_diffused_tracer("SubFilterAvailablePotentialEnergyDissipationRate", model)
    validate_closure_supplies_a_flux("SubFilterAvailablePotentialEnergyDissipationRate", model)
    validate_gravity_is_z_aligned("SubFilterAvailablePotentialEnergyDissipationRate", model)
    z✶, z✶ˡ = subfilter_reference_heights("SubFilterAvailablePotentialEnergyDissipationRate", model, filter, method,
                                          geopotential_height)

    εₐ  = AvailablePotentialEnergyDissipationRate(model, z✶)                # -qᵢ∂ᵢΥ off the closure's own flux
    εₐˡ = FilteredAvailablePotentialEnergyDissipationRate(model, filter, z✶ˡ)  # -q̄ᵢ∂ᵢΥˡ, filtered flux and Υ of b̄

    εₐˢ = Field(filter(Field(εₐ))) - εₐˡ

    return KernelFunctionOperation{Center, Center, Center}(subfilter_ape_dissipation_rate_ccc, model.grid, εₐˢ)
end

SubFilterAvailablePotentialEnergyDissipationRate(model; σ, dims = (1, 2, 3), boundary = :shrink, N = nothing, kwargs...) =
    SubFilterAvailablePotentialEnergyDissipationRate(model, GaussianFilter(; dims, σ, boundary, N); kwargs...)
#---

end # module
