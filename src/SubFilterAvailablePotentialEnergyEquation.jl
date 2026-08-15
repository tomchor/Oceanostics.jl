module SubFilterAvailablePotentialEnergyEquation

using DocStringExtensions

export SubFilterAvailablePotentialEnergy
export SubFilterAvailablePotentialEnergyDissipationRate, DissipationRate
# The shared reference profile both diagnostics measure against is built with
# `BackgroundPotentialEnergyEquation`'s machinery, so the pieces needed to construct and share one are
# re-exported here and this module can be used on its own.
export reference_height, reference_buoyancy, VerticalSort, ProfileLookup

using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Fields: Field
using Oceananigans.Grids: Center, Face
using Oceananigans.Models: model_geopotential_height
using Oceananigans.Operators
using Oceananigans.TurbulenceClosures: diffusive_flux_x, diffusive_flux_y, diffusive_flux_z

using Oceanostics: CustomKFO

using ..PotentialEnergyEquation: validate_buoyancy_is_a_diffused_tracer, validate_closure_supplies_a_flux,
                                 validate_gravity_is_z_aligned, buoyancy_diffusive_flux_arguments
using ..BackgroundPotentialEnergyEquation: reference_height, reference_buoyancy, VerticalSort, ProfileLookup,
                                           buoyancy_field
using ..AvailablePotentialEnergyEquation: AvailablePotentialEnergy, BuoyancyDisplacementPotential,
                                          AvailablePotentialEnergyDissipationRate
# `GaussianFilter` builds the convenience methods' filter; `BoxFilter` is imported only so its docstring
# `@ref` resolves in-module.
using ..SpatialFilters: GaussianFilter, BoxFilter

#+++ Shared reference profile
# The sub-filter split compares `eₐ` (or `ε_A`) of the full buoyancy with the same quantity of the
# filtered buoyancy, and the comparison only means anything when both are measured against one shared
# reference profile. `ProfileLookup` is the one method built to match a field into a profile that did
# not come from sorting that field, so it is the only method these diagnostics accept. The default
# `ProfileLookup()`, which would sort each field's own buoyancy, is resolved here into a lookup of the
# model's own (full) buoyancy: a `VerticalSort` column re-sorted on every `compute!`, so the reference
# state tracks the flow. A `ProfileLookup` already holding a profile is used as given, which is how a
# column is shared between diagnostics or a fixed profile (plain arrays) is held frozen in time.
function shared_profile_lookup(diagnostic, b, method::ProfileLookup)
    isnothing(method.profile) || return method
    return ProfileLookup(reference_height(b; method = VerticalSort()))
end

shared_profile_lookup(diagnostic, b, method) =
    throw(ArgumentError("`$diagnostic` measures the full and the filtered buoyancy against one shared \
                         reference profile, so `method` has to be a `ProfileLookup`, but it got a \
                         $(summary(method)). Use `ProfileLookup()` to sort the profile from the model's own \
                         buoyancy on every compute, `ProfileLookup(z✶_column)` to share a column built with \
                         `VerticalSort()`, or `ProfileLookup(b✶, z✶)` to hold a fixed reference profile."))

# The two reference heights every sub-filter APE diagnostic needs: the full buoyancy `b` and the
# filtered buoyancy `b̄ = filter(b)`, each looked up in the same shared profile. `b̄` is materialized as
# a `Field` (so the separable filter takes its fast staged path), and `compute!` on `z✶ˡ` recomputes it,
# so the filtered state follows the flow.
function subfilter_reference_heights(diagnostic, model, filter, method, geopotential_height)
    b = buoyancy_field(model, model.buoyancy, geopotential_height)
    b̄ = Field(filter(b))
    lookup = shared_profile_lookup(diagnostic, b, method)
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

where `eₐ` is the local available potential energy density ([`AvailablePotentialEnergy`](@ref)) and
both terms are measured against **one shared reference profile** `(b✶, z✶)`, following the filtered
APE framework of [Wenegrat, Chor & Barkan (2026)](https://arxiv.org/abs/2605.15879): the filtered
buoyancy is looked up in the same profile the full field is measured against, which is exactly what
[`ProfileLookup`](@ref) was built for. It is the potential-energy counterpart of the sub-filter
kinetic energy `Kˢ` ([`SubFilterKineticEnergy`](@ref Oceanostics.SubFilterKineticEnergyEquation.SubFilterKineticEnergy)).

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
buoyancies that fall between the profile's entries (the lookup then takes the nearest slot).

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

    eₐ  = AvailablePotentialEnergy(model, z✶)    # eₐ(b, z), the full buoyancy against the shared profile
    eₐˡ = AvailablePotentialEnergy(model, z✶ˡ)   # eₐ(b̄, z): `z✶ˡ` carries the filtered buoyancy as its own
    eₐˢ = Field(filter(Field(eₐ))) - eₐˡ

    return KernelFunctionOperation{Center, Center, Center}(subfilter_ape_ccc, model.grid, eₐˢ)
end

SubFilterAvailablePotentialEnergy(model; σ, dims = (1, 2, 3), boundary = :shrink, N = nothing, kwargs...) =
    SubFilterAvailablePotentialEnergy(model, GaussianFilter(; dims, σ, boundary, N); kwargs...)
#---

#+++ Sub-filter available potential energy dissipation rate
# ε_Aˡ = -q̄ᵢ∂ᵢΥˡ, the `ape_dissipation_rate_ccc` contraction with the pre-filtered flux `Field`s read
# directly in place of the inline `diffusive_flux_*` calls (the same substitution
# `FilteredKineticEnergyDissipationRate` makes for the viscous fluxes). Each product is formed on the
# face where both factors live and only then interpolated to the cell center.
@inline Axᶠᶜᶜ_δΥˡᶠᶜᶜ_q̄₁ᶠᶜᶜ(i, j, k, grid, Υˡ, q̄₁) = - Axᶠᶜᶜ(i, j, k, grid) * δxᶠᵃᵃ(i, j, k, grid, Υˡ) * @inbounds(q̄₁[i, j, k])
@inline Ayᶜᶠᶜ_δΥˡᶜᶠᶜ_q̄₂ᶜᶠᶜ(i, j, k, grid, Υˡ, q̄₂) = - Ayᶜᶠᶜ(i, j, k, grid) * δyᵃᶠᵃ(i, j, k, grid, Υˡ) * @inbounds(q̄₂[i, j, k])
@inline Azᶜᶜᶠ_δΥˡᶜᶜᶠ_q̄₃ᶜᶜᶠ(i, j, k, grid, Υˡ, q̄₃) = - Azᶜᶜᶠ(i, j, k, grid) * δzᵃᵃᶠ(i, j, k, grid, Υˡ) * @inbounds(q̄₃[i, j, k])

@inline filtered_ape_dissipation_rate_ccc(i, j, k, grid, Υˡ, q̄₁, q̄₂, q̄₃) =
    (ℑxᶜᵃᵃ(i, j, k, grid, Axᶠᶜᶜ_δΥˡᶠᶜᶜ_q̄₁ᶠᶜᶜ, Υˡ, q̄₁) + # F, C, C  → C, C, C
     ℑyᵃᶜᵃ(i, j, k, grid, Ayᶜᶠᶜ_δΥˡᶜᶠᶜ_q̄₂ᶜᶠᶜ, Υˡ, q̄₂) + # C, F, C  → C, C, C
     ℑzᵃᵃᶜ(i, j, k, grid, Azᶜᶜᶠ_δΥˡᶜᶜᶠ_q̄₃ᶜᶜᶠ, Υˡ, q̄₃)   # C, C, F  → C, C, C
     ) / Vᶜᶜᶜ(i, j, k, grid) # the division by volume, against the `A δΥˡ` above, is what makes it a derivative

# ε_Aˢ = filter(ε_A) - ε_Aˡ, the same wrapper trick as `SubFilterKineticEnergyDissipationRate`.
@inline subfilter_ape_dissipation_rate_ccc(i, j, k, grid, ε_Aˢ) = @inbounds ε_Aˢ[i, j, k]

const SubFilterAvailablePotentialEnergyDissipationRate = CustomKFO{<:typeof(subfilter_ape_dissipation_rate_ccc)}
const DissipationRate = SubFilterAvailablePotentialEnergyDissipationRate

"""
    $(SIGNATURES)

Return the sub-filter-scale (SFS) available potential energy dissipation rate `ε_Aˢ`, the APE
destruction by diffusion carried by the scales that a low-pass `filter` removes:

```
    ε_Aˢ = filter(ε_A) - ε_Aˡ ,   ε_Aˡ = -q̄ᵢ ∂ᵢΥˡ ,   q̄ᵢ = filter(qᵢ) ,   Υˡ = z✶(b̄) - z
```

where `ε_A = -qᵢ∂ᵢΥ` is the dissipation rate of the full field
([`AvailablePotentialEnergyDissipationRate`](@ref)), `qᵢ` is the closure's own diffusive buoyancy
flux, and `ε_Aˡ` is the same contraction evaluated on the filtered state: the filtered flux against
the displacement potential `Υˡ` ([`BuoyancyDisplacementPotential`](@ref)) of the filtered buoyancy
`b̄ = filter(b)`. Both states are measured against one shared reference profile, exactly as in
[`SubFilterAvailablePotentialEnergy`](@ref), whose budget this is the diffusive sink of
([Wenegrat, Chor & Barkan, 2026](https://arxiv.org/abs/2605.15879)); it mirrors what
[`SubFilterKineticEnergyDissipationRate`](@ref Oceanostics.SubFilterKineticEnergyEquation.SubFilterKineticEnergyDissipationRate)
is to the sub-filter kinetic energy.

The flux in `ε_Aˡ` is filtered, `q̄ᵢ = filter(qᵢ(b))`, not recomputed from the filtered buoyancy: the
filtered buoyancy equation carries the divergence of the filtered flux, so `-q̄ᵢ∂ᵢΥˡ` is the
dissipation that appears in the filtered-state budget. The two forms agree for a constant
diffusivity, where the filter commutes with the flux, and differ once `κ` varies in space — the same
distinction [`FilteredKineticEnergyDissipationRate`](@ref Oceanostics.FilteredKineticEnergyEquation.FilteredKineticEnergyDissipationRate)
draws for the viscous flux.

`method` has to be a [`ProfileLookup`](@ref), for the reason
[`SubFilterAvailablePotentialEnergy`](@ref) gives, and the lookup also makes each `z✶` a function of
buoyancy alone — the property that differentiating `Υ` and `Υˡ` needs (see
[`BuoyancyDisplacementPotential`](@ref)). Like [`AvailablePotentialEnergyDissipationRate`](@ref), this
diagnostic needs the buoyancy to be a tracer the closure diffuses (`BuoyancyTracer` only) and a
closure that supplies a diffusive flux.

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
└── computes: sub-filter available potential energy dissipation rate  ε_Aˢ = filter(ε_A) - ε_Aˡ
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

    ε_A = AvailablePotentialEnergyDissipationRate(model, z✶)   # -qᵢ∂ᵢΥ off the closure's own flux

    # q̄ᵢ = filter(qᵢ(b)): the closure's diffusive fluxes of the FULL buoyancy, each low-pass filtered and
    # materialized at its staggered location. The flux operation reads the live model fields, so the
    # filtered fluxes refresh when the diagnostic is recomputed.
    flux_args = buoyancy_diffusive_flux_arguments(model)
    filtered_flux(f, LX, LY, LZ) = Field(filter(KernelFunctionOperation{LX, LY, LZ}(f, model.grid, flux_args...)))
    q̄₁ = filtered_flux(diffusive_flux_x, Face,   Center, Center)
    q̄₂ = filtered_flux(diffusive_flux_y, Center, Face,   Center)
    q̄₃ = filtered_flux(diffusive_flux_z, Center, Center, Face)

    Υˡ = Field(BuoyancyDisplacementPotential(model, z✶ˡ))
    ε_Aˡ = KernelFunctionOperation{Center, Center, Center}(filtered_ape_dissipation_rate_ccc, model.grid, Υˡ, q̄₁, q̄₂, q̄₃)

    ε_Aˢ = Field(filter(Field(ε_A))) - ε_Aˡ

    return KernelFunctionOperation{Center, Center, Center}(subfilter_ape_dissipation_rate_ccc, model.grid, ε_Aˢ)
end

SubFilterAvailablePotentialEnergyDissipationRate(model; σ, dims = (1, 2, 3), boundary = :shrink, N = nothing, kwargs...) =
    SubFilterAvailablePotentialEnergyDissipationRate(model, GaussianFilter(; dims, σ, boundary, N); kwargs...)
#---

end # module
