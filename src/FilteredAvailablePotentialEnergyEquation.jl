module FilteredAvailablePotentialEnergyEquation

using DocStringExtensions

export FilteredAvailablePotentialEnergy
export FilteredAvailablePotentialEnergyDissipationRate, DissipationRate
export AvailablePotentialEnergyCrossScaleFlux, CrossScaleFlux
export FilteredAvailablePotentialToKineticEnergyConversion
# The reference profile the filtered buoyancy is measured against is built with
# `BackgroundPotentialEnergyEquation`'s machinery, so the pieces needed to construct and share one are
# re-exported here and this module can be used on its own.
export reference_height, reference_buoyancy, VerticalSort, ProfileLookup

using Oceananigans: NonhydrostaticModel
using Oceananigans.AbstractOperations: KernelFunctionOperation, ∂x, ∂y, ∂z
using Oceananigans.Fields: Field
using Oceananigans.Grids: Center, Face
using Oceananigans.Models: model_geopotential_height
using Oceananigans.Operators
using Oceananigans.TurbulenceClosures: diffusive_flux_x, diffusive_flux_y, diffusive_flux_z

using Oceanostics: CustomKFO

using ..PotentialEnergyEquation: validate_buoyancy_is_a_diffused_tracer, validate_closure_supplies_a_flux,
                                 validate_gravity_is_z_aligned, buoyancy_diffusive_flux_arguments
using ..BackgroundPotentialEnergyEquation: SortedReferenceHeightField, reference_height, reference_buoyancy,
                                           reference_buoyancy_at_height, VerticalSort, ProfileLookup,
                                           buoyancy_field
using ..AvailablePotentialEnergyEquation: AvailablePotentialEnergy, AvailablePotentialEnergyDisplacementPotential,
                                          AvailablePotentialEnergyDissipationRate, local_ape_ccc,
                                          validate_reference_height_grid
# `GaussianFilter` builds the convenience methods' filter; `BoxFilter` is imported only so its docstring
# `@ref` resolves in-module.
using ..SpatialFilters: GaussianFilter, BoxFilter
using ..FlowDiagnostics: validate_dims, subfilter_covariance, to_center

#+++ Shared reference profile
# The filtered-flow diagnostics measure the filtered buoyancy `b̄` against a reference profile that
# did not come from sorting `b̄` itself: ordinarily the sorted state of the *full* buoyancy, so that
# `eₐ(b̄, z)` and `eₐ(b, z)` are comparable and their difference (the subfilter APE) means something.
# `ProfileLookup` is the one method built to match a field into a profile it did not produce, so it is
# the only method these diagnostics accept. The default `ProfileLookup()`, which would sort each
# field's own buoyancy, is resolved here into a lookup of the model's own (full) buoyancy: a
# `VerticalSort` column re-sorted on every `compute!`, so the reference state tracks the flow. A
# `ProfileLookup` already holding a profile is used as given, which is how a column is shared between
# diagnostics or a fixed profile (plain arrays) is held frozen in time.
function shared_profile_lookup(diagnostic, b, method::ProfileLookup)
    isnothing(method.profile) || return method
    return ProfileLookup(reference_height(b; method = VerticalSort()))
end

shared_profile_lookup(diagnostic, b, method) =
    throw(ArgumentError("`$diagnostic` measures the filtered buoyancy against a shared reference profile \
                         (ordinarily the sorted state of the full buoyancy), so `method` has to be a \
                         `ProfileLookup`, but it got a $(summary(method)). Use `ProfileLookup()` to sort the \
                         profile from the model's own buoyancy on every compute, `ProfileLookup(z✶_column)` \
                         to share a column built with `VerticalSort()`, or `ProfileLookup(b✶, z✶)` to hold a \
                         fixed reference profile."))

# The full buoyancy `b`, its low-pass-filtered counterpart `b̄ = filter(b)`, and the shared lookup both
# are measured with. `b̄` is materialized as a `Field` (so the separable filter takes its fast staged
# path), and `compute!` on anything built from it re-filters it, so the filtered state follows the
# flow. `SubFilterAvailablePotentialEnergyEquation` builds its full-field reference height from the same
# three pieces, which is what guarantees the two states share one profile, and
# `AvailablePotentialEnergyCrossScaleFlux` shares the one `b̄` between its reference height and its
# subfilter buoyancy flux.
function filtered_buoyancy_and_lookup(diagnostic, model, filter, method, geopotential_height)
    b = buoyancy_field(model, model.buoyancy, geopotential_height)
    b̄ = Field(filter(b))
    lookup = shared_profile_lookup(diagnostic, b, method)
    return b, b̄, lookup
end

# The reference height of the filtered buoyancy, `z✶ˡ = z✶(b̄)`, looked up in the shared profile.
function filtered_reference_height(diagnostic, model, filter, method, geopotential_height)
    b, b̄, lookup = filtered_buoyancy_and_lookup(diagnostic, model, filter, method, geopotential_height)
    return reference_height(b̄; method = lookup)
end
#---

#+++ Filtered available potential energy
# eₐˡ = eₐ(b̄, z), the local APE of the filtered buoyancy. It reuses `AvailablePotentialEnergyEquation`'s
# `local_ape_ccc` kernel on the reference height `z✶ˡ` built from `b̄` (which carries `b̄` as its own
# buoyancy); wrapping it under a distinct kernel name gives `FilteredAvailablePotentialEnergy` its own type
# alias and `@diagnostic_show` display, exactly as `FilteredKineticEnergy` wraps `kinetic_energy_ccc`.
@inline filtered_ape_ccc(i, j, k, grid, args...) = local_ape_ccc(i, j, k, grid, args...)

const FilteredAvailablePotentialEnergy = CustomKFO{<:typeof(filtered_ape_ccc)}

"""
    $(SIGNATURES)

Return the available potential energy of the filtered buoyancy field `eₐˡ`, the local APE that the
scales a low-pass `filter` keeps would carry on their own:

```
    eₐˡ = eₐ(b̄, z) = ∫_{z✶(b̄)}^{z} [b✶(z̃) - b̄] dz̃ ,   b̄ = filter(b)
```

where `eₐ` is the local available potential energy density ([`AvailablePotentialEnergy`](@ref)) and
`b✶(z̃)` is a reference profile the filtered buoyancy is looked up in, following the filtered APE
framework of [Wenegrat, Chor & Barkan (2026)](https://arxiv.org/abs/2605.15879). It is the
potential-energy counterpart of the kinetic energy of the filtered flow `eₖˡ`
([`FilteredKineticEnergy`](@ref Oceanostics.FilteredKineticEnergyEquation.FilteredKineticEnergy)),
and the filter splits the APE into it and the subfilter remainder `eₐˢ = filter(eₐ) - eₐˡ`
([`SubFilterAvailablePotentialEnergy`](@ref Oceanostics.SubFilterAvailablePotentialEnergyEquation.SubFilterAvailablePotentialEnergy)).

For that split to mean anything the filtered buoyancy has to be measured against the **same reference
profile as the full one**, ordinarily the sorted state of the full buoyancy `b`, rather than against a
profile sorted from `b̄` itself. Looking a field up in a profile it did not produce is exactly what
[`ProfileLookup`](@ref) was built for, so `method` must be one:

  - `ProfileLookup()` (the default) sorts the profile from the model's own buoyancy — a
    [`VerticalSort`](@ref) column built internally and re-sorted on every `compute!`, so the reference
    state tracks the flow.
  - `ProfileLookup(z✶_column)` borrows a column you already built with
    `reference_height(model, method=VerticalSort())`, sharing its sort across several diagnostics.
  - `ProfileLookup(b✶, z✶)` with plain arrays holds the reference profile fixed in time, which also
    makes the diagnostic sort-free: each `compute!` is then a filter plus a binary-search lookup.

A second method, `FilteredAvailablePotentialEnergy(model, z✶ˡ)`, takes a reference height you built
yourself from a filtered buoyancy, `z✶ˡ = reference_height(Field(filter(b)); method=ProfileLookup(…))`,
which is how a single lookup is shared with [`FilteredAvailablePotentialEnergyDissipationRate`](@ref).
The filtered buoyancy is read off `z✶ˡ` itself, so no `filter` is needed there.

`filter` is any callable mapping a field to its low-pass-filtered counterpart, e.g. a reusable
[`GaussianFilter`](@ref) or [`BoxFilter`](@ref). The filtered buoyancy is materialized as a `Field`
internally (so the separable filter takes its fast staged path), and the returned object is a lazy
operation over it and the reference height, ready for `Field`, `Integral` and `OutputWriter`s. It
lives at `(Center, Center, Center)`, per unit mass (units `m² s⁻²`):

```jldoctest
using Oceananigans, Oceanostics

grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1), topology=(Periodic, Periodic, Bounded))
model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b)

filter = GaussianFilter(; dims=(1, 2, 3), σ=0.1)
FilteredAvailablePotentialEnergy(model, filter)

# output

FilteredAvailablePotentialEnergy KernelFunctionOperation at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: filtered_ape_ccc (generic function with 1 method)
└── arguments: ("Field", "Field", "Field")
└── computes: available potential energy of the filtered buoyancy  eₐ(b̄, z)
```

A convenience method `FilteredAvailablePotentialEnergy(model; σ, dims, boundary, N)` builds the
Gaussian `filter` for you from a standard deviation `σ` (with `σ = ℓ / (2√(2 ln 2))` for a FWHM `ℓ`).
`geopotential_height` enters the buoyancy construction exactly as in [`reference_height`](@ref).
"""
function FilteredAvailablePotentialEnergy(model, filter; method = ProfileLookup(),
                                          geopotential_height = model_geopotential_height(model))
    validate_gravity_is_z_aligned("FilteredAvailablePotentialEnergy", model)
    z✶ˡ = filtered_reference_height("FilteredAvailablePotentialEnergy", model, filter, method, geopotential_height)
    return FilteredAvailablePotentialEnergy(model, z✶ˡ)
end

# `AvailablePotentialEnergy(model, z✶ˡ)` validates the gravity direction, reads the filtered buoyancy `b̄`
# off `z✶ˡ`, and picks the model-grid or sorted-column argument list; only the kernel name is swapped.
function FilteredAvailablePotentialEnergy(model, z✶ˡ::SortedReferenceHeightField)
    eₐˡ = AvailablePotentialEnergy(model, z✶ˡ)
    return KernelFunctionOperation{Center, Center, Center}(filtered_ape_ccc, z✶ˡ.grid, eₐˡ.arguments...)
end

FilteredAvailablePotentialEnergy(model; σ, dims = (1, 2, 3), boundary = :shrink, N = nothing, kwargs...) =
    FilteredAvailablePotentialEnergy(model, GaussianFilter(; dims, σ, boundary, N); kwargs...)
#---

#+++ Filtered available potential energy dissipation rate
# εₐˡ = -q̄ᵢ∂ᵢΥˡ, the `ape_dissipation_rate_ccc` contraction with the pre-filtered flux `Field`s read
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

const FilteredAvailablePotentialEnergyDissipationRate = CustomKFO{<:typeof(filtered_ape_dissipation_rate_ccc)}
const DissipationRate = FilteredAvailablePotentialEnergyDissipationRate

"""
    $(SIGNATURES)

Return the available potential energy dissipation rate of the filtered buoyancy field `εₐˡ`, the rate at
which diffusion removes APE from the scales a low-pass `filter` keeps:

```
    εₐˡ = -q̄ᵢ ∂ᵢΥˡ ,   q̄ᵢ = filter(qᵢ) ,   Υˡ = z✶(b̄) - z ,   b̄ = filter(b)
```

It is the full-field contraction `εₐ = -qᵢ∂ᵢΥ` ([`AvailablePotentialEnergyDissipationRate`](@ref))
evaluated on the filtered state: the closure's diffusive buoyancy flux `qᵢ` low-pass filtered, against
the displacement potential `Υˡ` ([`AvailablePotentialEnergyDisplacementPotential`](@ref)) of the filtered buoyancy.
It is the diffusive sink of the budget of [`FilteredAvailablePotentialEnergy`](@ref)
([Wenegrat, Chor & Barkan, 2026](https://arxiv.org/abs/2605.15879)), and it mirrors what
[`FilteredKineticEnergyDissipationRate`](@ref Oceanostics.FilteredKineticEnergyEquation.FilteredKineticEnergyDissipationRate)
is to the filtered kinetic energy. The subfilter remainder `εₐˢ = filter(εₐ) - εₐˡ` is
[`SubFilterAvailablePotentialEnergyDissipationRate`](@ref Oceanostics.SubFilterAvailablePotentialEnergyEquation.SubFilterAvailablePotentialEnergyDissipationRate).

The flux is filtered, `q̄ᵢ = filter(qᵢ(b))`, not recomputed from the filtered buoyancy: the filtered
buoyancy equation carries the divergence of the filtered flux, so `-q̄ᵢ∂ᵢΥˡ` is the dissipation that
appears in the filtered-state budget. The two forms agree for a constant diffusivity, where the filter
commutes with the flux, and differ once `κ` varies in space — the same distinction
[`FilteredKineticEnergyDissipationRate`](@ref Oceanostics.FilteredKineticEnergyEquation.FilteredKineticEnergyDissipationRate)
draws for the viscous flux.

`method` has to be a [`ProfileLookup`](@ref), for the reason [`FilteredAvailablePotentialEnergy`](@ref)
gives, and the lookup also makes `z✶ˡ` a function of buoyancy alone — the property that differentiating
`Υˡ` needs (see [`AvailablePotentialEnergyDisplacementPotential`](@ref)). Like
[`AvailablePotentialEnergyDissipationRate`](@ref), this diagnostic needs the buoyancy to be a tracer
the closure diffuses (`BuoyancyTracer` only) and a closure that supplies a diffusive flux.

A second method, `FilteredAvailablePotentialEnergyDissipationRate(model, filter, z✶ˡ; upsilon)`, takes
a reference height you built from the filtered buoyancy (see [`FilteredAvailablePotentialEnergy`](@ref)),
so one lookup can serve both diagnostics; `upsilon` takes a `Υˡ` you already have, so writing both out
costs one `Υˡ` rather than two. `filter` is still needed there, to filter the fluxes.

`filter` is any callable mapping a field to its low-pass-filtered counterpart, e.g. a reusable
[`GaussianFilter`](@ref) or [`BoxFilter`](@ref). The filtered fluxes, the filtered buoyancy and `Υˡ`
are materialized as `Field`s internally, so the returned object is a lazy operation ready for `Field`,
`Integral` and `OutputWriter`s, recomputing (re-sorting included) as the simulation evolves. It lives
at `(Center, Center, Center)`, per unit mass (units `m² s⁻³`):

```jldoctest
using Oceananigans, Oceanostics

grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1), topology=(Periodic, Periodic, Bounded))
model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b, closure=ScalarDiffusivity(κ=1e-4))

filter = GaussianFilter(; dims=(1, 2, 3), σ=0.1)
FilteredAvailablePotentialEnergyDissipationRate(model, filter)

# output

FilteredAvailablePotentialEnergyDissipationRate KernelFunctionOperation at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: filtered_ape_dissipation_rate_ccc (generic function with 1 method)
└── arguments: ("Field", "Field", "Field", "Field")
└── computes: available potential energy dissipation rate of the filtered buoyancy  -q̄ᵢ∂ᵢΥˡ
```

A convenience method `FilteredAvailablePotentialEnergyDissipationRate(model; σ, dims, boundary, N)`
builds the Gaussian `filter` for you from a standard deviation `σ` (with `σ = ℓ / (2√(2 ln 2))` for a
FWHM `ℓ`).
"""
function FilteredAvailablePotentialEnergyDissipationRate(model, filter; method = ProfileLookup(),
                                                         geopotential_height = model_geopotential_height(model))
    validate_gravity_is_z_aligned("FilteredAvailablePotentialEnergyDissipationRate", model)
    z✶ˡ = filtered_reference_height("FilteredAvailablePotentialEnergyDissipationRate", model, filter, method,
                                    geopotential_height)
    return FilteredAvailablePotentialEnergyDissipationRate(model, filter, z✶ˡ)
end

function FilteredAvailablePotentialEnergyDissipationRate(model, filter, z✶ˡ::SortedReferenceHeightField; upsilon = nothing)
    validate_buoyancy_is_a_diffused_tracer("FilteredAvailablePotentialEnergyDissipationRate", model)
    validate_closure_supplies_a_flux("FilteredAvailablePotentialEnergyDissipationRate", model)
    validate_gravity_is_z_aligned("FilteredAvailablePotentialEnergyDissipationRate", model)
    validate_reference_height_grid("FilteredAvailablePotentialEnergyDissipationRate", model, z✶ˡ)

    Υˡ = isnothing(upsilon) ? Field(AvailablePotentialEnergyDisplacementPotential(model, z✶ˡ)) : upsilon

    # q̄ᵢ = filter(qᵢ(b)): the closure's diffusive fluxes of the FULL buoyancy, each low-pass filtered and
    # materialized at its staggered location. The flux operation reads the live model fields, so the
    # filtered fluxes refresh when the diagnostic is recomputed.
    flux_args = buoyancy_diffusive_flux_arguments(model)
    filtered_flux(f, LX, LY, LZ) = Field(filter(KernelFunctionOperation{LX, LY, LZ}(f, model.grid, flux_args...)))
    q̄₁ = filtered_flux(diffusive_flux_x, Face,   Center, Center)
    q̄₂ = filtered_flux(diffusive_flux_y, Center, Face,   Center)
    q̄₃ = filtered_flux(diffusive_flux_z, Center, Center, Face)

    return KernelFunctionOperation{Center, Center, Center}(filtered_ape_dissipation_rate_ccc, model.grid, Υˡ, q̄₁, q̄₂, q̄₃)
end

FilteredAvailablePotentialEnergyDissipationRate(model; σ, dims = (1, 2, 3), boundary = :shrink, N = nothing, kwargs...) =
    FilteredAvailablePotentialEnergyDissipationRate(model, GaussianFilter(; dims, σ, boundary, N); kwargs...)
#---

#+++ Cross-scale available-potential-energy flux
# Πₐ = -τᵢ ∂ᵢΥˡ, the APE analogue of `KineticEnergyCrossScaleFlux`'s Πₖ = -τᵢⱼS̄ᵢⱼ. The subfilter
# buoyancy flux τᵢ = filter(buᵢ) - b̄ūᵢ takes the place of the subfilter stress, and the gradient of
# the filtered-state displacement potential Υˡ takes the place of the resolved strain. Every factor is
# interpolated to (Center, Center, Center) before multiplying (via `FlowDiagnostics`' shared
# `to_center`), matching the contraction convention of the kinetic-energy flux.
"""
    $(SIGNATURES)

The subfilter buoyancy flux `τᵢ = filter(b uᵢ) - b̄ ūᵢ` for the directions in `dims`, as a `NamedTuple`
keyed `τ₁`, `τ₂`, `τ₃`:
[`subfilter_covariance`](@ref Oceanostics.FlowDiagnostics.subfilter_covariance) applied per direction,
with the caller's pre-filtered buoyancy `b̄` shared across the components through its `filtered_a`
keyword, so the buoyancy is filtered once rather than once per component.
"""
function subfilter_buoyancy_flux(filter, b, b̄, velocities, dims)
    pairs = map(dims) do d
        Symbol(:τ, ('₁', '₂', '₃')[d]) => subfilter_covariance(b, velocities[d], filter; filtered_a = b̄)
    end
    return (; pairs...)
end

# As in the kinetic-energy flux, the contraction is exposed as a single `KernelFunctionOperation` whose
# kernel just indexes the pre-assembled operation; its leaves are materialized `Field`s, so per-cell
# evaluation only reads them and does arithmetic — it never re-filters or re-sorts.
@inline cross_scale_ape_flux_ccc(i, j, k, grid, Πᵃ) = @inbounds Πᵃ[i, j, k]

const AvailablePotentialEnergyCrossScaleFlux = CustomKFO{<:typeof(cross_scale_ape_flux_ccc)}
const CrossScaleFlux = AvailablePotentialEnergyCrossScaleFlux

"""
    $(SIGNATURES)

Return the cross-scale (scale-to-scale) available-potential-energy flux `Πₐ`, the rate at which a
low-pass `filter` transfers available potential energy from the filtered to the subfilter scales
([Wenegrat, Chor & Barkan, 2026](https://arxiv.org/abs/2605.15879)):

```
    Πₐ = -τᵢ ∂ᵢΥˡ ,   τᵢ = filter(b uᵢ) - b̄ ūᵢ ,   Υˡ = z✶(b̄) - z
```

where `τᵢ` is the subfilter buoyancy flux and `Υˡ` is the
[`AvailablePotentialEnergyDisplacementPotential`](@ref Oceanostics.AvailablePotentialEnergyEquation.AvailablePotentialEnergyDisplacementPotential)
of the filtered buoyancy. It is the APE analogue of
[`KineticEnergyCrossScaleFlux`](@ref Oceanostics.FilteredKineticEnergyEquation.KineticEnergyCrossScaleFlux),
with the subfilter buoyancy flux in place of the subfilter stress and `∇Υˡ` in place of the resolved
strain; `Πₐ > 0` is forward (downscale, filtered → subfilter) transfer. It appears as `-Πₐ` in the
filtered APE budget and as `+Πₐ` in the subfilter one, which is what makes it a transfer rather than a
source or a sink. The result lives at `(Center, Center, Center)`, per unit mass (units `m² s⁻³`).

`method` has to be a [`ProfileLookup`](@ref), for the reason
[`FilteredAvailablePotentialEnergy`](@ref) gives: `Υˡ` measures the filtered buoyancy against a profile
it did not itself produce, ordinarily the sorted state of the full buoyancy. The lookup also makes `z✶`
a function of buoyancy alone, which is what differentiating `Υˡ` needs.

`dims` selects which directions are summed over and which velocities are filtered: the default
`dims = (1, 2, 3)` gives the full flux, while `dims = (1, 3)` gives the 2D `x`–`z` flux
`Πₐ = -(τ₁∂₁Υˡ + τ₃∂₃Υˡ)`. The filter's own directions are set independently inside `filter`.

`filter` is any callable mapping a field to its low-pass-filtered counterpart, e.g. a reusable
[`GaussianFilter`](@ref) or [`BoxFilter`](@ref). The filtered fields and `Υˡ` are materialized
internally, so the returned object is a lazy operation ready for `Field`, `Integral` and
`OutputWriter`s, re-filtering and re-sorting as the simulation evolves:

```jldoctest
using Oceananigans, Oceanostics

grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1), topology=(Periodic, Periodic, Bounded))
model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b)

filter = GaussianFilter(; dims=(1, 2, 3), σ=0.1)
AvailablePotentialEnergyCrossScaleFlux(model, filter)

# output

AvailablePotentialEnergyCrossScaleFlux KernelFunctionOperation at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: cross_scale_ape_flux_ccc (generic function with 1 method)
└── arguments: ("Oceananigans.AbstractOperations.UnaryOperation",)
└── computes: cross-scale available potential energy flux  -τᵢ∂ᵢΥˡ
```

A second method, `AvailablePotentialEnergyCrossScaleFlux(model, filter, z✶ˡ; upsilon, dims)`, takes a
reference height you built from the filtered buoyancy (see [`FilteredAvailablePotentialEnergy`](@ref)),
so one lookup — and, through `upsilon`, one `Υˡ` — can serve this flux and the other filtered-state
diagnostics. The filtered buoyancy is read off `z✶ˡ` itself; `filter` is still needed there, to build
the subfilter buoyancy flux.

A convenience method `AvailablePotentialEnergyCrossScaleFlux(model; σ, dims, boundary, N)` builds the
Gaussian `filter` for you from a standard deviation `σ` (with `σ = ℓ / (2√(2 ln 2))` for a FWHM `ℓ`).
"""
function AvailablePotentialEnergyCrossScaleFlux(model, filter; dims = (1, 2, 3), method = ProfileLookup(),
                                                geopotential_height = model_geopotential_height(model))
    validate_gravity_is_z_aligned("AvailablePotentialEnergyCrossScaleFlux", model)
    z✶ˡ = filtered_reference_height("AvailablePotentialEnergyCrossScaleFlux", model, filter, method,
                                    geopotential_height)
    return AvailablePotentialEnergyCrossScaleFlux(model, filter, z✶ˡ; dims, geopotential_height)
end

function AvailablePotentialEnergyCrossScaleFlux(model, filter, z✶ˡ::SortedReferenceHeightField;
                                                upsilon = nothing, dims = (1, 2, 3),
                                                geopotential_height = model_geopotential_height(model))
    validate_dims(dims)
    validate_gravity_is_z_aligned("AvailablePotentialEnergyCrossScaleFlux", model)
    validate_reference_height_grid("AvailablePotentialEnergyCrossScaleFlux", model, z✶ˡ)

    Υˡ = isnothing(upsilon) ? Field(AvailablePotentialEnergyDisplacementPotential(model, z✶ˡ)) : upsilon

    # The filtered buoyancy is `z✶ˡ`'s own (the field the caller filtered and looked up), so the one `b̄`
    # serves both the reference height — and so `Υˡ` — and the subfilter buoyancy flux: the buoyancy is
    # filtered once per compute rather than once for the lookup and again for the flux.
    b = buoyancy_field(model, model.buoyancy, geopotential_height)
    b̄ = reference_buoyancy(z✶ˡ.operand)
    τ = subfilter_buoyancy_flux(filter, b, b̄, model.velocities, dims)

    ∂ᵢ = (∂x, ∂y, ∂z)   # the τᵢ are already at cell centers, so only the gradient needs collocating
    Πᵃ = -sum(τ[Symbol(:τ, ('₁', '₂', '₃')[d])] * to_center(∂ᵢ[d](Υˡ)) for d in dims)

    return KernelFunctionOperation{Center, Center, Center}(cross_scale_ape_flux_ccc, model.grid, Πᵃ)
end

AvailablePotentialEnergyCrossScaleFlux(model; σ, dims = (1, 2, 3), boundary = :shrink, N = nothing, kwargs...) =
    AvailablePotentialEnergyCrossScaleFlux(model, GaussianFilter(; dims, σ, boundary, N); dims, kwargs...)
#---

#+++ Filtered available-potential-to-kinetic-energy conversion
# w̄b_rˡ, the term the filtered APE and the filtered KE budgets exchange. `b_rˡ = b̄ - b✶(z)` is the
# filtered buoyancy measured against the reference profile at the parcel's own height — the *unfiltered*
# reference, matching `eₐˡ = eₐ(b̄, z)` itself. `w̄` lives on the z face, so the product is formed there
# and only then interpolated to the cell center, exactly as `PotentialToKineticEnergyConversion` does.
@inline b_rˡᶜᶜᶜ(i, j, k, grid, b̄, b✶z) = @inbounds b̄[i, j, k] - b✶z[i, j, k]

@inline w̄b_rˡᶜᶜᶠ(i, j, k, grid, w̄, b̄, b✶z) = @inbounds w̄[i, j, k] * ℑzᵃᵃᶠ(i, j, k, grid, b_rˡᶜᶜᶜ, b̄, b✶z)

@inline filtered_ape_to_ke_conversion_ccc(i, j, k, grid, w̄, b̄, b✶z) =
    ℑzᵃᵃᶜ(i, j, k, grid, w̄b_rˡᶜᶜᶠ, w̄, b̄, b✶z)

const FilteredAvailablePotentialToKineticEnergyConversion = CustomKFO{<:typeof(filtered_ape_to_ke_conversion_ccc)}

"""
    $(SIGNATURES)

Return the conversion of filtered available potential energy into filtered kinetic energy `w̄b_rˡ`,
the rate at which the scales a low-pass `filter` keeps release their APE to the filtered flow
([Wenegrat, Chor & Barkan, 2026](https://arxiv.org/abs/2605.15879)):

```
    w̄ b_rˡ ,   b_rˡ = b̄ - b✶(z) ,   b̄ = filter(b)
```

Here `b_rˡ` is the buoyancy anomaly of the filtered field against the reference state, and `b✶(z)` is
the reference profile read at the parcel's **own height** `z` rather than at the reference height `z✶`
its buoyancy would take it to (`reference_buoyancy_at_height`). It is the term the filtered APE budget
carries as `-w̄b_rˡ` and the filtered kinetic energy budget as `+w̄b_rˡ`, which is what makes it a
reversible exchange rather than a source or a sink; `w̄b_rˡ > 0` converts filtered APE into filtered KE.

Note the reference profile is **not** filtered: `b_rˡ = b̄ - b✶(z)`, not `filter(b_r) = b̄ - filter(b✶(z))`.
The two differ once the filter acts in the vertical, and only the first is the conversion the filtered
budget carries, since it comes from differentiating `eₐˡ = eₐ(b̄, z)` — which is itself measured against
the full field's reference state ([`FilteredAvailablePotentialEnergy`](@ref)) — with respect to `z`.
For a purely horizontal filter the two coincide, `b✶` being a function of `z` alone.

`method` has to be a [`ProfileLookup`](@ref), for the reason [`FilteredAvailablePotentialEnergy`](@ref)
gives, and it supplies the profile `b✶(z)` is read from. Unlike the other diagnostics here this one
needs no reference *height* of its own, so it takes no `z✶ˡ`; share a profile by passing the same
`ProfileLookup(z✶_column)` the others get.

`filter` is any callable mapping a field to its low-pass-filtered counterpart, e.g. a reusable
[`GaussianFilter`](@ref) or [`BoxFilter`](@ref). The filtered buoyancy, the filtered vertical velocity
and `b✶(z)` are materialized as `Field`s internally, so the returned object is a lazy operation ready
for `Field`, `Integral` and `OutputWriter`s, re-filtering and re-sorting as the simulation evolves. It
lives at `(Center, Center, Center)`, per unit mass (units `m² s⁻³`):

```jldoctest
using Oceananigans, Oceanostics

grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1), topology=(Periodic, Periodic, Bounded))
model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b)

filter = GaussianFilter(; dims=(1, 2, 3), σ=0.1)
FilteredAvailablePotentialToKineticEnergyConversion(model, filter)

# output

FilteredAvailablePotentialToKineticEnergyConversion KernelFunctionOperation at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: filtered_ape_to_ke_conversion_ccc (generic function with 1 method)
└── arguments: ("Field", "Field", "Field")
└── computes: filtered APE to filtered KE conversion  w̄(b̄ - b✶(z))
```

A convenience method `FilteredAvailablePotentialToKineticEnergyConversion(model; σ, dims, boundary, N)`
builds the Gaussian `filter` for you from a standard deviation `σ` (with `σ = ℓ / (2√(2 ln 2))` for a
FWHM `ℓ`).
"""
function FilteredAvailablePotentialToKineticEnergyConversion(model, filter; method = ProfileLookup(),
                                                             geopotential_height = model_geopotential_height(model))
    validate_gravity_is_z_aligned("FilteredAvailablePotentialToKineticEnergyConversion", model)

    # The lookup's profile is the *unfiltered* reference state, which is what `b✶(z)` is read from; `b̄`
    # comes back from the same helper, so the buoyancy is filtered once.
    _, b̄, lookup = filtered_buoyancy_and_lookup("FilteredAvailablePotentialToKineticEnergyConversion", model, filter,
                                                method, geopotential_height)
    b✶z = reference_buoyancy_at_height(model.grid, lookup.profile)
    w̄  = Field(filter(model.velocities.w))

    return KernelFunctionOperation{Center, Center, Center}(filtered_ape_to_ke_conversion_ccc, model.grid, w̄, b̄, b✶z)
end

FilteredAvailablePotentialToKineticEnergyConversion(model; σ, dims = (1, 2, 3), boundary = :shrink, N = nothing, kwargs...) =
    FilteredAvailablePotentialToKineticEnergyConversion(model, GaussianFilter(; dims, σ, boundary, N); kwargs...)
#---

end # module
