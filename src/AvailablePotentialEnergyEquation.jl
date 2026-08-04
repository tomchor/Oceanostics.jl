module AvailablePotentialEnergyEquation

using DocStringExtensions

export AvailablePotentialEnergy, BuoyancyDisplacementPotential
export AvailablePotentialEnergyDissipationRate, DissipationRate
# `Φ` is a term of the `e_p` equation and lives in `PotentialEnergyEquation`; re-exported here because
# `ε_A` is defined as the diapycnal mixing rate less `Φ`, so the two are almost always wanted together.
export PotentialEnergyDiffusiveVerticalBuoyancyFlux
# `wb` is the term this budget exchanges with the kinetic energy one; see `PotentialEnergyEquation`.
export PotentialToKineticEnergyConversion, KineticEnergyConversion
# The reference state lives in `BackgroundPotentialEnergyEquation`; re-exported here so either module
# can be used on its own without reaching across for the pieces that build `z✶`.
export BackgroundPotentialEnergy, reference_height, reference_buoyancy
export ThreeDimensionalSort, HeavisideIntegral, VerticalSort, ProfileLookup

using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Fields: Field
using Oceananigans.Grids: Center, Face
using Oceananigans.Models: model_geopotential_height
using Oceananigans.BuoyancyFormulations: Zᶜᶜᶜ
using Oceananigans.Operators
using Oceananigans.TurbulenceClosures: diffusive_flux_x, diffusive_flux_y, diffusive_flux_z
using Oceanostics: validate_location, CustomKFO

# Imported so the docstring `@ref`s below resolve in-module, as well as for dispatch.
using ..PotentialEnergyEquation: PotentialEnergy, PotentialEnergyDiffusiveVerticalBuoyancyFlux,
                                 PotentialToKineticEnergyConversion, KineticEnergyConversion,
                                 validate_buoyancy_is_a_diffused_tracer, validate_closure_supplies_a_flux,
                                 buoyancy_diffusive_flux_arguments
using ..BackgroundPotentialEnergyEquation: BackgroundPotentialEnergy, SortedReferenceHeightField,
                                           AbstractReferenceHeightMethod, reference_height,
                                           reference_buoyancy, sorted_height, ThreeDimensionalSort,
                                           HeavisideIntegral, VerticalSort, ProfileLookup

#+++ Available potential energy
# The local available potential energy `eₐ = ∫_{z✶}^{z} [b✶(z̃) - b] dz̃`, split into the part only the
# reference profile can supply (`Ψδ = Ψ(z) - Ψ(z✶)`, see `fill_reference_potential!`) and the
# `-b(z - z✶)` the kernel does pointwise. The parcel's own height is the grid's `Zᶜᶜᶜ` on the model
# grid, a carried field on a column.
@inline local_ape_ccc(i, j, k, grid, Ψδ, b, z✶) = @inbounds Ψδ[i, j, k] - b[i, j, k] * (Zᶜᶜᶜ(i, j, k, grid) - z✶[i, j, k])
@inline local_ape_ccc(i, j, k, grid, Ψδ, b, z, z✶) = @inbounds Ψδ[i, j, k] - b[i, j, k] * (z[i, j, k] - z✶[i, j, k])

const AvailablePotentialEnergy = CustomKFO{<:typeof(local_ape_ccc)}

"""
    $(SIGNATURES)

Return a `KernelFunctionOperation` computing the **local** available potential energy density,

```
    eₐ(b, z) = ∫_{z✶}^{z} [b✶(z̃) - b] dz̃ ,   equivalently   (g/ρ₀) ∫_{z✶}^{z} [ρ - ρ✶(z̃)] dz̃
```

the work needed to bring a parcel from the reference height `z✶` it would occupy in the adiabatically
resorted state to the height `z` where it actually sits. The parcel's own buoyancy `b` is held fixed
along the path; only the reference profile `b✶` varies with `z̃`.

This is the spatially local APE density of
[Holliday & McIntyre (1981)](https://doi.org/10.1017/S0022112081001742) and it is also used in
[Wenegrat, Chor & Barkan (2026)](https://arxiv.org/abs/2605.15879) as a basis for a filtered APE
framework. When the reference state is sorted from the buoyancy itself (either from a single
buoyancy snapshot or the up-to-date buoyancy `Field`), it is **non-negative everywhere**, which
follows from the convexity of that integral. It is also possible to hand [`ProfileLookup`](@ref) a
different profile not produced by model's buoyancy, which breaks the `b = b✶(z✶)` step and, with
it, the non-negativity guarantee.

`z✶` is the reference height computed by [`reference_height`](@ref); pass one explicitly to share a
single sort with [`BackgroundPotentialEnergy`](@ref), or pass `method` through to choose how it is
built. All four give the same `eₐ` volume integral when each sorts the field itself.

`Integral(eₐ)` recovers the global APE `Integral(PotentialEnergy(model)) -
Integral(BackgroundPotentialEnergy(model))` only in the continuum limit: the local density samples the
reference profile at the model's cell centers while the global split effectively samples it at the
sorted column's, and the two midpoint quadratures differ at finite `Δz`. The gap is second order in
the vertical spacing, so it is a fraction of a percent on a well resolved grid but a few percent on a
coarse one. `eₐ` does vanish, cell by cell and exactly, for a statically stable and horizontally
uniform stratification.

The result lives at `(Center, Center, Center)`, per unit mass (units `m² s⁻²`), and is defined for the
same buoyancy formulations as [`PotentialEnergy`](@ref). Under [`VerticalSort`](@ref) it lands
on the sorted column, indexed by rank rather than by position in the flow.

```jldoctest
using Oceananigans, Oceanostics

grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1), topology=(Periodic, Periodic, Bounded))
model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b)

AvailablePotentialEnergy(model)

# output

AvailablePotentialEnergy KernelFunctionOperation at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: local_ape_ccc (generic function with 2 methods)
└── arguments: ("Field", "Field", "Field")
└── computes: local available potential energy density  eₐ = ∫[b✶(z̃) - b]dz̃ ≥ 0
```
"""
function AvailablePotentialEnergy(model; method = ThreeDimensionalSort(), geopotential_height = model_geopotential_height(model), location = (Center, Center, Center))
    validate_location(location, "AvailablePotentialEnergy")
    return AvailablePotentialEnergy(model, reference_height(model; method, geopotential_height))
end

AvailablePotentialEnergy(model, z✶::SortedReferenceHeightField) = available_potential_energy(z✶, reference_buoyancy(z✶.operand), sorted_height(z✶.operand))

# On the model grid the parcel's own height is the grid's; on a sorted column it has to be carried.
available_potential_energy(z✶, b, ::Nothing) = KernelFunctionOperation{Center, Center, Center}(local_ape_ccc, z✶.grid, z✶.operand.reference_potential, b, z✶)
available_potential_energy(z✶, b, z)         = KernelFunctionOperation{Center, Center, Center}(local_ape_ccc, z✶.grid, z✶.operand.reference_potential, b, z, z✶)
#---

#+++ Reference height on the model grid
# `Υ` and `ε_A` both read the parcel's own height off the grid `z✶` lives on, so both need that to be
# the model grid. [`VerticalSort`](@ref) answers on the sorted column instead, where the grid's own
# `Zᶜᶜᶜ` *is* `z✶` (which would make `Υ` silently zero) and a horizontal gradient of `b` means nothing.
validate_reference_height_grid(diagnostic, model, z✶) =
    z✶.grid === model.grid ||
        throw(ArgumentError("`$diagnostic` needs a reference height on the model grid, but this one lives on a \
                             $(summary(z✶.grid)). Use `HeavisideIntegral()`, `ThreeDimensionalSort()` or \
                             `ProfileLookup()` rather than `VerticalSort()`."))
#---

#+++ Buoyancy displacement potential
@inline upsilon_ccc(i, j, k, grid, z✶) = @inbounds z✶[i, j, k] - Zᶜᶜᶜ(i, j, k, grid)

const BuoyancyDisplacementPotential = CustomKFO{<:typeof(upsilon_ccc)}

"""
    $(SIGNATURES)

Return a `KernelFunctionOperation` computing the buoyancy displacement potential

```
    Υ = z✶ - z
```

how far below its actual height a parcel's reference height sits, and so how far it would have to
travel to reach the adiabatically resorted state. It is the derivative of the local available potential
energy with respect to buoyancy, `Υ = ∂eₐ/∂b`, which is what makes it the natural conjugate of `b`:
contracting it with a buoyancy gradient gives an APE dissipation rate
([`AvailablePotentialEnergyDissipationRate`](@ref)), and contracting it with a sub-filter buoyancy flux
gives a cross-scale APE flux.

This is the buoyancy form of `Υ(ρ, z) = g(z - z✶(ρ))/ρ₀` as
[Wenegrat, Chor & Barkan (2026)](https://arxiv.org/abs/2605.15879) write it in their Eq. (7) for
density. The two differ by the factor `-g/ρ₀` that converts between buoyancy and density, which cancels
wherever `Υ` is contracted with a buoyancy gradient. The result lives at `(Center, Center, Center)` and
is a length (units `m`).

`z✶` is the reference height computed by [`reference_height`](@ref); pass one explicitly to share a
single sort with the other reference-state diagnostics, or pass `method` through to choose how it is
built. It has to be one that lives on the model grid, so [`VerticalSort`](@ref) is rejected.

[`HeavisideIntegral`](@ref) is the default here rather than the package-wide
[`ThreeDimensionalSort`](@ref) because `Υ` is a map, and every use of it differentiates that map. Only
Eq. (11) of Winters et al. makes `z✶` a function of buoyancy alone, so tied cells share one reference
height; with [`ThreeDimensionalSort`](@ref) a run of equal buoyancy takes consecutive slots and spreads
`z✶` over the depth it fills, which is harmless in a volume integral but shows up in `∇Υ` as grid-scale
noise.

```jldoctest
using Oceananigans, Oceanostics

grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1), topology=(Periodic, Periodic, Bounded))
model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b)

BuoyancyDisplacementPotential(model)

# output

BuoyancyDisplacementPotential KernelFunctionOperation at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: upsilon_ccc (generic function with 1 method)
└── arguments: ("Field",)
└── computes: buoyancy displacement potential  Υ = z✶ - z
```
"""
function BuoyancyDisplacementPotential(model; method = HeavisideIntegral(),
                                       geopotential_height = model_geopotential_height(model),
                                       location = (Center, Center, Center))
    validate_location(location, "BuoyancyDisplacementPotential")
    return BuoyancyDisplacementPotential(model, reference_height(model; method, geopotential_height))
end

function BuoyancyDisplacementPotential(model, z✶::SortedReferenceHeightField)
    validate_reference_height_grid("BuoyancyDisplacementPotential", model, z✶)
    return KernelFunctionOperation{Center, Center, Center}(upsilon_ccc, z✶.grid, z✶)
end
#---

#+++ Available potential energy dissipation rate
# `ε_A = κ ∂ᵢb ∂ᵢΥ = -qᵢ ∂ᵢΥ`, where `qᵢ = -κ ∂ᵢb` is the buoyancy tracer's diffusive flux. Taking it
# from the closure's own `diffusive_flux_*` rather than from a diffusivity of our own makes this follow
# whatever closure the model runs with, and keeps the dissipation consistent with the diffusion the
# model actually applied — the same conservative formulation `TracerVarianceDissipationRate` uses for
# `χ = 2 ∂ⱼc·Fⱼ`. Each product is formed on the face where both factors live and only then interpolated
# to the cell center, so a no-flux boundary (where the tracer halo is mirrored, making `δb` there
# exactly zero) contributes nothing.
@inline Axᶠᶜᶜ_δΥᶠᶜᶜ_q₁ᶠᶜᶜ(i, j, k, grid, Υ, closure, closure_fields, id, c, args...) =
    - Axᶠᶜᶜ(i, j, k, grid) * δxᶠᵃᵃ(i, j, k, grid, Υ) * diffusive_flux_x(i, j, k, grid, closure, closure_fields, id, c, args...)

@inline Ayᶜᶠᶜ_δΥᶜᶠᶜ_q₂ᶜᶠᶜ(i, j, k, grid, Υ, closure, closure_fields, id, c, args...) =
    - Ayᶜᶠᶜ(i, j, k, grid) * δyᵃᶠᵃ(i, j, k, grid, Υ) * diffusive_flux_y(i, j, k, grid, closure, closure_fields, id, c, args...)

@inline Azᶜᶜᶠ_δΥᶜᶜᶠ_q₃ᶜᶜᶠ(i, j, k, grid, Υ, closure, closure_fields, id, c, args...) =
    - Azᶜᶜᶠ(i, j, k, grid) * δzᵃᵃᶠ(i, j, k, grid, Υ) * diffusive_flux_z(i, j, k, grid, closure, closure_fields, id, c, args...)

@inline ape_dissipation_rate_ccc(i, j, k, grid, args...) =
    (ℑxᶜᵃᵃ(i, j, k, grid, Axᶠᶜᶜ_δΥᶠᶜᶜ_q₁ᶠᶜᶜ, args...) + # F, C, C  → C, C, C
     ℑyᵃᶜᵃ(i, j, k, grid, Ayᶜᶠᶜ_δΥᶜᶠᶜ_q₂ᶜᶠᶜ, args...) + # C, F, C  → C, C, C
     ℑzᵃᵃᶜ(i, j, k, grid, Azᶜᶜᶠ_δΥᶜᶜᶠ_q₃ᶜᶜᶠ, args...)   # C, C, F  → C, C, C
     ) / Vᶜᶜᶜ(i, j, k, grid) # this division by volume, against the `A δΥ` above, is what makes it a derivative

const AvailablePotentialEnergyDissipationRate = CustomKFO{<:typeof(ape_dissipation_rate_ccc)}
const DissipationRate = AvailablePotentialEnergyDissipationRate

"""
    $(SIGNATURES)

Return a `KernelFunctionOperation` computing the rate at which diffusion destroys available potential
energy,

```
    ε_A = κ ∂ᵢb ∂ᵢΥ = κ [(∂z✶/∂b)|∇b|² - ∂b/∂z] ,
```

the sink of the local available potential energy equation of
[Wenegrat, Chor & Barkan (2026)](https://arxiv.org/abs/2605.15879) (their Eqs. 11 and 14, where it
appears as `-ε_A`), with `Υ` the [`BuoyancyDisplacementPotential`](@ref). It follows from
`∂eₐ/∂b = Υ`, which makes the diffusive part of `Deₐ/Dt` equal to `Υκ∇²b = ∇·(κΥ∇b) - κ∇Υ·∇b`: once
the flux divergence is set aside, `ε_A = κ∇Υ·∇b` is what remains.

Written out, the first part is the diapycnal mixing rate of
[Winters et al. (1995)](https://doi.org/10.1017/S002211209500125X), the work done rearranging the
reference state, and the second is
[`PotentialEnergyDiffusiveVerticalBuoyancyFlux`](@ref Oceanostics.PotentialEnergyEquation.DiffusiveVerticalBuoyancyFlux), the diffusion that state
undergoes on its own, which carries no APE with it. The two cancel exactly for a statically stable,
horizontally uniform stratification, where `z✶ = z` and there is no available energy to destroy, so
`ε_A` measures only the APE actually lost — it is not the sign-definite `κ|∇b|²`-like quantity the name
might suggest.

`κ ∂ᵢb` is taken from the closure's own diffusive flux rather than from a diffusivity supplied here, so
this follows whatever closure the model runs with, and is written in the same conservative form
[`TracerVarianceDissipationRate`](@ref Oceanostics.TracerVarianceEquation.TracerVarianceDissipationRate) uses. The
result lives at `(Center, Center, Center)`, per unit mass (units `m² s⁻³`).

The buoyancy has to be a tracer the closure diffuses, so this is defined for `BuoyancyTracer` models
only — `SeawaterBuoyancy` would need the diffusive fluxes of temperature and salinity combined through
the equation of state.

`upsilon` is a keyword of the two-argument form only. Anyone holding a `Υ` also holds the `z✶` it was
built from, and passing that is both cheaper and unambiguous: this form would have to sort the domain
to build a `z✶` it then uses for nothing but a grid check.

`z✶` is the reference height computed by [`reference_height`](@ref), and has to be one that lives on
the model grid, since `∇b` is taken there; [`HeavisideIntegral`](@ref) is the default for the reason
[`BuoyancyDisplacementPotential`](@ref) gives. `upsilon` takes a `Υ` you already have, so that writing
both out costs one sort and one `Υ` rather than two of each:

```jldoctest
using Oceananigans, Oceanostics

grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1), topology=(Periodic, Periodic, Bounded))
model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b, closure=ScalarDiffusivity(κ=1e-4))

z✶ = reference_height(model, method=HeavisideIntegral())
Υ = Field(BuoyancyDisplacementPotential(model, z✶))
AvailablePotentialEnergyDissipationRate(model, z✶; upsilon=Υ)

# output

AvailablePotentialEnergyDissipationRate KernelFunctionOperation at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: ape_dissipation_rate_ccc (generic function with 1 method)
└── arguments: ("Field", "ScalarDiffusivity", "Nothing", "Val", "Field", "Clock", "NamedTuple", "BuoyancyForce")
└── computes: available potential energy dissipation rate  ε_A = κ ∂ᵢb ∂ᵢΥ
```
"""
function AvailablePotentialEnergyDissipationRate(model; method = HeavisideIntegral(),
                                                 geopotential_height = model_geopotential_height(model),
                                                 location = (Center, Center, Center))
    validate_location(location, "AvailablePotentialEnergyDissipationRate")
    return AvailablePotentialEnergyDissipationRate(model, reference_height(model; method, geopotential_height))
end

function AvailablePotentialEnergyDissipationRate(model, z✶::SortedReferenceHeightField; upsilon = nothing)

    validate_buoyancy_is_a_diffused_tracer("AvailablePotentialEnergyDissipationRate", model)
    validate_closure_supplies_a_flux("AvailablePotentialEnergyDissipationRate", model)
    validate_reference_height_grid("AvailablePotentialEnergyDissipationRate", model, z✶)

    Υ = isnothing(upsilon) ? Field(BuoyancyDisplacementPotential(model, z✶)) : upsilon

    return KernelFunctionOperation{Center, Center, Center}(ape_dissipation_rate_ccc, model.grid,
                                                           Υ, buoyancy_diffusive_flux_arguments(model)...)
end
#---

end # module
