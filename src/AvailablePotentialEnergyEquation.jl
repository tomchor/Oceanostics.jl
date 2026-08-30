module AvailablePotentialEnergyEquation

using DocStringExtensions

export AvailablePotentialEnergy, AvailablePotentialEnergyDisplacementPotential, DisplacementPotential
export ReferenceBuoyancyAnomaly, AvailablePotentialToKineticEnergyConversion
export AvailablePotentialEnergyDissipationRate, DissipationRate
# `Φ` is a term of the `eₚ` equation and lives in `PotentialEnergyEquation`; re-exported here because
# `εₐ` is defined as the diapycnal mixing rate less `Φ`, so the two are almost always wanted together.
export PotentialEnergyDiffusiveVerticalBuoyancyFlux
# `wb` is the term this budget exchanges with the kinetic energy one; see `PotentialEnergyEquation`.
export PotentialToKineticEnergyConversion, KineticEnergyConversion
# The reference state lives in `BackgroundPotentialEnergyEquation`; re-exported here so either module
# can be used on its own without reaching across for the pieces that build `z✶`.
export BackgroundPotentialEnergy, reference_height, reference_buoyancy, reference_buoyancy_at_height
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
                                 validate_gravity_is_z_aligned,
                                 buoyancy_diffusive_flux_arguments
using ..BackgroundPotentialEnergyEquation: BackgroundPotentialEnergy, SortedReferenceHeightField,
                                           AbstractReferenceHeightMethod, reference_height,
                                           reference_buoyancy, reference_buoyancy_at_height,
                                           sorted_height, ThreeDimensionalSort,
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
framework. It is **non-negative everywhere** whenever the reference profile is one-dimensional and
gravitationally stable.

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

function AvailablePotentialEnergy(model, z✶::SortedReferenceHeightField)
    validate_gravity_is_z_aligned("AvailablePotentialEnergy", model)
    return available_potential_energy(z✶, reference_buoyancy(z✶.operand), sorted_height(z✶.operand))
end

# On the model grid the parcel's own height is the grid's; on a sorted column it has to be carried.
available_potential_energy(z✶, b, ::Nothing) = KernelFunctionOperation{Center, Center, Center}(local_ape_ccc, z✶.grid, z✶.operand.reference_potential, b, z✶)
available_potential_energy(z✶, b, z)         = KernelFunctionOperation{Center, Center, Center}(local_ape_ccc, z✶.grid, z✶.operand.reference_potential, b, z, z✶)
#---

#+++ Reference height on the model grid
# `Υ` and `εₐ` both read the parcel's own height off the grid `z✶` lives on, so both need that to be
# the model grid. [`VerticalSort`](@ref) answers on the sorted column instead, where the grid's own
# `Zᶜᶜᶜ` *is* `z✶` (which would make `Υ` silently zero) and a horizontal gradient of `b` means nothing.
validate_reference_height_grid(diagnostic, model, z✶) =
    z✶.grid === model.grid ||
        throw(ArgumentError("`$diagnostic` needs a reference height on the model grid, but this one lives on a \
                             $(summary(z✶.grid)). Use `HeavisideIntegral()`, `ThreeDimensionalSort()` or \
                             `ProfileLookup()` rather than `VerticalSort()`."))
#---

#+++ Displacement potential
@inline upsilon_ccc(i, j, k, grid, z✶) = @inbounds z✶[i, j, k] - Zᶜᶜᶜ(i, j, k, grid)

const AvailablePotentialEnergyDisplacementPotential = CustomKFO{<:typeof(upsilon_ccc)}
# Short enough to read beside the other terms of an `eₐ` budget, and scoped to this module the way
# `DissipationRate` is: `using Oceanostics` does not bring it in, since unprefixed it says nothing
# about which budget's displacement it is.
const DisplacementPotential = AvailablePotentialEnergyDisplacementPotential

"""
    $(SIGNATURES)

Return a `KernelFunctionOperation` computing the displacement potential

```
    Υ = z✶ - z
```

how far below its actual height a parcel's reference height sits, and so how far it would have to
travel to reach the adiabatically resorted state. It is the derivative of the local available potential
energy with respect to buoyancy, `Υ = ∂eₐ/∂b`, which is what makes it the natural conjugate of `b`:
contracting it with a buoyancy gradient gives an APE dissipation rate
([`AvailablePotentialEnergyDissipationRate`](@ref)), and contracting it with a subfilter buoyancy flux
gives a cross-scale APE flux.

This is the buoyancy form of `Υ(ρ, z) = g(z - z✶(ρ))/ρ₀` as
[Wenegrat, Chor & Barkan (2026)](https://arxiv.org/abs/2605.15879) write it in their Eq. (7) for
density. The two differ by the factor `-g/ρ₀` that converts between buoyancy and density, which cancels
wherever `Υ` is contracted with a buoyancy gradient. The result lives at `(Center, Center, Center)` and
is a length (units `m`).

`z✶` is the reference height computed by [`reference_height`](@ref); pass one explicitly to share a
single sort with the other reference-state diagnostics, or pass `method` through to choose how it is
built. It has to be one that lives on the model grid, so [`VerticalSort`](@ref) is rejected.

`using Oceanostics.AvailablePotentialEnergyEquation` additionally brings in `DisplacementPotential`,
short enough to read beside the other terms of an `eₐ` budget. That alias is scoped to this module, as
`DissipationRate` is: `using Oceanostics` does not bring it in, since unprefixed it says nothing about
which budget's displacement it names.

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

AvailablePotentialEnergyDisplacementPotential(model)

# output

AvailablePotentialEnergyDisplacementPotential KernelFunctionOperation at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: upsilon_ccc (generic function with 1 method)
└── arguments: ("Field",)
└── computes: displacement potential  Υ = z✶ - z
```
"""
function AvailablePotentialEnergyDisplacementPotential(model; method = HeavisideIntegral(),
                                                       geopotential_height = model_geopotential_height(model),
                                                       location = (Center, Center, Center))
    validate_location(location, "AvailablePotentialEnergyDisplacementPotential")
    return AvailablePotentialEnergyDisplacementPotential(model, reference_height(model; method, geopotential_height))
end

function AvailablePotentialEnergyDisplacementPotential(model, z✶::SortedReferenceHeightField)
    validate_gravity_is_z_aligned("AvailablePotentialEnergyDisplacementPotential", model)
    validate_reference_height_grid("AvailablePotentialEnergyDisplacementPotential", model, z✶)
    return KernelFunctionOperation{Center, Center, Center}(upsilon_ccc, z✶.grid, z✶)
end
#---

#+++ Reference buoyancy anomaly
@inline reference_buoyancy_anomaly_ccc(i, j, k, grid, b, b✶ᶻ) = @inbounds b[i, j, k] - b✶ᶻ[i, j, k]

const ReferenceBuoyancyAnomaly = CustomKFO{<:typeof(reference_buoyancy_anomaly_ccc)}

"""
    $(SIGNATURES)

Return a `KernelFunctionOperation` computing the buoyancy anomaly a parcel carries relative to the
adiabatically sorted reference profile taken at the parcel's **own** height,

```
    bᵣ = b - b✶(z) ,
```

the buoyancy form of `b_r(ρ, z) = -g(ρ - ρ✶(z))/ρ₀` in
[Wenegrat, Chor & Barkan (2026)](https://arxiv.org/abs/2605.15879), their Eq. (8). It is what the
available potential energy exchanges with the kinetic energy
([`AvailablePotentialToKineticEnergyConversion`](@ref)), and the buoyancy-space counterpart of
[`AvailablePotentialEnergyDisplacementPotential`](@ref): `Υ = z✶ - z` measures a parcel's displacement from the reference state
as a height, `bᵣ` measures it as a buoyancy, and both vanish exactly where the fluid is already sorted.

Note that `b✶(z)` is the reference profile at the height the parcel actually occupies, which is not
[`reference_buoyancy`](@ref): that pairs the profile with `z✶` instead, and `b✶(z✶)` is the parcel's own
buoyancy, so the anomaly built from it would be zero everywhere. The profile is sampled by
[`reference_buoyancy_at_height`](@ref), which reads it off the sort rather than repeating it.

The result lives at `(Center, Center, Center)` and is a buoyancy (units `m s⁻²`). `z✶` is the reference
height computed by [`reference_height`](@ref); pass one explicitly to share a single sort with the
other reference-state diagnostics, or pass `method` through to choose how it is built. It has to be one
that lives on the model grid, so [`VerticalSort`](@ref) is rejected.

```jldoctest
using Oceananigans, Oceanostics

grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1), topology=(Periodic, Periodic, Bounded))
model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b)

ReferenceBuoyancyAnomaly(model)

# output

ReferenceBuoyancyAnomaly KernelFunctionOperation at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: reference_buoyancy_anomaly_ccc (generic function with 1 method)
└── arguments: ("Field", "Field")
└── computes: reference buoyancy anomaly  bᵣ = b - b✶(z)
```
"""
function ReferenceBuoyancyAnomaly(model; method = HeavisideIntegral(),
                                  geopotential_height = model_geopotential_height(model),
                                  location = (Center, Center, Center))
    validate_location(location, "ReferenceBuoyancyAnomaly")
    return ReferenceBuoyancyAnomaly(model, reference_height(model; method, geopotential_height))
end

function ReferenceBuoyancyAnomaly(model, z✶::SortedReferenceHeightField)
    validate_gravity_is_z_aligned("ReferenceBuoyancyAnomaly", model)
    validate_reference_height_grid("ReferenceBuoyancyAnomaly", model, z✶)

    return KernelFunctionOperation{Center, Center, Center}(reference_buoyancy_anomaly_ccc, z✶.grid,
                                                           reference_buoyancy(z✶.operand),
                                                           reference_buoyancy_at_height(z✶))
end
#---

#+++ Available potential to kinetic energy conversion
# The product is formed on the face where `w` lives and only then interpolated to the cell center,
# which is the discretization `PotentialToKineticEnergyConversion` uses for `wb` (its `z_dot_g_bᶜᶜᶠ` is
# the same `ℑzᵃᵃᶠ` of a cell-centered buoyancy). Since `bᵣ = b - b✶(z)` and interpolation is linear,
# the two conversions then differ by exactly `w b✶(z)` cell by cell rather than only in the mean.
@inline w_bᵣᶜᶜᶠ(i, j, k, grid, w, bᵣ) = @inbounds w[i, j, k] * ℑzᵃᵃᶠ(i, j, k, grid, bᵣ)

@inline ape_to_ke_conversion_ccc(i, j, k, grid, w, bᵣ) = ℑzᵃᵃᶜ(i, j, k, grid, w_bᵣᶜᶜᶠ, w, bᵣ)

const AvailablePotentialToKineticEnergyConversion = CustomKFO{<:typeof(ape_to_ke_conversion_ccc)}

"""
    $(SIGNATURES)

Return a `KernelFunctionOperation` computing the rate at which available potential energy is converted
into kinetic energy,

```
    w bᵣ = w [b - b✶(z)] ,
```

the exchange term of the local available potential energy equation of
[Wenegrat, Chor & Barkan (2026)](https://arxiv.org/abs/2605.15879), which the `eₐ` budget takes with a
minus sign and the kinetic energy budget with a plus.

This is **not**
[`PotentialToKineticEnergyConversion`](@ref Oceanostics.KineticEnergyEquation.PotentialEnergyConversion),
which computes `uᵢbᵢ` over the total
buoyancy (`wb` under the vertical gravity these diagnostics require) and belongs to the `eₚ` budget.
The two differ by `w b✶(z)`, the exchange between the kinetic energy and the background state, and that
difference is why the pressure that goes with this budget is the deviation from the hydrostatic
pressure of the reference profile. As fields they are different maps; `w b✶(z)` is a flux divergence,
so the two only agree once integrated over a periodic or closed domain:

```
    ∫ w bᵣ dV = ∫ w b dV .
```

The anomaly `bᵣ` is [`ReferenceBuoyancyAnomaly`](@ref), built here unless one is passed as `anomaly`,
which is worth doing when both are wanted, since a `Field` of it is then computed once. `z✶` is the
reference height computed by [`reference_height`](@ref); it has to be one that lives on the model grid,
so [`VerticalSort`](@ref) is rejected. The result lives at `(Center, Center, Center)` and is a
conversion rate per unit mass (units `m² s⁻³`).

```jldoctest
using Oceananigans, Oceanostics

grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1), topology=(Periodic, Periodic, Bounded))
model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b)

z✶ = reference_height(model, method=HeavisideIntegral())
bᵣ = Field(ReferenceBuoyancyAnomaly(model, z✶))   # share the anomaly between the two
AvailablePotentialToKineticEnergyConversion(model, z✶; anomaly=bᵣ)

# output

AvailablePotentialToKineticEnergyConversion KernelFunctionOperation at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: ape_to_ke_conversion_ccc (generic function with 1 method)
└── arguments: ("Field", "Field")
└── computes: available potential to kinetic energy conversion  wbᵣ
```
"""
function AvailablePotentialToKineticEnergyConversion(model; method = HeavisideIntegral(),
                                                     geopotential_height = model_geopotential_height(model),
                                                     location = (Center, Center, Center))
    validate_location(location, "AvailablePotentialToKineticEnergyConversion")
    return AvailablePotentialToKineticEnergyConversion(model, reference_height(model; method, geopotential_height))
end

function AvailablePotentialToKineticEnergyConversion(model, z✶::SortedReferenceHeightField; anomaly = nothing)

    validate_gravity_is_z_aligned("AvailablePotentialToKineticEnergyConversion", model)
    validate_reference_height_grid("AvailablePotentialToKineticEnergyConversion", model, z✶)

    bᵣ = isnothing(anomaly) ? ReferenceBuoyancyAnomaly(model, z✶) : anomaly

    return KernelFunctionOperation{Center, Center, Center}(ape_to_ke_conversion_ccc, z✶.grid,
                                                           model.velocities.w, bᵣ)
end
#---

#+++ Available potential energy dissipation rate
# `εₐ = -qᵢ ∂ᵢΥ`, where `qᵢ` is the buoyancy tracer's diffusive flux (`-κ ∂ᵢb` for Fickian diffusion,
# but the form is never assumed: an LES closure's flux goes through unchanged). Taking it
# from the closure's own `diffusive_flux_*` rather than from a diffusivity of our own makes this follow
# whatever closure the model runs with, and keeps the dissipation consistent with the diffusion the
# model actually applied — the same conservative formulation `TracerVarianceDissipationRate` uses for
# `χ = -2 ∂ⱼc·qᶜⱼ`. Each product is formed on the face where both factors live and only then interpolated
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
    εₐ = -qᵢ ∂ᵢΥ = -(∂z✶/∂b) qᵢ ∂ᵢb + q₃ ,
```

the sink of the local available potential energy equation of
[Wenegrat, Chor & Barkan (2026)](https://arxiv.org/abs/2605.15879) (their Eqs. 11 and 14, where it
appears as `-εₐ`), with `Υ` the [`AvailablePotentialEnergyDisplacementPotential`](@ref) and `qᵢ` the
diffusive buoyancy flux the closure supplies. It follows from `∂eₐ/∂b = Υ`, which makes the diffusive
part of `Deₐ/Dt` equal to `-Υ ∂ᵢqᵢ = -∂ᵢ(Υqᵢ) + qᵢ∂ᵢΥ`: once the flux divergence is set aside,
`εₐ = -qᵢ∂ᵢΥ` is what remains. Nothing here assumes a form for `qᵢ`: it is `-κ∂ᵢb` for Fickian
diffusion, and whatever an LES closure returns otherwise.

Written out, the first part is the diapycnal mixing rate of
[Winters et al. (1995)](https://doi.org/10.1017/S002211209500125X), the work done rearranging the
reference state, and the second is
[`PotentialEnergyDiffusiveVerticalBuoyancyFlux`](@ref Oceanostics.PotentialEnergyEquation.DiffusiveVerticalBuoyancyFlux), the diffusion that state
undergoes on its own, which carries no APE with it. The two cancel exactly for a statically stable,
horizontally uniform stratification, where `z✶ = z` and there is no available energy to destroy, so
`εₐ` measures only the APE actually lost — it is not the sign-definite buoyancy-variance-like quantity
the name might suggest.

`qᵢ` is taken from the closure's own diffusive flux rather than from a diffusivity supplied here, so
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
[`AvailablePotentialEnergyDisplacementPotential`](@ref) gives. `upsilon` takes a `Υ` you already have, so that writing
both out costs one sort and one `Υ` rather than two of each:

```jldoctest
using Oceananigans, Oceanostics

grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1), topology=(Periodic, Periodic, Bounded))
model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b, closure=ScalarDiffusivity(κ=1e-4))

z✶ = reference_height(model, method=HeavisideIntegral())
Υ = Field(AvailablePotentialEnergyDisplacementPotential(model, z✶))
AvailablePotentialEnergyDissipationRate(model, z✶; upsilon=Υ)

# output

AvailablePotentialEnergyDissipationRate KernelFunctionOperation at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: ape_dissipation_rate_ccc (generic function with 1 method)
└── arguments: ("Field", "ScalarDiffusivity", "Nothing", "Val", "Field", "Clock", "NamedTuple", "BuoyancyForce")
└── computes: available potential energy dissipation rate  εₐ = -qᵢ∂ᵢΥ
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
    validate_gravity_is_z_aligned("AvailablePotentialEnergyDissipationRate", model)
    validate_reference_height_grid("AvailablePotentialEnergyDissipationRate", model, z✶)

    Υ = isnothing(upsilon) ? Field(DisplacementPotential(model, z✶)) : upsilon

    return KernelFunctionOperation{Center, Center, Center}(ape_dissipation_rate_ccc, model.grid,
                                                           Υ, buoyancy_diffusive_flux_arguments(model)...)
end
#---

end # module
