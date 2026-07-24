module AvailablePotentialEnergyEquation

using DocStringExtensions

export AvailablePotentialEnergy
# The reference state lives in `BackgroundPotentialEnergyEquation`; re-exported here so either module
# can be used on its own without reaching across for the pieces that build `z✶`.
export BackgroundPotentialEnergy, reference_height, reference_buoyancy
export ThreeDimensionalSort, HeavisideIntegral, VerticalSort, ProfileLookup

using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Fields: Field
using Oceananigans.Grids: Center, Face
using Oceananigans.Models: model_geopotential_height
using Oceananigans.BuoyancyFormulations: Zᶜᶜᶜ
using Oceanostics: validate_location, CustomKFO

# Imported so the docstring `@ref`s below resolve in-module, as well as for dispatch.
using ..PotentialEnergyEquation: PotentialEnergy
using ..BackgroundPotentialEnergyEquation: BackgroundPotentialEnergy, SortedReferenceHeightField,
                                           AbstractReferenceHeightMethod, reference_height,
                                           reference_buoyancy, sorted_height, ThreeDimensionalSort,
                                           HeavisideIntegral, VerticalSort, ProfileLookup

#+++ Available potential energy
# The local available potential energy `Eₐ = ∫_{z✶}^{z} [b✶(z̃) - b] dz̃`, split into the part only the
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
    Eₐ(b, z) = ∫_{z✶}^{z} [b✶(z̃) - b] dz̃ ,   equivalently   (g/ρ₀) ∫_{z✶}^{z} [ρ - ρ✶(z̃)] dz̃
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
built. All four give the same `Eₐ` volume integral when each sorts the field itself.

`Integral(Eₐ)` recovers the global APE `Integral(PotentialEnergy(model)) -
Integral(BackgroundPotentialEnergy(model))` only in the continuum limit: the local density samples the
reference profile at the model's cell centers while the global split effectively samples it at the
sorted column's, and the two midpoint quadratures differ at finite `Δz`. The gap is second order in
the vertical spacing, so it is a fraction of a percent on a well resolved grid but a few percent on a
coarse one. `Eₐ` does vanish, cell by cell and exactly, for a statically stable and horizontally
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
└── computes: local available potential energy density  Eₐ = ∫[b✶(z̃) - b]dz̃ ≥ 0
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

end # module
