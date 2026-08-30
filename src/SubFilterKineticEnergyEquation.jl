module SubFilterKineticEnergyEquation

using DocStringExtensions

export SubFilterKineticEnergy, SubFilterKineticEnergyDissipationRate, DissipationRate
# Πₖ is a source term of the subfilter KE budget (and a sink of the filtered budget), so it is
# re-exported here from `FilteredKineticEnergyEquation`, where it is defined.
export KineticEnergyCrossScaleFlux
# τˡ(w, bᵣ) is the other source of this budget — the APE the subfilter scales release to it — and a sink
# of the subfilter APE one, so it is re-exported here from `SubFilterAvailablePotentialEnergyEquation`.
export SubFilterAvailablePotentialToKineticEnergyConversion

using Oceananigans.Fields: Field
using Oceananigans.Grids: Center
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceanostics: CustomKFO

using ..KineticEnergyEquation: KineticEnergyDissipationRate, KineticEnergy
using ..FilteredKineticEnergyEquation: FilteredKineticEnergy, FilteredKineticEnergyDissipationRate,
                                       KineticEnergyCrossScaleFlux, filtered_kinetic_energy_ccc, filtered_velocities
using ..FilteredKineticEnergyEquation: subfilter_stress_tensor # re-exported for convenience
# `GaussianFilter` is used by the convenience methods; `BoxFilter` is imported only so its docstring
# `@ref` resolves in-module.
using ..SpatialFilters: GaussianFilter, BoxFilter
using ..SubFilterAvailablePotentialEnergyEquation: SubFilterAvailablePotentialToKineticEnergyConversion

#+++ Subfilter kinetic energy
# eₖˢ = filter(eₖ) - eₖˡ: the filtered full kinetic energy minus the kinetic energy of the filtered flow.
# Because `KineticEnergy` and `FilteredKineticEnergy` use the same interpolate-the-square (½⟨uᵢ²⟩)
# discretization, the discrete decomposition filter(eₖ) = eₖˡ + eₖˢ then holds exactly, by construction, on
# any grid. The kernel reads the materialized filtered full KE `k̄ = filter(eₖ)` and recomputes eₖˡ in place
# via `filtered_kinetic_energy_ccc` on the materialized filtered velocities `ūᵢ = filter(uᵢ)`.
@inline subfilter_kinetic_energy_ccc(i, j, k, grid, k̄, ū, v̄, w̄) = @inbounds k̄[i, j, k] - filtered_kinetic_energy_ccc(i, j, k, grid, ū, v̄, w̄)

const SubFilterKineticEnergy = CustomKFO{<:typeof(subfilter_kinetic_energy_ccc)}

"""
    $(SIGNATURES)

Return the subfilter-scale (SFS) kinetic energy `eₖˢ`, the kinetic energy carried by the scales that a
low-pass `filter` removes from the flow — the filtered full kinetic energy minus the kinetic energy of the
filtered flow:

```
    eₖˢ = filter(eₖ) - eₖˡ ,   eₖ = ½ uᵢuᵢ ,   eₖˡ = ½ ūᵢūᵢ ,   ūᵢ = filter(uᵢ)
```

equivalently `eₖˢ = ½ τᵢᵢ` with the subfilter stress `τᵢⱼ = filter(uᵢuⱼ) - ūᵢūⱼ` (filtering
framework of Aluie et al., 2018, *J. Phys. Oceanogr.*, doi:10.1175/JPO-D-17-0100.1). It is assembled from
the full kinetic energy `eₖ` and [`FilteredKineticEnergy`](@ref) `eₖˡ`, which share the same
interpolate-the-square (`½⟨uᵢ²⟩`) discretization, so the discrete decomposition `filter(eₖ) = eₖˡ + eₖˢ` holds
exactly by construction (on any grid, not just where the filter and interpolation commute).

`filter` is any callable mapping a field to its low-pass-filtered counterpart, e.g. a reusable
[`GaussianFilter`](@ref) or [`BoxFilter`](@ref). The full kinetic energy is materialized as a `Field`
before it is filtered (so the separable filter takes its fast staged path); the result lives at
`(Center, Center, Center)`, per unit mass (units `m² s⁻²`):

```jldoctest
using Oceananigans, Oceanostics

grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1), topology=(Periodic, Periodic, Bounded))
model = NonhydrostaticModel(grid)

filter = GaussianFilter(; dims=(1, 2, 3), σ=0.1)
SubFilterKineticEnergy(model, filter)

# output

SubFilterKineticEnergy KernelFunctionOperation at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: subfilter_kinetic_energy_ccc (generic function with 1 method)
└── arguments: ("Field", "Field", "Field", "Field")
└── computes: subfilter kinetic energy  ½τᵢᵢ
```

A convenience method `SubFilterKineticEnergy(model; σ, dims, boundary, N)` builds the Gaussian `filter`
for you from a standard deviation `σ` (with `σ = ℓ / (2√(2 ln 2))` for a FWHM `ℓ`).
"""
function SubFilterKineticEnergy(model, filter)
    u, v, w = model.velocities
    k  = KineticEnergy(model, u, v, w)           # full kinetic energy eₖ = ½uᵢuᵢ (kinetic_energy_ccc)
    k̄ = Field(filter(Field(k)))                  # filter(eₖ) materialized so the filter takes its staged path
    ū, v̄, w̄ = filtered_velocities(filter, (1, 2, 3), u, v, w) # ūᵢ = filter(uᵢ), materialized as `Field`s
    return KernelFunctionOperation{Center, Center, Center}(subfilter_kinetic_energy_ccc, model.grid, k̄, ū, v̄, w̄)
end

SubFilterKineticEnergy(model; σ, dims = (1, 2, 3), boundary = :shrink, N = nothing) =
    SubFilterKineticEnergy(model, GaussianFilter(; dims, σ, boundary, N))
#---

#+++ Subfilter kinetic energy dissipation
# Exposed as a single `KernelFunctionOperation` using the same wrapper trick as `KineticEnergyCrossScaleFlux`:
# the kernel just indexes the pre-assembled operation εₖˢ = filter(εₖ) - εₖˡ, whose leaves are the materialized
# filtered `Field`s, so per-cell evaluation only reads those fields and subtracts (it never re-filters).
@inline subfilter_ke_dissipation_rate_ccc(i, j, k, grid, εₖˢ) = @inbounds εₖˢ[i, j, k]

const SubFilterKineticEnergyDissipationRate = CustomKFO{<:typeof(subfilter_ke_dissipation_rate_ccc)}
const DissipationRate = SubFilterKineticEnergyDissipationRate

"""
    $(SIGNATURES)

Return the subfilter-scale (SFS) kinetic-energy dissipation rate `εₖˢ`, the viscous dissipation carried
by the scales that a low-pass `filter` removes:

```
    εₖˢ = filter(εₖ) - εₖˡ
```

where `εₖ` is the dissipation rate of the full flow
([`KineticEnergyDissipationRate`](@ref Oceanostics.KineticEnergyEquation.DissipationRate)) and `εₖˡ` is the
dissipation rate of the filtered flow ([`FilteredKineticEnergyDissipationRate`](@ref)). It is the viscous
sink in the budget of the subfilter kinetic energy `eₖˢ` ([`SubFilterKineticEnergy`](@ref);
filtering framework of Aluie et al., 2018, *J. Phys. Oceanogr.*, doi:10.1175/JPO-D-17-0100.1). For a
constant viscosity it reduces to `2ν[filter(SᵢⱼSᵢⱼ) - S̄ᵢⱼ S̄ᵢⱼ] ≥ 0`, a strictly positive sink.

`filter` is any callable mapping a field to its low-pass-filtered counterpart, e.g. a reusable
[`GaussianFilter`](@ref) or [`BoxFilter`](@ref). Following the `KineticEnergyCrossScaleFlux` pattern, the
result is a single `KernelFunctionOperation` whose kernel indexes a pre-assembled operation with
materialized filtered `Field` leaves (the full-flow dissipation `εₖ` is materialized before it is filtered,
and the filtered result is wrapped in a `Field` so the separable filter takes its fast staged path). It
lives at `(Center, Center, Center)`, per unit mass (units `m² s⁻³`). The model needs a closure whose
viscous fluxes are defined, exactly as [`FilteredKineticEnergyDissipationRate`](@ref) requires:

```jldoctest
using Oceananigans, Oceanostics

grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1), topology=(Periodic, Periodic, Bounded))
model = NonhydrostaticModel(grid; closure=ScalarDiffusivity(ν=1e-4))

filter = GaussianFilter(; dims=(1, 2, 3), σ=0.1)
SubFilterKineticEnergyDissipationRate(model, filter)

# output

SubFilterKineticEnergyDissipationRate KernelFunctionOperation at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: subfilter_ke_dissipation_rate_ccc (generic function with 1 method)
└── arguments: ("Oceananigans.AbstractOperations.BinaryOperation",)
└── computes: subfilter kinetic energy dissipation rate  filter(εₖ) - εₖˡ
```

A convenience method `SubFilterKineticEnergyDissipationRate(model; σ, dims, boundary, N)` builds the
Gaussian `filter` for you from a standard deviation `σ` (with `σ = ℓ / (2√(2 ln 2))` for a FWHM `ℓ`).
"""
function SubFilterKineticEnergyDissipationRate(model, filter)
    εₖ  = KineticEnergyDissipationRate(model)                 # dissipation of the full flow
    εₖˡ = FilteredKineticEnergyDissipationRate(model, filter)  # dissipation of the filtered flow
    εₖˢ = Field(filter(Field(εₖ))) - εₖˡ                       # εₖˢ = filter(εₖ) - εₖˡ; leaves are materialized Fields
    return KernelFunctionOperation{Center, Center, Center}(subfilter_ke_dissipation_rate_ccc, model.grid, εₖˢ)
end

SubFilterKineticEnergyDissipationRate(model; σ, dims = (1, 2, 3), boundary = :shrink, N = nothing) =
    SubFilterKineticEnergyDissipationRate(model, GaussianFilter(; dims, σ, boundary, N))
#---

end # module
