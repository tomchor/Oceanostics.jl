module SubFilterKineticEnergyEquation

using DocStringExtensions

export SubFilterKineticEnergy, SubFilterKineticEnergyDissipationRate, DissipationRate
# Πₖ is a source term of the sub-filter KE budget (and a sink of the filtered budget), so it is
# re-exported here from `FilteredKineticEnergyEquation`, where it is defined.
export KineticEnergyCrossScaleFlux

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

#+++ Sub-filter kinetic energy
# Kˢ = filter(K) - Kˡ: the filtered full kinetic energy minus the kinetic energy of the filtered flow.
# Because `KineticEnergy` and `FilteredKineticEnergy` use the same interpolate-the-square (½⟨uᵢ²⟩)
# discretization, the discrete decomposition filter(K) = Kˡ + Kˢ then holds exactly, by construction, on
# any grid. The kernel reads the materialized filtered full KE `k̄ = filter(K)` and recomputes Kˡ in place
# via `filtered_kinetic_energy_ccc` on the materialized filtered velocities `ūᵢ = filter(uᵢ)`.
@inline subfilter_kinetic_energy_ccc(i, j, k, grid, k̄, ū, v̄, w̄) = @inbounds k̄[i, j, k] - filtered_kinetic_energy_ccc(i, j, k, grid, ū, v̄, w̄)

const SubFilterKineticEnergy = CustomKFO{<:typeof(subfilter_kinetic_energy_ccc)}

"""
    $(SIGNATURES)

Return the sub-filter-scale (SFS) kinetic energy `Kˢ`, the kinetic energy carried by the scales that a
low-pass `filter` removes from the flow — the filtered full kinetic energy minus the kinetic energy of the
filtered flow:

```
    Kˢ = filter(K) - Kˡ ,   K = ½ uᵢuᵢ ,   Kˡ = ½ ūᵢūᵢ ,   ūᵢ = filter(uᵢ)
```

equivalently `Kˢ = ½ τⁱⁱ` with the sub-filter stress `τⁱʲ = filter(uⁱuʲ) - ūⁱūʲ` (filtering
framework of Aluie et al., 2018, *J. Phys. Oceanogr.*, doi:10.1175/JPO-D-17-0100.1). It is assembled from
the full kinetic energy `K` and [`FilteredKineticEnergy`](@ref) `Kˡ`, which share the same
interpolate-the-square (`½⟨uᵢ²⟩`) discretization, so the discrete decomposition `filter(K) = Kˡ + Kˢ` holds
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
└── computes: sub-filter kinetic energy  Kˢ = ½τⁱⁱ
```

A convenience method `SubFilterKineticEnergy(model; σ, dims, boundary, N)` builds the Gaussian `filter`
for you from a standard deviation `σ` (with `σ = ℓ / (2√(2 ln 2))` for a FWHM `ℓ`).
"""
function SubFilterKineticEnergy(model, filter)
    u, v, w = model.velocities
    k  = KineticEnergy(model, u, v, w)           # full kinetic energy ½uᵢuᵢ (kinetic_energy_ccc)
    k̄ = Field(filter(Field(k)))                  # filter(K) materialized so the filter takes its staged path
    ū, v̄, w̄ = filtered_velocities(filter, (1, 2, 3), u, v, w) # ūᵢ = filter(uᵢ), materialized as `Field`s
    return KernelFunctionOperation{Center, Center, Center}(subfilter_kinetic_energy_ccc, model.grid, k̄, ū, v̄, w̄)
end

SubFilterKineticEnergy(model; σ, dims = (1, 2, 3), boundary = :shrink, N = nothing) =
    SubFilterKineticEnergy(model, GaussianFilter(; dims, σ, boundary, N))
#---

#+++ Sub-filter kinetic energy dissipation
# Exposed as a single `KernelFunctionOperation` using the same wrapper trick as `KineticEnergyCrossScaleFlux`:
# the kernel just indexes the pre-assembled operation εˢ = filter(ε) - εˡ, whose leaves are the materialized
# filtered `Field`s, so per-cell evaluation only reads those fields and subtracts (it never re-filters).
@inline subfilter_ke_dissipation_rate_ccc(i, j, k, grid, εˢ) = @inbounds εˢ[i, j, k]

const SubFilterKineticEnergyDissipationRate = CustomKFO{<:typeof(subfilter_ke_dissipation_rate_ccc)}
const DissipationRate = SubFilterKineticEnergyDissipationRate

"""
    $(SIGNATURES)

Return the sub-filter-scale (SFS) kinetic-energy dissipation rate `εˢ`, the viscous dissipation carried
by the scales that a low-pass `filter` removes:

```
    εˢ = filter(ε) - εˡ
```

where `ε` is the dissipation rate of the full flow
([`KineticEnergyDissipationRate`](@ref Oceanostics.KineticEnergyEquation.DissipationRate)) and `εˡ` is the
dissipation rate of the filtered flow ([`FilteredKineticEnergyDissipationRate`](@ref)). It is the viscous
sink in the budget of the sub-filter kinetic energy `Kˢ` ([`SubFilterKineticEnergy`](@ref);
filtering framework of Aluie et al., 2018, *J. Phys. Oceanogr.*, doi:10.1175/JPO-D-17-0100.1). For a
constant viscosity it reduces to `2ν[filter(SⁱʲSⁱʲ) - S̄ⁱʲ S̄ⁱʲ] ≥ 0`, a strictly positive sink.

`filter` is any callable mapping a field to its low-pass-filtered counterpart, e.g. a reusable
[`GaussianFilter`](@ref) or [`BoxFilter`](@ref). Following the `KineticEnergyCrossScaleFlux` pattern, the
result is a single `KernelFunctionOperation` whose kernel indexes a pre-assembled operation with
materialized filtered `Field` leaves (the full-flow dissipation `ε` is materialized before it is filtered,
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
└── computes: sub-filter kinetic energy dissipation rate  εˢ = filter(ε) - εˡ
```

A convenience method `SubFilterKineticEnergyDissipationRate(model; σ, dims, boundary, N)` builds the
Gaussian `filter` for you from a standard deviation `σ` (with `σ = ℓ / (2√(2 ln 2))` for a FWHM `ℓ`).
"""
function SubFilterKineticEnergyDissipationRate(model, filter)
    ε  = KineticEnergyDissipationRate(model)                 # dissipation of the full flow
    εˡ = FilteredKineticEnergyDissipationRate(model, filter) # dissipation of the filtered flow
    εˢ = Field(filter(Field(ε))) - εˡ                        # εˢ = filter(ε) - εˡ; leaves are materialized Fields
    return KernelFunctionOperation{Center, Center, Center}(subfilter_ke_dissipation_rate_ccc, model.grid, εˢ)
end

SubFilterKineticEnergyDissipationRate(model; σ, dims = (1, 2, 3), boundary = :shrink, N = nothing) =
    SubFilterKineticEnergyDissipationRate(model, GaussianFilter(; dims, σ, boundary, N))
#---

end # module
