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

using ..KineticEnergyEquation: KineticEnergyDissipationRate
using ..FilteredKineticEnergyEquation: subfilter_stress_tensor, FilteredKineticEnergyDissipationRate, KineticEnergyCrossScaleFlux
using ..FlowDiagnostics: validate_dims   # to validate `dims` before the per-direction diagonal loop
# `GaussianFilter` is used by the convenience methods; `BoxFilter` is imported only so its docstring
# `@ref` resolves in-module.
using ..SpatialFilters: GaussianFilter, BoxFilter

#+++ Sub-filter kinetic energy
# Exposed as a single `KernelFunctionOperation` using the same wrapper trick as `KineticEnergyCrossScaleFlux`:
# the kernel indexes the pre-assembled operation Kˢ = ½τⁱⁱ, whose leaves are the materialized filtered
# `Field`s of `subfilter_stress_tensor`, so per-cell evaluation only reads those fields and sums.
@inline subfilter_kinetic_energy_ccc(i, j, k, grid, Kˢ) = @inbounds Kˢ[i, j, k]

const SubFilterKineticEnergy = CustomKFO{<:typeof(subfilter_kinetic_energy_ccc)}

"""
    $(SIGNATURES)

Return the sub-filter-scale (SFS) kinetic energy `Kˢ`, the kinetic energy carried by the scales that a
low-pass `filter` removes from the flow:

```
    Kˢ = ½ τⁱⁱ = ½ (τ₁₁ + τ₂₂ + τ₃₃) ,   τⁱʲ = filter(uⁱuʲ) - ūⁱ ūʲ ,   ūⁱ = filter(uⁱ)
```

where `τⁱʲ` is the sub-filter stress tensor ([`subfilter_stress_tensor`](@ref)). This is the sub-filter
counterpart of the filtered (coarse-grained) kinetic energy `Kˡ = ½ ūⁱ ūⁱ` (coarse-graining framework of
Aluie et al., 2018, *J. Phys. Oceanogr.*, doi:10.1175/JPO-D-17-0100.1).

`filter` is any callable mapping a field to its low-pass-filtered counterpart, e.g. a reusable
[`GaussianFilter`](@ref) or [`BoxFilter`](@ref). Following the `KineticEnergyCrossScaleFlux` pattern, the
result is a single `KernelFunctionOperation` whose kernel indexes a pre-assembled operation with
materialized filtered `Field` leaves (the diagonal stress components are formed at cell centers with
`collocate_diagonals = true`). It lives at `(Center, Center, Center)`, per unit mass (units `m² s⁻²`):

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
└── arguments: ("Oceananigans.AbstractOperations.BinaryOperation",)
└── computes: sub-filter kinetic energy  Kˢ = ½τⁱⁱ
```

`dims` selects which directions enter the stress tensor, exactly as in [`subfilter_stress_tensor`](@ref):
`Kˢ` sums the diagonal components the choice keeps (the default `dims = (1, 2, 3)` uses all three, while
`dims = (1, 3)` gives `½(τ₁₁ + τ₃₃)`). A convenience method
`SubFilterKineticEnergy(model; σ, dims, boundary, N)` builds the Gaussian `filter` for you from a
standard deviation `σ` (with `σ = ℓ / (2√(2 ln 2))` for a FWHM `ℓ`).
"""
function SubFilterKineticEnergy(model, filter; dims = (1, 2, 3))
    validate_dims(dims)
    # Build only the diagonal sub-filter stresses τᵈᵈ, one call per direction, so no off-diagonal fluxes
    # are ever materialized; Kˢ = ½ Σ_d τᵈᵈ.
    diagonals = (only(subfilter_stress_tensor(model, filter; dims=(d,), collocate_diagonals = true)) for d in dims)
    Kˢ = sum(diagonals) / 2
    return KernelFunctionOperation{Center, Center, Center}(subfilter_kinetic_energy_ccc, model.grid, Kˢ)
end

SubFilterKineticEnergy(model; σ, dims = (1, 2, 3), boundary = :shrink, N = nothing) =
    SubFilterKineticEnergy(model, GaussianFilter(; dims, σ, boundary, N); dims)
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
coarse-graining framework of Aluie et al., 2018, *J. Phys. Oceanogr.*, doi:10.1175/JPO-D-17-0100.1). For a
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
