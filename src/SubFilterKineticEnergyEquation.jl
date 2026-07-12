module SubFilterKineticEnergyEquation

using DocStringExtensions

export subfilter_kinetic_energy, subfilter_kinetic_energy_dissipation_rate

using Oceananigans.Fields: Field

using ..KineticEnergyEquation: KineticEnergyDissipationRate
using ..FilteredKineticEnergyEquation: subfilter_stress_tensor, CoarseGrainedKineticEnergyDissipationRate
using ..SpatialFilters: GaussianFilter, BoxFilter   # GaussianFilter is used by the convenience methods; BoxFilter is imported only so its docstring `@ref` resolves in-module

#+++ Sub-filter kinetic energy
"""
    $(SIGNATURES)

Return a lazy `AbstractOperation` for the sub-filter-scale (SFS) kinetic energy `Kˢ`, the kinetic
energy carried by the scales that a low-pass `filter` removes from the flow:

```
    Kˢ = ½ τⁱⁱ = ½ (τ₁₁ + τ₂₂ + τ₃₃) ,   τⁱʲ = filter(uⁱuʲ) - ūⁱ ūʲ ,   ūⁱ = filter(uⁱ)
```

where `τⁱʲ` is the sub-filter stress tensor ([`subfilter_stress_tensor`](@ref)). This is the sub-filter
counterpart of the filtered (coarse-grained) kinetic energy `Kˡ = ½ ūⁱ ūⁱ` (coarse-graining framework of
Aluie et al., 2018, *J. Phys. Oceanogr.*, doi:10.1175/JPO-D-17-0100.1).

`filter` is any callable mapping a field to its low-pass-filtered counterpart, e.g. a reusable
[`GaussianFilter`](@ref) or [`BoxFilter`](@ref). The diagonal stress components are formed at cell
centers (`collocate_diagonals = true`), so the result lives at `(Center, Center, Center)` and is per unit
mass (units `m² s⁻²`):

```jldoctest
using Oceananigans, Oceanostics
using Oceananigans.Fields: location

grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1), topology=(Periodic, Periodic, Bounded))
model = NonhydrostaticModel(grid)

filter = GaussianFilter(; dims=(1, 2, 3), σ=0.1)
Kˢ = subfilter_kinetic_energy(model, filter)

location(Kˢ)

# output

(Center, Center, Center)
```

`dims` selects which directions enter the stress tensor, exactly as in [`subfilter_stress_tensor`](@ref):
`Kˢ` sums the diagonal components the choice keeps (the default `dims = (1, 2, 3)` uses all three, while
`dims = (1, 3)` gives `½(τ₁₁ + τ₃₃)`). A convenience method
`subfilter_kinetic_energy(model; σ, dims, boundary, N)` builds the Gaussian `filter` for you from a
standard deviation `σ` (with `σ = ℓ / (2√(2 ln 2))` for a FWHM `ℓ`).
"""
function subfilter_kinetic_energy(model, filter; dims = (1, 2, 3))
    τ = subfilter_stress_tensor(model, filter; dims, collocate_diagonals = true)
    diagonals = (τ[k] for k in (:τ₁₁, :τ₂₂, :τ₃₃) if haskey(τ, k))   # keep only the diagonals `dims` retains
    return sum(diagonals) / 2
end

subfilter_kinetic_energy(model; σ, dims = (1, 2, 3), boundary = :shrink, N = nothing) =
    subfilter_kinetic_energy(model, GaussianFilter(; dims, σ, boundary, N); dims)
#---

#+++ Sub-filter kinetic energy dissipation
"""
    $(SIGNATURES)

Return a lazy `AbstractOperation` for the sub-filter-scale (SFS) kinetic-energy dissipation rate `εˢ`,
the viscous dissipation carried by the scales that a low-pass `filter` removes:

```
    εˢ = filter(ε) - εˡ
```

where `ε` is the dissipation rate of the full flow
([`KineticEnergyDissipationRate`](@ref Oceanostics.KineticEnergyEquation.DissipationRate)) and `εˡ` is
the dissipation rate of the filtered flow ([`CoarseGrainedKineticEnergyDissipationRate`](@ref)). It is
the viscous sink in the budget of the sub-filter kinetic energy `Kˢ` ([`subfilter_kinetic_energy`](@ref);
coarse-graining framework of Aluie et al., 2018, *J. Phys. Oceanogr.*, doi:10.1175/JPO-D-17-0100.1). For
a constant viscosity it reduces to `2ν[filter(SⁱʲSⁱʲ) - S̄ⁱʲ S̄ⁱʲ] ≥ 0`, a strictly positive sink.

`filter` is any callable mapping a field to its low-pass-filtered counterpart, e.g. a reusable
[`GaussianFilter`](@ref). The full-flow dissipation `ε` is materialized as a `Field` before it is
filtered (so it is not recomputed at every filter tap), and the filtered result is itself wrapped in a
`Field` so the separable filter takes its fast staged path; the result lives at `(Center, Center,
Center)`, per unit mass (units `m² s⁻³`). The model needs a closure whose viscous fluxes are defined,
exactly as [`CoarseGrainedKineticEnergyDissipationRate`](@ref) requires:

```jldoctest
using Oceananigans, Oceanostics
using Oceananigans.Fields: location

grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1), topology=(Periodic, Periodic, Bounded))
model = NonhydrostaticModel(grid; closure=ScalarDiffusivity(ν=1e-4))

filter = GaussianFilter(; dims=(1, 2, 3), σ=0.1)
εˢ = subfilter_kinetic_energy_dissipation_rate(model, filter)

location(εˢ)

# output

(Center, Center, Center)
```

A convenience method `subfilter_kinetic_energy_dissipation_rate(model; σ, dims, boundary, N)` builds the
Gaussian `filter` for you from a standard deviation `σ` (with `σ = ℓ / (2√(2 ln 2))` for a FWHM `ℓ`).
"""
function subfilter_kinetic_energy_dissipation_rate(model, filter)
    ε  = KineticEnergyDissipationRate(model)                      # dissipation of the full flow
    εˡ = CoarseGrainedKineticEnergyDissipationRate(model, filter) # dissipation of the filtered flow
    return Field(filter(Field(ε))) - εˡ   # εˢ = filter(ε) - εˡ; ε materialized so it is filtered via the fast staged path
end

subfilter_kinetic_energy_dissipation_rate(model; σ, dims = (1, 2, 3), boundary = :shrink, N = nothing) =
    subfilter_kinetic_energy_dissipation_rate(model, GaussianFilter(; dims, σ, boundary, N))
#---

end # module
