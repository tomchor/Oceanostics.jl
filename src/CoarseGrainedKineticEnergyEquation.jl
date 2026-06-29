module CoarseGrainedKineticEnergyEquation

using DocStringExtensions

export subfilter_stress_tensor, KineticEnergyCrossScaleFlux, CrossScaleFlux
export CoarseGrainedKineticEnergyDissipationRate, DissipationRate

using Oceananigans.Grids: Center, Face
using Oceananigans.Fields: Field
using Oceananigans.AbstractOperations: @at, KernelFunctionOperation

using Oceanostics: CustomKFO, perturbation_fields
using ..FlowDiagnostics: StressTensor, StrainRateTensor
import ..FlowDiagnostics            # for the (unexported) `validate_dims`
using ..KineticEnergyEquation: viscous_dissipation_rate_ccc   # reused by the coarse-grained dissipation
using ..SpatialFilters: GaussianFilter, BoxFilter   # BoxFilter is imported so its docstring `@ref` resolves in-module

#+++ Shared helpers
# Filter only the velocities that the requested `dims` actually use: component τᵢⱼ / S̄ᵢⱼ needs uᵢ and
# uⱼ, so velocity component d is filtered iff d ∈ dims; the others are passed through untouched (they
# never enter the kept tensor components). Each filtered velocity is materialized as a `Field` so the
# separable filter takes its fast staged path and is computed once and reused by both tensors.
function filtered_velocities(filter, dims, u, v, w)
    ū = (1 in dims) ? Field(filter(u)) : u
    v̄ = (2 in dims) ? Field(filter(v)) : v
    w̄ = (3 in dims) ? Field(filter(w)) : w
    return ū, v̄, w̄
end

# Low-level method of `subfilter_stress_tensor` taking already-filtered velocities `ū, v̄, w̄`, so the
# (expensive) velocity filtering can be done once and shared — `KineticEnergyCrossScaleFlux` reuses
# `ū, v̄, w̄` for both the strain rate and this stress. Computes τⁱʲ = filter(uⁱuʲ) - ūⁱūʲ component by
# component, reusing `StressTensor` to build the momentum-flux tensor uⁱuʲ of both the full and the
# filtered velocity; `filter` is applied to the (materialized) full flux and the filtered flux is
# subtracted. The result is a `NamedTuple` with the same keys/locations as `StressTensor`.
function subfilter_stress_tensor(filter, grid, u, v, w, ū, v̄, w̄; dims, collocate_diagonals=false)
    flux_full = StressTensor(grid, u, v, w; dims, collocate_diagonals)   # uⁱuʲ
    flux_filt = StressTensor(grid, ū, v̄, w̄; dims, collocate_diagonals)   # ūⁱūʲ
    subfilter(full, coarse) = Field(filter(Field(full))) - coarse        # filter(uⁱuʲ) - ūⁱūʲ
    ks = keys(flux_full)
    return NamedTuple{ks}(map(subfilter, values(flux_full), values(flux_filt)))
end
#---

#+++ Subfilter (sub-grid) stress tensor
"""
    $(SIGNATURES)

Return the components of the subfilter-scale (SFS) stress tensor `τ`, the residual momentum flux that
a low-pass `filter` removes from the filtered scales:

```
    τⁱʲ = filter(uⁱuʲ) - ūⁱ ūʲ ,   ūⁱ = filter(uⁱ)
```

(also called the sub-grid stress in LES, or the generalized central moment in the coarse-graining
framework of Aluie et al., 2018, *J. Phys. Oceanogr.*, doi:10.1175/JPO-D-17-0100.1). It is the
quantity contracted with the filtered strain rate to form the cross-scale kinetic-energy flux — see
[`KineticEnergyCrossScaleFlux`](@ref).

`filter` is any callable that maps a field to its low-pass-filtered counterpart, e.g. a reusable
[`GaussianFilter`](@ref) or [`BoxFilter`](@ref):

```jldoctest
using Oceananigans, Oceanostics

grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1), topology=(Periodic, Periodic, Bounded))
model = NonhydrostaticModel(grid)

filter = GaussianFilter(; dims=(1, 2, 3), σ=0.1)
τ = subfilter_stress_tensor(model, filter)

keys(τ)

# output

(:τ₁₁, :τ₂₂, :τ₃₃, :τ₁₂, :τ₁₃, :τ₂₃)
```

The result is a `NamedTuple` with the independent components, each living at the same staggered
location as the corresponding [`StressTensor`](@ref) component; `collocate_diagonals` has the same
meaning as there and is forwarded to it (use `collocate_diagonals = true` to put the diagonals at
`ccc`, e.g. to form the subfilter kinetic energy `½(τ₁₁ + τ₂₂ + τ₃₃)`). The filtered velocities `ūⁱ`
and the filtered momentum fluxes `filter(uⁱuʲ)` are materialized as `Field`s internally (the filter's
fast staged path only fires when wrapped directly in a `Field`), so each returned component is a lazy
operation over those computed fields and recomputes correctly when written by an `OutputWriter`.

`dims` selects which spatial directions enter the tensor, exactly as in [`StressTensor`](@ref):
component `τⁱʲ` is kept only when both `i` and `j` are in `dims`, and only the velocities used by the
kept components are filtered. The default `dims = (1, 2, 3)` returns the full tensor; `dims = (1, 3)`
returns the `x`–`z` subset (`τ₁₁`, `τ₃₃`, `τ₁₃`).

A convenience method `subfilter_stress_tensor(model; σ, dims, boundary, N, collocate_diagonals)` builds
the Gaussian `filter` for you from a standard deviation `σ` (a Gaussian of full width at half maximum
`ℓ` has `σ = ℓ / (2√(2 ln 2))`).
"""
function subfilter_stress_tensor(model, filter; dims = (1, 2, 3), collocate_diagonals = false)
    FlowDiagnostics.validate_dims(dims)
    grid = model.grid
    u, v, w = model.velocities
    ū, v̄, w̄ = filtered_velocities(filter, dims, u, v, w)
    return subfilter_stress_tensor(filter, grid, u, v, w, ū, v̄, w̄; dims, collocate_diagonals)
end

subfilter_stress_tensor(model; σ, dims = (1, 2, 3), boundary = :shrink, N = nothing, collocate_diagonals = false) =
    subfilter_stress_tensor(model, GaussianFilter(; dims, σ, boundary, N); dims, collocate_diagonals)
#---

#+++ Cross-scale kinetic-energy flux
# Πₖ = -τⁱʲ S̄ⁱʲ contracted at cell centers. τ and S̄ share keys/ordering (both built with the same
# `dims`), so we pair them component-by-component and weight the off-diagonals by 2 (tensor symmetry).
# Each component is interpolated to (Center, Center, Center) before multiplying, matching the offline
# postprocessing convention.
to_center(ψ) = @at (Center, Center, Center) ψ

const _CONTRACTION = ((:τ₁₁, :S₁₁, 1), (:τ₂₂, :S₂₂, 1), (:τ₃₃, :S₃₃, 1),
                      (:τ₁₂, :S₁₂, 2), (:τ₁₃, :S₁₃, 2), (:τ₂₃, :S₂₃, 2))

function _cross_scale_ke_flux(τ, S̄)
    terms = (weight * to_center(τ[kτ]) * to_center(S̄[kS]) for (kτ, kS, weight) in _CONTRACTION if haskey(τ, kτ))
    return -sum(terms)
end

# Expose the flux as a single `KernelFunctionOperation` so it displays like the other diagnostics (via
# `@diagnostic_show` in `Oceanostics`) and composes inside larger operation trees. The kernel just
# evaluates the contraction operation `Πᵏ` built above; `Πᵏ`'s leaves are the materialized filtered
# `Field`s, so this per-cell evaluation only reads those fields and does arithmetic — it never re-filters.
@inline cross_scale_ke_flux_ccc(i, j, k, grid, Πᵏ) = @inbounds Πᵏ[i, j, k]

const KineticEnergyCrossScaleFlux = CustomKFO{<:typeof(cross_scale_ke_flux_ccc)}
const CrossScaleFlux = KineticEnergyCrossScaleFlux

"""
    $(SIGNATURES)

Return the cross-scale (scale-to-scale) kinetic-energy flux `Πₖ`, the rate at which a low-pass
`filter` transfers kinetic energy from the filtered to the subfilter scales (Aluie et al., 2018,
*J. Phys. Oceanogr.*, doi:10.1175/JPO-D-17-0100.1):

```
    Πₖ = -τⁱʲ S̄ⁱʲ
```

where `τⁱʲ = filter(uⁱuʲ) - ūⁱūʲ` is the subfilter-scale stress tensor ([`subfilter_stress_tensor`](@ref))
and `S̄ⁱʲ = ½(∂ūⁱ/∂xʲ + ∂ūʲ/∂xⁱ)` is the strain rate tensor of the filtered velocity
([`StrainRateTensor`](@ref) applied to `ūⁱ`). The contraction is evaluated at `(Center, Center,
Center)`, with off-diagonal components counted twice by symmetry. `Πₖ > 0` is forward (downscale,
filtered → subfilter) transfer. The result is per unit mass (units `m² s⁻³`); multiply by a reference
density `ρ₀` for a volumetric power.

`filter` is any callable mapping a field to its filtered counterpart, e.g. a reusable
[`GaussianFilter`](@ref):

```jldoctest
using Oceananigans, Oceanostics

grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1), topology=(Periodic, Periodic, Bounded))
model = NonhydrostaticModel(grid)

ℓ = 0.2  # filter scale (full width at half maximum)
filter = GaussianFilter(; dims=(1, 2, 3), σ=ℓ / (2√(2log(2))))

KineticEnergyCrossScaleFlux(model, filter)

# output

KineticEnergyCrossScaleFlux KernelFunctionOperation at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: cross_scale_ke_flux_ccc (generic function with 1 method)
└── arguments: ("Oceananigans.AbstractOperations.UnaryOperation",)
└── computes: cross-scale kinetic energy flux  Πₖ = -τⁱʲS̄ⁱʲ
```

The returned object is a lazy operation over internally materialized filtered `Field`s, so it is
ready for `Field`, `Integral`, and `OutputWriter`s and recomputes as the simulation evolves.

`dims` selects which directions enter the tensors (only their `i,j` components are summed, and only
the velocities they use are filtered): the default `dims = (1, 2, 3)` gives the full 3D flux, while
`dims = (1, 3)` gives the 2D `x`–`z` flux `Πₖ = -(τ₁₁S̄₁₁ + τ₃₃S̄₃₃ + 2τ₁₃S̄₁₃)`. The filter's own
directions are set independently inside `filter`, so you can filter horizontally yet contract the
full tensor.

A convenience method `KineticEnergyCrossScaleFlux(model; σ, dims, boundary, N)` builds the Gaussian
`filter` for you from a standard deviation `σ` (with `σ = ℓ / (2√(2 ln 2))` for a FWHM `ℓ`).
"""
function KineticEnergyCrossScaleFlux(model, filter; dims = (1, 2, 3))
    FlowDiagnostics.validate_dims(dims)
    grid = model.grid
    u, v, w = model.velocities
    ū, v̄, w̄ = filtered_velocities(filter, dims, u, v, w)

    # Strain S̄ⁱʲ of the filtered velocities, and the subfilter stress τⁱʲ. The
    # contraction interpolates every component to cell centers, so the components can stay at their
    # natural staggered locations here; the result is wrapped in a `KernelFunctionOperation`.
    S̄ = StrainRateTensor(grid, ū, v̄, w̄; dims)
    τ = subfilter_stress_tensor(filter, grid, u, v, w, ū, v̄, w̄; dims)
    return KernelFunctionOperation{Center, Center, Center}(cross_scale_ke_flux_ccc, grid, _cross_scale_ke_flux(τ, S̄))
end

KineticEnergyCrossScaleFlux(model; σ, dims = (1, 2, 3), boundary = :shrink, N = nothing) =
    KineticEnergyCrossScaleFlux(model, GaussianFilter(; dims, σ, boundary, N); dims)
#---

#+++ Coarse-grained kinetic-energy dissipation
# A distinct kernel — delegating to `KineticEnergyEquation`'s `viscous_dissipation_rate_ccc` — so this
# diagnostic gets its own type alias and display while reusing the exact ∂ⱼuᵢ·Fᵢⱼ viscous contraction.
@inline coarse_grained_dissipation_rate_ccc(i, j, k, grid, closure_fields, filtered_model_fields, p) =
    viscous_dissipation_rate_ccc(i, j, k, grid, closure_fields, filtered_model_fields, p)

const CoarseGrainedKineticEnergyDissipationRate = CustomKFO{<:typeof(coarse_grained_dissipation_rate_ccc)}
const DissipationRate = CoarseGrainedKineticEnergyDissipationRate

"""
    $(SIGNATURES)

Return the coarse-grained (filtered-flow) kinetic-energy dissipation rate `ε̄`, the rate at which
viscosity removes kinetic energy from the *filtered* velocity field `ūᵢ = filter(uᵢ)`:

```
    ε̄ = ∂ⱼūᵢ · Fᵢⱼ(ū)
```

where `Fᵢⱼ` is the viscous stress (flux) tensor supplied by the model's closure. This is exactly the
[`KineticEnergyDissipationRate`](@ref Oceanostics.KineticEnergyEquation.DissipationRate) `ε = ∂ⱼuᵢ·Fᵢⱼ` — the same viscous contraction and the same closure
machinery — evaluated on the filtered velocities instead of the full ones, so it is the viscous sink in
the budget of the filtered kinetic energy `K̄ = ½ūᵢūᵢ` (coarse-graining framework of Aluie et al., 2018,
*J. Phys. Oceanogr.*, doi:10.1175/JPO-D-17-0100.1). For a constant-viscosity `ScalarDiffusivity` it
reduces to `2ν S̄ᵢⱼS̄ᵢⱼ`, the dissipation of the resolved strain. It is evaluated at `(Center, Center,
Center)`, per unit mass (units `m² s⁻³`); multiply by a reference density `ρ₀` for a volumetric power.

`filter` is any callable mapping a field to its filtered counterpart, e.g. a reusable
[`GaussianFilter`](@ref):

```jldoctest
using Oceananigans, Oceanostics

grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1), topology=(Periodic, Periodic, Bounded))
model = NonhydrostaticModel(grid; closure=ScalarDiffusivity(ν=1e-4))

ℓ = 0.2  # filter scale (full width at half maximum)
filter = GaussianFilter(; dims=(1, 2, 3), σ=ℓ / (2√(2log(2))))

CoarseGrainedKineticEnergyDissipationRate(model, filter)

# output

CoarseGrainedKineticEnergyDissipationRate KernelFunctionOperation at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: coarse_grained_dissipation_rate_ccc (generic function with 1 method)
└── arguments: ("Nothing", "NamedTuple", "NamedTuple")
└── computes: coarse-grained kinetic energy dissipation rate  ε̄ = ∂ⱼūᵢ·Fᵢⱼ
```

The viscosity is taken from `model.closure`/`model.closure_fields`, exactly as in
[`KineticEnergyDissipationRate`](@ref Oceanostics.KineticEnergyEquation.DissipationRate), so the model needs a closure whose viscous fluxes are defined,
just as that diagnostic does. The filtered velocities are materialized as `Field`s internally (and
refreshed on recompute), so the returned object is a lazy operation ready for `Field`, `Integral`, and
`OutputWriter`s and recomputes as the simulation evolves.

Unlike the cross-scale flux and the stress tensor, this diagnostic takes no `dims` argument: it always
forms the full viscous contraction (matching [`KineticEnergyDissipationRate`](@ref Oceanostics.KineticEnergyEquation.DissipationRate)). The directions the
filter acts in are set inside `filter`.

A convenience method `CoarseGrainedKineticEnergyDissipationRate(model; σ, dims, boundary, N)` builds the
Gaussian `filter` for you from a standard deviation `σ` (with `σ = ℓ / (2√(2 ln 2))` for a FWHM `ℓ`);
`dims` here selects the directions the Gaussian filter acts in.
"""
function CoarseGrainedKineticEnergyDissipationRate(model, filter)
    grid = model.grid
    u, v, w = model.velocities

    # Filter every velocity component (the dissipation contracts all of ∂ⱼūᵢ). Materializing as `Field`s
    # fires the separable filter's fast staged path and matches the field types the flux kernels read.
    ū = Field(filter(u))
    v̄ = Field(filter(v))
    w̄ = Field(filter(w))

    # The field set the viscous fluxes need (velocities + tracers + auxiliary fields), but with the
    # resolved velocities swapped for their filtered counterparts, so `viscous_dissipation_rate_ccc`
    # evaluates ∂ⱼūᵢ·Fᵢⱼ(ū): the dissipation of the filtered flow.
    filtered_model_fields = merge(perturbation_fields(model), (; u = ū, v = v̄, w = w̄))
    parameters = (; model.closure, model.clock, model.buoyancy)
    return KernelFunctionOperation{Center, Center, Center}(coarse_grained_dissipation_rate_ccc, grid,
                                                           model.closure_fields, filtered_model_fields, parameters)
end

CoarseGrainedKineticEnergyDissipationRate(model; σ, dims = (1, 2, 3), boundary = :shrink, N = nothing) =
    CoarseGrainedKineticEnergyDissipationRate(model, GaussianFilter(; dims, σ, boundary, N))
#---

end # module
