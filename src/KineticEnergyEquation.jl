module KineticEnergyEquation
using DocStringExtensions

export KineticEnergy
export KineticEnergyTendency
export Advection, KineticEnergyAdvection
export Stress, KineticEnergyStress
export Forcing, KineticEnergyForcing
export PressureRedistribution, KineticEnergyPressureRedistribution
export PotentialEnergyConversion, PotentialToKineticEnergyConversion
export DissipationRate, KineticEnergyDissipationRate
export KineticEnergyIsotropicDissipationRate

using Oceananigans: NonhydrostaticModel, HydrostaticFreeSurfaceModel, fields
using Oceananigans.Operators
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Grids: Center
using Oceananigans.Fields: Field, ZeroField
using Oceananigans.Models.NonhydrostaticModels: u_velocity_tendency, v_velocity_tendency, w_velocity_tendency
using Oceananigans.Models.HydrostaticFreeSurfaceModels: hydrostatic_free_surface_u_velocity_tendency,
                                                        hydrostatic_free_surface_v_velocity_tendency
using Oceananigans.Advection: div_𝐯u, div_𝐯v, div_𝐯w, U_dot_∇u, U_dot_∇v
using Oceananigans.TurbulenceClosures: viscous_flux_ux, viscous_flux_uy, viscous_flux_uz,
                                       viscous_flux_vx, viscous_flux_vy, viscous_flux_vz,
                                       viscous_flux_wx, viscous_flux_wy, viscous_flux_wz,
                                       ∂ⱼ_τ₁ⱼ, ∂ⱼ_τ₂ⱼ, ∂ⱼ_τ₃ⱼ
using Oceananigans.BuoyancyFormulations: x_dot_g_bᶠᶜᶜ, y_dot_g_bᶜᶠᶜ, z_dot_g_bᶜᶜᶠ

using Oceanostics: _νᶜᶜᶜ
using Oceanostics: validate_location, validate_dissipative_closure, validate_closure_supplies_a_flux,
                   perturbation_fields, CustomKFO, momentum_advection

# Some useful operators
@inline ψ²(i, j, k, grid, ψ) = @inbounds ψ[i, j, k]^2
@inline fψ_plus_gφ²(i, j, k, grid, f, ψ, g, φ) = (f(i, j, k, grid, ψ) + g(i, j, k, grid, φ))^2

# `uᵢ` times the term `f` of the `uᵢ` equation. A row whose velocity is a `ZeroField` is skipped without
# evaluating `f`, which is how the `w` row is dropped on a hydrostatic model (see `prognostic_velocities`).
@inline ψf(i, j, k, grid, ψ, f, args...) = @inbounds ψ[i, j, k] * f(i, j, k, grid, args...)
@inline ψf(i, j, k, grid, ::ZeroField, f, args...) = zero(grid)

const SupportedModel = Union{NonhydrostaticModel, HydrostaticFreeSurfaceModel}

# The velocity components whose momentum equations are contracted with them to form the kinetic energy
# budget. A HydrostaticFreeSurfaceModel diagnoses w from continuity and has no w-momentum equation, so
# w is replaced by a ZeroField there; the momentum fluxes are still computed with the diagnostic w, as
# in the model's own u and v tendencies.
prognostic_velocities(model, velocities) = (u = velocities.u, v = velocities.v, w = velocities.w)
prognostic_velocities(::HydrostaticFreeSurfaceModel, velocities) = (u = velocities.u, v = velocities.v, w = ZeroField())

# The same choice as a type, for a kernel that takes the velocities themselves from `fields(model)`:
# `Val(true)` keeps the `w` row and `Val(false)` drops it
prognostic_w_row(model) = Val(true)
prognostic_w_row(::HydrostaticFreeSurfaceModel) = Val(false)

# The forcing of each momentum equation. A hydrostatic model has no `w` equation and so no `forcing.w`;
# the zero forcing put in its place keeps the `w` row of `uᵢFᵤᵢ` at zero.
momentum_forcing(model) = model.forcing
momentum_forcing(model::HydrostaticFreeSurfaceModel) =
    (u = model.forcing.u, v = model.forcing.v, w = Returns(zero(eltype(model.grid))))

#+++ KineticEnergy
@inline kinetic_energy_ccc(i, j, k, grid, u, v, w) = (ℑxᶜᵃᵃ(i, j, k, grid, ψ², u) +
                                                      ℑyᵃᶜᵃ(i, j, k, grid, ψ², v) +
                                                      ℑzᵃᵃᶜ(i, j, k, grid, ψ², w)) / 2

const KineticEnergy = CustomKFO{<:typeof(kinetic_energy_ccc)}

function KineticEnergy(model, u, v, w; location = (Center, Center, Center))
    validate_location(location, "KineticEnergy")
    return KernelFunctionOperation{Center, Center, Center}(kinetic_energy_ccc, model.grid, u, v, w)
end

"""
    $(SIGNATURES)

Calculate the kinetic energy per unit mass `eₖ = ½uᵢuᵢ` of `model`. On a `HydrostaticFreeSurfaceModel`,
which steps `u` and `v` and diagnoses `w` from continuity, this is `½(u² + v²)`.

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(grid);

julia> KE = KineticEnergyEquation.KineticEnergy(model)
KineticEnergy KernelFunctionOperation at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: kinetic_energy_ccc (generic function with 1 method)
└── arguments: ("Field", "Field", "Field")
└── computes: kinetic energy  ½uᵢuᵢ
```
"""
KineticEnergy(model; kwargs...) = KineticEnergy(model, prognostic_velocities(model, model.velocities)...; kwargs...)
#---

#+++ KineticEnergyTendency
@inline function uᵢGᵢᶜᶜᶜ(i, j, k, grid, advection,
                                        coriolis,
                                        stokes_drift,
                                        closure,
                                        u_immersed_bc,
                                        v_immersed_bc,
                                        w_immersed_bc,
                                        buoyancy,
                                        background_fields,
                                        velocities,
                                        tracers,
                                        auxiliary_fields,
                                        closure_fields,
                                        pHY′,
                                        clock,
                                        forcings)
    common_args = (buoyancy, background_fields, velocities, tracers, auxiliary_fields, closure_fields, pHY′, clock)
    u∂ₜu = ℑxᶜᵃᵃ(i, j, k, grid, ψf, velocities.u, u_velocity_tendency, advection, coriolis, stokes_drift, closure, u_immersed_bc, common_args..., forcings.u)
    v∂ₜv = ℑyᵃᶜᵃ(i, j, k, grid, ψf, velocities.v, v_velocity_tendency, advection, coriolis, stokes_drift, closure, v_immersed_bc, common_args..., forcings.v)
    w∂ₜw = ℑzᵃᵃᶜ(i, j, k, grid, ψf, velocities.w, w_velocity_tendency, advection, coriolis, stokes_drift, closure, w_immersed_bc, common_args..., forcings.w)
    return u∂ₜu + v∂ₜv + w∂ₜw
end

# A hydrostatic model has tendencies for `u` and `v` only, and Oceananigans' kernels for them take a
# different set of arguments
@inline function uₕGₕᶜᶜᶜ(i, j, k, grid, advection,
                                        coriolis,
                                        closure,
                                        u_immersed_bc,
                                        v_immersed_bc,
                                        velocities,
                                        free_surface,
                                        tracers,
                                        buoyancy,
                                        closure_fields,
                                        pHY′,
                                        auxiliary_fields,
                                        vertical_coordinate,
                                        clock,
                                        forcings)
    common_args = (velocities, free_surface, tracers, buoyancy, closure_fields, pHY′, auxiliary_fields,
                   vertical_coordinate, clock)
    u∂ₜu = ℑxᶜᵃᵃ(i, j, k, grid, ψf, velocities.u, hydrostatic_free_surface_u_velocity_tendency,
                 advection, coriolis, closure, u_immersed_bc, common_args..., forcings.u)
    v∂ₜv = ℑyᵃᶜᵃ(i, j, k, grid, ψf, velocities.v, hydrostatic_free_surface_v_velocity_tendency,
                 advection, coriolis, closure, v_immersed_bc, common_args..., forcings.v)
    return u∂ₜu + v∂ₜv
end

const KineticEnergyTendency = Union{CustomKFO{<:typeof(uᵢGᵢᶜᶜᶜ)}, CustomKFO{<:typeof(uₕGₕᶜᶜᶜ)}}

"""
    $(SIGNATURES)

Return a `KernelFunctionOperation` that computes the tendency uᵢGᵢ of the kinetic energy `eₖ`,
excluding the nonhydrostatic pressure contribution:

    ∂ₜeₖ = ½∂ₜuᵢ² = uᵢGᵢ - uᵢ∂ᵢpₙₕₛ

On a `HydrostaticFreeSurfaceModel` the sum runs over `u` and `v` only, and `Gᵢ` is Oceananigans'
hydrostatic tendency, which leaves out the part of the barotropic pressure gradient the free surface
treats implicitly (all of it for an `ImplicitFreeSurface` or a `SplitExplicitFreeSurface`).

On either model `Gᵢ` is the explicit tendency. A closure with a `VerticallyImplicitTimeDiscretization`
(CATKE's default, and an option of `ScalarDiffusivity` and `VerticalScalarDiffusivity`) takes the `∂zu`
and `∂zv` parts of its vertical fluxes out of `Gᵢ` and applies them in the implicit step, so this term,
like [`KineticEnergyStress`](@ref), leaves out the vertical viscous work the model does. The change of
`eₖ` over a time step then differs from `uᵢGᵢ - uᵢ∂ᵢpₙₕₛ` by that work, which [`DissipationRate`](@ref)
still accounts for. On an `ImmersedBoundaryGrid` `Gᵢ` also carries the stress through the immersed
faces, which [`KineticEnergyStress`](@ref) leaves out.

```jldoctest
julia> using Oceananigans

julia> grid = RectilinearGrid(size = (1, 1, 4), extent = (1, 1, 1));

julia> model = NonhydrostaticModel(grid);

julia> using Oceanostics.KineticEnergyEquation: KineticEnergyTendency

julia> ke_tendency = KineticEnergyTendency(model)
KineticEnergyTendency KernelFunctionOperation at (Center, Center, Center)
├── grid: 1×1×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 1×1×3 halo
├── kernel_function: uᵢGᵢᶜᶜᶜ (generic function with 1 method)
└── arguments: ("Centered", "Nothing", "Nothing", "Nothing", "Nothing", "Nothing", "Nothing", "Nothing", "BackgroundFields", "NamedTuple", "NamedTuple", "NamedTuple", "Nothing", "Nothing", "Clock", "NamedTuple")
└── computes: kinetic energy tendency  uᵢGᵢ (excl. nonhydrostatic pressure)
```
"""
function KineticEnergyTendency(model::NonhydrostaticModel; location = (Center, Center, Center))
    validate_location(location, "KineticEnergyTendency")
    dependencies = (advection = momentum_advection(model),
                    model.coriolis,
                    model.stokes_drift,
                    model.closure,
                    u_immersed_bc = model.velocities.u.boundary_conditions.immersed,
                    v_immersed_bc = model.velocities.v.boundary_conditions.immersed,
                    w_immersed_bc = model.velocities.w.boundary_conditions.immersed,
                    model.buoyancy,
                    model.background_fields,
                    model.velocities,
                    model.tracers,
                    model.auxiliary_fields,
                    model.closure_fields,
                    model.pressures.pHY′,
                    model.clock,
                    model.forcing,)
    return KernelFunctionOperation{Center, Center, Center}(uᵢGᵢᶜᶜᶜ, model.grid, dependencies...)
end

function KineticEnergyTendency(model::HydrostaticFreeSurfaceModel; location = (Center, Center, Center))
    validate_location(location, "KineticEnergyTendency")
    dependencies = (advection = momentum_advection(model),
                    model.coriolis,
                    model.closure,
                    u_immersed_bc = model.velocities.u.boundary_conditions.immersed,
                    v_immersed_bc = model.velocities.v.boundary_conditions.immersed,
                    model.velocities,
                    model.free_surface,
                    model.tracers,
                    model.buoyancy,
                    model.closure_fields,
                    model.pressure.pHY′,
                    model.auxiliary_fields,
                    model.vertical_coordinate,
                    model.clock,
                    model.forcing,)
    return KernelFunctionOperation{Center, Center, Center}(uₕGₕᶜᶜᶜ, model.grid, dependencies...)
end
#---

#+++ KineticEnergyAdvection
@inline function uᵢ∂ⱼuⱼuᵢᶜᶜᶜ(i, j, k, grid, velocities, advection)
    u∂ⱼuⱼu = ℑxᶜᵃᵃ(i, j, k, grid, ψf, velocities.u, div_𝐯u, advection, velocities, velocities.u)
    v∂ⱼuⱼv = ℑyᵃᶜᵃ(i, j, k, grid, ψf, velocities.v, div_𝐯v, advection, velocities, velocities.v)
    w∂ⱼuⱼw = ℑzᵃᵃᶜ(i, j, k, grid, ψf, velocities.w, div_𝐯w, advection, velocities, velocities.w)
    return u∂ⱼuⱼu + v∂ⱼuⱼv + w∂ⱼuⱼw
end

# The hydrostatic `u` and `v` equations advect momentum through `U_dot_∇u` and `U_dot_∇v`, which take
# either a flux-form scheme or a `VectorInvariant` one
@inline function uₕU_dot_∇uₕᶜᶜᶜ(i, j, k, grid, velocities, advection)
    uU_dot_∇u = ℑxᶜᵃᵃ(i, j, k, grid, ψf, velocities.u, U_dot_∇u, advection, velocities)
    vU_dot_∇v = ℑyᵃᶜᵃ(i, j, k, grid, ψf, velocities.v, U_dot_∇v, advection, velocities)
    return uU_dot_∇u + vU_dot_∇v
end

const KineticEnergyAdvection = Union{CustomKFO{<:typeof(uᵢ∂ⱼuⱼuᵢᶜᶜᶜ)}, CustomKFO{<:typeof(uₕU_dot_∇uₕᶜᶜᶜ)}}
const Advection = KineticEnergyAdvection

"""
    $(SIGNATURES)

Return a `KernelFunctionOperation` that computes the advection term, defined as

    ADV = uᵢ∂ⱼ(uᵢuⱼ)

On a `HydrostaticFreeSurfaceModel` it is `uₕ⋅(u⃗⋅∇)uₕ` over `u` and `v` only, built on the kernels
(`U_dot_∇u`, `U_dot_∇v`) the hydrostatic momentum equations advect with, so it follows the model's
scheme, `VectorInvariant` included. The hydrostatic metric terms that a flux-form scheme picks up on a
curvilinear grid are left out, as they are by
[`UMomentumEquation.Advection`](@ref Oceanostics.UMomentumEquation.Advection); they vanish on a
`RectilinearGrid` and for a `VectorInvariant` scheme.

By default, the advection will be calculated using the resolved `velocities` and
users can use the keyword `velocities` to modify that behavior:

```jldoctest
julia> using Oceananigans

julia> grid = RectilinearGrid(size = (1, 1, 4), extent = (1,1,1));

julia> model = NonhydrostaticModel(grid);

julia> using Oceanostics.KineticEnergyEquation: KineticEnergyAdvection

julia> ADV = KineticEnergyAdvection(model)
KineticEnergyAdvection KernelFunctionOperation at (Center, Center, Center)
├── grid: 1×1×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 1×1×3 halo
├── kernel_function: uᵢ∂ⱼuⱼuᵢᶜᶜᶜ (generic function with 1 method)
└── arguments: ("NamedTuple", "Centered")
└── computes: kinetic energy advection  uᵢ∂ⱼ(uᵢuⱼ)
```
"""
function KineticEnergyAdvection(model::NonhydrostaticModel; velocities = model.velocities, location = (Center, Center, Center))
    validate_location(location, "KineticEnergyAdvection")
    return KernelFunctionOperation{Center, Center, Center}(uᵢ∂ⱼuⱼuᵢᶜᶜᶜ, model.grid, velocities, momentum_advection(model))
end

function KineticEnergyAdvection(model::HydrostaticFreeSurfaceModel; velocities = model.velocities,
                                location = (Center, Center, Center))
    validate_location(location, "KineticEnergyAdvection")
    return KernelFunctionOperation{Center, Center, Center}(uₕU_dot_∇uₕᶜᶜᶜ, model.grid, velocities, momentum_advection(model))
end
#---

#+++ KineticEnergyStress
# `uᵢ` times `∂ⱼ_τᵢⱼ`, the interior viscous term of the momentum equations. Both tendency kernels also
# subtract `immersed_∂ⱼ_τᵢⱼ`, the stress through the faces of an `ImmersedBoundaryGrid`, which is zero off
# such a grid and under its default no-flux immersed boundary conditions, and nonzero with an immersed
# `ValueBoundaryCondition` or a drag `FluxBoundaryCondition` on `u`, `v` or `w`. No diagnostic here mirrors
# it, so there `KineticEnergyTendency` minus the other terms leaves the wall-stress work as a residual; the
# momentum modules expose it per component as `ImmersedViscousDissipation`.
@inline function uᵢ∂ⱼ_τᵢⱼᶜᶜᶜ(i, j, k, grid, velocities,
                                            closure,
                                            closure_fields,
                                            clock,
                                            model_fields,
                                            buoyancy)

    u∂ⱼ_τ₁ⱼ = ℑxᶜᵃᵃ(i, j, k, grid, ψf, velocities.u, ∂ⱼ_τ₁ⱼ, closure, closure_fields, clock, model_fields, buoyancy)
    v∂ⱼ_τ₂ⱼ = ℑyᵃᶜᵃ(i, j, k, grid, ψf, velocities.v, ∂ⱼ_τ₂ⱼ, closure, closure_fields, clock, model_fields, buoyancy)
    w∂ⱼ_τ₃ⱼ = ℑzᵃᵃᶜ(i, j, k, grid, ψf, velocities.w, ∂ⱼ_τ₃ⱼ, closure, closure_fields, clock, model_fields, buoyancy)

    return u∂ⱼ_τ₁ⱼ+ v∂ⱼ_τ₂ⱼ + w∂ⱼ_τ₃ⱼ
end

const KineticEnergyStress = CustomKFO{<:typeof(uᵢ∂ⱼ_τᵢⱼᶜᶜᶜ)}
const Stress = KineticEnergyStress

"""
    $(SIGNATURES)

Return a `KernelFunctionOperation` that computes the diffusive term of the `eₖ` prognostic equation:

```
    DIFF = uᵢ∂ⱼτᵢⱼ
```

where `uᵢ` are the velocity components and `τᵢⱼ` is the viscous momentum flux (minus the stress
tensor). It enters the kinetic energy budget as `-DIFF`, and splits into a transport divergence plus
the dissipation rate `εₖ` ([`DissipationRate`](@ref)).

A `HydrostaticFreeSurfaceModel` has no `w`-momentum equation, so on that model the sum runs over
`i = 1, 2` only and `DIFF` is the stress term of the budget of `½(u² + v²)`. The fluxes `τ₁ⱼ` and
`τ₂ⱼ` are still computed with the diagnostic `w`, as in the model's own `u` and `v` tendencies.

`∂ⱼτᵢⱼ` is Oceananigans' own explicit viscous term, with the closure's time discretization applied. A
closure with a `VerticallyImplicitTimeDiscretization` (CATKE's default, and an option of
`ScalarDiffusivity` and `VerticalScalarDiffusivity`) takes the `∂zu` and `∂zv` parts of its vertical
fluxes out of that term and applies them in the implicit step, so `DIFF` then leaves out the vertical
viscous work the model does, as [`KineticEnergyTendency`](@ref) does, while [`DissipationRate`](@ref)
keeps the full explicit flux. Under such a closure `∫DIFF dV = ∫εₖ dV` no longer holds, and an
integrated budget should be written with `εₖ` rather than with `DIFF`.

On an `ImmersedBoundaryGrid` the tendency kernels also subtract the stress through the immersed faces,
`immersed_∂ⱼτᵢⱼ`, which `DIFF` does not include. It vanishes under the default no-flux immersed boundary
conditions, but with an immersed `ValueBoundaryCondition` or a drag `FluxBoundaryCondition` on the
velocities the wall-stress work is left out of the budget; the momentum modules expose it per component
as `ImmersedViscousDissipation`.

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(grid; closure=ScalarDiffusivity(ν=1e-4));

julia> DIFF = KineticEnergyEquation.KineticEnergyStress(model)
KineticEnergyStress KernelFunctionOperation at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: uᵢ∂ⱼ_τᵢⱼᶜᶜᶜ (generic function with 1 method)
└── arguments: ("NamedTuple", "ScalarDiffusivity", "Nothing", "Clock", "NamedTuple", "Nothing")
└── computes: kinetic energy stress/diffusion  uᵢ∂ⱼτᵢⱼ
```
"""
function KineticEnergyStress(model; location = (Center, Center, Center))
    validate_location(location, "KineticEnergyStress")
    dependencies = (prognostic_velocities(model, model.velocities),
                    model.closure,
                    model.closure_fields,
                    model.clock,
                    fields(model),
                    model.buoyancy)
    return KernelFunctionOperation{Center, Center, Center}(uᵢ∂ⱼ_τᵢⱼᶜᶜᶜ, model.grid, dependencies...)
end
#---

#+++ KineticEnergyForcing
@inline function uᵢFᵤᵢᶜᶜᶜ(i, j, k, grid, forcings,
                                         clock,
                                         model_fields)

    uFᵘ = ℑxᶜᵃᵃ(i, j, k, grid, ψf, model_fields.u, forcings.u, clock, model_fields)
    vFᵛ = ℑyᵃᶜᵃ(i, j, k, grid, ψf, model_fields.v, forcings.v, clock, model_fields)
    wFʷ = ℑzᵃᵃᶜ(i, j, k, grid, ψf, model_fields.w, forcings.w, clock, model_fields)

    return uFᵘ+ vFᵛ + wFʷ
end

const KineticEnergyForcing = CustomKFO{<:typeof(uᵢFᵤᵢᶜᶜᶜ)}
const Forcing = KineticEnergyForcing

"""
    $(SIGNATURES)

Return a `KernelFunctionOperation` that computes the forcing term of the KE prognostic equation:

```
    FORC = uᵢFᵤᵢ
```

where `uᵢ` are the velocity components and `Fᵤᵢ` is the forcing term(s) in the `uᵢ`
prognostic equation (i.e. the forcing for `uᵢ`). On a `HydrostaticFreeSurfaceModel` the sum runs over
`u` and `v` only, since `w` has no momentum equation, and so no forcing, there.

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(grid);

julia> FORC = KineticEnergyEquation.KineticEnergyForcing(model)
KineticEnergyForcing KernelFunctionOperation at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: uᵢFᵤᵢᶜᶜᶜ (generic function with 1 method)
└── arguments: ("NamedTuple", "Clock", "NamedTuple")
└── computes: kinetic energy forcing  uᵢFᵤᵢ
```
"""
function KineticEnergyForcing(model::SupportedModel; location = (Center, Center, Center))
    validate_location(location, "KineticEnergyForcing")
    dependencies = (momentum_forcing(model),
                    model.clock,
                    fields(model))
    return KernelFunctionOperation{Center, Center, Center}(uᵢFᵤᵢᶜᶜᶜ, model.grid, dependencies...)
end
#---

#+++ KineticEnergyPressureRedistribution
@inline function uᵢ∂ᵢpᶜᶜᶜ(i, j, k, grid, velocities, pressure)
    u∂x_p = ℑxᶜᵃᵃ(i, j, k, grid, ψf, velocities.u, ∂xᶠᶜᶜ, pressure)
    v∂y_p = ℑyᵃᶜᵃ(i, j, k, grid, ψf, velocities.v, ∂yᶜᶠᶜ, pressure)
    w∂z_p = ℑzᵃᵃᶜ(i, j, k, grid, ψf, velocities.w, ∂zᶜᶜᶠ, pressure)
    return u∂x_p + v∂y_p + w∂z_p
end

const KineticEnergyPressureRedistribution = CustomKFO{<:typeof(uᵢ∂ᵢpᶜᶜᶜ)}
const PressureRedistribution = KineticEnergyPressureRedistribution

# The default pressure on a NonhydrostaticModel: `pNHS`, plus the hydrostatic anomaly when the model
# carries one, as a computed `Field` so that computing the term fills its halos. Read directly, `pNHS`
# would be differentiated across halos that Oceananigans leaves at `p Δt` after a time step (#335).
@inline total_pressure_ccc(i, j, k, grid, pNHS, pHY′) = @inbounds pNHS[i, j, k] + pHY′[i, j, k]
@inline total_pressure_ccc(i, j, k, grid, pNHS, ::Nothing) = @inbounds pNHS[i, j, k]

total_pressure(model::NonhydrostaticModel) =
    Field(KernelFunctionOperation{Center, Center, Center}(total_pressure_ccc, model.grid,
                                                          model.pressures.pNHS, model.pressures.pHY′))

"""
    $(SIGNATURES)

Return a `KernelFunctionOperation` that computes the pressure redistribution term:

    PR = uᵢ∂ᵢp

where `p` is the pressure. By default `p` is taken to be the total pressure (nonhydrostatic + hydrostatic):

```jldoctest ∇u⃗p_example
julia> using Oceananigans

julia> grid = RectilinearGrid(size = (1, 1, 4), extent = (1,1,1));

julia> model = NonhydrostaticModel(grid);

julia> using Oceanostics.KineticEnergyEquation: KineticEnergyPressureRedistribution

julia> ∇u⃗p = KineticEnergyPressureRedistribution(model)
KineticEnergyPressureRedistribution KernelFunctionOperation at (Center, Center, Center)
├── grid: 1×1×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 1×1×3 halo
├── kernel_function: uᵢ∂ᵢpᶜᶜᶜ (generic function with 1 method)
└── arguments: ("NamedTuple", "Field")
└── computes: kinetic energy pressure redistribution  uᵢ∂ᵢp
```

We can also pass `velocities` and `pressure` keywords to perform more specific calculations. The
example below illustrates calculation of the nonhydrostatic contribution to the pressure
redistrubution term:

```jldoctest ∇u⃗p_example
julia> ∇u⃗pNHS = KineticEnergyPressureRedistribution(model, pressure=model.pressures.pNHS)
KineticEnergyPressureRedistribution KernelFunctionOperation at (Center, Center, Center)
├── grid: 1×1×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 1×1×3 halo
├── kernel_function: uᵢ∂ᵢpᶜᶜᶜ (generic function with 1 method)
└── arguments: ("NamedTuple", "Field")
└── computes: kinetic energy pressure redistribution  uᵢ∂ᵢp
```

On a `HydrostaticFreeSurfaceModel` the sum runs over `u` and `v` only, and `p` defaults to the
hydrostatic pressure anomaly `model.pressure.pHY′`, which is the pressure the hydrostatic momentum
tendencies carry. The free surface's own pressure is not part of it.

On a `NonhydrostaticModel` that has taken a time step, Oceananigans (0.106 to 0.116 at least) leaves the
halos of `pNHS` at the pressure times `Δt`: it corrects the velocities with `p Δt` and then divides the
interior alone by `Δt`. The default `p` is a computed `Field`, which fills its own halos when the term
is computed, so it is not affected. A pressure passed by hand, such as `model.pressures.pNHS` above, is
differentiated across its own halos, so in the cells next to a periodic boundary the term is wrong until
they are refilled with `fill_halo_regions!(model.pressures.pNHS)` right before it is evaluated. The
error is a redistribution and barely changes a volume integral.
"""
function KineticEnergyPressureRedistribution(model::NonhydrostaticModel; velocities = model.velocities,
                                             pressure = total_pressure(model),
                                             location = (Center, Center, Center))
    validate_location(location, "KineticEnergyPressureRedistribution")
    return KernelFunctionOperation{Center, Center, Center}(uᵢ∂ᵢpᶜᶜᶜ, model.grid, velocities, pressure)
end

function KineticEnergyPressureRedistribution(model::HydrostaticFreeSurfaceModel;
                                             velocities = prognostic_velocities(model, model.velocities),
                                             pressure = model.pressure.pHY′,
                                             location = (Center, Center, Center))
    validate_location(location, "KineticEnergyPressureRedistribution")
    return KernelFunctionOperation{Center, Center, Center}(uᵢ∂ᵢpᶜᶜᶜ, model.grid, velocities, pressure)
end
#---

#+++ Potential-to-kinetic energy conversion
@inline function uᵢbᵢᶜᶜᶜ(i, j, k, grid, velocities, buoyancy_model, tracers)
    ubˣ = ℑxᶜᵃᵃ(i, j, k, grid, ψf, velocities.u, x_dot_g_bᶠᶜᶜ, buoyancy_model, tracers)
    vbʸ = ℑyᵃᶜᵃ(i, j, k, grid, ψf, velocities.v, y_dot_g_bᶜᶠᶜ, buoyancy_model, tracers)
    wbᶻ = ℑzᵃᵃᶜ(i, j, k, grid, ψf, velocities.w, z_dot_g_bᶜᶜᶠ, buoyancy_model, tracers)
    return ubˣ + vbʸ + wbᶻ
end

const PotentialToKineticEnergyConversion = CustomKFO{<:typeof(uᵢbᵢᶜᶜᶜ)}
const PotentialEnergyConversion = PotentialToKineticEnergyConversion

"""
    $(SIGNATURES)

Return a `KernelFunctionOperation` that computes the conversion of potential energy into kinetic
energy, defined as

    uᵢbᵢ

where bᵢ is the component of the buoyancy acceleration in the `i`-th direction (which is zero for x
and y, except when `gravity_unit_vector` isn't aligned with the grid's z-direction) and all three
components of `i=1,2,3` are added up.

On a `HydrostaticFreeSurfaceModel`, `w` is the vertical velocity diagnosed from continuity. Buoyancy
enters the hydrostatic momentum equations through the hydrostatic pressure, so `wb` is not one of the
terms of `uₕGₕ` there; it is the conversion that the pressure work `-uₕ⋅∇ₕp` contains, since
`∂p/∂z = b`.

By default, the conversion will be calculated using the resolved `velocities` and `tracers`:

```jldoctest wb_example
julia> using Oceananigans

julia> grid = RectilinearGrid(size = (1, 1, 4), extent = (1,1,1));

julia> model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b);

julia> using Oceanostics.KineticEnergyEquation: PotentialEnergyConversion

julia> wb = PotentialEnergyConversion(model)
PotentialToKineticEnergyConversion KernelFunctionOperation at (Center, Center, Center)
├── grid: 1×1×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 1×1×3 halo
├── kernel_function: uᵢbᵢᶜᶜᶜ (generic function with 1 method)
└── arguments: ("NamedTuple", "BuoyancyForce", "NamedTuple")
└── computes: potential to kinetic energy conversion  uᵢbᵢ
```

If we want to calculate only the _turbulent_ conversion rate, we can do so by passing turbulent
perturbations to the `velocities` and/or `tracers` options):

```jldoctest wb_example
julia> w′ = Field(model.velocities.w - Field(Average(model.velocities.w)));

julia> b′ = Field(model.tracers.b - Field(Average(model.tracers.b)));

julia> w′b′ = PotentialEnergyConversion(model, velocities=(u=model.velocities.u, v=model.velocities.v, w=w′), tracers=(b=b′,))
PotentialToKineticEnergyConversion KernelFunctionOperation at (Center, Center, Center)
├── grid: 1×1×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 1×1×3 halo
├── kernel_function: uᵢbᵢᶜᶜᶜ (generic function with 1 method)
└── arguments: ("NamedTuple", "BuoyancyForce", "NamedTuple")
└── computes: potential to kinetic energy conversion  uᵢbᵢ
```
"""
function PotentialEnergyConversion(model::SupportedModel; velocities = model.velocities, tracers = model.tracers,
                                   location = (Center, Center, Center))
    validate_location(location, "PotentialEnergyConversion")
    return KernelFunctionOperation{Center, Center, Center}(uᵢbᵢᶜᶜᶜ, model.grid, velocities, model.buoyancy, tracers)
end
#---

#+++ KineticEnergyDissipationRate
# Each term takes the velocity component it differentiates apart from the fields the momentum flux is
# computed with, so that a HydrostaticFreeSurfaceModel can drop w from the former and keep it in the latter.
# The fluxes are the bare `viscous_flux_*`, the closure's explicit flux, without the time discretization
# that `∂ⱼ_τᵢⱼ` applies and that zeroes the interior `∂zu` flux of a vertically implicit closure; so εₖ
# keeps a term that `KineticEnergyStress` and `KineticEnergyTendency` leave to the implicit step. Nor are
# they masked across the faces of an `ImmersedBoundaryGrid`, as the `_viscous_flux_*` behind `∂ⱼ_τᵢⱼ` are.

# ∂ⱼu₁ ⋅ τ₁ⱼ; `args` are the closure, its fields, the clock, the model fields and the buoyancy
@inline Axᶜᶜᶜ_δuᶜᶜᶜ_τ₁₁ᶜᶜᶜ(i, j, k, grid, u, args...) =
    -Axᶜᶜᶜ(i, j, k, grid) * δxᶜᵃᵃ(i, j, k, grid, u) * viscous_flux_ux(i, j, k, grid, args...)
@inline Ayᶠᶠᶜ_δuᶠᶠᶜ_τ₁₂ᶠᶠᶜ(i, j, k, grid, u, args...) =
    -Ayᶠᶠᶜ(i, j, k, grid) * δyᵃᶠᵃ(i, j, k, grid, u) * viscous_flux_uy(i, j, k, grid, args...)
@inline Azᶠᶜᶠ_δuᶠᶜᶠ_τ₁₃ᶠᶜᶠ(i, j, k, grid, u, args...) =
    -Azᶠᶜᶠ(i, j, k, grid) * δzᵃᵃᶠ(i, j, k, grid, u) * viscous_flux_uz(i, j, k, grid, args...)

# ∂ⱼu₂ ⋅ τ₂ⱼ
@inline Axᶠᶠᶜ_δvᶠᶠᶜ_τ₂₁ᶠᶠᶜ(i, j, k, grid, v, args...) =
    -Axᶠᶠᶜ(i, j, k, grid) * δxᶠᵃᵃ(i, j, k, grid, v) * viscous_flux_vx(i, j, k, grid, args...)
@inline Ayᶜᶜᶜ_δvᶜᶜᶜ_τ₂₂ᶜᶜᶜ(i, j, k, grid, v, args...) =
    -Ayᶜᶜᶜ(i, j, k, grid) * δyᵃᶜᵃ(i, j, k, grid, v) * viscous_flux_vy(i, j, k, grid, args...)
@inline Azᶜᶠᶠ_δvᶜᶠᶠ_τ₂₃ᶜᶠᶠ(i, j, k, grid, v, args...) =
    -Azᶜᶠᶠ(i, j, k, grid) * δzᵃᵃᶠ(i, j, k, grid, v) * viscous_flux_vz(i, j, k, grid, args...)

# ∂ⱼu₃ ⋅ τ₃ⱼ
@inline Axᶠᶜᶠ_δwᶠᶜᶠ_τ₃₁ᶠᶜᶠ(i, j, k, grid, w, args...) =
    -Axᶠᶜᶠ(i, j, k, grid) * δxᶠᵃᵃ(i, j, k, grid, w) * viscous_flux_wx(i, j, k, grid, args...)
@inline Ayᶜᶠᶠ_δwᶜᶠᶠ_τ₃₂ᶜᶠᶠ(i, j, k, grid, w, args...) =
    -Ayᶜᶠᶠ(i, j, k, grid) * δyᵃᶠᵃ(i, j, k, grid, w) * viscous_flux_wy(i, j, k, grid, args...)
@inline Azᶜᶜᶜ_δwᶜᶜᶜ_τ₃₃ᶜᶜᶜ(i, j, k, grid, w, args...) =
    -Azᶜᶜᶜ(i, j, k, grid) * δzᵃᵃᶜ(i, j, k, grid, w) * viscous_flux_wz(i, j, k, grid, args...)

# A hydrostatic model has no `w` row (see `prognostic_velocities`), so its three `w` terms are skipped
# without evaluating their fluxes, as `ψf` does for a `ZeroField`
@inline Axᶠᶜᶠ_δwᶠᶜᶠ_τ₃₁ᶠᶜᶠ(i, j, k, grid, ::ZeroField, args...) = zero(grid)
@inline Ayᶜᶠᶠ_δwᶜᶠᶠ_τ₃₂ᶜᶠᶠ(i, j, k, grid, ::ZeroField, args...) = zero(grid)
@inline Azᶜᶜᶜ_δwᶜᶜᶜ_τ₃₃ᶜᶜᶜ(i, j, k, grid, ::ZeroField, args...) = zero(grid)

# The `w` of the `w` row: the one in `fields`, or a `ZeroField` that skips the row on a hydrostatic
# model. The kernel takes the velocities from `fields` and gets only this choice on its own.
# Oceananigans computes the fields of every `NamedTuple` argument each time the operation is computed,
# so a second `NamedTuple` of velocities would compute a perturbation field built from a mean, and the
# mean, twice per evaluation.
@inline differentiated_w(fields, ::Val{true}) = fields.w
@inline differentiated_w(fields, ::Val{false}) = ZeroField()

@inline function viscous_dissipation_rate_ccc(i, j, k, grid, closure_fields, fields, p)
    args = (p.closure, closure_fields, p.clock, fields, p.buoyancy)
    u, v, w = fields.u, fields.v, differentiated_w(fields, p.w_row)
    return (Axᶜᶜᶜ_δuᶜᶜᶜ_τ₁₁ᶜᶜᶜ(i, j, k, grid,         u, args...) + # C, C, C
            ℑxyᶜᶜᵃ(i, j, k, grid, Ayᶠᶠᶜ_δuᶠᶠᶜ_τ₁₂ᶠᶠᶜ, u, args...) + # F, F, C  → C, C, C
            ℑxzᶜᵃᶜ(i, j, k, grid, Azᶠᶜᶠ_δuᶠᶜᶠ_τ₁₃ᶠᶜᶠ, u, args...) + # F, C, F  → C, C, C

            ℑxyᶜᶜᵃ(i, j, k, grid, Axᶠᶠᶜ_δvᶠᶠᶜ_τ₂₁ᶠᶠᶜ, v, args...) + # F, F, C  → C, C, C
            Ayᶜᶜᶜ_δvᶜᶜᶜ_τ₂₂ᶜᶜᶜ(i, j, k, grid,         v, args...) + # C, C, C
            ℑyzᵃᶜᶜ(i, j, k, grid, Azᶜᶠᶠ_δvᶜᶠᶠ_τ₂₃ᶜᶠᶠ, v, args...) + # C, F, F  → C, C, C

            ℑxzᶜᵃᶜ(i, j, k, grid, Axᶠᶜᶠ_δwᶠᶜᶠ_τ₃₁ᶠᶜᶠ, w, args...) + # F, C, F  → C, C, C
            ℑyzᵃᶜᶜ(i, j, k, grid, Ayᶜᶠᶠ_δwᶜᶠᶠ_τ₃₂ᶜᶠᶠ, w, args...) + # C, F, F  → C, C, C
            Azᶜᶜᶜ_δwᶜᶜᶜ_τ₃₃ᶜᶜᶜ(i, j, k, grid,         w, args...)   # C, C, C
            ) / Vᶜᶜᶜ(i, j, k, grid) # The division by volume, with the A*δuᵢ in each term, makes a derivative
end

const KineticEnergyDissipationRate = CustomKFO{<:typeof(viscous_dissipation_rate_ccc)}
const DissipationRate = KineticEnergyDissipationRate

"""
    $(SIGNATURES)

Calculate the kinetic energy dissipation rate, defined as

    εₖ = -∂ⱼuᵢ ⋅ τᵢⱼ

where ∂ⱼuᵢ is the velocity gradient tensor and τᵢⱼ is the closure's viscous momentum flux (minus the
stress tensor), in the same convention Oceananigans uses. That definition holds for any closure. For
a Newtonian flux with a scalar viscosity, `τᵢⱼ = -2νSᵢⱼ`, it reduces on a `NonhydrostaticModel` to
`εₖ = 2νSᵢⱼSᵢⱼ ≥ 0`, which [`KineticEnergyIsotropicDissipationRate`](@ref) computes directly; neither
that reduction nor, for a flux that is not down-gradient, the sign carries over to other closures.

A `HydrostaticFreeSurfaceModel` has no `w`-momentum equation, so on that model the sum runs over
`i = 1, 2` only and `εₖ` is the dissipation rate of `½(u² + v²)`. The fluxes `τ₁ⱼ` and `τ₂ⱼ` are
still computed with the diagnostic `w`, as in the model's own `u` and `v` tendencies. The reduction
to `2νSᵢⱼSᵢⱼ` needs the `w` row, so there `εₖ` is not that, and the isotropic rate, which keeps the
row, is not a term of the hydrostatic budget. `∫εₖ dV` is still the viscous sink of `½(u² + v²)`.
Pointwise, `εₖ ≥ 0` with a horizontal or vertical formulation (`HorizontalScalarDiffusivity`,
`VerticalScalarDiffusivity`, CATKE), whose `τ` is built from `u` and `v` alone. With a
three-dimensional `ScalarDiffusivity`, whose `τ₁₃` carries `∂ₓw`, the cross term `ν ∂zu ∂ₓw` has no
`w` row to complete its square, so `εₖ` is indefinite cell by cell, by a term of order aspect ratio
squared relative to `ν(∂zu)²`, which is the order the hydrostatic approximation neglects elsewhere.

The fluxes are the closure's explicit `viscous_flux_*`, with no time discretization applied, so under a
closure with a `VerticallyImplicitTimeDiscretization` (CATKE's default, and an option of
`ScalarDiffusivity` and `VerticalScalarDiffusivity`) `εₖ` still includes the vertical shear term that
the model applies in the implicit step. [`KineticEnergyStress`](@ref) and
[`KineticEnergyTendency`](@ref) leave that term out, so `∫DIFF dV = ∫εₖ dV` fails there and an
integrated budget should be written with `εₖ`.

On an `ImmersedBoundaryGrid` the same bare fluxes are not masked across immersed faces, as the
`_viscous_flux_*` behind `∂ⱼτᵢⱼ` are, so `εₖ` there includes the shear across faces the model treats as
solid and `∫DIFF dV = ∫εₖ dV` does not hold either.

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(grid; closure=ScalarDiffusivity(ν=1e-4));

julia> ε = KineticEnergyEquation.DissipationRate(model)
KineticEnergyDissipationRate KernelFunctionOperation at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: viscous_dissipation_rate_ccc (generic function with 1 method)
└── arguments: ("Nothing", "NamedTuple", "NamedTuple")
└── computes: kinetic energy dissipation rate  -∂ⱼuᵢ·τᵢⱼ
```
"""
function DissipationRate(model; U=ZeroField(), V=ZeroField(), W=ZeroField(),
                                               location = (Center, Center, Center))
    validate_location(location, "DissipationRate")
    validate_closure_supplies_a_flux("KineticEnergyDissipationRate", model)
    mean_velocities = (u=U, v=V, w=W)
    model_fields = perturbation_fields(model; mean_velocities...)
    parameters = (; model.closure,
                  model.clock,
                  model.buoyancy,
                  w_row = prognostic_w_row(model))

    return KernelFunctionOperation{Center, Center, Center}(viscous_dissipation_rate_ccc, model.grid,
                                                           model.closure_fields, model_fields, parameters)
end
#---

#+++ KineticEnergyIsotropicDissipationRate
@inline function isotropic_viscous_dissipation_rate_ccc(i, j, k, grid, u, v, w, p)

    Σˣˣ² = ∂xᶜᶜᶜ(i, j, k, grid, u)^2
    Σʸʸ² = ∂yᶜᶜᶜ(i, j, k, grid, v)^2
    Σᶻᶻ² = ∂zᶜᶜᶜ(i, j, k, grid, w)^2

    Σˣʸ² = ℑxyᶜᶜᵃ(i, j, k, grid, fψ_plus_gφ², ∂yᶠᶠᶜ, u, ∂xᶠᶠᶜ, v) / 4
    Σˣᶻ² = ℑxzᶜᵃᶜ(i, j, k, grid, fψ_plus_gφ², ∂zᶠᶜᶠ, u, ∂xᶠᶜᶠ, w) / 4
    Σʸᶻ² = ℑyzᵃᶜᶜ(i, j, k, grid, fψ_plus_gφ², ∂zᶜᶠᶠ, v, ∂yᶜᶠᶠ, w) / 4

    ν = _νᶜᶜᶜ(i, j, k, grid, p.closure, p.closure_fields, p.clock, p.model_fields)

    return 2ν * (Σˣˣ² + Σʸʸ² + Σᶻᶻ² + 2 * (Σˣʸ² + Σˣᶻ² + Σʸᶻ²))
end

const KineticEnergyIsotropicDissipationRate = CustomKFO{<:typeof(isotropic_viscous_dissipation_rate_ccc)}
const IsotropicDissipationRate = KineticEnergyIsotropicDissipationRate

"""
    $(SIGNATURES)

Calculate the isotropic kinetic energy dissipation rate as

    εₖ = 2 ν SᵢⱼSᵢⱼ,

where Sᵢⱼ is the strain rate tensor, for a fluid with an isotropic turbulence closure (i.e., a
turbulence closure where ν (eddy or not) is the same for all directions). It keeps the `w` row on both
model types, so on a `HydrostaticFreeSurfaceModel` it is not the dissipation term of the budget of
`½(u² + v²)`, which [`DissipationRate`](@ref) computes: it includes the dissipation of `½w²`, which that
budget does not carry.

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(grid; closure=ScalarDiffusivity(ν=1e-4));

julia> ε = KineticEnergyEquation.KineticEnergyIsotropicDissipationRate(model)
KineticEnergyIsotropicDissipationRate KernelFunctionOperation at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: isotropic_viscous_dissipation_rate_ccc (generic function with 1 method)
└── arguments: ("Field", "Field", "Field", "NamedTuple")
└── computes: isotropic kinetic energy dissipation rate  2νSᵢⱼSᵢⱼ
```
"""
function KineticEnergyIsotropicDissipationRate(u, v, w, closure, closure_fields, model_fields, clock; location = (Center, Center, Center))
    validate_location(location, "KineticEnergyIsotropicDissipationRate")
    validate_dissipative_closure(closure)

    parameters = (; closure, closure_fields, clock, model_fields)
    return KernelFunctionOperation{Center, Center, Center}(isotropic_viscous_dissipation_rate_ccc, u.grid,
                                                           u, v, w, parameters)
end

@inline KineticEnergyIsotropicDissipationRate(model; location = (Center, Center, Center)) =
    KineticEnergyIsotropicDissipationRate(model.velocities..., model.closure, model.closure_fields, fields(model), model.clock; location = location)
#---

end # module
