module KineticEnergyEquation
using DocStringExtensions

export KineticEnergy
export KineticEnergyTendency
export Advection, KineticEnergyAdvection
export Stress, KineticEnergyStress
export Forcing, KineticEnergyForcing
export PressureRedistribution, KineticEnergyPressureRedistribution
export BarotropicPressureRedistribution, KineticEnergyBarotropicPressureRedistribution
export PotentialEnergyConversion, PotentialToKineticEnergyConversion
export DissipationRate, KineticEnergyDissipationRate
export KineticEnergyIsotropicDissipationRate

using Oceananigans: NonhydrostaticModel, HydrostaticFreeSurfaceModel, fields
using Oceananigans.Operators
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Grids: Center
using Oceananigans.Fields: ZeroField, Field
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
using Oceanostics: validate_location, validate_dissipative_closure, perturbation_fields, CustomKFO, momentum_advection
using Oceanostics: model_free_surface, free_surface_displacement, free_surface_gravitational_acceleration, SurfaceDisplacement

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
[`KineticEnergyBarotropicPressureRedistribution`](@ref) supplies that work, so the change of `eₖ` over a
step is `uₕGₕ - BPR` there.

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
tendencies carry. The free surface's own pressure `gη` is not part of it;
[`KineticEnergyBarotropicPressureRedistribution`](@ref) provides that work separately. On a
`NonhydrostaticModel` with a free surface, `gη` is the surface value of `pNHS`, so the default here
already contains it.

On a `NonhydrostaticModel` that has taken a time step, Oceananigans (0.106 to 0.116 at least) leaves the
halos of `pNHS` at the pressure times `Δt`: it corrects the velocities with `p Δt` and then divides the
interior alone by `Δt`. This term differentiates `p` across those halos, so in the cells next to a
periodic boundary it is wrong until the halos are refilled, with
`fill_halo_regions!(model.pressures.pNHS)` right before it is evaluated. The error is a redistribution
and barely changes a volume integral.
"""
function KineticEnergyPressureRedistribution(model::NonhydrostaticModel; velocities = model.velocities,
                                    pressure = model.pressures.pHY′ == nothing ? model.pressures.pNHS : sum(model.pressures),
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

#+++ KineticEnergyBarotropicPressureRedistribution
# The barotropic pressure is `gη`. `η` lives on the top face of the grid, k = Nz+1, and has no vertical
# structure, so its gradient at every level is the top-face difference that the free-surface step of an
# `ImplicitFreeSurface` applies to `u` and `v` and that an `ExplicitFreeSurface` puts in the tendency.
@inline ∂x_gηᶠᶜᶜ(i, j, k, grid, g, η) = g * ∂xᶠᶜᶠ(i, j, grid.Nz+1, grid, η)
@inline ∂y_gηᶜᶠᶜ(i, j, k, grid, g, η) = g * ∂yᶜᶠᶠ(i, j, grid.Nz+1, grid, η)

@inline function uₕ∇ₕgηᶜᶜᶜ(i, j, k, grid, velocities, g, η)
    u∂x_gη = ℑxᶜᵃᵃ(i, j, k, grid, ψf, velocities.u, ∂x_gηᶠᶜᶜ, g, η)
    v∂y_gη = ℑyᵃᶜᵃ(i, j, k, grid, ψf, velocities.v, ∂y_gηᶜᶠᶜ, g, η)
    return u∂x_gη + v∂y_gη
end

# A rigid lid (`free_surface = nothing`) has no barotropic pressure
@inline uₕ∇ₕgηᶜᶜᶜ(i, j, k, grid, velocities, ::Nothing, ::Nothing) = zero(grid)

const KineticEnergyBarotropicPressureRedistribution = CustomKFO{<:typeof(uₕ∇ₕgηᶜᶜᶜ)}
const BarotropicPressureRedistribution = KineticEnergyBarotropicPressureRedistribution

# The differences above read `η` in the halos along the lateral boundaries. A `HydrostaticFreeSurfaceModel`
# refills them every time it steps `η`; a `NonhydrostaticModel` fills them when `η` is set and never again
# (Oceananigans 0.113 and 0.116 at least), so on that model the term reads a copy of `η` that `compute!`
# rebuilds, halos included, each time the operation is computed.
barotropic_pressure_displacement(model, η) = η
barotropic_pressure_displacement(::NonhydrostaticModel, η) = Field(1 * η)
barotropic_pressure_displacement(model, ::Nothing) = nothing
barotropic_pressure_displacement(::NonhydrostaticModel, ::Nothing) = nothing

"""
    $(SIGNATURES)

Return a `KernelFunctionOperation` that computes the redistribution of kinetic energy by the barotropic
pressure `gη` of a free surface,

    BPR = uₕ⋅∇ₕ(gη) ,

where `η` is the free-surface displacement, read on the top face of the grid, and `g` the free surface's
gravitational acceleration. It is the part of the pressure work `uᵢ∂ᵢp` that the free surface
contributes, with the sign convention of [`KineticEnergyPressureRedistribution`](@ref): it enters the
budget of `eₖ` as `-BPR`, and `∇ₕη` is the same top-face difference the model applies to `u` and `v`.

Its volume integral is the exchange with the potential energy of the free surface,
[`FreeSurfacePotentialEnergy`](@ref) `eₛ = ½gη²`. With closed or periodic lateral boundaries and `U` the
depth-integrated transport,

    ∫ BPR dV = ∫ U⋅∇ₕ(gη) dA = -∫ gη ∇ₕ⋅U dA = ∫ gη ∂ₜη dA = ∂ₜ ∫ eₛ dA ,

which [`FreeSurfacePotentialEnergyTendency`](@ref) measures column by column as `gη w` at the surface.
Pointwise the term is of the order of the other pressure work; integrated it is small whenever the
barotropic flow is narrow compared with the external deformation radius `√(gH)/f`.

Where it sits in the budget depends on the model:

- A `HydrostaticFreeSurfaceModel` with an `ImplicitFreeSurface` or a `SplitExplicitFreeSurface` applies
  `-g∇ₕη` in its free-surface step, outside the tendency kernel, so [`KineticEnergyTendency`](@ref)
  leaves it out and the change of `eₖ` over a step is `uₕGₕ - BPR`. With an `ExplicitFreeSurface` the
  gradient is inside `Gₕ`, and this term is already part of the tendency.
- A `NonhydrostaticModel` with a free surface carries `gη` as the surface value of its nonhydrostatic
  pressure, so [`KineticEnergyPressureRedistribution`](@ref) already contains this term; it is the
  barotropic part of that pressure work, and the remainder integrates to zero. A `NonhydrostaticModel`
  with a rigid lid, the default, has no free surface and the term is zero.

The model applies `∇ₕη` at the new time level, or its average over the barotropic substeps, while this
diagnostic reads `η` as it is when evaluated, so a budget built from it closes to `O(Δt)`, like one built
with a `TimeDerivative`. Under a `MutableVerticalDiscretization` the cells move with `η` and the budget
takes a different form, which this term does not describe.

The differences of `η` reach into its halos along the lateral boundaries. A `NonhydrostaticModel` fills
those when `η` is set and never again (Oceananigans 0.116 and earlier), so on that model the term reads
a copy of `η` that is refilled whenever the operation is computed: evaluate it through a `Field` or an
output writer, not by indexing the operation directly.

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = HydrostaticFreeSurfaceModel(grid);

julia> BPR = KineticEnergyBarotropicPressureRedistribution(model)
KineticEnergyBarotropicPressureRedistribution KernelFunctionOperation at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: uₕ∇ₕgηᶜᶜᶜ (generic function with 2 methods)
└── arguments: ("NamedTuple", "Float64", "SurfaceDisplacement")
└── computes: kinetic energy barotropic pressure redistribution  uₕ⋅∇ₕ(gη)
```
"""
function KineticEnergyBarotropicPressureRedistribution(model; velocities = model.velocities,
                                                      free_surface = model_free_surface(model),
                                                      location = (Center, Center, Center))
    validate_location(location, "KineticEnergyBarotropicPressureRedistribution")
    g = free_surface_gravitational_acceleration(free_surface)
    η = SurfaceDisplacement(barotropic_pressure_displacement(model, free_surface_displacement(free_surface)))
    return KernelFunctionOperation{Center, Center, Center}(uₕ∇ₕgηᶜᶜᶜ, model.grid, velocities, g, η)
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

@inline function viscous_dissipation_rate_ccc(i, j, k, grid, velocities, closure_fields, fields, p)
    args = (p.closure, closure_fields, p.clock, fields, p.buoyancy)
    return (Axᶜᶜᶜ_δuᶜᶜᶜ_τ₁₁ᶜᶜᶜ(i, j, k, grid,         velocities.u, args...) + # C, C, C
            ℑxyᶜᶜᵃ(i, j, k, grid, Ayᶠᶠᶜ_δuᶠᶠᶜ_τ₁₂ᶠᶠᶜ, velocities.u, args...) + # F, F, C  → C, C, C
            ℑxzᶜᵃᶜ(i, j, k, grid, Azᶠᶜᶠ_δuᶠᶜᶠ_τ₁₃ᶠᶜᶠ, velocities.u, args...) + # F, C, F  → C, C, C

            ℑxyᶜᶜᵃ(i, j, k, grid, Axᶠᶠᶜ_δvᶠᶠᶜ_τ₂₁ᶠᶠᶜ, velocities.v, args...) + # F, F, C  → C, C, C
            Ayᶜᶜᶜ_δvᶜᶜᶜ_τ₂₂ᶜᶜᶜ(i, j, k, grid,         velocities.v, args...) + # C, C, C
            ℑyzᵃᶜᶜ(i, j, k, grid, Azᶜᶠᶠ_δvᶜᶠᶠ_τ₂₃ᶜᶠᶠ, velocities.v, args...) + # C, F, F  → C, C, C

            ℑxzᶜᵃᶜ(i, j, k, grid, Axᶠᶜᶠ_δwᶠᶜᶠ_τ₃₁ᶠᶜᶠ, velocities.w, args...) + # F, C, F  → C, C, C
            ℑyzᵃᶜᶜ(i, j, k, grid, Ayᶜᶠᶠ_δwᶜᶠᶠ_τ₃₂ᶜᶠᶠ, velocities.w, args...) + # C, F, F  → C, C, C
            Azᶜᶜᶜ_δwᶜᶜᶜ_τ₃₃ᶜᶜᶜ(i, j, k, grid,         velocities.w, args...)   # C, C, C
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
└── arguments: ("NamedTuple", "Nothing", "NamedTuple", "NamedTuple")
└── computes: kinetic energy dissipation rate  -∂ⱼuᵢ·τᵢⱼ
```
"""
function DissipationRate(model; U=ZeroField(), V=ZeroField(), W=ZeroField(),
                                               location = (Center, Center, Center))
    validate_location(location, "DissipationRate")
    mean_velocities = (u=U, v=V, w=W)
    model_fields = perturbation_fields(model; mean_velocities...)
    parameters = (; model.closure,
                  model.clock,
                  model.buoyancy)

    return KernelFunctionOperation{Center, Center, Center}(viscous_dissipation_rate_ccc, model.grid,
                                                           prognostic_velocities(model, model_fields),
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
