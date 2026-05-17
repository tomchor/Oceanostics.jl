module WMomentumEquation
using DocStringExtensions

using Oceananigans: fields, Face, Center, KernelFunctionOperation, AbstractModel
using Oceananigans.Models: HydrostaticFreeSurfaceModel
using Oceananigans.Models.NonhydrostaticModels: w_velocity_tendency
using Oceananigans.Advection: div_𝐯w
using Oceananigans.BuoyancyFormulations: z_dot_g_bᶜᶜᶠ
using Oceananigans.Coriolis: z_f_cross_U
using Oceananigans.TurbulenceClosures: ∂ⱼ_τ₃ⱼ, immersed_∂ⱼ_τ₃ⱼ
using Oceananigans.StokesDrifts: z_curl_Uˢ_cross_U, ∂t_wˢ

using Oceanostics: validate_location, CustomKFO

# NB: no `PressureGradient` here, unlike U/V. `w_velocity_tendency` has no `-∂z(pHY′)` line —
# Oceananigans treats the vertical hydrostatic balance as a property of the pressure projection
# step rather than as an explicit term, so the only mode-dependent piece is whether buoyancy
# itself appears (case `hydrostatic_pressure_anomaly = nothing`) or not (default, with splitting).
export Advection, BuoyancyAcceleration, CoriolisAcceleration,
       ViscousDissipation, ImmersedViscousDissipation, TotalViscousDissipation,
       StokesShear, StokesTendency, Forcing, Tendency,
       WAdvection, WBuoyancyAcceleration, WCoriolisAcceleration,
       WViscousDissipation, WImmersedViscousDissipation, WTotalViscousDissipation,
       WStokesShear, WStokesTendency, WForcing, WTendency

# Inline function for total viscous dissipation
@inline total_∂ⱼ_τ₃ⱼ(i, j, k, grid, velocities, w_immersed_bc, closure, diffusivities, clock, model_fields, buoyancy) =
    ∂ⱼ_τ₃ⱼ(i, j, k, grid, closure, diffusivities, clock, model_fields, buoyancy) +
    immersed_∂ⱼ_τ₃ⱼ(i, j, k, grid, velocities, w_immersed_bc, closure, diffusivities, clock, model_fields)

# Type aliases for major functions
const Advection = CustomKFO{<:typeof(div_𝐯w)}
const BuoyancyAcceleration = CustomKFO{<:typeof(z_dot_g_bᶜᶜᶠ)}
const CoriolisAcceleration = CustomKFO{<:typeof(z_f_cross_U)}
const ViscousDissipation = CustomKFO{<:typeof(∂ⱼ_τ₃ⱼ)}
const ImmersedViscousDissipation = CustomKFO{<:typeof(immersed_∂ⱼ_τ₃ⱼ)}
const TotalViscousDissipation = CustomKFO{<:typeof(total_∂ⱼ_τ₃ⱼ)}
const StokesShear = CustomKFO{<:typeof(z_curl_Uˢ_cross_U)}
const StokesTendency = CustomKFO{<:typeof(∂t_wˢ)}
const Forcing = KernelFunctionOperation{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any}
const Tendency = CustomKFO{<:typeof(w_velocity_tendency)}

# Aliases for consistency with TracerEquation naming
const WAdvection = Advection
const WBuoyancyAcceleration = BuoyancyAcceleration
const WCoriolisAcceleration = CoriolisAcceleration
const WViscousDissipation = ViscousDissipation
const WImmersedViscousDissipation = ImmersedViscousDissipation
const WTotalViscousDissipation = TotalViscousDissipation
const WStokesShear = StokesShear
const WStokesTendency = StokesTendency
const WForcing = Forcing
const WTendency = Tendency

#+++ Advection
"""
    $(SIGNATURES)

Calculate the advection of w-momentum as

    ADV = ∂ⱼ (uⱼ w)

using Oceananigans' kernel [`div_𝐯w`.](https://clima.github.io/OceananigansDocumentation/stable/appendix/library/#Oceananigans.Advection.div_𝐯w-NTuple{7,%20Any})

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(grid);

julia> ADV = WMomentumEquation.Advection(model)
KernelFunctionOperation at (Center, Center, Face)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: div_𝐯w (generic function with 10 methods)
└── arguments: ("Centered", "NamedTuple", "Field")
```
"""
function Advection(model, u, v, w, advection_scheme; location = (Center, Center, Face))
    validate_location(location, "Advection", (Center, Center, Face))
    total_velocities = (; u, v, w)
    return KernelFunctionOperation{Center, Center, Face}(div_𝐯w, model.grid, advection_scheme, total_velocities, w)
end

Advection(model; kwargs...) =
    Advection(model, model.velocities..., model.advection; kwargs...)

Advection(model::HydrostaticFreeSurfaceModel; kwargs...) =
    Advection(model, model.velocities..., model.advection.momentum; kwargs...)
#---

#+++ Buoyancy acceleration
"""
    $(SIGNATURES)

Calculate the buoyancy acceleration in the z-direction as

    BUOY = ĝ_z b

where ĝ_z is the z-component of the gravitational unit vector and b is the buoyancy.

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b);

julia> BUOY = WMomentumEquation.BuoyancyAcceleration(model)
KernelFunctionOperation at (Center, Center, Face)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: z_dot_g_bᶜᶜᶠ (generic function with 10 methods)
└── arguments: ("BuoyancyForce", "NamedTuple")
```
"""
function BuoyancyAcceleration(model, buoyancy, tracers; location = (Center, Center, Face))
    validate_location(location, "BuoyancyAcceleration", (Center, Center, Face))
    return KernelFunctionOperation{Center, Center, Face}(z_dot_g_bᶜᶜᶠ, model.grid, buoyancy, tracers)
end

BuoyancyAcceleration(model; kwargs...) =
    BuoyancyAcceleration(model, model.buoyancy, model.tracers; kwargs...)
#---

#+++ Coriolis acceleration
"""
    $(SIGNATURES)

Calculate the Coriolis acceleration in the z-direction as

    COR = - (f × u)_z

where f is the Coriolis parameter vector and u is the velocity vector.

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(grid; coriolis=FPlane(f=1e-4));

julia> COR = WMomentumEquation.CoriolisAcceleration(model)
KernelFunctionOperation at (Center, Center, Face)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: z_f_cross_U (generic function with 10 methods)
└── arguments: ("FPlane", "NamedTuple")
```
"""
function CoriolisAcceleration(model, coriolis, velocities; location = (Center, Center, Face))
    validate_location(location, "CoriolisAcceleration", (Center, Center, Face))
    return KernelFunctionOperation{Center, Center, Face}(z_f_cross_U, model.grid, coriolis, velocities)
end

CoriolisAcceleration(model; kwargs...) =
    CoriolisAcceleration(model, model.coriolis, model.velocities; kwargs...)
#---

#+++ Viscous dissipation
"""
    $(SIGNATURES)

Calculate the viscous dissipation term (excluding immersed boundaries) as

    VISC = - ∂ⱼ τ₃ⱼ,

where τ₃ⱼ is the viscous stress tensor for the z-momentum equation.

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(grid);

julia> VISC = WMomentumEquation.ViscousDissipation(model)
KernelFunctionOperation at (Center, Center, Face)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: ∂ⱼ_τ₃ⱼ (generic function with 10 methods)
└── arguments: ("Nothing", "Nothing", "Clock", "NamedTuple", "Nothing")
```
"""
function ViscousDissipation(model, closure, diffusivities, clock, model_fields, buoyancy; location = (Center, Center, Face))
    validate_location(location, "ViscousDissipation", (Center, Center, Face))
    return KernelFunctionOperation{Center, Center, Face}(∂ⱼ_τ₃ⱼ, model.grid, closure, diffusivities, clock, model_fields, buoyancy)
end

ViscousDissipation(model; kwargs...) =
    ViscousDissipation(model, model.closure, model.closure_fields, model.clock, fields(model), model.buoyancy; kwargs...)

"""
    $(SIGNATURES)

Calculate the viscous dissipation term due to immersed boundaries as

    VISC = - ∂ⱼ τ₃ⱼ,

where τ₃ⱼ is the immersed boundary viscous stress tensor for the z-momentum equation.

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(grid);

julia> VISC = WMomentumEquation.ImmersedViscousDissipation(model)
KernelFunctionOperation at (Center, Center, Face)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: immersed_∂ⱼ_τ₃ⱼ (generic function with 2 methods)
└── arguments: ("NamedTuple", "Nothing", "Nothing", "Nothing", "Clock", "NamedTuple")
```
"""
function ImmersedViscousDissipation(model, velocities, w_immersed_bc, closure, diffusivities, clock, model_fields; location = (Center, Center, Face))
    validate_location(location, "ImmersedViscousDissipation", (Center, Center, Face))
    return KernelFunctionOperation{Center, Center, Face}(immersed_∂ⱼ_τ₃ⱼ, model.grid, velocities, w_immersed_bc, closure, diffusivities, clock, model_fields)
end

function ImmersedViscousDissipation(model; kwargs...)
    w_immersed_bc = model.velocities.w.boundary_conditions.immersed
    return ImmersedViscousDissipation(model, model.velocities, w_immersed_bc, model.closure, model.closure_fields, model.clock, fields(model); kwargs...)
end

"""
    $(SIGNATURES)

Calculate the total viscous dissipation term as

    VISC = - ∂ⱼ τ₃ⱼ - ∂ⱼ τ₃ⱼ_immersed,

where τ₃ⱼ is the interior viscous stress tensor and τ₃ⱼ_immersed is the immersed boundary
viscous stress tensor for the z-momentum equation.

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(grid);

julia> VISC = WMomentumEquation.TotalViscousDissipation(model)
KernelFunctionOperation at (Center, Center, Face)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: total_∂ⱼ_τ₃ⱼ (generic function with 1 method)
└── arguments: ("NamedTuple", "Nothing", "Nothing", "Nothing", "Clock", "NamedTuple", "Nothing")
```
"""
function TotalViscousDissipation(model, velocities, w_immersed_bc, closure, diffusivities, clock, model_fields, buoyancy; location = (Center, Center, Face))
    validate_location(location, "TotalViscousDissipation", (Center, Center, Face))
    return KernelFunctionOperation{Center, Center, Face}(total_∂ⱼ_τ₃ⱼ, model.grid, velocities, w_immersed_bc, closure, diffusivities, clock, model_fields, buoyancy)
end

function TotalViscousDissipation(model; kwargs...)
    w_immersed_bc = model.velocities.w.boundary_conditions.immersed
    return TotalViscousDissipation(model, model.velocities, w_immersed_bc, model.closure, model.closure_fields, model.clock, fields(model), model.buoyancy; kwargs...)
end
#---

#+++ Stokes drift terms
"""
    $(SIGNATURES)

Calculate the Stokes shear term as

    STOKES_SHEAR = ((∇ × uˢ) × u)_z

where uˢ is the Stokes drift velocity and u is the velocity vector.

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(grid);

julia> STOKES = WMomentumEquation.StokesShear(model)
KernelFunctionOperation at (Center, Center, Face)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: z_curl_Uˢ_cross_U (generic function with 10 methods)
└── arguments: ("Nothing", "NamedTuple", "Float64")
```
"""
function StokesShear(model, stokes_drift, velocities, time; location = (Center, Center, Face))
    validate_location(location, "StokesShear", (Center, Center, Face))
    return KernelFunctionOperation{Center, Center, Face}(z_curl_Uˢ_cross_U, model.grid, stokes_drift, velocities, time)
end

StokesShear(model::HydrostaticFreeSurfaceModel; kwargs...) =
    throw(ArgumentError("WMomentumEquation.StokesShear is not defined for HydrostaticFreeSurfaceModel: " *
                        "Stokes drift is not part of the hydrostatic free-surface model."))

StokesShear(model; kwargs...) =
    StokesShear(model, model.stokes_drift, model.velocities, model.clock.time; kwargs...)

"""
    $(SIGNATURES)

Calculate the Stokes tendency term as

    STOKES_TEND = ∂wˢ/∂t

where wˢ is the z-component of the Stokes drift velocity.

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(grid);

julia> STOKES = WMomentumEquation.StokesTendency(model)
KernelFunctionOperation at (Center, Center, Face)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: ∂t_wˢ (generic function with 10 methods)
└── arguments: ("Nothing", "Float64")
```
"""
function StokesTendency(model, stokes_drift, time; location = (Center, Center, Face))
    validate_location(location, "StokesTendency", (Center, Center, Face))
    return KernelFunctionOperation{Center, Center, Face}(∂t_wˢ, model.grid, stokes_drift, time)
end

StokesTendency(model::HydrostaticFreeSurfaceModel; kwargs...) =
    throw(ArgumentError("WMomentumEquation.StokesTendency is not defined for HydrostaticFreeSurfaceModel: " *
                        "Stokes drift is not part of the hydrostatic free-surface model."))

StokesTendency(model; kwargs...) =
    StokesTendency(model, model.stokes_drift, model.clock.time; kwargs...)
#---

#+++ Forcing
"""
    $(SIGNATURES)

Calculate the forcing term `Fʷ` on the z-momentum equation for `model`.

`Forcing` is a type alias for the generic `KernelFunctionOperation` (no narrowing
on the kernel function), so a constructor on `Forcing(model)` would clash across the
U/V/W modules. To disambiguate, the W-momentum convenience constructor takes an
explicit `Val(:w)` tag.

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(grid);

julia> FORC = WMomentumEquation.Forcing(model, Val(:w))
KernelFunctionOperation at (Center, Center, Face)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: Returns (generic function with 1 method)
└── arguments: ("Clock", "NamedTuple")
```
"""
function Forcing(model, forcing_func, clock, model_fields, ::Val{:w}; location = (Center, Center, Face))
    validate_location(location, "Forcing", (Center, Center, Face))
    return KernelFunctionOperation{Center, Center, Face}(forcing_func, model.grid, clock, model_fields)
end

Forcing(model::HydrostaticFreeSurfaceModel, ::Val{:w}; kwargs...) =
    throw(ArgumentError("WMomentumEquation.Forcing is not defined for HydrostaticFreeSurfaceModel: " *
                        "w is diagnosed from continuity rather than evolved by a prognostic equation."))

Forcing(model, ::Val{:w}; kwargs...) =
    Forcing(model, model.forcing.w, model.clock, fields(model), Val(:w); kwargs...)
#---

#+++ Total tendency
"""
    $(SIGNATURES)

Calculate the total tendency of the w-momentum equation as computed by Oceananigans.

For NonhydrostaticModel, this includes:
- Advection: -∇⋅(𝐯w)
- Background advection terms
- Buoyancy: ĝ_z b
- Coriolis: -(f × u)_z
- Pressure gradient: -∂p/∂z
- Viscous dissipation: -∇⋅τ₃
- Immersed viscous dissipation
- Stokes shear: ((∇ × uˢ) × u)_z
- Stokes tendency: ∂wˢ/∂t
- Forcing: Fʷ

`HydrostaticFreeSurfaceModel` does not have a prognostic w-momentum equation
(w is diagnosed from continuity), so `Tendency` is only defined for
`NonhydrostaticModel`.

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(grid);

julia> TEND = WMomentumEquation.Tendency(model)
KernelFunctionOperation at (Center, Center, Face)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: w_velocity_tendency (generic function with 1 method)
└── arguments: ("Centered", "Nothing", "Nothing", "Nothing", "Nothing", "Nothing", "Oceananigans.Models.NonhydrostaticModels.BackgroundFields", "NamedTuple", "NamedTuple", "NamedTuple", "Nothing", "Nothing", "Clock", "Returns")
```
"""
function Tendency(model, advection_scheme, coriolis, stokes_drift, closure, w_immersed_bc, buoyancy, background_fields, velocities, tracers, auxiliary_fields, diffusivities, hydrostatic_pressure, clock, forcing_func; location = (Center, Center, Face))
    validate_location(location, "Tendency", (Center, Center, Face))
    return KernelFunctionOperation{Center, Center, Face}(w_velocity_tendency, model.grid, advection_scheme, coriolis, stokes_drift, closure, w_immersed_bc, buoyancy, background_fields, velocities, tracers, auxiliary_fields, diffusivities, hydrostatic_pressure, clock, forcing_func)
end

Tendency(model::HydrostaticFreeSurfaceModel; kwargs...) =
    throw(ArgumentError("WMomentumEquation.Tendency is not defined for HydrostaticFreeSurfaceModel: " *
                        "w is diagnosed from continuity rather than evolved by a prognostic equation."))

function Tendency(model; kwargs...)
    w_immersed_bc = model.velocities.w.boundary_conditions.immersed
    return Tendency(model, model.advection, model.coriolis, model.stokes_drift, model.closure, w_immersed_bc, model.buoyancy, model.background_fields, model.velocities, model.tracers, model.auxiliary_fields, model.closure_fields, model.pressures.pHY′, model.clock, model.forcing.w; kwargs...)
end
#---

end # module
