module VMomentumEquation
using DocStringExtensions

using Oceananigans: fields, Face, Center, KernelFunctionOperation, AbstractModel
using Oceananigans.Models: HydrostaticFreeSurfaceModel
using Oceananigans.Models.NonhydrostaticModels: v_velocity_tendency
using Oceananigans.Models.HydrostaticFreeSurfaceModels: hydrostatic_free_surface_v_velocity_tendency
using Oceananigans.Advection: div_𝐯v
using Oceananigans.BuoyancyFormulations: y_dot_g_bᶜᶠᶜ
using Oceananigans.Coriolis: y_f_cross_U
using Oceananigans.TurbulenceClosures: ∂ⱼ_τ₂ⱼ, immersed_∂ⱼ_τ₂ⱼ
using Oceananigans.StokesDrifts: y_curl_Uˢ_cross_U, ∂t_vˢ
using Oceananigans.Operators: ∂yᶜᶠᶜ

using Oceanostics: validate_location, CustomKFO

export Advection, BuoyancyAcceleration, CoriolisAcceleration, PressureGradient,
       ViscousDissipation, ImmersedViscousDissipation, TotalViscousDissipation,
       StokesShear, StokesTendency, Forcing, Tendency,
       VAdvection, VBuoyancyAcceleration, VCoriolisAcceleration, VPressureGradient,
       VViscousDissipation, VImmersedViscousDissipation, VTotalViscousDissipation,
       VStokesShear, VStokesTendency, VForcing, VTendency

# Inline function for total viscous dissipation
@inline total_∂ⱼ_τ₂ⱼ(i, j, k, grid, velocities, v_immersed_bc, closure, diffusivities, clock, model_fields, buoyancy) =
    ∂ⱼ_τ₂ⱼ(i, j, k, grid, closure, diffusivities, clock, model_fields, buoyancy) +
    immersed_∂ⱼ_τ₂ⱼ(i, j, k, grid, velocities, v_immersed_bc, closure, diffusivities, clock, model_fields)

# Inline function for hydrostatic pressure gradient
@inline hydrostatic_pressure_gradient_y(i, j, k, grid, hydrostatic_pressure) = ∂yᶜᶠᶜ(i, j, k, grid, hydrostatic_pressure)
@inline hydrostatic_pressure_gradient_y(i, j, k, grid, ::Nothing) = zero(grid)

# Type aliases for major functions
const Advection = CustomKFO{<:typeof(div_𝐯v)}
const BuoyancyAcceleration = CustomKFO{<:typeof(y_dot_g_bᶜᶠᶜ)}
const CoriolisAcceleration = CustomKFO{<:typeof(y_f_cross_U)}
const PressureGradient = CustomKFO{<:typeof(hydrostatic_pressure_gradient_y)}
const ViscousDissipation = CustomKFO{<:typeof(∂ⱼ_τ₂ⱼ)}
const ImmersedViscousDissipation = CustomKFO{<:typeof(immersed_∂ⱼ_τ₂ⱼ)}
const TotalViscousDissipation = CustomKFO{<:typeof(total_∂ⱼ_τ₂ⱼ)}
const StokesShear = CustomKFO{<:typeof(y_curl_Uˢ_cross_U)}
const StokesTendency = CustomKFO{<:typeof(∂t_vˢ)}
const Forcing = KernelFunctionOperation{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any}
const Tendency = Union{CustomKFO{<:typeof(v_velocity_tendency)},
                            CustomKFO{<:typeof(hydrostatic_free_surface_v_velocity_tendency)}}

const VAdvection = Advection
const VBuoyancyAcceleration = BuoyancyAcceleration
const VCoriolisAcceleration = CoriolisAcceleration
const VPressureGradient = PressureGradient
const VViscousDissipation = ViscousDissipation
const VImmersedViscousDissipation = ImmersedViscousDissipation
const VTotalViscousDissipation = TotalViscousDissipation
const VStokesShear = StokesShear
const VStokesTendency = StokesTendency
const VForcing = Forcing
const VTendency = Tendency

#+++ Advection
"""
    $(SIGNATURES)

Calculate the advection of v-momentum as

    ADV = ∂ⱼ (uⱼ v)

using Oceananigans' kernel [`div_𝐯v`.](https://clima.github.io/OceananigansDocumentation/stable/appendix/library/#Oceananigans.Advection.div_𝐯v-NTuple{7,%20Any})

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(grid);

julia> ADV = VMomentumEquation.Advection(model)
KernelFunctionOperation at (Center, Face, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: div_𝐯v (generic function with 10 methods)
└── arguments: ("Centered", "NamedTuple", "Field")
```
"""
function Advection(model, u, v, w, advection_scheme; location = (Center, Face, Center))
    validate_location(location, "Advection", (Center, Face, Center))
    total_velocities = (; u, v, w)
    return KernelFunctionOperation{Center, Face, Center}(div_𝐯v, model.grid, advection_scheme, total_velocities, v)
end

Advection(model; kwargs...)                              = Advection(model, model.velocities..., model.advection; kwargs...)
Advection(model::HydrostaticFreeSurfaceModel; kwargs...) = Advection(model, model.velocities..., model.advection.momentum; kwargs...)
#---

#+++ Buoyancy acceleration
"""
    $(SIGNATURES)

Calculate the buoyancy acceleration in the y-direction as

    BUOY = ĝᵧ b

where ĝᵧ is the y-component of the gravitational unit vector and b is the buoyancy.

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b);

julia> BUOY = VMomentumEquation.BuoyancyAcceleration(model)
KernelFunctionOperation at (Center, Face, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: y_dot_g_bᶜᶠᶜ (generic function with 10 methods)
└── arguments: ("BuoyancyForce", "NamedTuple")
```
"""
function BuoyancyAcceleration(model, buoyancy, tracers; location = (Center, Face, Center))
    validate_location(location, "BuoyancyAcceleration", (Center, Face, Center))
    return KernelFunctionOperation{Center, Face, Center}(y_dot_g_bᶜᶠᶜ, model.grid, buoyancy, tracers)
end

BuoyancyAcceleration(model; kwargs...) = BuoyancyAcceleration(model, model.buoyancy, model.tracers; kwargs...)
#---

#+++ Coriolis acceleration
"""
    $(SIGNATURES)

Calculate the Coriolis acceleration in the y-direction as

    COR = - (f × u)ᵧ

where f is the Coriolis parameter vector and u is the velocity vector.

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(grid; coriolis=FPlane(f=1e-4));

julia> COR = VMomentumEquation.CoriolisAcceleration(model)
KernelFunctionOperation at (Center, Face, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: y_f_cross_U (generic function with 10 methods)
└── arguments: ("FPlane", "NamedTuple")
```
"""
function CoriolisAcceleration(model, coriolis, velocities; location = (Center, Face, Center))
    validate_location(location, "CoriolisAcceleration", (Center, Face, Center))
    return KernelFunctionOperation{Center, Face, Center}(y_f_cross_U, model.grid, coriolis, velocities)
end

CoriolisAcceleration(model; kwargs...) = CoriolisAcceleration(model, model.coriolis, model.velocities; kwargs...)
#---

#+++ Pressure gradient
"""
    $(SIGNATURES)

Calculate the hydrostatic pressure gradient force in the y-direction as

    PRES = - ∂p/∂y

where p is the hydrostatic pressure anomaly.

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(grid);

julia> PRES = VMomentumEquation.PressureGradient(model)
KernelFunctionOperation at (Center, Face, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: hydrostatic_pressure_gradient_y (generic function with 2 methods)
└── arguments: ("Nothing",)
```
"""
function PressureGradient(model, hydrostatic_pressure; location = (Center, Face, Center))
    validate_location(location, "PressureGradient", (Center, Face, Center))
    return KernelFunctionOperation{Center, Face, Center}(hydrostatic_pressure_gradient_y, model.grid, hydrostatic_pressure)
end

function PressureGradient(model; kwargs...)
    # Both NH and HFS keep the hydrostatic pressure anomaly (`pHY′`) under different field names:
    # NH has `model.pressures.pHY′` (NamedTuple of pNHS, pHY′);
    # HFS has `model.pressure.pHY′` (NamedTuple with just pHY′). Pull whichever is present.
    hydrostatic_pressure = if hasfield(typeof(model), :pressures)
        model.pressures.pHY′
    elseif hasfield(typeof(model), :pressure)
        model.pressure.pHY′
    else
        nothing
    end
    return PressureGradient(model, hydrostatic_pressure; kwargs...)
end
#---

#+++ Viscous dissipation
"""
    $(SIGNATURES)

Calculate the viscous dissipation term (excluding immersed boundaries) as

    VISC = - ∂ⱼ τ₂ⱼ,

where τ₂ⱼ is the viscous stress tensor for the y-momentum equation.

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(grid);

julia> VISC = VMomentumEquation.ViscousDissipation(model)
KernelFunctionOperation at (Center, Face, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: ∂ⱼ_τ₂ⱼ (generic function with 10 methods)
└── arguments: ("Nothing", "Nothing", "Clock", "NamedTuple", "Nothing")
```
"""
function ViscousDissipation(model, closure, diffusivities, clock, model_fields, buoyancy; location = (Center, Face, Center))
    validate_location(location, "ViscousDissipation", (Center, Face, Center))
    return KernelFunctionOperation{Center, Face, Center}(∂ⱼ_τ₂ⱼ, model.grid, closure, diffusivities, clock, model_fields, buoyancy)
end

ViscousDissipation(model; kwargs...) = ViscousDissipation(model, model.closure, model.closure_fields, model.clock, fields(model), model.buoyancy; kwargs...)

"""
    $(SIGNATURES)

Calculate the viscous dissipation term due to immersed boundaries as

    VISC = - ∂ⱼ τ₂ⱼ,

where τ₂ⱼ is the immersed boundary viscous stress tensor for the y-momentum equation.

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(grid);

julia> VISC = VMomentumEquation.ImmersedViscousDissipation(model)
KernelFunctionOperation at (Center, Face, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: immersed_∂ⱼ_τ₂ⱼ (generic function with 2 methods)
└── arguments: ("NamedTuple", "Nothing", "Nothing", "Nothing", "Clock", "NamedTuple")
```
"""
function ImmersedViscousDissipation(model, velocities, v_immersed_bc, closure, diffusivities, clock, model_fields; location = (Center, Face, Center))
    validate_location(location, "ImmersedViscousDissipation", (Center, Face, Center))
    return KernelFunctionOperation{Center, Face, Center}(immersed_∂ⱼ_τ₂ⱼ, model.grid, velocities, v_immersed_bc, closure, diffusivities, clock, model_fields)
end

function ImmersedViscousDissipation(model; kwargs...)
    v_immersed_bc = model.velocities.v.boundary_conditions.immersed
    return ImmersedViscousDissipation(model, model.velocities, v_immersed_bc, model.closure, model.closure_fields, model.clock, fields(model); kwargs...)
end

"""
    $(SIGNATURES)

Calculate the total viscous dissipation term as

    VISC = - ∂ⱼ τ₂ⱼ - ∂ⱼ τ₂ⱼ_immersed,

where τ₂ⱼ is the interior viscous stress tensor and τ₂ⱼ_immersed is the immersed boundary
viscous stress tensor for the y-momentum equation.

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(grid);

julia> VISC = VMomentumEquation.TotalViscousDissipation(model)
KernelFunctionOperation at (Center, Face, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: total_∂ⱼ_τ₂ⱼ (generic function with 1 method)
└── arguments: ("NamedTuple", "Nothing", "Nothing", "Nothing", "Clock", "NamedTuple", "Nothing")
```
"""
function TotalViscousDissipation(model, velocities, v_immersed_bc, closure, diffusivities, clock, model_fields, buoyancy; location = (Center, Face, Center))
    validate_location(location, "TotalViscousDissipation", (Center, Face, Center))
    return KernelFunctionOperation{Center, Face, Center}(total_∂ⱼ_τ₂ⱼ, model.grid, velocities, v_immersed_bc, closure, diffusivities, clock, model_fields, buoyancy)
end

function TotalViscousDissipation(model; kwargs...)
    v_immersed_bc = model.velocities.v.boundary_conditions.immersed
    return TotalViscousDissipation(model, model.velocities, v_immersed_bc, model.closure, model.closure_fields, model.clock, fields(model), model.buoyancy; kwargs...)
end
#---

#+++ Stokes drift terms
"""
    $(SIGNATURES)

Calculate the Stokes shear term as

    STOKES_SHEAR = ((∇ × uˢ) × u)ᵧ

where uˢ is the Stokes drift velocity and u is the velocity vector.

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(grid);

julia> STOKES = VMomentumEquation.StokesShear(model)
KernelFunctionOperation at (Center, Face, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: y_curl_Uˢ_cross_U (generic function with 10 methods)
└── arguments: ("Nothing", "NamedTuple", "Float64")
```
"""
function StokesShear(model, stokes_drift, velocities, time; location = (Center, Face, Center))
    validate_location(location, "StokesShear", (Center, Face, Center))
    return KernelFunctionOperation{Center, Face, Center}(y_curl_Uˢ_cross_U, model.grid, stokes_drift, velocities, time)
end

StokesShear(model::HydrostaticFreeSurfaceModel; kwargs...) = throw(ArgumentError("VMomentumEquation.StokesShear is not defined for HydrostaticFreeSurfaceModel: " *
                                                                                 "Stokes drift is not part of the hydrostatic free-surface model."))

StokesShear(model; kwargs...) = StokesShear(model, model.stokes_drift, model.velocities, model.clock.time; kwargs...)

"""
    $(SIGNATURES)

Calculate the Stokes tendency term as

    STOKES_TEND = ∂vˢ/∂t

where vˢ is the y-component of the Stokes drift velocity.

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(grid);

julia> STOKES = VMomentumEquation.StokesTendency(model)
KernelFunctionOperation at (Center, Face, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: ∂t_vˢ (generic function with 10 methods)
└── arguments: ("Nothing", "Float64")
```
"""
function StokesTendency(model, stokes_drift, time; location = (Center, Face, Center))
    validate_location(location, "StokesTendency", (Center, Face, Center))
    return KernelFunctionOperation{Center, Face, Center}(∂t_vˢ, model.grid, stokes_drift, time)
end

StokesTendency(model::HydrostaticFreeSurfaceModel; kwargs...) = throw(ArgumentError("VMomentumEquation.StokesTendency is not defined for HydrostaticFreeSurfaceModel: " *
                                                                                    "Stokes drift is not part of the hydrostatic free-surface model."))

StokesTendency(model; kwargs...) = StokesTendency(model, model.stokes_drift, model.clock.time; kwargs...)
#---

#+++ Forcing
"""
    $(SIGNATURES)

Calculate the forcing term `Fᵛ` on the y-momentum equation for `model`.

`Forcing` is a type alias for the generic `KernelFunctionOperation` (no narrowing
on the kernel function), so a constructor on `Forcing(model)` would clash across the
U/V/W modules. To disambiguate, the V-momentum convenience constructor takes an
explicit `Val(:v)` tag.

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(grid);

julia> FORC = VMomentumEquation.Forcing(model, Val(:v))
KernelFunctionOperation at (Center, Face, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: Returns (generic function with 1 method)
└── arguments: ("Clock", "NamedTuple")
```
"""
function Forcing(model, forcing_func, clock, model_fields, ::Val{:v}; location = (Center, Face, Center))
    validate_location(location, "Forcing", (Center, Face, Center))
    return KernelFunctionOperation{Center, Face, Center}(forcing_func, model.grid, clock, model_fields)
end

Forcing(model, ::Val{:v}; kwargs...) = Forcing(model, model.forcing.v, model.clock, fields(model), Val(:v); kwargs...)
#---

#+++ Total tendency
"""
    $(SIGNATURES)

Calculate the total tendency of the v-momentum equation as computed by Oceananigans.

For NonhydrostaticModel, this includes:
- Advection: -∇⋅(𝐯v)
- Background advection terms
- Buoyancy: ĝᵧ b
- Coriolis: -(f × u)ᵧ
- Pressure gradient: -∂p/∂y
- Viscous dissipation: -∇⋅τ₂
- Immersed viscous dissipation
- Stokes shear: ((∇ × uˢ) × u)ᵧ
- Stokes tendency: ∂vˢ/∂t
- Forcing: Fᵛ

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(grid);

julia> TEND = VMomentumEquation.Tendency(model)
KernelFunctionOperation at (Center, Face, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: v_velocity_tendency (generic function with 1 method)
└── arguments: ("Centered", "Nothing", "Nothing", "Nothing", "Nothing", "Nothing", "Oceananigans.Models.NonhydrostaticModels.BackgroundFields", "NamedTuple", "NamedTuple", "NamedTuple", "Nothing", "Nothing", "Clock", "Returns")
```
"""
function Tendency(model::HydrostaticFreeSurfaceModel, advection_scheme, coriolis, closure, v_immersed_bc, velocities, free_surface, tracers, buoyancy, closure_fields, hydrostatic_pressure_anomaly, auxiliary_fields, vertical_coordinate, clock, forcing_func; location = (Center, Face, Center))
    validate_location(location, "Tendency", (Center, Face, Center))
    return KernelFunctionOperation{Center, Face, Center}(hydrostatic_free_surface_v_velocity_tendency, model.grid, advection_scheme, coriolis, closure, v_immersed_bc, velocities, free_surface, tracers, buoyancy, closure_fields, hydrostatic_pressure_anomaly, auxiliary_fields, vertical_coordinate, clock, forcing_func)
end

function Tendency(model, advection_scheme, coriolis, stokes_drift, closure, v_immersed_bc, buoyancy, background_fields, velocities, tracers, auxiliary_fields, diffusivities, hydrostatic_pressure, clock, forcing_func; location = (Center, Face, Center))
    validate_location(location, "Tendency", (Center, Face, Center))
    return KernelFunctionOperation{Center, Face, Center}(v_velocity_tendency, model.grid, advection_scheme, coriolis, stokes_drift, closure, v_immersed_bc, buoyancy, background_fields, velocities, tracers, auxiliary_fields, diffusivities, hydrostatic_pressure, clock, forcing_func)
end

function Tendency(model; kwargs...)
    v_immersed_bc = model.velocities.v.boundary_conditions.immersed

    if model isa HydrostaticFreeSurfaceModel
        return Tendency(model, model.advection.momentum, model.coriolis, model.closure, v_immersed_bc, model.velocities, model.free_surface, model.tracers, model.buoyancy, model.closure_fields, model.pressure.pHY′, model.auxiliary_fields, model.vertical_coordinate, model.clock, model.forcing.v; kwargs...)
    else
        return Tendency(model, model.advection, model.coriolis, model.stokes_drift, model.closure, v_immersed_bc, model.buoyancy, model.background_fields, model.velocities, model.tracers, model.auxiliary_fields, model.closure_fields, model.pressures.pHY′, model.clock, model.forcing.v; kwargs...)
    end
end
#---

end # module
