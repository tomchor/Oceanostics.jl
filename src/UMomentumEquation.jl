module UMomentumEquation
using DocStringExtensions

using Oceananigans: fields, Face, Center, KernelFunctionOperation, AbstractModel
using Oceananigans.Models: HydrostaticFreeSurfaceModel
using Oceananigans.Models.NonhydrostaticModels: u_velocity_tendency
using Oceananigans.Models.HydrostaticFreeSurfaceModels: hydrostatic_free_surface_u_velocity_tendency
using Oceananigans.Advection: div_𝐯u
using Oceananigans.BuoyancyFormulations: x_dot_g_bᶠᶜᶜ
using Oceananigans.Coriolis: x_f_cross_U
using Oceananigans.TurbulenceClosures: ∂ⱼ_τ₁ⱼ, immersed_∂ⱼ_τ₁ⱼ
using Oceananigans.StokesDrifts: x_curl_Uˢ_cross_U, ∂t_uˢ
using Oceananigans.Operators: ∂xᶠᶜᶜ

using Oceanostics: validate_location, CustomKFO

export Advection, BuoyancyAcceleration, CoriolisAcceleration, PressureGradient,
       ViscousDissipation, ImmersedViscousDissipation, TotalViscousDissipation,
       StokesShear, StokesTendency, Forcing, TotalTendency,
       UAdvection, UBuoyancyAcceleration, UCoriolisAcceleration, UPressureGradient,
       UViscousDissipation, UImmersedViscousDissipation, UTotalViscousDissipation,
       UStokesShear, UStokesTendency, UForcing, UTotalTendency

# Inline function for total viscous dissipation
@inline total_∂ⱼ_τ₁ⱼ(i, j, k, grid, velocities, u_immersed_bc, closure, diffusivities, clock, model_fields, buoyancy) =
    ∂ⱼ_τ₁ⱼ(i, j, k, grid, closure, diffusivities, clock, model_fields, buoyancy) +
    immersed_∂ⱼ_τ₁ⱼ(i, j, k, grid, velocities, u_immersed_bc, closure, diffusivities, clock, model_fields)

# Inline function for hydrostatic pressure gradient
@inline hydrostatic_pressure_gradient_x(i, j, k, grid, hydrostatic_pressure) = ∂xᶠᶜᶜ(i, j, k, grid, hydrostatic_pressure)
@inline hydrostatic_pressure_gradient_x(i, j, k, grid, ::Nothing) = zero(grid)

# Type aliases for major functions
const Advection = CustomKFO{<:typeof(div_𝐯u)}
const BuoyancyAcceleration = CustomKFO{<:typeof(x_dot_g_bᶠᶜᶜ)}
const CoriolisAcceleration = CustomKFO{<:typeof(x_f_cross_U)}
const PressureGradient = CustomKFO{<:typeof(hydrostatic_pressure_gradient_x)}
const ViscousDissipation = CustomKFO{<:typeof(∂ⱼ_τ₁ⱼ)}
const ImmersedViscousDissipation = CustomKFO{<:typeof(immersed_∂ⱼ_τ₁ⱼ)}
const TotalViscousDissipation = CustomKFO{<:typeof(total_∂ⱼ_τ₁ⱼ)}
const StokesShear = CustomKFO{<:typeof(x_curl_Uˢ_cross_U)}
const StokesTendency = CustomKFO{<:typeof(∂t_uˢ)}
const Forcing = KernelFunctionOperation{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any}
const TotalTendency = CustomKFO{<:typeof(u_velocity_tendency)}

# Aliases for consistency with TracerEquation naming
const UAdvection = Advection
const UBuoyancyAcceleration = BuoyancyAcceleration
const UCoriolisAcceleration = CoriolisAcceleration
const UPressureGradient = PressureGradient
const UViscousDissipation = ViscousDissipation
const UImmersedViscousDissipation = ImmersedViscousDissipation
const UTotalViscousDissipation = TotalViscousDissipation
const UStokesShear = StokesShear
const UStokesTendency = StokesTendency
const UForcing = Forcing
const UTotalTendency = TotalTendency

#+++ Advection
"""
    $(SIGNATURES)

Calculates the advection of u-momentum as

    ADV = ∂ⱼ (uⱼ u)

using Oceananigans' kernel [`div_𝐯u`.](https://clima.github.io/OceananigansDocumentation/stable/appendix/library/#Oceananigans.Advection.div_𝐯u-NTuple{7,%20Any})

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(; grid);

julia> ADV = UMomentumEquation.Advection(model)
KernelFunctionOperation at (Face, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: div_𝐯u (generic function with 10 methods)
└── arguments: ("Centered", "NamedTuple", "Field")
```
"""
function Advection(model, u, v, w, advection; location = (Face, Center, Center))
    validate_location(location, "Advection", (Face, Center, Center))
    total_velocities = (; u, v, w)
    return KernelFunctionOperation{Face, Center, Center}(div_𝐯u, model.grid, advection, total_velocities, u)
end

function Advection(model; kwargs...)
    return Advection(model, model.velocities..., model.advection; kwargs...)
end

function Advection(model::HydrostaticFreeSurfaceModel; kwargs...)
    return Advection(model, model.velocities..., model.advection.momentum; kwargs...)
end
#---

#+++ Buoyancy acceleration
"""
    $(SIGNATURES)

Calculates the buoyancy acceleration in the x-direction as

    BUOY = ĝₓ b

where ĝₓ is the x-component of the gravitational unit vector and b is the buoyancy.

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(; grid, buoyancy=BuoyancyTracer(), tracers=:b);

julia> BUOY = UMomentumEquation.BuoyancyAcceleration(model)
KernelFunctionOperation at (Face, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: x_dot_g_bᶠᶜᶜ (generic function with 10 methods)
└── arguments: ("BuoyancyTracer", "NamedTuple")
```
"""
function BuoyancyAcceleration(model, buoyancy, tracers; location = (Face, Center, Center))
    validate_location(location, "BuoyancyAcceleration", (Face, Center, Center))
    return KernelFunctionOperation{Face, Center, Center}(x_dot_g_bᶠᶜᶜ, model.grid, buoyancy, tracers)
end

function BuoyancyAcceleration(model; kwargs...)
    return BuoyancyAcceleration(model, model.buoyancy, model.tracers; kwargs...)
end
#---

#+++ Coriolis acceleration
"""
    $(SIGNATURES)

Calculates the Coriolis acceleration in the x-direction as

    COR = - f × u

where f is the Coriolis parameter vector and u is the velocity vector.

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(; grid, coriolis=FPlane(1e-4));

julia> COR = UMomentumEquation.CoriolisAcceleration(model)
KernelFunctionOperation at (Face, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: x_f_cross_U (generic function with 10 methods)
└── arguments: ("FPlane", "NamedTuple")
```
"""
function CoriolisAcceleration(model, coriolis, velocities; location = (Face, Center, Center))
    validate_location(location, "CoriolisAcceleration", (Face, Center, Center))
    return KernelFunctionOperation{Face, Center, Center}(x_f_cross_U, model.grid, coriolis, velocities)
end

function CoriolisAcceleration(model; kwargs...)
    return CoriolisAcceleration(model, model.coriolis, model.velocities; kwargs...)
end
#---

#+++ Pressure gradient
"""
    $(SIGNATURES)

Calculates the pressure gradient force in the x-direction as

    PRES = - ∂p/∂x

where p is the pressure field.

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(; grid);

julia> PRES = UMomentumEquation.PressureGradient(model)
KernelFunctionOperation at (Face, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: hydrostatic_pressure_gradient_x (generic function with 2 methods)
└── arguments: ("Nothing",)
```
"""
function PressureGradient(model, hydrostatic_pressure; location = (Face, Center, Center))
    validate_location(location, "PressureGradient", (Face, Center, Center))
    return KernelFunctionOperation{Face, Center, Center}(hydrostatic_pressure_gradient_x, model.grid, hydrostatic_pressure)
end

function PressureGradient(model; kwargs...)
    # For NonhydrostaticModel, hydrostatic_pressure is typically nothing
    # For HydrostaticFreeSurfaceModel, it would be the free surface
    hydrostatic_pressure = hasfield(typeof(model), :free_surface) ? model.free_surface : nothing
    return PressureGradient(model, hydrostatic_pressure; kwargs...)
end
#---

#+++ Viscous dissipation
"""
    $(SIGNATURES)

Calculates the viscous dissipation term (excluding immersed boundaries) as

    VISC = - ∂ⱼ τ₁ⱼ,

where τ₁ⱼ is the viscous stress tensor for the x-momentum equation.

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(; grid);

julia> VISC = UMomentumEquation.ViscousDissipation(model)
KernelFunctionOperation at (Face, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: ∂ⱼ_τ₁ⱼ (generic function with 10 methods)
└── arguments: ("Nothing", "Nothing", "Clock", "NamedTuple", "Nothing")
```
"""
function ViscousDissipation(model, closure, diffusivities, clock, model_fields, buoyancy; location = (Face, Center, Center))
    validate_location(location, "ViscousDissipation", (Face, Center, Center))
    return KernelFunctionOperation{Face, Center, Center}(∂ⱼ_τ₁ⱼ, model.grid, closure, diffusivities, clock, model_fields, buoyancy)
end

function ViscousDissipation(model; kwargs...)
    return ViscousDissipation(model, model.closure, model.diffusivity_fields, model.clock, fields(model), model.buoyancy; kwargs...)
end

"""
    $(SIGNATURES)

Calculates the viscous dissipation term due to immersed boundaries as

    VISC = - ∂ⱼ τ₁ⱼ,

where τ₁ⱼ is the immersed boundary viscous stress tensor for the x-momentum equation.

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(; grid);

julia> VISC = UMomentumEquation.ImmersedViscousDissipation(model)
KernelFunctionOperation at (Face, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: immersed_∂ⱼ_τ₁ⱼ (generic function with 2 methods)
└── arguments: ("NamedTuple", "Nothing", "Nothing", "Nothing", "Clock", "NamedTuple")
```
"""
function ImmersedViscousDissipation(model, velocities, u_immersed_bc, closure, diffusivities, clock, model_fields; location = (Face, Center, Center))
    validate_location(location, "ImmersedViscousDissipation", (Face, Center, Center))
    return KernelFunctionOperation{Face, Center, Center}(immersed_∂ⱼ_τ₁ⱼ, model.grid, velocities, u_immersed_bc, closure, diffusivities, clock, model_fields)
end

function ImmersedViscousDissipation(model; kwargs...)
    u_immersed_bc = model.velocities.u.boundary_conditions.immersed
    return ImmersedViscousDissipation(model, model.velocities, u_immersed_bc, model.closure, model.diffusivity_fields, model.clock, fields(model); kwargs...)
end

"""
    $(SIGNATURES)

Calculates the total viscous dissipation term as

    VISC = - ∂ⱼ τ₁ⱼ - ∂ⱼ τ₁ⱼ_immersed,

where τ₁ⱼ is the interior viscous stress tensor and τ₁ⱼ_immersed is the immersed boundary
viscous stress tensor for the x-momentum equation.

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(; grid);

julia> VISC = UMomentumEquation.TotalViscousDissipation(model)
KernelFunctionOperation at (Face, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: total_∂ⱼ_τ₁ⱼ (generic function with 1 method)
└── arguments: ("NamedTuple", "Nothing", "Nothing", "Nothing", "Clock", "NamedTuple", "Nothing")
```
"""
function TotalViscousDissipation(model, velocities, u_immersed_bc, closure, diffusivities, clock, model_fields, buoyancy; location = (Face, Center, Center))
    validate_location(location, "TotalViscousDissipation", (Face, Center, Center))
    return KernelFunctionOperation{Face, Center, Center}(total_∂ⱼ_τ₁ⱼ, model.grid, velocities, u_immersed_bc, closure, diffusivities, clock, model_fields, buoyancy)
end

function TotalViscousDissipation(model; kwargs...)
    u_immersed_bc = model.velocities.u.boundary_conditions.immersed
    return TotalViscousDissipation(model, model.velocities, u_immersed_bc, model.closure, model.diffusivity_fields, model.clock, fields(model), model.buoyancy; kwargs...)
end
#---

#+++ Stokes drift terms
"""
    $(SIGNATURES)

Calculates the Stokes shear term as

    STOKES_SHEAR = (∇ × uˢ) × u

where uˢ is the Stokes drift velocity and u is the velocity vector.

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(; grid);

julia> STOKES = UMomentumEquation.StokesShear(model)
KernelFunctionOperation at (Face, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: x_curl_Uˢ_cross_U (generic function with 10 methods)
└── arguments: ("Nothing", "NamedTuple", "Float64")
```
"""
function StokesShear(model, stokes_drift, velocities, time; location = (Face, Center, Center))
    validate_location(location, "StokesShear", (Face, Center, Center))
    return KernelFunctionOperation{Face, Center, Center}(x_curl_Uˢ_cross_U, model.grid, stokes_drift, velocities, time)
end

function StokesShear(model; kwargs...)
    return StokesShear(model, model.stokes_drift, model.velocities, model.clock.time; kwargs...)
end

"""
    $(SIGNATURES)

Calculates the Stokes tendency term as

    STOKES_TEND = ∂uˢ/∂t

where uˢ is the Stokes drift velocity.

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(; grid);

julia> STOKES = UMomentumEquation.StokesTendency(model)
KernelFunctionOperation at (Face, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: ∂t_uˢ (generic function with 10 methods)
└── arguments: ("Nothing", "Float64")
```
"""
function StokesTendency(model, stokes_drift, time; location = (Face, Center, Center))
    validate_location(location, "StokesTendency", (Face, Center, Center))
    return KernelFunctionOperation{Face, Center, Center}(∂t_uˢ, model.grid, stokes_drift, time)
end

function StokesTendency(model; kwargs...)
    return StokesTendency(model, model.stokes_drift, model.clock.time; kwargs...)
end
#---

#+++ Forcing
"""
    $(SIGNATURES)

Calculate the forcing term `Fᵘ` on the x-momentum equation for `model`.

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(; grid);

julia> FORC = UMomentumEquation.Forcing(model)
KernelFunctionOperation at (Face, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: zeroforcing (generic function with 1 method)
└── arguments: ("Clock", "NamedTuple")
```
"""
function Forcing(model, forcing, clock, model_fields, ::Val{:u}; location = (Face, Center, Center))
    return KernelFunctionOperation{Face, Center, Center}(forcing, model.grid, clock, model_fields)
end

function Forcing(model; kwargs...)
    return Forcing(model, model.forcing.u, model.clock, fields(model), Val(:u); kwargs...)
end
#---

#+++ Total tendency
"""
    $(SIGNATURES)

Calculate the total tendency of the u-momentum equation as computed by Oceananigans.

For NonhydrostaticModel, this includes:
- Advection: -∇⋅(𝐯u)
- Background advection terms
- Buoyancy: ĝₓ b
- Coriolis: -f × u
- Pressure gradient: -∂p/∂x
- Viscous dissipation: -∇⋅τ₁
- Immersed viscous dissipation
- Stokes shear: (∇ × uˢ) × u
- Stokes tendency: ∂uˢ/∂t
- Forcing: Fᵘ

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(; grid);

julia> TEND = UMomentumEquation.TotalTendency(model)
KernelFunctionOperation at (Face, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: u_velocity_tendency (generic function with 1 method)
└── arguments: ("Centered", "Nothing", "Nothing", "Nothing", "Nothing", "Nothing", "NamedTuple", "NamedTuple", "NamedTuple", "Nothing", "Nothing", "Clock", "zeroforcing")
```
"""
function TotalTendency(model::HydrostaticFreeSurfaceModel, advection, coriolis, stokes_drift, closure, u_immersed_bc, buoyancy, background_fields, velocities, tracers, auxiliary_fields, diffusivities, free_surface, clock, forcing; location = (Face, Center, Center))
    validate_location(location, "TotalTendency", (Face, Center, Center))
    return KernelFunctionOperation{Face, Center, Center}(hydrostatic_free_surface_u_velocity_tendency, model.grid, advection, coriolis, stokes_drift, closure, u_immersed_bc, buoyancy, background_fields, velocities, tracers, auxiliary_fields, diffusivities, free_surface, clock, forcing)
end

function TotalTendency(model, advection, coriolis, stokes_drift, closure, u_immersed_bc, buoyancy, background_fields, velocities, tracers, auxiliary_fields, diffusivities, hydrostatic_pressure, clock, forcing; location = (Face, Center, Center))
    validate_location(location, "TotalTendency", (Face, Center, Center))
    return KernelFunctionOperation{Face, Center, Center}(u_velocity_tendency, model.grid, advection, coriolis, stokes_drift, closure, u_immersed_bc, buoyancy, background_fields, velocities, tracers, auxiliary_fields, diffusivities, hydrostatic_pressure, clock, forcing)
end

function TotalTendency(model; kwargs...)
    u_immersed_bc = model.velocities.u.boundary_conditions.immersed
    hydrostatic_pressure = hasfield(typeof(model), :free_surface) ? model.free_surface : nothing

    if model isa HydrostaticFreeSurfaceModel
        return TotalTendency(model, model.advection.momentum, model.coriolis, model.stokes_drift, model.closure, u_immersed_bc, model.buoyancy, model.background_fields, model.velocities, model.tracers, model.auxiliary_fields, model.diffusivity_fields, hydrostatic_pressure, model.clock, model.forcing.u; kwargs...)
    else
        return TotalTendency(model, model.advection, model.coriolis, model.stokes_drift, model.closure, u_immersed_bc, model.buoyancy, model.background_fields, model.velocities, model.tracers, model.auxiliary_fields, model.diffusivity_fields, hydrostatic_pressure, model.clock, model.forcing.u; kwargs...)
    end
end
#---

end # module
