module TKEBudgetTerms
using DocStringExtensions

export TurbulentKineticEnergy, KineticEnergy
export IsotropicViscousDissipationRate, IsotropicPseudoViscousDissipationRate
export XPressureRedistribution, YPressureRedistribution, ZPressureRedistribution
export XShearProductionRate, YShearProductionRate, ZShearProductionRate

using Oceananigans.Operators
using Oceananigans.AbstractOperations
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Grids: Center, Face
using Oceananigans.Fields: ZeroField
import Oceananigans.TurbulenceClosures: viscous_flux_ux, viscous_flux_uy, viscous_flux_uz, 
                                        viscous_flux_vx, viscous_flux_vy, viscous_flux_vz,
                                        viscous_flux_wx, viscous_flux_wy, viscous_flux_wz

using Oceanostics: _νᶜᶜᶜ
using Oceanostics: validate_location, validate_dissipative_closure

# Some useful operators
@inline ψ²(i, j, k, grid, ψ) = @inbounds ψ[i, j, k]^2

@inline ψ′²(i, j, k, grid, ψ, Ψ) = @inbounds (ψ[i, j, k] - Ψ[i, j, k])^2
@inline ψ′²(i, j, k, grid, ψ, Ψ::Number) = @inbounds (ψ[i, j, k] - Ψ)^2

@inline fψ²(i, j, k, grid, f, ψ) = @inbounds f(i, j, k, grid, ψ)^2

@inline fψ_plus_gφ²(i, j, k, grid, f, ψ, g, φ) = @inbounds (f(i, j, k, grid, ψ) + g(i, j, k, grid, φ))^2

#++++ Turbulent kinetic energy
@inline function turbulent_kinetic_energy_ccc(i, j, k, grid, u, v, w, U, V, W)
    return (ℑxᶜᵃᵃ(i, j, k, grid, ψ′², u, U) +
            ℑyᵃᶜᵃ(i, j, k, grid, ψ′², v, V) +
            ℑzᵃᵃᶜ(i, j, k, grid, ψ′², w, W)) / 2
end

"""
    $(SIGNATURES)

Calculate the turbulent kinetic energy of `model` manually specifying `u`, `v`, `w` and optionally
background velocities `U`, `V` and `W`.
"""
function TurbulentKineticEnergy(model, u, v, w; U=0, V=0, W=0, location = (Center, Center, Center))
    validate_location(location, "TurbulentKineticEnergy")
    return KernelFunctionOperation{Center, Center, Center}(turbulent_kinetic_energy_ccc, model.grid,
                                                           u, v, w, U, V, W)
end

"""
    $(SIGNATURES)

Calculate the turbulent kinetic energy of `model`.
"""
TurbulentKineticEnergy(model; kwargs...) = TurbulentKineticEnergy(model, model.velocities...; kwargs...)

"""
    $(SIGNATURES)

Calculate the kinetic energy of `model` manually specifying `u`, `v` and `w`.
"""
KineticEnergy(model, u, v, w; location = (Center, Center, Center), kwargs...) =
    TurbulentKineticEnergy(model, u, v, w; location, kwargs...)

"""
    $(SIGNATURES)
    
Calculate the kinetic energy of `model`.
"""
KineticEnergy(model; kwargs...) = KineticEnergy(model, model.velocities...; kwargs...)
#------

#+++ Energy dissipation rate for a fluid with isotropic viscosity
@inline function isotropic_viscous_dissipation_rate_ccc(i, j, k, grid, u, v, w, p)

    Σˣˣ² = ∂xᶜᶜᶜ(i, j, k, grid, u)^2
    Σʸʸ² = ∂yᶜᶜᶜ(i, j, k, grid, v)^2
    Σᶻᶻ² = ∂zᶜᶜᶜ(i, j, k, grid, w)^2

    Σˣʸ² = ℑxyᶜᶜᵃ(i, j, k, grid, fψ_plus_gφ², ∂yᶠᶠᶜ, u, ∂xᶠᶠᶜ, v) / 4
    Σˣᶻ² = ℑxzᶜᵃᶜ(i, j, k, grid, fψ_plus_gφ², ∂zᶠᶜᶠ, u, ∂xᶠᶜᶠ, w) / 4
    Σʸᶻ² = ℑyzᵃᶜᶜ(i, j, k, grid, fψ_plus_gφ², ∂zᶜᶠᶠ, v, ∂yᶜᶠᶠ, w) / 4

    ν = _νᶜᶜᶜ(i, j, k, grid, p.closure, p.diffusivity_fields, p.clock)

    return 2ν * (Σˣˣ² + Σʸʸ² + Σᶻᶻ² + 2 * (Σˣʸ² + Σˣᶻ² + Σʸᶻ²))
end

"""
    $(SIGNATURES)

Calculate the Viscous Dissipation Rate, defined as

    ε = 2 ν SᵢⱼSᵢⱼ,

where Sᵢⱼ is the strain rate tensor, for a fluid with an isotropic turbulence closure (i.e., a 
turbulence closure where ν (eddy or not) is the same for all directions.
"""
function IsotropicViscousDissipationRate(model; U=0, V=0, W=0,
                                         location = (Center, Center, Center))

    validate_location(location, "IsotropicViscousDissipationRate")
    validate_dissipative_closure(model.closure)

    u, v, w = model.velocities

    parameters = (closure = model.closure,
                  diffusivity_fields = model.diffusivity_fields,
                  clock = model.clock)

    return KernelFunctionOperation{Center, Center, Center}(isotropic_viscous_dissipation_rate_ccc, model.grid,
                                                           (u - U), (v - V), (w - W), parameters)
end

for viscous_flux in (:viscous_flux_ux, :viscous_flux_uy, :viscous_flux_uz,
                     :viscous_flux_vx, :viscous_flux_vy, :viscous_flux_vz,
                     :viscous_flux_wx, :viscous_flux_wy, :viscous_flux_wz)
    @eval $viscous_flux(i, j, k, grid, closure_tuple::Tuple, diffusivity_fields, args...) = 
        sum($viscous_flux(i, j, k, closure, diffusivities, args...) for (closure, diffusivities) in zip(closure_tuple, diffusivity_fields))
end


# ∂ⱼuᵢ: Fᵢⱼ
δˣuFuxᶜᶜᶜ(i, j, k, grid, closure, K_fields, clo, fields, b) = - Axᶜᶜᶜ(i, j, k, grid) * δxᶜᵃᵃ(i, j, k, grid, fields.u) * viscous_flux_ux(i, j, k, grid, closure, K_fields, clo, fields, b)
δʸuFuyᶠᶠᶜ(i, j, k, grid, closure, K_fields, clo, fields, b) = - Ayᶠᶠᶜ(i, j, k, grid) * δyᵃᶠᵃ(i, j, k, grid, fields.u) * viscous_flux_uy(i, j, k, grid, closure, K_fields, clo, fields, b)
δᶻuFuzᶠᶜᶠ(i, j, k, grid, closure, K_fields, clo, fields, b) = - Azᶠᶜᶠ(i, j, k, grid) * δzᵃᵃᶠ(i, j, k, grid, fields.u) * viscous_flux_uz(i, j, k, grid, closure, K_fields, clo, fields, b)

δˣvFvxᶠᶠᶜ(i, j, k, grid, closure, K_fields, clo, fields, b) = - Axᶠᶠᶜ(i, j, k, grid) * δxᶠᵃᵃ(i, j, k, grid, fields.v) * viscous_flux_vx(i, j, k, grid, closure, K_fields, clo, fields, b)
δʸvFvyᶜᶜᶜ(i, j, k, grid, closure, K_fields, clo, fields, b) = - Ayᶜᶜᶜ(i, j, k, grid) * δyᵃᶜᵃ(i, j, k, grid, fields.v) * viscous_flux_vy(i, j, k, grid, closure, K_fields, clo, fields, b)
δᶻvFvzᶜᶠᶠ(i, j, k, grid, closure, K_fields, clo, fields, b) = - Azᶜᶠᶠ(i, j, k, grid) * δzᵃᵃᶠ(i, j, k, grid, fields.v) * viscous_flux_vz(i, j, k, grid, closure, K_fields, clo, fields, b)

δˣwFwxᶠᶜᶠ(i, j, k, grid, closure, K_fields, clo, fields, b) = - Axᶠᶜᶠ(i, j, k, grid) * δxᶠᵃᵃ(i, j, k, grid, fields.w) * viscous_flux_wx(i, j, k, grid, closure, K_fields, clo, fields, b)
δʸwFwyᶜᶠᶠ(i, j, k, grid, closure, K_fields, clo, fields, b) = - Ayᶜᶠᶠ(i, j, k, grid) * δyᵃᶠᵃ(i, j, k, grid, fields.w) * viscous_flux_wy(i, j, k, grid, closure, K_fields, clo, fields, b)
δᶻwFwzᶜᶜᶜ(i, j, k, grid, closure, K_fields, clo, fields, b) = - Azᶜᶜᶜ(i, j, k, grid) * δzᵃᵃᶜ(i, j, k, grid, fields.w) * viscous_flux_wz(i, j, k, grid, closure, K_fields, clo, fields, b)

@inline function isotropic_pseudo_viscous_dissipation_rate_ccc(i, j, k, grid, u, v, w, diffusivity_fields, fields, p)
return (δˣuFuxᶜᶜᶜ(i, j, k, grid,         p.closure, diffusivity_fields, p.clock, fields, p.buoyancy) + # C, C, C
        ℑxyᶜᶜᵃ(i, j, k, grid, δʸuFuyᶠᶠᶜ, p.closure, diffusivity_fields, p.clock, fields, p.buoyancy) + # F, F, C  → C, C, C
        ℑxzᶜᵃᶜ(i, j, k, grid, δᶻuFuzᶠᶜᶠ, p.closure, diffusivity_fields, p.clock, fields, p.buoyancy) + # F, C, F  → C, C, C

        ℑxyᶜᶜᵃ(i, j, k, grid, δˣvFvxᶠᶠᶜ, p.closure, diffusivity_fields, p.clock, fields, p.buoyancy) + # F, F, C  → C, C, C
        δʸvFvyᶜᶜᶜ(i, j, k, grid,         p.closure, diffusivity_fields, p.clock, fields, p.buoyancy) + # C, C, C
        ℑyzᵃᶜᶜ(i, j, k, grid, δᶻvFvzᶜᶠᶠ, p.closure, diffusivity_fields, p.clock, fields, p.buoyancy) + # C, F, F  → C, C, C

        ℑxzᶜᵃᶜ(i, j, k, grid, δˣwFwxᶠᶜᶠ, p.closure, diffusivity_fields, p.clock, fields, p.buoyancy) + # F, C, F  → C, C, C
        ℑyzᵃᶜᶜ(i, j, k, grid, δʸwFwyᶜᶠᶠ, p.closure, diffusivity_fields, p.clock, fields, p.buoyancy) + # C, F, F  → C, C, C
        δᶻwFwzᶜᶜᶜ(i, j, k, grid,         p.closure, diffusivity_fields, p.clock, fields, p.buoyancy)   # C, C, C
        ) / Vᶜᶜᶜ(i, j, k, grid) # This division by volume, coupled with the call to δ above, ensures a derivative operation
end

"""
    $(SIGNATURES)

Calculate the pseudo viscous Dissipation Rate, defined as

    ε = ν (∂uᵢ/∂xⱼ) (∂uᵢ/∂xⱼ)

for a fluid with an isotropic turbulence closure (i.e., a 
turbulence closure where ν (eddy or not) is the same for all directions.
"""
function IsotropicPseudoViscousDissipationRate(model; U=ZeroField(), V=ZeroField(), W=ZeroField(),
                                               location = (Center, Center, Center))

    validate_location(location, "IsotropicPseudoViscousDissipationRate")

    u, v, w = model.velocities

    parameters = (; model.closure, 
                  model.clock,
                  model.buoyancy)
    computed_dependencies = (u - U, v - V, w - W, 
                             model.diffusivity_fields,
                             fields(model))

    return KernelFunctionOperation{Center, Center, Center}(isotropic_pseudo_viscous_dissipation_rate_ccc, model.grid,
                                                           (u - U), (v - V), (w - W), parameters)
end
#---

#++++ Pressure redistribution terms
"""
    $(SIGNATURES)

Calculate the pressure redistribution term in the `x` direction. Here `u′` and `p′`
are the fluctuations around a mean.
"""
function XPressureRedistribution(model, u′, p′)
    return ∂x(u′*p′) # p is the total kinematic pressure (there's no need for ρ₀)
end

"""
    $(SIGNATURES)

Calculate the pressure redistribution term in the `y` direction. Here `v′` and `p′`
are the fluctuations around a mean.
"""
function YPressureRedistribution(model, v′, p′)
    return ∂y(v′*p′) # p is the total kinematic pressure (there's no need for ρ₀)
end

"""
    $(SIGNATURES)

Calculate the pressure redistribution term in the `z` direction. Here `w′` and `p′`
are the fluctuations around a mean.
"""
function ZPressureRedistribution(model, w′, p′)
    return ∂z(w′*p′) # p is the total kinematic pressure (there's no need for ρ₀)
end
#----

#++++ Shear production terms
@inline function shear_production_rate_x_ccc(i, j, k, grid, u, v, w, U, V, W)
    u_int = ℑxᶜᵃᵃ(i, j, k, grid, u) # F, C, C  → C, C, C

    ∂xU = ∂xᶜᶜᶜ(i, j, k, grid, U) # F, C, C  → C, C, C
    uu = ℑxᶜᵃᵃ(i, j, k, grid, ψ², u)
    uu∂xU = uu * ∂xU

    ∂xV = ℑxyᶜᶜᵃ(i, j, k, grid, ∂xᶠᶠᶜ, V) # C, F, C  → F, F, C  → C, C, C
    vu = ℑyᵃᶜᵃ(i, j, k, grid, v) * u_int
    vu∂xV = vu * ∂xV

    ∂xW = ℑxzᶜᵃᶜ(i, j, k, grid, ∂xᶠᶜᶠ, W) # C, C, F  → F, C, F  → C, C, C
    wu = ℑzᵃᵃᶜ(i, j, k, grid, w) * u_int
    wu∂xW = wu * ∂xW

    return -(uu∂xU + vu∂xV + wu∂xW)
end

"""
    $(SIGNATURES)

Calculate the shear production rate in the `model`'s `x` direction, considering velocities
`u`, `v`, `w` and background (or average) velocities `U`, `V` and `W`.
"""
function XShearProductionRate(model, u, v, w, U, V, W; location = (Center, Center, Center))
    validate_location(location, "XShearProductionRate")
    return KernelFunctionOperation{Center, Center, Center}(shear_production_rate_x_ccc, model.grid,
                                                           u, v, w, U, V, W)
end

"""
    $(SIGNATURES)

Calculate the shear production rate in the `model`'s `x` direction. At least one of the mean 
velocities `U`, `V` and `W` must be specified otherwise the output will be zero.
"""
function XShearProductionRate(model; U=0, V=0, W=0, kwargs...)
    u, v, w = model.velocities
    return XShearProductionRate(model, u-U, v-V, w-W, U, V, W; kwargs...)
end


@inline function shear_production_rate_y_ccc(i, j, k, grid, u, v, w, U, V, W)
    v_int = ℑyᵃᶜᵃ(i, j, k, grid, v) # C, F, C  → C, C, C

    ∂yU = ℑxyᶜᶜᵃ(i, j, k, grid, ∂yᶠᶠᶜ, U) # F, C, C  → F, F, C  → C, C, C
    uv = ℑxᶜᵃᵃ(i, j, k, grid, u) * v_int
    uv∂yU = uv * ∂yU

    ∂yV = ∂yᶜᶜᶜ(i, j, k, grid, V) # C, F, C  → C, C C
    vv = ℑyᵃᶜᵃ(i, j, k, grid, ψ², v) # C, F, C  → C, C, C
    vv∂yV = vv * ∂yV

    ∂yW = ℑyzᵃᶜᶜ(i, j, k, grid, ∂yᶜᶠᶠ, W) # C, C, F  → C, F, F  → C, C, C
    wv = ℑzᵃᵃᶜ(i, j, k, grid, w) * v_int
    wv∂yW = wv * ∂yW

    return -(uv∂yU + vv∂yV + wv∂yW)
end

"""
    $(SIGNATURES)

Calculate the shear production rate in the `model`'s `y` direction, considering velocities
`u`, `v`, `w` and background (or average) velocities `U`, `V` and `W`.
"""
function YShearProductionRate(model, u, v, w, U, V, W; location = (Center, Center, Center))
    validate_location(location, "YShearProductionRate")
    return KernelFunctionOperation{Center, Center, Center}(shear_production_rate_y_ccc, model.grid,
                                                           u, v, w, U, V, W)
end

"""
    $(SIGNATURES)

Calculate the shear production rate in the `model`'s `y` direction. At least one of the mean 
velocities `U`, `V` and `W` must be specified otherwise the output will be zero.
"""
function YShearProductionRate(model; U=0, V=0, W=0, kwargs...)
    u, v, w = model.velocities
    return YShearProductionRate(model, u-U, v-V, w-W, U, V, W; kwargs...)
end


@inline function shear_production_rate_z_ccc(i, j, k, grid, u, v, w, U, V, W)
    w_int = ℑzᵃᵃᶜ(i, j, k, grid, w) # C, C, F  → C, C, C

    ∂zU = ℑxzᶜᵃᶜ(i, j, k, grid, ∂zᶠᶜᶠ, U) # F, C, C  → F, C, F  → C, C, C
    uw = ℑxᶜᵃᵃ(i, j, k, grid, u) * w_int
    uw∂zU = uw * ∂zU

    ∂zV = ℑyzᵃᶜᶜ(i, j, k, grid, ∂zᶜᶠᶠ, V) # C, F, C  → C, F, F  → C, C, C
    vw = ℑyᵃᶜᵃ(i, j, k, grid, v) * w_int
    vw∂zV = vw * ∂zV

    ∂zW = ∂zᶜᶜᶜ(i, j, k, grid, W) # C, C, F  → C, C, C
    ww = ℑzᵃᵃᶜ(i, j, k, grid, ψ², w) # C, C, F  → C, C, C
    ww∂zW = ww * ∂zW

    return - (uw∂zU + vw∂zV + ww∂zW)
end

"""
    $(SIGNATURES)

Calculate the shear production rate in the `model`'s `z` direction, considering velocities
`u`, `v`, `w` and background (or average) velocities `U`, `V` and `W`.
"""
function ZShearProductionRate(model, u, v, w, U, V, W; location = (Center, Center, Center))
    validate_location(location, "ZShearProductionRate")
    return KernelFunctionOperation{Center, Center, Center}(shear_production_rate_z_ccc, model.grid,
                                                           u, v, w, U, V, W)
end

"""
    $(SIGNATURES)

Calculate the shear production rate in the `model`'s `z` direction. At least one of the mean 
velocities `U`, `V` and `W` must be specified otherwise the output will be zero.
"""
function ZShearProductionRate(model; U=0, V=0, W=0, kwargs...)
    u, v, w = model.velocities
    return ZShearProductionRate(model, u-U, v-V, w-W, U, V, W; kwargs...)
end
#----

end # module
