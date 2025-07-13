module TurbulentKineticEnergyEquation
using DocStringExtensions

export TurbulentKineticEnergy
export IsotropicDissipationRate, TurbulentKineticEnergyIsotropicDissipationRate
export XShearProductionRate, YShearProductionRate, ZShearProductionRate

using Oceananigans: NonhydrostaticModel, HydrostaticFreeSurfaceModel, fields
using Oceananigans.Operators
using Oceananigans.AbstractOperations
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Grids: Center, Face
using Oceananigans.Fields: ZeroField
using Oceananigans.Models.NonhydrostaticModels: u_velocity_tendency, v_velocity_tendency, w_velocity_tendency
using Oceananigans.Advection: div_𝐯u, div_𝐯v, div_𝐯w
using Oceananigans.TurbulenceClosures: viscous_flux_ux, viscous_flux_uy, viscous_flux_uz,
                                       viscous_flux_vx, viscous_flux_vy, viscous_flux_vz,
                                       viscous_flux_wx, viscous_flux_wy, viscous_flux_wz,
                                       ∂ⱼ_τ₁ⱼ, ∂ⱼ_τ₂ⱼ, ∂ⱼ_τ₃ⱼ
using Oceananigans.TurbulenceClosures: immersed_∂ⱼ_τ₁ⱼ, immersed_∂ⱼ_τ₂ⱼ, immersed_∂ⱼ_τ₃ⱼ
using Oceananigans.BuoyancyFormulations: x_dot_g_bᶠᶜᶜ, y_dot_g_bᶜᶠᶜ, z_dot_g_bᶜᶜᶠ

using Oceanostics: _νᶜᶜᶜ
using Oceanostics: validate_location, validate_dissipative_closure, perturbation_fields
using Oceanostics.KineticEnergyEquation: IsotropicKineticEnergyDissipationRate

# Some useful operators
@inline ψ²(i, j, k, grid, ψ) = @inbounds ψ[i, j, k]^2

@inline ψ′²(i, j, k, grid, ψ, ψ̄) = @inbounds (ψ[i, j, k] - ψ̄[i, j, k])^2
@inline ψ′²(i, j, k, grid, ψ, ψ̄::Number) = @inbounds (ψ[i, j, k] - ψ̄)^2

@inline fψ_plus_gφ²(i, j, k, grid, f, ψ, g, φ) = (f(i, j, k, grid, ψ) + g(i, j, k, grid, φ))^2

#++++ Turbulent kinetic energy
@inline turbulent_kinetic_energy_ccc(i, j, k, grid, u, v, w, U, V, W) = (ℑxᶜᵃᵃ(i, j, k, grid, ψ′², u, U) +
                                                                         ℑyᵃᶜᵃ(i, j, k, grid, ψ′², v, V) +
                                                                         ℑzᵃᵃᶜ(i, j, k, grid, ψ′², w, W)) / 2

const TurbulentKineticEnergy = KernelFunctionOperation{<:Any, <:Any, <:Any, <:Any, <:Any, <:typeof(turbulent_kinetic_energy_ccc)}

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
#------

#+++ Turbulent kinetic energy isotropic dissipation rate
"""
    $(SIGNATURES)

Calculate the Viscous Dissipation Rate, defined as

    ε = 2 ν S'ᵢⱼS'ᵢⱼ,

where S'ᵢⱼ is the strain rate tensor, for a fluid with an isotropic turbulence closure (i.e., a
turbulence closure where ν (eddy or not) is the same for all directions.
"""
@inline TurbulentKineticEnergyIsotropicDissipationRate(u, v, w, args...; U=ZeroField(), V=ZeroField(), W=ZeroField(), location = (Center, Center, Center)) =
    IsotropicKineticEnergyDissipationRate((u - U), (v - V), (w - W), args...; location)

const IsotropicDissipationRate = TurbulentKineticEnergyIsotropicDissipationRate
#---

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

const XShearProductionRate = KernelFunctionOperation{<:Any, <:Any, <:Any, <:Any, <:Any, <:typeof(shear_production_rate_x_ccc)}

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

const YShearProductionRate = KernelFunctionOperation{<:Any, <:Any, <:Any, <:Any, <:Any, <:typeof(shear_production_rate_y_ccc)}

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

const ZShearProductionRate = KernelFunctionOperation{<:Any, <:Any, <:Any, <:Any, <:Any, <:typeof(shear_production_rate_z_ccc)}

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