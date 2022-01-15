module TKEBudgetTerms

export TurbulentKineticEnergy, KineticEnergy
export IsotropicViscousDissipationRate, IsotropicPseudoViscousDissipationRate
export AnisotropicPseudoViscousDissipationRate
export XPressureRedistribution, YPressureRedistribution, ZPressureRedistribution
export XShearProduction, YShearProduction, ZShearProduction

using Oceananigans
using Oceananigans.Operators
using Oceananigans.AbstractOperations
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Grids: Center, Face
using Oceananigans.Fields: ZeroField

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

function TurbulentKineticEnergy(model, u, v, w;
                                U = 0,
                                V = 0,
                                W = 0,
                                location = (Center, Center, Center))

    if location == (Center, Center, Center)
        return KernelFunctionOperation{Center, Center, Center}(turbulent_kinetic_energy_ccc, model.grid,
                                       computed_dependencies=(u, v, w, U, V, W))
    else
        error("TurbulentKineticEnergy only supports location = (Center, Center, Center) for now.")
    end
end

KineticEnergy(model, u, v, w; location = (Center, Center, Center), kwargs...) =
    TurbulentKineticEnergy(model, u, v, w; location, kwargs...)

TurbulentKineticEnergy(model; kwargs...) = TurbulentKineticEnergy(model, model.velocities...; kwargs...)
KineticEnergy(model; kwargs...) = KineticEnergy(model, model.velocities...; kwargs...)
#------


#++++ Energy dissipation rate for a fluid with isotropic viscosity
function isotropic_viscous_dissipation_rate_les_ccc(i, j, k, grid, u, v, w, ν)

    Σˣˣ² = ∂xᶜᵃᵃ(i, j, k, grid, u)^2
    Σʸʸ² = ∂yᵃᶜᵃ(i, j, k, grid, v)^2
    Σᶻᶻ² = ∂zᵃᵃᶜ(i, j, k, grid, w)^2

    Σˣʸ² = ℑxyᶜᶜᵃ(i, j, k, grid, fψ_plus_gφ², ∂yᵃᶠᵃ, u, ∂xᶠᵃᵃ, v) / 4
    Σˣᶻ² = ℑxzᶜᵃᶜ(i, j, k, grid, fψ_plus_gφ², ∂zᵃᵃᶠ, u, ∂xᶠᵃᵃ, w) / 4
    Σʸᶻ² = ℑyzᵃᶜᶜ(i, j, k, grid, fψ_plus_gφ², ∂zᵃᵃᶠ, v, ∂yᵃᶠᵃ, w) / 4

    return ν[i, j, k] * 2 * (Σˣˣ² + Σʸʸ² + Σᶻᶻ² + 2 * (Σˣʸ² + Σˣᶻ² + Σʸᶻ²))
end

function isotropic_viscous_dissipation_rate_dns_ccc(i, j, k, grid, u, v, w, ν)

    Σˣˣ² = ∂xᶜᵃᵃ(i, j, k, grid, u)^2
    Σʸʸ² = ∂yᵃᶜᵃ(i, j, k, grid, v)^2
    Σᶻᶻ² = ∂zᵃᵃᶜ(i, j, k, grid, w)^2

    Σˣʸ² = ℑxyᶜᶜᵃ(i, j, k, grid, fψ_plus_gφ², ∂yᵃᶠᵃ, u, ∂xᶠᵃᵃ, v) / 4
    Σˣᶻ² = ℑxzᶜᵃᶜ(i, j, k, grid, fψ_plus_gφ², ∂zᵃᵃᶠ, u, ∂xᶠᵃᵃ, w) / 4
    Σʸᶻ² = ℑyzᵃᶜᶜ(i, j, k, grid, fψ_plus_gφ², ∂zᵃᵃᶠ, v, ∂yᵃᶠᵃ, w) / 4

    return ν * 2 * (Σˣˣ² + Σʸʸ² + Σᶻᶻ² + 2 * (Σˣʸ² + Σˣᶻ² + Σʸᶻ²))
end


"""
Calculates the Viscous Dissipation Rate for a fluid with an isotropic turbulence closure (i.e., a 
turbulence closure where ν (eddy or not) is the same for all directions.
"""
function IsotropicViscousDissipationRate(model; U=nothing, V=nothing, W=nothing, 
                                         location = (Center, Center, Center))
    if location != (Center, Center, Center)
        error("IsotropicViscousDissipationRate only supports location = (Center, Center, Center) for now.")
    end

    u, v, w = model.velocities
    if U != nothing
        u -= U
    end

    if V != nothing
        v -= V
    end

    if W != nothing
        w -= W
    end

    if model.closure isa Oceananigans.TurbulenceClosures.AbstractEddyViscosityClosure
        νₑ = model.diffusivity_fields.νₑ
        return KernelFunctionOperation{Center, Center, Center}(isotropic_viscous_dissipation_rate_les_ccc, model.grid;
                                                               computed_dependencies=(u, v, w, νₑ))

    elseif model.closure == nothing
        error("Trying to calculate a TKE viscous dissipation rate with `model.closure==nothing`.")

    else
        ν = model.closure.ν
        return KernelFunctionOperation{Center, Center, Center}(isotropic_viscous_dissipation_rate_dns_ccc, model.grid;
                                                               computed_dependencies=(u, v, w), parameters=ν)
    end


end




function isotropic_pseudo_viscous_dissipation_rate_les_ccc(i, j, k, grid, u, v, w, ν)
    ddx² = ∂xᶜᵃᵃ(i, j, k, grid, ψ², u) + ℑxyᶜᶜᵃ(i, j, k, grid, fψ², ∂xᶠᵃᵃ, v) + ℑxzᶜᵃᶜ(i, j, k, grid, fψ², ∂xᶠᵃᵃ, w)
    ddy² = ℑxyᶜᶜᵃ(i, j, k, grid, fψ², ∂yᵃᶠᵃ, u) + ∂yᵃᶜᵃ(i, j, k, grid, ψ², v) + ℑyzᵃᶜᶜ(i, j, k, grid, fψ², ∂yᵃᶠᵃ, w)
    ddz² = ℑxzᶜᵃᶜ(i, j, k, grid, fψ², ∂zᵃᵃᶠ, u) + ℑyzᵃᶜᶜ(i, j, k, grid, fψ², ∂zᵃᵃᶠ, v) + ∂zᵃᵃᶜ(i, j, k, grid, ψ², w)
    return ν[i,j,k] * (ddx² + ddy² + ddz²)
end

function isotropic_pseudo_viscous_dissipation_rate_dns_ccc(i, j, k, grid, u, v, w, ν)
    ddx² = ∂xᶜᵃᵃ(i, j, k, grid, ψ², u) + ℑxyᶜᶜᵃ(i, j, k, grid, fψ², ∂xᶠᵃᵃ, v) + ℑxzᶜᵃᶜ(i, j, k, grid, fψ², ∂xᶠᵃᵃ, w)
    ddy² = ℑxyᶜᶜᵃ(i, j, k, grid, fψ², ∂yᵃᶠᵃ, u) + ∂yᵃᶜᵃ(i, j, k, grid, ψ², v) + ℑyzᵃᶜᶜ(i, j, k, grid, fψ², ∂yᵃᶠᵃ, w)
    ddz² = ℑxzᶜᵃᶜ(i, j, k, grid, fψ², ∂zᵃᵃᶠ, u) + ℑyzᵃᶜᶜ(i, j, k, grid, fψ², ∂zᵃᵃᶠ, v) + ∂zᵃᵃᶜ(i, j, k, grid, ψ², w)
    return ν * (ddx² + ddy² + ddz²)
end

function IsotropicPseudoViscousDissipationRate(model; U=nothing, V=nothing, W=nothing,
                                               location = (Center, Center, Center))
    if location != (Center, Center, Center)
        error("IsotropicPseudoViscousDissipationRate only supports location = (Center, Center, Center) for now.")
    end

    u, v, w = model.velocities
    if U != nothing
        u -= U
    end

    if V != nothing
        v -= V
    end

    if W != nothing
        w -= W
    end

    if model.closure isa Oceananigans.TurbulenceClosures.AbstractEddyViscosityClosure
        νₑ = model.diffusivity_fields.νₑ
        return KernelFunctionOperation{Center, Center, Center}(isotropic_pseudo_viscous_dissipation_rate_les_ccc, model.grid;
                                       computed_dependencies=(u, v, w, νₑ))
    elseif model.closure == nothing
        error("Trying to calculate a TKE pseudo viscous dissipation rate with `model.closure==nothing`.")

    else
        ν = model.closure.ν
        return KernelFunctionOperation{Center, Center, Center}(isotropic_pseudo_viscous_dissipation_rate_dns_ccc, model.grid;
                                                               computed_dependencies=(u, v, w), parameters=ν)
    end

end
#------


#+++++ Energy dissipation rate for a fluid with constant anisotropic viscosity (closure)
function anisotropic_pseudo_viscous_dissipation_rate_ccc(i, j, k, grid, u, v, w, params)

    ddx² = ∂xᶜᵃᵃ(i, j, k, grid, ψ², u) + ℑxyᶜᶜᵃ(i, j, k, grid, fψ², ∂xᶠᵃᵃ, v) + ℑxzᶜᵃᶜ(i, j, k, grid, fψ², ∂xᶠᵃᵃ, w)
    ddy² = ℑxyᶜᶜᵃ(i, j, k, grid, fψ², ∂yᵃᶠᵃ, u) + ∂yᵃᶜᵃ(i, j, k, grid, ψ², v) + ℑyzᵃᶜᶜ(i, j, k, grid, fψ², ∂yᵃᶠᵃ, w)
    ddz² = ℑxzᶜᵃᶜ(i, j, k, grid, fψ², ∂zᵃᵃᶠ, u) + ℑyzᵃᶜᶜ(i, j, k, grid, fψ², ∂zᵃᵃᶠ, v) + ∂zᵃᵃᶜ(i, j, k, grid, ψ², w)

    return params.νx*ddx² + params.νy*ddy² + params.νz*ddz²
end

function AnisotropicPseudoViscousDissipationRate(model; U=nothing, V=nothing, W=nothing, 
                                                 location = (Center, Center, Center))

    if location != (Center, Center, Center)
        error("AnisotropicPseudoViscousDissipationRate only supports location = (Center, Center, Center) for now.")
    end

    u, v, w = model.velocities
    if U != nothing
        u -= U
    end

    if V != nothing
        v -= V
    end

    if W != nothing
        w -= W
    end

    νx = model.closure.νx
    νy = model.closure.νy
    νz = model.closure.νz
    return KernelFunctionOperation{Center, Center, Center}(anisotropic_pseudo_viscous_dissipation_rate_ccc, model.grid;
                                                           computed_dependencies=(u, v, w),
                                                           parameters=(νx=νx, νy=νy, νz=νz,))
end
#-----


#++++ Pressure redistribution terms
function XPressureRedistribution(model)
    u, v, w = model.velocities
    p = sum(model.pressures)
    return ∂x(u*p) # p is the total kinematic pressure (there's no need to ρ₀)
end

function YPressureRedistribution(model)
    u, v, w = model.velocities
    p = sum(model.pressures)
    return ∂y(v*p) # p is the total kinematic pressure (there's no need to ρ₀)
end

function ZPressureRedistribution(model)
    u, v, w = model.velocities
    p = sum(model.pressures)
    return ∂z(w*p) # p is the total kinematic pressure (there's no need to ρ₀)
end
#----


#++++ Shear production terms
function shear_production_x_ccc(i, j, k, grid, u, v, w, U, V, W)
    u_int = ℑxᶜᵃᵃ(i, j, k, grid, u) # F, C, C  → C, C, C

    ∂xU = ∂xᶜᵃᵃ(i, j, k, grid, U) # F, C, C  → C, C, C
    uu = ℑxᶜᵃᵃ(i, j, k, grid, ψ², u)
    uu∂xU = uu * ∂xU

    ∂xV = ℑxyᶜᶜᵃ(i, j, k, grid, ∂xᶠᵃᵃ, V) # C, F, C  → F, F, C  → C, C, C
    vu = ℑyᵃᶜᵃ(i, j, k, grid, v) * u_int
    vu∂xV = vu * ∂xV

    ∂xW = ℑxzᶜᵃᶜ(i, j, k, grid, ∂xᶠᵃᵃ, W) # C, C, F  → F, C, F  → C, C, C
    wu = ℑzᵃᵃᶜ(i, j, k, grid, w) * u_int
    wu∂xW = wu * ∂xW

    return -(uu∂xU + vu∂xV + wu∂xW)
end

function XShearProduction(model, u, v, w, U, V, W; location = (Center, Center, Center))
    if location == (Center, Center, Center)
        return KernelFunctionOperation{Center, Center, Center}(shear_production_x_ccc, model.grid;
                                       computed_dependencies=(u, v, w, U, V, W))
    else
        error("XShearProduction only supports location = (Center, Center, Center) for now.")
    end
end


function shear_production_y_ccc(i, j, k, grid, u, v, w, U, V, W)
    v_int = ℑyᵃᶜᵃ(i, j, k, grid, v) # C, F, C  → C, C, C

    ∂yU = ℑxyᶜᶜᵃ(i, j, k, grid, ∂yᵃᶠᵃ, U) # F, C, C  → F, F, C  → C, C, C
    uv = ℑxᶜᵃᵃ(i, j, k, grid, u) * v_int
    uv∂yU = uv * ∂yU

    ∂yV = ∂yᵃᶜᵃ(i, j, k, grid, V)
    vv = ℑyᵃᶜᵃ(i, j, k, grid, ψ², v) # C, F, C  → C, C, C
    vv∂yV = vv * ∂yV

    ∂yW = ℑyzᵃᶜᶜ(i, j, k, grid, ∂yᵃᶠᵃ, W) # C, C, F  → C, F, F  → C, C, C
    wv = ℑzᵃᵃᶜ(i, j, k, grid, w) * v_int
    wv∂yW = wv * ∂yW

    return -(uv∂yU + vv∂yV + wv∂yW)
end

function YShearProduction(model, u, v, w, U, V, W; location = (Center, Center, Center))
    if location == (Center, Center, Center)
        return KernelFunctionOperation{Center, Center, Center}(shear_production_y_ccc, model.grid;
                                       computed_dependencies=(u, v, w, U, V, W))
    else
        error("YShearProduction only supports location = (Center, Center, Center) for now.")
    end
end


function shear_production_z_ccc(i, j, k, grid, u, v, w, U, V, W)
    w_int = ℑzᵃᵃᶜ(i, j, k, grid, w) # C, C, F  → C, C, C

    ∂zU = ℑxzᶜᵃᶜ(i, j, k, grid, ∂zᵃᵃᶠ, U) # F, C, C  → F, C, F  → C, C, C
    uw = ℑxᶜᵃᵃ(i, j, k, grid, u) * w_int
    uw∂zU = uw * ∂zU

    ∂zV = ℑyzᵃᶜᶜ(i, j, k, grid, ∂zᵃᵃᶠ, V) # C, F, C  → C, F, F  → C, C, C
    vw = ℑyᵃᶜᵃ(i, j, k, grid, v) * w_int
    vw∂zV = vw * ∂zV

    ∂zW = ∂zᵃᵃᶜ(i, j, k, grid, W)
    ww = ℑzᵃᵃᶜ(i, j, k, grid, ψ², w) # C, C, F  → C, C, C
    ww∂zW = ww * ∂zW

    return - (uw∂zU + vw∂zV + ww∂zW)
end

function ZShearProduction(model, u, v, w, U, V, W; location = (Center, Center, Center))
    if location == (Center, Center, Center)
        return KernelFunctionOperation{Center, Center, Center}(shear_production_z_ccc, model.grid;
                                       computed_dependencies=(u, v, w, U, V, W))
    else
        error("ZShearProduction only supports location = (Center, Center, Center) for now.")
    end
end

ZShearProduction(model; U=ZeroField(), V=ZeroField(), W=ZeroField(), kwargs...) =
    ZShearProduction(model, model.velocities..., U, V, W; kwargs...)
#----

end # module
