module TKEBudgetTerms
using DocStringExtensions

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
using Oceananigans.TurbulenceClosures: νᶜᶜᶜ, AbstractScalarDiffusivity, ThreeDimensionalFormulation

using Oceanostics: _νᶜᶜᶜ

# Right now, all kernels must be located at ccc
validate_location(location, type, valid_location=(Center, Center, Center)) =
    location != valid_location &&
        error("$type only supports location = $valid_location for now.")


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
function TurbulentKineticEnergy(model, u, v, w; U = 0, V = 0, W = 0, location = (Center, Center, Center))
    validate_location(location, "TurbulentKineticEnergy")
    return KernelFunctionOperation{Center, Center, Center}(turbulent_kinetic_energy_ccc, model.grid,
                                                           computed_dependencies=(u, v, w, U, V, W))
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

#####
##### Dissipation rates
#####

validate_dissipative_closure(closure) = error("Cannot calculate dissipation rate for $closure")
validate_dissipative_closure(::AbstractScalarDiffusivity{<:Any, ThreeDimensionalFormulation}) = nothing
validate_dissipative_closure(closure_tuple::Tuple) = Tuple(validate_dissipative_closure(c) for c in closure_tuple)

#++++ Energy dissipation rate for a fluid with isotropic viscosity
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

    return KernelFunctionOperation{Center, Center, Center}(isotropic_viscous_dissipation_rate_ccc, model.grid;
                                                           computed_dependencies=(u - U, v - V, w - W),
                                                           parameters)
end
#------

@inline function isotropic_pseudo_viscous_dissipation_rate_ccc(i, j, k, grid, u, v, w, p)
    ddx² = ∂xᶜᶜᶜ(i, j, k, grid, ψ², u) + ℑxyᶜᶜᵃ(i, j, k, grid, fψ², ∂xᶠᶠᶜ, v) + ℑxzᶜᵃᶜ(i, j, k, grid, fψ², ∂xᶠᶜᶠ, w)
    ddy² = ℑxyᶜᶜᵃ(i, j, k, grid, fψ², ∂yᶠᶠᶜ, u) + ∂yᶜᶜᶜ(i, j, k, grid, ψ², v) + ℑyzᵃᶜᶜ(i, j, k, grid, fψ², ∂yᶜᶠᶠ, w)
    ddz² = ℑxzᶜᵃᶜ(i, j, k, grid, fψ², ∂zᶠᶜᶠ, u) + ℑyzᵃᶜᶜ(i, j, k, grid, fψ², ∂zᶜᶠᶠ, v) + ∂zᶜᶜᶜ(i, j, k, grid, ψ², w)
    ν = _νᶜᶜᶜ(i, j, k, grid, p.closure, p.diffusivity_fields, p.clock)
    return ν * (ddx² + ddy² + ddz²)
end

"""
    $(SIGNATURES)

Calculate the pseudo viscous Dissipation Rate, defined as

    ε = ν (∂uᵢ/∂xⱼ) (∂uᵢ/∂xⱼ)

for a fluid with an isotropic turbulence closure (i.e., a 
turbulence closure where ν (eddy or not) is the same for all directions.
"""
function IsotropicPseudoViscousDissipationRate(model; U=0, V=0, W=0,
                                               location = (Center, Center, Center))

    validate_location(location, "IsotropicPseudoViscousDissipationRate")
    validate_dissipative_closure(model.closure)

    u, v, w = model.velocities

    parameters = (closure = model.closure,
                  diffusivity_fields = model.diffusivity_fields,
                  clock = model.clock)

    return KernelFunctionOperation{Center, Center, Center}(isotropic_pseudo_viscous_dissipation_rate_ccc, model.grid;
                                                           computed_dependencies=(u - U, v - V, w - W),
                                                           parameters)
end

#++++ Pressure redistribution terms
function XPressureRedistribution(model)
    u, v, w = model.velocities
    p = sum(model.pressures)
    return ∂x(u*p) # p is the total kinematic pressure (there's no need for ρ₀)
end

function YPressureRedistribution(model)
    u, v, w = model.velocities
    p = sum(model.pressures)
    return ∂y(v*p) # p is the total kinematic pressure (there's no need for ρ₀)
end

function ZPressureRedistribution(model)
    u, v, w = model.velocities
    p = sum(model.pressures)
    return ∂z(w*p) # p is the total kinematic pressure (there's no need for ρ₀)
end
#----


#++++ Shear production terms
@inline function shear_production_x_ccc(i, j, k, grid, u, v, w, U, V, W)
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
function XShearProduction(model, u, v, w, U, V, W; location = (Center, Center, Center))
    validate_location(location, "XShearProduction")
    return KernelFunctionOperation{Center, Center, Center}(shear_production_x_ccc, model.grid;
                                                           computed_dependencies=(u, v, w, U, V, W))
end

@inline function shear_production_y_ccc(i, j, k, grid, u, v, w, U, V, W)
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
function YShearProduction(model, u, v, w, U, V, W; location = (Center, Center, Center))
    validate_location(location, "YShearProduction")
    return KernelFunctionOperation{Center, Center, Center}(shear_production_y_ccc, model.grid;
                                                           computed_dependencies=(u, v, w, U, V, W))
end

@inline function shear_production_z_ccc(i, j, k, grid, u, v, w, U, V, W)
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
function ZShearProduction(model, u, v, w, U, V, W; location = (Center, Center, Center))
    validate_location(location, "ZShearProduction")
    return KernelFunctionOperation{Center, Center, Center}(shear_production_z_ccc, model.grid;
                                                           computed_dependencies=(u, v, w, U, V, W))
end

ZShearProduction(model; U=ZeroField(), V=ZeroField(), W=ZeroField(), kwargs...) =
    ZShearProduction(model, model.velocities..., U, V, W; kwargs...)
#----

end # module
