module TurbulentKineticEnergyTerms

export TurbulentKineticEnergy, KineticEnergy

using Oceananigans.Operators
using KernelAbstractions: @index, @kernel
using Oceananigans.Grids: Center, Face
using Oceananigans.Fields: KernelComputedField, ZeroField

# Some useful operators
@inline ψ²(i, j, k, grid, ψ) = @inbounds ψ[i, j, k]^2
@inline ψ′²(i, j, k, grid, ψ, Ψ) = @inbounds (ψ[i, j, k] - Ψ[i, j, k])^2

@inline fψ²(i, j, k, grid, f, ψ) = @inbounds f(i, j, k, grid, ψ)^2

@inline fψ_plus_gφ²(i, j, k, grid, f, ψ, g, φ) = @inbounds (f(i, j, k, grid, ψ) + g(i, j, k, grid, φ))^2

@inline νfψ_plus_κgφ_times_fψ_plus_gφ(i, j, k, grid, ν, f, ψ, κ, g, φ) =
    @inbounds (ν*f(i, j, k, grid, ψ) + κ*g(i, j, k, grid, φ)) * (f(i, j, k, grid, ψ) + g(i, j, k, grid, φ))

@inline upᶠᵃᵃ(i, j, k, grid, u, p) = @inbounds u[i, j, k] * ℑxᶠᵃᵃ(i, j, k, grid, p)
@inline vpᵃᶠᵃ(i, j, k, grid, v, p) = @inbounds v[i, j, k] * ℑyᵃᶠᵃ(i, j, k, grid, p)
@inline wpᵃᵃᶠ(i, j, k, grid, w, p) = @inbounds w[i, j, k] * ℑzᵃᵃᶠ(i, j, k, grid, p)


#++++ Turbulent kinetic energy
@kernel function turbulent_kinetic_energy_ccc!(tke, grid, u, v, w, U, V, W)
    i, j, k = @index(Global, NTuple)

    @inbounds tke[i, j, k] = (
                              ℑxᶜᵃᵃ(i, j, k, grid, ψ′², u, U) +
                              ℑyᵃᶜᵃ(i, j, k, grid, ψ′², v, V) +
                              ℑzᵃᵃᶜ(i, j, k, grid, ψ′², w, W)
                             ) / 2
end

function TurbulentKineticEnergy(model, u, v, w;
                                U = ZeroField(),
                                V = ZeroField(),
                                W = ZeroField(),
                                location = (Center, Center, Center),
                                kwargs...)

    if location == (Center, Center, Center)
        return KernelComputedField(Center, Center, Center, turbulent_kinetic_energy_ccc!, model;
                                   computed_dependencies=(u, v, w, U, V, W), kwargs...)
    else
        error("TurbulentKineticEnergy only supports location = (Center, Center, Center) for now.")
    end
end

KineticEnergy(model, u, v, w; location = (Center, Center, Center), kwargs...) =
    TurbulentKineticEnergy(model, u, v, w; location, kwargs...)

TurbulentKineticEnergy(model; kwargs...) = TurbulentKineticEnergy(model, model.velocities...; kwargs...)
KineticEnergy(model; kwargs...) = KineticEnergy(model, model.velocities...; kwargs...)
#------


#++++ Energy dissipation rate for a fluid with constant isotropic viscosity
@kernel function isotropic_viscous_dissipation_rate_ccc!(ϵ, grid, u, v, w, ν)
    i, j, k = @index(Global, NTuple)

    Σˣˣ² = ∂xᶜᵃᵃ(i, j, k, grid, u)^2
    Σʸʸ² = ∂yᵃᶜᵃ(i, j, k, grid, v)^2
    Σᶻᶻ² = ∂zᵃᵃᶜ(i, j, k, grid, w)^2

    Σˣʸ² = ℑxyᶜᶜᵃ(i, j, k, grid, fψ_plus_gφ², ∂yᵃᶠᵃ, u, ∂xᶠᵃᵃ, v) / 4
    Σˣᶻ² = ℑxzᶜᵃᶜ(i, j, k, grid, fψ_plus_gφ², ∂zᵃᵃᶠ, u, ∂xᶠᵃᵃ, w) / 4
    Σʸᶻ² = ℑyzᵃᶜᶜ(i, j, k, grid, fψ_plus_gφ², ∂zᵃᵃᶠ, v, ∂yᵃᶠᵃ, w) / 4

    @inbounds ϵ[i, j, k] = ν[i, j, k] * 2 * (Σˣˣ² + Σʸʸ² + Σᶻᶻ² + 2 * (Σˣʸ² + Σˣᶻ² + Σʸᶻ²))
end

function IsotropicViscousDissipationRate(model, u, v, w, ν; location = (Center, Center, Center), kwargs...)
    if location == (Center, Center, Center)
        return KernelComputedField(Center, Center, Center, isotropic_viscous_dissipation_rate_ccc!, model;
                                   computed_dependencies=(u, v, w, ν), kwargs...)
    else
        error("IsotropicViscousDissipationRate only supports location = (Center, Center, Center) for now.")
    end
end


@kernel function isotropic_pseudo_viscous_dissipation_rate_ccc!(ϵ, grid, u, v, w, ν)
    i, j, k = @index(Global, NTuple)

    ddx² = ∂xᶜᵃᵃ(i, j, k, grid, ψ², u) + ℑxyᶜᶜᵃ(i, j, k, grid, fψ², ∂xᶠᵃᵃ, v) + ℑxzᶜᵃᶜ(i, j, k, grid, fψ², ∂xᶠᵃᵃ, w)
    ddy² = ℑxyᶜᶜᵃ(i, j, k, grid, fψ², ∂yᵃᶠᵃ, u) + ∂yᵃᶜᵃ(i, j, k, grid, ψ², v) + ℑyzᵃᶜᶜ(i, j, k, grid, fψ², ∂yᵃᶠᵃ, w)
    ddz² = ℑxzᶜᵃᶜ(i, j, k, grid, fψ², ∂zᵃᵃᶠ, u) + ℑyzᵃᶜᶜ(i, j, k, grid, fψ², ∂zᵃᵃᶠ, v) + ∂zᵃᵃᶜ(i, j, k, grid, ψ², w)

    @inbounds ϵ[i, j, k] = ν[i,j,k] * (ddx² + ddy² + ddz²)
end

function IsotropicPseudoViscousDissipationRate(model, u, v, w, ν; location = (Center, Center, Center), kwargs...)
    if location == (Center, Center, Center)
        return KernelComputedField(Center, Center, Center, isotropic_pseudo_viscous_dissipation_rate_ccc!, model;
                                   computed_dependencies=(u, v, w, ν), kwargs...)
    else
        error("IsotropicPseudoViscousDissipationRate only supports location = (Center, Center, Center) for now.")
    end
end
#------


#+++++ Energy dissipation rate for a fluid with constant anisotropic viscosity (closure)
@kernel function anisotropic_viscous_dissipation_rate_ccc!(ϵ, grid, u, v, w, params)
    i, j, k = @index(Global, NTuple)
    νx=params.νx; νy=params.νy; νz=params.νz;

    Σˣˣ² = νx * ∂xᶜᵃᵃ(i, j, k, grid, u)^2
    Σʸʸ² = νy * ∂yᵃᶜᵃ(i, j, k, grid, v)^2
    Σᶻᶻ² = νz * ∂zᵃᵃᶜ(i, j, k, grid, w)^2

    Σˣʸ² = ℑxyᶜᶜᵃ(i, j, k, grid, νfψ_plus_κgφ_times_fψ_plus_gφ, νy, ∂yᵃᶠᵃ, u, νx, ∂xᶠᵃᵃ, v) / 4
    Σˣᶻ² = ℑxzᶜᵃᶜ(i, j, k, grid, νfψ_plus_κgφ_times_fψ_plus_gφ, νz, ∂zᵃᵃᶠ, u, νx, ∂xᶠᵃᵃ, w) / 4
    Σʸᶻ² = ℑyzᵃᶜᶜ(i, j, k, grid, νfψ_plus_κgφ_times_fψ_plus_gφ, νz, ∂zᵃᵃᶠ, v, νy, ∂yᵃᶠᵃ, w) / 4

    @inbounds ϵ[i, j, k] = 2 * (Σˣˣ² + Σʸʸ² + Σᶻᶻ² + 2*(Σˣʸ² + Σˣᶻ² + Σʸᶻ²))
end

function AnisotropicViscousDissipationRate(model, u, v, w, νx, νy, νz; location = (Center, Center, Center), kwargs...)
    if location == (Center, Center, Center)
        return KernelComputedField(Center, Center, Center, anisotropic_viscous_dissipation_rate_ccc!, model;
                                   computed_dependencies=(u, v, w),
                                   parameters=(νx=νx, νy=νy, νz=νz), kwargs...)
    else
        error("AnisotropicViscousDissipationRate only supports location = (Center, Center, Center) for now.")
    end
end


@kernel function anisotropic_pseudo_viscous_dissipation_rate_ccc!(ϵ, grid, u, v, w, params)
    i, j, k = @index(Global, NTuple)

    ddx² = ∂xᶜᵃᵃ(i, j, k, grid, ψ², u) + ℑxyᶜᶜᵃ(i, j, k, grid, fψ², ∂xᶠᵃᵃ, v) + ℑxzᶜᵃᶜ(i, j, k, grid, fψ², ∂xᶠᵃᵃ, w)
    ddy² = ℑxyᶜᶜᵃ(i, j, k, grid, fψ², ∂yᵃᶠᵃ, u) + ∂yᵃᶜᵃ(i, j, k, grid, ψ², v) + ℑyzᵃᶜᶜ(i, j, k, grid, fψ², ∂yᵃᶠᵃ, w)
    ddz² = ℑxzᶜᵃᶜ(i, j, k, grid, fψ², ∂zᵃᵃᶠ, u) + ℑyzᵃᶜᶜ(i, j, k, grid, fψ², ∂zᵃᵃᶠ, v) + ∂zᵃᵃᶜ(i, j, k, grid, ψ², w)

    @inbounds ϵ[i, j, k] = params.νx*ddx² + params.νy*ddy² + params.νz*ddz²
end

function AnisotropicPseudoViscousDissipationRate(model, u, v, w, νx, νy, νz; location = (Center, Center, Center), kwargs...)
    if location == (Center, Center, Center)
        return KernelComputedField(Center, Center, Center, anisotropic_pseudo_viscous_dissipation_rate_ccc!, model;
                                   computed_dependencies=(u, v, w),
                                   parameters=(νx=νx, νy=νy, νz=νz,), kwargs...)
    else
        error("AnisotropicPseudoViscousDissipationRate only supports location = (Center, Center, Center) for now.")
    end
end
#-----


#++++ Pressure redistribution terms
@kernel function pressure_redistribution_x_ccc!(dupdx_ρ, grid, u, p, ρ₀)
    i, j, k = @index(Global, NTuple)
    @inbounds dupdx_ρ[i, j, k] = (1/ρ₀) * ∂xᶜᵃᵃ(i, j, k, grid, upᶠᵃᵃ, u, p) # C, C, F  → C, C, C
end

function PressureRedistribution_x(model, u, p, ρ₀; location = (Center, Center, Center), kwargs...)
    if location == (Center, Center, Center)
        return KernelComputedField(Center, Center, Center, pressure_redistribution_x_ccc!, model;
                                   computed_dependencies=(u, p), parameters=ρ₀, kwargs...)
    else
        error("PressureRedistribution_x only supports location = (Center, Center, Center) for now.")
    end
end


@kernel function pressure_redistribution_y_ccc!(dvpdy_ρ, grid, v, p, ρ₀)
    i, j, k = @index(Global, NTuple)
    @inbounds dvpdy_ρ[i, j, k] = (1/ρ₀) * ∂yᵃᶜᵃ(i, j, k, grid, vpᵃᶠᵃ, v, p) # C, C, F  → C, C, C
end

function PressureRedistribution_y(model, v, p, ρ₀; location = (Center, Center, Center), kwargs...)
    if location == (Center, Center, Center)
        return KernelComputedField(Center, Center, Center, pressure_redistribution_y_ccc!, model;
                                   computed_dependencies=(v, p), parameters=ρ₀, kwargs...)
    else
        error("PressureRedistribution_y only supports location = (Center, Center, Center) for now.")
    end
end


@kernel function pressure_redistribution_z_ccc!(dwpdz_ρ, grid, w, p, ρ₀)
    i, j, k = @index(Global, NTuple)
    @inbounds dwpdz_ρ[i, j, k] = (1/ρ₀) * ∂zᵃᵃᶜ(i, j, k, grid, wpᵃᵃᶠ, w, p) # C, C, F  → C, C, C
end

function PressureRedistribution_z(model, w, p, ρ₀; location = (Center, Center, Center), kwargs...)
    if location == (Center, Center, Center)
        return KernelComputedField(Center, Center, Center, pressure_redistribution_z_ccc!, model;
                                   computed_dependencies=(w, p), parameters=ρ₀, kwargs...)
    else
        error("PressureRedistribution_z only supports location = (Center, Center, Center) for now.")
    end
end
#----


#++++ Shear production terms
@kernel function shear_production_x_ccc!(shear_production, grid, u, v, w, U, V, W)
    i, j, k = @index(Global, NTuple)
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

    @inbounds shear_production[i, j, k] = -(uu∂xU + vu∂xV + wu∂xW)
end

function ShearProduction_x(model, u, v, w, U, V, W; location = (Center, Center, Center), kwargs...)
    if location == (Center, Center, Center)
        return KernelComputedField(Center, Center, Center, shear_production_x_ccc!, model;
                                   computed_dependencies=(u, v, w, U, V, W), kwargs...)
    else
        error("ShearProduction_x only supports location = (Center, Center, Center) for now.")
    end
end


@kernel function shear_production_y_ccc!(shear_production, grid, u, v, w, U, V, W)
    i, j, k = @index(Global, NTuple)
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

    @inbounds shear_production[i, j, k] = -(uv∂yU + vv∂yV + wv∂yW)
end

function ShearProduction_y(model, u, v, w, U, V, W; location = (Center, Center, Center), kwargs...)
    if location == (Center, Center, Center)
        return KernelComputedField(Center, Center, Center, shear_production_y_ccc!, model;
                                   computed_dependencies=(u, v, w, U, V, W), kwargs...)
    else
        error("ShearProduction_y only supports location = (Center, Center, Center) for now.")
    end
end


@kernel function shear_production_z_ccc!(shear_production, grid, u, v, w, U, V, W)
    i, j, k = @index(Global, NTuple)
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

    @inbounds shear_production[i, j, k] = - (uw∂zU + vw∂zV + ww∂zW)
end

function ShearProduction_z(model, u, v, w, U, V, W; location = (Center, Center, Center), kwargs...)
    if location == (Center, Center, Center)
        return KernelComputedField(Center, Center, Center, shear_production_z_ccc!, model;
                                   computed_dependencies=(u, v, w, U, V, W), kwargs...)
    else
        error("ShearProduction_z only supports location = (Center, Center, Center) for now.")
    end
end
#----

end # module
