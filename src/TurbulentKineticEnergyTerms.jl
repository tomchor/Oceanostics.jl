module TurbulentKineticEnergyTerms


using Oceananigans.Operators
using KernelAbstractions: @index, @kernel
using Oceananigans.Grids: Center, Face
using Oceananigans.Fields: KernelComputedField

@inline ψ²(i, j, k, grid, ψ) = @inbounds ψ[i, j, k]^2
@inline ψ′²(i, j, k, grid, ψ, Ψ) = @inbounds (ψ[i, j, k] - Ψ[i, j, k])^2


@kernel function kinetic_energy_ccc!(tke, grid, u, v, w)
    i, j, k = @index(Global, NTuple)

    @inbounds tke[i, j, k] = (
                              ℑxᶜᵃᵃ(i, j, k, grid, ψ², u) +
                              ℑyᵃᶜᵃ(i, j, k, grid, ψ², v) +
                              ℑzᵃᵃᶜ(i, j, k, grid, ψ², w)
                             ) / 2
end

function KineticEnergy(model, u, v, w, location = (Center, Center, Center), kwargs...)
    if location == (Center, Center, Center)
        return KernelComputedField(Center, Center, Center, kinetic_energy_ccc!, model;
                                   computed_dependencies=(u, v, w), kwargs...)
    else
        throw(Exception)
    end
end




#++++ TKE dissipation
@kernel function isotropic_viscous_dissipation_ccc!(ϵ, grid, ν, u, v, w)
    i, j, k = @index(Global, NTuple)

    Σˣˣ = ∂xᶜᵃᵃ(i, j, k, grid, u)
    Σʸʸ = ∂yᵃᶜᵃ(i, j, k, grid, v)
    Σᶻᶻ = ∂zᵃᵃᶜ(i, j, k, grid, w)

    Σˣʸ = (ℑxyᶜᶜᵃ(i, j, k, grid, ∂yᵃᶠᵃ, u) + ℑxyᶜᶜᵃ(i, j, k, grid, ∂xᶠᵃᵃ, v)) / 2
    Σˣᶻ = (ℑxzᶜᵃᶜ(i, j, k, grid, ∂zᵃᵃᶠ, u) + ℑxzᶜᵃᶜ(i, j, k, grid, ∂xᶠᵃᵃ, w)) / 2
    Σʸᶻ = (ℑyzᵃᶜᶜ(i, j, k, grid, ∂zᵃᵃᶠ, v) + ℑyzᵃᶜᶜ(i, j, k, grid, ∂yᵃᶠᵃ, w)) / 2

    @inbounds ϵ[i, j, k] = ν[i, j, k] * 2 * (Σˣˣ^2 + Σʸʸ^2 + Σᶻᶻ^2 + 2 * (Σˣʸ^2 + Σˣᶻ^2 + Σʸᶻ^2))
end

function IsotropicViscousDissipation(model, ν, u, v, w, location = (Center, Center, Center), kwargs...)
    if location == (Center, Center, Center)
        return KernelComputedField(Center, Center, Center, isotropic_viscous_dissipation_ccc!, model;
                                   computed_dependencies=(ν, u, v, w), kwargs...)
    else
        throw(Exception)
    end
end




@kernel function anisotropic_viscous_dissipation_ccc!(ϵ, grid, νx, νy, νz, u, v, w)
    i, j, k = @index(Global, NTuple)

    ddx² = ∂xᶜᵃᵃ(i, j, k, grid, u)^2 + ℑxyᶜᶜᵃ(i, j, k, grid, ∂xᶠᵃᵃ, v)^2 + ℑxzᶜᵃᶜ(i, j, k, grid, ∂xᶠᵃᵃ, w)^2

    ddy² = ℑxyᶜᶜᵃ(i, j, k, grid, ∂yᵃᶠᵃ, u)^2 + ∂yᵃᶜᵃ(i, j, k, grid, v)^2 + ℑyzᵃᶜᶜ(i, j, k, grid, ∂yᵃᶠᵃ, w)^2

    ddz² = ℑxzᶜᵃᶜ(i, j, k, grid, ∂zᵃᵃᶠ, u)^2 + ℑyzᵃᶜᶜ(i, j, k, grid, ∂zᵃᵃᶠ, v)^2 + ∂zᵃᵃᶜ(i, j, k, grid, w)^2

    @inbounds ϵ[i, j, k] = νx[i,j,k]*ddx² + νy[i,j,k]*ddy² + νz[i,j,k]*ddz²
end

function AnisotropicViscousDissipation(model, νx, νy, νz, u, v, w, location = (Center, Center, Center), kwargs...)
    if location == (Center, Center, Center)
        return KernelComputedField(Center, Center, Center, anisotropic_viscous_dissipation_ccc!, model;
                                   computed_dependencies=(νx, νy, νz, u, v, w), kwargs...)
    else
        throw(Exception)
    end
end
#-----


#++++ Pressure redistribution terms
@inline upᶠᵃᵃ(i, j, k, grid, u, p) = @inbounds u[i, j, k] * ℑxᶠᵃᵃ(i, j, k, grid, p)
@kernel function pressure_redistribution_x_ccc!(dupdx_ρ, grid, u, p, ρ₀)
    i, j, k = @index(Global, NTuple)
    @inbounds dupdx_ρ[i, j, k] = (1/ρ₀) * ∂xᶜᵃᵃ(i, j, k, grid, upᶠᵃᵃ, u, p) # C, C, F  → C, C, C
end

@inline vpᵃᶠᵃ(i, j, k, grid, v, p) = @inbounds v[i, j, k] * ℑyᵃᶠᵃ(i, j, k, grid, p)
@kernel function pressure_redistribution_y_ccc!(dvpdy_ρ, grid, v, p, ρ₀)
    i, j, k = @index(Global, NTuple)
    @inbounds dvpdy_ρ[i, j, k] = (1/ρ₀) * ∂yᵃᶜᵃ(i, j, k, grid, vpᵃᶠᵃ, v, p) # C, C, F  → C, C, C
end 

@inline wpᵃᵃᶠ(i, j, k, grid, w, p) = @inbounds w[i, j, k] * ℑzᵃᵃᶠ(i, j, k, grid, p)
@kernel function pressure_redistribution_z_ccc!(dwpdz_ρ, grid, w, p, ρ₀)
    i, j, k = @index(Global, NTuple)
    @inbounds dwpdz_ρ[i, j, k] = (1/ρ₀) * ∂zᵃᵃᶜ(i, j, k, grid, wpᵃᵃᶠ, w, p) # C, C, F  → C, C, C
end 
#----


#++++ Shear production terms
@kernel function shear_production_x_ccc!(shear_production, grid, u, v, w, U, V, W)
    i, j, k = @index(Global, NTuple)
    u_int = ℑxᶜᵃᵃ(i, j, k, grid, u) # F, C, C  → C, C, C

    ∂xU = ∂x(i, j, k, grid, U) # F, C, C  → C, C, C
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

function ShearProduction_x(model, u, v, w, U, V, W, location = (Center, Center, Center), kwargs...)
    if location == (Center, Center, Center)
        return KernelComputedField(Center, Center, Center, shear_production_x_ccc!, model;
                                   computed_dependencies=(u, v, w, U, V, W), kwargs...)
    else
        throw(Exception)
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

function ShearProduction_y(model, u, v, w, U, V, W, location = (Center, Center, Center), kwargs...)
    if location == (Center, Center, Center)
        return KernelComputedField(Center, Center, Center, shear_production_y_ccc!, model;
                                   computed_dependencies=(u, v, w, U, V, W), kwargs...)
    else
        throw(Exception)
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

function ShearProduction_z(model, u, v, w, U, V, W, location = (Center, Center, Center), kwargs...)
    if location == (Center, Center, Center)
        return KernelComputedField(Center, Center, Center, shear_production_z_ccc!, model;
                                   computed_dependencies=(u, v, w, U, V, W), kwargs...)
    else
        throw(Exception)
    end
end
#----

end # module
