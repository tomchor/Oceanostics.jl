using KernelAbstractions: @index, @kernel
using Oceananigans.Operators


@inline ψ′²(i, j, k, grid, ψ, Ψ) = @inbounds (ψ[i, j, k] - Ψ[i, j, k])^2
@kernel function turbulent_kinetic_energy_ccc!(tke, grid, u, v, w, U, V, W)
    i, j, k = @index(Global, NTuple)

    @inbounds tke[i, j, k] = (
                              ℑxᶜᵃᵃ(i, j, k, grid, ψ′², u, U) +
                              ℑyᵃᶜᵃ(i, j, k, grid, ψ′², v, V) +
                              ℑzᵃᵃᶜ(i, j, k, grid, ψ′², w, W)
                             ) / 2
end



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



@kernel function anisotropic_viscous_dissipation_ccc!(ϵ, grid, νx, νy, νz, u, v, w)
    i, j, k = @index(Global, NTuple)

    ddx² = ∂xᶜᵃᵃ(i, j, k, grid, u)^2 + ℑxyᶜᶜᵃ(i, j, k, grid, ∂xᶠᵃᵃ, v)^2 + ℑxzᶜᵃᶜ(i, j, k, grid, ∂xᶠᵃᵃ, w)^2

    ddy² = ℑxyᶜᶜᵃ(i, j, k, grid, ∂yᵃᶠᵃ, u)^2 + ∂yᵃᶜᵃ(i, j, k, grid, v)^2 + ℑyzᵃᶜᶜ(i, j, k, grid, ∂yᵃᶠᵃ, w)^2

    ddz² = ℑxzᶜᵃᶜ(i, j, k, grid, ∂zᵃᵃᶠ, u)^2 + ℑyzᵃᶜᶜ(i, j, k, grid, ∂zᵃᵃᶠ, v)^2 + ∂zᵃᵃᶜ(i, j, k, grid, w)^2

    @inbounds ϵ[i, j, k] = νx[i,j,k]*ddx² + νy[i,j,k]*ddy² + νz[i,j,k]*ddz²
end



@kernel function compute_vertical_pressure_term!(dwpdz, grid, w, p, ρ₀)
    i, j, k = @index(Global, NTuple)

    wp = ℑzᵃᵃᶠ(i, j, k, grid, p) * w[i, j, k]

    @inbounds dwpdz[i, j, k] = (1/ρ₀) * ∂zᵃᵃᶜ(i, j, k, grid, wp)
end


@inline ψ²(i, j, k, grid, ψ) = @inbounds ψ[i, j, k]^2
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



@kernel function shear_production_z_ccc!(shear_production, grid, u, v, w, U, V, W)
    i, j, k = @index(Global, NTuple)
    w_int = ℑyᵃᶜᵃ(i, j, k, grid, w) # C, F, C  → C, C, C

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


