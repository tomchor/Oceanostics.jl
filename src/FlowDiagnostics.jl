module FlowDiagnostics

export IsotropicBuoyancyMixingRate, AnisotropicBuoyancyMixingRate

using Oceananigans.Operators
using KernelAbstractions: @index, @kernel
using Oceananigans.Grids: Center, Face
using Oceananigans.Fields: KernelComputedField

# Some useful operators
@inline fψ²(i, j, k, grid, f, ψ) = @inbounds f(i, j, k, grid, ψ)^2

@kernel function richardson_number_ccf!(Ri, grid, u, v, b, params)
    i, j, k = @index(Global, NTuple)

    dBdz_tot = ∂zᵃᵃᶠ(i, j, k, grid, b)        + params.N2_bg   # dbdz(c, c, f)
    dUdz_tot = ℑxᶜᵃᵃ(i, j, k, grid, ∂zᵃᵃᶠ, u) + params.dUdz_bg # dudz(f, c, f) → dudz(c, c, f)
    dVdz_tot = ℑyᵃᶜᵃ(i, j, k, grid, ∂zᵃᵃᶠ, v) + params.dVdz_bg # dvdz(c, f, f) → dvdz(c, c, f)

    @inbounds Ri[i, j, k] = dBdz_tot / (dUdz_tot^2 + dVdz_tot^2)
end


@kernel function rossby_number_ffc!(Ro, grid, u, v, params)
    i, j, k = @index(Global, NTuple)

    dUdy_tot = ∂yᵃᶠᵃ(i, j, k, grid, u) + params.dUdy_bg
    dVdx_tot = ∂xᶠᵃᵃ(i, j, k, grid, v) + params.dVdx_bg

    @inbounds Ro[i, j, k] = (dVdx_tot - dUdy_tot) / params.f₀
end


@kernel function potential_vorticity_in_thermal_wind_fff!(PV, grid, u, v, b, f₀)
    i, j, k = @index(Global, NTuple)

    dVdx =  ℑzᵃᵃᶠ(i, j, k, grid, ∂xᶠᵃᵃ, v) # F, F, C → F, F, F
    dUdy =  ℑzᵃᵃᶠ(i, j, k, grid, ∂yᵃᶠᵃ, u) # F, F, C → F, F, F
    dbdz = ℑxyᶠᶠᵃ(i, j, k, grid, ∂zᵃᵃᶠ, b) # C, C, F → F, F, F

    pv_barot = (f₀ + dVdx - dUdy) * dbdz

    dUdz = ℑyᵃᶠᵃ(i, j, k, grid, ∂zᵃᵃᶠ, u) # F, C, F → F, F, F
    dVdz = ℑxᶠᵃᵃ(i, j, k, grid, ∂zᵃᵃᶠ, v) # C, F, F → F, F, F

    pv_baroc = -f₀*(dUdz^2 + dVdz^2)

    @inbounds PV[i, j, k] = pv_barot[i, j, k] + pv_baroc[i, j, k]
end


@kernel function ertel_potential_vorticity_fff!(PV, grid, u, v, w, b, f₀)
    i, j, k = @index(Global, NTuple)

    dWdy =  ℑxᶠᵃᵃ(i, j, k, grid, ∂yᵃᶠᵃ, w) # C, C, F  → C, F, F  → F, F, F
    dVdz =  ℑxᶠᵃᵃ(i, j, k, grid, ∂zᵃᵃᶠ, v) # C, F, C  → C, F, F  → F, F, F
    dbdx = ℑyzᵃᶠᶠ(i, j, k, grid, ∂xᶠᵃᵃ, b) # C, C, C  → F, C, C  → F, F, F
    pv_x = (dWdy - dVdz) * dbdx # F, F, F

    dUdz =  ℑyᵃᶠᵃ(i, j, k, grid, ∂zᵃᵃᶠ, u) # F, C, C  → F, C, F → F, F, F
    dWdx =  ℑyᵃᶠᵃ(i, j, k, grid, ∂xᶠᵃᵃ, w) # C, C, F  → F, C, F → F, F, F
    dbdy = ℑxzᶠᵃᶠ(i, j, k, grid, ∂yᵃᶠᵃ, b) # C, C, C  → C, F, C → F, F, F
    pv_y = (dUdz - dWdx) * dbdy # F, F, F

    dVdx =  ℑzᵃᵃᶠ(i, j, k, grid, ∂xᶠᵃᵃ, v) # C, F, C  → F, F, C → F, F, F
    dUdy =  ℑzᵃᵃᶠ(i, j, k, grid, ∂yᵃᶠᵃ, u) # F, C, C  → F, F, C → F, F, F
    dbdz = ℑxyᶠᶠᵃ(i, j, k, grid, ∂zᵃᵃᶠ, b) # C, C, C  → C, C, F → F, F, F
    pv_z = (f₀ + dVdx - dUdy) * dbdz

    @inbounds PV[i, j, k] = pv_x + pv_y + pv_z
end


#+++++ Mixing of buoyancy
@kernel function isotropic_buoyancy_mixing_rate_ccc!(mixing_rate, grid, b, κᵇ, N²₀)
    i, j, k = @index(Global, NTuple)
    dbdx² = ℑxᶜᵃᵃ(i, j, k, grid, fψ², ∂xᶠᵃᵃ, b) # C, C, C  → F, C, C  → C, C, C
    dbdy² = ℑyᵃᶜᵃ(i, j, k, grid, fψ², ∂yᵃᶠᵃ, b) # C, C, C  → C, F, C  → C, C, C
    dbdz² = ℑzᵃᵃᶜ(i, j, k, grid, fψ², ∂zᵃᵃᶠ, b) # C, C, C  → C, C, F  → C, C, C

    @inbounds mixing_rate[i, j, k] = κᵇ[i,j,k] * (dbdx² + dbdy² + dbdz²) / N²₀
end

function IsotropicBuoyancyMixingRate(model, b, κᵇ, N²₀; location = (Center, Center, Center), kwargs...)
    if location == (Center, Center, Center)
        return KernelComputedField(Center, Center, Center, isotropic_buoyancy_mixing_rate_ccc!, model;
                                   computed_dependencies=(b, κᵇ), parameters=N²₀, kwargs...)
    else
        error("IsotropicBuoyancyMixingRate only supports location = (Center, Center, Center) for now.")
    end
end


@kernel function anisotropic_buoyancy_mixing_rate_ccc!(mixing_rate, grid, b, params)
    i, j, k = @index(Global, NTuple)
    dbdx² = ℑxᶜᵃᵃ(i, j, k, grid, fψ², ∂xᶠᵃᵃ, b) # C, C, C  → F, C, C  → C, C, C
    dbdy² = ℑyᵃᶜᵃ(i, j, k, grid, fψ², ∂yᵃᶠᵃ, b) # C, C, C  → C, F, C  → C, C, C
    dbdz² = ℑzᵃᵃᶜ(i, j, k, grid, fψ², ∂zᵃᵃᶠ, b) # C, C, C  → C, C, F  → C, C, C

    @inbounds mixing_rate[i, j, k] = (params.κx*dbdx² + params.κy*dbdy² + params.κz*dbdz²)/params.N²₀
end

function AnisotropicBuoyancyMixingRate(model, b, κx, κy, κz, N²₀; location = (Center, Center, Center), kwargs...)
    if location == (Center, Center, Center)
        return KernelComputedField(Center, Center, Center, anisotropic_buoyancy_mixing_rate_ccc!, model;
                                   computed_dependencies=(b,),
                                   parameters=(κx=κx, κy=κy, κz=κz, N²₀=N²₀), kwargs...)
    else
        error("AnisotropicBuoyancyMixingRate only supports location = (Center, Center, Center) for now.")
    end
end
#-----

end # module
