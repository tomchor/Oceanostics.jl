module FlowDiagnostics

using Oceananigans.Operators
using KernelAbstractions: @index, @kernel


@kernel function richardson_number_ccf!(Ri, grid, u, v, b, dUdz_bg, dVdz_bg, N2)
    i, j, k = @index(Global, NTuple)

    dBdz = ∂zᵃᵃᶠ(i, j, k, grid, b) + N2 # dbdz(c, c, f)
    dUdz_tot = ℑxᶜᵃᵃ(i, j, k, grid, ∂zᵃᵃᶠ, u) + dUdz_bg # dudz(f, c, f) → dudz(c, c, f)
    dVdz_tot = ℑyᵃᶜᵃ(i, j, k, grid, ∂zᵃᵃᶠ, v) + dVdz_bg # dvdz(c, f, f) → dvdz(c, c, f)

    @inbounds Ri[i, j, k] = dBdz / (dUdz_tot^2 + dVdz_tot^2)
end





@kernel function rossby_number_ffc!(Ro, grid, u, v, dUdy_bg, dVdx_bg, f₀)
    i, j, k = @index(Global, NTuple)

    dUdy_tot = ∂yᵃᶠᵃ(i, j, k, grid, u) + dUdy_bg
    dVdx_tot = ∂xᶠᵃᵃ(i, j, k, grid, v) + dVdx_bg

    @inbounds Ro[i, j, k] = (dVdx_tot - dUdy_tot) / f₀
end



@kernel function compute_pv_from_Ro_Ri!(PV, grid, Ri, Ro, N², f₀)
    i, j, k = @index(Global, NTuple)

    Ro_int = ℑxyzᶜᶜᶠ(i, j, k, grid, Ro)
    @inbounds PV[i, j, k] = N²[i, j, k]*f₀ * (1 + Ro[i, j, k] - 1/Ri[i, j, k])
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





@kernel function wp_ccc!(wp, grid, w, p)
    i, j, k = @index(Global, NTuple)
    @inbounds wp[i, j, k] = ℑzᵃᵃᶜ(i, j, k, grid, w) * p[i, j, k]
end



end # module
