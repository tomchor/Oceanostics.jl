module FlowDiagnostics
using DocStringExtensions

export RichardsonNumber, RossbyNumber
export ErtelPotentialVorticity, ThermalWindPotentialVorticity, DirectionalErtelPotentialVorticity
export StrainRateTensorModulus, VorticityTensorModulus, Q, QVelocityGradientTensorInvariant

using Oceanostics: validate_location, validate_dissipative_closure, add_background_fields

using Oceananigans: NonhydrostaticModel, FPlane, ConstantCartesianCoriolis, BuoyancyField, BuoyancyTracer
using Oceananigans.Operators
using Oceananigans.AbstractOperations
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Grids: Center, Face, NegativeZDirection, ZDirection

#+++ Richardson number
@inline ψ²(i, j, k, grid, ψ) = @inbounds ψ[i, j, k]^2

"""
Get `w` from `û`, `v̂`, `ŵ` and based on the direction given by the unit vector `vertical_dir`.
"""
@inline function w²_from_u⃗_tilted_ccc(i, j, k, grid, û, v̂, ŵ, vertical_dir)
    û = ℑxᶜᵃᵃ(i, j, k, grid, û) # F, C, C  → C, C, C
    v̂ = ℑyᵃᶜᵃ(i, j, k, grid, v̂) # C, F, C  → C, C, C
    ŵ = ℑzᵃᵃᶜ(i, j, k, grid, ŵ) # C, C, F  → C, C, C
    return (û * vertical_dir[1] + v̂ * vertical_dir[2] + ŵ * vertical_dir[3])^2
end

"""
    $(SIGNATURES)

Return the (true) horizontal velocity magnitude.
"""
@inline function uₕ_norm_ccc(i, j, k, grid, û, v̂, ŵ, vertical_dir)
    û² = ℑxᶜᵃᵃ(i, j, k, grid, ψ², û) # F, C, C  → C, C, C
    v̂² = ℑyᵃᶜᵃ(i, j, k, grid, ψ², v̂) # C, F, C  → C, C, C
    ŵ² = ℑzᵃᵃᶜ(i, j, k, grid, ψ², ŵ) # C, C, F  → C, C, C
    return √(û² + v̂² + ŵ² - w²_from_u⃗_tilted_ccc(i, j, k, grid, û, v̂, ŵ, vertical_dir))
end

@inline function richardson_number_ccf(i, j, k, grid, û, v̂, ŵ, b, vertical_dir)

    dbdx̂ = ℑxzᶜᵃᶠ(i, j, k, grid, ∂xᶠᶜᶜ, b) # C, C, C  → F, C, C → C, C, F
    dbdŷ = ℑyzᵃᶜᶠ(i, j, k, grid, ∂yᶜᶠᶜ, b) # C, C, C  → C, F, C → C, C, F
    dbdẑ = ∂zᶜᶜᶠ(i, j, k, grid, b) # C, C, C  → C, C, F
    dbdz = dbdx̂ * vertical_dir[1] + dbdŷ * vertical_dir[2] + dbdẑ * vertical_dir[3]

    duₕdx̂ = ℑxᶜᵃᵃ(i, j, k, grid, ∂xᶠᶜᶜ, uₕ_norm_ccc, û, v̂, ŵ, vertical_dir)
    duₕdŷ = ℑyᵃᶜᵃ(i, j, k, grid, ∂yᶜᶠᶜ, uₕ_norm_ccc, û, v̂, ŵ, vertical_dir)
    duₕdẑ = ∂zᶜᶜᶠ(i, j, k, grid, uₕ_norm_ccc, û, v̂, ŵ, vertical_dir)
    duₕdz = duₕdx̂ * vertical_dir[1] + duₕdŷ * vertical_dir[2] + duₕdẑ * vertical_dir[3]

    return dbdz / duₕdz^2
end

"""
    $(SIGNATURES)

Calculate the Richardson Number as
    Ri = (∂b/∂z) / (|∂u⃗ₕ/∂z|²)
where `z` is the true vertical direction (ie anti-parallel to gravity).
"""
function RichardsonNumber(model; location = (Center, Center, Face), add_background=true)
    validate_location(location, "RichardsonNumber", (Center, Center, Face))

    if (model isa NonhydrostaticModel) & add_background
        full_fields = add_background_fields(model)
        u, v, w, b = full_fields.u, full_fields.v, full_fields.w, full_fields.b
    else
        u, v, w = model.velocities
        b = model.tracers.b
    end

    if model.buoyancy.gravity_unit_vector isa NegativeZDirection
        true_vertical_direction = (0, 0, 1)
    elseif model.buoyancy.gravity_unit_vector isa ZDirection
        true_vertical_direction = (0, 0, -1)
    else
        true_vertical_direction = .-model.buoyancy.gravity_unit_vector
    end
    return KernelFunctionOperation{Center, Center, Face}(richardson_number_ccf, model.grid,
                                                         u, v, w, b, true_vertical_direction)
end
#---

#+++ Rossby number
@inline function rossby_number_fff(i, j, k, grid, u, v, w, params)
    dwdy =  ℑxᶠᵃᵃ(i, j, k, grid, ∂yᶜᶠᶠ, w) # C, C, F  → C, F, F  → F, F, F
    dvdz =  ℑxᶠᵃᵃ(i, j, k, grid, ∂zᶜᶠᶠ, v) # C, F, C  → C, F, F  → F, F, F
    ω_x = (dwdy + params.dWdy_bg) - (dvdz + params.dVdz_bg)

    dudz =  ℑyᵃᶠᵃ(i, j, k, grid, ∂zᶠᶜᶠ, u) # F, C, C  → F, C, F → F, F, F
    dwdx =  ℑyᵃᶠᵃ(i, j, k, grid, ∂xᶠᶜᶠ, w) # C, C, F  → F, C, F → F, F, F
    ω_y = (dudz + params.dUdz_bg) - (dwdx + params.dWdx_bg)

    dvdx =  ℑzᵃᵃᶠ(i, j, k, grid, ∂xᶠᶠᶜ, v) # C, F, C  → F, F, C → F, F, F
    dudy =  ℑzᵃᵃᶠ(i, j, k, grid, ∂yᶠᶠᶜ, u) # F, C, C  → F, F, C → F, F, F
    ω_z = (dvdx + params.dVdx_bg) - (dudy + params.dUdy_bg)

    return (ω_x*params.fx + ω_y*params.fy + ω_z*params.fz)/(params.fx^2 + params.fy^2 + params.fz^2)
end

""" 
    $(SIGNATURES)

Calculate the Rossby number using the vorticity in the rotation axis direction according
to `model.coriolis`. Rossby number is defined as

    Ro = ωᶻ / f

where ωᶻ is the vorticity in the Coriolis axis of rotation and `f` is the Coriolis rotation frequency.
"""
function RossbyNumber(model; location = (Face, Face, Face),
                      dWdy_bg=0, dVdz_bg=0,
                      dUdz_bg=0, dWdx_bg=0,
                      dUdy_bg=0, dVdx_bg=0)
    validate_location(location, "RossbyNumber", (Face, Face, Face))

    if model isa NonhydrostaticModel
        full_fields = add_background_fields(model)
        u, v, w = full_fields.u, full_fields.v, full_fields.w
    else
        u, v, w = model.velocities
    end

    coriolis = model.coriolis
    if coriolis isa FPlane
        fx = fy = 0
        fz = model.coriolis.f
    elseif coriolis isa ConstantCartesianCoriolis
        fx = coriolis.fx
        fy = coriolis.fy
        fz = coriolis.fz
    else
        throw(ArgumentError("RossbyNumber only implemented for FPlane and ConstantCartesianCoriolis"))
    end

    parameters = (; fx, fy, fz, dWdy_bg, dVdz_bg, dUdz_bg, dWdx_bg, dUdy_bg, dVdx_bg)
    return KernelFunctionOperation{Face, Face, Face}(rossby_number_fff, model.grid,
                                                     u, v, w, parameters)
end
#---

#+++ Potential vorticity
@inline function potential_vorticity_in_thermal_wind_fff(i, j, k, grid, u, v, b, f)

    dVdx =  ℑzᵃᵃᶠ(i, j, k, grid, ∂xᶠᶠᶜ, v) # F, F, C → F, F, F
    dUdy =  ℑzᵃᵃᶠ(i, j, k, grid, ∂yᶠᶠᶜ, u) # F, F, C → F, F, F
    dbdz = ℑxyᶠᶠᵃ(i, j, k, grid, ∂zᶜᶜᶠ, b) # C, C, F → F, F, F

    pv_barot = (f + dVdx - dUdy) * dbdz

    dUdz = ℑyᵃᶠᵃ(i, j, k, grid, ∂zᶠᶜᶠ, u) # F, C, F → F, F, F
    dVdz = ℑxᶠᵃᵃ(i, j, k, grid, ∂zᶜᶠᶠ, v) # C, F, F → F, F, F

    pv_baroc = -f * (dUdz^2 + dVdz^2)

    return pv_barot + pv_baroc
end

"""
    $(SIGNATURES)

Calculate the Potential Vorticty assuming thermal wind balance for `model`, where the characteristics of
the Coriolis rotation are taken from `model.coriolis`. The Potential Vorticity in this case
is defined as

    TWPV = (f + ωᶻ) ∂b/∂z - f ((∂U/∂z)² + (∂V/∂z)²)

where `f` is the Coriolis frequency, `ωᶻ` is the relative vorticity in the `z` direction, `b` is the buoyancy, and
`∂U/∂z` and `∂V/∂z` comprise the thermal wind shear.
"""
function ThermalWindPotentialVorticity(model; f=nothing, location = (Face, Face, Face))
    validate_location(location, "ThermalWindPotentialVorticity", (Face, Face, Face))
    u, v, w = model.velocities
    b = BuoyancyField(model)
    if isnothing(f)
        f = model.coriolis.f
    end
    return KernelFunctionOperation{Face, Face, Face}(potential_vorticity_in_thermal_wind_fff, model.grid,
                                                     u, v, b, f)
end

@inline function ertel_potential_vorticity_fff(i, j, k, grid, u, v, w, b, fx, fy, fz)
    dWdy =  ℑxᶠᵃᵃ(i, j, k, grid, ∂yᶜᶠᶠ, w) # C, C, F  → C, F, F  → F, F, F
    dVdz =  ℑxᶠᵃᵃ(i, j, k, grid, ∂zᶜᶠᶠ, v) # C, F, C  → C, F, F  → F, F, F
    dbdx = ℑyzᵃᶠᶠ(i, j, k, grid, ∂xᶠᶜᶜ, b) # C, C, C  → F, C, C  → F, F, F
    pv_x = (fx + dWdy - dVdz) * dbdx # F, F, F

    dUdz =  ℑyᵃᶠᵃ(i, j, k, grid, ∂zᶠᶜᶠ, u) # F, C, C  → F, C, F → F, F, F
    dWdx =  ℑyᵃᶠᵃ(i, j, k, grid, ∂xᶠᶜᶠ, w) # C, C, F  → F, C, F → F, F, F
    dbdy = ℑxzᶠᵃᶠ(i, j, k, grid, ∂yᶜᶠᶜ, b) # C, C, C  → C, F, C → F, F, F
    pv_y = (fy + dUdz - dWdx) * dbdy # F, F, F

    dVdx =  ℑzᵃᵃᶠ(i, j, k, grid, ∂xᶠᶠᶜ, v) # C, F, C  → F, F, C → F, F, F
    dUdy =  ℑzᵃᵃᶠ(i, j, k, grid, ∂yᶠᶠᶜ, u) # F, C, C  → F, F, C → F, F, F
    dbdz = ℑxyᶠᶠᵃ(i, j, k, grid, ∂zᶜᶜᶠ, b) # C, C, C  → C, C, F → F, F, F
    pv_z = (fz + dVdx - dUdy) * dbdz

    return pv_x + pv_y + pv_z
end

"""
    $(SIGNATURES)

Calculate the Ertel Potential Vorticty for `model`, where the characteristics of
the Coriolis rotation are taken from `model.coriolis`. The Ertel Potential Vorticity
is defined as

    EPV = ωₜₒₜ ⋅ ∇b

where ωₜₒₜ is the total (relative + planetary) vorticity vector, `b` is the buoyancy and ∇ is the gradient
operator.
"""
function ErtelPotentialVorticity(model; location = (Face, Face, Face))
    validate_location(location, "ErtelPotentialVorticity", (Face, Face, Face))

    u, v, w = model.velocities
    b = model.tracers.b

    if model isa NonhydrostaticModel
        full_fields = add_background_fields(model)
        u, v, w, b = full_fields.u, full_fields.v, full_fields.w, full_fields.b
    else
        u, v, w = model.velocities
        b = model.tracers.b
    end

    coriolis = model.coriolis
    if coriolis isa FPlane
        fx = fy = 0
        fz = model.coriolis.f
    elseif coriolis isa ConstantCartesianCoriolis
        fx = coriolis.fx
        fy = coriolis.fy
        fz = coriolis.fz
    else
        throw(ArgumentError("ErtelPotentialVorticity only implemented for FPlane and ConstantCartesianCoriolis"))
    end

    return KernelFunctionOperation{Face, Face, Face}(ertel_potential_vorticity_fff, model.grid,
                                                     u, v, w, b, fx, fy, fz)
end

@inline function directional_ertel_potential_vorticity_fff(i, j, k, grid, u, v, w, b, params)

    dWdy =  ℑxᶠᵃᵃ(i, j, k, grid, ∂yᶜᶠᶠ, w) # C, C, F  → C, F, F → F, F, F
    dVdz =  ℑxᶠᵃᵃ(i, j, k, grid, ∂zᶜᶠᶠ, v) # C, F, C  → C, F, F → F, F, F
    ω̂_x = dWdy - dVdz # F, F, F

    dUdz =  ℑyᵃᶠᵃ(i, j, k, grid, ∂zᶠᶜᶠ, u) # F, C, C  → F, C, F → F, F, F
    dWdx =  ℑyᵃᶠᵃ(i, j, k, grid, ∂xᶠᶜᶠ, w) # C, C, F  → F, C, F → F, F, F
    ω̂_y = dUdz - dWdx # F, F, F

    dVdx =  ℑzᵃᵃᶠ(i, j, k, grid, ∂xᶠᶠᶜ, v) # C, F, C  → F, F, C → F, F, F
    dUdy =  ℑzᵃᵃᶠ(i, j, k, grid, ∂yᶠᶠᶜ, u) # F, C, C  → F, F, C → F, F, F
    ω̂_z = dVdx - dUdy # F, F, F

    dbdx̂ = ℑyzᵃᶠᶠ(i, j, k, grid, ∂xᶠᶜᶜ, b) # C, C, C  → F, C, C → F, F, F
    dbdŷ = ℑxzᶠᵃᶠ(i, j, k, grid, ∂yᶜᶠᶜ, b) # C, C, C  → C, F, C → F, F, F
    dbdẑ = ℑxyᶠᶠᵃ(i, j, k, grid, ∂zᶜᶜᶠ, b) # C, C, C  → C, C, F → F, F, F

    ω_dir = ω̂_x * params.dir_x + ω̂_y * params.dir_y + ω̂_z * params.dir_z
    dbddir = dbdx̂ * params.dir_x + dbdŷ * params.dir_y + dbdẑ * params.dir_z

    return (params.f_dir + ω_dir) * dbddir
end


"""
    $(SIGNATURES)

Calculate the contribution from a given `direction` to the Ertel Potential Vorticity
basde on a `model` and a `direction`. The Ertel Potential Vorticity is defined as

    EPV = ωₜₒₜ ⋅ ∇b

where ωₜₒₜ is the total (relative + planetary) vorticity vector, `b` is the buoyancy and ∇ is the gradient
operator.
"""
function DirectionalErtelPotentialVorticity(model, direction; location = (Face, Face, Face))
    validate_location(location, "DirectionalErtelPotentialVorticity", (Face, Face, Face))

    if model.buoyancy == nothing || !(model.buoyancy.model isa BuoyancyTracer)
        throw(ArgumentError("`DirectionalErtelPotentialVorticity` is only implemented for `BuoyancyTracer`"))
    end

    if model isa NonhydrostaticModel
        full_fields = add_background_fields(model)
        u, v, w, b = full_fields.u, full_fields.v, full_fields.w, full_fields.b
    else
        u, v, w = model.velocities
        b = model.tracers.b
    end

    coriolis = model.coriolis
    if coriolis != nothing
        if coriolis isa FPlane
            fx = fy = 0
            fz = coriolis.f
        elseif coriolis isa ConstantCartesianCoriolis
            fx = coriolis.fx
            fy = coriolis.fy
            fz = coriolis.fz
        else
        throw(ArgumentError("`DirectionalErtelPotentialVorticity` only implemented for `FPlane` and `ConstantCartesianCoriolis`"))
        end
        f_dir = sum([fx, fy, fz] .* direction)
    else
        f_dir = 0
    end

    dir_x, dir_y, dir_z = direction
    return KernelFunctionOperation{Face, Face, Face}(directional_ertel_potential_vorticity_fff, model.grid,
                                                     u, v, w, b, (; f_dir, dir_x, dir_y, dir_z))
end
#---

#+++ Velocity gradient tensor
@inline fψ_plus_gφ²(i, j, k, grid, f, ψ, g, φ) = @inbounds (f(i, j, k, grid, ψ) + g(i, j, k, grid, φ))^2

function strain_rate_tensor_modulus_ccc(i, j, k, grid, u, v, w)
    Sˣˣ² = ∂xᶜᶜᶜ(i, j, k, grid, u)^2
    Sʸʸ² = ∂yᶜᶜᶜ(i, j, k, grid, v)^2
    Sᶻᶻ² = ∂zᶜᶜᶜ(i, j, k, grid, w)^2

    Sˣʸ² = ℑxyᶜᶜᵃ(i, j, k, grid, fψ_plus_gφ², ∂yᶠᶠᶜ, u, ∂xᶠᶠᶜ, v) / 4
    Sˣᶻ² = ℑxzᶜᵃᶜ(i, j, k, grid, fψ_plus_gφ², ∂zᶠᶜᶠ, u, ∂xᶠᶜᶠ, w) / 4
    Sʸᶻ² = ℑyzᵃᶜᶜ(i, j, k, grid, fψ_plus_gφ², ∂zᶜᶠᶠ, v, ∂yᶜᶠᶠ, w) / 4

    return √(Sˣˣ² + Sʸʸ² + Sᶻᶻ² + 2 * (Sˣʸ² + Sˣᶻ² + Sʸᶻ²))
end

"""
    $(SIGNATURES)

Calculate the modulus (absolute value) of the strain rate tensor `S`, which is defined as the
symmetric part of the velocity gradient tensor:

    Sᵢⱼ = ½(∂ⱼuᵢ + ∂ᵢuⱼ)

Its modulus is then defined (using Einstein summation notation) as

    || Sᵢⱼ || = √( Sᵢⱼ Sᵢⱼ)

"""
function StrainRateTensorModulus(model; location = (Center, Center, Center))
    validate_location(location, "StrainRateTensorModulus", (Center, Center, Center))
    return KernelFunctionOperation{Center, Center, Center}(strain_rate_tensor_modulus_ccc, model.grid, model.velocities...)
end


@inline fψ_minus_gφ²(i, j, k, grid, f, ψ, g, φ) = @inbounds (f(i, j, k, grid, ψ) - g(i, j, k, grid, φ))^2

function vorticity_tensor_modulus_ccc(i, j, k, grid, u, v, w)
    Ωˣʸ² = ℑxyᶜᶜᵃ(i, j, k, grid, fψ_minus_gφ², ∂yᶠᶠᶜ, u, ∂xᶠᶠᶜ, v) / 4
    Ωˣᶻ² = ℑxzᶜᵃᶜ(i, j, k, grid, fψ_minus_gφ², ∂zᶠᶜᶠ, u, ∂xᶠᶜᶠ, w) / 4
    Ωʸᶻ² = ℑyzᵃᶜᶜ(i, j, k, grid, fψ_minus_gφ², ∂zᶜᶠᶠ, v, ∂yᶜᶠᶠ, w) / 4

    Ωʸˣ² = ℑxyᶜᶜᵃ(i, j, k, grid, fψ_minus_gφ², ∂xᶠᶠᶜ, v, ∂yᶠᶠᶜ, u) / 4
    Ωᶻˣ² = ℑxzᶜᵃᶜ(i, j, k, grid, fψ_minus_gφ², ∂xᶠᶜᶠ, w, ∂zᶠᶜᶠ, u) / 4
    Ωᶻʸ² = ℑyzᵃᶜᶜ(i, j, k, grid, fψ_minus_gφ², ∂yᶜᶠᶠ, w, ∂zᶜᶠᶠ, v) / 4

    return √(Ωˣʸ² + Ωˣᶻ² + Ωʸᶻ² + Ωʸˣ² + Ωᶻˣ² + Ωᶻʸ²)
end

"""
    $(SIGNATURES)

Calculate the modulus (absolute value) of the vorticity tensor `Ω`, which is defined as the
antisymmetric part of the velocity gradient tensor:

    Ωᵢⱼ = ½(∂ⱼuᵢ - ∂ᵢuⱼ)

Its modulus is then defined (using Einstein summation notation) as

    || Ωᵢⱼ || = √( Ωᵢⱼ Ωᵢⱼ)

"""
function VorticityTensorModulus(model; location = (Center, Center, Center))
    validate_location(location, "VorticityTensorModulus", (Center, Center, Center))
    return KernelFunctionOperation{Center, Center, Center}(vorticity_tensor_modulus_ccc, model.grid, model.velocities...)
end


# From 10.1063/1.5124245
@inline function Q_velocity_gradient_tensor_invariant_ccc(i, j, k, grid, u, v, w)
    S² = strain_rate_tensor_modulus_ccc(i, j, k, grid, u, v, w)^2
    Ω² = vorticity_tensor_modulus_ccc(i, j, k, grid, u, v, w)^2
    return (Ω² - S²) / 2
end

"""
    $(SIGNATURES)

Calculate the value of the `Q` velocity gradient tensor invariant. This is usually just called `Q`
and it is generally used for identifying and visualizing vortices in fluid flow.

The definition and nomenclature comes from the equation for the eigenvalues `λ` of the velocity
gradient tensor `∂ⱼuᵢ`:

    λ³ + P λ² + Q λ + T = 0

from where `Q` is defined as

    Q = ½ ( ΩᵢⱼΩᵢⱼ - SᵢⱼSᵢⱼ)

and where `Sᵢⱼ= ½(∂ⱼuᵢ + ∂ᵢuⱼ)` and `Ωᵢⱼ= ½(∂ⱼuᵢ - ∂ᵢuⱼ)`. More info about it can be found in
10.1063/1.5124245.
"""
function QVelocityGradientTensorInvariant(model; location = (Center, Center, Center))
    validate_location(location, "QVelocityGradientTensorInvariant", (Center, Center, Center))
    return KernelFunctionOperation{Center, Center, Center}(Q_velocity_gradient_tensor_invariant_ccc, model.grid, model.velocities...)
end

const Q = QVelocityGradientTensorInvariant
#---

end # module
