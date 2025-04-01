module FlowDiagnostics
using DocStringExtensions

export RichardsonNumber, RossbyNumber
export ErtelPotentialVorticity, ThermalWindPotentialVorticity, DirectionalErtelPotentialVorticity
export StrainRateTensorModulus, VorticityTensorModulus, Q, QVelocityGradientTensorInvariant
export MixedLayerDepth, DensityAnomalyCriterion

using Oceanostics: validate_location,
                   validate_dissipative_closure,
                   add_background_fields,
                   get_coriolis_frequency_components

using Oceananigans: NonhydrostaticModel, FPlane, ConstantCartesianCoriolis, BuoyancyField, BuoyancyTracer
using Oceananigans.BuoyancyFormulations: get_temperature_and_salinity, SeawaterBuoyancy
using Oceananigans.Operators
using Oceananigans.AbstractOperations
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.BuoyancyFormulations: buoyancy
using Oceananigans.Grids: Center, Face, NegativeZDirection, ZDirection, znode


using SeawaterPolynomials: ρ′, BoussinesqEquationOfState
using SeawaterPolynomials.SecondOrderSeawaterPolynomials: SecondOrderSeawaterPolynomial

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

```
    Ri = (∂b/∂z) / (|∂u⃗ₕ/∂z|²)
```

where `z` is the true vertical direction (ie anti-parallel to gravity).
"""
function RichardsonNumber(model; location = (Center, Center, Face))
    validate_location(location, "RichardsonNumber", (Center, Center, Face))
    return RichardsonNumber(model, model.velocities..., buoyancy(model); location)
end

function RichardsonNumber(model, u, v, w, b; location = (Center, Center, Face))
    validate_location(location, "RichardsonNumber", (Center, Center, Face))

    if model.buoyancy.gravity_unit_vector isa NegativeZDirection
        true_vertical_direction = (0, 0, 1)
    elseif model.buoyancy.gravity_unit_vector isa ZDirection
        true_vertical_direction = (0, 0, -1)
    else
        true_vertical_direction = .-model.buoyancy.gravity_unit_vector
    end
    return RichardsonNumber(model, u, v, w, b, true_vertical_direction; location = (Center, Center, Face))
end

function RichardsonNumber(model, u, v, w, b, true_vertical_direction; location = (Center, Center, Face))
    validate_location(location, "RichardsonNumber", (Center, Center, Face))
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

```
    Ro = ωᶻ / f
```
where ωᶻ is the vorticity in the Coriolis axis of rotation and `f` is the Coriolis rotation frequency.
"""
function RossbyNumber(model; location = (Face, Face, Face), add_background = true,
                      dWdy_bg=0, dVdz_bg=0,
                      dUdz_bg=0, dWdx_bg=0,
                      dUdy_bg=0, dVdx_bg=0)
    validate_location(location, "RossbyNumber", (Face, Face, Face))

    if (model isa NonhydrostaticModel) & add_background
        full_fields = add_background_fields(model)
        u, v, w = full_fields.u, full_fields.v, full_fields.w
    else
        u, v, w = model.velocities
    end

    return RossbyNumber(model, u, v, w, model.coriolis; location,
                        dWdy_bg, dVdz_bg, dUdz_bg, dWdx_bg, dUdy_bg, dVdx_bg)
end

function RossbyNumber(model, u, v, w, coriolis; location = (Face, Face, Face),
                      dWdy_bg=0, dVdz_bg=0,
                      dUdz_bg=0, dWdx_bg=0,
                      dUdy_bg=0, dVdx_bg=0)
    validate_location(location, "RossbyNumber", (Face, Face, Face))

    fx, fy, fz = get_coriolis_frequency_components(coriolis)

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

```
    TWPV = (f + ωᶻ) ∂b/∂z - f ((∂U/∂z)² + (∂V/∂z)²)
```
where `f` is the Coriolis frequency, `ωᶻ` is the relative vorticity in the `z` direction, `b` is the buoyancy, and
`∂U/∂z` and `∂V/∂z` comprise the thermal wind shear.
"""
function ThermalWindPotentialVorticity(model; tracer = :b, location = (Face, Face, Face))
    validate_location(location, "ThermalWindPotentialVorticity", (Face, Face, Face))
    u, v, w = model.velocities
    return ThermalWindPotentialVorticity(model, u, v, model.tracers[tracer], model.coriolis; location)
end

function ThermalWindPotentialVorticity(model, u, v, tracer, coriolis; location = (Face, Face, Face))
    validate_location(location, "ThermalWindPotentialVorticity", (Face, Face, Face))
    fx, fy, fz = get_coriolis_frequency_components(coriolis)
    return KernelFunctionOperation{Face, Face, Face}(potential_vorticity_in_thermal_wind_fff, model.grid,
                                                     u, v, tracer, fz)
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

```jldoctest
julia> using Oceananigans

julia> grid = RectilinearGrid(topology = (Flat, Flat, Bounded), size = 4, extent = 1);

julia> N² = 1e-6;

julia> b_bcs = FieldBoundaryConditions(top=GradientBoundaryCondition(N²));

julia> model = NonhydrostaticModel(; grid, coriolis=FPlane(1e-4), buoyancy=BuoyancyTracer(), tracers=:b, boundary_conditions=(; b=b_bcs));

julia> stratification(z) = N² * z;

julia> set!(model, b=stratification)

julia> using Oceanostics: ErtelPotentialVorticity

julia> EPV = ErtelPotentialVorticity(model)
KernelFunctionOperation at (Face, Face, Face)
├── grid: 1×1×4 RectilinearGrid{Float64, Flat, Flat, Bounded} on CPU with 0×0×3 halo
├── kernel_function: ertel_potential_vorticity_fff (generic function with 1 method)
└── arguments: ("1×1×4 Field{Face, Center, Center} on RectilinearGrid on CPU", "1×1×4 Field{Center, Face, Center} on RectilinearGrid on CPU", "1×1×5 Field{Center, Center, Face} on RectilinearGrid on CPU", "1×1×4 Field{Center, Center, Center} on RectilinearGrid on CPU", "0", "0", "0.0001")

julia> interior(compute!(Field(EPV)))
1×1×5 view(::Array{Float64, 3}, 1:1, 1:1, 4:8) with eltype Float64:
[:, :, 1] =
 0.0

[:, :, 2] =
 1.0000000000000002e-10

[:, :, 3] =
 9.999999999999998e-11

[:, :, 4] =
 1.0000000000000002e-10

[:, :, 5] =
 1.0e-10
```

Note that EPV values are correctly calculated both in the interior and the boundaries. In the
interior and top boundary, EPV = f×N² = 10⁻¹⁰, while EPV = 0 at the bottom boundary since ∂b/∂z
is zero there.
"""
function ErtelPotentialVorticity(model; tracer = :b, location = (Face, Face, Face))
    validate_location(location, "ErtelPotentialVorticity", (Face, Face, Face))
    return ErtelPotentialVorticity(model, model.velocities..., model.tracers[tracer], model.coriolis; location)
end

function ErtelPotentialVorticity(model, u, v, w, tracer, coriolis; location = (Face, Face, Face))
    validate_location(location, "ErtelPotentialVorticity", (Face, Face, Face))
    fx, fy, fz = get_coriolis_frequency_components(coriolis)
    return KernelFunctionOperation{Face, Face, Face}(ertel_potential_vorticity_fff, model.grid,
                                                     u, v, w, tracer, fx, fy, fz)
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
function DirectionalErtelPotentialVorticity(model, direction; tracer = :b, location = (Face, Face, Face))
    validate_location(location, "DirectionalErtelPotentialVorticity", (Face, Face, Face))
    return DirectionalErtelPotentialVorticity(model, direction, model.velocities..., model.tracers[tracer], model.coriolis; location)
end


function DirectionalErtelPotentialVorticity(model, direction, u, v, w, tracer, coriolis; location = (Face, Face, Face))
    validate_location(location, "DirectionalErtelPotentialVorticity", (Face, Face, Face))

    fx, fy, fz = get_coriolis_frequency_components(coriolis)
    f_dir = sum([fx, fy, fz] .* direction)

    dir_x, dir_y, dir_z = direction
    return KernelFunctionOperation{Face, Face, Face}(directional_ertel_potential_vorticity_fff, model.grid,
                                                     u, v, w, tracer, (; f_dir, dir_x, dir_y, dir_z))
end
#---

#+++ Velocity gradient tensor
@inline fψ_plus_gφ²(i, j, k, grid, f, ψ, g, φ) = (f(i, j, k, grid, ψ) + g(i, j, k, grid, φ))^2

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

```
    Sᵢⱼ = ½(∂ⱼuᵢ + ∂ᵢuⱼ)
```
Its modulus is then defined (using Einstein summation notation) as

```
    || Sᵢⱼ || = √(Sᵢⱼ Sᵢⱼ)
```
"""
function StrainRateTensorModulus(model; location = (Center, Center, Center))
    validate_location(location, "StrainRateTensorModulus", (Center, Center, Center))
    return KernelFunctionOperation{Center, Center, Center}(strain_rate_tensor_modulus_ccc, model.grid, model.velocities...)
end


@inline fψ_minus_gφ²(i, j, k, grid, f, ψ, g, φ) = (f(i, j, k, grid, ψ) - g(i, j, k, grid, φ))^2

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

```
    Ωᵢⱼ = ½(∂ⱼuᵢ - ∂ᵢuⱼ)
```
Its modulus is then defined (using Einstein summation notation) as

```
    || Ωᵢⱼ || = √(Ωᵢⱼ Ωᵢⱼ)
```
"""
function VorticityTensorModulus(model; location = (Center, Center, Center))
    validate_location(location, "VorticityTensorModulus", (Center, Center, Center))
    return KernelFunctionOperation{Center, Center, Center}(vorticity_tensor_modulus_ccc, model.grid, model.velocities...)
end


# From doi:10.1063/1.5124245
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

```
    λ³ + P λ² + Q λ + T = 0
```
from where `Q` is defined as

```
    Q = ½ (ΩᵢⱼΩᵢⱼ - SᵢⱼSᵢⱼ)
```
and where `Sᵢⱼ= ½(∂ⱼuᵢ + ∂ᵢuⱼ)` and `Ωᵢⱼ= ½(∂ⱼuᵢ - ∂ᵢuⱼ)`. More info about it can be found in
doi:10.1063/1.5124245.
"""
function QVelocityGradientTensorInvariant(model; location = (Center, Center, Center))
    validate_location(location, "QVelocityGradientTensorInvariant", (Center, Center, Center))
    return KernelFunctionOperation{Center, Center, Center}(Q_velocity_gradient_tensor_invariant_ccc, model.grid, model.velocities...)
end

const Q = QVelocityGradientTensorInvariant
struct MixedLayerDepth{C}
    criterion::C
end

"""
    $(SIGNATURES)

Returns the mixed layer depth defined as the depth at which `criterion` is true.

Defaults to `DensityAnomalyCriterion` where the depth is that at which the density
is some threshold (defaults to 0.125kg/m³) higher than the surface density.

When `DensityAnomalyCriterion` is used, the arguments `buoyancy` and `C` should be 
supploed where `buoyancy` should be the buoyancy model, and `C` should be a named 
tuple of `(; T, S)`, `(; T)` or `(; S)` (the latter two if the buoyancy model 
specifies a constant salinity or temperature).
"""
function MixedLayerDepth(grid, args...; criterion = DensityAnomalyCriterion(convert(eltype(grid), 1/8)))
    validate_criterion_model(criterion, args...)

    MLD = MixedLayerDepth(criterion)

    return KernelFunctionOperation{Center, Center, Nothing}(MLD, grid, args...)
end

function (MLD::MixedLayerDepth)(i, j, k, grid, args...)
    kₘₗ = -1

    for k in grid.Nz-1:-1:1
        below_mixed_layer = MLD.criterion(i, j, k, grid, args...)

        kₘₗ = ifelse(below_mixed_layer & (kₘₗ < 0), k, kₘₗ)
    end

    zₘₗ = interpolate_from_nearest_cell(MLD.criterion, i, j, kₘₗ, grid, args...)
    
    return ifelse(kₘₗ == -1, -Inf, zₘₗ)
end

"""
    $(SIGNATURES)

Defines the mixed layer to be the depth at which the density is more than `pertubation`
greater than the surface density.

When this model is used, the arguments `buoyancy` and `C` should be 
supploed where `buoyancy` should be the buoyancy model, and `C` should be a named 
tuple of `(; T, S)`, `(; T)` or `(; S)` (the latter two if the buoyancy model 
specifies a constant salinity or temperature).
"""
struct DensityAnomalyCriterion{FT}
    pertubation :: FT
end

validate_criterion_model(::DensityAnomalyCriterion, args...) = 
    @error "For DensityAnomalyCriterion you must supply the arguments buoyancy and C, where C is a named tuple of (; T, S), (; T) or (; S)"

validate_criterion_model(::DensityAnomalyCriterion, buoyancy, C) = 
    @error "DensityAnomalyCriterion is not implemented for your buoyancy, model as the density anomaly is not defined"
    
validate_criterion_model(::DensityAnomalyCriterion, ::SeawaterBuoyancy{<:Any, <:BoussinesqEquationOfState}, C::NamedTuple) = nothing

@inline function density(i, j, k, grid, b::SeawaterBuoyancy{<:Any, <:BoussinesqEquationOfState}, C)
    T, S = get_temperature_and_salinity(b, C)

    return ρ′(i, j, k, grid, b.equation_of_state, T, S)
end

@inline function (density_criterion::DensityAnomalyCriterion)(i, j, k, grid, buoyancy, C)
    δρ = density_criterion.pertubation

    ρᵣ = (density(i, j, grid.Nz, grid, buoyancy, C) + density(i, j, grid.Nz+1, grid, buoyancy, C)) * convert(eltype(grid), 0.5)

    ρₖ = density(i, j, k, grid, buoyancy, C)

    return ρₖ > ρᵣ + δρ
end

@inline function interpolate_from_nearest_cell(density_criterion::DensityAnomalyCriterion, i, j, k, grid, buoyancy, C)
    δρ = density_criterion.pertubation

    ρᵣ = (density(i, j, grid.Nz, grid, buoyancy, C) + density(i, j, grid.Nz+1, grid, buoyancy, C)) * convert(eltype(grid), 0.5)

    ρₖ = density(i, j, k, grid, buoyancy, C)
    ρ₊ = density(i, j, k+1, grid, buoyancy, C)

    zₖ = znode(i, j, k, grid, Center(), Center(), Center())
    z₊ = znode(i, j, k+1, grid, Center(), Center(), Center())

    return zₖ + (z₊ - zₖ) * (ρᵣ + δρ - ρₖ) / (ρ₊ - ρₖ)
end
end # module
