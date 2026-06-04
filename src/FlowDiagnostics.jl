module FlowDiagnostics
using DocStringExtensions

export RichardsonNumber, RossbyNumber
export ErtelPotentialVorticity, ThermalWindPotentialVorticity, DirectionalErtelPotentialVorticity
export StrainRateTensor, PrincipalStrainRates, StrainRateTensorModulus, VorticityTensorModulus, Q, QVelocityGradientTensorInvariant
export MixedLayerDepth, BuoyancyAnomalyCriterion, DensityAnomalyCriterion
export BottomCellValue

using Oceanostics: validate_location,
                   validate_dissipative_closure,
                   add_background_fields,
                   get_coriolis_frequency_components,
                   CustomKFO

using Oceananigans: NonhydrostaticModel, FPlane, ConstantCartesianCoriolis, BuoyancyTracer, location
using Oceananigans.BuoyancyFormulations: get_temperature_and_salinity, SeawaterBuoyancy, buoyancy_perturbationᶜᶜᶜ
using Oceananigans.Operators
using Oceananigans.AbstractOperations
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Models: buoyancy_operation
using Oceananigans.Grids: AbstractGrid, Center, Face, NegativeZDirection, ZDirection, znode, bottommost_active_node

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

const RichardsonNumber = CustomKFO{<:typeof(richardson_number_ccf)}

"""
    $(SIGNATURES)

Calculate the Richardson Number as

```
    Ri = (∂b/∂z) / (|∂u⃗ₕ/∂z|²)
```

where `z` is the true vertical direction (ie anti-parallel to gravity).
"""
function RichardsonNumber(model; loc = (Center, Center, Face))
    validate_location(loc, "RichardsonNumber", (Center, Center, Face))
    return RichardsonNumber(model, model.velocities..., buoyancy_operation(model); loc)
end

function RichardsonNumber(model, u, v, w, b; loc = (Center, Center, Face))
    validate_location(loc, "RichardsonNumber", (Center, Center, Face))

    if model.buoyancy.gravity_unit_vector isa NegativeZDirection
        true_vertical_direction = (0, 0, 1)
    elseif model.buoyancy.gravity_unit_vector isa ZDirection
        true_vertical_direction = (0, 0, -1)
    else
        true_vertical_direction = .-model.buoyancy.gravity_unit_vector
    end
    return RichardsonNumber(model, u, v, w, b, true_vertical_direction; loc = (Center, Center, Face))
end

function RichardsonNumber(model, u, v, w, b, true_vertical_direction; loc = (Center, Center, Face))
    validate_location(loc, "RichardsonNumber", (Center, Center, Face))
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

const RossbyNumber = CustomKFO{<:typeof(rossby_number_fff)}

"""
    $(SIGNATURES)

Calculate the Rossby number using the vorticity in the rotation axis direction according
to `model.coriolis`. Rossby number is defined as

```
    Ro = ωᶻ / f
```
where ωᶻ is the vorticity in the Coriolis axis of rotation and `f` is the Coriolis rotation frequency.
"""
function RossbyNumber(model; loc = (Face, Face, Face), add_background = true,
                      dWdy_bg=0, dVdz_bg=0,
                      dUdz_bg=0, dWdx_bg=0,
                      dUdy_bg=0, dVdx_bg=0)
    validate_location(loc, "RossbyNumber", (Face, Face, Face))

    if (model isa NonhydrostaticModel) & add_background
        full_fields = add_background_fields(model)
        u, v, w = full_fields.u, full_fields.v, full_fields.w
    else
        u, v, w = model.velocities
    end

    return RossbyNumber(model, u, v, w, model.coriolis; loc,
                        dWdy_bg, dVdz_bg, dUdz_bg, dWdx_bg, dUdy_bg, dVdx_bg)
end

function RossbyNumber(model, u, v, w, coriolis; loc = (Face, Face, Face),
                      dWdy_bg=0, dVdz_bg=0,
                      dUdz_bg=0, dWdx_bg=0,
                      dUdy_bg=0, dVdx_bg=0)
    validate_location(loc, "RossbyNumber", (Face, Face, Face))

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
    ThermalWindPotentialVorticity

Narrower type alias matching only the thermal-wind variant of
[`ErtelPotentialVorticity`](@ref). Useful for identifying or dispatching on the
thermal-wind variant via `isa`. Construct via
`ErtelPotentialVorticity(model; thermal_wind = true)`.
"""
const ThermalWindPotentialVorticity = CustomKFO{<:typeof(potential_vorticity_in_thermal_wind_fff)}

const ErtelPotentialVorticity = CustomKFO{<:Union{typeof(ertel_potential_vorticity_fff),
                                                  typeof(potential_vorticity_in_thermal_wind_fff)}}

"""
    $(SIGNATURES)

Calculate the Ertel Potential Vorticty for `model`, where the characteristics of
the Coriolis rotation are taken from `model.coriolis`. The Ertel Potential Vorticity
is defined as

    EPV = ωₜₒₜ ⋅ ∇b

where ωₜₒₜ is the total (relative + planetary) vorticity vector, `b` is the buoyancy and ∇ is the gradient
operator.

If `thermal_wind = true`, the thermal-wind approximation is used instead, giving

```
    EPV = (f + ωᶻ) ∂b/∂z - f ((∂U/∂z)² + (∂V/∂z)²)
```

where `f` is the (vertical component of the) Coriolis frequency, `ωᶻ` is the vertical relative vorticity,
and `∂U/∂z`, `∂V/∂z` comprise the thermal wind shear. The returned object is an instance of both
`ErtelPotentialVorticity` and `ThermalWindPotentialVorticity`, so the thermal-wind variant can be
identified separately.

```jldoctest
julia> using Oceananigans

julia> grid = RectilinearGrid(topology = (Flat, Flat, Bounded), size = 4, extent = 1);

julia> N² = 1e-6;

julia> b_bcs = FieldBoundaryConditions(top=GradientBoundaryCondition(N²));

julia> model = NonhydrostaticModel(grid; coriolis=FPlane(f=1e-4), buoyancy=BuoyancyTracer(), tracers=:b, boundary_conditions=(; b=b_bcs));

julia> stratification(z) = N² * z;

julia> set!(model, b=stratification)

julia> using Oceanostics: ErtelPotentialVorticity

julia> EPV = ErtelPotentialVorticity(model)
KernelFunctionOperation at (Face, Face, Face)
├── grid: 1×1×4 RectilinearGrid{Float64, Flat, Flat, Bounded} on CPU with 0×0×3 halo
├── kernel_function: ertel_potential_vorticity_fff (generic function with 1 method)
└── arguments: ("Field", "Field", "Field", "Field", "Int64", "Int64", "Float64")

julia> interior(Field(EPV))
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
function ErtelPotentialVorticity(model; tracer_name = :b, thermal_wind = false, loc = (Face, Face, Face))
    validate_location(loc, "ErtelPotentialVorticity", (Face, Face, Face))
    return ErtelPotentialVorticity(model, model.velocities..., model.tracers[tracer_name], model.coriolis;
                                   thermal_wind, loc)
end

function ErtelPotentialVorticity(model, u, v, w, tracer, coriolis; thermal_wind = false, loc = (Face, Face, Face))
    validate_location(loc, "ErtelPotentialVorticity", (Face, Face, Face))
    fx, fy, fz = get_coriolis_frequency_components(coriolis)
    if thermal_wind
        return KernelFunctionOperation{Face, Face, Face}(potential_vorticity_in_thermal_wind_fff, model.grid,
                                                         u, v, tracer, fz)
    end
    return KernelFunctionOperation{Face, Face, Face}(ertel_potential_vorticity_fff, model.grid,
                                                     u, v, w, tracer, fx, fy, fz)
end

function ErtelPotentialVorticity(model, u, v, tracer, coriolis; thermal_wind = true, loc = (Face, Face, Face))
    thermal_wind || throw(ArgumentError("ErtelPotentialVorticity called without `w` requires `thermal_wind = true`"))
    validate_location(loc, "ErtelPotentialVorticity", (Face, Face, Face))
    _, _, fz = get_coriolis_frequency_components(coriolis)
    return KernelFunctionOperation{Face, Face, Face}(potential_vorticity_in_thermal_wind_fff, model.grid,
                                                     u, v, tracer, fz)
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

const DirectionalErtelPotentialVorticity = CustomKFO{<:typeof(directional_ertel_potential_vorticity_fff)}

"""
    $(SIGNATURES)

Calculate the contribution from a given `direction` to the Ertel Potential Vorticity
basde on a `model` and a `direction`. The Ertel Potential Vorticity is defined as

    EPV = ωₜₒₜ ⋅ ∇b

where ωₜₒₜ is the total (relative + planetary) vorticity vector, `b` is the buoyancy and ∇ is the gradient
operator.
"""
function DirectionalErtelPotentialVorticity(model, direction; tracer_name = :b, loc = (Face, Face, Face))
    validate_location(loc, "DirectionalErtelPotentialVorticity", (Face, Face, Face))
    return DirectionalErtelPotentialVorticity(model, direction, model.velocities..., model.tracers[tracer_name], model.coriolis; loc)
end


function DirectionalErtelPotentialVorticity(model, direction, u, v, w, tracer, coriolis; loc = (Face, Face, Face))
    validate_location(loc, "DirectionalErtelPotentialVorticity", (Face, Face, Face))

    fx, fy, fz = get_coriolis_frequency_components(coriolis)
    f_dir = sum([fx, fy, fz] .* direction)

    dir_x, dir_y, dir_z = direction
    return KernelFunctionOperation{Face, Face, Face}(directional_ertel_potential_vorticity_fff, model.grid,
                                                     u, v, w, tracer, (; f_dir, dir_x, dir_y, dir_z))
end
#---

#+++ Velocity gradient and vorticity tensors
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

const StrainRateTensorModulus = CustomKFO{<:typeof(strain_rate_tensor_modulus_ccc)}

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
function StrainRateTensorModulus(model; loc = (Center, Center, Center))
    validate_location(loc, "StrainRateTensorModulus", (Center, Center, Center))
    return KernelFunctionOperation{Center, Center, Center}(strain_rate_tensor_modulus_ccc, model.grid, model.velocities...)
end

# Off-diagonal strain rate components, each evaluated at its natural staggered location.
@inline strain_rate_tensor_xy_ffc(i, j, k, grid, u, v) = (∂yᶠᶠᶜ(i, j, k, grid, u) + ∂xᶠᶠᶜ(i, j, k, grid, v)) / 2
@inline strain_rate_tensor_xz_fcf(i, j, k, grid, u, w) = (∂zᶠᶜᶠ(i, j, k, grid, u) + ∂xᶠᶜᶠ(i, j, k, grid, w)) / 2
@inline strain_rate_tensor_yz_cff(i, j, k, grid, v, w) = (∂zᶜᶠᶠ(i, j, k, grid, v) + ∂yᶜᶠᶠ(i, j, k, grid, w)) / 2

"""
    $(SIGNATURES)

Return the components of the strain rate tensor `S`, defined as the symmetric part of the velocity
gradient tensor:

```
    Sᵢⱼ = ½(∂ⱼuᵢ + ∂ᵢuⱼ)
```

The result is a `NamedTuple` with the 6 independent components, each a `KernelFunctionOperation`
living at its natural location on the staggered grid:

| Component | Definition         | Location |
|:---------:|:------------------:|:--------:|
| `S₁₁`     | `∂u/∂x`            | `ccc`    |
| `S₂₂`     | `∂v/∂y`            | `ccc`    |
| `S₃₃`     | `∂w/∂z`            | `ccc`    |
| `S₁₂`     | `½(∂u/∂y + ∂v/∂x)` | `ffc`    |
| `S₁₃`     | `½(∂u/∂z + ∂w/∂x)` | `fcf`    |
| `S₂₃`     | `½(∂v/∂z + ∂w/∂y)` | `cff`    |

The tensor is symmetric, so the remaining components follow from `Sⱼᵢ = Sᵢⱼ` (i.e. `S₂₁ = S₁₂`,
`S₃₁ = S₁₃`, `S₃₂ = S₂₃`). Each component can be wrapped in a `Field` and used with output writers,
time-averaging, etc. Can also be called as `StrainRateTensor(grid, u, v, w)` to build the components
from individual velocity fields. See also [`StrainRateTensorModulus`](@ref) for the scalar modulus
`√(SᵢⱼSᵢⱼ)`.
"""
StrainRateTensor(model) = StrainRateTensor(model.grid, model.velocities...)

function StrainRateTensor(grid::AbstractGrid, u, v, w)
    S₁₁ = KernelFunctionOperation{Center, Center, Center}(∂xᶜᶜᶜ, grid, u)
    S₂₂ = KernelFunctionOperation{Center, Center, Center}(∂yᶜᶜᶜ, grid, v)
    S₃₃ = KernelFunctionOperation{Center, Center, Center}(∂zᶜᶜᶜ, grid, w)

    S₁₂ = KernelFunctionOperation{Face, Face, Center}(strain_rate_tensor_xy_ffc, grid, u, v)
    S₁₃ = KernelFunctionOperation{Face, Center, Face}(strain_rate_tensor_xz_fcf, grid, u, w)
    S₂₃ = KernelFunctionOperation{Center, Face, Face}(strain_rate_tensor_yz_cff, grid, v, w)

    return (; S₁₁, S₂₂, S₃₃, S₁₂, S₁₃, S₂₃)
end

# Analytical eigenvalues of the symmetric 3×3 matrix [a d e; d b f; e f c], returned sorted as
# λ₁ ≥ λ₂ ≥ λ₃ (Smith's trigonometric method). Operates entirely on scalars, so it stays
# allocation-free and GPU-safe inside a kernel without pulling in an external tensor package.
@inline function eigvals_symmetric_3x3(a, b, c, d, e, f)
    p1 = d^2 + e^2 + f^2
    q  = (a + b + c) / 3
    p2 = (a - q)^2 + (b - q)^2 + (c - q)^2 + 2p1
    p  = sqrt(p2 / 6)
    p == 0 && return (q, q, q) # A is isotropic (A = qI), e.g. a quiescent flow

    a′ = a - q; b′ = b - q; c′ = c - q
    detB = (a′ * (b′*c′ - f^2) - d * (d*c′ - e*f) + e * (d*f - e*b′)) / p^3
    r = clamp(detB / 2, -one(p), one(p)) # clamp absorbs roundoff so acos stays in range
    φ = acos(r) / 3

    λ₁ = q + 2p * cos(φ)
    λ₃ = q + 2p * cos(φ + oftype(φ, 2π/3))
    λ₂ = 3q - λ₁ - λ₃ # the eigenvalues sum to the trace
    return (λ₁, λ₂, λ₃)
end

# Strain rate components co-located at ccc (off-diagonals interpolated from their natural edges),
# fed to the symmetric eigensolver.
@inline function strain_rate_eigenvalues_ccc(i, j, k, grid, u, v, w)
    S₁₁ = ∂xᶜᶜᶜ(i, j, k, grid, u)
    S₂₂ = ∂yᶜᶜᶜ(i, j, k, grid, v)
    S₃₃ = ∂zᶜᶜᶜ(i, j, k, grid, w)
    S₁₂ = ℑxyᶜᶜᵃ(i, j, k, grid, strain_rate_tensor_xy_ffc, u, v)
    S₁₃ = ℑxzᶜᵃᶜ(i, j, k, grid, strain_rate_tensor_xz_fcf, u, w)
    S₂₃ = ℑyzᵃᶜᶜ(i, j, k, grid, strain_rate_tensor_yz_cff, v, w)
    return eigvals_symmetric_3x3(S₁₁, S₂₂, S₃₃, S₁₂, S₁₃, S₂₃)
end

@inline strain_rate_eigenvalue_1_ccc(i, j, k, grid, u, v, w) = strain_rate_eigenvalues_ccc(i, j, k, grid, u, v, w)[1]
@inline strain_rate_eigenvalue_2_ccc(i, j, k, grid, u, v, w) = strain_rate_eigenvalues_ccc(i, j, k, grid, u, v, w)[2]
@inline strain_rate_eigenvalue_3_ccc(i, j, k, grid, u, v, w) = strain_rate_eigenvalues_ccc(i, j, k, grid, u, v, w)[3]

"""
    $(SIGNATURES)

Return the principal strain rates — the eigenvalues of the strain rate tensor
`Sᵢⱼ = ½(∂ⱼuᵢ + ∂ᵢuⱼ)` — as a `NamedTuple` `(; λ₁, λ₂, λ₃)` ordered so that `λ₁ ≥ λ₂ ≥ λ₃`. Each is a
`KernelFunctionOperation` at `(Center, Center, Center)`: the full tensor is assembled there (the
off-diagonal components are interpolated from their natural edge locations) and its eigenvalues are
computed analytically with a symmetric-3×3 eigensolver.

The principal strain rates describe stretching (`λ > 0`) and compression (`λ < 0`) along the
principal axes. They are rotation invariants of the tensor: `λ₁ + λ₂ + λ₃ = ∇·u` and
`λ₁² + λ₂² + λ₃² = SᵢⱼSᵢⱼ`, so for incompressible flow they sum to zero and their root-sum-of-squares
equals [`StrainRateTensorModulus`](@ref). Can also be called as
`PrincipalStrainRates(grid, u, v, w)`. See also [`StrainRateTensor`](@ref) for the tensor components.
"""
PrincipalStrainRates(model) = PrincipalStrainRates(model.grid, model.velocities...)

function PrincipalStrainRates(grid::AbstractGrid, u, v, w)
    λ₁ = KernelFunctionOperation{Center, Center, Center}(strain_rate_eigenvalue_1_ccc, grid, u, v, w)
    λ₂ = KernelFunctionOperation{Center, Center, Center}(strain_rate_eigenvalue_2_ccc, grid, u, v, w)
    λ₃ = KernelFunctionOperation{Center, Center, Center}(strain_rate_eigenvalue_3_ccc, grid, u, v, w)
    return (; λ₁, λ₂, λ₃)
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

const VorticityTensorModulus = CustomKFO{<:typeof(vorticity_tensor_modulus_ccc)}

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
function VorticityTensorModulus(model; loc = (Center, Center, Center))
    validate_location(loc, "VorticityTensorModulus", (Center, Center, Center))
    return KernelFunctionOperation{Center, Center, Center}(vorticity_tensor_modulus_ccc, model.grid, model.velocities...)
end


# From doi:10.1063/1.5124245
@inline function Q_velocity_gradient_tensor_invariant_ccc(i, j, k, grid, u, v, w)
    S² = strain_rate_tensor_modulus_ccc(i, j, k, grid, u, v, w)^2
    Ω² = vorticity_tensor_modulus_ccc(i, j, k, grid, u, v, w)^2
    return (Ω² - S²) / 2
end

const QVelocityGradientTensorInvariant = CustomKFO{<:typeof(Q_velocity_gradient_tensor_invariant_ccc)}
#---

#+++ Mixed layer depth
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
function QVelocityGradientTensorInvariant(model; loc = (Center, Center, Center))
    validate_location(loc, "QVelocityGradientTensorInvariant", (Center, Center, Center))
    return KernelFunctionOperation{Center, Center, Center}(Q_velocity_gradient_tensor_invariant_ccc, model.grid, model.velocities...)
end

const Q = QVelocityGradientTensorInvariant

"""
    $(TYPEDEF)
"""
struct MixedLayerDepthKernel{C}
    criterion::C
end

"""
    $(SIGNATURES)

Returns the mixed layer depth defined as the depth at which `criterion` is true.

Defaults to `DensityAnomalyCriterion` where the depth is that at which the density
is some threshold (defaults to 0.125kg/m³) higher than the surface density.

When `DensityAnomalyCriterion` is used, the arguments `buoyancy_formulation` and `C` should be
supplied where `buoyancy_formulation` should be the buoyancy model, and `C` should be a named
tuple of `(; T, S)`, `(; T)` or `(; S)` (the latter two if the buoyancy model
specifies a constant salinity or temperature).
"""
function MixedLayerDepth(grid::AbstractGrid, args...; criterion = BuoyancyAnomalyCriterion(convert(eltype(grid), -1e-4 * Oceananigans.defaults.gravitational_acceleration)))
    validate_criterion_model(criterion, args...)
    MLD = MixedLayerDepthKernel(criterion)
    return KernelFunctionOperation{Center, Center, Nothing}(MLD, grid, args...)
end

@inline function (MLD::MixedLayerDepthKernel)(i, j, k, grid, args...)
    kₘₗ = -1

    for k in grid.Nz-1:-1:1
        below_mixed_layer = MLD.criterion(i, j, k, grid, args...)
        kₘₗ = ifelse(below_mixed_layer & (kₘₗ < 0), k, kₘₗ)
    end

    zₘₗ = interpolate_from_nearest_cell(MLD.criterion, i, j, kₘₗ, grid, args...)
    return ifelse(kₘₗ == -1, -Inf, zₘₗ)
end

"""
    $(TYPEDEF)

An abstract mixed layer depth criterion where the mixed layer is defined to be
`anomaly` + `threshold` greater than the surface value of `anomaly`.

`AbstractAnomalyCriterion` types should provide a method for the function `anomaly` in the form
`anomaly(criterion, i, j, k, grid, args...)`, and should have a property `threshold`.
"""
abstract type AbstractAnomalyCriterion end

@inline function (criterion::AbstractAnomalyCriterion)(i, j, k, grid, args...)
    δ = criterion.threshold

    ref = (anomaly(criterion, i, j, grid.Nz, grid, args...) + anomaly(criterion, i, j, grid.Nz+1, grid, args...)) * convert(eltype(grid), 0.5)
    val = anomaly(criterion, i, j, k, grid,args...)

    return val < ref + δ
end

@inline function interpolate_from_nearest_cell(criterion::AbstractAnomalyCriterion, i, j, k, grid, args...)
    δ = criterion.threshold

    ref = (anomaly(criterion, i, j, grid.Nz, grid, args...) + anomaly(criterion, i, j, grid.Nz + 1, grid, args...)) * convert(eltype(grid), 0.5)

    k_val  = anomaly(criterion, i, j, k, grid, args...)
    k⁺_val = anomaly(criterion, i, j, k + 1, grid, args...)

    zₖ = znode(i, j, k, grid, Center(), Center(), Center())
    z₊ = znode(i, j, k+1, grid, Center(), Center(), Center())

    return zₖ + (z₊ - zₖ) * (ref + δ - k_val) / (k⁺_val - k_val)
end

"""
    $(TYPEDEF)

Defines the mixed layer to be the depth at which the buoyancy is more than `threshold` greater than
the surface buoyancy (but the pertubaton is usually negative).

When this model is used, the arguments `buoyancy_formulation` and `C` should be supplied where `C`
should be the named tuple `(; b)`, with `b` the buoyancy tracer.
"""
@kwdef struct BuoyancyAnomalyCriterion{FT} <: AbstractAnomalyCriterion
    threshold :: FT = -1e-4 * Oceananigans.defaults.gravitational_acceleration
end

validate_criterion_model(::BuoyancyAnomalyCriterion, args...) =
    @error "For BuoyancyAnomalyCriterion you must supply the arguments `buoyancy_formulation` and `C`, where `C` is the named tuple `(; b)`, with `b` the buoyancy tracer."

validate_criterion_model(::BuoyancyAnomalyCriterion, buoyancy_formulation, C) = nothing

@inline anomaly(::BuoyancyAnomalyCriterion, i, j, k, grid, buoyancy_formulation, C) = buoyancy_perturbationᶜᶜᶜ(i, j, k, grid, buoyancy_formulation, C)

"""
    $(TYPEDEF)

Defines the mixed layer to be the depth at which the density is more than `threshold`
greater than the surface density.

When this model is used, the arguments `buoyancy_formulation` and `C` should be supplied where
`buoyancy_formulation` should be the buoyancy model, and `C` should be a named tuple of `(; T, S)`,
`(; T)` or `(; S)` (the latter two if the buoyancy model specifies a constant salinity or
temperature).
"""
@kwdef struct DensityAnomalyCriterion{FT} <: AbstractAnomalyCriterion
             reference_density :: FT = 1020.0
    gravitational_acceleration :: FT = Oceananigans.defaults.gravitational_acceleration
                     threshold :: FT = 0.125
end

function DensityAnomalyCriterion(buoyancy_formulation::SeawaterBuoyancy{<:Any, <:BoussinesqEquationOfState}; threshold = 0.125)
    ρᵣ = buoyancy_formulation.equation_of_state.reference_density
    g  = buoyancy_formulation.gravitational_acceleration

    return DensityAnomalyCriterion(ρᵣ, g, threshold)
end

validate_criterion_model(::DensityAnomalyCriterion, args...) =
    @error "For DensityAnomalyCriterion you must supply the arguments buoyancy_formulation and C, where C is a named tuple of (; T, S), (; T) or (; S)"

validate_criterion_model(::DensityAnomalyCriterion, buoyancy_formulation, C) = nothing
    
@inline function anomaly(criterion::DensityAnomalyCriterion, i, j, k, grid, buoyancy_formulation, C)
    b = buoyancy_perturbationᶜᶜᶜ(i, j, k, grid, buoyancy_formulation, C)

    ρᵣ = criterion.reference_density
    g  = criterion.gravitational_acceleration
    return - ρᵣ * b / g
end
#---

#+++ Bottom value
"""
    $(SIGNATURES)

Returns the value of the given `diagnostic` at the bottom, which can be either the bottom of the
domain (lowest vertical level) or an immersed bottom.
"""
function BottomCellValue(diagnostic)
    loc = location(diagnostic)
    instantiated_location = (L() for L in loc)
    condition_to_ban = KernelFunctionOperation{loc...}(bottommost_active_node, diagnostic.grid, instantiated_location...)
    return diagnostic * condition_to_ban
end
#---

end # module
