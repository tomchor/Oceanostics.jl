module FlowDiagnostics
using DocStringExtensions

export RichardsonNumber, RossbyNumber
export ErtelPotentialVorticity, ThermalWindPotentialVorticity, DirectionalErtelPotentialVorticity
export StrainRateTensor, StrainRateTensorModulus, VorticityTensor, VorticityTensorModulus, Q, QVelocityGradientTensorInvariant
export StressTensor
export MixedLayerDepth, BuoyancyAnomalyCriterion, DensityAnomalyCriterion
export BottomCellValue

using Oceanostics: validate_location,
                   validate_dissipative_closure,
                   add_background_fields,
                   get_coriolis_frequency_components,
                   CustomKFO

import Oceananigans # so `Oceananigans.defaults.gravitational_acceleration` resolves in default kwargs/fields below
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

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b);

julia> Ri = RichardsonNumber(model)
RichardsonNumber (KernelFunctionOperation) at (Center, Center, Face)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: richardson_number_ccf (generic function with 1 method)
└── arguments: ("Field", "Field", "Field", "Field", "Tuple")
└── computes: Richardson number  Ri = (∂b/∂z) / |∂u⃗ₕ/∂z|²
```
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

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(grid; coriolis=FPlane(f=1e-4));

julia> Ro = RossbyNumber(model)
RossbyNumber (KernelFunctionOperation) at (Face, Face, Face)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: rossby_number_fff (generic function with 1 method)
└── arguments: ("Field", "Field", "Field", "NamedTuple")
└── computes: Rossby number  Ro = ωᶻ/f
```
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

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(grid; coriolis=FPlane(f=1e-4), buoyancy=BuoyancyTracer(), tracers=:b);

julia> PV = ErtelPotentialVorticity(model; thermal_wind = true);

julia> PV isa ThermalWindPotentialVorticity
true

julia> PV isa ErtelPotentialVorticity
true
```
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
ErtelPotentialVorticity (KernelFunctionOperation) at (Face, Face, Face)
├── grid: 1×1×4 RectilinearGrid{Float64, Flat, Flat, Bounded} on CPU with 0×0×3 halo
├── kernel_function: ertel_potential_vorticity_fff (generic function with 1 method)
└── arguments: ("Field", "Field", "Field", "Field", "Int64", "Int64", "Float64")
└── computes: Ertel potential vorticity  q = ω⃗ₜₒₜ · ∇b

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

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(grid; coriolis=FPlane(f=1e-4), buoyancy=BuoyancyTracer(), tracers=:b);

julia> DEPV = DirectionalErtelPotentialVorticity(model, (0, 0, 1))
DirectionalErtelPotentialVorticity (KernelFunctionOperation) at (Face, Face, Face)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: directional_ertel_potential_vorticity_fff (generic function with 1 method)
└── arguments: ("Field", "Field", "Field", "Field", "NamedTuple")
└── computes: directional contribution to Ertel PV  (f̂ + ω̂)·∇b along a direction
```
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

#+++ Strain rate tensor
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

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(grid);

julia> S = StrainRateTensorModulus(model)
StrainRateTensorModulus (KernelFunctionOperation) at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: strain_rate_tensor_modulus_ccc (generic function with 1 method)
└── arguments: ("Field", "Field", "Field")
└── computes: strain-rate tensor modulus  √(SᵢⱼSᵢⱼ)
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

validate_dims(dims::Tuple{Vararg{Int}}) =
    (!isempty(dims) & all(d -> d in (1, 2, 3), dims) & allunique(dims)) ||
        throw(ArgumentError("`dims` must be a non-empty tuple of distinct integers drawn from (1, 2, 3); got $dims"))
validate_dims(dims) = throw(ArgumentError("`dims` must be a tuple of integers; got $(typeof(dims))"))

"""
    $(SIGNATURES)

Return the components of the strain rate tensor `S`, defined as the symmetric part of the velocity
gradient tensor:

```
    Sᵢⱼ = ½(∂ⱼuᵢ + ∂ᵢuⱼ)
```

The result is a `NamedTuple` with the independent components, each a `KernelFunctionOperation`
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
`S₃₁ = S₁₃`, `S₃₂ = S₂₃`).

`dims` selects which spatial directions (`1 → x`, `2 → y`, `3 → z`) enter the tensor: component
`Sᵢⱼ` is included only when both `i` and `j` are in `dims`. The default `dims = (1, 2, 3)` returns
the full tensor, while e.g. `dims = (1, 3)` returns the 2D strain rate tensor in the `x`–`z` plane
(`S₁₁`, `S₃₃`, `S₁₃`). Components are always ordered diagonals-first, independently of the order of
`dims`.

Each component can be wrapped in a `Field` and used with output writers, time-averaging, etc. Can
also be called as `StrainRateTensor(grid, u, v, w; dims)` to build the components from individual
velocity fields. See also [`StrainRateTensorModulus`](@ref) for the scalar modulus `√(SᵢⱼSᵢⱼ)`.

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(grid);

julia> S = StrainRateTensor(model);

julia> keys(S)
(:S₁₁, :S₂₂, :S₃₃, :S₁₂, :S₁₃, :S₂₃)

julia> S.S₁₃
KernelFunctionOperation at (Face, Center, Face)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: strain_rate_tensor_xz_fcf (generic function with 1 method)
└── arguments: ("Field", "Field")
```
"""
StrainRateTensor(model; dims = (1, 2, 3)) = StrainRateTensor(model.grid, model.velocities...; dims)

function StrainRateTensor(grid::AbstractGrid, u, v, w; dims = (1, 2, 3))
    validate_dims(dims)
    want(ij...) = all(in(dims), ij) # keep component Sᵢⱼ only if every index it needs is in `dims`

    components = (
        S₁₁ = want(1)    ? KernelFunctionOperation{Center, Center, Center}(∂xᶜᶜᶜ, grid, u) : nothing,
        S₂₂ = want(2)    ? KernelFunctionOperation{Center, Center, Center}(∂yᶜᶜᶜ, grid, v) : nothing,
        S₃₃ = want(3)    ? KernelFunctionOperation{Center, Center, Center}(∂zᶜᶜᶜ, grid, w) : nothing,
        S₁₂ = want(1, 2) ? KernelFunctionOperation{Face, Face, Center}(strain_rate_tensor_xy_ffc, grid, u, v) : nothing,
        S₁₃ = want(1, 3) ? KernelFunctionOperation{Face, Center, Face}(strain_rate_tensor_xz_fcf, grid, u, w) : nothing,
        S₂₃ = want(2, 3) ? KernelFunctionOperation{Center, Face, Face}(strain_rate_tensor_yz_cff, grid, v, w) : nothing,
    )

    return (; (k => op for (k, op) in pairs(components) if op !== nothing)...)
end
#---

#+++ Vorticity tensor
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

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(grid);

julia> Ω = VorticityTensorModulus(model)
VorticityTensorModulus (KernelFunctionOperation) at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: vorticity_tensor_modulus_ccc (generic function with 1 method)
└── arguments: ("Field", "Field", "Field")
└── computes: vorticity tensor modulus  √(ΩᵢⱼΩᵢⱼ)
```
"""
function VorticityTensorModulus(model; loc = (Center, Center, Center))
    validate_location(loc, "VorticityTensorModulus", (Center, Center, Center))
    return KernelFunctionOperation{Center, Center, Center}(vorticity_tensor_modulus_ccc, model.grid, model.velocities...)
end

# Off-diagonal vorticity components, each evaluated at its natural staggered location. The diagonal
# components vanish identically (Ωᵢᵢ = 0), so they are not built.
@inline vorticity_tensor_xy_ffc(i, j, k, grid, u, v) = (∂yᶠᶠᶜ(i, j, k, grid, u) - ∂xᶠᶠᶜ(i, j, k, grid, v)) / 2
@inline vorticity_tensor_xz_fcf(i, j, k, grid, u, w) = (∂zᶠᶜᶠ(i, j, k, grid, u) - ∂xᶠᶜᶠ(i, j, k, grid, w)) / 2
@inline vorticity_tensor_yz_cff(i, j, k, grid, v, w) = (∂zᶜᶠᶠ(i, j, k, grid, v) - ∂yᶜᶠᶠ(i, j, k, grid, w)) / 2

"""
    $(SIGNATURES)

Return the components of the vorticity tensor `Ω`, defined as the antisymmetric part of the velocity
gradient tensor:

```
    Ωᵢⱼ = ½(∂ⱼuᵢ - ∂ᵢuⱼ)
```

The tensor is antisymmetric, so its diagonal vanishes (`Ω₁₁ = Ω₂₂ = Ω₃₃ = 0`) and only the
independent off-diagonal components are returned, as a `NamedTuple` of `KernelFunctionOperation`s,
each living at its natural location on the staggered grid:

| Component | Definition         | Location |
|:---------:|:------------------:|:--------:|
| `Ω₁₂`     | `½(∂u/∂y - ∂v/∂x)` | `ffc`    |
| `Ω₁₃`     | `½(∂u/∂z - ∂w/∂x)` | `fcf`    |
| `Ω₂₃`     | `½(∂v/∂z - ∂w/∂y)` | `cff`    |

The remaining off-diagonal components follow from antisymmetry, `Ωⱼᵢ = -Ωᵢⱼ` (i.e. `Ω₂₁ = -Ω₁₂`,
`Ω₃₁ = -Ω₁₃`, `Ω₃₂ = -Ω₂₃`).

`dims` selects which spatial directions (`1 → x`, `2 → y`, `3 → z`) enter the tensor: component
`Ωᵢⱼ` is included only when both `i` and `j` are in `dims`. The default `dims = (1, 2, 3)` returns
all three off-diagonal components, while e.g. `dims = (1, 3)` returns the single component in the
`x`–`z` plane (`Ω₁₃`). Because every component couples two distinct directions, a single-direction
`dims` (e.g. `dims = (1,)`) yields an empty tensor. Components are always ordered `Ω₁₂`, `Ω₁₃`,
`Ω₂₃`, independently of the order of `dims`.

Each component can be wrapped in a `Field` and used with output writers, time-averaging, etc. Can
also be called as `VorticityTensor(grid, u, v, w; dims)` to build the components from individual
velocity fields. See also [`VorticityTensorModulus`](@ref) for the scalar modulus `√(ΩᵢⱼΩᵢⱼ)`.

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(grid);

julia> Ω = VorticityTensor(model);

julia> keys(Ω)
(:Ω₁₂, :Ω₁₃, :Ω₂₃)

julia> Ω.Ω₁₃
KernelFunctionOperation at (Face, Center, Face)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: vorticity_tensor_xz_fcf (generic function with 1 method)
└── arguments: ("Field", "Field")
```
"""
VorticityTensor(model; dims = (1, 2, 3)) = VorticityTensor(model.grid, model.velocities...; dims)

function VorticityTensor(grid::AbstractGrid, u, v, w; dims = (1, 2, 3))
    validate_dims(dims)
    want(ij...) = all(in(dims), ij) # keep component Ωᵢⱼ only if every index it needs is in `dims`

    components = (
        Ω₁₂ = want(1, 2) ? KernelFunctionOperation{Face, Face, Center}(vorticity_tensor_xy_ffc, grid, u, v) : nothing,
        Ω₁₃ = want(1, 3) ? KernelFunctionOperation{Face, Center, Face}(vorticity_tensor_xz_fcf, grid, u, w) : nothing,
        Ω₂₃ = want(2, 3) ? KernelFunctionOperation{Center, Face, Face}(vorticity_tensor_yz_cff, grid, v, w) : nothing,
    )

    return (; (k => op for (k, op) in pairs(components) if op !== nothing)...)
end

# From doi:10.1063/1.5124245
@inline function Q_velocity_gradient_tensor_invariant_ccc(i, j, k, grid, u, v, w)
    S² = strain_rate_tensor_modulus_ccc(i, j, k, grid, u, v, w)^2
    Ω² = vorticity_tensor_modulus_ccc(i, j, k, grid, u, v, w)^2
    return (Ω² - S²) / 2
end

const QVelocityGradientTensorInvariant = CustomKFO{<:typeof(Q_velocity_gradient_tensor_invariant_ccc)}
#---

#+++ Stress tensor
# Stress tensor τᵢⱼ = uᵢuⱼ kernels. Diagonals come in two flavors (see `collocate_diagonals` in
# `StressTensor`): collocated `_ccc` = (ℑ uᵢ)² at ccc, or interpolation-free at uᵢ's own location.
@inline stress_tensor_xx_ccc(i, j, k, grid, u)    = ℑxᶜᵃᵃ(i, j, k, grid, u)^2
@inline stress_tensor_yy_ccc(i, j, k, grid, v)    = ℑyᵃᶜᵃ(i, j, k, grid, v)^2
@inline stress_tensor_zz_ccc(i, j, k, grid, w)    = ℑzᵃᵃᶜ(i, j, k, grid, w)^2
@inline stress_tensor_xx_fcc(i, j, k, grid, u)    = @inbounds u[i, j, k]^2
@inline stress_tensor_yy_cfc(i, j, k, grid, v)    = @inbounds v[i, j, k]^2
@inline stress_tensor_zz_ccf(i, j, k, grid, w)    = @inbounds w[i, j, k]^2
@inline stress_tensor_xy_ffc(i, j, k, grid, u, v) = ℑyᵃᶠᵃ(i, j, k, grid, u) * ℑxᶠᵃᵃ(i, j, k, grid, v)
@inline stress_tensor_xz_fcf(i, j, k, grid, u, w) = ℑzᵃᵃᶠ(i, j, k, grid, u) * ℑxᶠᵃᵃ(i, j, k, grid, w)
@inline stress_tensor_yz_cff(i, j, k, grid, v, w) = ℑzᵃᵃᶠ(i, j, k, grid, v) * ℑyᵃᶠᵃ(i, j, k, grid, w)

"""
    $(SIGNATURES)

Return the components of the (kinematic) stress tensor `τ`, defined as the outer product of the
velocity field with itself:

```
    τᵢⱼ = uᵢ uⱼ
```

The result is a `NamedTuple` of the independent components, each a `KernelFunctionOperation` living
at a location on the staggered grid.

The **off-diagonal** components are always evaluated at their natural edge location. Because each
couples two *different*, mutually-staggered velocities, the two factors must be interpolated to a
common point before multiplying — this is unavoidable, and the edge locations below are the
interpolation-minimal choice (one interpolation per factor):

| Component | Definition | Location |
|:---------:|:----------:|:--------:|
| `τ₁₂`     | `u v`      | `ffc`    |
| `τ₁₃`     | `u w`      | `fcf`    |
| `τ₂₃`     | `v w`      | `cff`    |

The **`collocate_diagonals`** keyword controls *only* where the diagonal components `τ₁₁, τ₂₂, τ₃₃`
live. It trades interpolation against collocation, and **defaults to `false`** (minimal
interpolation):

- `collocate_diagonals = false` (**default**): each diagonal is computed as `τᵢᵢ = uᵢ²` read
  *directly at `uᵢ`'s own location*, performing **no interpolation at all**. This is the cheapest
  and most accurate option. The trade-off is that the three diagonals end up at three *different*
  locations, so they are not collocated with each other or with the off-diagonals:

  | Component | Definition | Location |
  |:---------:|:----------:|:--------:|
  | `τ₁₁`     | `u u`      | `fcc`    |
  | `τ₂₂`     | `v v`      | `cfc`    |
  | `τ₃₃`     | `w w`      | `ccf`    |

- `collocate_diagonals = true`: each velocity is first interpolated to `ccc` and *then* squared,
  `τᵢᵢ = (ℑ uᵢ)²`, so all three diagonals share the single location `ccc`. Choose this when you need
  the diagonals collocated — e.g. to form the trace `τ₁₁ + τ₂₂ + τ₃₃` (twice the kinetic energy) or
  to treat `τ` as a single collocated tensor. The cost is one interpolation per diagonal:

  | Component | Definition | Location |
  |:---------:|:----------:|:--------:|
  | `τ₁₁`     | `u u`      | `ccc`    |
  | `τ₂₂`     | `v v`      | `ccc`    |
  | `τ₃₃`     | `w w`      | `ccc`    |

!!! warning "The two diagonal modes return different numbers"
    Interpolation and squaring do not commute, `(ℑ uᵢ)² ≠ ℑ(uᵢ²)`. The diagonals obtained with
    `collocate_diagonals = true` are therefore *not* the `false` values resampled at another point —
    they differ by an interpolation error. Pick the mode deliberately.

The tensor is symmetric, so the remaining components follow from `τⱼᵢ = τᵢⱼ` (i.e. `τ₂₁ = τ₁₂`,
`τ₃₁ = τ₁₃`, `τ₃₂ = τ₂₃`).

`dims` selects which spatial directions (`1 → x`, `2 → y`, `3 → z`) enter the tensor: component
`τᵢⱼ` is included only when both `i` and `j` are in `dims`. The default `dims = (1, 2, 3)` returns
the full tensor, while e.g. `dims = (1, 3)` returns the 2D stress tensor in the `x`–`z` plane
(`τ₁₁`, `τ₃₃`, `τ₁₃`). Components are always ordered diagonals-first, independently of the order of
`dims`.

Each component can be wrapped in a `Field` and used with output writers, time-averaging, etc. Can
also be called as `StressTensor(grid, u, v, w; dims, collocate_diagonals)` to build the components
from individual velocity fields. Building the tensor from perturbation velocities (e.g. via
`perturbation_fields`) yields the kinematic Reynolds stress tensor.

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(grid);

julia> τ = StressTensor(model);

julia> keys(τ)
(:τ₁₁, :τ₂₂, :τ₃₃, :τ₁₂, :τ₁₃, :τ₂₃)

julia> τ.τ₁₁
KernelFunctionOperation at (Face, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: stress_tensor_xx_fcc (generic function with 1 method)
└── arguments: ("Field",)
```
"""
StressTensor(model; dims = (1, 2, 3), collocate_diagonals = false) =
    StressTensor(model.grid, model.velocities...; dims, collocate_diagonals)

function StressTensor(grid::AbstractGrid, u, v, w; dims = (1, 2, 3), collocate_diagonals = false)
    validate_dims(dims)
    want(ij...) = all(in(dims), ij) # keep component τᵢⱼ only if every index it needs is in `dims`

    if collocate_diagonals # τᵢᵢ = (ℑ uᵢ)² interpolated to a shared ccc location (one interpolation each)
        τ₁₁ = want(1) ? KernelFunctionOperation{Center, Center, Center}(stress_tensor_xx_ccc, grid, u) : nothing
        τ₂₂ = want(2) ? KernelFunctionOperation{Center, Center, Center}(stress_tensor_yy_ccc, grid, v) : nothing
        τ₃₃ = want(3) ? KernelFunctionOperation{Center, Center, Center}(stress_tensor_zz_ccc, grid, w) : nothing
    else # τᵢᵢ = uᵢ² read at each velocity's own location (no interpolation)
        τ₁₁ = want(1) ? KernelFunctionOperation{Face, Center, Center}(stress_tensor_xx_fcc, grid, u) : nothing
        τ₂₂ = want(2) ? KernelFunctionOperation{Center, Face, Center}(stress_tensor_yy_cfc, grid, v) : nothing
        τ₃₃ = want(3) ? KernelFunctionOperation{Center, Center, Face}(stress_tensor_zz_ccf, grid, w) : nothing
    end

    components = (;
        τ₁₁, τ₂₂, τ₃₃,
        τ₁₂ = want(1, 2) ? KernelFunctionOperation{Face, Face, Center}(stress_tensor_xy_ffc, grid, u, v) : nothing,
        τ₁₃ = want(1, 3) ? KernelFunctionOperation{Face, Center, Face}(stress_tensor_xz_fcf, grid, u, w) : nothing,
        τ₂₃ = want(2, 3) ? KernelFunctionOperation{Center, Face, Face}(stress_tensor_yz_cff, grid, v, w) : nothing,
    )

    return (; (k => op for (k, op) in pairs(components) if op !== nothing)...)
end
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

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(grid);

julia> Qcrit = QVelocityGradientTensorInvariant(model)
QVelocityGradientTensorInvariant (KernelFunctionOperation) at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: Q_velocity_gradient_tensor_invariant_ccc (generic function with 1 method)
└── arguments: ("Field", "Field", "Field")
└── computes: Q velocity-gradient invariant  Q = ½(ΩᵢⱼΩᵢⱼ - SᵢⱼSᵢⱼ)
```
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

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> C = (; b = CenterField(grid));

julia> h = MixedLayerDepth(grid, BuoyancyTracer(), C);

julia> h isa KernelFunctionOperation
true
```
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

```jldoctest
julia> using Oceanostics

julia> BuoyancyAnomalyCriterion(threshold = -1e-4)
BuoyancyAnomalyCriterion{Float64}(-0.0001)
```
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

```jldoctest
julia> using Oceanostics

julia> DensityAnomalyCriterion()
DensityAnomalyCriterion{Float64}(1020.0, 9.80665, 0.125)
```
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

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> c = CenterField(grid);

julia> set!(c, (x, y, z) -> z);

julia> cᵇ = BottomCellValue(c);

julia> Field(cᵇ) isa Field
true
```
"""
function BottomCellValue(diagnostic)
    loc = location(diagnostic)
    instantiated_location = (L() for L in loc)
    condition_to_ban = KernelFunctionOperation{loc...}(bottommost_active_node, diagnostic.grid, instantiated_location...)
    return diagnostic * condition_to_ban
end
#---

end # module
