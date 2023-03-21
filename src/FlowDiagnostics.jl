module FlowDiagnostics
using DocStringExtensions

export RichardsonNumber, RossbyNumber
export ErtelPotentialVorticity, ThermalWindPotentialVorticity, DirectionalErtelPotentialVorticity
export TracerVarianceDissipationRate

using Oceanostics: validate_location, validate_dissipative_closure

using Oceananigans
using Oceananigans.Operators
using Oceananigans.AbstractOperations
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Grids: Center, Face
import Oceananigans.TurbulenceClosures: diffusive_flux_x, diffusive_flux_y, diffusive_flux_z

#+++ Useful operators and functions
"""
    $(SIGNATURES)

Add background fields (velocities and tracers only) to their perturbations.
"""
function add_background_fields(model)

    velocities = model.velocities
    # Adds background velocities to their perturbations only if background velocity isn't ZeroField
    full_velocities = NamedTuple{keys(velocities)}((model.background_fields.velocities[key] isa Oceananigans.Fields.ZeroField) ? 
                                                   val : 
                                                   val + model.background_fields.velocities[key] 
                                                   for (key,val) in zip(keys(velocities), velocities))
    tracers = model.tracers
    # Adds background tracer fields to their perturbations only if background tracer field isn't ZeroField
    full_tracers = NamedTuple{keys(tracers)}((model.background_fields.tracers[key] isa Oceananigans.Fields.ZeroField) ? 
                                                   val : 
                                                   val + model.background_fields.tracers[key] 
                                                   for (key,val) in zip(keys(tracers), tracers))

    return merge(full_velocities, full_tracers)
end
#---

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

    if model.buoyancy.gravity_unit_vector isa Oceananigans.Grids.ZDirection
        true_vertical_direction = (0, 0, 1)
    else
        true_vertical_direction =  model.buoyancy.gravity_unit_vector
    end
    return KernelFunctionOperation{Center, Center, Face}(richardson_number_ccf, model.grid;
                                                         computed_dependencies=(u, v, w, b), parameters=Tuple(true_vertical_direction))
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
    return KernelFunctionOperation{Face, Face, Face}(rossby_number_fff, model.grid;
                                                     computed_dependencies=(u, v, w), parameters=parameters)
end
#---

#+++ Potential vorticity
@inline function potential_vorticity_in_thermal_wind_fff(i, j, k, grid, u, v, b, p)

    dVdx =  ℑzᵃᵃᶠ(i, j, k, grid, ∂xᶠᶠᶜ, v) # F, F, C → F, F, F
    dUdy =  ℑzᵃᵃᶠ(i, j, k, grid, ∂yᶠᶠᶜ, u) # F, F, C → F, F, F
    dbdz = ℑxyᶠᶠᵃ(i, j, k, grid, ∂zᶜᶜᶠ, b) # C, C, F → F, F, F

    pv_barot = (p.f + dVdx - dUdy) * dbdz

    dUdz = ℑyᵃᶠᵃ(i, j, k, grid, ∂zᶠᶜᶠ, u) # F, C, F → F, F, F
    dVdz = ℑxᶠᵃᵃ(i, j, k, grid, ∂zᶜᶠᶠ, v) # C, F, F → F, F, F

    pv_baroc = -p.f * (dUdz^2 + dVdz^2)

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
function ThermalWindPotentialVorticity(model; f=nothing)
    u, v, w = model.velocities
    b = BuoyancyField(model)
    if isnothing(f)
        f = model.coriolis.f
    end
    return KernelFunctionOperation{Face, Face, Face}(potential_vorticity_in_thermal_wind_fff, model.grid;
                                                     computed_dependencies=(u, v, b), parameters= (; f,))
end

@inline function ertel_potential_vorticity_fff(i, j, k, grid, u, v, w, b, params)
    dWdy =  ℑxᶠᵃᵃ(i, j, k, grid, ∂yᶜᶠᶠ, w) # C, C, F  → C, F, F  → F, F, F
    dVdz =  ℑxᶠᵃᵃ(i, j, k, grid, ∂zᶜᶠᶠ, v) # C, F, C  → C, F, F  → F, F, F
    dbdx = ℑyzᵃᶠᶠ(i, j, k, grid, ∂xᶠᶜᶜ, b) # C, C, C  → F, C, C  → F, F, F
    pv_x = (params.fx + dWdy - dVdz) * dbdx # F, F, F

    dUdz =  ℑyᵃᶠᵃ(i, j, k, grid, ∂zᶠᶜᶠ, u) # F, C, C  → F, C, F → F, F, F
    dWdx =  ℑyᵃᶠᵃ(i, j, k, grid, ∂xᶠᶜᶠ, w) # C, C, F  → F, C, F → F, F, F
    dbdy = ℑxzᶠᵃᶠ(i, j, k, grid, ∂yᶜᶠᶜ, b) # C, C, C  → C, F, C → F, F, F
    pv_y = (params.fy + dUdz - dWdx) * dbdy # F, F, F

    dVdx =  ℑzᵃᵃᶠ(i, j, k, grid, ∂xᶠᶠᶜ, v) # C, F, C  → F, F, C → F, F, F
    dUdy =  ℑzᵃᵃᶠ(i, j, k, grid, ∂yᶠᶠᶜ, u) # F, C, C  → F, F, C → F, F, F
    dbdz = ℑxyᶠᶠᵃ(i, j, k, grid, ∂zᶜᶜᶠ, b) # C, C, C  → C, C, F → F, F, F
    pv_z = (params.fz + dVdx - dUdy) * dbdz

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
        if ~(model.background_fields.velocities.u isa Oceananigans.Fields.ZeroField)
            u += model.background_fields.velocities.u
        end

        if ~(model.background_fields.velocities.v isa Oceananigans.Fields.ZeroField)
            v += model.background_fields.velocities.v
        end

        if ~(model.background_fields.velocities.w isa Oceananigans.Fields.ZeroField)
            w += model.background_fields.velocities.w
        end

        if ~(model.background_fields.tracers.b isa Oceananigans.Fields.ZeroField)
            b += model.background_fields.tracers.b
        end
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

    return KernelFunctionOperation{Face, Face, Face}(ertel_potential_vorticity_fff, model.grid;
                                                     computed_dependencies=(u, v, w, b), parameters=(; fx, fy, fz))
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

    if model.buoyancy == nothing || !(model.buoyancy.model isa Oceananigans.BuoyancyTracer)
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
    return KernelFunctionOperation{Face, Face, Face}(directional_ertel_potential_vorticity_fff, model.grid;
                                                     computed_dependencies=(u, v, w, b), parameters=(; f_dir, dir_x, dir_y, dir_z))
end
#---

#+++ Tracer variance dissipation
for diff_flux in (:diffusive_flux_x, :diffusive_flux_y, :diffusive_flux_z)
    # Unroll the loop over a tuple. This is necessary because no arrays can be dynamically
    # allocated inside a GPU Kernel, so something like
    # `sum(diff_flux(ijk, closure) for closure in closure_tuple)`
    # doesn't work, since it allocates a dynamically-sized Array.
    @eval @inline $diff_flux(i, j, k, grid, closure_tuple::Tuple, diffusivity_fields, args...) = 
        $diff_flux(i, j, k, grid, closure_tuple[1], diffusivity_fields[1], args...) + 
        $diff_flux(i, j, k, grid, closure_tuple[2:end], diffusivity_fields[2:end], args...)

    # End of the line
    @eval @inline $diff_flux(i, j, k, grid, closure_tuple::Tuple{}, args...) = zero(grid)
end

# Variance dissipation at fcc
@inline χxᶠᶜᶜ(i, j, k, grid, closure, diffusivity_fields, id, c, args...) =
    - Axᶠᶜᶜ(i, j, k, grid) * δxᶠᵃᵃ(i, j, k, grid, c) * diffusive_flux_x(i, j, k, grid, closure, diffusivity_fields, id, c, args...)

# Variance dissipation at cfc
@inline χyᶜᶠᶜ(i, j, k, grid, closure, diffusivity_fields, id, c, args...) =
    - Ayᶜᶠᶜ(i, j, k, grid) * δyᵃᶠᵃ(i, j, k, grid, c) * diffusive_flux_y(i, j, k, grid, closure, diffusivity_fields, id, c, args...)

# Variance dissipation at ccf
@inline χzᶜᶜᶠ(i, j, k, grid, closure, diffusivity_fields, id, c, args...) =
    - Azᶜᶜᶠ(i, j, k, grid) * δzᵃᵃᶠ(i, j, k, grid, c) * diffusive_flux_z(i, j, k, grid, closure, diffusivity_fields, id, c, args...)

@inline function tracer_variance_dissipation_rate_ccc(i, j, k, grid, diffusivity_fields, c, fields, p)
return 2 * (ℑxᶜᵃᵃ(i, j, k, grid, χxᶠᶜᶜ, p.closure, diffusivity_fields, p.id, c, p.clock, fields, p.buoyancy) + # F, C, C  → C, C, C
            ℑyᵃᶜᵃ(i, j, k, grid, χyᶜᶠᶜ, p.closure, diffusivity_fields, p.id, c, p.clock, fields, p.buoyancy) + # C, F, C  → C, C, C
            ℑzᵃᵃᶜ(i, j, k, grid, χzᶜᶜᶠ, p.closure, diffusivity_fields, p.id, c, p.clock, fields, p.buoyancy)   # C, C, F  → C, C, C
            ) / Vᶜᶜᶜ(i, j, k, grid) # This division by volume, coupled with the call to δ above, ensures a derivative operation
end

"""
    $(SIGNATURES)

Return a `KernelFunctionOperation` that computes the isotropic variance dissipation rate
for `tracer_name` in `model.tracers`. The isotropic variance dissipation rate is defined as 

    χ = 2 ∇c ⋅ F⃗

where `F⃗` is the diffusive flux of `c` and `∇` is the gradient operator. `χ` is implemented in its
conservative formulation based on the equation above. 

Note that often χ is written as `χ = 2κ (∇c ⋅ ∇c)`, which is the special case for Fickian diffusion
(`κ` is the tracer diffusivity).

Here `tracer_name` is needed even when passing `tracer` in order to get the appropriate `tracer_index`.
When passing `tracer`, this function should be used as

```julia
grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1))
model = NonhydrostaticModel(grid=grid, tracers=:b, closure=SmagorinskyLilly())

b̄ = Field(Average(model.tracers.b, dims=(1,2)))
b′ = model.tracers.b - b̄

χb = TracerVarianceDissipationRate(model, :b, tracer=b′)
```
"""
function TracerVarianceDissipationRate(model, tracer_name; tracer = nothing, location = (Center, Center, Center))
    tracer_index = findfirst(n -> n === tracer_name, propertynames(model.tracers))

    parameters = (; model.closure, model.clock, model.buoyancy,
                  id = Val(tracer_index))

    tracer = tracer == nothing ? model.tracers[tracer_name] : tracer

    return KernelFunctionOperation{Center, Center, Center}(tracer_variance_dissipation_rate_ccc, model.grid;
                                                           computed_dependencies=(model.diffusivity_fields, tracer, fields(model)),
                                                           parameters=parameters)
end
#---

end # module
