module FlowDiagnostics

export RichardsonNumber, RossbyNumber
export ErtelPotentialVorticity, ThermalWindPotentialVorticity, DirectionalErtelPotentialVorticity
export IsotropicBuoyancyMixingRate, AnisotropicBuoyancyMixingRate
export IsotropicTracerVarianceDissipationRate, AnisotropicTracerVarianceDissipationRate

using ..TKEBudgetTerms: validate_location

using Oceanostics: _calc_κᶜᶜᶜ

using Oceananigans
using Oceananigans.Operators
using Oceananigans.AbstractOperations
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Grids: Center, Face
using Oceananigans.TurbulenceClosures: diffusive_flux_x, diffusive_flux_y, diffusive_flux_z

#+++ Useful operators and functions
@inline fψ²(i, j, k, grid, f, ψ) = @inbounds f(i, j, k, grid, ψ)^2

"""
Adds background fields (velocities and tracers only) to their perturbations
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

function RichardsonNumber(model; b=BuoyancyField(model), N²_bg=0, dUdz_bg=0, dVdz_bg=0)
    u, v, w = model.velocities

    dBdz_tot = ∂z(b) + N²_bg
    dUdz_tot = ∂z(u) + dUdz_bg
    dVdz_tot = ∂z(v) + dVdz_bg

    return dBdz_tot / (dUdz_tot^2 + dVdz_tot^2)
end

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
Calculates the Rossby number using the vorticity in the rotation axis direction according
to `model.coriolis`.
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

#++++ Potential vorticity
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
Calculates the contribution from a given `direction` to the Ertel Potential Vorticty
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
#----

#+++++ Tracer variance dissipation
# Variance dissipation at fcc
@inline χxᶠᶜᶜ(i, j, k, grid, closure, diffusivities, id, c, args...) = - δxᶠᵃᵃ(i, j, k, grid, c) * diffusive_flux_x(i, j, k, grid, closure, diffusivities, id, c, args...) 

# Variance dissipation at cfc
@inline χyᶜᶠᶜ(i, j, k, grid, closure, diffusivities, id, c, args...) = - δyᵃᶠᵃ(i, j, k, grid, c) * diffusive_flux_y(i, j, k, grid, closure, diffusivities, id, c, args...)

# Variance dissipation at ccf
@inline χzᶜᶜᶠ(i, j, k, grid, closure, diffusivities, id, c, args...) = - δzᵃᵃᶠ(i, j, k, grid, c) * diffusive_flux_z(i, j, k, grid, closure, diffusivities, id, c, args...)
# Function call is 
# diffusive_flux_x(i, j, k, grid, cl::AIDorAHD, K, ::Val{id}, c, clk, fields, b)
# Which can be written more verbose as
# diffusive_flux_x(i, j, k, grid, closure, diffusivities, id, c, clock, fields, model.buoyancy)

@inline function isotropic_tracer_variance_dissipation_rate_ccc(i, j, k, grid, diffusivity_fields, c, fields, p)
#    dcdx² = ℑxᶜᵃᵃ(i, j, k, grid, fψ², ∂xᶠᶜᶜ, c) # C, C, C  → F, C, C  → C, C, C
#    dcdy² = ℑyᵃᶜᵃ(i, j, k, grid, fψ², ∂yᶜᶠᶜ, c) # C, C, C  → C, F, C  → C, C, C
#    dcdz² = ℑzᵃᵃᶜ(i, j, k, grid, fψ², ∂zᶜᶜᶠ, c) # C, C, C  → C, C, F  → C, C, C

return 2 * (ℑxᶜᵃᵃ(i, j, k, grid, χxᶠᶜᶜ, p.closure, diffusivity_fields, p.id, c, p.clock, fields, p.buoyancy) + # F, C, C  → C, C, C
            ℑyᵃᶜᵃ(i, j, k, grid, χyᶜᶠᶜ, p.closure, diffusivity_fields, p.id, c, p.clock, fields, p.buoyancy) + # C, F, C  → C, C, C
            ℑzᵃᵃᶜ(i, j, k, grid, χzᶜᶜᶠ, p.closure, diffusivity_fields, p.id, c, p.clock, fields, p.buoyancy)   # C, C, F  → C, C, C
            ) / Vᶜᶜᶜ(i, j, k, grid) # This division by volume, coupled with the call to δ above, ensures a derivative operation

#    κ = _calc_κᶜᶜᶜ(i, j, k, grid, p.closure, p.buoyancy, velocities, tracers, p.id)
#    return 2κ * (dcdx² + dcdy² + dcdz²)
end

"""
    IsotropicTracerVarianceDissipationRate(model, tracer_name)

Return a `KernelFunctionOperation` that computes the isotropic variance dissipation rate
for `tracer_name` in `model.tracers`.
"""
function IsotropicTracerVarianceDissipationRate(model, tracer_name; location = (Center, Center, Center))
    tracer_index = findfirst(n -> n === tracer_name, propertynames(model.tracers))

    parameters = (; model.closure, model.clock, model.buoyancy,
                  id = Val(tracer_index))

    return KernelFunctionOperation{Center, Center, Center}(isotropic_tracer_variance_dissipation_rate_ccc, model.grid;
                                                           computed_dependencies=(model.diffusivity_fields, model.tracers[tracer_name], fields(model)),
                                                           parameters=parameters)
end
#-----

end # module
