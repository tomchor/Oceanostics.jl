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

#+++ Richardson number
@inline ψ²(i, j, k, grid, ψ) = @inbounds ψ[i, j, k]^2

"""
Get `w` from `û`, `v̂`, `ŵ` and based on the direction given by the unit vector `(k_x, k_y, k_z)`.
"""
@inline function w²_from_u⃗_tilted(i, j, k, grid, û, v̂, ŵ, k_x, k_y, k_z)
    û = ℑxᶜᵃᵃ(i, j, k, grid, û) # F, C, C  → C, C, C
    v̂ = ℑyᵃᶜᵃ(i, j, k, grid, v̂) # C, F, C  → C, C, C
    ŵ = ℑzᵃᵃᶜ(i, j, k, grid, ŵ) # C, C, F  → C, C, C
    return (û * k_x + v̂ * k_y + ŵ * k_z)^2
end

"""
Return the (true) horizontal velocity magnitude.
"""
@inline function uₕ_norm(i, j, k, grid, û, v̂, ŵ, k_x, k_y, k_z)
    û² = ℑxᶜᵃᵃ(i, j, k, grid, ψ², û) # F, C, C  → C, C, C
    v̂² = ℑyᵃᶜᵃ(i, j, k, grid, ψ², v̂) # C, F, C  → C, C, C
    ŵ² = ℑzᵃᵃᶜ(i, j, k, grid, ψ², ŵ) # C, C, F  → C, C, C
    return √(û² + v̂² + ŵ² - w²_from_u⃗_tilted(i, j, k, grid, û, v̂, ŵ, k_x, k_y, k_z))
end

@inline function richardson_number_ccf(i, j, k, grid, û, v̂, ŵ, b, params)

    dbdx̂ = ℑxzᶜᵃᶠ(i, j, k, grid, ∂xᶠᶜᶜ, b) # C, C, C  → F, C, C → C, C, F
    dbdŷ = ℑyzᵃᶜᶠ(i, j, k, grid, ∂yᶜᶠᶜ, b) # C, C, C  → C, F, C → C, C, F
    dbdẑ = ∂zᶜᶜᶠ(i, j, k, grid, b) # C, C, C  → C, C, F
    dbdz = dbdx̂ * params.vertical_dir_x + dbdŷ * params.vertical_dir_y + dbdẑ * params.vertical_dir_z

    duₕdx̂ = ℑxᶜᵃᵃ(i, j, k, grid, ∂xᶠᶜᶜ, uₕ_norm, û, v̂, ŵ, params.vertical_dir_x, params.vertical_dir_y, params.vertical_dir_z)
    duₕdŷ = ℑyᵃᶜᵃ(i, j, k, grid, ∂yᶜᶠᶜ, uₕ_norm, û, v̂, ŵ, params.vertical_dir_x, params.vertical_dir_y, params.vertical_dir_z)
    duₕdẑ = ∂zᶜᶜᶠ(i, j, k, grid, uₕ_norm, û, v̂, ŵ, params.vertical_dir_x, params.vertical_dir_y, params.vertical_dir_z)
    duₕdz = duₕdx̂ * params.vertical_dir_x + duₕdŷ * params.vertical_dir_y + duₕdẑ * params.vertical_dir_z

    return dbdz / duₕdz^2
end
function RichardsonNumber(model; location = (Face, Face, Face), add_background=true)
    validate_location(location, "RichardsonNumber", (Face, Face, Face))

    if (model isa NonhydrostaticModel) & add_background
        full_fields = add_background_fields(model)
        u, v, w, b = full_fields.u, full_fields.v, full_fields.w, full_fields.b
    else
        u, v, w = model.velocities
        b = model.tracers.b
    end

    vertical_dir_x, vertical_dir_y, vertical_dir_z = model.buoyancy.gravity_unit_vector
    return KernelFunctionOperation{Center, Center, Face}(richardson_number_ccf, model.grid;
                                                     computed_dependencies=(u, v, w, b), parameters=(; vertical_dir_x, vertical_dir_y, vertical_dir_z))
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
@inline function isotropic_tracer_variance_dissipation_rate_ccc(i, j, k, grid, c, velocities, p)
    dcdx² = ℑxᶜᵃᵃ(i, j, k, grid, fψ², ∂xᶠᶜᶜ, c) # C, C, C  → F, C, C  → C, C, C
    dcdy² = ℑyᵃᶜᵃ(i, j, k, grid, fψ², ∂yᶜᶠᶜ, c) # C, C, C  → C, F, C  → C, C, C
    dcdz² = ℑzᵃᵃᶜ(i, j, k, grid, fψ², ∂zᶜᶜᶠ, c) # C, C, C  → C, C, F  → C, C, C

    κ = _calc_κᶜᶜᶜ(i, j, k, grid, p.closure, c, p.id, velocities)

    return 2κ * (dcdx² + dcdy² + dcdz²)
end

"""
    IsotropicTracerVarianceDissipationRate(model, tracer_name)

Return a `KernelFunctionOperation` that computes the isotropic variance dissipation rate
for `tracer_name` in `model.tracers`.
"""
function IsotropicTracerVarianceDissipationRate(model, tracer_name; location = (Center, Center, Center))
    tracer_index = findfirst(n -> n === tracer_name, propertynames(model.tracers))

    parameters = (closure = model.closure,
                  id = Val(tracer_index))

    return KernelFunctionOperation{Center, Center, Center}(isotropic_tracer_variance_dissipation_rate_ccc, model.grid;
                                                           computed_dependencies=(model.tracers[tracer_name], model.velocities),
                                                           parameters=parameters)
end
#-----

end # module
