module Oceanostics
using DocStringExtensions

#+++ Module export
export TracerEquation, KineticEnergyEquation, TurbulentKineticEnergyEquation, TracerVarianceEquation, PotentialEnergyEquation
#---

#+++ TracerEquation exports
export TracerAdvection, TracerDiffusion, TracerImmersedDiffusion, TracerTotalDiffusion, TracerForcing
#---

#+++ TracerVarianceEquation exports
export TracerVarianceDissipationRate, TracerVarianceTendency, TracerVarianceDiffusiveTerm
#---

#+++ KineticEnergyEquation exports
export KineticEnergyForcing, KineticEnergyPressureRedistribution, KineticEnergyBuoyancyProduction, KineticEnergyDissipationRate
#---

#+++ FlowDiagnostics exports
export RichardsonNumber, RossbyNumber
export ErtelPotentialVorticity, ThermalWindPotentialVorticity
export DirectionalErtelPotentialVorticity
export StrainRateTensorModulus, VorticityTensorModulus, Q, QVelocityGradientTensorInvariant
export MixedLayerDepth, BuoyancyAnomalyCriterion, DensityAnomalyCriterion
export BottomCellValue
#---

#+++ PotentialEnergyEquationTerms exports
export PotentialEnergy
#---

#+++ ProgressMessengers
export ProgressMessengers
#---

#+++ Utils for validation
# Right now, all kernels must be located at ccc
using Oceananigans.TurbulenceClosures: AbstractScalarDiffusivity, ThreeDimensionalFormulation
using Oceananigans.Grids: Center, Face

validate_location(location, type, valid_location=(Center, Center, Center)) =
    location != valid_location &&
        error("$type only supports location = $valid_location for now.")

validate_dissipative_closure(closure) = error("Cannot calculate dissipation rate for $closure")
validate_dissipative_closure(::AbstractScalarDiffusivity{<:Any, ThreeDimensionalFormulation}) = nothing
validate_dissipative_closure(closure_tuple::Tuple) = Tuple(validate_dissipative_closure(c) for c in closure_tuple)
#---

#+++ Utils for background fields
using Oceananigans.Fields: Field, ZeroField
"""
    $(SIGNATURES)

Add background fields (velocities and tracers only) to their perturbations.
"""
function add_background_fields(model)

    velocities = model.velocities
    # Adds background velocities to their perturbations only if background velocity isn't ZeroField
    full_velocities = NamedTuple{keys(velocities)}((model.background_fields.velocities[key] isa ZeroField) ?
                                                    val :
                                                    Field(val + model.background_fields.velocities[key])
                                                    for (key, val) in zip(keys(velocities), velocities))
    tracers = model.tracers
    # Adds background tracer fields to their perturbations only if background tracer field isn't ZeroField
    full_tracers = NamedTuple{keys(tracers)}((model.background_fields.tracers[key] isa ZeroField) ?
                                              val :
                                              Field(val + model.background_fields.tracers[key])
                                              for (key, val) in zip(keys(tracers), tracers))

    return merge(full_velocities, full_tracers)
end
#---

#+++ Utils for removing mean fields
using Oceananigans: prognostic_fields, HydrostaticFreeSurfaceModel
using Oceananigans.Biogeochemistry: biogeochemical_auxiliary_fields
"""
    $(SIGNATURES)

Remove mean fields from the model resolved fields.
"""
function perturbation_fields(model; kwargs...)

    resolved_fields = prognostic_fields(model)
    if model isa HydrostaticFreeSurfaceModel
        resolved_fields = (; resolved_fields..., w=ZeroField())
    end

    mean_fields = values(kwargs)
    # Removes mean fields only if mean field is provided
    pert_fields = NamedTuple{keys(resolved_fields)}(haskey(mean_fields, key) ?
                                                    Field(val - mean_fields[key]) :
                                                    val
                                                    for (key,val) in zip(keys(resolved_fields), resolved_fields))
    return merge(pert_fields,
                 model.auxiliary_fields,
                 biogeochemical_auxiliary_fields(model.biogeochemistry))
end
#---

#+++ Utils for Coriolis frequency
using Oceananigans: FPlane, ConstantCartesianCoriolis, AbstractModel
function get_coriolis_frequency_components(coriolis)
    if coriolis isa FPlane
        fx = fy = 0
        fz = coriolis.f
    elseif coriolis isa ConstantCartesianCoriolis
        fx = coriolis.fx
        fy = coriolis.fy
        fz = coriolis.fz
    elseif coriolis == nothing
        fx = fy = fz = 0
    else
        throw(ArgumentError("Extraction of rotation components is only implemented for `FPlane`, `ConstantCartesianCoriolis` and `nothing`."))
    end
    return fx, fy, fz
end

get_coriolis_frequency_components(model::AbstractModel) = get_coriolis_frequency_components(model.coriolis)
#---

#+++ A few utils for closure tuples:
using Oceananigans.TurbulenceClosures: νᶜᶜᶜ

# Fallbacks that capture "single closure" case
@inline _νᶜᶜᶜ(args...) = νᶜᶜᶜ(args...)

# End point
@inline _νᶜᶜᶜ(i, j, k, grid, closure_tuple::Tuple{}, K::Tuple{}, clock) = zero(grid)

# Unroll the loop over a tuple
@inline _νᶜᶜᶜ(i, j, k, grid, closure_tuple::Tuple, K::Tuple, clock) = νᶜᶜᶜ(i, j, k, grid, closure_tuple[1],     K[1],     clock) +
                                                                     _νᶜᶜᶜ(i, j, k, grid, closure_tuple[2:end], K[2:end], clock)
#---

include("TracerEquation.jl")
include("TracerVarianceEquation.jl")
include("KineticEnergyEquation.jl")
include("TurbulentKineticEnergyEquation.jl")
include("PotentialEnergyEquation.jl")
include("FlowDiagnostics.jl")
include("ProgressMessengers/ProgressMessengers.jl")

using .TracerEquation, .TracerVarianceEquation, .KineticEnergyEquation, .TurbulentKineticEnergyEquation, .PotentialEnergyEquation
using .FlowDiagnostics
using .ProgressMessengers

end # module
