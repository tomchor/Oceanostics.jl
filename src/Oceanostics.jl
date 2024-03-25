module Oceanostics
using DocStringExtensions

#+++ TKEBudgetTerms exports
export TurbulentKineticEnergy, KineticEnergy
export KineticEnergyTendency, KineticEnergyStressTerm, KineticEnergyForcingTerm
export IsotropicKineticEnergyDissipationRate, KineticEnergyDissipationRate
export XPressureRedistribution, YPressureRedistribution, ZPressureRedistribution
export XShearProductionRate, YShearProductionRate, ZShearProductionRate
#---

#+++ TracerVarianceBudgetTerms exports
export TracerVarianceTendency
export TracerVarianceDiffusiveTerm
export TracerVarianceDissipationRate
#---

#+++ FlowDiagnostics exports
export RichardsonNumber, RossbyNumber
export ErtelPotentialVorticity, ThermalWindPotentialVorticity
export DirectionalErtelPotentialVorticity
export StrainRateTensorModulus, VorticityTensorModulus, Q, QVelocityGradientTensorInvariant
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
using Oceananigans.BuoyancyModels: Buoyancy, BuoyancyTracer, SeawaterBuoyancy, LinearEquationOfState
using SeawaterPolynomials: BoussinesqEquationOfState
using Oceananigans.Grids: Center, Face

validate_location(location, type, valid_location=(Center, Center, Center)) =
    location != valid_location &&
        error("$type only supports location = $valid_location for now.")

validate_dissipative_closure(closure) = error("Cannot calculate dissipation rate for $closure")
validate_dissipative_closure(::AbstractScalarDiffusivity{<:Any, ThreeDimensionalFormulation}) = nothing
validate_dissipative_closure(closure_tuple::Tuple) = Tuple(validate_dissipative_closure(c) for c in closure_tuple)

validate_buoyancy(buoyancy::Nothing) = throw(ArgumentError("Cannot calculate gravitational potential energy without a buoyancy model."))
validate_buoyancy(buoyancy::Buoyancy{<:BuoyancyTracer, g}) where g =
    throw(ArgumentError("Cannot calculate gravitational potential energy for the buoyancy model $(buoyancy.model)."))
validate_buoyancy(buoyancy::Buoyancy{<:SeawaterBuoyancy{FT, <:LinearEquationOfState, T, S}, g}) where {FT, T, S, g} =
    throw(ArgumentError("To calculate gravitational potential energy with a linear equation of state use a linear `BoussinesqEquationOfState`."))
validate_buoyancy(buoyancy::Buoyancy{<:SeawaterBuoyancy{FT, <:BoussinesqEquationOfState, T, S}, g}) where {FT, T, S, g} = nothing

#---

#+++ Utils for background fields
using Oceananigans.Fields: ZeroField
"""
    $(SIGNATURES)

Add background fields (velocities and tracers only) to their perturbations.
"""
function add_background_fields(model)

    velocities = model.velocities
    # Adds background velocities to their perturbations only if background velocity isn't ZeroField
    full_velocities = NamedTuple{keys(velocities)}((model.background_fields.velocities[key] isa ZeroField) ?
                                                   val :
                                                   val + model.background_fields.velocities[key]
                                                   for (key,val) in zip(keys(velocities), velocities))
    tracers = model.tracers
    # Adds background tracer fields to their perturbations only if background tracer field isn't ZeroField
    full_tracers = NamedTuple{keys(tracers)}((model.background_fields.tracers[key] isa ZeroField) ?
                                              val :
                                              val + model.background_fields.tracers[key]
                                              for (key,val) in zip(keys(tracers), tracers))

    return merge(full_velocities, full_tracers)
end
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

include("TKEBudgetTerms.jl")
include("TracerVarianceBudgetTerms.jl")
include("FlowDiagnostics.jl")
include("PotentialEnergyEquationTerms.jl")
include("ProgressMessengers/ProgressMessengers.jl")

using .TKEBudgetTerms, .TracerVarianceBudgetTerms, .FlowDiagnostics, .ProgressMessengers
using .PotentialEnergyEquationTerms

end # module
