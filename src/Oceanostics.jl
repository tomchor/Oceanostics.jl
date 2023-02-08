module Oceanostics
using DocStringExtensions

#++++ TKEBudgetTerms exports
export TurbulentKineticEnergy, KineticEnergy
export IsotropicViscousDissipationRate, IsotropicPseudoViscousDissipationRate
export AnisotropicPseudoViscousDissipationRate
export XPressureRedistribution, YPressureRedistribution, ZPressureRedistribution
export XShearProductionRate, YShearProductionRate, ZShearProductionRate
#----

#++++ FlowDiagnostics exports
export RichardsonNumber, RossbyNumber
export ErtelPotentialVorticity, ThermalWindPotentialVorticity
export IsotropicBuoyancyMixingRate, AnisotropicBuoyancyMixingRate
export IsotropicTracerVarianceDissipationRate, AnisotropicTracerVarianceDissipationRate
#----

using Oceananigans.TurbulenceClosures: νᶜᶜᶜ, calc_nonlinear_κᶜᶜᶜ

#####
##### A few utils for closure tuples:
#####

# Fallbacks that capture "single closure" case
@inline _νᶜᶜᶜ(args...) = νᶜᶜᶜ(args...)
@inline _calc_nonlinear_κᶜᶜᶜ(args...) = calc_nonlinear_κᶜᶜᶜ(args...)

# End point
@inline _νᶜᶜᶜ(i, j, k, grid, closure_tuple::Tuple{}, K::Tuple{}, clock) = zero(eltype(grid))
@inline _calc_nonlinear_κᶜᶜᶜ(i, j, k, grid, closure_tuple::Tuple{}, c, id, velocities) = zero(eltype(grid))

# "Inner-outer" form (hopefully) makes the compiler "unroll" the loop over a tuple:
@inline _νᶜᶜᶜ(i, j, k, grid, closure_tuple::Tuple, K::Tuple, clock) =
     νᶜᶜᶜ(i, j, k, grid, closure_tuple[1],     K[1],     clock) + 
    _νᶜᶜᶜ(i, j, k, grid, closure_tuple[2:end], K[2:end], clock)

@inline _calc_nonlinear_κᶜᶜᶜ(i, j, k, grid, closure_tuple::Tuple, c, id, velocities) = 
    calc_nonlinear_κᶜᶜᶜ(i, j, k, grid, closure_tuple[1], c, id, velocities) + 
    _calc_nonlinear_κᶜᶜᶜ(i, j, k, grid, closure_tuple[2:end], c, id, velocities)


include("TKEBudgetTerms.jl")
include("FlowDiagnostics.jl")
include("progress_messengers.jl")

using .TKEBudgetTerms

end # module
