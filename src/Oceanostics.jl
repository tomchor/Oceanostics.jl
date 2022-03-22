module Oceanostics

#++++ TKEBudgetTerms exports
export TurbulentKineticEnergy, KineticEnergy
export IsotropicViscousDissipationRate, IsotropicPseudoViscousDissipationRate
export AnisotropicPseudoViscousDissipationRate
export XPressureRedistribution, YPressureRedistribution, ZPressureRedistribution
export XShearProduction, YShearProduction, ZShearProduction
#----

#++++ FlowDiagnostics exports
export RichardsonNumber, RossbyNumber
export ErtelPotentialVorticity, ThermalWindPotentialVorticity
export IsotropicBuoyancyMixingRate, AnisotropicBuoyancyMixingRate
export IsotropicTracerVarianceDissipationRate, AnisotropicTracerVarianceDissipationRate
#----

#####
##### A few utils for closure tuples:
#####

# Fallbacks that capture "single closure" case
@inline _νᶜᶜᶜ(args...) = νᶜᶜᶜ(args...)
@inline _κᶜᶜᶜ(args...) = κᶜᶜᶜ(args...)

# "Inner-outer" form (hopefully) makes the compiler "unroll" the loop over a tuple:
@inline _νᶜᶜᶜ(i, j, k, grid, closure_tuple::Tuple, K::Tuple, clock) =
     νᶜᶜᶜ(i, j, k, grid, closure_tuple[1],     K[1],     clock) + 
    _νᶜᶜᶜ(i, j, k, grid, closure_tuple[2:end], K[2:end], clock)

@inline _κᶜᶜᶜ(i, j, k, grid, closure_tuple::Tuple, K::Tuple, id, clock) =
     κᶜᶜᶜ(i, j, k, grid, closure_tuple[1],     K[1],     id, clock) + 
    _κᶜᶜᶜ(i, j, k, grid, closure_tuple[2:end], K[2:end], id, clock)

include("TKEBudgetTerms.jl")
include("FlowDiagnostics.jl")
include("progress_messengers.jl")

using .TKEBudgetTerms

end # module
