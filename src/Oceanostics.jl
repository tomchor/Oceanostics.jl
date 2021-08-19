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
export ErtelPotentialVorticityᶠᶠᶠ, ThermalWindPotentialVorticityᶠᶠᶠ
export IsotropicBuoyancyMixingRate, AnisotropicBuoyancyMixingRate
export IsotropicTracerVarianceDissipationRate, AnisotropicTracerVarianceDissipationRate
#----

include("TKEBudgetTerms.jl")
include("FlowDiagnostics.jl")
include("progress_messengers.jl")

using .TKEBudgetTerms

end # module
