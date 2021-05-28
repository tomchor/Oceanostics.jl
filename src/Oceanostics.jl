module Oceanostics

export TurbulentKineticEnergy, KineticEnergy
export IsotropicViscousDissipationRate, IsotropicPseudoViscousDissipationRate
export AnisotropicViscousDissipationRate, AnisotropicPseudoViscousDissipationRate
export PressureRedistribution_x, PressureRedistribution_y, PressureRedistribution_z
export ShearProduction_x, ShearProduction_y, ShearProduction_z

include("TKEEquationTerms.jl")
include("FlowDiagnostics.jl")
include("progress_messengers.jl")

using .TKEEquationTerms

end # module
