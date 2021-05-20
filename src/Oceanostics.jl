module Oceanostics

export TurbulentKineticEnergy, KineticEnergy

include("TurbulentKineticEnergyTerms.jl")
include("FlowDiagnostics.jl")
include("progress_messengers.jl")

using .TurbulentKineticEnergyTerms

end # module
