module TKEBudgetTerms
using DocStringExtensions


using Oceananigans: NonhydrostaticModel, HydrostaticFreeSurfaceModel, fields
using Oceananigans.Models: tracer_tendency, div_Uc

using Oceanostics: validate_location

#+++ Advection
function TracerAdvection(model, u, v, w, c, advection=model.advection; location = (Center, Center, Center))
    return KernelFunctionOperation{Center, Center, Center}(div_Uc, model.grid, advection, (; u, v, w), c)
end

function TracerAdvection(model, tracer_index)
    @inbounds c = model.tracers[tracer_index]
    return TracerAdvection(model, model.velocities..., c, model.advection; location)
end
#---

end # module
