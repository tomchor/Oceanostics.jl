module TracerBudgetTerms
using DocStringExtensions

using Oceananigans: fields, Center, KernelFunctionOperation
using Oceananigans.Models.NonhydrostaticModels: div_Uc

using Oceanostics: validate_location

export TracerAdvection

#+++ Advection
"""
    $(SIGNATURES)

Calculates the advection of the tracer `c` as

    ADV = ∂ⱼ (uⱼ c)

using Oceananigans' kernel `div_Uc`.

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(; grid, tracers=:a);

julia> ADV = TracerAdvection(model, :a)
KernelFunctionOperation at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: div_Uc (generic function with 10 methods)
└── arguments: ("Centered(order=2)", "(u=4×4×4 Field{Face, Center, Center} on RectilinearGrid on CPU, v=4×4×4 Field{Center, Face, Center} on RectilinearGrid on CPU, w=4×4×5 Field{Center, Center, Face} on RectilinearGrid on CPU)", "4×4×4 Field{Center, Center, Center} on RectilinearGrid on CPU")
```
"""
function TracerAdvection(model, u, v, w, c, advection=model.advection; location = (Center, Center, Center))
    validate_location(location, "TracerAdvection", (Center, Center, Center))
    return KernelFunctionOperation{Center, Center, Center}(div_Uc, model.grid, advection, (; u, v, w), c)
end

function TracerAdvection(model, tracer_index; location = (Center, Center, Center))
    @inbounds c = model.tracers[tracer_index]
    return TracerAdvection(model, model.velocities..., c, model.advection; location)
end
#---

end # module
