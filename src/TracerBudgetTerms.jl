module TracerBudgetTerms
using DocStringExtensions

using Oceananigans: fields, Center, KernelFunctionOperation
using Oceananigans.Models: HydrostaticFreeSurfaceModel
using Oceananigans.Models.NonhydrostaticModels: div_Uc, ∇_dot_qᶜ, immersed_∇_dot_qᶜ, biogeochemical_transition

using Oceanostics: validate_location

export TracerAdvection, TracerDiffusion, ImmersedTracerDiffusion, TracerForcing, TracerBiogeochemistry

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
function TracerAdvection(model, u, v, w, c, advection; location = (Center, Center, Center))
    validate_location(location, "TracerAdvection", (Center, Center, Center))
    return KernelFunctionOperation{Center, Center, Center}(div_Uc, model.grid, advection, (; u, v, w), c)
end

function TracerAdvection(model, tracer_index; location = (Center, Center, Center))
    @inbounds c = model.tracers[tracer_index]
    return TracerAdvection(model, model.velocities..., c, model.advection; location)
end

function TracerAdvection(model::HydrostaticFreeSurfaceModel, tracer_index; location = (Center, Center, Center))
    @inbounds c = model.tracers[tracer_index]
    advection = model.advection[tracer_index]
    return TracerAdvection(model, model.velocities..., c, advection; location)
end
#---

#+++ Diffusion

"""
    $(SIGNATURES)


```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(; grid, tracers=:a);

julia> DIFF = TracerDiffusion(model, :a)
KernelFunctionOperation at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: ∇_dot_qᶜ (generic function with 10 methods)
└── arguments: ("Nothing", "Nothing", "Val{:a}", "4×4×4 Field{Center, Center, Center} on RectilinearGrid on CPU", "Clock{Float64, Float64}(time=0 seconds, iteration=0, last_Δt=Inf days)", "(u=4×4×4 Field{Face, Center, Center} on RectilinearGrid on CPU, v=4×4×4 Field{Center, Face, Center} on RectilinearGrid on CPU, w=4×4×5 Field{Center, Center, Face} on RectilinearGrid on CPU, a=4×4×4 Field{Center, Center, Center} on RectilinearGrid on CPU)", "Nothing")
```
"""
function TracerDiffusion(model, val_tracer_index, c, closure=model.closure, diffusivity_fields=model.diffusivity_fields; location = (Center, Center, Center))
    validate_location(location, "TracerDiffusion", (Center, Center, Center))
    return KernelFunctionOperation{Center, Center, Center}(∇_dot_qᶜ, model.grid, closure, diffusivity_fields, val_tracer_index, c, model.clock, fields(model), model.buoyancy)
end

function TracerDiffusion(model, tracer_index; location = (Center, Center, Center))
    @inbounds c = model.tracers[tracer_index]
    return TracerDiffusion(model, Val(tracer_index), c, model.closure, model.diffusivity_fields; location)
end


"""
    $(SIGNATURES)


```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(; grid, tracers=:a);

julia> DIFF = ImmersedTracerDiffusion(model, :a)
KernelFunctionOperation at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: ∇_dot_qᶜ (generic function with 10 methods)
└── arguments: ("Nothing", "Nothing", "Val{:a}", "4×4×4 Field{Center, Center, Center} on RectilinearGrid on CPU", "Clock{Float64, Float64}(time=0 seconds, iteration=0, last_Δt=Inf days)", "(u=4×4×4 Field{Face, Center, Center} on RectilinearGrid on CPU, v=4×4×4 Field{Center, Face, Center} on RectilinearGrid on CPU, w=4×4×5 Field{Center, Center, Face} on RectilinearGrid on CPU, a=4×4×4 Field{Center, Center, Center} on RectilinearGrid on CPU)", "Nothing")
```
"""
function ImmersedTracerDiffusion(model, c, c_immersed_bc=c.boundary_conditions.immersed,
                                 closure=model.closure, diffusivity_fields=model.diffusivity_fields, val_tracer_index=1, clock=model.clock, model_fields=fields(model);
                                 location = (Center, Center, Center))
    validate_location(location, "ImmersedTracerDiffusion", (Center, Center, Center))
    return KernelFunctionOperation{Center, Center, Center}(immersed_∇_dot_qᶜ, model.grid, c, c_immersed_bc, closure, diffusivity_fields, val_tracer_index, clock, model_fields)
end
#---

#+++
#---

end # module
