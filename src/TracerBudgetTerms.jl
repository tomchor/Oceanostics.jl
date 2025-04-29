module TracerBudgetTerms
using DocStringExtensions

using Oceananigans: fields, Center, KernelFunctionOperation
using Oceananigans.Models: HydrostaticFreeSurfaceModel
using Oceananigans.Models.NonhydrostaticModels: div_Uc, ∇_dot_qᶜ, immersed_∇_dot_qᶜ, biogeochemical_transition

using Oceanostics: validate_location

export TracerAdvection, TracerDiffusion, ImmersedTracerDiffusion, TotalTracerDiffusion, TracerForcing

#+++ Advection
"""
    $(SIGNATURES)

Calculates the advection of the tracer `c` as

    ADV = ∂ⱼ (uⱼ c)

using Oceananigans' kernel [`div_Uc`.](https://clima.github.io/OceananigansDocumentation/stable/appendix/library/#Oceananigans.Advection.div_Uc-NTuple{7,%20Any})

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

function TracerAdvection(model, tracer_index; kwargs...)
    @inbounds c = model.tracers[tracer_index]
    return TracerAdvection(model, model.velocities..., c, model.advection; kwargs...)
end

function TracerAdvection(model::HydrostaticFreeSurfaceModel, tracer_index; kwargs...)
    @inbounds c = model.tracers[tracer_index]
    tracer_advection = model.advection[tracer_index]
    return TracerAdvection(model, model.velocities..., c, tracer_advection; kwargs...)
end
#---

#+++ Diffusion
"""
    $(SIGNATURES)

Calculates the diffusion term (excluding anything due to the bathymetry) as

    DIFF = ∂ⱼ qᶜⱼ,

where qᶜⱼ is the diffusion tensor for tracer `c`, using the Oceananigans' kernel `∇_dot_qᶜ`.

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
function TracerDiffusion(model, val_tracer_index, c, closure, diffusivity_fields, clock, model_fields, buoyancy; location = (Center, Center, Center))
    validate_location(location, "TracerDiffusion", (Center, Center, Center))
    return KernelFunctionOperation{Center, Center, Center}(∇_dot_qᶜ, model.grid, closure, diffusivity_fields, val_tracer_index, c, clock, model_fields, buoyancy)
end

function TracerDiffusion(model, tracer_index; kwargs...)
    @inbounds c = model.tracers[tracer_index]
    return TracerDiffusion(model, Val(tracer_index), c, model.closure, model.diffusivity_fields, model.clock, fields(model), model.buoyancy; kwargs...)
end


"""
    $(SIGNATURES)

Calculates the diffusion term due to the bathymetry term as

    DIFF = ∂ⱼ 𝓆ᶜⱼ,

where 𝓆ᶜⱼ is the bathyemtry-led diffusion tensor for tracer `c`, using the Oceananigans' kernel
`immersed_∇_dot_qᶜ`.


```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(; grid, tracers=:a);

julia> DIFF = ImmersedTracerDiffusion(model, :a)
KernelFunctionOperation at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: immersed_∇_dot_qᶜ (generic function with 2 methods)
└── arguments: ("4×4×4 Field{Center, Center, Center} on RectilinearGrid on CPU", "FluxBoundaryCondition: Nothing", "Nothing", "Nothing", "Val{:a}", "Clock{Float64, Float64}(time=0 seconds, iteration=0, last_Δt=Inf days)", "(u=4×4×4 Field{Face, Center, Center} on RectilinearGrid on CPU, v=4×4×4 Field{Center, Face, Center} on RectilinearGrid on CPU, w=4×4×5 Field{Center, Center, Face} on RectilinearGrid on CPU, a=4×4×4 Field{Center, Center, Center} on RectilinearGrid on CPU)")
```
"""
function ImmersedTracerDiffusion(model, c, c_immersed_bc, closure, diffusivity_fields, val_tracer_index, clock, model_fields; location = (Center, Center, Center))
    validate_location(location, "ImmersedTracerDiffusion", (Center, Center, Center))
    return KernelFunctionOperation{Center, Center, Center}(immersed_∇_dot_qᶜ, model.grid, c, c_immersed_bc, closure, diffusivity_fields, val_tracer_index, clock, model_fields)
end

function ImmersedTracerDiffusion(model, tracer_index; kwargs...)
    tracer = model.tracers[tracer_index]
    immersed_bc = tracer.boundary_conditions.immersed
    return ImmersedTracerDiffusion(model, tracer, immersed_bc, model.closure, model.diffusivity_fields, Val(tracer_index), model.clock, fields(model); kwargs...)
end

@inline total_∇_dot_qᶜ(i, j, k, grid, c, c_immersed_bc, closure, diffusivity_fields, val_tracer_index, clock, model_fields, buoyancy) =
    ∇_dot_qᶜ(i, j, k, grid, closure, diffusivity_fields, val_tracer_index, c, clock, model_fields, buoyancy) +
    immersed_∇_dot_qᶜ(i, j, k, grid, c, c_immersed_bc, closure, diffusivity_fields, val_tracer_index, clock, model_fields, buoyancy)

"""
    $(SIGNATURES)

Calculates the total diffusion term as

    DIFF = ∂ⱼ qᶜⱼ + ∂ⱼ 𝓆ᶜⱼ,

where qᶜⱼ is the interior diffusion tensor and 𝓆ᶜⱼ is the bathyemtry-led diffusion tensor for tracer
`c`. The calculation is done using the Oceananigans' kernels `∇_dot_qᶜ` and `immersed_∇_dot_qᶜ`.

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(; grid, tracers=:a);

julia> DIFF = TotalTracerDiffusion(model, :a)
KernelFunctionOperation at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: total_∇_dot_qᶜ (generic function with 1 method)
└── arguments: ("4×4×4 Field{Center, Center, Center} on RectilinearGrid on CPU", "FluxBoundaryCondition: Nothing", "Nothing", "Nothing", "Val{:a}", "Clock{Float64, Float64}(time=0 seconds, iteration=0, last_Δt=Inf days)", "(u=4×4×4 Field{Face, Center, Center} on RectilinearGrid on CPU, v=4×4×4 Field{Center, Face, Center} on RectilinearGrid on CPU, w=4×4×5 Field{Center, Center, Face} on RectilinearGrid on CPU, a=4×4×4 Field{Center, Center, Center} on RectilinearGrid on CPU)", "Nothing")
```
"""
function TotalTracerDiffusion(model, c, c_immersed_bc, closure, diffusivity_fields, val_tracer_index, clock, model_fields, buoyancy; location = (Center, Center, Center))
    validate_location(location, "TotalTracerDiffusion", (Center, Center, Center))
    return KernelFunctionOperation{Center, Center, Center}(total_∇_dot_qᶜ, model.grid, c, c_immersed_bc, closure, diffusivity_fields, val_tracer_index, clock, model_fields, buoyancy)
end

function TotalTracerDiffusion(model, tracer_index; kwargs...)
    tracer = model.tracers[tracer_index]
    immersed_bc = tracer.boundary_conditions.immersed
    return TotalTracerDiffusion(model, tracer, immersed_bc, model.closure, model.diffusivity_fields, Val(tracer_index), model.clock, fields(model), model.buoyancy; kwargs...)
end
#---

#+++ Forcing
"""
    $(SIGNATURES)

Calculate the forcing term `Fᶜ` on the equation for tracer `c` for `model`.

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(; grid, tracers=:a);

julia> FORC = TracerForcing(model, :a)
KernelFunctionOperation at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: zeroforcing (generic function with 1 method)
└── arguments: ("Clock{Float64, Float64}(time=0 seconds, iteration=0, last_Δt=Inf days)", "(u=4×4×4 Field{Face, Center, Center} on RectilinearGrid on CPU, v=4×4×4 Field{Center, Face, Center} on RectilinearGrid on CPU, w=4×4×5 Field{Center, Center, Face} on RectilinearGrid on CPU, a=4×4×4 Field{Center, Center, Center} on RectilinearGrid on CPU)")
```
"""
function TracerForcing(model, forcing, clock, model_fields; location = (Center, Center, Center))
    return KernelFunctionOperation{Center, Center, Center}(forcing, model.grid, clock, model_fields)
end

function TracerForcing(model, tracer_index; kwargs...)
    return TracerForcing(model, model.forcing[tracer_index], model.clock, fields(model); kwargs...)
end
#---

end # module
