module TracerEquation
using DocStringExtensions

using Oceananigans: fields, Center, Face, KernelFunctionOperation
using Oceananigans.Models: HydrostaticFreeSurfaceModel
using Oceananigans.Models.NonhydrostaticModels: div_Uc, ∇_dot_qᶜ, immersed_∇_dot_qᶜ, biogeochemical_transition
using Oceananigans.TurbulenceClosures: diffusive_flux_x, diffusive_flux_y, diffusive_flux_z

using Oceanostics: validate_location, CustomKFO

export Advection, Diffusion, ImmersedDiffusion, TotalDiffusion, Forcing,
       TracerAdvection, TracerDiffusion, TracerImmersedDiffusion, TracerTotalDiffusion, 
       TracerDiffusiveFluxX, TracerDiffusiveFluxY, TracerDiffusiveFluxZ, TracerForcing

# Inline function for total diffusion
@inline total_∇_dot_qᶜ(i, j, k, grid, c, c_immersed_bc, closure, closure_fields, val_tracer_index, clock, model_fields, buoyancy) =
    ∇_dot_qᶜ(i, j, k, grid, closure, closure_fields, val_tracer_index, c, clock, model_fields, buoyancy) +
    immersed_∇_dot_qᶜ(i, j, k, grid, c, c_immersed_bc, closure, closure_fields, val_tracer_index, clock, model_fields, buoyancy)

# Type aliases for major functions
const Advection = CustomKFO{<:typeof(div_Uc)}
const Diffusion = CustomKFO{<:typeof(∇_dot_qᶜ)}
const ImmersedDiffusion = CustomKFO{<:typeof(immersed_∇_dot_qᶜ)}
const TotalDiffusion = CustomKFO{<:typeof(total_∇_dot_qᶜ)}
const DiffusiveFluxX = CustomKFO{<:typeof(diffusive_flux_x)}
const DiffusiveFluxY = CustomKFO{<:typeof(diffusive_flux_y)}
const DiffusiveFluxZ = CustomKFO{<:typeof(diffusive_flux_z)}
const Forcing = KernelFunctionOperation{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any}

const TracerAdvection = Advection
const TracerDiffusion = Diffusion
const TracerImmersedDiffusion = ImmersedDiffusion
const TracerTotalDiffusion = TotalDiffusion
const TracerDiffusiveFluxX = DiffusiveFluxX
const TracerDiffusiveFluxY = DiffusiveFluxY
const TracerDiffusiveFluxZ = DiffusiveFluxZ
const TracerForcing = Forcing

#+++ Advection
"""
    $(SIGNATURES)

Calculates the advection of the tracer `c` as

    ADV = ∂ⱼ (uⱼ c)

using Oceananigans' kernel [`div_Uc`.](https://clima.github.io/OceananigansDocumentation/stable/appendix/library/#Oceananigans.Advection.div_Uc-NTuple{7,%20Any})

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(grid; tracers=:a);

julia> ADV = TracerEquation.Advection(model, :a)
TracerAdvection KernelFunctionOperation at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: div_Uc (generic function with 12 methods)
└── arguments: ("Centered", "NamedTuple", "Field")
└── computes: tracer advection  ∂ⱼ(uⱼc)
```
"""
function Advection(model, u, v, w, c, advection; location = (Center, Center, Center))
    validate_location(location, "Advection", (Center, Center, Center))
    return KernelFunctionOperation{Center, Center, Center}(div_Uc, model.grid, advection, (; u, v, w), c)
end

function Advection(model, tracer_name; kwargs...)
    @inbounds c = model.tracers[tracer_name]
    return Advection(model, model.velocities..., c, model.advection; kwargs...)
end

function Advection(model::HydrostaticFreeSurfaceModel, tracer_name; kwargs...)
    @inbounds c = model.tracers[tracer_name]
    tracer_advection = model.advection[tracer_name]
    return Advection(model, model.velocities..., c, tracer_advection; kwargs...)
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

julia> model = NonhydrostaticModel(grid; tracers=:a);

julia> DIFF = TracerEquation.Diffusion(model, :a)
TracerDiffusion KernelFunctionOperation at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: ∇_dot_qᶜ (generic function with 10 methods)
└── arguments: ("Nothing", "Nothing", "Val", "Field", "Clock", "NamedTuple", "Nothing")
└── computes: tracer diffusion (interior)  ∂ⱼqᶜⱼ
```
"""
function Diffusion(model, val_tracer_index, c, closure, closure_fields, clock, model_fields, buoyancy; location = (Center, Center, Center))
    validate_location(location, "Diffusion", (Center, Center, Center))
    return KernelFunctionOperation{Center, Center, Center}(∇_dot_qᶜ, model.grid, closure, closure_fields, val_tracer_index, c, clock, model_fields, buoyancy)
end

function Diffusion(model, tracer_name; kwargs...)
    tracer_index = findfirst(x -> x == tracer_name, keys(model.tracers))
    @inbounds c = model.tracers[tracer_name]
    return Diffusion(model, Val(tracer_index), c, model.closure, model.closure_fields, model.clock, fields(model), model.buoyancy; kwargs...)
end


"""
    $(SIGNATURES)

Calculates the diffusion term due to the bathymetry term as

    DIFF = ∂ⱼ 𝓆ᶜⱼ,

where 𝓆ᶜⱼ is the bathymetry-led diffusion tensor for tracer `c`, using the Oceananigans' kernel
`immersed_∇_dot_qᶜ`.

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(grid; tracers=:a);

julia> DIFF = TracerEquation.ImmersedDiffusion(model, :a)
TracerImmersedDiffusion KernelFunctionOperation at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: immersed_∇_dot_qᶜ (generic function with 2 methods)
└── arguments: ("Field", "Nothing", "Nothing", "Nothing", "Val", "Clock", "NamedTuple")
└── computes: tracer diffusion through immersed boundaries  ∂ⱼ𝓆ᶜⱼ
```
"""
function ImmersedDiffusion(model, c, c_immersed_bc, closure, closure_fields, val_tracer_index, clock, model_fields; location = (Center, Center, Center))
    validate_location(location, "ImmersedDiffusion", (Center, Center, Center))
    return KernelFunctionOperation{Center, Center, Center}(immersed_∇_dot_qᶜ, model.grid, c, c_immersed_bc, closure, closure_fields, val_tracer_index, clock, model_fields)
end

function ImmersedDiffusion(model, tracer_name; kwargs...)
    tracer_index = findfirst(x -> x == tracer_name, keys(model.tracers))
    tracer = model.tracers[tracer_name]
    immersed_bc = tracer.boundary_conditions.immersed
    return ImmersedDiffusion(model, tracer, immersed_bc, model.closure, model.closure_fields, Val(tracer_index), model.clock, fields(model); kwargs...)
end

"""
    $(SIGNATURES)

Calculates the total diffusion term as

    DIFF = ∂ⱼ qᶜⱼ + ∂ⱼ 𝓆ᶜⱼ,

`c`. The calculation is done using the Oceananigans' kernels `∇_dot_qᶜ` and `immersed_∇_dot_qᶜ`.
where qᶜⱼ is the interior diffusion tensor and 𝓆ᶜⱼ is the bathymetry-led diffusion tensor for tracer

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(grid; tracers=:a);

julia> DIFF = TracerEquation.TotalDiffusion(model, :a)
TracerTotalDiffusion KernelFunctionOperation at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: total_∇_dot_qᶜ (generic function with 1 method)
└── arguments: ("Field", "Nothing", "Nothing", "Nothing", "Val", "Clock", "NamedTuple", "Nothing")
└── computes: total tracer diffusion (interior + immersed)  ∂ⱼqᶜⱼ + ∂ⱼ𝓆ᶜⱼ
```
"""
function TotalDiffusion(model, c, c_immersed_bc, closure, closure_fields, val_tracer_index, clock, model_fields, buoyancy; location = (Center, Center, Center))
    validate_location(location, "TotalDiffusion", (Center, Center, Center))
    return KernelFunctionOperation{Center, Center, Center}(total_∇_dot_qᶜ, model.grid, c, c_immersed_bc, closure, closure_fields, val_tracer_index, clock, model_fields, buoyancy)
end

function TotalDiffusion(model, tracer_name; kwargs...)
    tracer_index = findfirst(x -> x == tracer_name, keys(model.tracers))
    tracer = model.tracers[tracer_index]
    immersed_bc = tracer.boundary_conditions.immersed
    return TotalDiffusion(model, tracer, immersed_bc, model.closure, model.closure_fields, Val(tracer_index), model.clock, fields(model), model.buoyancy; kwargs...)
end

"""
    $(SIGNATURES)

Calculates the sub-grid diffusive flux in the x-direction as determined by the 
configured closure, using the Oceananigans kernel `diffusive_flux_x`.

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> closure = AnisotropicMinimumDissipation()

julia> model = NonhydrostaticModel(grid; closure, tracers=:a);

julia> DIFF_FLUX_X = TracerEquation.DiffusiveFluxX(model, :a)
TracerDiffusiveFluxX KernelFunctionOperation at (Face, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: diffusive_flux_x (generic function with 15 methods)
└── arguments: ("AnisotropicMinimumDissipation", "NamedTuple", "Val", "Field", "Clock", "NamedTuple", "Nothing")
└── computes: sub-grid diffusive flux given by the configured closure
```
"""
function DiffusiveFluxX(model, grid, closure, closure_fields, val_tracer_index, tracer, clock, model_fields, buoyancy; location = (Face, Center, Center))
    validate_location(location, "DiffusiveFluxX", (Face, Center, Center))
    return KernelFunctionOperation{Face, Center, Center}(diffusive_flux_x, grid, closure, closure_fields, val_tracer_index, tracer, clock, model_fields, buoyancy)
end

function DiffusiveFluxX(model, tracer_name; kwargs...)
    tracer_index = findfirst(x -> x == tracer_name, keys(model.tracers))
    tracer = model.tracers[tracer_index]
    return DiffusiveFluxX(model, model.grid, model.closure, model.closure_fields, Val(tracer_index), tracer, model.clock, fields(model), model.buoyancy; kwargs...)
end

"""
    $(SIGNATURES)

Calculates the sub-grid diffusive flux in the y-direction as determined by the 
configured closure, using the Oceananigans kernel `diffusive_flux_y`.

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> closure = AnisotropicMinimumDissipation()

julia> model = NonhydrostaticModel(grid; closure, tracers=:a);

julia> DIFF_FLUX_Y = TracerEquation.DiffusiveFluxY(model, :a)
TracerDiffusiveFluxY KernelFunctionOperation at (Center, Face, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: diffusive_flux_y (generic function with 15 methods)
└── arguments: ("AnisotropicMinimumDissipation", "NamedTuple", "Val", "Field", "Clock", "NamedTuple", "Nothing")
└── computes: sub-grid diffusive flux given by the configured closure
```
"""
function DiffusiveFluxY(model, grid, closure, closure_fields, val_tracer_index, tracer, clock, model_fields, buoyancy; location = (Center, Face, Center))
    validate_location(location, "DiffusiveFluxY", (Center, Face, Center))
    return KernelFunctionOperation{Center, Face, Center}(diffusive_flux_y, grid, closure, closure_fields, val_tracer_index, tracer, clock, model_fields, buoyancy)
end

function DiffusiveFluxY(model, tracer_name; kwargs...)
    tracer_index = findfirst(x -> x == tracer_name, keys(model.tracers))
    tracer = model.tracers[tracer_index]
    return DiffusiveFluxY(model, model.grid, model.closure, model.closure_fields, Val(tracer_index), tracer, model.clock, fields(model), model.buoyancy; kwargs...)
end

"""
    $(SIGNATURES)

Calculates the sub-grid diffusive flux in the z-direction as determined by the 
configured closure, using the Oceananigans kernel `diffusive_flux_z`.

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> closure = AnisotropicMinimumDissipation()

julia> model = NonhydrostaticModel(grid; closure, tracers=:a);

julia> DIFF_FLUX_Z = TracerEquation.DiffusiveFluxZ(model, :a)
TracerDiffusiveFluxZ KernelFunctionOperation at (Center, Center, Face)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: diffusive_flux_z (generic function with 17 methods)
└── arguments: ("AnisotropicMinimumDissipation", "NamedTuple", "Val", "Field", "Clock", "NamedTuple", "Nothing")
└── computes: sub-grid diffusive flux given by the configured closure
```
"""
function DiffusiveFluxZ(model, grid, closure, closure_fields, val_tracer_index, tracer, clock, model_fields, buoyancy; location = (Center, Center, Face))
    validate_location(location, "DiffusiveFluxZ", (Center, Center, Face))
    return KernelFunctionOperation{Center, Center, Face}(diffusive_flux_z, grid, closure, closure_fields, val_tracer_index, tracer, clock, model_fields, buoyancy)
end

function DiffusiveFluxZ(model, tracer_name; kwargs...)
    tracer_index = findfirst(x -> x == tracer_name, keys(model.tracers))
    tracer = model.tracers[tracer_index]
    return DiffusiveFluxZ(model, model.grid, model.closure, model.closure_fields, Val(tracer_index), tracer, model.clock, fields(model), model.buoyancy; kwargs...)
end
#---

#+++ Forcing
"""
    $(SIGNATURES)

Calculate the forcing term `Fᶜ` on the equation for tracer `c` for `model`.

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(grid; tracers=:a);

julia> FORC = TracerEquation.Forcing(model, :a)
KernelFunctionOperation at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: Returns (generic function with 1 method)
└── arguments: ("Clock", "NamedTuple")
```
"""
function Forcing(model, forcing, clock, model_fields; location = (Center, Center, Center))
    return KernelFunctionOperation{Center, Center, Center}(forcing, model.grid, clock, model_fields)
end

function Forcing(model, tracer_name; kwargs...)
    return Forcing(model, model.forcing[tracer_name], model.clock, fields(model); kwargs...)
end
#---

end # module
