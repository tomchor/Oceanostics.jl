module TracerVarianceBudgetTerms
using DocStringExtensions

export TracerVarianceDissipationRate, TracerVarianceTendency, TracerVarianceDiffusiveTerm

using Oceanostics: validate_location, validate_dissipative_closure, add_background_fields

using Oceananigans.Operators
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Grids: Center, Face
using Oceananigans: NonhydrostaticModel, fields
using Oceananigans.Models.NonhydrostaticModels: tracer_tendency
using Oceananigans.TurbulenceClosures: ∇_dot_qᶜ

import Oceananigans.TurbulenceClosures: diffusive_flux_x, diffusive_flux_y, diffusive_flux_z

#+++ Tracer variance tendency
@inline c∂ₜcᶜᶜᶜ(i, j, k, grid, val_tracer_index::Val{tracer_index},
                               val_tracer_name,
                               advection,
                               closure,
                               c_immersed_bc,
                               buoyancy,
                               biogeochemistry,
                               background_fields,
                               velocities,
                               tracers, args...) where tracer_index =
    @inbounds 2 * tracers[tracer_index][i, j, k] * tracer_tendency(i, j, k, grid,
                                                                   val_tracer_index,
                                                                   val_tracer_name,
                                                                   advection,
                                                                   closure,
                                                                   c_immersed_bc,
                                                                   buoyancy,
                                                                   biogeochemistry,
                                                                   background_fields,
                                                                   velocities,
                                                                   tracers,
                                                                   args...)

"""
    $(SIGNATURES)

Return a `KernelFunctionOperation` that computes the tracer variance tendency:

    TEND = 2 c ∂ₜc

where `c` is the tracer and `∂ₜc` is the tracer tendency (computed using
Oceananigans' tracer tendency kernel).

```julia
grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1))
model = NonhydrostaticModel(grid=grid, tracers=:b, closure=SmagorinskyLilly())

DIFF = TracerVarianceTendency(model, :b)
```
"""
function TracerVarianceTendency(model::NonhydrostaticModel, tracer_name; location = (Center, Center, Center))
    validate_location(location, "TracerVarianceTendency")
    tracer_index = findfirst(n -> n === tracer_name, propertynames(model.tracers))

    dependencies = (Val(tracer_index),
                    Val(tracer_name),
                    model.advection,
                    model.closure,
                    model.tracers[tracer_name].boundary_conditions.immersed,
                    model.buoyancy,
                    model.biogeochemistry,
                    model.background_fields,
                    model.velocities,
                    model.tracers,
                    model.auxiliary_fields,
                    model.diffusivity_fields,
                    model.forcing[tracer_name],
                    model.clock)

    return KernelFunctionOperation{Center, Center, Center}(c∂ₜcᶜᶜᶜ, model.grid, dependencies...)
end
#---

#+++ Tracer variance diffusive term
@inline c∇_dot_qᶜ(i, j, k, grid, closure,
                                 diffusivities,
                                 val_tracer_index,
                                 tracer,
                                 args...) =
    @inbounds 2 * tracer[i, j, k] * ∇_dot_qᶜ(i, j, k, grid, closure, diffusivities, val_tracer_index, tracer, args...)

"""
    $(SIGNATURES)

Return a `KernelFunctionOperation` that computes the diffusive term of the tracer variance
prognostic equation using Oceananigans' diffusive tracer flux divergence kernel:

```
    DIFF = 2 c ∂ⱼFⱼ
```
where `c` is the tracer, and `Fⱼ` is the tracer's diffusive flux in the `j`-th direction.

```julia
grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1))
model = NonhydrostaticModel(grid=grid, tracers=:b, closure=SmagorinskyLilly())

DIFF = TracerVarianceDiffusiveTerm(model, :b)
```
"""
function TracerVarianceDiffusiveTerm(model, tracer_name; location = (Center, Center, Center))
    validate_location(location, "TracerVarianceDiffusiveTerm")
    tracer_index = findfirst(n -> n === tracer_name, propertynames(model.tracers))

    dependencies = (model.closure,
                    model.diffusivity_fields,
                    Val(tracer_index),
                    model.tracers[tracer_name],
                    model.clock,
                    fields(model),
                    model.buoyancy)
    return KernelFunctionOperation{Center, Center, Center}(c∇_dot_qᶜ, model.grid, dependencies...)
end
#---

#+++ Tracer variance dissipation rate
for diff_flux in (:diffusive_flux_x, :diffusive_flux_y, :diffusive_flux_z)
    # Unroll the loop over a tuple
    @eval @inline $diff_flux(i, j, k, grid, closure_tuple::Tuple, diffusivity_fields, args...) =
        $diff_flux(i, j, k, grid, closure_tuple[1], diffusivity_fields[1], args...) + 
        $diff_flux(i, j, k, grid, closure_tuple[2:end], diffusivity_fields[2:end], args...)

    # End of the line
    @eval @inline $diff_flux(i, j, k, grid, closure_tuple::Tuple{}, args...) = zero(grid)
end

# Variance dissipation rate at fcc
@inline Axᶠᶜᶜ_δcᶠᶜᶜ_q₁ᶠᶜᶜ(i, j, k, grid, closure, diffusivity_fields, id, c, args...) =
    - Axᶠᶜᶜ(i, j, k, grid) * δxᶠᵃᵃ(i, j, k, grid, c) * diffusive_flux_x(i, j, k, grid, closure, diffusivity_fields, id, c, args...)

# Variance dissipation rate at cfc
@inline Ayᶜᶠᶜ_δcᶜᶠᶜ_q₂ᶜᶠᶜ(i, j, k, grid, closure, diffusivity_fields, id, c, args...) =
    - Ayᶜᶠᶜ(i, j, k, grid) * δyᵃᶠᵃ(i, j, k, grid, c) * diffusive_flux_y(i, j, k, grid, closure, diffusivity_fields, id, c, args...)

# Variance dissipation rate at ccf
@inline Azᶜᶜᶠ_δcᶜᶜᶠ_q₃ᶜᶜᶠ(i, j, k, grid, closure, diffusivity_fields, id, c, args...) =
    - Azᶜᶜᶠ(i, j, k, grid) * δzᵃᵃᶠ(i, j, k, grid, c) * diffusive_flux_z(i, j, k, grid, closure, diffusivity_fields, id, c, args...)

@inline tracer_variance_dissipation_rate_ccc(i, j, k, grid, args...) =
    2 * (ℑxᶜᵃᵃ(i, j, k, grid, Axᶠᶜᶜ_δcᶠᶜᶜ_q₁ᶠᶜᶜ, args...) + # F, C, C  → C, C, C
         ℑyᵃᶜᵃ(i, j, k, grid, Ayᶜᶠᶜ_δcᶜᶠᶜ_q₂ᶜᶠᶜ, args...) + # C, F, C  → C, C, C
         ℑzᵃᵃᶜ(i, j, k, grid, Azᶜᶜᶠ_δcᶜᶜᶠ_q₃ᶜᶜᶠ, args...)   # C, C, F  → C, C, C
         ) / Vᶜᶜᶜ(i, j, k, grid) # This division by volume, coupled with the call to A*δc above, ensures a derivative operation

"""
    $(SIGNATURES)

Return a `KernelFunctionOperation` that computes the isotropic variance dissipation rate
for `tracer_name` in `model.tracers`. The isotropic variance dissipation rate is defined as 

```
    χ = 2 ∂ⱼc ⋅ Fⱼ
```
where `Fⱼ` is the diffusive flux of `c` in the `j`-th direction and `∂ⱼ` is the gradient operator.
`χ` is implemented in its conservative formulation based on the equation above. 

Note that often `χ` is written as `χ = 2κ (∇c ⋅ ∇c)`, which is the special case for Fickian diffusion
(`κ` is the tracer diffusivity).

Here `tracer_name` is needed even when passing `tracer` in order to get the appropriate `tracer_index`.
When passing `tracer`, this function should be used as

```julia
grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1))
model = NonhydrostaticModel(grid=grid, tracers=:b, closure=SmagorinskyLilly())

b̄ = Field(Average(model.tracers.b, dims=(1,2)))
b′ = model.tracers.b - b̄

χb = TracerVarianceDissipationRate(model, :b, tracer=b′)
```
"""
function TracerVarianceDissipationRate(model, tracer_name; tracer = nothing, location = (Center, Center, Center))
    validate_location(location, "TracerVarianceDissipationRate")
    tracer_index = findfirst(n -> n === tracer_name, propertynames(model.tracers))

    parameters = (; model.closure, model.clock, model.buoyancy,
                  id = Val(tracer_index))

    tracer = tracer == nothing ? model.tracers[tracer_name] : tracer
    return KernelFunctionOperation{Center, Center, Center}(tracer_variance_dissipation_rate_ccc, model.grid,
                                                           model.closure,
                                                           model.diffusivity_fields,
                                                           Val(tracer_index),
                                                           tracer, 
                                                           model.clock,
                                                           fields(model),
                                                           model.buoyancy)
end
#---

end # module
