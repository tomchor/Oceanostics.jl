using Oceananigans.Operators

#+++ Energy conservation
import Oceananigans.Models.NonhydrostaticModels: u_velocity_tendency, v_velocity_tendency, w_velocity_tendency
using Oceananigans.TurbulenceClosures: ∂ⱼ_τ₁ⱼ, ∂ⱼ_τ₂ⱼ, ∂ⱼ_τ₃ⱼ

@inline ψf(i, j, k, grid, ψ, f, args...) = ψ[i, j, k] * f(i, j, k, grid, args...)

@inline function uᵢ∂ⱼ_τᵢⱼᶜᶜᶜ(i, j, k, grid, closure,
                                            diffusivity_fields,
                                            clock,
                                            model_fields,
                                            buoyancy,
                                            )
    u∂ⱼ_τ₁ⱼ = ℑxᶜᵃᵃ(i, j, k, grid, ψf, model_fields.u, ∂ⱼ_τ₁ⱼ, closure, diffusivity_fields, clock, model_fields, buoyancy)
    v∂ⱼ_τ₂ⱼ = ℑyᵃᶜᵃ(i, j, k, grid, ψf, model_fields.v, ∂ⱼ_τ₂ⱼ, closure, diffusivity_fields, clock, model_fields, buoyancy)
    w∂ⱼ_τ₃ⱼ = ℑzᵃᵃᶜ(i, j, k, grid, ψf, model_fields.w, ∂ⱼ_τ₃ⱼ, closure, diffusivity_fields, clock, model_fields, buoyancy)

    return u∂ⱼ_τ₁ⱼ+ v∂ⱼ_τ₂ⱼ + w∂ⱼ_τ₃ⱼ
end

dependencies2 = (model.closure,
                 model.diffusivity_fields,
                 model.clock,
                 fields(model),
                 model.buoyancy,
                 )
#---

