using Test
using CUDA: has_cuda_gpu
using Oceananigans
using Oceananigans.Fields: location
using Oceananigans.AbstractOperations: compute_at!

using Oceanostics
using Oceanostics: subfilter_stress_tensor, CoarseGrainedKineticEnergyCrossScaleFlux, GaussianFilter
using Oceanostics: CoarseGrainedKineticEnergyDissipationRate, KineticEnergyDissipationRate
using Oceanostics: StressTensor, StrainRateTensor

arch = has_cuda_gpu() ? GPU() : CPU()

# Interpolate any operation to cell centers, the location where the flux is contracted.
center(x) = @at (Center, Center, Center) x

#+++ Test functions
function test_subfilter_stress_tensor(model, filt)
    grid = model.grid
    τ = subfilter_stress_tensor(model, filt)
    @test keys(τ) == (:τ₁₁, :τ₂₂, :τ₃₃, :τ₁₂, :τ₁₃, :τ₂₃)

    # each component lives at the same location as the corresponding `StressTensor` component
    ref = StressTensor(grid, model.velocities...)
    for k in keys(τ)
        @test location(τ[k]) == location(ref[k])
    end
    for τᵢⱼ in τ
        @test Field(τᵢⱼ) isa Field # every component is computable
    end

    # `collocate_diagonals` is forwarded to `StressTensor`: diagonals move to ccc, off-diagonals stay
    τc = subfilter_stress_tensor(model, filt; collocate_diagonals=true)
    @test location(τc.τ₁₁) == (Center, Center, Center)
    @test location(τc.τ₂₂) == (Center, Center, Center)
    @test location(τc.τ₃₃) == (Center, Center, Center)
    @test (location(τc.τ₁₂), location(τc.τ₁₃), location(τc.τ₂₃)) ==
          (location(τ.τ₁₂),  location(τ.τ₁₃),  location(τ.τ₂₃))

    # `dims` selects sub-dimensional tensors, exactly like `StressTensor`
    @test keys(subfilter_stress_tensor(model, filt; dims=(1, 3))) == (:τ₁₁, :τ₃₃, :τ₁₃)
    @test keys(subfilter_stress_tensor(model, filt; dims=(2,)))   == (:τ₂₂,)

    # invalid `dims` are rejected
    @test_throws ArgumentError subfilter_stress_tensor(model, filt; dims=(1, 4))
    @test_throws ArgumentError subfilter_stress_tensor(model, filt; dims=())
    return nothing
end

# Build Πₖ = -τⁱʲ S̄ⁱʲ by hand from the same building blocks and check the module reproduces it. This
# guards the wiring of the contraction: the right components, the ×2 on off-diagonals, and the sign.
function test_cross_scale_ke_flux_matches_manual(model, filt)
    grid = model.grid
    u, v, w = model.velocities
    ū = Field(filt(u)); v̄ = Field(filt(v)); w̄ = Field(filt(w))

    full   = StressTensor(grid, u, v, w)    # uⁱuʲ
    coarse = StressTensor(grid, ū, v̄, w̄)    # ūⁱūʲ
    sub(f, c) = Field(filt(Field(f))) - c   # filter(uⁱuʲ) - ūⁱūʲ
    τ₁₁ = sub(full.τ₁₁, coarse.τ₁₁); τ₂₂ = sub(full.τ₂₂, coarse.τ₂₂); τ₃₃ = sub(full.τ₃₃, coarse.τ₃₃)
    τ₁₂ = sub(full.τ₁₂, coarse.τ₁₂); τ₁₃ = sub(full.τ₁₃, coarse.τ₁₃); τ₂₃ = sub(full.τ₂₃, coarse.τ₂₃)

    S̄ = StrainRateTensor(grid, ū, v̄, w̄)
    Π_manual = -(center(τ₁₁) * center(S̄.S₁₁) + center(τ₂₂) * center(S̄.S₂₂) + center(τ₃₃) * center(S̄.S₃₃) +
                 2center(τ₁₂) * center(S̄.S₁₂) + 2center(τ₁₃) * center(S̄.S₁₃) + 2center(τ₂₃) * center(S̄.S₂₃))

    Π = CoarseGrainedKineticEnergyCrossScaleFlux(model, filt)
    @test location(Π) == (Center, Center, Center)
    @test interior(Field(Π)) ≈ interior(Field(Π_manual))

    # the flux is a single KernelFunctionOperation with a custom display (cf. PR #250): the two-arg
    # `show` is a one-line summary, while the three-arg `MIME"text/plain"` show adds the `computes:` line
    @test Π isa CoarseGrainedKineticEnergyCrossScaleFlux
    @test occursin("CoarseGrainedKineticEnergyCrossScaleFlux", sprint(show, Π))
    @test occursin("computes:", sprint(show, MIME("text/plain"), Π))

    # reachable by the short name CoarseGrainedKineticEnergyEquation.CrossScaleFlux too (same type alias)
    @test CoarseGrainedKineticEnergyEquation.CrossScaleFlux === CoarseGrainedKineticEnergyCrossScaleFlux
    @test CoarseGrainedKineticEnergyEquation.CrossScaleFlux(model, filt) isa CoarseGrainedKineticEnergyCrossScaleFlux

    # invalid `dims` are rejected here too
    @test_throws ArgumentError CoarseGrainedKineticEnergyCrossScaleFlux(model, filt; dims=(1, 1))
    return nothing
end

# The Gaussian convenience method must reproduce the explicit filter-factory call with matching kwargs.
function test_convenience_method(model)
    σ = 0.12
    filt = ψ -> GaussianFilter(ψ; dims=(1, 2, 3), σ, boundary=:shrink) # :shrink is the convenience default
    @test keys(subfilter_stress_tensor(model; σ)) == keys(subfilter_stress_tensor(model, filt))
    @test interior(Field(CoarseGrainedKineticEnergyCrossScaleFlux(model; σ))) ≈
          interior(Field(CoarseGrainedKineticEnergyCrossScaleFlux(model, filt)))
    return nothing
end

# A uniform flow uⁱ = const has filter(uⁱuʲ) = ūⁱūʲ and ∂ūⁱ = 0, so both the subfilter stress and the
# cross-scale flux vanish identically.
function test_uniform_flow_vanishes(grid, filt; U=2, V=-3)
    model = NonhydrostaticModel(grid)
    set!(model, u=U, v=V) # w ≡ 0; a uniform horizontal flow is divergence-free

    τ = subfilter_stress_tensor(model, filt)
    for τᵢⱼ in τ
        @test all(abs.(interior(Field(τᵢⱼ))) .< 1e-10)
    end
    @test all(abs.(interior(Field(CoarseGrainedKineticEnergyCrossScaleFlux(model, filt)))) .< 1e-10)
    return nothing
end

# A reusable filter object (the documented idiom, `GaussianFilter(; …)`) can be passed directly as
# `filter`, giving the same result as an equivalent field-first closure.
function test_reusable_filter_object(model)
    reusable = GaussianFilter(; dims=(1, 2, 3), σ=0.1, boundary=:edge)
    closure  = ψ -> GaussianFilter(ψ; dims=(1, 2, 3), σ=0.1, boundary=:edge)
    @test keys(subfilter_stress_tensor(model, reusable)) == keys(subfilter_stress_tensor(model, closure))
    @test interior(Field(CoarseGrainedKineticEnergyCrossScaleFlux(model, reusable))) ≈
          interior(Field(CoarseGrainedKineticEnergyCrossScaleFlux(model, closure)))
    return nothing
end

# The diagnostic holds internally materialized filtered `Field`s; recomputing it at a new time — as an
# `OutputWriter` does each output — must reflect the updated flow, not stay frozen at construction.
function test_recomputes_on_evolution(model, filt)
    Πf = Field(CoarseGrainedKineticEnergyCrossScaleFlux(model, filt))
    compute_at!(Πf, 0.0)
    snapshot = Array(interior(Πf))

    set!(model, u=(x, y, z) -> 2randn(), v=(x, y, z) -> 2randn(), w=(x, y, z) -> 2randn())
    compute_at!(Πf, 1.0)

    fresh = Field(CoarseGrainedKineticEnergyCrossScaleFlux(model, filt))
    compute_at!(fresh, 2.0)

    @test !(Array(interior(Πf)) ≈ snapshot)   # tracked the change in the flow
    @test interior(Πf) ≈ interior(fresh)      # equals a flux built fresh on the new state
    return nothing
end

# Coarse-grained dissipation = the KE dissipation of the *filtered* flow. Reference: the existing
# `KineticEnergyDissipationRate` (same ∂ⱼuᵢ·Fᵢⱼ machinery) through its perturbation mechanism, with the
# mean set to the subfilter part `u - ū`, so the velocities it dissipates are exactly the filtered `ūᵢ`.
# This guards that the constructor swaps in the filtered velocities and reuses the viscous contraction.
function test_coarse_grained_dissipation_matches_filtered_flow(model, filt)
    u, v, w = model.velocities
    ū = Field(filt(u)); v̄ = Field(filt(v)); w̄ = Field(filt(w))

    ε     = CoarseGrainedKineticEnergyDissipationRate(model, filt)
    ε_ref = KineticEnergyDissipationRate(model; U=Field(u - ū), V=Field(v - v̄), W=Field(w - w̄))

    @test location(ε) == (Center, Center, Center)
    @test interior(Field(ε)) ≈ interior(Field(ε_ref))

    # its own type/display, distinct from KineticEnergyDissipationRate even though it reuses the contraction
    @test ε isa CoarseGrainedKineticEnergyDissipationRate
    @test !(ε isa KineticEnergyDissipationRate)
    @test occursin("CoarseGrainedKineticEnergyDissipationRate", sprint(show, ε))
    @test occursin("computes:", sprint(show, MIME("text/plain"), ε))

    # reachable by the short name CoarseGrainedKineticEnergyEquation.DissipationRate (same alias)
    @test CoarseGrainedKineticEnergyEquation.DissipationRate === CoarseGrainedKineticEnergyDissipationRate
    @test CoarseGrainedKineticEnergyEquation.DissipationRate(model, filt) isa CoarseGrainedKineticEnergyDissipationRate
    return nothing
end

# The Gaussian convenience method must reproduce the explicit filter-factory call with matching kwargs.
function test_coarse_grained_dissipation_convenience(model)
    σ = 0.12
    filt = ψ -> GaussianFilter(ψ; dims=(1, 2, 3), σ, boundary=:shrink) # :shrink is the convenience default
    @test interior(Field(CoarseGrainedKineticEnergyDissipationRate(model; σ))) ≈
          interior(Field(CoarseGrainedKineticEnergyDissipationRate(model, filt)))
    return nothing
end

# A uniform flow has S̄ᵢⱼ ≡ 0, so the filtered-flow dissipation vanishes identically.
function test_coarse_grained_dissipation_uniform_vanishes(grid, filt; ν=1e-3, U=2, V=-3)
    model = NonhydrostaticModel(grid; closure=ScalarDiffusivity(; ν))
    set!(model, u=U, v=V) # w ≡ 0; a uniform horizontal flow is divergence-free
    @test all(abs.(interior(Field(CoarseGrainedKineticEnergyDissipationRate(model, filt)))) .< 1e-10)
    return nothing
end

# Holds materialized filtered `Field`s — recomputing at a new time must reflect the new flow, not stay
# frozen. The filtered velocities sit in a NamedTuple argument, so this also checks that `compute_at!`
# refreshes them through that NamedTuple.
function test_coarse_grained_dissipation_recomputes(model, filt)
    εf = Field(CoarseGrainedKineticEnergyDissipationRate(model, filt))
    compute_at!(εf, 0.0)
    snapshot = Array(interior(εf))

    set!(model, u=(x, y, z) -> 2randn(), v=(x, y, z) -> 2randn(), w=(x, y, z) -> 2randn())
    compute_at!(εf, 1.0)

    fresh = Field(CoarseGrainedKineticEnergyDissipationRate(model, filt))
    compute_at!(fresh, 2.0)

    @test !(Array(interior(εf)) ≈ snapshot)   # tracked the change in the flow
    @test interior(εf) ≈ interior(fresh)      # equals a dissipation built fresh on the new state
    return nothing
end
#---

@testset "Coarse-grained kinetic energy equation" begin
    @info "  Testing coarse-grained kinetic energy diagnostics"
    grid = RectilinearGrid(arch, size=(8, 8, 8), extent=(1, 1, 1), topology=(Periodic, Periodic, Bounded))
    model = NonhydrostaticModel(grid)
    set!(model, u=(x, y, z) -> randn(), v=(x, y, z) -> randn(), w=(x, y, z) -> randn())

    filt = ψ -> GaussianFilter(ψ; dims=(1, 2, 3), σ=0.1, boundary=:edge)

    @info "    Subfilter stress tensor"
    test_subfilter_stress_tensor(model, filt)

    @info "    Cross-scale KE flux matches manual contraction"
    test_cross_scale_ke_flux_matches_manual(model, filt)

    @info "    Gaussian convenience method"
    test_convenience_method(model)

    @info "    Uniform flow vanishes"
    test_uniform_flow_vanishes(grid, ψ -> GaussianFilter(ψ; dims=(1, 2, 3), σ=0.1, boundary=:edge))

    @info "    Reusable filter object as `filter`"
    test_reusable_filter_object(model)

    @info "    Recomputes as the flow evolves"
    test_recomputes_on_evolution(model, filt)

    # The coarse-grained dissipation reuses KineticEnergyDissipationRate's viscous contraction, so it
    # needs a model with a closure; a constant-ν ScalarDiffusivity keeps the comparison clean.
    @info "    Coarse-grained (filtered-flow) KE dissipation"
    model_ν = NonhydrostaticModel(grid; closure=ScalarDiffusivity(ν=1e-3))
    set!(model_ν, u=(x, y, z) -> randn(), v=(x, y, z) -> randn(), w=(x, y, z) -> randn())
    test_coarse_grained_dissipation_matches_filtered_flow(model_ν, filt)
    test_coarse_grained_dissipation_convenience(model_ν)
    test_coarse_grained_dissipation_uniform_vanishes(grid, ψ -> GaussianFilter(ψ; dims=(1, 2, 3), σ=0.1, boundary=:edge))
    test_coarse_grained_dissipation_recomputes(model_ν, filt) # mutates model_ν; keep last
end
