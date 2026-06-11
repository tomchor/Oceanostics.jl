using Test
using CUDA: has_cuda_gpu, @allowscalar
using Oceananigans
using Oceananigans.Fields: location

using Oceanostics
using Oceanostics: SubfilterStressTensor, KineticEnergyCrossScaleFlux, GaussianFilter
using Oceanostics: StressTensor, StrainRateTensor

arch = has_cuda_gpu() ? GPU() : CPU()

# Interpolate any operation to cell centers, the location where the flux is contracted.
center(x) = @at (Center, Center, Center) x

#+++ Test functions
function test_subfilter_stress_tensor(model, filt)
    grid = model.grid
    τ = SubfilterStressTensor(model, filt)
    @test keys(τ) == (:τ₁₁, :τ₂₂, :τ₃₃, :τ₁₂, :τ₁₃, :τ₂₃)
    @test all(τᵢⱼ -> τᵢⱼ isa SubfilterStressTensor, τ) # every component is recognized by the type alias
    @test occursin("computes:", sprint(show, MIME"text/plain"(), τ.τ₁₃)) # and has the diagnostic display

    # each component lives at the same location as the corresponding `StressTensor` component
    ref = StressTensor(grid, model.velocities...)
    for k in keys(τ)
        @test location(τ[k]) == location(ref[k])
    end
    for τᵢⱼ in τ
        @test Field(τᵢⱼ) isa Field # every component is computable
    end

    # `collocate_diagonals` is forwarded to `StressTensor`: diagonals move to ccc, off-diagonals stay
    τc = SubfilterStressTensor(model, filt; collocate_diagonals=true)
    @test location(τc.τ₁₁) == (Center, Center, Center)
    @test location(τc.τ₂₂) == (Center, Center, Center)
    @test location(τc.τ₃₃) == (Center, Center, Center)
    @test (location(τc.τ₁₂), location(τc.τ₁₃), location(τc.τ₂₃)) ==
          (location(τ.τ₁₂),  location(τ.τ₁₃),  location(τ.τ₂₃))

    # `dims` selects sub-dimensional tensors, exactly like `StressTensor`
    @test keys(SubfilterStressTensor(model, filt; dims=(1, 3))) == (:τ₁₁, :τ₃₃, :τ₁₃)
    @test keys(SubfilterStressTensor(model, filt; dims=(2,)))   == (:τ₂₂,)

    # invalid `dims` are rejected
    @test_throws ArgumentError SubfilterStressTensor(model, filt; dims=(1, 4))
    @test_throws ArgumentError SubfilterStressTensor(model, filt; dims=())
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

    Π = KineticEnergyCrossScaleFlux(model, filt)
    @test location(Π) == (Center, Center, Center)
    @test interior(Field(Π)) ≈ interior(Field(Π_manual))

    # the flux is a single KernelFunctionOperation with a custom display (cf. PR #250, #254)
    @test Π isa KineticEnergyCrossScaleFlux
    @test occursin("KineticEnergyCrossScaleFlux", sprint(show, Π)) # two-arg show = compact summary
    @test occursin("computes:", sprint(show, MIME"text/plain"(), Π)) # three-arg show = full tree + description

    # reachable by the short name CoarseGrainedKineticEnergyEquation.CrossScaleFlux too (same type alias)
    @test CoarseGrainedKineticEnergyEquation.CrossScaleFlux === KineticEnergyCrossScaleFlux
    @test CoarseGrainedKineticEnergyEquation.CrossScaleFlux(model, filt) isa KineticEnergyCrossScaleFlux

    # invalid `dims` are rejected here too
    @test_throws ArgumentError KineticEnergyCrossScaleFlux(model, filt; dims=(1, 1))
    return nothing
end

# The Gaussian convenience method must reproduce the explicit filter-factory call with matching kwargs.
function test_convenience_method(model)
    σ = 0.12
    filt = ψ -> GaussianFilter(ψ; dims=(1, 2, 3), σ, boundary=:shrink) # :shrink is the convenience default
    @test keys(SubfilterStressTensor(model; σ)) == keys(SubfilterStressTensor(model, filt))
    @test interior(Field(KineticEnergyCrossScaleFlux(model; σ))) ≈
          interior(Field(KineticEnergyCrossScaleFlux(model, filt)))
    return nothing
end

# A uniform flow uⁱ = const has filter(uⁱuʲ) = ūⁱūʲ and ∂ūⁱ = 0, so both the subfilter stress and the
# cross-scale flux vanish identically.
function test_uniform_flow_vanishes(grid, filt; U=2, V=-3)
    model = NonhydrostaticModel(grid)
    set!(model, u=U, v=V) # w ≡ 0; a uniform horizontal flow is divergence-free

    τ = SubfilterStressTensor(model, filt)
    for τᵢⱼ in τ
        @test all(abs.(interior(Field(τᵢⱼ))) .< 1e-10)
    end
    @test all(abs.(interior(Field(KineticEnergyCrossScaleFlux(model, filt)))) .< 1e-10)
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
end
