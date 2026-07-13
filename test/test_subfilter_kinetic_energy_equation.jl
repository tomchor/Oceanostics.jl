using Test
using CUDA: has_cuda_gpu
using Oceananigans
using Oceananigans.Fields: location
using Oceananigans.AbstractOperations: compute_at!

using Oceanostics
using Oceanostics: subfilter_kinetic_energy, SubFilterKineticEnergyDissipationRate
using Oceanostics: subfilter_stress_tensor, KineticEnergyDissipationRate,
                   FilteredKineticEnergyDissipationRate, GaussianFilter

arch = has_cuda_gpu() ? GPU() : CPU()

#+++ Test functions
# Kˢ = ½τⁱⁱ must equal the hand-built sum of the collocated diagonal sub-filter stresses (the way the
# Rayleigh-Taylor example builds it).
function test_subfilter_kinetic_energy_matches_manual(model, filt)
    τ = subfilter_stress_tensor(model, filt; collocate_diagonals=true)
    Kˢ_manual = (τ.τ₁₁ + τ.τ₂₂ + τ.τ₃₃) / 2

    Kˢ = subfilter_kinetic_energy(model, filt)
    @test location(Kˢ) == (Center, Center, Center)
    @test interior(Field(Kˢ)) ≈ interior(Field(Kˢ_manual))

    # a `dims` subset keeps only the diagonals it retains: ½(τ₁₁ + τ₃₃)
    τ13 = subfilter_stress_tensor(model, filt; dims=(1, 3), collocate_diagonals=true)
    Kˢ13 = subfilter_kinetic_energy(model, filt; dims=(1, 3))
    @test interior(Field(Kˢ13)) ≈ interior(Field((τ13.τ₁₁ + τ13.τ₃₃) / 2))
    return nothing
end

# A uniform flow has τⁱʲ ≡ 0, so the sub-filter kinetic energy vanishes identically.
function test_subfilter_kinetic_energy_uniform_vanishes(grid, filt; U=2, V=-3)
    model = NonhydrostaticModel(grid)
    set!(model, u=U, v=V) # w ≡ 0; a uniform horizontal flow is divergence-free
    @test all(abs.(interior(Field(subfilter_kinetic_energy(model, filt)))) .< 1e-10)
    return nothing
end

# The Gaussian convenience method must reproduce the explicit filter-factory call with matching kwargs.
function test_subfilter_kinetic_energy_convenience(model)
    σ = 0.12
    filt = ψ -> GaussianFilter(ψ; dims=(1, 2, 3), σ, boundary=:shrink) # :shrink is the convenience default
    @test interior(Field(subfilter_kinetic_energy(model; σ))) ≈
          interior(Field(subfilter_kinetic_energy(model, filt)))
    return nothing
end

# εˢ = filter(ε) - εˡ must equal the hand-built difference of the full-flow and filtered-flow dissipations.
function test_subfilter_dissipation_matches_manual(model, filt)
    ε  = KineticEnergyDissipationRate(model)
    εˡ = FilteredKineticEnergyDissipationRate(model, filt)
    εˢ_manual = Field(filt(ε)) - εˡ

    εˢ = SubFilterKineticEnergyDissipationRate(model, filt)
    @test location(εˢ) == (Center, Center, Center)
    @test interior(Field(εˢ)) ≈ interior(Field(εˢ_manual))

    # it is a single KernelFunctionOperation with its own type/display
    @test εˢ isa SubFilterKineticEnergyDissipationRate
    @test occursin("SubFilterKineticEnergyDissipationRate", sprint(show, εˢ))
    @test occursin("computes:", sprint(show, MIME("text/plain"), εˢ))
    return nothing
end

# The Gaussian convenience method must reproduce the explicit filter-factory call with matching kwargs.
function test_subfilter_dissipation_convenience(model)
    σ = 0.12
    filt = ψ -> GaussianFilter(ψ; dims=(1, 2, 3), σ, boundary=:shrink) # :shrink is the convenience default
    @test interior(Field(SubFilterKineticEnergyDissipationRate(model; σ))) ≈
          interior(Field(SubFilterKineticEnergyDissipationRate(model, filt)))
    return nothing
end

# Both diagnostics hold internally materialized filtered `Field`s (Kˢ via `subfilter_stress_tensor`, εˢ via
# the nested `Field(filter(Field(ε)))`); recomputing at a new time — as an `OutputWriter` does each output —
# must reflect the updated flow through those nested fields, not stay frozen at construction. This mutates
# the model, so both are called last for their model.
function test_subfilter_kinetic_energy_recomputes(model, filt)
    Kf = Field(subfilter_kinetic_energy(model, filt))
    compute_at!(Kf, 0.0)
    snapshot = Array(interior(Kf))

    set!(model, u=(x, y, z) -> 2randn(), v=(x, y, z) -> 2randn(), w=(x, y, z) -> 2randn())
    compute_at!(Kf, 1.0)

    fresh = Field(subfilter_kinetic_energy(model, filt))
    compute_at!(fresh, 2.0)

    @test !(Array(interior(Kf)) ≈ snapshot)   # tracked the change in the flow
    @test interior(Kf) ≈ interior(fresh)      # equals a Kˢ built fresh on the new state
    return nothing
end

function test_subfilter_dissipation_recomputes(model, filt)
    εf = Field(SubFilterKineticEnergyDissipationRate(model, filt))
    compute_at!(εf, 0.0)
    snapshot = Array(interior(εf))

    set!(model, u=(x, y, z) -> 2randn(), v=(x, y, z) -> 2randn(), w=(x, y, z) -> 2randn())
    compute_at!(εf, 1.0)

    fresh = Field(SubFilterKineticEnergyDissipationRate(model, filt))
    compute_at!(fresh, 2.0)

    @test !(Array(interior(εf)) ≈ snapshot)   # tracked the change in the flow
    @test interior(εf) ≈ interior(fresh)      # equals an εˢ built fresh on the new state
    return nothing
end
#---

@testset "Sub-filter kinetic energy equation" begin
    @info "  Testing sub-filter kinetic energy diagnostics"
    grid = RectilinearGrid(arch, size=(8, 8, 8), extent=(1, 1, 1), topology=(Periodic, Periodic, Bounded))
    filt = ψ -> GaussianFilter(ψ; dims=(1, 2, 3), σ=0.1, boundary=:edge)

    model = NonhydrostaticModel(grid)
    set!(model, u=(x, y, z) -> randn(), v=(x, y, z) -> randn(), w=(x, y, z) -> randn())

    @info "    Sub-filter kinetic energy Kˢ = ½τⁱⁱ matches manual"
    test_subfilter_kinetic_energy_matches_manual(model, filt)

    @info "    Gaussian convenience method (Kˢ)"
    test_subfilter_kinetic_energy_convenience(model)

    @info "    Uniform flow vanishes (Kˢ)"
    test_subfilter_kinetic_energy_uniform_vanishes(grid, ψ -> GaussianFilter(ψ; dims=(1, 2, 3), σ=0.1))

    @info "    Kˢ recomputes as the flow evolves"
    test_subfilter_kinetic_energy_recomputes(model, filt) # mutates model; keep after the other `model` tests

    # εˢ needs a dissipative closure so the full- and filtered-flow dissipations are defined.
    @info "    Sub-filter KE dissipation εˢ = filter(ε) - εˡ matches manual"
    model_ν = NonhydrostaticModel(grid; closure=ScalarDiffusivity(ν=1e-3))
    set!(model_ν, u=(x, y, z) -> randn(), v=(x, y, z) -> randn(), w=(x, y, z) -> randn())
    test_subfilter_dissipation_matches_manual(model_ν, filt)
    test_subfilter_dissipation_convenience(model_ν)

    @info "    εˢ recomputes as the flow evolves"
    test_subfilter_dissipation_recomputes(model_ν, filt) # mutates model_ν; keep last
end
