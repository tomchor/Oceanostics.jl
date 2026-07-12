using Test
using CUDA: has_cuda_gpu
using Oceananigans
using Oceananigans.Fields: location

using Oceanostics
using Oceanostics: subfilter_kinetic_energy, subfilter_kinetic_energy_dissipation_rate
using Oceanostics: subfilter_stress_tensor, KineticEnergyDissipationRate,
                   CoarseGrainedKineticEnergyDissipationRate, GaussianFilter

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
    εˡ = CoarseGrainedKineticEnergyDissipationRate(model, filt)
    εˢ_manual = Field(filt(ε)) - εˡ

    εˢ = subfilter_kinetic_energy_dissipation_rate(model, filt)
    @test location(εˢ) == (Center, Center, Center)
    @test interior(Field(εˢ)) ≈ interior(Field(εˢ_manual))
    return nothing
end

# The Gaussian convenience method must reproduce the explicit filter-factory call with matching kwargs.
function test_subfilter_dissipation_convenience(model)
    σ = 0.12
    filt = ψ -> GaussianFilter(ψ; dims=(1, 2, 3), σ, boundary=:shrink) # :shrink is the convenience default
    @test interior(Field(subfilter_kinetic_energy_dissipation_rate(model; σ))) ≈
          interior(Field(subfilter_kinetic_energy_dissipation_rate(model, filt)))
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

    # εˢ needs a dissipative closure so the full- and filtered-flow dissipations are defined.
    @info "    Sub-filter KE dissipation εˢ = filter(ε) - εˡ matches manual"
    model_ν = NonhydrostaticModel(grid; closure=ScalarDiffusivity(ν=1e-3))
    set!(model_ν, u=(x, y, z) -> randn(), v=(x, y, z) -> randn(), w=(x, y, z) -> randn())
    test_subfilter_dissipation_matches_manual(model_ν, filt)
    test_subfilter_dissipation_convenience(model_ν)
end
