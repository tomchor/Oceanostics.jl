using Test
using CUDA: has_cuda_gpu
using Oceananigans
using Oceananigans.Fields: location
using Oceananigans.AbstractOperations: compute_at!

using Oceanostics
using Oceanostics: SubFilterKineticEnergy, SubFilterKineticEnergyDissipationRate
using Oceanostics: KineticEnergyDissipationRate,
                   FilteredKineticEnergyDissipationRate, GaussianFilter

arch = has_cuda_gpu() ? GPU() : CPU()

#+++ Test functions
# eₖˢ = filter(eₖ) - eₖˡ: the filtered full kinetic energy minus the filtered-flow kinetic energy.
function test_subfilter_kinetic_energy_matches_manual(model, filt)
    u, v, w = model.velocities
    eₖˢ_manual = Field(filt(Field(Oceanostics.KineticEnergy(model, u, v, w)))) - FilteredKineticEnergy(model, filt)

    eₖˢ = SubFilterKineticEnergy(model, filt)
    @test location(eₖˢ) == (Center, Center, Center)
    @test interior(Field(eₖˢ)) ≈ interior(Field(eₖˢ_manual))

    # it is a single KernelFunctionOperation with its own type/display
    @test eₖˢ isa SubFilterKineticEnergy
    @test occursin("SubFilterKineticEnergy", sprint(show, eₖˢ))
    @test occursin("computes:", sprint(show, MIME("text/plain"), eₖˢ))
    return nothing
end

# The discrete energy decomposition filter(½uᵢuᵢ) = eₖˡ + eₖˢ holds exactly by construction — eₖˢ is defined
# as filter(eₖ) - eₖˡ — on any grid (here a bounded one), not just where the filter and interpolation commute.
function test_subfilter_kinetic_energy_decomposition(grid, filt)
    model = NonhydrostaticModel(grid)
    set!(model, u=(x, y, z) -> randn(), v=(x, y, z) -> randn(), w=(x, y, z) -> randn())
    filtered_K = Field(filt(Oceanostics.KineticEnergy(model)))
    eₖˡ = Field(FilteredKineticEnergy(model, filt))
    eₖˢ = Field(SubFilterKineticEnergy(model, filt))
    @test interior(filtered_K) ≈ interior(eₖˡ) .+ interior(eₖˢ)
    return nothing
end

# A uniform flow has τᵢⱼ ≡ 0, so the sub-filter kinetic energy vanishes identically.
function test_subfilter_kinetic_energy_uniform_vanishes(grid, filt; U=2, V=-3)
    model = NonhydrostaticModel(grid)
    set!(model, u=U, v=V) # w ≡ 0; a uniform horizontal flow is divergence-free
    @test all(abs.(interior(Field(SubFilterKineticEnergy(model, filt)))) .< 1e-10)
    return nothing
end

# The Gaussian convenience method must reproduce the explicit filter-factory call with matching kwargs.
function test_subfilter_kinetic_energy_convenience(model)
    σ = 0.12
    filt = ψ -> GaussianFilter(ψ; dims=(1, 2, 3), σ, boundary=:shrink) # :shrink is the convenience default
    @test interior(Field(SubFilterKineticEnergy(model; σ))) ≈
          interior(Field(SubFilterKineticEnergy(model, filt)))
    return nothing
end

# εₖˢ = filter(εₖ) - εₖˡ must equal the hand-built difference of the full-flow and filtered-flow dissipations.
function test_subfilter_dissipation_matches_manual(model, filt)
    εₖ  = KineticEnergyDissipationRate(model)
    εₖˡ = FilteredKineticEnergyDissipationRate(model, filt)
    εₖˢ_manual = Field(filt(εₖ)) - εₖˡ

    εₖˢ = SubFilterKineticEnergyDissipationRate(model, filt)
    @test location(εₖˢ) == (Center, Center, Center)
    @test interior(Field(εₖˢ)) ≈ interior(Field(εₖˢ_manual))

    # it is a single KernelFunctionOperation with its own type/display
    @test εₖˢ isa SubFilterKineticEnergyDissipationRate
    @test occursin("SubFilterKineticEnergyDissipationRate", sprint(show, εₖˢ))
    @test occursin("computes:", sprint(show, MIME("text/plain"), εₖˢ))
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

# Both diagnostics hold internally materialized filtered `Field`s (eₖˢ via `subfilter_stress_tensor`, εₖˢ via
# the nested `Field(filter(Field(ε)))`); recomputing at a new time — as an `OutputWriter` does each output —
# must reflect the updated flow through those nested fields, not stay frozen at construction. This mutates
# the model, so both are called last for their model.
function test_subfilter_kinetic_energy_recomputes(model, filt)
    Kf = Field(SubFilterKineticEnergy(model, filt))
    compute_at!(Kf, 0.0)
    snapshot = Array(interior(Kf))

    set!(model, u=(x, y, z) -> 2randn(), v=(x, y, z) -> 2randn(), w=(x, y, z) -> 2randn())
    compute_at!(Kf, 1.0)

    fresh = Field(SubFilterKineticEnergy(model, filt))
    compute_at!(fresh, 2.0)

    @test !(Array(interior(Kf)) ≈ snapshot)   # tracked the change in the flow
    @test interior(Kf) ≈ interior(fresh)      # equals an eₖˢ built fresh on the new state
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
    @test interior(εf) ≈ interior(fresh)      # equals an εₖˢ built fresh on the new state
    return nothing
end

# The module re-exports Πₖ (a source term of this budget) from `FilteredKineticEnergyEquation`, and
# aliases its dissipation rate as `DissipationRate`.
function test_subfilter_module_reexports()
    @test SubFilterKineticEnergyEquation.KineticEnergyCrossScaleFlux === KineticEnergyCrossScaleFlux
    @test SubFilterKineticEnergyEquation.DissipationRate === SubFilterKineticEnergyDissipationRate
    return nothing
end
#---

@testset "Sub-filter kinetic energy equation" begin
    @info "  Testing sub-filter kinetic energy diagnostics"
    grid = RectilinearGrid(arch, size=(8, 8, 8), extent=(1, 1, 1), topology=(Periodic, Periodic, Bounded))
    filt = ψ -> GaussianFilter(ψ; dims=(1, 2, 3), σ=0.1, boundary=:edge)

    model = NonhydrostaticModel(grid)
    set!(model, u=(x, y, z) -> randn(), v=(x, y, z) -> randn(), w=(x, y, z) -> randn())

    @info "    Sub-filter kinetic energy eₖˢ = ½τᵢᵢ matches manual"
    test_subfilter_kinetic_energy_matches_manual(model, filt)

    @info "    Gaussian convenience method (eₖˢ)"
    test_subfilter_kinetic_energy_convenience(model)

    @info "    Uniform flow vanishes (eₖˢ)"
    test_subfilter_kinetic_energy_uniform_vanishes(grid, ψ -> GaussianFilter(ψ; dims=(1, 2, 3), σ=0.1))

    @info "    eₖˢ recomputes as the flow evolves"
    test_subfilter_kinetic_energy_recomputes(model, filt) # mutates model; keep after the other `model` tests

    @info "    Discrete decomposition filter(eₖ) = eₖˡ + eₖˢ (bounded grid)"
    test_subfilter_kinetic_energy_decomposition(grid, filt)

    # εₖˢ needs a dissipative closure so the full- and filtered-flow dissipations are defined.
    @info "    Sub-filter KE dissipation εₖˢ = filter(εₖ) - εₖˡ matches manual"
    model_ν = NonhydrostaticModel(grid; closure=ScalarDiffusivity(ν=1e-3))
    set!(model_ν, u=(x, y, z) -> randn(), v=(x, y, z) -> randn(), w=(x, y, z) -> randn())
    test_subfilter_dissipation_matches_manual(model_ν, filt)
    test_subfilter_dissipation_convenience(model_ν)

    @info "    εₖˢ recomputes as the flow evolves"
    test_subfilter_dissipation_recomputes(model_ν, filt) # mutates model_ν; keep last

    @info "    Module re-exports (Πₖ) and aliases (DissipationRate)"
    test_subfilter_module_reexports()
end
