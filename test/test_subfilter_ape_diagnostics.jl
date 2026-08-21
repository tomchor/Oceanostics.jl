using Test
using CUDA: has_cuda_gpu
using Oceananigans
using Oceananigans.Fields: location, compute_at!

using Oceanostics
using Oceanostics: SubFilterAvailablePotentialEnergy, SubFilterAvailablePotentialEnergyDissipationRate
using Oceanostics: FilteredAvailablePotentialEnergy, FilteredAvailablePotentialEnergyDissipationRate,
                   AvailablePotentialEnergyCrossScaleFlux
using Oceanostics: AvailablePotentialEnergy, AvailablePotentialEnergyDissipationRate,
                   reference_height, reference_buoyancy, VerticalSort, ProfileLookup, HeavisideIntegral,
                   GaussianFilter

arch = has_cuda_gpu() ? GPU() : CPU()

# A random stably stratified buoyancy, shared by most tests below.
random_stratified_b(x, y, z) = 1e-2 * z + 1e-3 * randn()

#+++ Test functions
# eₐˢ = filter(eₐ) - eₐˡ must equal the hand-built difference, with both terms measured against one
# shared reference profile: the full and the filtered buoyancy each looked up in the same VerticalSort
# column via ProfileLookup.
function test_subfilter_ape_matches_manual(model, filt)
    col    = reference_height(model, method=VerticalSort())
    lookup = ProfileLookup(col)
    b  = model.tracers.b
    b̄  = Field(filt(b))
    z✶  = reference_height(b;  method=lookup)
    z✶ˡ = reference_height(b̄; method=lookup)
    eₐˢ_manual = Field(filt(Field(AvailablePotentialEnergy(model, z✶)))) - AvailablePotentialEnergy(model, z✶ˡ)

    eₐˢ = SubFilterAvailablePotentialEnergy(model, filt)
    @test location(eₐˢ) == (Center, Center, Center)
    @test interior(Field(eₐˢ)) ≈ interior(Field(eₐˢ_manual))

    # it is a single KernelFunctionOperation with its own type/display
    @test eₐˢ isa SubFilterAvailablePotentialEnergy
    @test occursin("SubFilterAvailablePotentialEnergy", sprint(show, eₐˢ))
    @test occursin("computes:", sprint(show, MIME("text/plain"), eₐˢ))
    return nothing
end

# A filter that is numerically the identity (σ ≪ Δx truncated at N=3, so the off-center Gaussian
# weights underflow against the center one) makes b̄ = b bit for bit, hence z✶ˡ = z✶, eₐˡ = eₐ and
# q̄ᵢ = qᵢ, Υˡ = Υ. Both diagnostics must then vanish identically, which checks the filtered-state
# kernels against the full-state ones with no reimplementation: `filtered_ape_dissipation_rate_ccc`
# reproduces `ape_dissipation_rate_ccc` exactly when handed the unfiltered fluxes.
function test_subfilter_ape_identity_filter_vanishes(model)
    identity_filter = ψ -> GaussianFilter(ψ; dims=(1, 2, 3), σ=1e-4, N=3)
    @test all(interior(Field(SubFilterAvailablePotentialEnergy(model, identity_filter))) .== 0)
    @test all(interior(Field(SubFilterAvailablePotentialEnergyDissipationRate(model, identity_filter))) .== 0)
    return nothing
end

# A horizontally uniform, stable stratification filtered horizontally: b̄ = b up to roundoff, and both
# eₐ and eₐˡ vanish (z✶ = z cell by cell), so eₐˢ ≈ 0. eₐ is blind to where in a tied run the lookup
# lands, so the roundoff in b̄ cannot leak an O(Δz) error into this test.
function test_subfilter_ape_uniform_stratification_vanishes(grid, filt_horizontal)
    model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b)
    set!(model, b=(x, y, z) -> 1e-2 * z)
    eₐˢ = Field(SubFilterAvailablePotentialEnergy(model, filt_horizontal))
    @test maximum(abs, interior(eₐˢ)) < 1e-12
    return nothing
end

# eₐ is convex in buoyancy, so for a filter with no vertical component Jensen's inequality makes
# eₐˢ ≥ 0 pointwise (exactly, up to roundoff, since the full field's buoyancies are the profile's own
# entries). A constant buoyancy is the degenerate case: no available energy at any scale.
function test_subfilter_ape_signs(grid, filt_horizontal)
    model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b)
    set!(model, b=random_stratified_b)
    eₐˢ = Field(SubFilterAvailablePotentialEnergy(model, filt_horizontal))
    @test minimum(interior(eₐˢ)) ≥ -1e-12

    set!(model, b=1e-2)   # constant buoyancy: eₐ ≡ 0 and eₐˡ is roundoff-sized
    eₐˢ_const = Field(SubFilterAvailablePotentialEnergy(model, filt_horizontal))
    @test maximum(abs, interior(eₐˢ_const)) < 1e-12
    return nothing
end

# The Gaussian convenience methods must reproduce the explicit filter-factory call with matching kwargs.
function test_subfilter_ape_convenience(model)
    σ = 0.12
    filt = ψ -> GaussianFilter(ψ; dims=(1, 2, 3), σ, boundary=:shrink) # :shrink is the convenience default
    @test interior(Field(SubFilterAvailablePotentialEnergy(model; σ))) ≈
          interior(Field(SubFilterAvailablePotentialEnergy(model, filt)))
    return nothing
end

function test_subfilter_ape_dissipation_convenience(model)
    σ = 0.12
    filt = ψ -> GaussianFilter(ψ; dims=(1, 2, 3), σ, boundary=:shrink)
    @test interior(Field(SubFilterAvailablePotentialEnergyDissipationRate(model; σ))) ≈
          interior(Field(SubFilterAvailablePotentialEnergyDissipationRate(model, filt)))
    return nothing
end

# A fixed profile (plain arrays snapshotted off a column) must agree with borrowing the live column,
# as long as the flow has not moved between the snapshot and the compute.
function test_subfilter_ape_fixed_profile(model, filt)
    col = reference_height(model, method=VerticalSort())
    compute!(col)
    b✶ = Array(vec(interior(reference_buoyancy(col))))
    z✶ = Array(vec(interior(col)))

    fixed    = Field(SubFilterAvailablePotentialEnergy(model, filt; method=ProfileLookup(b✶, z✶)))
    borrowed = Field(SubFilterAvailablePotentialEnergy(model, filt; method=ProfileLookup(col)))
    @test interior(fixed) ≈ interior(borrowed)
    return nothing
end

# εₐˢ display/type checks, on a model with a closure so the diffusive fluxes exist.
function test_subfilter_ape_dissipation_basics(model, filt)
    εₐˢ = SubFilterAvailablePotentialEnergyDissipationRate(model, filt)
    @test location(εₐˢ) == (Center, Center, Center)
    @test εₐˢ isa SubFilterAvailablePotentialEnergyDissipationRate
    @test occursin("SubFilterAvailablePotentialEnergyDissipationRate", sprint(show, εₐˢ))
    @test occursin("computes:", sprint(show, MIME("text/plain"), εₐˢ))
    @test all(isfinite, interior(Field(εₐˢ)))
    return nothing
end

# Both diagnostics hold materialized filtered `Field`s and re-sorted reference states; recomputing at a
# new time — as an `OutputWriter` does each output — must reflect the updated flow all the way through
# that chain (filter, column sort, lookups), not stay frozen at construction. This mutates the model,
# so these are called last for their model.
function test_subfilter_ape_recomputes(model, filt)
    ef = Field(SubFilterAvailablePotentialEnergy(model, filt))
    compute_at!(ef, 0.0)
    snapshot = Array(interior(ef))

    set!(model, b=(x, y, z) -> 1e-2 * z + 2e-3 * randn())
    compute_at!(ef, 1.0)

    fresh = Field(SubFilterAvailablePotentialEnergy(model, filt))
    compute_at!(fresh, 2.0)

    @test !(Array(interior(ef)) ≈ snapshot)   # tracked the change in the flow
    @test interior(ef) ≈ interior(fresh)      # equals an eₐˢ built fresh on the new state
    return nothing
end

function test_subfilter_ape_dissipation_recomputes(model, filt)
    εf = Field(SubFilterAvailablePotentialEnergyDissipationRate(model, filt))
    compute_at!(εf, 0.0)
    snapshot = Array(interior(εf))

    set!(model, b=(x, y, z) -> 1e-2 * z + 2e-3 * randn())
    compute_at!(εf, 1.0)

    fresh = Field(SubFilterAvailablePotentialEnergyDissipationRate(model, filt))
    compute_at!(fresh, 2.0)

    @test !(Array(interior(εf)) ≈ snapshot)   # tracked the change in the flow
    @test interior(εf) ≈ interior(fresh)      # equals an εₐˢ built fresh on the new state
    return nothing
end

# The sub-filter split needs one shared profile, so any method that is not a `ProfileLookup` is
# rejected; the dissipation additionally needs the buoyancy to be a diffused tracer and a closure that
# supplies a flux, exactly like `AvailablePotentialEnergyDissipationRate`.
function test_subfilter_ape_errors(grid, filt)
    model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b, closure=ScalarDiffusivity(κ=1e-4))
    @test_throws ArgumentError SubFilterAvailablePotentialEnergy(model, filt; method=HeavisideIntegral())
    @test_throws ArgumentError SubFilterAvailablePotentialEnergyDissipationRate(model, filt; method=HeavisideIntegral())

    model_no_buoyancy = NonhydrostaticModel(grid)
    @test_throws ArgumentError SubFilterAvailablePotentialEnergy(model_no_buoyancy, filt)

    model_no_closure = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b)
    @test_throws ArgumentError SubFilterAvailablePotentialEnergyDissipationRate(model_no_closure, filt)

    model_seawater = NonhydrostaticModel(grid; buoyancy=SeawaterBuoyancy(), tracers=(:T, :S),
                                         closure=ScalarDiffusivity(κ=1e-4))
    @test_throws ArgumentError SubFilterAvailablePotentialEnergyDissipationRate(model_seawater, filt)
    return nothing
end

# The module re-exports the filtered-flow APE and its dissipation (the "ˡ" halves of both splits) from
# `FilteredAvailablePotentialEnergyEquation`, aliases its own dissipation rate as `DissipationRate`,
# and re-exports the reference-profile machinery so it can be used on its own.
function test_subfilter_ape_module_reexports()
    @test SubFilterAvailablePotentialEnergyEquation.FilteredAvailablePotentialEnergy === FilteredAvailablePotentialEnergy
    @test SubFilterAvailablePotentialEnergyEquation.FilteredAvailablePotentialEnergyDissipationRate === FilteredAvailablePotentialEnergyDissipationRate
    @test SubFilterAvailablePotentialEnergyEquation.AvailablePotentialEnergyCrossScaleFlux === AvailablePotentialEnergyCrossScaleFlux
    @test :FilteredAvailablePotentialEnergy in names(SubFilterAvailablePotentialEnergyEquation)
    @test :FilteredAvailablePotentialEnergyDissipationRate in names(SubFilterAvailablePotentialEnergyEquation)
    @test :AvailablePotentialEnergyCrossScaleFlux in names(SubFilterAvailablePotentialEnergyEquation)
    @test SubFilterAvailablePotentialEnergyEquation.DissipationRate === SubFilterAvailablePotentialEnergyDissipationRate
    @test SubFilterAvailablePotentialEnergyEquation.ProfileLookup === ProfileLookup
    @test SubFilterAvailablePotentialEnergyEquation.VerticalSort === VerticalSort
    @test SubFilterAvailablePotentialEnergyEquation.reference_height === reference_height
    @test SubFilterAvailablePotentialEnergyEquation.reference_buoyancy === reference_buoyancy
    return nothing
end
#---

@testset "Sub-filter available potential energy equation" begin
    @info "  Testing sub-filter available potential energy diagnostics"
    grid = RectilinearGrid(arch, size=(8, 8, 8), extent=(1, 1, 1), topology=(Periodic, Periodic, Bounded))
    filt = ψ -> GaussianFilter(ψ; dims=(1, 2, 3), σ=0.1, boundary=:edge)
    filt_horizontal = ψ -> GaussianFilter(ψ; dims=(1, 2), σ=0.1)

    model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b, closure=ScalarDiffusivity(κ=1e-4))
    set!(model, b=random_stratified_b)

    @info "    Sub-filter APE eₐˢ = filter(eₐ) - eₐˡ matches manual"
    test_subfilter_ape_matches_manual(model, filt)

    @info "    Identity filter makes eₐˢ and εₐˢ vanish identically"
    test_subfilter_ape_identity_filter_vanishes(model)

    @info "    Horizontally uniform stratification vanishes (eₐˢ)"
    test_subfilter_ape_uniform_stratification_vanishes(grid, filt_horizontal)

    @info "    Horizontal filter keeps eₐˢ ≥ 0 (Jensen); constant buoyancy vanishes"
    test_subfilter_ape_signs(grid, filt_horizontal)

    @info "    Gaussian convenience methods"
    test_subfilter_ape_convenience(model)
    test_subfilter_ape_dissipation_convenience(model)

    @info "    Fixed profile (arrays) agrees with borrowing the live column"
    test_subfilter_ape_fixed_profile(model, filt)

    @info "    Sub-filter APE dissipation εₐˢ = filter(εₐ) - εₐˡ basics"
    test_subfilter_ape_dissipation_basics(model, filt)

    @info "    eₐˢ and εₐˢ recompute as the flow evolves"
    test_subfilter_ape_recomputes(model, filt)              # mutates model; keep after the other `model` tests
    test_subfilter_ape_dissipation_recomputes(model, filt)

    @info "    Validation errors (method, buoyancy, closure)"
    test_subfilter_ape_errors(grid, filt)

    @info "    Module re-exports and aliases"
    test_subfilter_ape_module_reexports()
end
