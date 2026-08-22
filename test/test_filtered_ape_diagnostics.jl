using Test
using CUDA: has_cuda_gpu
using Oceananigans
using Oceananigans.Fields: location, compute_at!

using Oceanostics
using Oceanostics: FilteredAvailablePotentialEnergy, FilteredAvailablePotentialEnergyDissipationRate
using Oceanostics: AvailablePotentialEnergyCrossScaleFlux, subfilter_covariance
using Oceanostics: AvailablePotentialEnergy, AvailablePotentialEnergyDissipationRate, DisplacementPotential,
                   reference_height, reference_buoyancy, VerticalSort, ProfileLookup, HeavisideIntegral,
                   GaussianFilter

arch = has_cuda_gpu() ? GPU() : CPU()

# A random stably stratified buoyancy, shared by most tests below.
random_stratified_b(x, y, z) = 1e-2 * z + 1e-3 * randn()

# The shared profile the filtered buoyancy is looked up in: a column sorted from the model's own
# buoyancy, borrowed through `ProfileLookup` so the tests can build the same `z✶ˡ` by hand.
shared_lookup(model) = ProfileLookup(reference_height(model, method=VerticalSort()))

#+++ Test functions
# eₐˡ = eₐ(b̄, z): the local APE kernel evaluated on the reference height of the filtered buoyancy,
# looked up in the shared profile. Built by hand from `AvailablePotentialEnergy` on that `z✶ˡ`.
function test_filtered_ape_matches_manual(model, filt)
    lookup = shared_lookup(model)
    z✶ˡ = reference_height(Field(filt(model.tracers.b)); method=lookup)
    eₐˡ_manual = AvailablePotentialEnergy(model, z✶ˡ)

    eₐˡ = FilteredAvailablePotentialEnergy(model, filt; method=lookup)
    @test location(eₐˡ) == (Center, Center, Center)
    @test interior(Field(eₐˡ)) ≈ interior(Field(eₐˡ_manual))

    # it is a single KernelFunctionOperation with its own type/display
    @test eₐˡ isa FilteredAvailablePotentialEnergy
    @test occursin("FilteredAvailablePotentialEnergy", sprint(show, eₐˡ))
    @test occursin("computes:", sprint(show, MIME("text/plain"), eₐˡ))

    # the low-level form takes the reference height you built, and needs no filter of its own
    @test interior(Field(FilteredAvailablePotentialEnergy(model, z✶ˡ))) == interior(Field(eₐˡ))
    return nothing
end

# A filter that is numerically the identity (σ ≪ Δx truncated at N=3, so the off-center Gaussian
# weights underflow against the center one) makes b̄ = b bit for bit, hence z✶ˡ = z✶, so both filtered
# diagnostics must reproduce the full-field ones exactly when measured against the same profile. This
# checks the filtered kernels against the full-field ones with no reimplementation: `filtered_ape_ccc`
# reduces to `local_ape_ccc`, and `filtered_ape_dissipation_rate_ccc` reproduces
# `ape_dissipation_rate_ccc` when handed the unfiltered fluxes.
function test_filtered_ape_identity_filter_reproduces_full(model)
    identity_filter = ψ -> GaussianFilter(ψ; dims=(1, 2, 3), σ=1e-4, N=3)
    lookup = shared_lookup(model)
    z✶ = reference_height(model.tracers.b; method=lookup)

    eₐ  = Field(AvailablePotentialEnergy(model, z✶))
    eₐˡ = Field(FilteredAvailablePotentialEnergy(model, identity_filter; method=lookup))
    @test interior(eₐˡ) == interior(eₐ)

    εₐ  = Field(AvailablePotentialEnergyDissipationRate(model, z✶))
    εₐˡ = Field(FilteredAvailablePotentialEnergyDissipationRate(model, identity_filter; method=lookup))
    @test interior(εₐˡ) == interior(εₐ)
    return nothing
end

# A horizontally uniform, stable stratification filtered horizontally: b̄ = b up to roundoff, and the
# filtered buoyancy is its own sorted state, so eₐˡ ≈ 0. εₐˡ ≈ 0 is the sharper of the two: b̄ misses
# the profile's entries by roundoff, and `ProfileLookup` has to place such a near-match at its level's
# run mid-height, exactly as it places an exact match, for Υˡ to vanish cell by cell. Sending it to the
# nearest *slot* instead would put it at the top or the bottom of the run depending on the sign of the
# roundoff, and ∂zΥˡ·q̄₃ would turn that ±half-cell comb into a spurious dissipation of order κN².
function test_filtered_ape_uniform_stratification_vanishes(grid, filt_horizontal)
    model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b, closure=ScalarDiffusivity(κ=1e-4))
    set!(model, b=(x, y, z) -> 1e-2 * z)
    @test maximum(abs, interior(Field(FilteredAvailablePotentialEnergy(model, filt_horizontal)))) < 1e-12
    @test maximum(abs, interior(Field(FilteredAvailablePotentialEnergyDissipationRate(model, filt_horizontal)))) < 1e-12
    return nothing
end

# eₐ ≥ 0 rests on the parcel carrying b = b✶(z✶) exactly, which a filtered buoyancy looked up in the
# *full* field's profile does not: b̄ lands between the profile's entries and is placed at the nearest
# class, so with δ = b̄ - b✶(z✶) the local APE is eₐˡ = eₐ(b✶(z✶), z) - δ(z - z✶), and the second
# term can dip below zero by at most half a class gap times the displacement — a discretization-sized
# amount, not roundoff. Looking b̄ up in a profile sorted from b̄ itself puts it on the profile, and
# there eₐˡ ≥ 0 to roundoff, which is the sharp check that the kernel's sign is right.
function test_filtered_ape_nonnegative(grid, filt_horizontal)
    model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b)
    set!(model, b=random_stratified_b)
    b̄ = Field(filt_horizontal(model.tracers.b))

    # against the full field's profile: bounded by the profile's resolution
    col = reference_height(model, method=VerticalSort())
    compute!(col)
    b✶  = Array(vec(interior(reference_buoyancy(col))))
    z✶ˡ = reference_height(b̄; method=ProfileLookup(col))
    Υˡ  = Field(DisplacementPotential(model, z✶ˡ))
    eₐˡ = Field(FilteredAvailablePotentialEnergy(model, z✶ˡ))
    @test minimum(interior(eₐˡ)) ≥ -0.5 * maximum(diff(b✶)) * maximum(abs, interior(Υˡ))

    # against its own profile: on the profile, so non-negative to roundoff
    eₐˡ_own = Field(FilteredAvailablePotentialEnergy(model, reference_height(b̄; method=ProfileLookup())))
    @test minimum(interior(eₐˡ_own)) ≥ -sqrt(eps(eltype(grid))) * maximum(abs, interior(eₐˡ_own))
    return nothing
end

# The Gaussian convenience methods must reproduce the explicit filter-factory call with matching kwargs.
function test_filtered_ape_convenience(model)
    σ = 0.12
    filt = ψ -> GaussianFilter(ψ; dims=(1, 2, 3), σ, boundary=:shrink) # :shrink is the convenience default
    @test interior(Field(FilteredAvailablePotentialEnergy(model; σ))) ≈
          interior(Field(FilteredAvailablePotentialEnergy(model, filt)))
    @test interior(Field(FilteredAvailablePotentialEnergyDissipationRate(model; σ))) ≈
          interior(Field(FilteredAvailablePotentialEnergyDissipationRate(model, filt)))
    return nothing
end

# A fixed profile (plain arrays snapshotted off a column) must agree with borrowing the live column,
# as long as the flow has not moved between the snapshot and the compute.
function test_filtered_ape_fixed_profile(model, filt)
    col = reference_height(model, method=VerticalSort())
    compute!(col)
    b✶ = Array(vec(interior(reference_buoyancy(col))))
    z✶ = Array(vec(interior(col)))

    fixed    = Field(FilteredAvailablePotentialEnergy(model, filt; method=ProfileLookup(b✶, z✶)))
    borrowed = Field(FilteredAvailablePotentialEnergy(model, filt; method=ProfileLookup(col)))
    @test interior(fixed) ≈ interior(borrowed)
    return nothing
end

# εₐˡ display/type checks, and the low-level `(model, filter, z✶ˡ; upsilon)` form sharing a Υˡ.
function test_filtered_ape_dissipation_basics(model, filt)
    εₐˡ = FilteredAvailablePotentialEnergyDissipationRate(model, filt)
    @test location(εₐˡ) == (Center, Center, Center)
    @test εₐˡ isa FilteredAvailablePotentialEnergyDissipationRate
    @test occursin("FilteredAvailablePotentialEnergyDissipationRate", sprint(show, εₐˡ))
    @test occursin("computes:", sprint(show, MIME("text/plain"), εₐˡ))
    @test all(isfinite, interior(Field(εₐˡ)))

    lookup = shared_lookup(model)
    z✶ˡ = reference_height(Field(filt(model.tracers.b)); method=lookup)
    Υˡ  = Field(DisplacementPotential(model, z✶ˡ))
    @test interior(Field(FilteredAvailablePotentialEnergyDissipationRate(model, filt, z✶ˡ; upsilon=Υˡ))) ≈
          interior(Field(FilteredAvailablePotentialEnergyDissipationRate(model, filt; method=lookup)))
    return nothing
end

# Both diagnostics hold materialized filtered `Field`s and re-sorted reference states; recomputing at a
# new time — as an `OutputWriter` does each output — must reflect the updated flow all the way through
# that chain (filter, column sort, lookup), not stay frozen at construction. This mutates the model, so
# these are called last for their model.
function test_filtered_ape_recomputes(model, filt)
    ef = Field(FilteredAvailablePotentialEnergy(model, filt))
    compute_at!(ef, 0.0)
    snapshot = Array(interior(ef))

    set!(model, b=(x, y, z) -> 1e-2 * z + 2e-3 * randn())
    compute_at!(ef, 1.0)

    fresh = Field(FilteredAvailablePotentialEnergy(model, filt))
    compute_at!(fresh, 2.0)

    @test !(Array(interior(ef)) ≈ snapshot)   # tracked the change in the flow
    @test interior(ef) ≈ interior(fresh)      # equals an eₐˡ built fresh on the new state
    return nothing
end

function test_filtered_ape_dissipation_recomputes(model, filt)
    εf = Field(FilteredAvailablePotentialEnergyDissipationRate(model, filt))
    compute_at!(εf, 0.0)
    snapshot = Array(interior(εf))

    set!(model, b=(x, y, z) -> 1e-2 * z + 2e-3 * randn())
    compute_at!(εf, 1.0)

    fresh = Field(FilteredAvailablePotentialEnergyDissipationRate(model, filt))
    compute_at!(fresh, 2.0)

    @test !(Array(interior(εf)) ≈ snapshot)   # tracked the change in the flow
    @test interior(εf) ≈ interior(fresh)      # equals an εₐˡ built fresh on the new state
    return nothing
end

# The filtered buoyancy has to be looked up in a shared profile, so any method that is not a
# `ProfileLookup` is rejected; the dissipation additionally needs the buoyancy to be a diffused tracer
# and a closure that supplies a flux, exactly like `AvailablePotentialEnergyDissipationRate`.
function test_filtered_ape_errors(grid, filt)
    model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b, closure=ScalarDiffusivity(κ=1e-4))
    @test_throws ArgumentError FilteredAvailablePotentialEnergy(model, filt; method=HeavisideIntegral())
    @test_throws ArgumentError FilteredAvailablePotentialEnergyDissipationRate(model, filt; method=HeavisideIntegral())

    model_no_buoyancy = NonhydrostaticModel(grid)
    @test_throws ArgumentError FilteredAvailablePotentialEnergy(model_no_buoyancy, filt)

    model_no_closure = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b)
    @test_throws ArgumentError FilteredAvailablePotentialEnergyDissipationRate(model_no_closure, filt)

    model_seawater = NonhydrostaticModel(grid; buoyancy=SeawaterBuoyancy(), tracers=(:T, :S),
                                         closure=ScalarDiffusivity(κ=1e-4))
    @test_throws ArgumentError FilteredAvailablePotentialEnergyDissipationRate(model_seawater, filt)
    return nothing
end

# The module aliases its dissipation rate as `DissipationRate` and re-exports the reference-profile
# machinery so it can be used on its own.
function test_filtered_ape_module_reexports()
    @test FilteredAvailablePotentialEnergyEquation.DissipationRate === FilteredAvailablePotentialEnergyDissipationRate
    @test FilteredAvailablePotentialEnergyEquation.ProfileLookup === ProfileLookup
    @test FilteredAvailablePotentialEnergyEquation.VerticalSort === VerticalSort
    @test FilteredAvailablePotentialEnergyEquation.reference_height === reference_height
    @test FilteredAvailablePotentialEnergyEquation.reference_buoyancy === reference_buoyancy
    return nothing
end
#---


"""
Πₐ = -τᵢ ∂ᵢΥˡ rebuilt from raw filter calls (`filter(b uᵢ) - b̄ ūᵢ`, every factor collocated at cell
centers) contracted with the gradient of the filtered-state displacement potential. The diagnostic
itself builds τᵢ through `subfilter_covariance`, so the manual construction here deliberately does
not: the comparison pins the covariance, the sign, the directions summed over, and the co-location of
every factor independently — which no approximate check could separate.
"""
function test_ape_cross_scale_flux_matches_manual(model, filt)

    lookup = shared_lookup(model)
    Πₐ = AvailablePotentialEnergyCrossScaleFlux(model, filt; method = lookup)
    @test Πₐ isa AvailablePotentialEnergyCrossScaleFlux

    b = model.tracers.b
    u, v, w = model.velocities
    z✶ˡ = reference_height(Field(filt(b)); method = lookup)
    Υˡ = Field(DisplacementPotential(model, z✶ˡ))

    ccc = (Center, Center, Center)
    b̄ = Field(filt(b))
    raw_τ(uᵈ_ccc) = Field(filt(Field(b * uᵈ_ccc))) - b̄ * Field(filt(uᵈ_ccc))
    manual = -sum(Field(@at ccc raw_τ(Field(@at ccc uᵈ)) * ∂ᵈ(Υˡ))
                  for (uᵈ, ∂ᵈ) in zip((u, v, w), (∂x, ∂y, ∂z)))

    @test interior(Field(Πₐ)) ≈ interior(Field(manual))

    # the low-level form on a prebuilt z✶ˡ (one lookup, and through `upsilon` one Υˡ, shared with the
    # other filtered-state diagnostics) matches the high-level form
    @test interior(Field(AvailablePotentialEnergyCrossScaleFlux(model, filt, z✶ˡ; upsilon=Υˡ))) ≈
          interior(Field(Πₐ))

    return nothing
end

"""
The flux is a *transfer*: with no motion there is no sub-filter buoyancy flux to carry APE across the
filter scale, so τᵢ and hence Πₐ vanish identically however sharp the buoyancy is. This is the check
that a stray sign or a leftover term would break, since Υˡ itself is nowhere near zero here.
"""
function test_ape_cross_scale_flux_vanishes_without_motion(grid, filt)

    model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b)
    set!(model, b = random_stratified_b)   # velocities left at zero

    Πₐ = Field(AvailablePotentialEnergyCrossScaleFlux(model, filt))
    Υˡ = Field(DisplacementPotential(model, reference_height(Field(filt(model.tracers.b)); method = shared_lookup(model))))

    @test maximum(abs, interior(Υˡ)) > 0   # otherwise the assertion below would hold trivially
    @test maximum(abs, interior(Πₐ)) == 0

    return nothing
end

"""
`dims` selects which directions are summed, exactly as it does for the kinetic-energy flux: the 2D
`x`–`z` flux has to be the full one less its `y` term.
"""
function test_ape_cross_scale_flux_dims(model, filt)

    lookup = shared_lookup(model)
    full = Field(AvailablePotentialEnergyCrossScaleFlux(model, filt; method = lookup))
    xz   = Field(AvailablePotentialEnergyCrossScaleFlux(model, filt; dims = (1, 3), method = lookup))

    b = model.tracers.b
    Υˡ = Field(DisplacementPotential(model, reference_height(Field(filt(b)); method = lookup)))
    y_term = Field(@at (Center, Center, Center) -subfilter_covariance(b, model.velocities[2], filt) * ∂y(Υˡ))

    @test interior(xz) ≈ interior(full) .- interior(y_term)

    return nothing
end

"""
Like the other filtered-state diagnostics, the flux measures the filtered buoyancy against a profile it
did not produce, so anything but a `ProfileLookup` has to be refused rather than silently sorting `b̄`
into its own reference state.
"""
function test_ape_cross_scale_flux_errors(grid, filt)

    model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b)
    set!(model, b = random_stratified_b)

    @test_throws "ProfileLookup" AvailablePotentialEnergyCrossScaleFlux(model, filt; method = HeavisideIntegral())
    @test_throws ArgumentError AvailablePotentialEnergyCrossScaleFlux(model, filt; dims = (1, 4))

    return nothing
end

@testset "Filtered available potential energy equation" begin
    @info "  Testing filtered available potential energy diagnostics"
    grid = RectilinearGrid(arch, size=(8, 8, 8), extent=(1, 1, 1), topology=(Periodic, Periodic, Bounded))
    filt = ψ -> GaussianFilter(ψ; dims=(1, 2, 3), σ=0.1, boundary=:edge)
    filt_horizontal = ψ -> GaussianFilter(ψ; dims=(1, 2), σ=0.1)

    model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b, closure=ScalarDiffusivity(κ=1e-4))
    set!(model, b=random_stratified_b)

    @info "    Filtered APE eₐˡ = eₐ(b̄, z) matches manual"
    test_filtered_ape_matches_manual(model, filt)

    @info "    Identity filter reproduces the full-field eₐ and εₐ exactly"
    test_filtered_ape_identity_filter_reproduces_full(model)

    @info "    Horizontally uniform stratification vanishes"
    test_filtered_ape_uniform_stratification_vanishes(grid, filt_horizontal)

    @info "    eₐˡ ≥ 0 (to the profile's resolution against the full field's profile; to roundoff against its own)"
    test_filtered_ape_nonnegative(grid, filt_horizontal)

    @info "    Gaussian convenience methods"
    test_filtered_ape_convenience(model)

    @info "    Fixed profile (arrays) agrees with borrowing the live column"
    test_filtered_ape_fixed_profile(model, filt)

    @info "    Filtered APE dissipation εₐˡ = -q̄ᵢ∂ᵢΥˡ basics and shared Υˡ"
    test_filtered_ape_dissipation_basics(model, filt)

    @info "    eₐˡ and εₐˡ recompute as the flow evolves"
    test_filtered_ape_recomputes(model, filt)              # mutates model; keep after the other `model` tests
    test_filtered_ape_dissipation_recomputes(model, filt)

    @info "    Validation errors (method, buoyancy, closure)"
    test_filtered_ape_errors(grid, filt)

    @info "    Module re-exports and aliases"
    test_filtered_ape_module_reexports()

    @info "  Testing the cross-scale available potential energy flux Πₐ"
    test_ape_cross_scale_flux_matches_manual(model, filt)
    test_ape_cross_scale_flux_dims(model, filt)
    test_ape_cross_scale_flux_vanishes_without_motion(grid, filt)
    test_ape_cross_scale_flux_errors(grid, filt)
end
