using Test
using CUDA: has_cuda_gpu
using Oceananigans
using Oceananigans.Fields: location, compute_at!

using Oceanostics
using Oceanostics: FilteredAvailablePotentialEnergy, FilteredAvailablePotentialEnergyDissipationRate
using Oceanostics: AvailablePotentialEnergyCrossScaleFlux, subfilter_covariance
using Oceanostics: FilteredAvailablePotentialToKineticEnergyConversion
using Oceanostics.BackgroundPotentialEnergyEquation: reference_buoyancy_at_height
using Oceanostics: AvailablePotentialEnergy, AvailablePotentialEnergyDissipationRate, BuoyancyDisplacementPotential,
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
    Υˡ  = Field(BuoyancyDisplacementPotential(model, z✶ˡ))
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
    Υˡ  = Field(BuoyancyDisplacementPotential(model, z✶ˡ))
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
    Υˡ = Field(BuoyancyDisplacementPotential(model, z✶ˡ))

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
    Υˡ = Field(BuoyancyDisplacementPotential(model, reference_height(Field(filt(model.tracers.b)); method = shared_lookup(model))))

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
    Υˡ = Field(BuoyancyDisplacementPotential(model, reference_height(Field(filt(b)); method = lookup)))
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

"""
w̄b_rˡ = w̄(b̄ − b✶(z)) rebuilt by hand from the filtered vertical velocity, the filtered buoyancy and the
reference profile read at each cell's own height. The product is formed on the `w` face and only then
interpolated to the center, so the manual construction pins that co-location too.
"""
function test_filtered_ape_ke_conversion_matches_manual(model, filt)

    lookup = shared_lookup(model)
    b✶z = reference_buoyancy_at_height(model.grid, lookup.profile)
    w̄  = Field(filt(model.velocities.w))
    b̄  = Field(filt(model.tracers.b))
    b_rˡ_ccf = Field(@at (Center, Center, Face) Field(b̄ - b✶z))
    manual = Field(@at (Center, Center, Center) (w̄ * b_rˡ_ccf))

    conversion = FilteredAvailablePotentialToKineticEnergyConversion(model, filt; method = lookup)
    @test location(conversion) == (Center, Center, Center)
    @test conversion isa FilteredAvailablePotentialToKineticEnergyConversion
    @test occursin("FilteredAvailablePotentialToKineticEnergyConversion", sprint(show, conversion))
    @test occursin("computes:", sprint(show, MIME("text/plain"), conversion))
    @test interior(Field(conversion)) ≈ interior(manual)

    return nothing
end

"""
The reference profile is read at the parcel's own height, so a horizontally uniform stable
stratification is its own reference state: b✶(z) = b(z) cell by cell, hence b_rˡ ≡ 0 and no energy is
converted however vigorous the vertical velocity. This is the sharp check on the lookup, since a
half-slot error in b✶(z) would show up here as an O(N²Δz) anomaly rather than as roundoff.
"""
function test_filtered_ape_ke_conversion_uniform_stratification_vanishes(grid, filt)

    model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b)
    set!(model, b = (x, y, z) -> 1e-2 * z, w = (x, y, z) -> 1e-2 * randn())

    b✶z = Field(reference_buoyancy_at_height(grid, ProfileLookup(reference_height(model, method=VerticalSort())).profile))
    @test interior(b✶z) == interior(model.tracers.b)   # its own reference state, to the bit

    @test maximum(abs, interior(Field(FilteredAvailablePotentialToKineticEnergyConversion(model, filt)))) < 1e-18

    return nothing
end

"""
The conversion is a flux carried by the filtered vertical velocity, so it vanishes identically with no
motion however sharp the buoyancy — the check a stray sign or a leftover term would break, since b_rˡ
is nowhere near zero here.
"""
function test_filtered_ape_ke_conversion_vanishes_without_motion(grid, filt)

    model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b)
    set!(model, b = random_stratified_b)   # velocities left at zero

    lookup = shared_lookup(model)
    b_rˡ = Field(Field(filt(model.tracers.b)) - reference_buoyancy_at_height(grid, lookup.profile))
    @test maximum(abs, interior(b_rˡ)) > 0   # otherwise the assertion below would hold trivially
    @test maximum(abs, interior(Field(FilteredAvailablePotentialToKineticEnergyConversion(model, filt; method = lookup)))) == 0

    return nothing
end

"""
b_rˡ = b̄ − b✶(z) measures the filtered buoyancy against the *unfiltered* reference profile, which is
what differentiating eₐˡ produces; it is not filter(b_r) = b̄ − filter(b✶(z)), which filters the
reference too. The two differ once the filter acts in the vertical and coincide for a purely horizontal
one, since b✶ is a function of z alone — exactly the distinction this term is defined by.
"""
function test_filtered_ape_ke_conversion_unfiltered_reference(model, filt, filt_horizontal)

    lookup = shared_lookup(model)
    b = model.tracers.b

    for (filter, coincide) in ((filt, false), (filt_horizontal, true))
        b✶z = reference_buoyancy_at_height(model.grid, lookup.profile)
        b_rˡ         = Field(Field(filter(b)) - b✶z)     # filtered buoyancy, unfiltered reference
        filtered_b_r = Field(filter(Field(b - b✶z)))     # filters the reference too
        @test (interior(b_rˡ) ≈ interior(filtered_b_r)) == coincide
    end

    return nothing
end

# The Gaussian convenience method must reproduce the explicit filter-factory call with matching kwargs.
function test_filtered_ape_ke_conversion_convenience(model)
    σ = 0.12
    filt = ψ -> GaussianFilter(ψ; dims=(1, 2, 3), σ, boundary=:shrink) # :shrink is the convenience default
    @test interior(Field(FilteredAvailablePotentialToKineticEnergyConversion(model; σ))) ≈
          interior(Field(FilteredAvailablePotentialToKineticEnergyConversion(model, filt)))
    return nothing
end

# The conversion holds a filtered `Field` and a profile that is re-sorted on every `compute!`, so it has
# to track the flow like the other filtered-state diagnostics. This mutates the model, so it runs last.
function test_filtered_ape_ke_conversion_recomputes(model, filt)

    cf = Field(FilteredAvailablePotentialToKineticEnergyConversion(model, filt))
    compute_at!(cf, 0.0)
    snapshot = Array(interior(cf))

    set!(model, b = (x, y, z) -> 1e-2 * z + 2e-3 * randn(), w = (x, y, z) -> 2e-2 * randn())
    compute_at!(cf, 1.0)

    fresh = Field(FilteredAvailablePotentialToKineticEnergyConversion(model, filt))
    compute_at!(fresh, 2.0)

    @test !(Array(interior(cf)) ≈ snapshot)   # tracked the change in the flow
    @test interior(cf) ≈ interior(fresh)      # equals a conversion built fresh on the new state
    return nothing
end

"""
Like the other filtered-state diagnostics, the conversion measures the filtered buoyancy against a
profile it did not produce, so anything but a `ProfileLookup` has to be refused.
"""
function test_filtered_ape_ke_conversion_errors(grid, filt)

    model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b)
    set!(model, b = random_stratified_b)

    @test_throws "ProfileLookup" FilteredAvailablePotentialToKineticEnergyConversion(model, filt; method = HeavisideIntegral())
    @test_throws ArgumentError FilteredAvailablePotentialToKineticEnergyConversion(NonhydrostaticModel(grid), filt)

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

    @info "  Testing the filtered APE to filtered KE conversion w̄b_rˡ"
    test_filtered_ape_ke_conversion_matches_manual(model, filt)
    test_filtered_ape_ke_conversion_unfiltered_reference(model, filt, filt_horizontal)
    test_filtered_ape_ke_conversion_convenience(model)
    test_filtered_ape_ke_conversion_uniform_stratification_vanishes(grid, filt_horizontal)
    test_filtered_ape_ke_conversion_vanishes_without_motion(grid, filt)
    test_filtered_ape_ke_conversion_errors(grid, filt)

    @info "  Testing the cross-scale available potential energy flux Πₐ"
    test_ape_cross_scale_flux_matches_manual(model, filt)
    test_ape_cross_scale_flux_dims(model, filt)
    test_ape_cross_scale_flux_vanishes_without_motion(grid, filt)
    test_ape_cross_scale_flux_errors(grid, filt)

    @info "    w̄b_rˡ recomputes as the flow evolves"
    test_filtered_ape_ke_conversion_recomputes(model, filt)   # mutates model; keep last
end
