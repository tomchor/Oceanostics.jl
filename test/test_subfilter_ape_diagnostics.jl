using Test
using CUDA: has_cuda_gpu
using Oceananigans
using Oceananigans.Fields: location, compute_at!

using Oceanostics
using Oceanostics: SubFilterAvailablePotentialEnergy, SubFilterAvailablePotentialEnergyDissipationRate
using Oceanostics: FilteredAvailablePotentialEnergy, FilteredAvailablePotentialEnergyDissipationRate,
                   AvailablePotentialEnergyCrossScaleFlux, FilteredAvailablePotentialToKineticEnergyConversion
using Oceanostics: SubFilterAvailablePotentialToKineticEnergyConversion, AvailablePotentialToKineticEnergyConversion,
                   reference_buoyancy_at_height
using Oceanostics: AvailablePotentialEnergy, AvailablePotentialEnergyDissipationRate,
                   reference_height, reference_buoyancy, VerticalSort, ProfileLookup, HeavisideIntegral,
                   GaussianFilter, BoxFilter

arch = has_cuda_gpu() ? GPU() : CPU()

# A random stably stratified buoyancy, shared by most tests below.
random_stratified_b(x, y, z) = 1e-2 * z + 1e-3 * randn()

# A deterministic stand-in for it, for the checks whose verdict depends on the particular field rather
# than on an identity: the same stratification with a smooth three-dimensional disturbance, so that a
# failure is reproducible rather than a property of one random draw. It uses no RNG, so `set!` can
# evaluate it inside a GPU kernel.
wavy_stratified_b(x, y, z) = 1e-2 * z + 1e-3 * sinpi(2x) * cospi(2y) * sinpi(4z)

# Filters that are neither a `BoxFilter` nor a `GaussianFilter`: three-point means along z and along x,
# written as plain kernels, which the diagnostics can only recognize by trying them on a probe field.
@inline z_mean(i, j, k, grid, ψ) = @inbounds (ψ[i, j, k-1] + ψ[i, j, k] + ψ[i, j, k+1]) / 3
@inline x_mean(i, j, k, grid, ψ) = @inbounds (ψ[i-1, j, k] + ψ[i, j, k] + ψ[i+1, j, k]) / 3
z_mean_filter(ψ) = KernelFunctionOperation{location(ψ)...}(z_mean, ψ.grid, ψ)
x_mean_filter(ψ) = KernelFunctionOperation{location(ψ)...}(x_mean, ψ.grid, ψ)

# Filters whose direction depends on the field they are given: along z for fields on z faces, or for lazy
# operations, and along x otherwise. Both act horizontally on the buoyancy, so only a diagnostic that
# also filters such a field can catch them.
z_faces_z_mean_filter(ψ) = location(ψ)[3] === Face ? z_mean_filter(ψ) : x_mean_filter(ψ)
lazy_z_mean_filter(ψ) = ψ isa Field ? x_mean_filter(ψ) : z_mean_filter(ψ)

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
    identity_filter = ψ -> GaussianFilter(ψ; dims=(1, 2), σ=1e-4, N=3)
    @test all(interior(Field(SubFilterAvailablePotentialEnergy(model, identity_filter))) .== 0)
    @test all(interior(Field(SubFilterAvailablePotentialEnergyDissipationRate(model, identity_filter))) .== 0)
    return nothing
end

# A horizontally uniform, stable stratification is its own reference state, so it carries no available
# energy at any scale and eₐˢ has to vanish. Filtered horizontally, b̄ = b up to roundoff and both eₐ and
# eₐˡ vanish cell by cell (z✶ = z); eₐ is blind to where in a tied run the lookup lands, so the roundoff
# in b̄ cannot leak an O(Δz) error into this test.
function test_subfilter_ape_uniform_stratification_vanishes(grid, filt)
    model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b)
    set!(model, b=(x, y, z) -> 1e-2 * z)
    eₐˢ = Field(SubFilterAvailablePotentialEnergy(model, filt))
    @test maximum(abs, interior(eₐˢ)) < 1e-12
    return nothing
end

# eₐ is convex in buoyancy, so Jensen's inequality makes eₐˢ ≥ 0 pointwise for a filter that averages at
# fixed z (exactly, up to roundoff, since the full field's buoyancies are the profile's own entries),
# which is the only kind the diagnostics accept. A filter with vertical extent would average (b, z)
# jointly, and there the bound needs joint convexity of eₐ, which fails wherever the stratification at
# the parcel's own height is weaker than at its reference height (the resting-fluid tests below show
# it). A constant buoyancy is the degenerate case: no available energy at any scale.
function test_subfilter_ape_signs(grid, filt)
    model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b)
    set!(model, b=wavy_stratified_b)
    eₐˢ = Field(SubFilterAvailablePotentialEnergy(model, filt))
    @test minimum(interior(eₐˢ)) ≥ -1e-12

    set!(model, b=1e-2)   # constant buoyancy: eₐ ≡ 0 and eₐˡ is roundoff-sized
    eₐˢ_const = Field(SubFilterAvailablePotentialEnergy(model, filt))
    @test maximum(abs, interior(eₐˢ_const)) < 1e-12
    return nothing
end

# The sharpest test of that bound is a fluid at rest in its own reference state, b = b✶(z). Every parcel
# already sits at its reference height, so eₐ ≡ 0 everywhere and any split of it has to give zero for
# both parts. A horizontal filter does that. A filter with vertical extent averages the stratification
# itself, so b̄ leaves the profile wherever the profile is curved, and within reach of a wall even where
# it is straight: the filtered state then picks up eₐˡ > 0, which the subfilter state pays for with
# eₐˢ < 0. That failure is why every filtered and subfilter APE diagnostic rejects such a filter, and
# these tests keep it pinned, for a curved and for a straight resting profile under a horizontal, a
# vertical and a 3D filter.
resting_tanh_b(x, y, z) = tanh((z - 1) / 0.25)   # stable and curved, flattening towards both walls
resting_linear_b(x, y, z) = 0.5 * z              # stable and straight: no curvature for the filter to find

# eₐ, eₐˡ and eₐˢ against one shared profile, brought back to the host, where the checks below are plain
# array reductions. The diagnostics build the split for the horizontal filter. They refuse the others,
# so for those the split is built by hand the way `SubFilterAvailablePotentialEnergy` builds it, with the
# full-field `AvailablePotentialEnergy`, whose kernel `FilteredAvailablePotentialEnergy` forwards to.
function resting_energies(grid, setter, filt; vertical_filter)
    model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b)
    set!(model, b=setter)
    lookup = ProfileLookup(reference_height(model, method=VerticalSort()))
    z✶ˡ = reference_height(Field(filt(model.tracers.b)); method=lookup)
    eₐ  = AvailablePotentialEnergy(model, reference_height(model.tracers.b; method=lookup))
    if vertical_filter
        eₐˡ = AvailablePotentialEnergy(model, z✶ˡ)
        eₐˢ = Field(filt(Field(eₐ))) - eₐˡ
    else
        eₐˡ = FilteredAvailablePotentialEnergy(model, z✶ˡ)
        eₐˢ = SubFilterAvailablePotentialEnergy(model, filt; method=lookup)
    end
    return map(op -> Array(interior(Field(op))), (eₐ, eₐˡ, eₐˢ))
end

function test_subfilter_ape_resting_fluid(grid, setter, filt; vertical_filter=false)
    eₐ, eₐˡ, eₐˢ = resting_energies(grid, setter, filt; vertical_filter)
    @test maximum(abs, eₐ) < 1e-14          # the premise: a resting fluid has no available energy
    @test maximum(abs, eₐˢ .+ eₐˡ) < 1e-14  # and the split is exact, eₐˢ = -eₐˡ, whatever their signs
    if vertical_filter                      # the failure that keeps such filters out of the diagnostics:
        @test minimum(eₐˢ) < -1e-12         # a negative subfilter reservoir ...
        @test sum(eₐˢ) < 0                  # ... in the volume integral too (uniform cells)
    else
        @test minimum(eₐˢ) ≥ -1e-12         # the bound: no negative subfilter reservoir ...
        @test sum(eₐˢ) ≥ 0                  # ... and none in the volume integral either (uniform cells)
        @test maximum(abs, eₐˡ) < 1e-12     # equivalently, no large-scale reservoir to pay for it
    end
    return nothing
end

# The Gaussian convenience methods must reproduce the explicit filter-factory call with matching kwargs.
function test_subfilter_ape_convenience(model)
    σ = 0.12
    filt = ψ -> GaussianFilter(ψ; dims=(1, 2), σ, boundary=:shrink) # the convenience defaults
    @test interior(Field(SubFilterAvailablePotentialEnergy(model; σ))) ≈
          interior(Field(SubFilterAvailablePotentialEnergy(model, filt)))
    return nothing
end

function test_subfilter_ape_dissipation_convenience(model)
    σ = 0.12
    filt = ψ -> GaussianFilter(ψ; dims=(1, 2), σ, boundary=:shrink)
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

# The subfilter split needs one shared profile, so any method that is not a `ProfileLookup` is
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

# A filter that averages along z makes eₐˢ negative even for a fluid at rest (the resting-fluid tests
# above), so every subfilter diagnostic refuses one, whether it comes as a filter object, as a closure,
# as a hand-written kernel or through the convenience methods' `dims`.
function test_subfilter_ape_rejects_vertical_filters(grid, vertical_filters)
    model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b, closure=ScalarDiffusivity(κ=1e-4))
    set!(model, b=random_stratified_b)
    for diagnostic in (SubFilterAvailablePotentialEnergy, SubFilterAvailablePotentialEnergyDissipationRate,
                       SubFilterAvailablePotentialToKineticEnergyConversion)
        for vertical in vertical_filters
            @test_throws "acts in the horizontal" diagnostic(model, vertical)
        end
        for dims in ((1, 2, 3), (3,))
            @test_throws "acts in the horizontal" diagnostic(model; σ=0.1, dims)
        end
    end
    return nothing
end

# The filter is checked on every field a diagnostic applies it to, so a filter that acts along z only on
# some kinds of field is refused exactly by the diagnostics that filter such a field: εₐˢ filters the
# fluxes, which are lazy operations and one of which lives on z faces, and τˡ(w, bᵣ) filters `w`, a
# `Field` on z faces. eₐˢ filters only stored, centred fields, on which both filters act along x.
function test_subfilter_ape_checks_every_filtered_field(grid)
    model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b, closure=ScalarDiffusivity(κ=1e-4))
    set!(model, b=random_stratified_b)

    # (diagnostic, refuses `z_faces_z_mean_filter`, refuses `lazy_z_mean_filter`)
    cases = ((SubFilterAvailablePotentialEnergy,                    false, false),
             (SubFilterAvailablePotentialEnergyDissipationRate,     true,  true),
             (SubFilterAvailablePotentialToKineticEnergyConversion, true,  false))

    for (diagnostic, refuses_z_faces, refuses_lazy) in cases,
        (filt, refuses) in ((z_faces_z_mean_filter, refuses_z_faces), (lazy_z_mean_filter, refuses_lazy))
        if refuses
            @test_throws "acts in the horizontal" diagnostic(model, filt)
        else
            @test diagnostic(model, filt) isa diagnostic
        end
    end
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
    # τˡ(w, bᵣ) is a source of the subfilter KE budget too, so that module re-exports it from here
    @test SubFilterKineticEnergyEquation.SubFilterAvailablePotentialToKineticEnergyConversion === SubFilterAvailablePotentialToKineticEnergyConversion
    @test :SubFilterAvailablePotentialToKineticEnergyConversion in names(SubFilterKineticEnergyEquation)
    @test SubFilterAvailablePotentialEnergyEquation.DissipationRate === SubFilterAvailablePotentialEnergyDissipationRate
    @test SubFilterAvailablePotentialEnergyEquation.ProfileLookup === ProfileLookup
    @test SubFilterAvailablePotentialEnergyEquation.VerticalSort === VerticalSort
    @test SubFilterAvailablePotentialEnergyEquation.reference_height === reference_height
    @test SubFilterAvailablePotentialEnergyEquation.reference_buoyancy === reference_buoyancy
    return nothing
end
#---

# τˡ(w, bᵣ) is defined as filter(wbᵣ) - w̄b_rˡ, so the two halves have to add back up to the filtered
# full-field conversion. That is the decomposition claim, and it pins the shared b✶(z) and the shared
# discretization: were either half built against a different reference or collocated differently, the
# sum would drift from filter(wbᵣ).
function test_subfilter_ape_ke_conversion_decomposition(model, filt)
    lookup = ProfileLookup(reference_height(model, method=VerticalSort()))
    z✶ = reference_height(model.tracers.b; method=lookup)

    τ  = Field(SubFilterAvailablePotentialToKineticEnergyConversion(model, filt; method=lookup))
    wl = Field(FilteredAvailablePotentialToKineticEnergyConversion(model, filt; method=lookup))
    filtered_full = Field(filt(Field(AvailablePotentialToKineticEnergyConversion(model, z✶))))

    @test location(τ) == (Center, Center, Center)
    @test maximum(abs, interior(τ)) > 0   # otherwise the sum below would hold trivially
    @test interior(filtered_full) ≈ interior(wl) .+ interior(τ)

    @test occursin("SubFilterAvailablePotentialToKineticEnergyConversion", sprint(show, τ.operand))
    @test occursin("computes:", sprint(show, MIME("text/plain"), τ.operand))
    return nothing
end

# An identity-scale filter makes both halves the full-field conversion, so their difference vanishes to
# the bit — the check that the two are the same expression evaluated on different fields, not two
# separately built ones.
function test_subfilter_ape_ke_conversion_identity_filter_vanishes(model)
    identity_filter = ψ -> GaussianFilter(ψ; dims=(1, 2), σ=1e-4, N=3)
    @test all(interior(Field(SubFilterAvailablePotentialToKineticEnergyConversion(model, identity_filter))) .== 0)
    return nothing
end

# The conversion is carried by the vertical velocity, so with no motion it vanishes identically however
# sharp the buoyancy.
function test_subfilter_ape_ke_conversion_vanishes_without_motion(grid, filt)
    model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b)
    set!(model, b=random_stratified_b)   # velocities left at zero
    @test maximum(abs, interior(Field(SubFilterAvailablePotentialToKineticEnergyConversion(model, filt)))) == 0
    return nothing
end

# Both halves measure the buoyancy against the *unfiltered* reference profile, so the filtered half uses
# b_rˡ = b̄ - b✶(z) rather than filter(bᵣ) = filter(b - b✶(z)), which filters the reference too. The two
# differ once the filter acts in the vertical and coincide for a purely horizontal one, b✶ being a
# function of z alone — the convention that makes the two halves an exact decomposition. The diagnostic
# itself takes only the horizontal kind, so the vertical filter here is applied to the fields directly.
function test_subfilter_ape_ke_conversion_unfiltered_reference(model, filt_vertical, filt_horizontal)
    lookup = ProfileLookup(reference_height(model, method=VerticalSort()))
    b   = model.tracers.b
    b✶ᶻ = reference_buoyancy_at_height(model.grid, lookup.profile)

    for (filter, coincide) in ((filt_vertical, false), (filt_horizontal, true))
        b_rˡ        = Field(Field(filter(b)) - b✶ᶻ)   # filtered buoyancy, unfiltered reference
        filtered_bᵣ = Field(filter(Field(b - b✶ᶻ)))   # filters the reference too
        @test (interior(b_rˡ) ≈ interior(filtered_bᵣ)) == coincide
    end
    return nothing
end

# The Gaussian convenience method must reproduce the explicit filter-factory call with matching kwargs.
function test_subfilter_ape_ke_conversion_convenience(model)
    σ = 0.12
    filt = ψ -> GaussianFilter(ψ; dims=(1, 2), σ, boundary=:shrink) # the convenience defaults
    @test interior(Field(SubFilterAvailablePotentialToKineticEnergyConversion(model; σ))) ≈
          interior(Field(SubFilterAvailablePotentialToKineticEnergyConversion(model, filt)))
    return nothing
end

# Like its siblings, it measures the filtered buoyancy against a profile it did not produce, so anything
# but a `ProfileLookup` has to be refused.
function test_subfilter_ape_ke_conversion_errors(grid, filt)
    model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b)
    set!(model, b=random_stratified_b)
    @test_throws "ProfileLookup" SubFilterAvailablePotentialToKineticEnergyConversion(model, filt; method=HeavisideIntegral())
    @test_throws ArgumentError SubFilterAvailablePotentialToKineticEnergyConversion(NonhydrostaticModel(grid), filt)
    return nothing
end

# It holds filtered `Field`s and a profile re-sorted on every `compute!`, so it has to track the flow.
# This mutates the model, so it runs last.
function test_subfilter_ape_ke_conversion_recomputes(model, filt)
    cf = Field(SubFilterAvailablePotentialToKineticEnergyConversion(model, filt))
    compute_at!(cf, 0.0)
    snapshot = Array(interior(cf))

    set!(model, b=(x, y, z) -> 1e-2 * z + 2e-3 * randn(), w=(x, y, z) -> 2e-2 * randn())
    compute_at!(cf, 1.0)

    fresh = Field(SubFilterAvailablePotentialToKineticEnergyConversion(model, filt))
    compute_at!(fresh, 2.0)

    @test !(Array(interior(cf)) ≈ snapshot)   # tracked the change in the flow
    @test interior(cf) ≈ interior(fresh)      # equals a conversion built fresh on the new state
    return nothing
end

@testset "Subfilter available potential energy equation" begin
    @info "  Testing subfilter available potential energy diagnostics"
    grid = RectilinearGrid(arch, size=(8, 8, 8), extent=(1, 1, 1), topology=(Periodic, Periodic, Bounded))
    filt = ψ -> GaussianFilter(ψ; dims=(1, 2), σ=0.1)   # the diagnostics take only filters that act in the horizontal

    # Filters with a vertical component, in the forms a caller can hand one over: a filter object, a
    # closure over the one-step form, a `BoxFilter` that cuts along y and z, and a hand-written kernel.
    filt_vertical = ψ -> GaussianFilter(ψ; dims=(3,), σ=0.1, boundary=:edge)
    vertical_filters = (GaussianFilter(; dims=(1, 2, 3), σ=0.1, boundary=:edge), filt_vertical, BoxFilter(; dims=(2, 3), N=3),
                        z_mean_filter)

    model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b, closure=ScalarDiffusivity(κ=1e-4))
    set!(model, b=random_stratified_b)

    @info "    Subfilter APE eₐˢ = filter(eₐ) - eₐˡ matches manual"
    test_subfilter_ape_matches_manual(model, filt)

    @info "    Identity filter makes eₐˢ and εₐˢ vanish identically"
    test_subfilter_ape_identity_filter_vanishes(model)

    @info "    Horizontally uniform stratification vanishes (eₐˢ)"
    test_subfilter_ape_uniform_stratification_vanishes(grid, filt)

    @info "    eₐˢ ≥ 0 (Jensen); constant buoyancy vanishes"
    test_subfilter_ape_signs(grid, filt)

    # The resting-fluid checks run over all three ways a filter can cut, each in its own labelled testset
    # so the summary names the case. The two cuts that reach in z carry `vertical_filter=true` (the third
    # element of each entry): the diagnostics refuse them, so their split is built by hand, and they
    # assert the failure rather than the bound. The comments on the test functions carry the mechanism.
    #
    # A curved profile has to be curved *on the grid*, and the filter has to be wide enough that the
    # displacement it creates clears the reference profile's own class spacing, so these carry their own
    # tall grid and their own filters rather than the shared 8×8×8 box.
    @info "    Fluid at rest: no APE at any scale with a horizontal filter, eₐˡ > 0 and eₐˢ < 0 with one along z"
    resting_grid = RectilinearGrid(arch, size=(4, 4, 64), x=(0, 1), y=(0, 1), z=(0, 2),
                                   topology=(Periodic, Periodic, Bounded))
    resting_cuts = (("horizontal", ψ -> GaussianFilter(ψ; dims=(1, 2), σ=0.2), false),
                    ("vertical",   ψ -> GaussianFilter(ψ; dims=(3,), σ=0.2, boundary=:edge), true),
                    ("3D",         ψ -> GaussianFilter(ψ; dims=(1, 2, 3), σ=0.2, boundary=:edge), true))
    resting_profiles = ("curved" => resting_tanh_b, "straight" => resting_linear_b)
    for (profile, setter) in resting_profiles, (cut, f, vertical_filter) in resting_cuts
        @testset "resting $profile profile, $cut filter" begin
            test_subfilter_ape_resting_fluid(resting_grid, setter, f; vertical_filter)
        end
    end

    @info "    Gaussian convenience methods"
    test_subfilter_ape_convenience(model)
    test_subfilter_ape_dissipation_convenience(model)

    @info "    Fixed profile (arrays) agrees with borrowing the live column"
    test_subfilter_ape_fixed_profile(model, filt)

    @info "    Subfilter APE dissipation εₐˢ = filter(εₐ) - εₐˡ basics"
    test_subfilter_ape_dissipation_basics(model, filt)

    @info "    eₐˢ and εₐˢ recompute as the flow evolves"
    test_subfilter_ape_recomputes(model, filt)              # mutates model; keep after the other `model` tests
    test_subfilter_ape_dissipation_recomputes(model, filt)

    @info "    Validation errors (method, buoyancy, closure)"
    test_subfilter_ape_errors(grid, filt)

    @info "    Filters with a vertical component are rejected"
    test_subfilter_ape_rejects_vertical_filters(grid, vertical_filters)
    test_subfilter_ape_checks_every_filtered_field(grid)

    @info "    Subfilter APE to KE conversion τˡ(w, bᵣ)"
    # The shared model is buoyancy-only, and the conversion is carried by the vertical velocity, so
    # without this every assertion below would hold trivially on a field of zeros.
    set!(model, u=(x, y, z) -> 1e-2 * randn(), w=(x, y, z) -> 1e-2 * randn())
    test_subfilter_ape_ke_conversion_decomposition(model, filt)
    test_subfilter_ape_ke_conversion_identity_filter_vanishes(model)
    test_subfilter_ape_ke_conversion_unfiltered_reference(model, filt_vertical, filt)
    test_subfilter_ape_ke_conversion_convenience(model)
    test_subfilter_ape_ke_conversion_vanishes_without_motion(grid, filt)
    test_subfilter_ape_ke_conversion_errors(grid, filt)
    test_subfilter_ape_ke_conversion_recomputes(model, filt)   # mutates model; keep last of the `model` tests

    @info "    Module re-exports and aliases"
    test_subfilter_ape_module_reexports()
end
