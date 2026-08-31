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
# entries). A constant buoyancy is the degenerate case: no available energy at any scale. The bound is
# not universal, and `test_subfilter_ape_resting_fluid` below is the case that breaks it.
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

# The Jensen argument above bounds an average taken at *fixed z*: eₐ(·, z) is convex in b, so averaging
# buoyancies that all sit at one height cannot lower it. A filter with vertical extent averages (b, z)
# jointly, and what it would need is joint convexity. The Hessian of eₐ is [[1/N²(z✶), -1], [-1, N²(z)]],
# so its determinant is N²(z)/N²(z✶) - 1 and eₐ is jointly convex only where the stratification at the
# parcel's own height is at least as strong as at its reference height. Outside that region a symmetric
# stencil no longer gives filter(eₐ) ≥ eₐ(b̄, z), and eₐˢ is free to go negative.
#
# The sharpest case is a fluid at rest in its own reference state, b = b✶(z). Every parcel already sits
# at its reference height, so eₐ ≡ 0 everywhere and any split of it has to give zero for both parts.
#
# It does not. Writing z + r for the heights the filter reaches, b̄(z) = filter(b✶(z + r)) ≠ b✶(z)
# wherever b✶ is curved (to leading order b̄ - b✶ ≈ ½ b✶'' ⟨r²⟩), so the filtered field is no longer a
# reference state and carries a spurious eₐˡ > 0. With filter(eₐ) = 0 the subfilter part is its mirror
# image, eₐˢ = -eₐˡ < 0: two reservoirs that are equal, opposite, and both fictitious.
#
# Unlike the random-field tests above this one is deterministic and exact. eₐ vanishes to the bit, so a
# negative eₐˢ cannot be read as roundoff or as the nearest-class fallback for buoyancies off the
# profile; it is the whole of what the split reports.
resting_tanh_b(x, y, z) = tanh((z - 1) / 0.25)   # stable and curved, flattening towards both walls
resting_linear_b(x, y, z) = 0.5 * z              # stable and straight: no curvature for the filter to find

# eₐ, eₐˡ and eₐˢ against one shared profile, the way `SubFilterAvailablePotentialEnergy` builds them,
# brought back to the host so the checks below can pick out a single level without tripping the GPU's
# scalar indexing guard.
function resting_energies(grid, setter, filt)
    model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b)
    set!(model, b=setter)
    lookup = ProfileLookup(reference_height(model, method=VerticalSort()))
    eₐ  = AvailablePotentialEnergy(model, reference_height(model.tracers.b; method=lookup))
    eₐˡ = FilteredAvailablePotentialEnergy(model, reference_height(Field(filt(model.tracers.b)); method=lookup))
    eₐˢ = SubFilterAvailablePotentialEnergy(model, filt; method=lookup)
    return map(op -> Array(interior(Field(op))), (eₐ, eₐˡ, eₐˢ))
end

function test_subfilter_ape_resting_fluid(grid, filt_vertical)
    eₐ, eₐˡ, eₐˢ = resting_energies(grid, resting_tanh_b, filt_vertical)

    @test maximum(abs, eₐ) < 1e-14                  # the premise: a resting fluid has no available energy
    @test maximum(eₐˡ) > 1e-4                       # yet the split reports a large-scale reservoir
    @test minimum(eₐˢ) < -1e-4                      # and pays for it with a negative subfilter one
    @test maximum(abs, eₐˢ .+ eₐˡ) < 1e-14          # mirror images, since filter(eₐ) = 0 leaves eₐˢ = -eₐˡ
    @test sum(eₐˢ) < 0                              # and it does not cancel in the integral (uniform cells)

    # the size of the effect is set by the curvature of b✶, so it pinches out at the inflection point
    k_inflection = argmin(abs.(vec(Array(znodes(grid, Center()))) .- 1))
    @test maximum(eₐˡ[:, :, k_inflection]) < 1e-3 * maximum(eₐˡ)
    return nothing
end

# Vertical extent and curvature are both necessary, and removing either restores the bound. A filter
# with no vertical component reaches only along z = const, where b✶(z) is a single value, so b̄ = b✶(z)
# and both reservoirs vanish identically. A straight profile is returned unchanged by any symmetric
# stencil, so it vanishes too. In the language above, a uniform N² sits exactly on the joint-convexity
# boundary: the Hessian is singular there, eₐ = bᵣ²/2N² is a degenerate quadratic in (b, z), and the
# inequality holds with equality rather than being violated, which is the exact zero checked below.
#
# The exception is within 2σ of a wall, where the stencil is truncated and renormalized. That tilts it,
# leaving ⟨r⟩ ≠ 0, so it no longer averages to the target height and a boundary-sized remainder
# survives. That 2σ is the reach `infer_width` gives the Gaussian, so the rows to skip below follow
# from σ and the grid spacing.
function test_subfilter_ape_resting_fluid_ingredients(grid, filt_vertical, filt_horizontal, σ)
    _, eₐˡ_h, eₐˢ_h = resting_energies(grid, resting_tanh_b, filt_horizontal)
    @test maximum(abs, eₐˡ_h) < 1e-14
    @test maximum(abs, eₐˢ_h) < 1e-14

    _, eₐˡ_lin, eₐˢ_lin = resting_energies(grid, resting_linear_b, filt_vertical)
    n_wall = ceil(Int, 2σ / minimum(zspacings(grid)))
    inside = (n_wall + 1):(size(eₐˡ_lin, 3) - n_wall)
    @test maximum(abs, eₐˡ_lin[:, :, inside]) < 1e-14
    @test maximum(abs, eₐˢ_lin[:, :, inside]) < 1e-14
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
    identity_filter = ψ -> GaussianFilter(ψ; dims=(1, 2, 3), σ=1e-4, N=3)
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
# function of z alone — the convention that makes the two halves an exact decomposition.
function test_subfilter_ape_ke_conversion_unfiltered_reference(model, filt, filt_horizontal)
    lookup = ProfileLookup(reference_height(model, method=VerticalSort()))
    b   = model.tracers.b
    b✶ᶻ = reference_buoyancy_at_height(model.grid, lookup.profile)

    for (filter, coincide) in ((filt, false), (filt_horizontal, true))
        b_rˡ        = Field(Field(filter(b)) - b✶ᶻ)   # filtered buoyancy, unfiltered reference
        filtered_bᵣ = Field(filter(Field(b - b✶ᶻ)))   # filters the reference too
        @test (interior(b_rˡ) ≈ interior(filtered_bᵣ)) == coincide
    end
    return nothing
end

# The Gaussian convenience method must reproduce the explicit filter-factory call with matching kwargs.
function test_subfilter_ape_ke_conversion_convenience(model)
    σ = 0.12
    filt = ψ -> GaussianFilter(ψ; dims=(1, 2, 3), σ, boundary=:shrink) # :shrink is the convenience default
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
    filt = ψ -> GaussianFilter(ψ; dims=(1, 2, 3), σ=0.1, boundary=:edge)
    filt_horizontal = ψ -> GaussianFilter(ψ; dims=(1, 2), σ=0.1)

    model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b, closure=ScalarDiffusivity(κ=1e-4))
    set!(model, b=random_stratified_b)

    @info "    Subfilter APE eₐˢ = filter(eₐ) - eₐˡ matches manual"
    test_subfilter_ape_matches_manual(model, filt)

    @info "    Identity filter makes eₐˢ and εₐˢ vanish identically"
    test_subfilter_ape_identity_filter_vanishes(model)

    @info "    Horizontally uniform stratification vanishes (eₐˢ)"
    test_subfilter_ape_uniform_stratification_vanishes(grid, filt_horizontal)

    @info "    Horizontal filter keeps eₐˢ ≥ 0 (Jensen); constant buoyancy vanishes"
    test_subfilter_ape_signs(grid, filt_horizontal)

    # A curved profile has to be curved *on the grid*, and the filter has to be wide enough that the
    # displacement it creates clears the reference profile's own class spacing, so these two carry their
    # own tall grid and their own filter rather than the shared 8×8×8 box.
    @info "    A fluid at rest: a vertical filter splits a zero eₐ into ±eₐ (Jensen does not apply)"
    resting_grid = RectilinearGrid(arch, size=(4, 4, 64), x=(0, 1), y=(0, 1), z=(0, 2),
                                   topology=(Periodic, Periodic, Bounded))
    σ_resting = 0.2
    filt_resting = ψ -> GaussianFilter(ψ; dims=(3,), σ=σ_resting, boundary=:edge)
    test_subfilter_ape_resting_fluid(resting_grid, filt_resting)
    test_subfilter_ape_resting_fluid_ingredients(resting_grid, filt_resting, filt_horizontal, σ_resting)

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

    @info "    Subfilter APE to KE conversion τˡ(w, bᵣ)"
    # The shared model is buoyancy-only, and the conversion is carried by the vertical velocity, so
    # without this every assertion below would hold trivially on a field of zeros.
    set!(model, u=(x, y, z) -> 1e-2 * randn(), w=(x, y, z) -> 1e-2 * randn())
    test_subfilter_ape_ke_conversion_decomposition(model, filt)
    test_subfilter_ape_ke_conversion_identity_filter_vanishes(model)
    test_subfilter_ape_ke_conversion_unfiltered_reference(model, filt, filt_horizontal)
    test_subfilter_ape_ke_conversion_convenience(model)
    test_subfilter_ape_ke_conversion_vanishes_without_motion(grid, filt)
    test_subfilter_ape_ke_conversion_errors(grid, filt)
    test_subfilter_ape_ke_conversion_recomputes(model, filt)   # mutates model; keep last of the `model` tests

    @info "    Module re-exports and aliases"
    test_subfilter_ape_module_reexports()
end
