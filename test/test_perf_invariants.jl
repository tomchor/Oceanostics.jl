using Test
using Statistics: median

using Oceananigans
using Oceananigans: fill_halo_regions!
using Oceananigans.Fields: Field, location

using Oceanostics
using Oceanostics: BoxFilter, GaussianFilter
using Oceanostics.TracerEquation
using Oceanostics.UMomentumEquation
using Oceanostics.KineticEnergyEquation
using Oceanostics.TurbulentKineticEnergyEquation
using Oceanostics.TracerVarianceEquation
using Oceanostics.PotentialEnergyEquation
using Oceanostics.FlowDiagnostics

# These tests defend "the repo doesn't get accidentally slower" without
# encoding hardware-specific wall-time numbers. Two complementary checks:
#
#   • Allocation + type-stability invariants (option 1). Per-cell evaluation
#     of any `KernelFunctionOperation` we expose must be type-stable and
#     allocate zero bytes. Both are hardware-independent: the typical
#     perf-regression in this codebase (an accidental `Any`, a closure that
#     captures a non-isbits, a kernel that boxes its accumulator) shows up
#     immediately as allocations or as a non-concrete inferred type. We
#     check this on representative KFOs from every src/ module.
#
#   • Self-relative ratio invariants (option 2). For separable filters we
#     assert that the staged `compute!` path is no slower than the fused
#     path on the same runner (well, no more than a generous factor), and
#     that the wide-stencil 3D case where staging is supposed to dominate
#     actually dominates. Same-runner ratios cancel out hardware noise.

#+++ Shared setup
# Small grid so the suite finishes quickly. We need enough cells away from
# the boundary that every kernel can read its full stencil at index (3,3,3)
# without hitting halo logic. A halo of 3 lets us probe the densest 7-point
# stencils we have. The grid is triply Periodic so no boundary policy
# branch fires.
const N        = 8
const PROBE    = (3, 3, 3)
const PROBE_I  = PROBE[1]
const PROBE_J  = PROBE[2]
const PROBE_K  = PROBE[3]

# Build one model that covers the union of fields most KFOs in the package
# need: velocities, a tracer `b` used as the buoyancy field, an FPlane.
function build_probe_model()
    grid = RectilinearGrid(CPU(), size=(N, N, N), extent=(1, 1, 1),
                           halo=(3, 3, 3),
                           topology=(Periodic, Periodic, Periodic))
    closure = ScalarDiffusivity(ν=1e-3, κ=1e-3)
    model = NonhydrostaticModel(grid; closure,
                                buoyancy=BuoyancyForce(BuoyancyTracer()),
                                coriolis=FPlane(f=1e-4),
                                tracers=:b)
    # Fill with something non-trivial so kernels that do `s/n` divisions
    # don't hit pathological 0/0 cases.
    set!(model, u=(x,y,z) -> sin(2π*x),
                v=(x,y,z) -> cos(2π*y),
                w=(x,y,z) -> sin(2π*z),
                b=(x,y,z) -> 1e-3*(x + y + z))
    return model
end
#---

#+++ Helpers
# Touch the KFO once so type-inference and compilation happen *outside*
# the `@allocated` block, then assert the per-cell evaluation is allocation
# free and type-stable. Wrapped in a function so the macro context itself
# can't introduce stray allocations.
@inline _probe(kfo, i, j, k) = @inbounds kfo[i, j, k]

function expect_zero_alloc(label, kfo; i=PROBE_I, j=PROBE_J, k=PROBE_K)
    _probe(kfo, i, j, k)                 # warm up: compile, infer
    allocs = @allocated _probe(kfo, i, j, k)
    @test allocs == 0
    return nothing
end

# `Test.@inferred` asserts the return type Julia infers for the call is
# concrete. If a kernel becomes type-unstable (e.g. a captured field gets
# boxed) inference falls back to `Any`/`Union` and the test fails.
function expect_type_stable(label, kfo; i=PROBE_I, j=PROBE_J, k=PROBE_K)
    @test @inferred(_probe(kfo, i, j, k)) isa Number
    return nothing
end

# Convenience: both checks for one KFO.
function test_kfo_invariants(label, kfo)
    @testset "$label" begin
        expect_type_stable(label, kfo)
        expect_zero_alloc(label, kfo)
    end
end

# Best-of-N wall time for a `compute!` on a Field — used only for the
# self-relative ratio tests, never compared against an absolute number.
function time_compute(field; samples)
    compute!(field)            # warm up: compile + first-call costs out of band
    compute!(field)
    times = Float64[]
    for _ in 1:samples
        t = @elapsed compute!(field)
        push!(times, t)
    end
    return minimum(times)
end
#---

@testset "Performance invariants" begin

    model = build_probe_model()
    grid  = model.grid

    #+++ Allocation + type-stability across modules
    @testset "Filters" begin
        c = model.tracers.b
        for dims in ((1,), (1,2), (1,2,3))
            test_kfo_invariants("BoxFilter dims=$dims",      BoxFilter(c; dims=dims, N=3))
            test_kfo_invariants("GaussianFilter dims=$dims", GaussianFilter(c; dims=dims, σ=2/N))
        end

        # Stretched (variably spaced) directions take the StretchedGaussianFilterKernel,
        # which reads node coordinates/spacings and evaluates `exp` per offset instead
        # of looking up a precomputed weight. That extra machinery (including the
        # location-tuple splat) must still be allocation-free and type-stable per cell —
        # an accidental boxing of the location tuple or a non-concrete spacing lookup
        # would show up here immediately. The grid is stretched in z (the canonical
        # case) and uniform in the periodic x and y.
        zfaces(k) = -1 + (k-1)/N + 0.08 * sin(2π*(k-1)/N)
        stretched_grid = RectilinearGrid(CPU(), size=(N, N, N), x=(0, 1), y=(0, 1), z=zfaces,
                                         halo=(3, 3, 3), topology=(Periodic, Periodic, Bounded))
        cs = CenterField(stretched_grid)
        set!(cs, (x, y, z) -> sin(2π*x) + z^2)
        # Pass an explicit (small) N: the alloc/type-stability invariant is
        # independent of stencil width, and a fixed N keeps the fused 3D probe's
        # compile time bounded regardless of the grid's minimum spacing.
        for dims in ((3,), (1, 3), (1, 2, 3))
            test_kfo_invariants("GaussianFilter stretched dims=$dims", GaussianFilter(cs; dims=dims, σ=2/N, N=5))
        end
    end

    @testset "TracerEquation" begin
        test_kfo_invariants("TracerAdvection",         TracerEquation.TracerAdvection(model, :b))
        test_kfo_invariants("TracerDiffusion",         TracerEquation.TracerDiffusion(model, :b))
    end

    @testset "UMomentumEquation" begin
        test_kfo_invariants("UAdvection",              UMomentumEquation.UAdvection(model))
        test_kfo_invariants("UCoriolisAcceleration",   UMomentumEquation.UCoriolisAcceleration(model))
    end

    @testset "KineticEnergyEquation" begin
        test_kfo_invariants("KineticEnergy",                  KineticEnergyEquation.KineticEnergy(model))
        test_kfo_invariants("KineticEnergyDissipationRate",   KineticEnergyEquation.KineticEnergyDissipationRate(model))
    end

    @testset "TurbulentKineticEnergyEquation" begin
        test_kfo_invariants("TurbulentKineticEnergy",  TurbulentKineticEnergyEquation.TurbulentKineticEnergy(model))
    end

    @testset "TracerVarianceEquation" begin
        test_kfo_invariants("TracerVarianceTendency",  TracerVarianceEquation.TracerVarianceTendency(model, :b))
    end

    @testset "PotentialEnergyEquation" begin
        test_kfo_invariants("PotentialEnergy",         PotentialEnergyEquation.PotentialEnergy(model))
    end

    @testset "FlowDiagnostics" begin
        test_kfo_invariants("RossbyNumber",            RossbyNumber(model))
        test_kfo_invariants("ErtelPotentialVorticity", ErtelPotentialVorticity(model))
        test_kfo_invariants("StrainRateTensorModulus", StrainRateTensorModulus(model))

        # Off-diagonal strain components live at edge locations (ffc/fcf/cff) and exercise the
        # new per-component kernels; the diagonals reuse Oceananigans' ∂ᵢᶜᶜᶜ operators.
        Sij = StrainRateTensor(model)
        test_kfo_invariants("StrainRateTensor.S₁₂", Sij.S₁₂)
        test_kfo_invariants("StrainRateTensor.S₁₃", Sij.S₁₃)
        test_kfo_invariants("StrainRateTensor.S₂₃", Sij.S₂₃)

        # The vorticity tensor is antisymmetric, so all its components are off-diagonal edge kernels
        # (ffc/fcf/cff); there are no diagonal components to check.
        Ωij = VorticityTensor(model)
        test_kfo_invariants("VorticityTensor.Ω₁₂", Ωij.Ω₁₂)
        test_kfo_invariants("VorticityTensor.Ω₁₃", Ωij.Ω₁₃)
        test_kfo_invariants("VorticityTensor.Ω₂₃", Ωij.Ω₂₃)
    end
    #---

    #+++ Self-relative ratio invariants
    # Build the same multi-direction filter two ways:
    #   • staged: `Field(filter)` invokes the new `compute!` override that
    #     runs `d` 1D passes through intermediate fields.
    #   • fused:  `Field(1.0 * filter)` wraps the filter in a UnaryOperation
    #     so the override does not match — the original fused single-kernel
    #     N^d evaluation runs.
    # The staged path was added specifically because the fused path becomes
    # very expensive for ≥2D filters at wide stencils. If a future change
    # accidentally disables the override (e.g. the type alias stops
    # matching) the fused path will run instead and one of these ratios
    # will collapse.
    @testset "Filter staged vs fused ratio" begin
        # Small grid + moderately wide stencil: small enough that CI
        # wall-time stays under a few seconds, large enough that the
        # fused-vs-staged work difference (N³ vs 3·N reads per cell)
        # dominates timing noise.
        bench_grid = RectilinearGrid(CPU(), size=(16, 16, 16), extent=(1, 1, 1),
                                     halo=(5, 5, 5),
                                     topology=(Periodic, Periodic, Periodic))
        c = CenterField(bench_grid)
        set!(c, (x, y, z) -> sin(2π*x) * cos(2π*y))
        fill_halo_regions!(c)

        # Same grid size but stretched in z, to track the variably spaced
        # GaussianFilter: its 1D passes do more per-cell work (node/spacing reads
        # plus `exp`) than the uniform path, but the staged evaluation must still
        # beat the fused `N³` path by the same wide margin at a wide 3D stencil.
        stretched_bench_grid = RectilinearGrid(CPU(), size=(16, 16, 16),
                                               x=(0, 1), y=(0, 1),
                                               z=k -> -1 + (k-1)/16 + 0.08*sin(2π*(k-1)/16),
                                               halo=(5, 5, 5),
                                               topology=(Periodic, Periodic, Bounded))
        cs = CenterField(stretched_bench_grid)
        set!(cs, (x, y, z) -> sin(2π*x) * cos(2π*y))
        fill_halo_regions!(cs)

        # Each `(filter_name, build)` pair returns a fresh KFO for given
        # `(dims, width)`.
        configs = (("BoxFilter",
                    (dims, w) -> BoxFilter(c; dims=dims, N=2*w+1)),
                   ("GaussianFilter",
                    (dims, w) -> GaussianFilter(c; dims=dims, σ=(w/2)/16, N=2*w+1)),
                   ("GaussianFilter (stretched z)",
                    (dims, w) -> GaussianFilter(cs; dims=dims, σ=(w/2)/16, N=2*w+1)))

        # Sanity: at a wide stencil width, the fused 3D path should be
        # *substantially* slower than the staged path. The threshold (2×)
        # is well below the 10–100× we measure in practice, so it has lots
        # of headroom against CI noise but still fails loudly if staging
        # silently regresses to the fused path.
        for (name, build) in configs
            kfo = build((1, 2, 3), 4)              # N_sten = 9, 3D
            t_staged = time_compute(Field(kfo); samples=5)
            t_fused  = time_compute(Field(1.0 * kfo); samples=5)
            @test t_staged < t_fused / 2
        end

        # And for small-N 2D filters the staged path may be slightly slower
        # due to extra launches + an intermediate allocation, but it must
        # not be more than ~5× slower — that would indicate the staged
        # implementation grew significant overhead.
        for (name, build) in configs
            kfo = build((1, 2), 1)                 # N_sten = 3, 2D
            t_staged = time_compute(Field(kfo); samples=5)
            t_fused  = time_compute(Field(1.0 * kfo); samples=5)
            @test t_staged < 5 * t_fused
        end
    end
    #---
end
