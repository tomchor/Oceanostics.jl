using Test
using Oceananigans
using Oceananigans: fill_halo_regions!, location
using Oceananigans.AbstractOperations: KernelFunctionOperation

using Oceanostics
using Oceanostics: BoxFilter

# -------- Test helpers --------
#
# These helpers factor out the setup and reference computations that the
# BoxFilter testsets share, so each @testset body focuses on *what* is being
# checked rather than on the bookkeeping of building grids, setting fields,
# and accumulating explicit stencil sums.

make_grid(; Nx=8, Ny=8, Nz=8, halo=(2, 2, 2), topology=(Periodic, Periodic, Periodic)) =
    RectilinearGrid(size = (Nx, Ny, Nz),
                    x = (0, 1), y = (0, 1), z = (0, 1),
                    halo = halo,
                    topology = topology)

function center_field_from(grid, f)
    c = CenterField(grid)
    set!(c, f)
    fill_halo_regions!(c)
    return c
end

function compute_box_filter(ψ, dims, width; kwargs...)
    cf = Field(BoxFilter(ψ; dims=dims, width=width, kwargs...))
    compute!(cf)
    return cf
end

# Explicit reference for a periodic (mod1-wrapped) box average.
function reference_box_average_periodic(ic, dims, width, Ns)
    Nx, Ny, Nz = Ns
    rx = 1 in dims ? (-width:width) : (0:0)
    ry = 2 in dims ? (-width:width) : (0:0)
    rz = 3 in dims ? (-width:width) : (0:0)
    n = length(rx) * length(ry) * length(rz)
    ref = similar(ic)
    for i in 1:Nx, j in 1:Ny, k in 1:Nz
        s = zero(eltype(ic))
        for di in rx, dj in ry, dk in rz
            s += ic[mod1(i + di, Nx), mod1(j + dj, Ny), mod1(k + dk, Nz)]
        end
        ref[i, j, k] = s / n
    end
    return ref
end

# Explicit reference for a shrink-and-renormalize box average (out-of-bounds
# offsets are dropped from *both* the sum and the count).
function reference_box_average_shrink(ic, dims, width, Ns)
    Nx, Ny, Nz = Ns
    rx = 1 in dims ? (-width:width) : (0:0)
    ry = 2 in dims ? (-width:width) : (0:0)
    rz = 3 in dims ? (-width:width) : (0:0)
    ref = similar(ic)
    for i in 1:Nx, j in 1:Ny, k in 1:Nz
        s = zero(eltype(ic)); n = 0
        for di in rx, dj in ry, dk in rz
            ii = i + di; jj = j + dj; kk = k + dk
            x_ok = !(1 in dims) || (1 <= ii <= Nx)
            y_ok = !(2 in dims) || (1 <= jj <= Ny)
            z_ok = !(3 in dims) || (1 <= kk <= Nz)
            if x_ok && y_ok && z_ok
                s += ic[ii, jj, kk]; n += 1
            end
        end
        ref[i, j, k] = s / n
    end
    return ref
end

# Explicit reference for an edge-replicated box average (out-of-bounds
# offsets clamp the index to the nearest boundary cell).
function reference_box_average_edge(ic, dims, width, Ns)
    Nx, Ny, Nz = Ns
    rx = 1 in dims ? (-width:width) : (0:0)
    ry = 2 in dims ? (-width:width) : (0:0)
    rz = 3 in dims ? (-width:width) : (0:0)
    n = length(rx) * length(ry) * length(rz)
    ref = similar(ic)
    for i in 1:Nx, j in 1:Ny, k in 1:Nz
        s = zero(eltype(ic))
        for di in rx, dj in ry, dk in rz
            ii = 1 in dims ? clamp(i + di, 1, Nx) : i + di
            jj = 2 in dims ? clamp(j + dj, 1, Ny) : j + dj
            kk = 3 in dims ? clamp(k + dk, 1, Nz) : k + dk
            s += ic[ii, jj, kk]
        end
        ref[i, j, k] = s / n
    end
    return ref
end

# Explicit reference for a 1-D constant-pad box average along direction `d`
# (`left` on the low-index side, `right` on the high-index side).
function reference_box_average_constant_1d(ic, d, width, Ns, left, right)
    Nx, Ny, Nz = Ns
    N_d = Ns[d]
    ref = similar(ic)
    for i in 1:Nx, j in 1:Ny, k in 1:Nz
        s = zero(eltype(ic))
        for di in -width:width
            center = d == 1 ? i : d == 2 ? j : k
            off    = center + di
            if off < 1
                s += left
            elseif off > N_d
                s += right
            else
                idx = d == 1 ? (off, j, k) : d == 2 ? (i, off, k) : (i, j, off)
                s += ic[idx...]
            end
        end
        ref[i, j, k] = s / (2 * width + 1)
    end
    return ref
end

@testset "Filters" begin
    @testset "BoxFilter" begin
        Nx = Ny = Nz = 8
        Ns = (Nx, Ny, Nz)
        grid = make_grid(; Nx=Nx, Ny=Ny, Nz=Nz)

        @testset "Constructor returns KernelFunctionOperation" begin
            # Sanity check that BoxFilter returns a KernelFunctionOperation and that
            # the `BoxFilter` type alias matches the constructor's output. Downstream
            # Oceananigans machinery (Field, compute!, Output writers) dispatches on
            # KFO, so a regression here would break every user of BoxFilter.
            bf = BoxFilter(CenterField(grid); dims=(1,), width=1)
            @test bf isa KernelFunctionOperation
            @test bf isa BoxFilter
        end

        @testset "Linear field is unchanged on interior (1D, 2D, 3D)" begin
            # For a linear field, a symmetric (2w+1)-point running mean reproduces
            # the field exactly on cells whose stencil does not cross a periodic
            # boundary (where a linear function is not truly periodic). We compare
            # only interior cells of the filtered directions.
            c = center_field_from(grid, (x, y, z) -> x + 2y + 3z)

            for dims in [(1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
                cf = compute_box_filter(c, dims, 1)
                rx = 1 in dims ? (2:Nx-1) : (1:Nx)
                ry = 2 in dims ? (2:Ny-1) : (1:Ny)
                rz = 3 in dims ? (2:Nz-1) : (1:Nz)
                @test interior(cf)[rx, ry, rz] ≈ interior(c)[rx, ry, rz]
            end
        end

        @testset "Constant field is unchanged" begin
            # A constant field is the trivial fixed point of any averaging operator.
            # This immediately catches normalization mistakes (e.g. dividing by n+1
            # instead of n, or by (2w+1) instead of (2w+1)^d for multi-dim filters).
            c = center_field_from(grid, (x, y, z) -> 3.14)
            cf = compute_box_filter(c, (1, 2, 3), 2)
            @test all(interior(cf) .≈ 3.14)
        end

        @testset "Averaging matches an explicit stencil sum (periodic)" begin
            # Strongest correctness check for periodic dims: compare BoxFilter
            # output cell-by-cell against a hand-written reference using explicit
            # mod1 wrapping, on a non-trivial (trig + quadratic) field. Covers 1D
            # (width=1), 3D (width=1, 27-point box), and 2D / 3D with width=2
            # (25- and 125-point boxes). The wider-stencil cases matter because
            # they stress the inner-loop indexing over strides > 1 and the
            # (2w+1)^d normalization.
            c  = center_field_from(grid, (x, y, z) -> sin(2π * x) * cos(2π * y) + z^2)
            ic = interior(c)

            for (dims, width) in [((1,),      1),
                                  ((1, 2, 3), 1),
                                  ((1, 2),    2),
                                  ((1, 2, 3), 2)]
                cf  = compute_box_filter(c, dims, width)
                ref = reference_box_average_periodic(ic, dims, width, Ns)
                @test interior(cf) ≈ ref
            end
        end

        @testset "Exact hand-computed values on a 1D periodic grid" begin
            # Pins down actual numerical output on a tiny, trivially-verifiable
            # case: ψ = [0, 1, 2, ..., 9] on a 10-cell Periodic 1D grid, filtered
            # at width=1 and width=2. Every expected value was computed by hand
            # (see comments below). This anchors the filter's output to concrete
            # numbers — any regression in normalization, indexing offsets, or
            # periodic wrap-around would immediately surface as a value mismatch
            # rather than hiding inside a self-consistent numerical reference.
            grid_1d = RectilinearGrid(size = (10,),
                                      x = (0, 1),
                                      halo = (2,),
                                      topology = (Periodic, Flat, Flat))
            c = CenterField(grid_1d)
            interior(c)[:] .= Float64.(0:9)
            fill_halo_regions!(c)

            # width = 1, (ψ[i-1] + ψ[i] + ψ[i+1]) / 3 with periodic wrap:
            #   i=1:  (9 + 0 + 1)/3 = 10/3
            #   i=2..9: just i-1 (the arithmetic-sequence mean)
            #   i=10: (8 + 9 + 0)/3 = 17/3
            expected1 = [10/3, 1, 2, 3, 4, 5, 6, 7, 8, 17/3]
            @test interior(compute_box_filter(c, (1,), 1))[:] ≈ expected1

            # width = 2, 5-point mean with periodic wrap:
            #   i=1:  (8+9+0+1+2)/5 = 4
            #   i=2:  (9+0+1+2+3)/5 = 3
            #   i=3..8: just i-1 (the arithmetic-sequence mean)
            #   i=9:  (6+7+8+9+0)/5 = 6
            #   i=10: (7+8+9+0+1)/5 = 5
            expected2 = [4, 3, 2, 3, 4, 5, 6, 7, 6, 5]
            @test interior(compute_box_filter(c, (1,), 2))[:] ≈ expected2
        end

        @testset "Output location matches input location" begin
            # A symmetric (2w+1)-point average preserves the grid location of its
            # input, so filtering a Face-located field must yield output on the
            # same Face. A regression (e.g. hard-coding ccc) would silently move
            # the output to the wrong grid and break any downstream gradient /
            # operator.
            @test location(BoxFilter(CenterField(grid); dims=(1,),      width=1)) == (Center, Center, Center)
            @test location(BoxFilter(XFaceField(grid);  dims=(1,),      width=1)) == (Face,   Center, Center)
            @test location(BoxFilter(YFaceField(grid);  dims=(1, 2),    width=1)) == (Center, Face,   Center)
            @test location(BoxFilter(ZFaceField(grid);  dims=(1, 2, 3), width=1)) == (Center, Center, Face)
        end

        @testset "Accepts AbstractOperation as input" begin
            # BoxFilter should accept any Oceananigans AbstractOperation, not just
            # Fields, so users can chain it with algebraic expressions like `2*c`
            # without having to materialize the intermediate. The numerical check
            # confirms the filter sees the operation's values (not zeros or the
            # underlying Field unscaled).
            c  = center_field_from(grid, (x, y, z) -> x + y)
            op = 2 * c  # BinaryOperation at ccc
            @test BoxFilter(op; dims=(1, 2), width=1) isa KernelFunctionOperation

            f = compute_box_filter(op, (1, 2), 1)
            @test interior(f)[2:Nx-1, 2:Ny-1, :] ≈ 2 .* interior(c)[2:Nx-1, 2:Ny-1, :]
        end

        @testset "Accepts another KernelFunctionOperation as input" begin
            # Filters must compose: wrapping a BoxFilter inside another BoxFilter
            # (e.g. a 1D-y filter of a 1D-x filter) is a legitimate user pattern
            # and also exercises the recursive `f::Function` method of the kernel.
            c     = center_field_from(grid, (x, y, z) -> sin(2π * x))
            inner = BoxFilter(c;     dims=(1,), width=1)
            outer = BoxFilter(inner; dims=(2,), width=1)
            @test outer isa KernelFunctionOperation
            @test location(outer) == (Center, Center, Center)
        end

        @testset "Validation of dims" begin
            # `dims` must be a non-empty tuple of distinct integers from (1,2,3).
            # We reject each misuse up front with a clear ArgumentError rather
            # than letting it fall through to a cryptic indexing or dispatch
            # failure deep inside the kernel.
            c = CenterField(grid)
            @test_throws ArgumentError BoxFilter(c; dims=(),     width=1)
            @test_throws ArgumentError BoxFilter(c; dims=(0,),   width=1)
            @test_throws ArgumentError BoxFilter(c; dims=(4,),   width=1)
            @test_throws ArgumentError BoxFilter(c; dims=(1, 1), width=1)
            @test_throws ArgumentError BoxFilter(c; dims=(:x,),  width=1)
            @test_throws ArgumentError BoxFilter(c; dims=[1, 2], width=1)
        end

        @testset "Validation of width" begin
            # `width` must be a positive integer (it is the half-width in cells
            # of a (2w+1)-point stencil). Zero, negative, and non-integer values
            # must be rejected at construction time so users get an immediate,
            # clear error rather than a silent miscount or a later dispatch error.
            c = CenterField(grid)
            @test_throws ArgumentError BoxFilter(c; dims=(1,), width=0)
            @test_throws ArgumentError BoxFilter(c; dims=(1,), width=-1)
            @test_throws ArgumentError BoxFilter(c; dims=(1,), width=1.5)
        end

        @testset "Small halo is allowed for any width or topology" begin
            # Every boundary policy (periodic mod1, shrink, edge, constant-pad)
            # wraps / clamps / skips indices before reading the field, so the
            # halo never has to carry the stencil. This test verifies that
            # BoxFilter constructs successfully with width ≫ halo on a variety
            # of topologies — a regression that reintroduces a halo-size check
            # would fail here.
            tiny_periodic = make_grid(; halo=(1, 1, 1), topology=(Periodic, Periodic, Periodic))
            tiny_bounded  = make_grid(; halo=(1, 1, 1), topology=(Bounded,  Bounded,  Bounded))
            tiny_mixed    = make_grid(; halo=(1, 1, 1), topology=(Periodic, Bounded,  Bounded))

            for g in (tiny_periodic, tiny_bounded, tiny_mixed)
                c = CenterField(g)
                @test BoxFilter(c; dims=(1,),      width=5)                      isa KernelFunctionOperation
                @test BoxFilter(c; dims=(1, 2, 3), width=5, boundary=:edge)      isa KernelFunctionOperation
                @test BoxFilter(c; dims=(1, 2, 3), width=5,
                                boundary=(left=0.0, right=0.0))                  isa KernelFunctionOperation
            end
        end

        @testset ":shrink (default) on a bounded grid matches explicit reference" begin
            # Default boundary policy is :shrink: out-of-bounds offsets are
            # dropped from *both* the sum and the count, so the near-wall average
            # is over whatever interior cells the stencil actually covers.
            # Verified cell-by-cell against an explicit shrink reference, across
            # 1D, 2D, and 3D filters with width=2.
            bounded_grid = make_grid(; halo=(1, 1, 1), topology=(Bounded, Bounded, Bounded))
            c  = center_field_from(bounded_grid, (x, y, z) -> sin(2π * x) * cos(2π * y) + z^2)
            ic = interior(c)

            for dims in [(1,), (2,), (3,), (1, 2), (1, 2, 3)]
                cf  = compute_box_filter(c, dims, 2)  # default boundary=:shrink
                ref = reference_box_average_shrink(ic, dims, 2, Ns)
                @test interior(cf) ≈ ref
            end
        end

        @testset ":edge on a bounded grid matches explicit reference" begin
            # :edge replicates the boundary-cell value for out-of-bounds offsets
            # (reads ψ[1] on the low side, ψ[N] on the high side). Verified
            # against an explicit reference that uses clamp for the same effect.
            bounded_grid = make_grid(; halo=(1, 1, 1), topology=(Bounded, Bounded, Bounded))
            c  = center_field_from(bounded_grid, (x, y, z) -> sin(2π * x) * cos(2π * y) + z^2)
            ic = interior(c)

            for dims in [(1,), (2,), (3,), (1, 2, 3)]
                cf  = compute_box_filter(c, dims, 2; boundary=:edge)
                ref = reference_box_average_edge(ic, dims, 2, Ns)
                @test interior(cf) ≈ ref
            end
        end

        @testset "Constant-pad (left, right) matches explicit reference" begin
            # Constant-pad fills out-of-bounds offsets with `left` on the low
            # side and `right` on the high side — supplied via a NamedTuple so
            # the two values are visually labeled at the call site.
            bounded_grid = make_grid(; halo=(1, 1, 1), topology=(Bounded, Bounded, Bounded))
            c  = center_field_from(bounded_grid, (x, y, z) -> sin(2π * x) * cos(2π * y))
            ic = interior(c)

            left, right = -1.25, 2.5
            for (d, width) in [(1, 2), (2, 3), (3, 2)]
                dims = (d,)
                cf  = compute_box_filter(c, dims, width;
                                         boundary=(left=left, right=right))
                ref = reference_box_average_constant_1d(ic, d, width, Ns, left, right)
                @test interior(cf) ≈ ref
            end
        end

        @testset "Exact hand-computed values on a 1D bounded grid" begin
            # ψ = [0, 1, 2, ..., 9] on a 10-cell Bounded 1D grid with tiny halo,
            # filtered under each boundary policy at width=2. Every expected
            # value is hand-computed (see comments). Anchors each policy's
            # near-wall behavior to concrete numbers so a subtle change in
            # counting / padding would show up as a value mismatch.
            grid_1d = RectilinearGrid(size = (10,),
                                      x = (0, 1),
                                      halo = (1,),
                                      topology = (Bounded, Flat, Flat))
            c = CenterField(grid_1d)
            interior(c)[:] .= Float64.(0:9)
            fill_halo_regions!(c)

            # :shrink, width=2. Near-wall stencils shrink; interior cells use
            # the full 5-point mean (which equals i-1 for an arithmetic sequence).
            #   i=1:  (ψ[1]+ψ[2]+ψ[3])/3 = (0+1+2)/3 = 1
            #   i=2:  (ψ[1]+ψ[2]+ψ[3]+ψ[4])/4 = 6/4 = 1.5
            #   i=3..8: i-1
            #   i=9:  (ψ[7]+ψ[8]+ψ[9]+ψ[10])/4 = 30/4 = 7.5
            #   i=10: (ψ[8]+ψ[9]+ψ[10])/3 = 24/3 = 8
            shrink_expected = [1, 1.5, 2, 3, 4, 5, 6, 7, 7.5, 8]
            @test interior(compute_box_filter(c, (1,), 2))[:] ≈ shrink_expected

            # :edge, width=2. Out-of-bounds offsets clamp to ψ[1] or ψ[10].
            #   i=1: (ψ[1]+ψ[1]+ψ[1]+ψ[2]+ψ[3])/5 = 3/5 = 0.6
            #   i=2: (ψ[1]+ψ[1]+ψ[2]+ψ[3]+ψ[4])/5 = 6/5 = 1.2
            #   i=3..8: full-stencil mean = i-1
            #   i=9:  (ψ[7]+ψ[8]+ψ[9]+ψ[10]+ψ[10])/5 = 39/5 = 7.8
            #   i=10: (ψ[8]+ψ[9]+ψ[10]+ψ[10]+ψ[10])/5 = 42/5 = 8.4
            edge_expected = [0.6, 1.2, 2, 3, 4, 5, 6, 7, 7.8, 8.4]
            @test interior(compute_box_filter(c, (1,), 2; boundary=:edge))[:] ≈ edge_expected

            # Constant-pad (left=-1, right=-2), width=2. Near-wall cells include
            # the pads; interior cells use only real values.
            #   i=1:  (-1 + -1 + ψ[1]+ψ[2]+ψ[3])/5 = (-2 + 3)/5 = 0.2
            #   i=2:  (-1 + ψ[1]+ψ[2]+ψ[3]+ψ[4])/5 = (-1 + 6)/5 = 1
            #   i=3..8: i-1
            #   i=9:  (ψ[7]+ψ[8]+ψ[9]+ψ[10] + -2)/5 = (30 - 2)/5 = 5.6
            #   i=10: (ψ[8]+ψ[9]+ψ[10] + -2 + -2)/5 = (24 - 4)/5 = 4
            const_expected = [0.2, 1, 2, 3, 4, 5, 6, 7, 5.6, 4]
            @test interior(compute_box_filter(c, (1,), 2;
                                              boundary=(left=-1.0, right=-2.0)))[:] ≈ const_expected
        end

        @testset "Per-dim boundary tuple mixes policies across dims" begin
            # A tuple `boundary` assigns one policy to each filtered dim (in the
            # same order as `dims`). We check that dims=(1, 2) with
            # (:shrink, :edge) matches the sequential application of a 1D shrink
            # along x followed by a 1D edge along y — confirming both the tuple
            # plumbing and the fact that each dim's policy is actually used.
            bounded_grid = make_grid(; halo=(1, 1, 1), topology=(Bounded, Bounded, Bounded))
            c  = center_field_from(bounded_grid, (x, y, z) -> sin(2π * x) * cos(2π * y))
            ic = interior(c)

            cf = compute_box_filter(c, (1, 2), 2; boundary=(:shrink, :edge))
            shrink_in_x       = reference_box_average_shrink(ic,         (1,), 2, Ns)
            edge_of_shrink_xy = reference_box_average_edge(shrink_in_x,  (2,), 2, Ns)
            @test interior(cf) ≈ edge_of_shrink_xy
        end

        @testset "Periodic dims ignore the boundary spec" begin
            # On a (Periodic, Bounded, Bounded) grid, a `boundary` spec for the
            # x-direction must be silently overridden by the periodic wrap.
            # We verify this by computing the same filter twice with very
            # different x-boundary specs: the outputs must agree.
            mixed_grid = make_grid(; halo=(1, 1, 1), topology=(Periodic, Bounded, Bounded))
            c = center_field_from(mixed_grid, (x, y, z) -> sin(2π * x) + y)

            cf_default = compute_box_filter(c, (1,), 3)  # default :shrink is silently overridden to periodic
            cf_absurd  = compute_box_filter(c, (1,), 3; boundary=(left=1e6, right=-1e6))
            @test interior(cf_default) ≈ interior(cf_absurd)
        end

        @testset "Validation of boundary spec" begin
            # The `boundary` kwarg must be a known Symbol, a `(left, right)`
            # NamedTuple, or a tuple of those matching `dims` length. Invalid
            # specs are rejected at construction time — and since the validation
            # runs up front (before the periodic-override), a malformed spec
            # errors even on a fully periodic grid.
            c = CenterField(grid)
            @test_throws ArgumentError BoxFilter(c; dims=(1,),   width=1, boundary=:foo)
            @test_throws ArgumentError BoxFilter(c; dims=(1,),   width=1, boundary=(foo=1,))
            @test_throws ArgumentError BoxFilter(c; dims=(1,),   width=1, boundary=(left=1, right=2, mid=3))
            @test_throws ArgumentError BoxFilter(c; dims=(1,),   width=1, boundary=42)
            # Mismatched per-dim tuple length:
            @test_throws ArgumentError BoxFilter(c; dims=(1, 2), width=1, boundary=(:shrink,))
            @test_throws ArgumentError BoxFilter(c; dims=(1,),   width=1, boundary=(:shrink, :edge))
        end
    end
end
