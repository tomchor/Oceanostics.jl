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

# Create a CenterField, fill it with `f(x, y, z)`, and sync halos so any
# subsequent boundary-adjacent stencil evaluation sees a consistent state.
function center_field_from(grid, f)
    c = CenterField(grid)
    set!(c, f)
    fill_halo_regions!(c)
    return c
end

# Build a BoxFilter over `ψ` along `dims` with the given `width`, wrap it in a
# Field, compute it, and return the resulting Field.
function compute_box_filter(ψ, dims, width)
    cf = Field(BoxFilter(ψ; dims=dims, width=width))
    compute!(cf)
    return cf
end

# Explicit reference for a (2*width + 1)^length(dims) box average over `dims`,
# with mod1-wrapping along each selected direction. `Ns = (Nx, Ny, Nz)` gives
# the wrap lengths. Used as an independent check against the compiled kernel.
function reference_box_average(ic, dims, width, Ns)
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

        @testset "Averaging matches an explicit stencil sum" begin
            # Strongest correctness check: compare BoxFilter output cell-by-cell
            # against a hand-written reference using explicit mod1 wrapping, on a
            # non-trivial (trig + quadratic) field. Covers 1D (width=1), 3D
            # (width=1, 27-point box), and 2D / 3D with width=2 (25- and 125-point
            # boxes). The wider-stencil cases matter because they stress the
            # inner-loop indexing over strides > 1 and the (2w+1)^d normalization.
            c  = center_field_from(grid, (x, y, z) -> sin(2π * x) * cos(2π * y) + z^2)
            ic = interior(c)

            for (dims, width) in [((1,),      1),
                                  ((1, 2, 3), 1),
                                  ((1, 2),    2),
                                  ((1, 2, 3), 2)]
                cf  = compute_box_filter(c, dims, width)
                ref = reference_box_average(ic, dims, width, Ns)
                @test interior(cf) ≈ ref
            end
        end

        @testset "Exact hand-computed values on a 1D periodic grid" begin
            # Pins down actual numerical output on a tiny, trivially-verifiable case:
            # ψ = [0, 1, 2, ..., 9] on a 10-cell Periodic 1D grid, filtered at
            # width=1 and width=2. Every expected value was computed by hand (see
            # comments below). This complements the algebraic-identity tests
            # (linear/constant fields, explicit stencil sums with trig inputs) by
            # anchoring the filter's output to concrete numbers — any regression
            # in normalization, indexing offsets, or periodic wrap-around would
            # immediately surface as a value mismatch rather than hiding inside a
            # self-consistent numerical reference.
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
            # input, so filtering a Face-located field must yield output on the same
            # Face. A regression (e.g. hard-coding ccc) would silently move the
            # output to the wrong grid and break any downstream gradient/operator.
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
            # We reject each misuse up front with a clear ArgumentError rather than
            # letting it fall through to a cryptic indexing or dispatch failure
            # deep inside the kernel.
            c = CenterField(grid)
            @test_throws ArgumentError BoxFilter(c; dims=(),     width=1)
            @test_throws ArgumentError BoxFilter(c; dims=(0,),   width=1)
            @test_throws ArgumentError BoxFilter(c; dims=(4,),   width=1)
            @test_throws ArgumentError BoxFilter(c; dims=(1, 1), width=1)
            @test_throws ArgumentError BoxFilter(c; dims=(:x,),  width=1)
            @test_throws ArgumentError BoxFilter(c; dims=[1, 2], width=1)
        end

        @testset "Validation of width" begin
            # `width` must be a positive integer (it is the half-width in cells of
            # a (2w+1)-point stencil). Zero, negative, and non-integer values must
            # be rejected at construction time so users get an immediate, clear
            # error rather than a silent miscount or a later dispatch error.
            c = CenterField(grid)
            @test_throws ArgumentError BoxFilter(c; dims=(1,), width=0)
            @test_throws ArgumentError BoxFilter(c; dims=(1,), width=-1)
            @test_throws ArgumentError BoxFilter(c; dims=(1,), width=1.5)
        end

        @testset "Periodic directions need no halo" begin
            # Fully periodic grid with a tiny halo; width ≫ halo is allowed because
            # every selected direction is periodic and wraps via mod1. The numerical
            # check confirms that the mod1 path actually matches the explicit
            # reference when the halo is too small to carry the stencil.
            small_halo_grid = make_grid(; Nx=Nx, Ny=Ny, Nz=Nz, halo=(1, 1, 1))
            @test BoxFilter(CenterField(small_halo_grid); dims=(1,),      width=3) isa KernelFunctionOperation
            @test BoxFilter(CenterField(small_halo_grid); dims=(1, 2, 3), width=3) isa KernelFunctionOperation

            c   = center_field_from(small_halo_grid, (x, y, z) -> sin(2π * x) + cos(2π * y))
            cf  = compute_box_filter(c, (1,), 3)
            ref = reference_box_average(interior(c), (1,), 3, Ns)
            @test interior(cf) ≈ ref
        end

        @testset "Mixed topology: halo enforced only on non-periodic dims" begin
            # (Periodic, Bounded, Bounded) with small halos in y and z. Verifies
            # that halo validation is per-direction: periodic dims skip the check
            # entirely, while each bounded dim independently constrains width.
            mixed_grid = make_grid(; Nx=Nx, Ny=Ny, Nz=Nz,
                                   halo=(1, 2, 2),
                                   topology=(Periodic, Bounded, Bounded))
            c = CenterField(mixed_grid)

            # Filtering only along x (periodic): width may exceed x-halo freely.
            @test BoxFilter(c; dims=(1,), width=5) isa KernelFunctionOperation

            # Filtering along y (bounded): width must be ≤ y-halo.
            @test BoxFilter(c; dims=(2,), width=2) isa KernelFunctionOperation
            @test_throws ArgumentError BoxFilter(c; dims=(2,), width=3)

            # Filtering along z (bounded): same.
            @test_throws ArgumentError BoxFilter(c; dims=(3,), width=3)

            # Filtering on mixed dims: the bounded dim still constrains width.
            @test_throws ArgumentError BoxFilter(c; dims=(1, 2), width=3)
            @test BoxFilter(c; dims=(1, 2, 3), width=2) isa KernelFunctionOperation
        end

        @testset "Halo validation (fully bounded)" begin
            # On a fully bounded grid with halo = (2, 2, 2), width = 3 must fail
            # for any selected dim since there is no periodic wrap to fall back on.
            bounded_grid = make_grid(; Nx=Nx, Ny=Ny, Nz=Nz,
                                     halo=(2, 2, 2),
                                     topology=(Bounded, Bounded, Bounded))
            c = CenterField(bounded_grid)
            @test_throws ArgumentError BoxFilter(c; dims=(1,),      width=3)
            @test_throws ArgumentError BoxFilter(c; dims=(2,),      width=3)
            @test_throws ArgumentError BoxFilter(c; dims=(3,),      width=3)
            @test_throws ArgumentError BoxFilter(c; dims=(1, 2, 3), width=3)

            # width = 2 is fine since halo is exactly 2.
            @test BoxFilter(c; dims=(1, 2, 3), width=2) isa KernelFunctionOperation
        end
    end
end
