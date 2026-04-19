using Test
using Oceananigans
using Oceananigans: fill_halo_regions!, location
using Oceananigans.AbstractOperations: KernelFunctionOperation

using Oceanostics
using Oceanostics: BoxFilter

@testset "Filters" begin
    @testset "BoxFilter" begin
        Nx = Ny = Nz = 8
        Hx = Hy = Hz = 2
        grid = RectilinearGrid(size = (Nx, Ny, Nz),
                               x = (0, 1), y = (0, 1), z = (0, 1),
                               halo = (Hx, Hy, Hz),
                               topology = (Periodic, Periodic, Periodic))

        @testset "Constructor returns KernelFunctionOperation" begin
            c = CenterField(grid)
            bf = BoxFilter(c; dims=(1,), width=1)
            @test bf isa KernelFunctionOperation
            @test bf isa BoxFilter
        end

        @testset "Linear field is unchanged on interior (1D, 2D, 3D)" begin
            # For a linear field, a symmetric (2w+1)-point running mean reproduces the field
            # exactly on cells whose stencil does not cross a periodic boundary (where a
            # linear function is not truly periodic).
            c = CenterField(grid)
            set!(c, (x, y, z) -> x + 2y + 3z)
            fill_halo_regions!(c)

            interior_range_x = 2:Nx-1
            interior_range_y = 2:Ny-1
            interior_range_z = 2:Nz-1

            for dims in [(1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
                cf = Field(BoxFilter(c; dims=dims, width=1))
                compute!(cf)

                rx = 1 in dims ? interior_range_x : (1:Nx)
                ry = 2 in dims ? interior_range_y : (1:Ny)
                rz = 3 in dims ? interior_range_z : (1:Nz)

                @test interior(cf)[rx, ry, rz] ≈ interior(c)[rx, ry, rz]
            end
        end

        @testset "Constant field is unchanged" begin
            c = CenterField(grid)
            set!(c, (x, y, z) -> 3.14)
            fill_halo_regions!(c)
            cf = Field(BoxFilter(c; dims=(1, 2, 3), width=2))
            compute!(cf)
            @test all(interior(cf) .≈ 3.14)
        end

        @testset "Averaging matches an explicit stencil sum" begin
            c = CenterField(grid)
            set!(c, (x, y, z) -> sin(2π * x) * cos(2π * y) + z^2)
            fill_halo_regions!(c)

            ic = interior(c)

            # Reference: explicit 1D average in x with width = 1 (wrap via mod1)
            cf = Field(BoxFilter(c; dims=(1,), width=1))
            compute!(cf)

            ref = similar(ic)
            for i in 1:Nx, j in 1:Ny, k in 1:Nz
                ref[i, j, k] = (ic[mod1(i-1, Nx), j, k] + ic[i, j, k] + ic[mod1(i+1, Nx), j, k]) / 3
            end
            @test interior(cf) ≈ ref

            # Reference: explicit 3D average with width = 1 (27-point box, wrap via mod1)
            cf3 = Field(BoxFilter(c; dims=(1, 2, 3), width=1))
            compute!(cf3)

            ref3 = similar(ic)
            for i in 1:Nx, j in 1:Ny, k in 1:Nz
                s = 0.0
                for di in -1:1, dj in -1:1, dk in -1:1
                    s += ic[mod1(i+di, Nx), mod1(j+dj, Ny), mod1(k+dk, Nz)]
                end
                ref3[i, j, k] = s / 27
            end
            @test interior(cf3) ≈ ref3
        end

        @testset "Output location matches input location" begin
            c = CenterField(grid)
            u = XFaceField(grid)
            v = YFaceField(grid)
            w = ZFaceField(grid)

            @test location(BoxFilter(c; dims=(1,),      width=1)) == (Center, Center, Center)
            @test location(BoxFilter(u; dims=(1,),      width=1)) == (Face,   Center, Center)
            @test location(BoxFilter(v; dims=(1, 2),    width=1)) == (Center, Face,   Center)
            @test location(BoxFilter(w; dims=(1, 2, 3), width=1)) == (Center, Center, Face)
        end

        @testset "Accepts AbstractOperation as input" begin
            c = CenterField(grid)
            set!(c, (x, y, z) -> x + y)
            fill_halo_regions!(c)

            op = 2 * c  # BinaryOperation at ccc
            bf = BoxFilter(op; dims=(1, 2), width=1)
            @test bf isa KernelFunctionOperation

            f = Field(bf)
            compute!(f)
            @test interior(f)[2:Nx-1, 2:Ny-1, :] ≈ 2 .* interior(c)[2:Nx-1, 2:Ny-1, :]
        end

        @testset "Accepts another KernelFunctionOperation as input" begin
            c = CenterField(grid)
            set!(c, (x, y, z) -> sin(2π * x))
            fill_halo_regions!(c)

            inner = BoxFilter(c; dims=(1,), width=1)
            outer = BoxFilter(inner; dims=(2,), width=1)
            @test outer isa KernelFunctionOperation
            @test location(outer) == (Center, Center, Center)
        end

        @testset "Validation of dims" begin
            c = CenterField(grid)
            @test_throws ArgumentError BoxFilter(c; dims=(),     width=1)
            @test_throws ArgumentError BoxFilter(c; dims=(0,),   width=1)
            @test_throws ArgumentError BoxFilter(c; dims=(4,),   width=1)
            @test_throws ArgumentError BoxFilter(c; dims=(1, 1), width=1)
            @test_throws ArgumentError BoxFilter(c; dims=(:x,),  width=1)
            @test_throws ArgumentError BoxFilter(c; dims=[1, 2], width=1)
        end

        @testset "Validation of width" begin
            c = CenterField(grid)
            @test_throws ArgumentError BoxFilter(c; dims=(1,), width=0)
            @test_throws ArgumentError BoxFilter(c; dims=(1,), width=-1)
            @test_throws ArgumentError BoxFilter(c; dims=(1,), width=1.5)
        end

        @testset "Periodic directions need no halo" begin
            # Fully periodic grid with a tiny halo; width ≫ halo is allowed because every
            # selected direction is periodic and wraps via mod1.
            small_halo_grid = RectilinearGrid(size = (Nx, Ny, Nz),
                                              x = (0, 1), y = (0, 1), z = (0, 1),
                                              halo = (1, 1, 1),
                                              topology = (Periodic, Periodic, Periodic))
            c = CenterField(small_halo_grid)
            @test BoxFilter(c; dims=(1,),      width=3) isa KernelFunctionOperation
            @test BoxFilter(c; dims=(1, 2, 3), width=3) isa KernelFunctionOperation

            # Numerical check: 1D filter of width = 3 on a known pattern must match a
            # reference computed with explicit mod1 wrapping.
            set!(c, (x, y, z) -> sin(2π * x) + cos(2π * y))
            fill_halo_regions!(c)

            width = 3
            cf = Field(BoxFilter(c; dims=(1,), width=width))
            compute!(cf)

            ic = interior(c)
            ref = similar(ic)
            for i in 1:Nx, j in 1:Ny, k in 1:Nz
                s = 0.0
                for di in -width:width
                    s += ic[mod1(i+di, Nx), j, k]
                end
                ref[i, j, k] = s / (2*width + 1)
            end
            @test interior(cf) ≈ ref
        end

        @testset "Mixed topology: halo enforced only on non-periodic dims" begin
            # (Periodic, Bounded, Bounded), small halo in y and z.
            mixed_grid = RectilinearGrid(size = (Nx, Ny, Nz),
                                         x = (0, 1), y = (0, 1), z = (0, 1),
                                         halo = (1, 2, 2),
                                         topology = (Periodic, Bounded, Bounded))
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
            # On a fully bounded grid with halo = (2, 2, 2), width = 3 must fail for any dim.
            bounded_grid = RectilinearGrid(size = (Nx, Ny, Nz),
                                           x = (0, 1), y = (0, 1), z = (0, 1),
                                           halo = (2, 2, 2),
                                           topology = (Bounded, Bounded, Bounded))
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
