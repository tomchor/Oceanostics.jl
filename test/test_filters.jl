using Test
using CUDA: has_cuda_gpu
using Oceananigans
using Oceananigans: fill_halo_regions!, location
using Oceananigans.AbstractOperations: KernelFunctionOperation

using Oceanostics
using Oceanostics: BoxFilter

arch = has_cuda_gpu() ? GPU() : CPU()

#+++ Test helpers
make_grid(; Nx=8, Ny=8, Nz=8, halo=(2, 2, 2), topology=(Periodic, Periodic, Periodic)) =
    RectilinearGrid(arch, size = (Nx, Ny, Nz),
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
    return cf
end
#---

#+++ Reference implementations
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
#---

#+++ Test functions
function test_constructor(grid)
    bf = BoxFilter(CenterField(grid); dims=(1,), width=1)
    @test bf isa KernelFunctionOperation
    @test bf isa BoxFilter
end

function test_linear_field_unchanged(grid, Nx, Ny, Nz)
    c = center_field_from(grid, (x, y, z) -> x + 2y + 3z)
    for dims in [(1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
        cf = compute_box_filter(c, dims, 1)
        rx = 1 in dims ? (2:Nx-1) : (1:Nx)
        ry = 2 in dims ? (2:Ny-1) : (1:Ny)
        rz = 3 in dims ? (2:Nz-1) : (1:Nz)
        @test interior(cf)[rx, ry, rz] ≈ interior(c)[rx, ry, rz]
    end
end

function test_constant_field_unchanged(grid)
    c = center_field_from(grid, (x, y, z) -> 3.14)
    cf = compute_box_filter(c, (1, 2, 3), 2)
    @test all(interior(cf) .≈ 3.14)
end

function test_periodic_stencil_sum(grid, Ns)
    c  = center_field_from(grid, (x, y, z) -> sin(2π * x) * cos(2π * y) + z^2)
    ic = Array(interior(c))
    for (dims, width) in [((1,),      1),
                          ((1, 2, 3), 1),
                          ((1, 2),    2),
                          ((1, 2, 3), 2)]
        cf  = compute_box_filter(c, dims, width)
        ref = reference_box_average_periodic(ic, dims, width, Ns)
        @test Array(interior(cf)) ≈ ref
    end
end

function test_1d_periodic_hand_computed()
    grid_1d = RectilinearGrid(arch, size = (10,),
                              x = (0, 1),
                              halo = (2,),
                              topology = (Periodic, Flat, Flat))
    c = CenterField(grid_1d)
    interior(c)[:] .= Float64.(0:9)
    fill_halo_regions!(c)

    expected1 = [10/3, 1, 2, 3, 4, 5, 6, 7, 8, 17/3]
    @test Array(interior(compute_box_filter(c, (1,), 1)))[:] ≈ expected1

    expected2 = [4, 3, 2, 3, 4, 5, 6, 7, 6, 5]
    @test Array(interior(compute_box_filter(c, (1,), 2)))[:] ≈ expected2
end

function test_output_location(grid)
    @test location(BoxFilter(CenterField(grid); dims=(1,),      width=1)) == (Center, Center, Center)
    @test location(BoxFilter(XFaceField(grid);  dims=(1,),      width=1)) == (Face,   Center, Center)
    @test location(BoxFilter(YFaceField(grid);  dims=(1, 2),    width=1)) == (Center, Face,   Center)
    @test location(BoxFilter(ZFaceField(grid);  dims=(1, 2, 3), width=1)) == (Center, Center, Face)
end

function test_abstract_operation_input(grid, Nx, Ny)
    c  = center_field_from(grid, (x, y, z) -> x + y)
    op = 2 * c
    @test BoxFilter(op; dims=(1, 2), width=1) isa KernelFunctionOperation

    f = compute_box_filter(op, (1, 2), 1)
    @test interior(f)[2:Nx-1, 2:Ny-1, :] ≈ 2 .* interior(c)[2:Nx-1, 2:Ny-1, :]
end

function test_kfo_input(grid)
    c     = center_field_from(grid, (x, y, z) -> sin(2π * x))
    inner = BoxFilter(c;     dims=(1,), width=1)
    outer = BoxFilter(inner; dims=(2,), width=1)
    @test outer isa KernelFunctionOperation
    @test location(outer) == (Center, Center, Center)
end

function test_dims_validation(grid)
    c = CenterField(grid)
    @test_throws ArgumentError BoxFilter(c; dims=(),     width=1)
    @test_throws ArgumentError BoxFilter(c; dims=(0,),   width=1)
    @test_throws ArgumentError BoxFilter(c; dims=(4,),   width=1)
    @test_throws ArgumentError BoxFilter(c; dims=(1, 1), width=1)
    @test_throws ArgumentError BoxFilter(c; dims=(:x,),  width=1)
    @test_throws ArgumentError BoxFilter(c; dims=[1, 2], width=1)
end

function test_width_validation(grid)
    c = CenterField(grid)
    @test_throws ArgumentError BoxFilter(c; dims=(1,), width=0)
    @test_throws ArgumentError BoxFilter(c; dims=(1,), width=-1)
    @test_throws ArgumentError BoxFilter(c; dims=(1,), width=1.5)
end

function test_small_halo()
    tiny_periodic = make_grid(; halo=(1, 1, 1), topology=(Periodic, Periodic, Periodic))
    tiny_bounded  = make_grid(; halo=(1, 1, 1), topology=(Bounded,  Bounded,  Bounded))
    tiny_mixed    = make_grid(; halo=(1, 1, 1), topology=(Periodic, Bounded,  Bounded))

    for g in (tiny_periodic, tiny_bounded, tiny_mixed)
        c = CenterField(g)
        @test BoxFilter(c; dims=(1,),      width=5)                                 isa KernelFunctionOperation
        @test BoxFilter(c; dims=(1, 2, 3), width=5, boundary=:edge)                 isa KernelFunctionOperation
        @test BoxFilter(c; dims=(1, 2, 3), width=5, boundary=(left=0.0, right=0.0)) isa KernelFunctionOperation
    end
end

function test_shrink_boundary(Ns)
    bounded_grid = make_grid(; halo=(1, 1, 1), topology=(Bounded, Bounded, Bounded))
    c  = center_field_from(bounded_grid, (x, y, z) -> sin(2π * x) * cos(2π * y) + z^2)
    ic = Array(interior(c))
    for dims in [(1,), (2,), (3,), (1, 2), (1, 2, 3)]
        cf  = compute_box_filter(c, dims, 2)
        ref = reference_box_average_shrink(ic, dims, 2, Ns)
        @test Array(interior(cf)) ≈ ref
    end
end

function test_edge_boundary(Ns)
    bounded_grid = make_grid(; halo=(1, 1, 1), topology=(Bounded, Bounded, Bounded))
    c  = center_field_from(bounded_grid, (x, y, z) -> sin(2π * x) * cos(2π * y) + z^2)
    ic = Array(interior(c))
    for dims in [(1,), (2,), (3,), (1, 2, 3)]
        cf  = compute_box_filter(c, dims, 2; boundary=:edge)
        ref = reference_box_average_edge(ic, dims, 2, Ns)
        @test Array(interior(cf)) ≈ ref
    end
end

function test_constant_pad_boundary(Ns)
    bounded_grid = make_grid(; halo=(1, 1, 1), topology=(Bounded, Bounded, Bounded))
    c  = center_field_from(bounded_grid, (x, y, z) -> sin(2π * x) * cos(2π * y))
    ic = Array(interior(c))
    left, right = -1.25, 2.5
    for (d, width) in [(1, 2), (2, 3), (3, 2)]
        dims = (d,)
        cf  = compute_box_filter(c, dims, width; boundary=(left=left, right=right))
        ref = reference_box_average_constant_1d(ic, d, width, Ns, left, right)
        @test Array(interior(cf)) ≈ ref
    end
end

function test_1d_bounded_hand_computed()
    grid_1d = RectilinearGrid(arch, size = (10,),
                              x = (0, 1),
                              halo = (1,),
                              topology = (Bounded, Flat, Flat))
    c = CenterField(grid_1d)
    interior(c)[:] .= Float64.(0:9)
    fill_halo_regions!(c)

    # :shrink, width=2
    shrink_expected = [1, 1.5, 2, 3, 4, 5, 6, 7, 7.5, 8]
    @test Array(interior(compute_box_filter(c, (1,), 2)))[:] ≈ shrink_expected

    # :shrink, width=3
    shrink_expected_w3 = [1.5, 2, 2.5, 3, 4, 5, 6, 6.5, 7, 7.5]
    @test Array(interior(compute_box_filter(c, (1,), 3)))[:] ≈ shrink_expected_w3

    # :edge, width=2
    edge_expected = [0.6, 1.2, 2, 3, 4, 5, 6, 7, 7.8, 8.4]
    @test Array(interior(compute_box_filter(c, (1,), 2; boundary=:edge)))[:] ≈ edge_expected

    # :edge, width=3
    edge_expected_w3 = [6/7, 10/7, 15/7, 3, 4, 5, 6, 48/7, 53/7, 57/7]
    @test Array(interior(compute_box_filter(c, (1,), 3; boundary=:edge)))[:] ≈ edge_expected_w3

    # Constant-pad (left=-1, right=-2), width=2
    const_expected = [0.2, 1, 2, 3, 4, 5, 6, 7, 5.6, 4]
    @test Array(interior(compute_box_filter(c, (1,), 2;
                                            boundary=(left=-1.0, right=-2.0))))[:] ≈ const_expected

    # Constant-pad (left=-1, right=-2), width=3
    const_expected_w3 = [3/7, 8/7, 2, 3, 4, 5, 6, 37/7, 31/7, 24/7]
    @test Array(interior(compute_box_filter(c, (1,), 3;
                                            boundary=(left=-1.0, right=-2.0))))[:] ≈ const_expected_w3
end

function test_per_dim_boundary(Ns)
    bounded_grid = make_grid(; halo=(1, 1, 1), topology=(Bounded, Bounded, Bounded))
    c  = center_field_from(bounded_grid, (x, y, z) -> sin(2π * x) * cos(2π * y))
    ic = Array(interior(c))
    cf = compute_box_filter(c, (1, 2), 2; boundary=(:shrink, :edge))
    shrink_in_x       = reference_box_average_shrink(ic,         (1,), 2, Ns)
    edge_of_shrink_xy = reference_box_average_edge(shrink_in_x,  (2,), 2, Ns)
    @test Array(interior(cf)) ≈ edge_of_shrink_xy
end

function test_periodic_ignores_boundary()
    mixed_grid = make_grid(; halo=(1, 1, 1), topology=(Periodic, Bounded, Bounded))
    c = center_field_from(mixed_grid, (x, y, z) -> sin(2π * x) + y)
    cf_default = compute_box_filter(c, (1,), 3)
    cf_absurd  = compute_box_filter(c, (1,), 3; boundary=(left=1e6, right=-1e6))
    @test Array(interior(cf_default)) ≈ Array(interior(cf_absurd))
end

function test_boundary_validation(grid)
    c = CenterField(grid)
    @test_throws ArgumentError BoxFilter(c; dims=(1,),   width=1, boundary=:foo)
    @test_throws ArgumentError BoxFilter(c; dims=(1,),   width=1, boundary=(foo=1,))
    @test_throws ArgumentError BoxFilter(c; dims=(1,),   width=1, boundary=(left=1, right=2, mid=3))
    @test_throws ArgumentError BoxFilter(c; dims=(1,),   width=1, boundary=42)
    @test_throws ArgumentError BoxFilter(c; dims=(1, 2), width=1, boundary=(:shrink,))
    @test_throws ArgumentError BoxFilter(c; dims=(1,),   width=1, boundary=(:shrink, :edge))
end
#---

#+++ Run tests
@testset "Filters on $(typeof(arch).name.wrapper)" begin
    @testset "BoxFilter" begin
        Nx = Ny = Nz = 8
        Ns = (Nx, Ny, Nz)
        grid = make_grid(; Nx, Ny, Nz)

        @testset "Constructor returns KernelFunctionOperation" begin
            test_constructor(grid)
        end

        @testset "Linear field is unchanged on interior (1D, 2D, 3D)" begin
            test_linear_field_unchanged(grid, Nx, Ny, Nz)
        end

        @testset "Constant field is unchanged" begin
            test_constant_field_unchanged(grid)
        end

        @testset "Averaging matches an explicit stencil sum (periodic)" begin
            test_periodic_stencil_sum(grid, Ns)
        end

        @testset "Exact hand-computed values on a 1D periodic grid" begin
            test_1d_periodic_hand_computed()
        end

        @testset "Output location matches input location" begin
            test_output_location(grid)
        end

        @testset "Accepts AbstractOperation as input" begin
            test_abstract_operation_input(grid, Nx, Ny)
        end

        @testset "Accepts another KernelFunctionOperation as input" begin
            test_kfo_input(grid)
        end

        @testset "Validation of dims" begin
            test_dims_validation(grid)
        end

        @testset "Validation of width" begin
            test_width_validation(grid)
        end

        @testset "Small halo is allowed for any width or topology" begin
            test_small_halo()
        end

        @testset ":shrink (default) on a bounded grid matches explicit reference" begin
            test_shrink_boundary(Ns)
        end

        @testset ":edge on a bounded grid matches explicit reference" begin
            test_edge_boundary(Ns)
        end

        @testset "Constant-pad (left, right) matches explicit reference" begin
            test_constant_pad_boundary(Ns)
        end

        @testset "Exact hand-computed values on a 1D bounded grid" begin
            test_1d_bounded_hand_computed()
        end

        @testset "Per-dim boundary tuple mixes policies across dims" begin
            test_per_dim_boundary(Ns)
        end

        @testset "Periodic dims ignore the boundary spec" begin
            test_periodic_ignores_boundary()
        end

        @testset "Validation of boundary spec" begin
            test_boundary_validation(grid)
        end
    end
end
#---
