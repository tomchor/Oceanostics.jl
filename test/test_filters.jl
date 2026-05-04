using Test
using CUDA: has_cuda_gpu, @allowscalar
using Oceananigans
using Oceananigans: fill_halo_regions!, location
using Oceananigans.AbstractOperations: KernelFunctionOperation

using Oceanostics
using Oceanostics: BoxFilter, GaussianFilter

arch = has_cuda_gpu() ? GPU() : CPU()

#+++ Test helpers
function make_grid(; Nx=8, Ny=8, Nz=8, halo=(2, 2, 2), topology=(Periodic, Periodic, Periodic))
    return RectilinearGrid(arch, size = (Nx, Ny, Nz),
                           x = (0, 1), y = (0, 1), z = (0, 1),
                           halo = halo,
                           topology = topology)
end

function center_field_from(grid, f)
    c = CenterField(grid)
    set!(c, f)
    fill_halo_regions!(c)
    return c
end

# Throughout the test functions, `width` is the half-width of the stencil
# (cells on each side of the centre cell), which is the natural variable for
# the reference implementations below. The filter API takes the total
# `n_points = 2*width + 1`, so this helper does the conversion in one place.
function compute_filter(ψ, Filter, dims, width; kwargs...)
    cf = Field(Filter(ψ; dims=dims, n_points=2*width + 1, kwargs...))
    return cf
end

box_weights(width) = ntuple(_ -> 1.0, 2*width + 1)
gauss_weights(width, σ) = ntuple(idx -> exp(-(idx - width - 1)^2 / (2σ^2)), 2*width + 1)
#---

#+++ Reference implementations
function reference_weighted_average_periodic(ic, dims, width, Ns, w1d)
    Nx, Ny, Nz = Ns
    rx = 1 in dims ? (-width:width) : (0:0)
    ry = 2 in dims ? (-width:width) : (0:0)
    rz = 3 in dims ? (-width:width) : (0:0)
    ref = similar(ic)
    for i in 1:Nx, j in 1:Ny, k in 1:Nz
        s = zero(eltype(ic)); w_sum = zero(eltype(ic))
        for di in rx, dj in ry, dk in rz
            wx = 1 in dims ? w1d[di + width + 1] : 1.0
            wy = 2 in dims ? w1d[dj + width + 1] : 1.0
            wz = 3 in dims ? w1d[dk + width + 1] : 1.0
            w = wx * wy * wz
            s += w * ic[mod1(i + di, Nx), mod1(j + dj, Ny), mod1(k + dk, Nz)]
            w_sum += w
        end
        ref[i, j, k] = s / w_sum
    end
    return ref
end

function reference_weighted_average_shrink(ic, dims, width, Ns, w1d)
    Nx, Ny, Nz = Ns
    rx = 1 in dims ? (-width:width) : (0:0)
    ry = 2 in dims ? (-width:width) : (0:0)
    rz = 3 in dims ? (-width:width) : (0:0)
    ref = similar(ic)
    for i in 1:Nx, j in 1:Ny, k in 1:Nz
        s = zero(eltype(ic)); w_sum = zero(eltype(ic))
        for di in rx, dj in ry, dk in rz
            ii = i + di; jj = j + dj; kk = k + dk
            x_ok = !(1 in dims) || (1 <= ii <= Nx)
            y_ok = !(2 in dims) || (1 <= jj <= Ny)
            z_ok = !(3 in dims) || (1 <= kk <= Nz)
            if x_ok && y_ok && z_ok
                wx = 1 in dims ? w1d[di + width + 1] : 1.0
                wy = 2 in dims ? w1d[dj + width + 1] : 1.0
                wz = 3 in dims ? w1d[dk + width + 1] : 1.0
                w = wx * wy * wz
                s += w * ic[ii, jj, kk]
                w_sum += w
            end
        end
        ref[i, j, k] = s / w_sum
    end
    return ref
end

function reference_weighted_average_edge(ic, dims, width, Ns, w1d)
    Nx, Ny, Nz = Ns
    rx = 1 in dims ? (-width:width) : (0:0)
    ry = 2 in dims ? (-width:width) : (0:0)
    rz = 3 in dims ? (-width:width) : (0:0)
    ref = similar(ic)
    for i in 1:Nx, j in 1:Ny, k in 1:Nz
        s = zero(eltype(ic)); w_sum = zero(eltype(ic))
        for di in rx, dj in ry, dk in rz
            ii = 1 in dims ? clamp(i + di, 1, Nx) : i + di
            jj = 2 in dims ? clamp(j + dj, 1, Ny) : j + dj
            kk = 3 in dims ? clamp(k + dk, 1, Nz) : k + dk
            wx = 1 in dims ? w1d[di + width + 1] : 1.0
            wy = 2 in dims ? w1d[dj + width + 1] : 1.0
            wz = 3 in dims ? w1d[dk + width + 1] : 1.0
            w = wx * wy * wz
            s += w * ic[ii, jj, kk]
            w_sum += w
        end
        ref[i, j, k] = s / w_sum
    end
    return ref
end

function reference_weighted_average_constant_1d(ic, d, width, Ns, w1d, left, right)
    Nx, Ny, Nz = Ns
    N_d = Ns[d]
    ref = similar(ic)
    for i in 1:Nx, j in 1:Ny, k in 1:Nz
        s = zero(eltype(ic)); w_sum = zero(eltype(ic))
        for di in -width:width
            w = w1d[di + width + 1]
            center = d == 1 ? i : d == 2 ? j : k
            off    = center + di
            if off < 1
                s += w * left
            elseif off > N_d
                s += w * right
            else
                idx = d == 1 ? (off, j, k) : d == 2 ? (i, off, k) : (i, j, off)
                s += w * ic[idx...]
            end
            w_sum += w
        end
        ref[i, j, k] = s / w_sum
    end
    return ref
end
#---

#+++ Test functions (shared across filters)
function test_constructor(grid, Filter, fkw)
    bf = Filter(CenterField(grid); dims=(1,), n_points=3, fkw...)
    @test bf isa KernelFunctionOperation
    @test bf isa Filter
end

function test_linear_field_unchanged(grid, Nx, Ny, Nz, Filter, fkw)
    c = center_field_from(grid, (x, y, z) -> x + 2y + 3z)
    for dims in [(1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
        cf = compute_filter(c, Filter, dims, 1; fkw...)
        rx = 1 in dims ? (2:Nx-1) : (1:Nx)
        ry = 2 in dims ? (2:Ny-1) : (1:Ny)
        rz = 3 in dims ? (2:Nz-1) : (1:Nz)
        @test interior(cf)[rx, ry, rz] ≈ interior(c)[rx, ry, rz]
    end
end

function test_constant_field_unchanged(grid, Filter, fkw)
    c = center_field_from(grid, (x, y, z) -> 3.14)
    cf = compute_filter(c, Filter, (1, 2, 3), 2; fkw...)
    @test all(interior(cf) .≈ 3.14)
end

function test_periodic_stencil_sum(grid, Ns, Filter, make_weights, fkw)
    c  = center_field_from(grid, (x, y, z) -> sin(2π * x) * cos(2π * y) + z^2)
    ic = Array(interior(c))
    for (dims, width) in [((1,),      1),
                          ((1, 2, 3), 1),
                          ((1, 2),    2),
                          ((1, 2, 3), 2)]
        cf  = compute_filter(c, Filter, dims, width; fkw...)
        w1d = make_weights(width)
        ref = reference_weighted_average_periodic(ic, dims, width, Ns, w1d)
        @test Array(interior(cf)) ≈ ref
    end
end

function test_output_location(grid, Filter, fkw)
    @test location(Filter(CenterField(grid); dims=(1,),      n_points=3, fkw...)) == (Center, Center, Center)
    @test location(Filter(XFaceField(grid);  dims=(1,),      n_points=3, fkw...)) == (Face,   Center, Center)
    @test location(Filter(YFaceField(grid);  dims=(1, 2),    n_points=3, fkw...)) == (Center, Face,   Center)
    @test location(Filter(ZFaceField(grid);  dims=(1, 2, 3), n_points=3, fkw...)) == (Center, Center, Face)
end

function test_abstract_operation_input(grid, Nx, Ny, Filter, fkw)
    c  = center_field_from(grid, (x, y, z) -> x + y)
    op = 2 * c
    @test Filter(op; dims=(1, 2), n_points=3, fkw...) isa KernelFunctionOperation

    f = compute_filter(op, Filter, (1, 2), 1; fkw...)
    @test interior(f)[2:Nx-1, 2:Ny-1, :] ≈ 2 .* interior(c)[2:Nx-1, 2:Ny-1, :]
end

function test_kfo_input(grid, Filter, fkw)
    c     = center_field_from(grid, (x, y, z) -> sin(2π * x))
    inner = Filter(c;     dims=(1,), n_points=3, fkw...)
    outer = Filter(inner; dims=(2,), n_points=3, fkw...)
    @test outer isa KernelFunctionOperation
    @test location(outer) == (Center, Center, Center)
end

function test_dims_validation(grid, Filter, fkw)
    c = CenterField(grid)
    @test_throws ArgumentError Filter(c; dims=(),     n_points=3, fkw...)
    @test_throws ArgumentError Filter(c; dims=(0,),   n_points=3, fkw...)
    @test_throws ArgumentError Filter(c; dims=(4,),   n_points=3, fkw...)
    @test_throws ArgumentError Filter(c; dims=(1, 1), n_points=3, fkw...)
    @test_throws ArgumentError Filter(c; dims=(:x,),  n_points=3, fkw...)
    @test_throws ArgumentError Filter(c; dims=[1, 2], n_points=3, fkw...)
end

function test_n_points_validation(grid, Filter, fkw)
    c = CenterField(grid)
    @test_throws ArgumentError Filter(c; dims=(1,), n_points=0,   fkw...)  # zero
    @test_throws ArgumentError Filter(c; dims=(1,), n_points=-1,  fkw...)  # negative
    @test_throws ArgumentError Filter(c; dims=(1,), n_points=1,   fkw...)  # below minimum (3)
    @test_throws ArgumentError Filter(c; dims=(1,), n_points=2,   fkw...)  # even
    @test_throws ArgumentError Filter(c; dims=(1,), n_points=4,   fkw...)  # even
    @test_throws ArgumentError Filter(c; dims=(1,), n_points=3.5, fkw...)  # non-integer
end

# For a periodic direction with N cells, wrap_periodic_index is only valid when
# the stencil spans at most one period, i.e. n_points ≤ 2N+1.
function test_periodic_n_points_too_large(Filter, fkw)
    # N=4 in each direction → 2N+1 = 9; n_points=11 exceeds this limit.
    small_periodic = make_grid(; Nx=4, Ny=4, Nz=4, topology=(Periodic, Periodic, Periodic))
    small_bounded  = make_grid(; Nx=4, Ny=4, Nz=4, topology=(Bounded,  Bounded,  Bounded))
    small_mixed    = make_grid(; Nx=4, Ny=4, Nz=4, topology=(Periodic, Bounded,  Bounded))

    c_p = CenterField(small_periodic)
    c_b = CenterField(small_bounded)
    c_m = CenterField(small_mixed)

    # n_points=11 > 2*4+1=9 in a periodic direction: must fail
    @test_throws ArgumentError Filter(c_p; dims=(1,), n_points=11, fkw...)
    # n_points=9 == 2*4+1=9 in a periodic direction: edge of valid range, must succeed
    @test Filter(c_p; dims=(1,), n_points=9, fkw...) isa KernelFunctionOperation
    # n_points=11 in a bounded direction only: no periodic constraint, must succeed
    @test Filter(c_b; dims=(1,), n_points=11, fkw...) isa KernelFunctionOperation
    # Mixed topology: n_points=11 in periodic x must fail, in bounded y must succeed
    @test_throws ArgumentError Filter(c_m; dims=(1,), n_points=11, fkw...)
    @test Filter(c_m; dims=(2,), n_points=11, fkw...) isa KernelFunctionOperation
end

function test_small_halo(Filter, fkw)
    tiny_periodic = make_grid(; halo=(1, 1, 1), topology=(Periodic, Periodic, Periodic))
    tiny_bounded  = make_grid(; halo=(1, 1, 1), topology=(Bounded,  Bounded,  Bounded))
    tiny_mixed    = make_grid(; halo=(1, 1, 1), topology=(Periodic, Bounded,  Bounded))

    for g in (tiny_periodic, tiny_bounded, tiny_mixed)
        c = CenterField(g)
        @test Filter(c; dims=(1,),      n_points=11, fkw...)                                 isa KernelFunctionOperation
        @test Filter(c; dims=(1, 2, 3), n_points=11, boundary=:edge, fkw...)                 isa KernelFunctionOperation
        @test Filter(c; dims=(1, 2, 3), n_points=11, boundary=(left=0.0, right=0.0), fkw...) isa KernelFunctionOperation
    end
end

function test_shrink_boundary(Ns, Filter, make_weights, fkw)
    bounded_grid = make_grid(; halo=(1, 1, 1), topology=(Bounded, Bounded, Bounded))
    c  = center_field_from(bounded_grid, (x, y, z) -> sin(2π * x) * cos(2π * y) + z^2)
    ic = Array(interior(c))
    for dims in [(1,), (2,), (3,), (1, 2), (1, 2, 3)]
        width = 2
        cf  = compute_filter(c, Filter, dims, width; fkw...)
        w1d = make_weights(width)
        ref = reference_weighted_average_shrink(ic, dims, width, Ns, w1d)
        @test Array(interior(cf)) ≈ ref
    end
end

function test_edge_boundary(Ns, Filter, make_weights, fkw)
    bounded_grid = make_grid(; halo=(1, 1, 1), topology=(Bounded, Bounded, Bounded))
    c  = center_field_from(bounded_grid, (x, y, z) -> sin(2π * x) * cos(2π * y) + z^2)
    ic = Array(interior(c))
    for dims in [(1,), (2,), (3,), (1, 2, 3)]
        width = 2
        cf  = compute_filter(c, Filter, dims, width; boundary=:edge, fkw...)
        w1d = make_weights(width)
        ref = reference_weighted_average_edge(ic, dims, width, Ns, w1d)
        @test Array(interior(cf)) ≈ ref
    end
end

function test_constant_pad_boundary(Ns, Filter, make_weights, fkw)
    bounded_grid = make_grid(; halo=(1, 1, 1), topology=(Bounded, Bounded, Bounded))
    c  = center_field_from(bounded_grid, (x, y, z) -> sin(2π * x) * cos(2π * y))
    ic = Array(interior(c))
    left, right = -1.25, 2.5
    for (d, width) in [(1, 2), (2, 3), (3, 2)]
        dims = (d,)
        cf  = compute_filter(c, Filter, dims, width; boundary=(left=left, right=right), fkw...)
        w1d = make_weights(width)
        ref = reference_weighted_average_constant_1d(ic, d, width, Ns, w1d, left, right)
        @test Array(interior(cf)) ≈ ref
    end
end

function test_per_dim_boundary(Ns, Filter, make_weights, fkw)
    bounded_grid = make_grid(; halo=(1, 1, 1), topology=(Bounded, Bounded, Bounded))
    c  = center_field_from(bounded_grid, (x, y, z) -> sin(2π * x) * cos(2π * y))
    ic = Array(interior(c))
    width = 2
    cf = compute_filter(c, Filter, (1, 2), width; boundary=(:shrink, :edge), fkw...)
    w1d = make_weights(width)
    shrink_in_x       = reference_weighted_average_shrink(ic,        (1,), width, Ns, w1d)
    edge_of_shrink_xy = reference_weighted_average_edge(shrink_in_x, (2,), width, Ns, w1d)
    @test Array(interior(cf)) ≈ edge_of_shrink_xy
end

function test_periodic_ignores_boundary(Filter, fkw)
    mixed_grid = make_grid(; halo=(1, 1, 1), topology=(Periodic, Bounded, Bounded))
    c = center_field_from(mixed_grid, (x, y, z) -> sin(2π * x) + y)
    cf_default = compute_filter(c, Filter, (1,), 3; fkw...)
    cf_absurd  = compute_filter(c, Filter, (1,), 3; boundary=(left=1e6, right=-1e6), fkw...)
    @test Array(interior(cf_default)) ≈ Array(interior(cf_absurd))
end

function test_boundary_validation(grid, Filter, fkw)
    c = CenterField(grid)
    @test_throws ArgumentError Filter(c; dims=(1,),   n_points=3, boundary=:foo, fkw...)
    @test_throws ArgumentError Filter(c; dims=(1,),   n_points=3, boundary=(foo=1,), fkw...)
    @test_throws ArgumentError Filter(c; dims=(1,),   n_points=3, boundary=(left=1, right=2, mid=3), fkw...)
    @test_throws ArgumentError Filter(c; dims=(1,),   n_points=3, boundary=42, fkw...)
    @test_throws ArgumentError Filter(c; dims=(1, 2), n_points=3, boundary=(:shrink,), fkw...)
    @test_throws ArgumentError Filter(c; dims=(1,),   n_points=3, boundary=(:shrink, :edge), fkw...)
end
#---

#+++ BoxFilter-specific hand-computed tests
function test_1d_periodic_hand_computed()
    grid_1d = RectilinearGrid(arch, size = (10,),
                              x = (0, 1),
                              halo = (2,),
                              topology = (Periodic, Flat, Flat))
    c = CenterField(grid_1d)
    interior(c)[:] .= Float64.(0:9)
    fill_halo_regions!(c)

    expected1 = [10/3, 1, 2, 3, 4, 5, 6, 7, 8, 17/3]
    @test Array(interior(compute_filter(c, BoxFilter, (1,), 1)))[:] ≈ expected1

    expected2 = [4, 3, 2, 3, 4, 5, 6, 7, 6, 5]
    @test Array(interior(compute_filter(c, BoxFilter, (1,), 2)))[:] ≈ expected2
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
    @test Array(interior(compute_filter(c, BoxFilter, (1,), 2)))[:] ≈ shrink_expected

    # :shrink, width=3
    shrink_expected_w3 = [1.5, 2, 2.5, 3, 4, 5, 6, 6.5, 7, 7.5]
    @test Array(interior(compute_filter(c, BoxFilter, (1,), 3)))[:] ≈ shrink_expected_w3

    # :edge, width=2
    edge_expected = [0.6, 1.2, 2, 3, 4, 5, 6, 7, 7.8, 8.4]
    @test Array(interior(compute_filter(c, BoxFilter, (1,), 2; boundary=:edge)))[:] ≈ edge_expected

    # :edge, width=3
    edge_expected_w3 = [6/7, 10/7, 15/7, 3, 4, 5, 6, 48/7, 53/7, 57/7]
    @test Array(interior(compute_filter(c, BoxFilter, (1,), 3; boundary=:edge)))[:] ≈ edge_expected_w3

    # Constant-pad (left=-1, right=-2), width=2
    const_expected = [0.2, 1, 2, 3, 4, 5, 6, 7, 5.6, 4]
    @test Array(interior(compute_filter(c, BoxFilter, (1,), 2;
                                        boundary=(left=-1.0, right=-2.0))))[:] ≈ const_expected

    # Constant-pad (left=-1, right=-2), width=3
    const_expected_w3 = [3/7, 8/7, 2, 3, 4, 5, 6, 37/7, 31/7, 24/7]
    @test Array(interior(compute_filter(c, BoxFilter, (1,), 3;
                                        boundary=(left=-1.0, right=-2.0))))[:] ≈ const_expected_w3
end
#---

#+++ GaussianFilter-specific tests
function test_σ_validation(grid)
    c = CenterField(grid)
    @test_throws ArgumentError GaussianFilter(c; dims=(1,), n_points=3, σ=0)
    @test_throws ArgumentError GaussianFilter(c; dims=(1,), n_points=3, σ=-1.0)
    @test_throws ArgumentError GaussianFilter(c; dims=(1,), n_points=3, σ="foo")
end

# Test the n_points=nothing inference path. On a uniform grid with spacing Δ,
# infer_width(σ, grid, d) = ceil(2σ/Δ). Here σ = 1.5*Δ → width = 3, n_points = 7.
# The inferred filter must produce the same output as the explicit n_points=7 filter.
function test_gaussian_n_points_inferred()
    N = 8; Δ = 1/N
    σ = 1.5 * Δ   # infer_width = ceil(2*1.5) = 3 → n_points_inferred = 7
    grid = make_grid(; Nx=N, Ny=N, Nz=N)
    c = center_field_from(grid, (x, y, z) -> sin(2π * x) + cos(2π * z))

    # Smoke: omitting n_points produces a valid KFO for both 1D and 2D filtering
    @test GaussianFilter(c; dims=(1,),    σ=σ) isa KernelFunctionOperation
    @test GaussianFilter(c; dims=(1, 3), σ=σ) isa KernelFunctionOperation

    # Numerical: inferred path equals explicit n_points=7
    f_inferred = Field(GaussianFilter(c; dims=(1,), σ=σ))
    f_explicit  = Field(GaussianFilter(c; dims=(1,), σ=σ, n_points=7))
    @test Array(interior(f_inferred)) ≈ Array(interior(f_explicit))

    # Same check for a 2D filter
    f_inferred_2d = Field(GaussianFilter(c; dims=(1, 3), σ=σ))
    f_explicit_2d = Field(GaussianFilter(c; dims=(1, 3), σ=σ, n_points=7))
    @test Array(interior(f_inferred_2d)) ≈ Array(interior(f_explicit_2d))
end

# Test per-dimension n_points::Tuple with dims listed out of sorted order.
# dims=(3, 1), n_points=(5, 7) means: 5 points for direction 3, 7 for direction 1.
# This should equal dims=(1, 3), n_points=(7, 5) — the same widths, dims just reordered.
function test_gaussian_n_points_tuple_order()
    N = 8; Δ = 1/N
    σ = 1.5 * Δ
    grid = make_grid(; Nx=N, Ny=N, Nz=N)
    c = center_field_from(grid, (x, y, z) -> sin(2π * x) * cos(2π * z))

    # Two ways to spell the same filter: width=3 in x (dim 1), width=2 in z (dim 3)
    f_a = Field(GaussianFilter(c; dims=(3, 1), σ=σ, n_points=(5, 7)))  # z→5, x→7
    f_b = Field(GaussianFilter(c; dims=(1, 3), σ=σ, n_points=(7, 5)))  # x→7, z→5
    @test Array(interior(f_a)) ≈ Array(interior(f_b))

    # Validation: wrong tuple length must error
    @test_throws ArgumentError GaussianFilter(c; dims=(1, 3), σ=σ, n_points=(5,))
    @test_throws ArgumentError GaussianFilter(c; dims=(1, 3), σ=σ, n_points=(5, 7, 9))
end

# GaussianFilter precomputes its weights using one Δ per direction, so it
# only supports uniform spacing along every filtered direction. Non-uniform
# directions must error at construction time with a clear message. Other
# directions on the same grid may be non-uniform, as long as they aren't
# being filtered.
function test_gaussian_uniform_spacing_required()
    zfaces(k) = -1 + (k-1)/8 + 0.05 * sin(2π*(k-1)/8)
    grid = RectilinearGrid(arch, size=(8, 8, 8), x=(0, 1), y=(0, 1), z=zfaces,
                           topology=(Periodic, Periodic, Bounded))
    c = CenterField(grid)

    # Filtering the stretched z direction must error — alone or alongside other dims.
    @test_throws ArgumentError GaussianFilter(c; dims=(3,),   σ=0.1)
    @test_throws ArgumentError GaussianFilter(c; dims=(1, 3), σ=0.1)

    # Filtering only the uniform x and y directions must still succeed.
    @test GaussianFilter(c; dims=(1,),   σ=0.1) isa KernelFunctionOperation
    @test GaussianFilter(c; dims=(1, 2), σ=0.1) isa KernelFunctionOperation

    # BoxFilter is unaffected: its weights don't depend on Δ.
    @test BoxFilter(c; dims=(3,), n_points=3) isa KernelFunctionOperation
end

# Apply the GaussianFilter to a discrete Dirac delta (a 1D field that is zero
# everywhere except at one interior point). The output must equal the
# normalized Gaussian kernel — this is the filter's impulse response, and
# verifies that GaussianFilter actually applies the kernel it advertises.
function test_gaussian_dirac_delta()
    N = 21
    grid_1d = RectilinearGrid(arch, size = (N,),
                              x = (0, 1),
                              halo = (3,),
                              topology = (Periodic, Flat, Flat))
    Δ = 1/N  # uniform spacing of the test grid
    i₀ = N ÷ 2 + 1  # interior point — chosen so the stencil never wraps

    # σ_cells is the standard deviation in cells (drives the reference
    # weights); σ_physical = σ_cells * Δ is what the API takes.
    for (width, σ_cells) in [(3, 1.0), (3, 2.0), (5, 1.5)]
        c = CenterField(grid_1d)
        parent(c) .= 0
        @allowscalar interior(c)[i₀] = 1
        fill_halo_regions!(c)

        cf = compute_filter(c, GaussianFilter, (1,), width; σ=σ_cells * Δ)

        weights = gauss_weights(width, σ_cells)
        W_sum = sum(weights)
        expected = zeros(N)
        for j in 1:N
            Δij = i₀ - j
            if abs(Δij) <= width
                expected[j] = weights[Δij + width + 1] / W_sum
            end
        end

        @test Array(interior(cf))[:] ≈ expected
    end
end
#---

#+++ Run tests
# Reference weights are computed in cells; the GaussianFilter API takes σ in
# physical units. The shared test grid is uniform with Δ = 1/8, so a physical
# σ of 1/8 corresponds to σ-in-cells = 1.0 (which is what the reference
# weights use).
σ_test_cells = 1.0
σ_test_physical = σ_test_cells * (1/8)

filter_configs = [
    ("BoxFilter",      BoxFilter,      box_weights,                          NamedTuple()),
    ("GaussianFilter", GaussianFilter, w -> gauss_weights(w, σ_test_cells),  (; σ=σ_test_physical)),
]

@testset "Filters on $(typeof(arch).name.wrapper)" begin
    for (filter_name, Filter, make_weights, fkw) in filter_configs
        @testset "$filter_name" begin
            Nx = Ny = Nz = 8
            Ns = (Nx, Ny, Nz)
            grid = make_grid(; Nx, Ny, Nz)

            @testset "Smoke tests" begin
                test_constructor(grid, Filter, fkw)
                test_output_location(grid, Filter, fkw)
                test_small_halo(Filter, fkw)
            end

            @testset "Identity properties" begin
                test_linear_field_unchanged(grid, Nx, Ny, Nz, Filter, fkw)
                test_constant_field_unchanged(grid, Filter, fkw)
            end

            @testset "Numerical correctness against explicit references" begin
                test_periodic_stencil_sum(grid, Ns, Filter, make_weights, fkw)
                test_shrink_boundary(Ns, Filter, make_weights, fkw)
                test_edge_boundary(Ns, Filter, make_weights, fkw)
                test_constant_pad_boundary(Ns, Filter, make_weights, fkw)
                test_per_dim_boundary(Ns, Filter, make_weights, fkw)
                test_periodic_ignores_boundary(Filter, fkw)
            end

            @testset "Composability" begin
                test_abstract_operation_input(grid, Nx, Ny, Filter, fkw)
                test_kfo_input(grid, Filter, fkw)
            end

            @testset "Argument validation" begin
                test_dims_validation(grid, Filter, fkw)
                test_n_points_validation(grid, Filter, fkw)
                test_periodic_n_points_too_large(Filter, fkw)
                test_boundary_validation(grid, Filter, fkw)
            end
        end
    end

    @testset "BoxFilter hand-computed" begin
        test_1d_periodic_hand_computed()
        test_1d_bounded_hand_computed()
    end

    @testset "GaussianFilter-specific" begin
        test_σ_validation(make_grid())
        test_gaussian_dirac_delta()
        test_gaussian_n_points_inferred()
        test_gaussian_n_points_tuple_order()
        test_gaussian_uniform_spacing_required()
    end
end
#---
