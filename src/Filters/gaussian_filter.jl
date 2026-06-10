"""
    AbstractGaussianFilterKernel{D} <: Function

Supertype for the two 1D Gaussian-filter kernels along direction `D` (1, 2, or
3). The two concrete kernels share the same stencil shape and accumulation but
differ in how they obtain the per-offset weight:

  - [`GaussianFilterKernel`](@ref) — for **uniformly spaced** directions; looks
    up precomputed weights (the fast path, used on regular grids).
  - [`StretchedGaussianFilterKernel`](@ref) — for **variably spaced** directions;
    computes each weight on the fly from the physical node positions.

Both are tagged with `D` so the staged multi-direction evaluation and the
dispatch aliases can treat a mixed-spacing filter (e.g. uniform `x`, stretched
`z`) uniformly.
"""
abstract type AbstractGaussianFilterKernel{D} <: Function end

"""
    GaussianFilterKernel{D, W} <: AbstractGaussianFilterKernel{D}

Callable struct that computes a 1D Gaussian-weighted average along a
**uniformly spaced** direction `D` (1, 2, or 3). Stores precomputed
unnormalized weights in `weights::W`. Like `BoxFilterKernel`, has terminal
(indexable input) and recursive (function input) methods.
"""
struct GaussianFilterKernel{D, W} <: AbstractGaussianFilterKernel{D}
    weights::W
end

const GaussianFilter = CustomKFO{<:AbstractGaussianFilterKernel}

GaussianFilterKernel{D}(weights::W) where {D, W} = GaussianFilterKernel{D, W}(weights)

# `@unroll_full` (LLVM `llvm.loop.unroll.full` hint, defined in Filters.jl)
# is essential here: the weights tuple is captured by-value in each thread's
# register file, and tuple indexing by a non-constant `idx` would force
# spilling the tuple to per-thread local memory. Forcing a full unroll keeps
# every tuple access at a constant index, so the weights live in registers
# (or are folded into the IR as constants), avoiding a ~20× cliff at
# width ≳ 8.

#+++ Terminal methods (indexable input).
@inline function (kern::GaussianFilterKernel{1})(i, j, k, grid, ::Val{width}, policy, ψ) where {width}
    Nx = size(grid, 1)
    s = zero(grid); w_sum = zero(grid)
    @unroll_full for idx in 1:(2*width+1)
        Δi = idx - width - 1
        @inbounds w = kern.weights[idx]
        val, cnt = x_stencil_fetch(policy, ψ, i + Δi, j, k, Nx)
        s += w * val
        w_sum += w * cnt
    end
    return s / w_sum
end

@inline function (kern::GaussianFilterKernel{2})(i, j, k, grid, ::Val{width}, policy, ψ) where {width}
    Ny = size(grid, 2)
    s = zero(grid); w_sum = zero(grid)
    @unroll_full for idx in 1:(2*width+1)
        Δj = idx - width - 1
        @inbounds w = kern.weights[idx]
        val, cnt = y_stencil_fetch(policy, ψ, i, j + Δj, k, Ny)
        s += w * val
        w_sum += w * cnt
    end
    return s / w_sum
end

@inline function (kern::GaussianFilterKernel{3})(i, j, k, grid, ::Val{width}, policy, ψ) where {width}
    Nz = size(grid, 3)
    s = zero(grid); w_sum = zero(grid)
    @unroll_full for idx in 1:(2*width+1)
        Δk = idx - width - 1
        @inbounds w = kern.weights[idx]
        val, cnt = z_stencil_fetch(policy, ψ, i, j, k + Δk, Nz)
        s += w * val
        w_sum += w * cnt
    end
    return s / w_sum
end
#---

#+++ Recursive methods (function input — typically another GaussianFilterKernel).
@inline function (kern::GaussianFilterKernel{1})(i, j, k, grid, ::Val{width}, policy, f::Function, fargs...) where {width}
    Nx = size(grid, 1)
    s = zero(grid); w_sum = zero(grid)
    @unroll_full for idx in 1:(2*width+1)
        Δi = idx - width - 1
        @inbounds w = kern.weights[idx]
        val, cnt = x_stencil_call(policy, f, i + Δi, j, k, Nx, grid, fargs...)
        s += w * val
        w_sum += w * cnt
    end
    return s / w_sum
end

@inline function (kern::GaussianFilterKernel{2})(i, j, k, grid, ::Val{width}, policy, f::Function, fargs...) where {width}
    Ny = size(grid, 2)
    s = zero(grid); w_sum = zero(grid)
    @unroll_full for idx in 1:(2*width+1)
        Δj = idx - width - 1
        @inbounds w = kern.weights[idx]
        val, cnt = y_stencil_call(policy, f, i, j + Δj, k, Ny, grid, fargs...)
        s += w * val
        w_sum += w * cnt
    end
    return s / w_sum
end

@inline function (kern::GaussianFilterKernel{3})(i, j, k, grid, ::Val{width}, policy, f::Function, fargs...) where {width}
    Nz = size(grid, 3)
    s = zero(grid); w_sum = zero(grid)
    @unroll_full for idx in 1:(2*width+1)
        Δk = idx - width - 1
        @inbounds w = kern.weights[idx]
        val, cnt = z_stencil_call(policy, f, i, j, k + Δk, Nz, grid, fargs...)
        s += w * val
        w_sum += w * cnt
    end
    return s / w_sum
end
#---

#+++ Stretched (variably spaced) direction kernel
"""
    StretchedGaussianFilterKernel{D, S, L} <: AbstractGaussianFilterKernel{D}

Callable struct that computes a 1D Gaussian-weighted average along a
**variably spaced** direction `D` (1, 2, or 3). Because the grid spacing varies
along the direction, the per-offset weights cannot be precomputed once; instead
each weight is evaluated on the fly as

```
w = Δₘ · exp(-(xₘ - xᵢ)² / 2σ²),
```

where `xᵢ` is the centre cell's coordinate, `xₘ` the stencil cell's coordinate,
and `Δₘ` the stencil cell's width along `D`. The `Δₘ` factor is the quadrature
weight that makes the normalized sum a consistent approximation of the
continuous Gaussian convolution `∫ G_σ(x-x') ψ(x') dx' / ∫ G_σ(x-x') dx'`; it
keeps the filter from over-weighting finely resolved regions and preserves
constants (and, to quadrature accuracy, linear fields) on a stretched grid.

Fields:
  - `σ::S` — the Gaussian standard deviation in physical units.
  - `loc::L` — the field's location triple (instances, e.g. `(Center(), Center(),
    Face())`), used to read node coordinates and spacings along `D`.
  - `period::S` — the domain length along `D`, used only for `Periodic`
    directions to recover the unwrapped image coordinate of a wrapped stencil
    point.

This kernel reduces *exactly* to [`GaussianFilterKernel`](@ref) on a uniform
grid: there `Δₘ` is constant (cancels in the normalization) and `xₘ - xᵢ = Δm·Δ`,
matching the precomputed cell-offset weights. It has terminal (indexable input)
and recursive (function input) methods, like the uniform kernel.
"""
struct StretchedGaussianFilterKernel{D, S, L} <: AbstractGaussianFilterKernel{D}
    σ::S
    loc::L
    period::S
end

StretchedGaussianFilterKernel{D}(σ::S, loc::L, period::S) where {D, S, L} =
    StretchedGaussianFilterKernel{D, S, L}(σ, loc, period)

# `(coordinate, cell width)` of the stencil cell at the (possibly out-of-range)
# index `m` along a filtered direction, honoring the boundary policy's geometry.
# Only *interior* nodes/spacings are read (the index is wrapped or clamped into
# `1:N` up front), so these never index past the grid's coordinate arrays no
# matter how wide the stencil is:
#   • Periodic — the *unwrapped* image coordinate: the wrapped-index node shifted
#     by ±`period`, so the distance to the centre cell is the geometric distance
#     to the near periodic image rather than across the whole domain. The
#     periodic-`N` validation guarantees the stencil spans at most one period, so
#     a single ±`period` correction is exact (even on a stretched periodic grid,
#     where node positions tile with the period). The width at the wrapped index
#     equals the width of the true image cell.
#   • Bounded (shrink/edge/constant) — the clamped boundary cell's coordinate and
#     width. For `:shrink` this is irrelevant (out-of-range offsets carry count
#     0); for `:edge`/constant padding it places the contributing value at the
#     boundary, matching where that value is read from.
@inline function x_node_geometry(::PeriodicBoundary, grid, loc, i, j, k, m, N, L)
    mr = wrap_periodic_index(m, N)
    return xnode(mr, j, k, grid, loc...) - L * (m < 1) + L * (m > N), xspacing(mr, j, k, grid, loc...)
end
@inline function x_node_geometry(::AbstractBoundaryPolicy, grid, loc, i, j, k, m, N, L)
    mr = clamp(m, 1, N)
    return xnode(mr, j, k, grid, loc...), xspacing(mr, j, k, grid, loc...)
end
@inline function y_node_geometry(::PeriodicBoundary, grid, loc, i, j, k, m, N, L)
    mr = wrap_periodic_index(m, N)
    return ynode(i, mr, k, grid, loc...) - L * (m < 1) + L * (m > N), yspacing(i, mr, k, grid, loc...)
end
@inline function y_node_geometry(::AbstractBoundaryPolicy, grid, loc, i, j, k, m, N, L)
    mr = clamp(m, 1, N)
    return ynode(i, mr, k, grid, loc...), yspacing(i, mr, k, grid, loc...)
end
@inline function z_node_geometry(::PeriodicBoundary, grid, loc, i, j, k, m, N, L)
    mr = wrap_periodic_index(m, N)
    return znode(i, j, mr, grid, loc...) - L * (m < 1) + L * (m > N), zspacing(i, j, mr, grid, loc...)
end
@inline function z_node_geometry(::AbstractBoundaryPolicy, grid, loc, i, j, k, m, N, L)
    mr = clamp(m, 1, N)
    return znode(i, j, mr, grid, loc...), zspacing(i, j, mr, grid, loc...)
end

#+++ Terminal methods (indexable input).
@inline function (kern::StretchedGaussianFilterKernel{1})(i, j, k, grid, ::Val{width}, policy, ψ) where {width}
    Nx = size(grid, 1); loc = kern.loc
    x₀ = xnode(i, j, k, grid, loc...); σ = kern.σ
    s = zero(grid); w_sum = zero(grid)
    @unroll_full for idx in 1:(2*width+1)
        Δi = idx - width - 1
        x, Δx = x_node_geometry(policy, grid, loc, i, j, k, i + Δi, Nx, kern.period)
        w = Δx * exp(-(x - x₀)^2 / (2σ^2))
        val, cnt = x_stencil_fetch(policy, ψ, i + Δi, j, k, Nx)
        s += w * val
        w_sum += w * cnt
    end
    return s / w_sum
end

@inline function (kern::StretchedGaussianFilterKernel{2})(i, j, k, grid, ::Val{width}, policy, ψ) where {width}
    Ny = size(grid, 2); loc = kern.loc
    y₀ = ynode(i, j, k, grid, loc...); σ = kern.σ
    s = zero(grid); w_sum = zero(grid)
    @unroll_full for idx in 1:(2*width+1)
        Δj = idx - width - 1
        y, Δy = y_node_geometry(policy, grid, loc, i, j, k, j + Δj, Ny, kern.period)
        w = Δy * exp(-(y - y₀)^2 / (2σ^2))
        val, cnt = y_stencil_fetch(policy, ψ, i, j + Δj, k, Ny)
        s += w * val
        w_sum += w * cnt
    end
    return s / w_sum
end

@inline function (kern::StretchedGaussianFilterKernel{3})(i, j, k, grid, ::Val{width}, policy, ψ) where {width}
    Nz = size(grid, 3); loc = kern.loc
    z₀ = znode(i, j, k, grid, loc...); σ = kern.σ
    s = zero(grid); w_sum = zero(grid)
    @unroll_full for idx in 1:(2*width+1)
        Δk = idx - width - 1
        z, Δz = z_node_geometry(policy, grid, loc, i, j, k, k + Δk, Nz, kern.period)
        w = Δz * exp(-(z - z₀)^2 / (2σ^2))
        val, cnt = z_stencil_fetch(policy, ψ, i, j, k + Δk, Nz)
        s += w * val
        w_sum += w * cnt
    end
    return s / w_sum
end
#---

#+++ Recursive methods (function input — typically another AbstractGaussianFilterKernel).
@inline function (kern::StretchedGaussianFilterKernel{1})(i, j, k, grid, ::Val{width}, policy, f::Function, fargs...) where {width}
    Nx = size(grid, 1); loc = kern.loc
    x₀ = xnode(i, j, k, grid, loc...); σ = kern.σ
    s = zero(grid); w_sum = zero(grid)
    @unroll_full for idx in 1:(2*width+1)
        Δi = idx - width - 1
        x, Δx = x_node_geometry(policy, grid, loc, i, j, k, i + Δi, Nx, kern.period)
        w = Δx * exp(-(x - x₀)^2 / (2σ^2))
        val, cnt = x_stencil_call(policy, f, i + Δi, j, k, Nx, grid, fargs...)
        s += w * val
        w_sum += w * cnt
    end
    return s / w_sum
end

@inline function (kern::StretchedGaussianFilterKernel{2})(i, j, k, grid, ::Val{width}, policy, f::Function, fargs...) where {width}
    Ny = size(grid, 2); loc = kern.loc
    y₀ = ynode(i, j, k, grid, loc...); σ = kern.σ
    s = zero(grid); w_sum = zero(grid)
    @unroll_full for idx in 1:(2*width+1)
        Δj = idx - width - 1
        y, Δy = y_node_geometry(policy, grid, loc, i, j, k, j + Δj, Ny, kern.period)
        w = Δy * exp(-(y - y₀)^2 / (2σ^2))
        val, cnt = y_stencil_call(policy, f, i, j + Δj, k, Ny, grid, fargs...)
        s += w * val
        w_sum += w * cnt
    end
    return s / w_sum
end

@inline function (kern::StretchedGaussianFilterKernel{3})(i, j, k, grid, ::Val{width}, policy, f::Function, fargs...) where {width}
    Nz = size(grid, 3); loc = kern.loc
    z₀ = znode(i, j, k, grid, loc...); σ = kern.σ
    s = zero(grid); w_sum = zero(grid)
    @unroll_full for idx in 1:(2*width+1)
        Δk = idx - width - 1
        z, Δz = z_node_geometry(policy, grid, loc, i, j, k, k + Δk, Nz, kern.period)
        w = Δz * exp(-(z - z₀)^2 / (2σ^2))
        val, cnt = z_stencil_call(policy, f, i, j, k + Δk, Nz, grid, fargs...)
        s += w * val
        w_sum += w * cnt
    end
    return s / w_sum
end
#---

gaussian_weights(width, σ) = ntuple(idx -> exp(-(idx - width - 1)^2 / (2σ^2)), 2*width + 1)

validate_σ(σ::Real) = σ > 0 || throw(ArgumentError("`σ` must be a positive number; got $σ"))
validate_σ(σ) = throw(ArgumentError("`σ` must be a positive number; got $(typeof(σ))"))

direction_min_spacing(grid, d) =
    d == 1 ? minimum_xspacing(grid) :
    d == 2 ? minimum_yspacing(grid) :
             minimum_zspacing(grid)

direction_spacings(grid, d) =
    d == 1 ? xspacings(grid) :
    d == 2 ? yspacings(grid) :
             zspacings(grid)

direction_extent(grid, d) =
    d == 1 ? grid.Lx :
    d == 2 ? grid.Ly :
             grid.Lz

# A direction is "uniform" when every spacing along it is identical. Uniform
# directions take the fast precomputed-weights path (`GaussianFilterKernel`);
# variably spaced directions take the on-the-fly node-distance path
# (`StretchedGaussianFilterKernel`). The check is done once per direction at
# construction time, so it never touches the per-cell hot loop.
function direction_is_uniform(grid, d)
    sp_min, sp_max = extrema(direction_spacings(grid, d))
    return sp_min == sp_max
end

# Build the 1D kernel for filtered direction `d` (the `i`-th entry of
# `sorted_dims`). Uniform directions get a `GaussianFilterKernel` carrying
# weights precomputed in cell-offset units (`σ_cells = σ / Δ`); variably spaced
# directions get a `StretchedGaussianFilterKernel` that evaluates weights from
# physical node distances at run time.
function gaussian_kernel(grid, loc, σT, d, width)
    if direction_is_uniform(grid, d)
        σ_cells = σT / direction_min_spacing(grid, d)
        return GaussianFilterKernel{d}(gaussian_weights(width, σ_cells))
    else
        loc_instances = map(ℓ -> ℓ(), loc)                 # (Center, Center, Face) types → instances
        period = convert(eltype(grid), direction_extent(grid, d))
        return StretchedGaussianFilterKernel{d}(σT, loc_instances, period)
    end
end

"""
    $(SIGNATURES)

Return a `KernelFunctionOperation` that computes a Gaussian-weighted local average of `ψ` over the
directions listed in `dims`.

`σ` is the standard deviation of the Gaussian kernel in physical units (the same units as the grid
spacing). The filter approximates the continuous Gaussian convolution
`∫ G_σ(x-x') ψ(x') dx' / ∫ G_σ(x-x') dx'`: a stencil cell of width `Δₘ` whose centre sits a
physical distance `r` from the current cell centre contributes weight `Δₘ · exp(-r² / 2σ²)`, and
the result is normalized by the sum of the surviving weights, so all boundary policies behave
consistently. On a uniform direction the `Δₘ` factor is constant and cancels, recovering the plain
`exp(-r² / 2σ²)` weighting.

`GaussianFilter` supports **both uniformly and variably spaced (stretched) directions**, choosing
the implementation per direction at construction time so the regular-grid case keeps its original
speed:

  - **Uniform direction** — the weights are identical for every cell, so they are precomputed once
    in cell-offset units (`σ_cells = σ / Δ`, weight `exp(-Δi² / 2σ_cells²)` at cell offset `Δi`)
    and looked up in the hot loop. This is the fast path used on regular grids.
  - **Stretched direction** — the physical footprint of a fixed cell offset varies from cell to
    cell, so the weights cannot be precomputed; each is evaluated on the fly from the node
    coordinates and widths (`Δₘ · exp(-(xₘ - xᵢ)² / 2σ²)`). The cell-width factor `Δₘ` is the
    quadrature weight of the continuous convolution; it stops the average from being biased toward
    finely resolved regions and keeps constants (and, to quadrature accuracy, linear fields)
    preserved. The two paths agree exactly where they overlap (a uniform direction has
    `xₘ - xᵢ = Δm·Δ` and constant `Δₘ`). Directions are decided independently, so a grid that is
    uniform in `x` but stretched in `z` uses the fast path for `x` and the node-distance path for
    `z`.

`N` is the **total number of grid points used by the filter stencil** along each filtered
direction — i.e. how many cells contribute to a single filtered output value. `N` must be an
**odd integer ≥ 3** so the stencil is symmetric around the current cell. (This is the size of the
filter stencil — *not* the size of the grid.) If unspecified, `N` is inferred per-direction from
`σ` and the *smallest* spacing along that direction so that the stencil extends at least `2σ` to
each side of the current cell everywhere (on a stretched direction it therefore reaches farther
than `2σ` where cells are large, which is harmless since the Gaussian weights there are tiny). To
override, pass either a single odd integer (applied to every filtered dim) or a tuple with one
odd-integer count per dim in `dims` (in the order the user passed them). For `Periodic`
directions the stencil must span at most one period (`N ≤ 2*Nd_grid + 1`, where `Nd_grid` is the
number of cells along that direction); this is enforced at construction time.

See `BoxFilter` for the `dims` and `boundary` keyword documentation.

## Performance notes

A multi-direction Gaussian filter is mathematically separable. The constructor still returns a
single composable `KernelFunctionOperation`, but when that operation is the operand of a `Field`
(the standard `Field(GaussianFilter(...))` / `compute!` path), the implementation evaluates the
filter as a sequence of 1D passes through intermediate fields. This reduces the per-output read
count from `N^d` to `d × N`, which is the main reason multi-direction filters with wide stencils
are competitive on GPUs. Mixed-spacing filters stage the same way — each direction's 1D pass uses
its own (uniform or stretched) kernel. If the filter is composed into another `AbstractOperation`
(e.g. `2 * GaussianFilter(c; dims=(1,2,3))`) it falls back to the fused, single-kernel evaluation.

## Examples

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(8, 8), x=(0, 1), z=(0, 1),
                              topology=(Periodic, Flat, Bounded));

julia> c = CenterField(grid);

julia> GaussianFilter(c; dims=(1, 3), σ=0.1) isa KernelFunctionOperation
true
```
"""
function GaussianFilter(ψ; dims, σ, N=nothing, boundary=:shrink)
    validate_σ(σ)
    grid, loc, sorted_dims, policies = resolve_filter_policies(ψ, dims, boundary)

    sorted_widths = resolve_gaussian_widths(N, σ, grid, dims, sorted_dims)
    validate_periodic_widths(grid, sorted_dims, policies, sorted_widths)
    σT = convert(eltype(grid), σ)

    return build_filter_kfo((d, i) -> gaussian_kernel(grid, loc, σT, d, sorted_widths[i]),
                            grid, loc, sorted_dims, sorted_widths, policies, ψ)
end

#+++ Reusable (field-less) Gaussian filter
"""
    GaussianFilterOperator{D, S, NN, B}

A reusable, field-less Gaussian filter. Stores the `GaussianFilter` parameters
(`dims`, `σ`, `N`, `boundary`) and, when called on a field `ψ`, returns
`GaussianFilter(ψ; dims, σ, N, boundary)` — the very same
`KernelFunctionOperation` that the field-first constructor would build.
Construct one once with [`GaussianFilter`](@ref)`(; …)` and apply it to many
fields.
"""
struct GaussianFilterOperator{D, S, NN, B}
    dims::D
    σ::S
    N::NN
    boundary::B
end

(F::GaussianFilterOperator)(ψ) = GaussianFilter(ψ; dims=F.dims, σ=F.σ, N=F.N, boundary=F.boundary)

"""
    GaussianFilter(; dims, σ, N=nothing, boundary=:shrink)

Build a reusable [`GaussianFilterOperator`](@ref) capturing the Gaussian-filter
parameters without binding them to a field. The returned object is callable:
`F(ψ)` returns `GaussianFilter(ψ; dims, σ, N, boundary)`. Useful for applying the
same filter to many fields or for passing a preconfigured filter to other
diagnostics.

See the field-first [`GaussianFilter`](@ref)`(ψ; …)` method for the meaning of
the keyword arguments.
"""
GaussianFilter(; dims, σ, N=nothing, boundary=:shrink) = GaussianFilterOperator(dims, σ, N, boundary)

Base.show(io::IO, F::GaussianFilterOperator) =
    print(io, "GaussianFilter(dims=", F.dims, ", σ=", F.σ, ", boundary=", repr(F.boundary), ")")
#---

infer_width(σ, grid, d) = ceil(Int, 2σ / direction_min_spacing(grid, d))

function resolve_gaussian_widths(N, σ, grid, dims, sorted_dims)
    if N === nothing
        return ntuple(i -> infer_width(σ, grid, sorted_dims[i]), length(sorted_dims))
    elseif N isa Tuple
        error_message = "`N` tuple must have one entry per dim in `dims`; got length $(length(N)) for dims=$dims"
        length(N) == length(dims) || throw(ArgumentError(error_message))
        foreach(validate_N, N)
        return ntuple(i -> begin
            user_idx = findfirst(==(sorted_dims[i]), dims)
            (N[user_idx] - 1) ÷ 2
        end, length(sorted_dims))
    else
        validate_N(N)
        return ntuple(_ -> (N - 1) ÷ 2, length(sorted_dims))
    end
end
#---

#+++ Staged multi-direction evaluation
#
# Multi-direction GaussianFilters are evaluated via the shared
# `_compute_staged_filter!` machinery defined in `Filters.jl`. The aliases
# below pin the dispatch — 1D filters (`length(args) == 3`) fall through to
# the default `compute!` and use the unrolled single-direction kernel.
# Match a multi-direction GaussianFilter regardless of whether each direction's
# 1D kernel is the uniform (`GaussianFilterKernel`) or stretched
# (`StretchedGaussianFilterKernel`) variant — a mixed-spacing filter (e.g.
# uniform x, stretched z) carries one of each. Pinning on the shared supertype
# `AbstractGaussianFilterKernel` lets all combinations stage through the
# separable `d × N` path.
const _GaussianFilter2D = KernelFunctionOperation{LX, LY, LZ, G, T,
                                                  <:AbstractGaussianFilterKernel,
                                                  <:Tuple{Val, AbstractBoundaryPolicy,
                                                          AbstractGaussianFilterKernel, Val, AbstractBoundaryPolicy,
                                                          Any}} where {LX, LY, LZ, G, T}

const _GaussianFilter3D = KernelFunctionOperation{LX, LY, LZ, G, T,
                                                  <:AbstractGaussianFilterKernel,
                                                  <:Tuple{Val, AbstractBoundaryPolicy,
                                                          AbstractGaussianFilterKernel, Val, AbstractBoundaryPolicy,
                                                          AbstractGaussianFilterKernel, Val, AbstractBoundaryPolicy,
                                                          Any}} where {LX, LY, LZ, G, T}

compute!(comp::Field{<:Any, <:Any, <:Any, <:_GaussianFilter2D}, time=nothing) = _compute_staged_filter!(comp, time)
compute!(comp::Field{<:Any, <:Any, <:Any, <:_GaussianFilter3D}, time=nothing) = _compute_staged_filter!(comp, time)
#---
