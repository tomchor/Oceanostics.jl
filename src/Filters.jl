module Filters
using DocStringExtensions

export BoxFilter

using Oceananigans: location
using Oceananigans.Grids: topology, Periodic
using Oceananigans.AbstractOperations: KernelFunctionOperation

using Oceanostics: CustomKFO

# -------- Boundary policies --------

"""
    AbstractBoundaryPolicy

Supertype for `BoxFilter`'s per-direction boundary handling. A policy tells the
kernel how to treat stencil offsets that fall outside the interior range `1:N`
along a direction. `Periodic` directions always use `PeriodicBoundary`
(silently overriding any user choice); `Bounded` directions use the user's
pick.
"""
abstract type AbstractBoundaryPolicy end

"""Wrap out-of-bounds offsets to the interior via `mod1`. Used automatically
for every `Periodic` direction, regardless of the user's `boundary` choice."""
struct PeriodicBoundary <: AbstractBoundaryPolicy end

"""Drop out-of-bounds offsets from *both* the sum and the count, giving an
honest local average over whatever interior cells the stencil actually covers.
The effective stencil shrinks within `width` cells of a wall."""
struct ShrinkBoundary <: AbstractBoundaryPolicy end

"""Replicate the boundary-cell value: an offset past either end reads the
nearest interior cell (`ψ[1]` on the low side, `ψ[N]` on the high side)."""
struct EdgeBoundary <: AbstractBoundaryPolicy end

"""
    ConstantBoundary(left, right)

Pad the field outside the interior with `left` on the low-index side and
`right` on the high-index side. `left` and `right` are promoted to a common
type.
"""
struct ConstantBoundary{T} <: AbstractBoundaryPolicy
    left::T
    right::T
    ConstantBoundary{T}(l, r) where {T} = new{T}(l, r)
end
ConstantBoundary(left, right) = ConstantBoundary{promote_type(typeof(left), typeof(right))}(promote(left, right)...)

# -------- Fetch / call dispatchers --------
#
# `_fetchD(policy, ψ, ii, jj, kk, N)` returns `(value, count)` when `ψ` is an
# indexable field; `_callD(policy, f, ii, jj, kk, N, grid, fargs...)` is the
# same thing when the inner value comes from another kernel function. `count`
# is 1 in every case except `ShrinkBoundary` out-of-bounds, where it is 0 so
# the offset is excluded from the running mean.

@inline _fetch1(::PeriodicBoundary, ψ, ii, jj, kk, N) = (ψ[mod1(ii, N), jj, kk], 1)
@inline _fetch2(::PeriodicBoundary, ψ, ii, jj, kk, N) = (ψ[ii, mod1(jj, N), kk], 1)
@inline _fetch3(::PeriodicBoundary, ψ, ii, jj, kk, N) = (ψ[ii, jj, mod1(kk, N)], 1)

@inline _fetch1(::EdgeBoundary, ψ, ii, jj, kk, N) = (ψ[clamp(ii, 1, N), jj, kk], 1)
@inline _fetch2(::EdgeBoundary, ψ, ii, jj, kk, N) = (ψ[ii, clamp(jj, 1, N), kk], 1)
@inline _fetch3(::EdgeBoundary, ψ, ii, jj, kk, N) = (ψ[ii, jj, clamp(kk, 1, N)], 1)

@inline function _fetch1(c::ConstantBoundary, ψ, ii, jj, kk, N)
    ii < 1 && return (c.left,  1)
    ii > N && return (c.right, 1)
    return (ψ[ii, jj, kk], 1)
end
@inline function _fetch2(c::ConstantBoundary, ψ, ii, jj, kk, N)
    jj < 1 && return (c.left,  1)
    jj > N && return (c.right, 1)
    return (ψ[ii, jj, kk], 1)
end
@inline function _fetch3(c::ConstantBoundary, ψ, ii, jj, kk, N)
    kk < 1 && return (c.left,  1)
    kk > N && return (c.right, 1)
    return (ψ[ii, jj, kk], 1)
end

@inline _fetch1(::ShrinkBoundary, ψ, ii, jj, kk, N) =
    (1 <= ii <= N) ? (ψ[ii, jj, kk], 1) : (zero(eltype(ψ)), 0)
@inline _fetch2(::ShrinkBoundary, ψ, ii, jj, kk, N) =
    (1 <= jj <= N) ? (ψ[ii, jj, kk], 1) : (zero(eltype(ψ)), 0)
@inline _fetch3(::ShrinkBoundary, ψ, ii, jj, kk, N) =
    (1 <= kk <= N) ? (ψ[ii, jj, kk], 1) : (zero(eltype(ψ)), 0)

@inline _call1(::PeriodicBoundary, f, ii, jj, kk, N, grid, fargs...) =
    (f(mod1(ii, N), jj, kk, grid, fargs...), 1)
@inline _call2(::PeriodicBoundary, f, ii, jj, kk, N, grid, fargs...) =
    (f(ii, mod1(jj, N), kk, grid, fargs...), 1)
@inline _call3(::PeriodicBoundary, f, ii, jj, kk, N, grid, fargs...) =
    (f(ii, jj, mod1(kk, N), grid, fargs...), 1)

@inline _call1(::EdgeBoundary, f, ii, jj, kk, N, grid, fargs...) =
    (f(clamp(ii, 1, N), jj, kk, grid, fargs...), 1)
@inline _call2(::EdgeBoundary, f, ii, jj, kk, N, grid, fargs...) =
    (f(ii, clamp(jj, 1, N), kk, grid, fargs...), 1)
@inline _call3(::EdgeBoundary, f, ii, jj, kk, N, grid, fargs...) =
    (f(ii, jj, clamp(kk, 1, N), grid, fargs...), 1)

@inline function _call1(c::ConstantBoundary, f, ii, jj, kk, N, grid, fargs...)
    ii < 1 && return (c.left,  1)
    ii > N && return (c.right, 1)
    return (f(ii, jj, kk, grid, fargs...), 1)
end
@inline function _call2(c::ConstantBoundary, f, ii, jj, kk, N, grid, fargs...)
    jj < 1 && return (c.left,  1)
    jj > N && return (c.right, 1)
    return (f(ii, jj, kk, grid, fargs...), 1)
end
@inline function _call3(c::ConstantBoundary, f, ii, jj, kk, N, grid, fargs...)
    kk < 1 && return (c.left,  1)
    kk > N && return (c.right, 1)
    return (f(ii, jj, kk, grid, fargs...), 1)
end

@inline _call1(::ShrinkBoundary, f, ii, jj, kk, N, grid, fargs...) =
    (1 <= ii <= N) ? (f(ii, jj, kk, grid, fargs...), 1) : (zero(grid), 0)
@inline _call2(::ShrinkBoundary, f, ii, jj, kk, N, grid, fargs...) =
    (1 <= jj <= N) ? (f(ii, jj, kk, grid, fargs...), 1) : (zero(grid), 0)
@inline _call3(::ShrinkBoundary, f, ii, jj, kk, N, grid, fargs...) =
    (1 <= kk <= N) ? (f(ii, jj, kk, grid, fargs...), 1) : (zero(grid), 0)

# -------- Kernel struct + methods --------

"""
    BoxFilterKernel{D} <: Function

Callable singleton that computes a 1D box average along direction `D` (1, 2, or 3).
Has two methods: a terminal one that indexes into an indexable input, and a
recursive one that invokes another kernel function at each stencil point.
"""
struct BoxFilterKernel{D} <: Function end

# Terminal methods (indexable input).

@inline function (::BoxFilterKernel{1})(i, j, k, grid, width, policy, ψ)
    Nx = size(grid, 1)
    s = zero(grid); n = 0
    @inbounds for di in -width:width
        val, cnt = _fetch1(policy, ψ, i + di, j, k, Nx)
        s += val; n += cnt
    end
    return s / n
end

@inline function (::BoxFilterKernel{2})(i, j, k, grid, width, policy, ψ)
    Ny = size(grid, 2)
    s = zero(grid); n = 0
    @inbounds for dj in -width:width
        val, cnt = _fetch2(policy, ψ, i, j + dj, k, Ny)
        s += val; n += cnt
    end
    return s / n
end

@inline function (::BoxFilterKernel{3})(i, j, k, grid, width, policy, ψ)
    Nz = size(grid, 3)
    s = zero(grid); n = 0
    @inbounds for dk in -width:width
        val, cnt = _fetch3(policy, ψ, i, j, k + dk, Nz)
        s += val; n += cnt
    end
    return s / n
end

# Recursive methods (function input — typically another BoxFilterKernel).

@inline function (::BoxFilterKernel{1})(i, j, k, grid, width, policy, f::Function, fargs...)
    Nx = size(grid, 1)
    s = zero(grid); n = 0
    @inbounds for di in -width:width
        val, cnt = _call1(policy, f, i + di, j, k, Nx, grid, fargs...)
        s += val; n += cnt
    end
    return s / n
end

@inline function (::BoxFilterKernel{2})(i, j, k, grid, width, policy, f::Function, fargs...)
    Ny = size(grid, 2)
    s = zero(grid); n = 0
    @inbounds for dj in -width:width
        val, cnt = _call2(policy, f, i, j + dj, k, Ny, grid, fargs...)
        s += val; n += cnt
    end
    return s / n
end

@inline function (::BoxFilterKernel{3})(i, j, k, grid, width, policy, f::Function, fargs...)
    Nz = size(grid, 3)
    s = zero(grid); n = 0
    @inbounds for dk in -width:width
        val, cnt = _call3(policy, f, i, j, k + dk, Nz, grid, fargs...)
        s += val; n += cnt
    end
    return s / n
end

const BoxFilter = CustomKFO{<:BoxFilterKernel}

"""
    $(SIGNATURES)

Return a `KernelFunctionOperation` that computes a local box-average of `ψ`
over the directions listed in `dims`.

`dims` is a tuple of distinct integers drawn from `(1, 2, 3)` (where `1`, `2`,
`3` correspond to `x`, `y`, `z`). `width` is the half-width of the stencil in
cells, so each selected direction uses a `(2*width + 1)`-point running mean
centered on the current cell.

A multi-directional filter is assembled as a single `KernelFunctionOperation`
whose kernel function is a 1D `BoxFilterKernel{d₁}`, with the next dimension's
`BoxFilterKernel{d₂}` (and so on) threaded into the argument list. The nested
1D kernels inline into a single fused read pass at compile time.

## Boundary handling

Stencil offsets that leave the interior `1:N` of a direction are handled
per-direction. For `Periodic` directions offsets are always wrapped via
`mod1`, independent of the `boundary` keyword. For `Bounded` directions the
`boundary` keyword picks the policy:

  - `:shrink` (default) — drop out-of-bounds offsets from *both* the sum and
    the count, so the filter is an honest local average whose effective
    stencil shrinks within `width` cells of a wall.
  - `:edge` — replicate the boundary-cell value (reads `ψ[1]` or `ψ[N]` for
    offsets past either end).
  - `(left=a, right=b)` — pad with constant `a` on the low-index side and
    `b` on the high-index side (`a` and `b` are promoted to a common type).

`boundary` may be a single spec applied to every filtered dim, or a tuple
with one entry per dim in `dims` (in the order the user passed them):

    BoxFilter(ψ; dims=(1, 2), width=3, boundary=:edge)
    BoxFilter(ψ; dims=(1, 2), width=3, boundary=(:shrink, :edge))
    BoxFilter(ψ; dims=(1,),   width=3, boundary=(left=0.0, right=1.0))

Because every policy wraps, clamps, or skips indices up front, `halo_size(grid)`
does not constrain `width`: a small halo on a bounded direction is fine. The
output location matches the location of `ψ`, and `ψ` can be any input that
supports the standard Oceananigans `ψ[i, j, k]` indexing contract (e.g. a
`Field` or any `AbstractOperation`).
"""
function BoxFilter(ψ; dims, width, boundary=:shrink)
    validate_dims(dims)
    validate_width(width)

    grid = ψ.grid
    LX, LY, LZ = location(ψ)

    per_user_dim_specs = if boundary isa Tuple
        length(boundary) == length(dims) ||
            throw(ArgumentError("BoxFilter `boundary` must be a single spec or a tuple with one entry per dim in `dims`; got length $(length(boundary)) for dims=$dims"))
        boundary
    else
        ntuple(_ -> boundary, length(dims))
    end

    # Validate every user-provided spec up front so a malformed spec errors
    # immediately, even on a grid where the corresponding dim is `Periodic`
    # (and the spec would otherwise be silently overridden).
    foreach(parse_boundary_spec, per_user_dim_specs)

    # Canonical kernel nesting order (1 → 2 → 3), with boundary specs
    # reordered to match so each dim still gets its intended policy.
    sorted_dims = Tuple(d for d in (1, 2, 3) if d in dims)
    sorted_specs = ntuple(length(sorted_dims)) do i
        user_idx = findfirst(==(sorted_dims[i]), dims)
        per_user_dim_specs[user_idx]
    end

    policies = ntuple(length(sorted_dims)) do i
        d = sorted_dims[i]
        if topology(grid, d) === Periodic
            PeriodicBoundary()
        else
            parse_boundary_spec(sorted_specs[i])
        end
    end

    return build_box_filter_kfo(grid, (LX, LY, LZ), sorted_dims, width, policies, ψ)
end

function build_box_filter_kfo(grid, loc, dims::Tuple{Int}, width, policies, ψ)
    d = dims[1]
    return KernelFunctionOperation{loc...}(BoxFilterKernel{d}(), grid,
                                           width, policies[1], ψ)
end

function build_box_filter_kfo(grid, loc, dims::NTuple{2, Int}, width, policies, ψ)
    d1, d2 = dims
    return KernelFunctionOperation{loc...}(BoxFilterKernel{d1}(), grid,
                                           width, policies[1],
                                           BoxFilterKernel{d2}(), width, policies[2],
                                           ψ)
end

function build_box_filter_kfo(grid, loc, dims::NTuple{3, Int}, width, policies, ψ)
    d1, d2, d3 = dims
    return KernelFunctionOperation{loc...}(BoxFilterKernel{d1}(), grid,
                                           width, policies[1],
                                           BoxFilterKernel{d2}(), width, policies[2],
                                           BoxFilterKernel{d3}(), width, policies[3],
                                           ψ)
end

# -------- Validation --------

validate_dims(dims::Tuple{Vararg{Int}}) =
    (!isempty(dims) && all(d -> d in (1, 2, 3), dims) && allunique(dims)) ||
        throw(ArgumentError("BoxFilter `dims` must be a non-empty tuple of distinct integers drawn from (1, 2, 3); got $dims"))

validate_dims(dims) =
    throw(ArgumentError("BoxFilter `dims` must be a tuple of integers; got $(typeof(dims))"))

validate_width(width::Integer) =
    width >= 1 || throw(ArgumentError("BoxFilter `width` must be a positive integer; got $width"))

validate_width(width) =
    throw(ArgumentError("BoxFilter `width` must be a positive integer; got $(typeof(width))"))

parse_boundary_spec(s::Symbol) =
    s === :shrink ? ShrinkBoundary() :
    s === :edge   ? EdgeBoundary()   :
    throw(ArgumentError("BoxFilter `boundary` symbol must be :shrink or :edge; got :$s"))

function parse_boundary_spec(nt::NamedTuple)
    (length(nt) == 2 && haskey(nt, :left) && haskey(nt, :right)) ||
        throw(ArgumentError("BoxFilter `boundary` NamedTuple must have exactly keys `:left` and `:right`; got keys $(keys(nt))"))
    return ConstantBoundary(nt.left, nt.right)
end

parse_boundary_spec(p::AbstractBoundaryPolicy) = p

parse_boundary_spec(x) =
    throw(ArgumentError("BoxFilter `boundary` must be :shrink, :edge, or (left=a, right=b); got $(repr(x))"))

end # module
