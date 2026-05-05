module Filters
using DocStringExtensions

export BoxFilter, GaussianFilter

using Oceananigans: location
using Oceananigans.Grids: topology, Periodic,
                          minimum_xspacing, minimum_yspacing, minimum_zspacing,
                          xspacings, yspacings, zspacings
using Oceananigans.AbstractOperations: KernelFunctionOperation

using Oceanostics: CustomKFO

#+++ Boundary policies
"""
    AbstractBoundaryPolicy

Supertype for per-direction boundary handling. A policy tells the kernel how to
treat stencil offsets that fall outside the interior range `1:N` along a
direction. `Periodic` directions always use `PeriodicBoundary` (silently
overriding any user choice); `Bounded` directions use the user's pick.
"""
abstract type AbstractBoundaryPolicy end

"""Wrap out-of-bounds offsets to the interior, modulo `N`. Used automatically
for every `Periodic` direction, regardless of the user's `boundary` choice."""
struct PeriodicBoundary <: AbstractBoundaryPolicy end

"""Drop out-of-bounds offsets from *both* the sum and the count, giving an
honest local average over whatever interior cells the stencil actually covers.
The effective stencil shrinks near a wall."""
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
#---

#+++ Stencil value readers
#
# Read a single stencil-offset value while applying the boundary policy along
# one direction. Each function returns `(value, count)`: `count` is 1 for every
# policy except `ShrinkBoundary` out-of-bounds, where it is 0 so the offset is
# excluded from the running mean.
#
# `x/y/z_stencil_fetch` read from an indexable field `ψ`.
# `x/y/z_stencil_call`  evaluate a kernel function `f` at the adjusted index.

@inline wrap_periodic_index(i, N) = i + N * (i < 1) - N * (i > N)

@inline x_stencil_fetch(::PeriodicBoundary, ψ, i, j, k, N) = (@inbounds ψ[wrap_periodic_index(i, N), j, k], 1)
@inline y_stencil_fetch(::PeriodicBoundary, ψ, i, j, k, N) = (@inbounds ψ[i, wrap_periodic_index(j, N), k], 1)
@inline z_stencil_fetch(::PeriodicBoundary, ψ, i, j, k, N) = (@inbounds ψ[i, j, wrap_periodic_index(k, N)], 1)

@inline x_stencil_fetch(::EdgeBoundary, ψ, i, j, k, N) = (@inbounds ψ[clamp(i, 1, N), j, k], 1)
@inline y_stencil_fetch(::EdgeBoundary, ψ, i, j, k, N) = (@inbounds ψ[i, clamp(j, 1, N), k], 1)
@inline z_stencil_fetch(::EdgeBoundary, ψ, i, j, k, N) = (@inbounds ψ[i, j, clamp(k, 1, N)], 1)

@inline x_stencil_fetch(c::ConstantBoundary, ψ, i, j, k, N) = (ifelse(i < 1, c.left, ifelse(i > N, c.right, @inbounds ψ[clamp(i, 1, N), j, k])), 1)
@inline y_stencil_fetch(c::ConstantBoundary, ψ, i, j, k, N) = (ifelse(j < 1, c.left, ifelse(j > N, c.right, @inbounds ψ[i, clamp(j, 1, N), k])), 1)
@inline z_stencil_fetch(c::ConstantBoundary, ψ, i, j, k, N) = (ifelse(k < 1, c.left, ifelse(k > N, c.right, @inbounds ψ[i, j, clamp(k, 1, N)])), 1)

@inline function x_stencil_fetch(::ShrinkBoundary, ψ, i, j, k, N)
    in_bounds = (1 <= i) & (i <= N)
    return ifelse(in_bounds, @inbounds(ψ[clamp(i, 1, N), j, k]), zero(eltype(ψ))), Int(in_bounds)
end
@inline function y_stencil_fetch(::ShrinkBoundary, ψ, i, j, k, N)
    in_bounds = (1 <= j) & (j <= N)
    return ifelse(in_bounds, @inbounds(ψ[i, clamp(j, 1, N), k]), zero(eltype(ψ))), Int(in_bounds)
end
@inline function z_stencil_fetch(::ShrinkBoundary, ψ, i, j, k, N)
    in_bounds = (1 <= k) & (k <= N)
    return ifelse(in_bounds, @inbounds(ψ[i, j, clamp(k, 1, N)]), zero(eltype(ψ))), Int(in_bounds)
end

@inline x_stencil_call(::PeriodicBoundary, f, i, j, k, N, grid, fargs...) = (f(wrap_periodic_index(i, N), j, k, grid, fargs...), 1)
@inline y_stencil_call(::PeriodicBoundary, f, i, j, k, N, grid, fargs...) = (f(i, wrap_periodic_index(j, N), k, grid, fargs...), 1)
@inline z_stencil_call(::PeriodicBoundary, f, i, j, k, N, grid, fargs...) = (f(i, j, wrap_periodic_index(k, N), grid, fargs...), 1)

@inline x_stencil_call(::EdgeBoundary, f, i, j, k, N, grid, fargs...) = (f(clamp(i, 1, N), j, k, grid, fargs...), 1)
@inline y_stencil_call(::EdgeBoundary, f, i, j, k, N, grid, fargs...) = (f(i, clamp(j, 1, N), k, grid, fargs...), 1)
@inline z_stencil_call(::EdgeBoundary, f, i, j, k, N, grid, fargs...) = (f(i, j, clamp(k, 1, N), grid, fargs...), 1)

@inline x_stencil_call(c::ConstantBoundary, f, i, j, k, N, grid, fargs...) = (ifelse(i < 1, c.left, ifelse(i > N, c.right, f(clamp(i, 1, N), j, k, grid, fargs...))), 1)
@inline y_stencil_call(c::ConstantBoundary, f, i, j, k, N, grid, fargs...) = (ifelse(j < 1, c.left, ifelse(j > N, c.right, f(i, clamp(j, 1, N), k, grid, fargs...))), 1)
@inline z_stencil_call(c::ConstantBoundary, f, i, j, k, N, grid, fargs...) = (ifelse(k < 1, c.left, ifelse(k > N, c.right, f(i, j, clamp(k, 1, N), grid, fargs...))), 1)

@inline function x_stencil_call(::ShrinkBoundary, f, i, j, k, N, grid, fargs...)
    in_bounds = (1 <= i) & (i <= N)
    return ifelse(in_bounds, f(clamp(i, 1, N), j, k, grid, fargs...), zero(grid)), Int(in_bounds)
end
@inline function y_stencil_call(::ShrinkBoundary, f, i, j, k, N, grid, fargs...)
    in_bounds = (1 <= j) & (j <= N)
    return ifelse(in_bounds, f(i, clamp(j, 1, N), k, grid, fargs...), zero(grid)), Int(in_bounds)
end
@inline function z_stencil_call(::ShrinkBoundary, f, i, j, k, N, grid, fargs...)
    in_bounds = (1 <= k) & (k <= N)
    return ifelse(in_bounds, f(i, j, clamp(k, 1, N), grid, fargs...), zero(grid)), Int(in_bounds)
end
#---

#+++ Shared filter infrastructure
function resolve_filter_policies(ψ, dims, boundary)
    validate_dims(dims)

    grid = ψ.grid
    loc = location(ψ)

    per_user_dim_specs = if boundary isa Tuple
        error_message = "`boundary` must be a single spec or a tuple with one entry per dim in `dims`; got length $(length(boundary)) for dims=$dims"
        length(boundary) == length(dims) || throw(ArgumentError(error_message))
        boundary
    else
        ntuple(_ -> boundary, length(dims))
    end

    foreach(parse_boundary_spec, per_user_dim_specs)

    sorted_dims = Tuple(d for d in (1, 2, 3) if d in dims)
    sorted_specs = ntuple(i -> begin
        user_idx = findfirst(==(sorted_dims[i]), dims)
        per_user_dim_specs[user_idx]
    end, length(sorted_dims))

    policies = ntuple(i -> begin
        d = sorted_dims[i]
        if topology(grid, d) === Periodic
            PeriodicBoundary()
        else
            parse_boundary_spec(sorted_specs[i])
        end
    end, length(sorted_dims))

    return grid, loc, sorted_dims, policies
end

# `make_kernel(d, i)` takes both the grid direction `d` and the index `i` of
# that direction within `sorted_dims`, so kernels can pick up per-direction
# state (e.g. precomputed weights) by index. `widths` is a tuple with one
# entry per filtered dim (also in `sorted_dims` order).
function build_filter_kfo(make_kernel, grid, loc, dims::Tuple{Int}, widths, policies, ψ)
    d = dims[1]
    return KernelFunctionOperation{loc...}(make_kernel(d, 1), grid,
                                           Val(widths[1]), policies[1], ψ)
end

function build_filter_kfo(make_kernel, grid, loc, dims::NTuple{2, Int}, widths, policies, ψ)
    d1, d2 = dims
    return KernelFunctionOperation{loc...}(make_kernel(d1, 1), grid,
                                           Val(widths[1]), policies[1],
                                           make_kernel(d2, 2), Val(widths[2]), policies[2],
                                           ψ)
end

function build_filter_kfo(make_kernel, grid, loc, dims::NTuple{3, Int}, widths, policies, ψ)
    d1, d2, d3 = dims
    return KernelFunctionOperation{loc...}(make_kernel(d1, 1), grid,
                                           Val(widths[1]), policies[1],
                                           make_kernel(d2, 2), Val(widths[2]), policies[2],
                                           make_kernel(d3, 3), Val(widths[3]), policies[3],
                                           ψ)
end

#---

#+++ Validation
validate_dims(dims::Tuple{Vararg{Int}}) =
    (!isempty(dims) & all(d -> d in (1, 2, 3), dims) & allunique(dims)) ||
        throw(ArgumentError("`dims` must be a non-empty tuple of distinct integers drawn from (1, 2, 3); got $dims"))

validate_dims(dims) = throw(ArgumentError("`dims` must be a tuple of integers; got $(typeof(dims))"))

validate_N(N::Integer) = ((N >= 3) & isodd(N)) || throw(ArgumentError("`N` must be an odd integer ≥ 3; got $N"))
validate_N(N) = throw(ArgumentError("`N` must be an odd integer ≥ 3; got $(typeof(N))"))

function validate_periodic_widths(grid, sorted_dims, policies, widths)
    for (i, (d, policy)) in enumerate(zip(sorted_dims, policies))
        if policy isa PeriodicBoundary
            Nd_grid = size(grid, d)
            N = 2 * widths[i] + 1
            N <= 2 * Nd_grid + 1 ||
                throw(ArgumentError("for the periodic direction d=$d, `N` ($N) exceeds the maximum allowed 2*Nd_grid+1 = $(2*Nd_grid+1) (this direction has $Nd_grid cells); the periodic wrapping assumes the stencil spans at most one period"))
        end
    end
end

parse_boundary_spec(s::Symbol) =
    s === :shrink ? ShrinkBoundary() :
    s === :edge   ? EdgeBoundary()   :
    throw(ArgumentError("`boundary` symbol must be :shrink or :edge; got :$s"))

function parse_boundary_spec(nt::NamedTuple)
    ((length(nt) == 2) & haskey(nt, :left) & haskey(nt, :right)) ||
        throw(ArgumentError("`boundary` NamedTuple must have exactly keys `:left` and `:right`; got keys $(keys(nt))"))
    return ConstantBoundary(nt.left, nt.right)
end

parse_boundary_spec(p::AbstractBoundaryPolicy) = p
parse_boundary_spec(x) = throw(ArgumentError("`boundary` must be :shrink, :edge, or (left=a, right=b); got $(repr(x))"))
#---

include("box_filter.jl")
include("gaussian_filter.jl")

end # module
