module Filters
using DocStringExtensions

export BoxFilter, GaussianFilter

using Oceananigans: location
using Oceananigans.Grids: topology, Periodic,
                          minimum_xspacing, minimum_yspacing, minimum_zspacing,
                          xspacings, yspacings, zspacings,
                          xnode, ynode, znode
using Oceananigans.Operators: xspacing, yspacing, zspacing
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

#+++ Shared staged-compute infrastructure
#
# Both `BoxFilter` and `GaussianFilter` are separable: a multi-direction
# filter equals a sequence of 1D passes. The single fused
# `KernelFunctionOperation` built by `build_filter_kfo` evaluates `Nᵈ`
# stencil points per output cell; staging through `d` intermediate fields
# evaluates `d × N`. For each filter we override
# `Oceananigans.Fields.compute!` on `Field{<:Any,<:Any,<:Any,<:_FilterND}`
# so the standard `Field(filter)` path picks up the staged evaluation. When
# the filter is *nested* inside another `AbstractOperation` (e.g.
# `2 * BoxFilter(c; dims=(1,2))`) the override doesn't match and the original
# inlined-fused kernel runs.
#
# The machinery in this section is filter-agnostic — it walks the KFO's
# `arguments` tuple by length (3 → 1D, 6 → 2D, 9 → 3D), which is the shape
# produced by `build_filter_kfo` above. The kernel-specific files just
# define type aliases and attach the dispatch.

# A self-contained "fully unroll this loop" macro. This is the same pattern
# as `KernelAbstractions.Extras.@unroll`: attach the LLVM
# `llvm.loop.unroll.full` loopinfo node to the body so the optimizer is
# required to unroll the loop. Done inline so the `Filters` submodule does
# not need to add `KernelAbstractions` as a direct dependency.
macro unroll_full(expr)
    expr.head === :for || error("@unroll_full needs a `for` loop")
    i, iter = expr.args[1].args
    body = expr.args[2]
    return esc(quote
        for $i in $iter
            $body
            $(Expr(:loopinfo, (Symbol("llvm.loop.unroll.full"), 1)))
        end
    end)
end

import Oceananigans.Fields: compute!
using Oceananigans.Fields: Field, offset_index, set_status!
using Oceananigans.AbstractOperations: compute_at!, _compute!
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Architectures: architecture
using Oceananigans.Utils: KernelParameters, launch!

# Build a single-direction KFO for one stage of the chain at the filter's
# `loc`. `kern` is the 1D filter kernel (`BoxFilterKernel{D}` or
# `GaussianFilterKernel{D}`).
@inline _single_dim_kfo(loc, grid, kern, valw, pol, input) =
    KernelFunctionOperation{loc...}(kern, grid, valw, pol, input)

# Run `kfo` over its iteration space and write the result into `data`.
# Reuses Oceananigans' trivial copy kernel `_compute!`
# (`data[i,j,k] = operand[i,j,k]`).
function _launch_compute_into!(data, indices, grid, kfo)
    arch = architecture(grid)
    params = KernelParameters(size(kfo), map(offset_index, indices))
    launch!(arch, grid, params, _compute!, data, kfo)
    return nothing
end

# Allocate an intermediate Field and compute the 1D pass into it *without*
# filling halo regions. The next pass reads only the interior of the
# intermediate (every `*_stencil_fetch` / `*_stencil_call` clamps or wraps
# offsets into `1:N`), so halo data is irrelevant. Skipping the halo fill
# saves several small kernel launches per intermediate.
function _stage_into_temp(loc, grid, kern, valw, pol, input)
    kfo = _single_dim_kfo(loc, grid, kern, valw, pol, input)
    temp = Field(kfo, compute=false)
    _launch_compute_into!(temp.data, temp.indices, grid, kfo)
    return temp
end

# Generic staged compute for a 2D or 3D separable filter. Both `BoxFilter`
# and `GaussianFilter` share this body — the only filter-specific things are
# the type aliases that pin the dispatch, defined in each filter's file.
function _compute_staged_filter!(comp, time)
    op    = comp.operand
    grid  = op.grid
    loc   = location(op)
    args  = op.arguments
    kern1 = op.kernel_function

    # If ψ is itself a computed field that needs refreshing, do that first.
    ψ = args[end]
    compute_at!(ψ, time)

    if length(args) == 6        # 2D filter
        valw1, pol1        = args[1], args[2]
        kern2, valw2, pol2 = args[3], args[4], args[5]

        temp1 = _stage_into_temp(loc, grid, kern1, valw1, pol1, ψ)
        final = _single_dim_kfo(loc, grid, kern2, valw2, pol2, temp1)
    else                        # 3D filter, length(args) == 9
        valw1, pol1        = args[1], args[2]
        kern2, valw2, pol2 = args[3], args[4], args[5]
        kern3, valw3, pol3 = args[6], args[7], args[8]

        temp1 = _stage_into_temp(loc, grid, kern1, valw1, pol1, ψ)
        temp2 = _stage_into_temp(loc, grid, kern2, valw2, pol2, temp1)
        final = _single_dim_kfo(loc, grid, kern3, valw3, pol3, temp2)
    end

    _launch_compute_into!(comp.data, comp.indices, grid, final)
    fill_halo_regions!(comp)
    set_status!(comp.status, time)
    return comp
end
#---

include("box_filter.jl")
include("gaussian_filter.jl")

end # module
