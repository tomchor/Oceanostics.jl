module SpatialFilters
using DocStringExtensions

export BoxFilter, GaussianFilter, check_filter_staging

using Oceananigans: location
using Oceananigans.Grids: topology, Periodic, instantiate,
                          minimum_xspacing, minimum_yspacing, minimum_zspacing,
                          xspacings, yspacings, zspacings,
                          xnode, ynode, znode
using Oceananigans.Operators: xspacing, yspacing, zspacing
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Architectures: Adapt

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

"""
    SizedBoundary(policy, N)

Internal. A boundary policy paired with the operand's extent `N` along the direction it
governs.

The extent has to be attached here, on the host, because it cannot be recovered inside a
kernel. `stencil_length` derives it from `location(ψ)`, and on a GPU the `ψ` a kernel sees is
the *adapted* operand, which carries no location: `location(ψ)[d]` comes back `Nothing`, and
Oceananigans' `length(::Nothing, topo, N)` is `1`. Every filtered direction then looks one
cell long, and the failure is silent in two ways at once — `wrap_periodic_index(i, 1) == i-1`
displaces the whole field by a cell, and `ShrinkBoundary` keeps only the single tap at index
1. Neither is visible on a constant field, which is why CPU-only CI never caught it.
"""
struct SizedBoundary{P <: AbstractBoundaryPolicy, I} <: AbstractBoundaryPolicy
    policy::P
    N::I
end

# The extent the kernels filter over, and the underlying policy for dispatch.
@inline stencil_extent(sb::SizedBoundary) = sb.N
@inline unwrap_policy(sb::SizedBoundary) = sb.policy
@inline unwrap_policy(p::AbstractBoundaryPolicy) = p
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

# Unwrap a `SizedBoundary` so the readers below keep dispatching on the underlying policy.
@inline x_stencil_fetch(sb::SizedBoundary, ψ, i, j, k, N) = x_stencil_fetch(sb.policy, ψ, i, j, k, N)
@inline y_stencil_fetch(sb::SizedBoundary, ψ, i, j, k, N) = y_stencil_fetch(sb.policy, ψ, i, j, k, N)
@inline z_stencil_fetch(sb::SizedBoundary, ψ, i, j, k, N) = z_stencil_fetch(sb.policy, ψ, i, j, k, N)

@inline x_stencil_call(sb::SizedBoundary, f, i, j, k, N, grid, fargs...) = x_stencil_call(sb.policy, f, i, j, k, N, grid, fargs...)
@inline y_stencil_call(sb::SizedBoundary, f, i, j, k, N, grid, fargs...) = y_stencil_call(sb.policy, f, i, j, k, N, grid, fargs...)
@inline z_stencil_call(sb::SizedBoundary, f, i, j, k, N, grid, fargs...) = z_stencil_call(sb.policy, f, i, j, k, N, grid, fargs...)

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

#+++ Operand extent along a filtered direction
# Filtering does not move a field off its location, so a kernel iterates the operand's own index
# space, whose length along direction `d` is Oceananigans' `length(loc, topo, N)`: `N` cells, but
# `N + 1` faces when the direction is `Bounded` (under `Periodic` the wrap identifies face `N + 1`
# with face `1`, so `Face` and `Center` coincide at `N`). `size(grid, d)` alone under-counts by one
# exactly there, which would make every boundary policy treat the last face as out of range:
# `:shrink` would drop the point's own weight at the top face (a bias toward the interior, or `0/0`
# for a stencil whose off-center weights vanish) and `:edge` would read the face below instead. The
# operand is the last argument at every level of the fused recursion, so the recursive kernel
# methods read it from `fargs[end]`; everything here is singleton types, so the lookup constant-folds.
@inline stencil_length(grid, d, ψ) = length(instantiate(location(ψ)[d]), instantiate(topology(grid, d)), size(grid, d))
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

    # Each policy carries the operand's extent along its own direction, measured here where
    # `location(ψ)` is still available. See `SizedBoundary` for why this cannot be deferred
    # to the kernel.
    policies = ntuple(i -> begin
        d = sorted_dims[i]
        base = if topology(grid, d) === Periodic
            PeriodicBoundary()
        else
            parse_boundary_spec(sorted_specs[i])
        end
        SizedBoundary(base, stencil_length(grid, d, ψ))
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

# Normalize a `dims` argument to canonical tuple form: a bare integer is treated as a
# one-element tuple (a single direction), and anything else is passed through unchanged.
tuplefy_dims(dims::Integer) = (Int(dims),)
tuplefy_dims(dims) = dims

validate_N(N::Integer) = ((N >= 3) & isodd(N)) || throw(ArgumentError("`N` must be an odd integer ≥ 3; got $N"))
validate_N(N) = throw(ArgumentError("`N` must be an odd integer ≥ 3; got $(typeof(N))"))

function validate_periodic_widths(grid, sorted_dims, policies, widths)
    for (i, (d, policy)) in enumerate(zip(sorted_dims, policies))
        if unwrap_policy(policy) isa PeriodicBoundary
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
# required to unroll the loop. Done inline so the `SpatialFilters` submodule does
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
using Oceananigans.Fields: Field, AbstractField, offset_index, set_status!
using Oceananigans.AbstractOperations: compute_at!, _compute!
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Architectures: architecture
using Oceananigans.Utils: KernelParameters, launch!

# Build a single-direction KFO for one stage of the chain at the filter's
# `loc`. `kern` is the 1D filter kernel (`BoxFilterKernel{D}` or
# `GaussianFilterKernel{D}`).
@inline _single_dim_kfo(loc, grid, kern, valw, pol, input) =
    KernelFunctionOperation{loc...}(kern, grid, valw, pol, input)

# Evaluate `kfo` into `dest` via Oceananigans' copy kernel. The iteration space comes from
# `dest`, not `kfo`: a windowed `dest` (e.g. an output writer's `indices=(:, :, Nz)` slice)
# holds only the window, so sizing the launch from `kfo` would write out of bounds.
function _launch_compute_into!(dest, grid, kfo)
    arch = architecture(grid)
    params = KernelParameters(size(dest), map(offset_index, dest.indices))
    launch!(arch, grid, params, _compute!, dest.data, kfo)
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
    _launch_compute_into!(temp, grid, kfo)
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

    _launch_compute_into!(comp, grid, final)
    fill_halo_regions!(comp)
    set_status!(comp.status, time)
    return comp
end
#---

include("box_filter.jl")
include("gaussian_filter.jl")

#+++ Staged-vs-fused diagnostic
# A multi-direction filter takes its fast staged (separable) path only as the direct operand of a
# `Field`; nesting it in any other operation falls back to the slow fused kernel. `check_filter_staging`
# walks an operation tree structurally (a node is staged only if it is a `Field`'s direct operand) and
# flags every filter that will run fused. See the "Spatial filters" docs ("Performance notes") for why.

const _MultiDirFilter = Union{_BoxFilter2D, _BoxFilter3D, _GaussianFilter2D, _GaussianFilter3D}

@inline _push_filter_child!(children, x::AbstractField) = (push!(children, x); nothing)
@inline _push_filter_child!(children, x::Tuple) = (foreach(el -> _push_filter_child!(children, el), x); nothing)
@inline _push_filter_child!(children, x) = nothing

function _operation_children(node)
    children = Any[]
    for i in 1:nfields(node)
        _push_filter_child!(children, getfield(node, i))
    end
    return children
end

function _collect_fused_filters!(found, seen, node, staged_ok)
    (node in seen) && return found
    push!(seen, node)

    if node isa Field
        op = node.operand
        op === nothing || _collect_fused_filters!(found, seen, op, true)
        return found
    end

    (node isa _MultiDirFilter) && !staged_ok && push!(found, node)

    for child in _operation_children(node)
        _collect_fused_filters!(found, seen, child, false)
    end
    return found
end

"""
    check_filter_staging(op; warn=true)

Inspect the operation tree `op` (an `AbstractOperation`, a `Field`, or a reduction such as
`Integral(...)`) for multi-direction Oceanostics filters — a [`BoxFilter`](@ref) or
[`GaussianFilter`](@ref) over two or three directions — that will evaluate on the slow *fused*
single-kernel path instead of the fast staged (separable) path.

A multi-direction filter is evaluated by its staged kernel only when it is the **direct operand of a
`Field`**, e.g. `Field(filter(ψ))`. Composing the filter into any other operation — `Field(filter(ψ)
- φ)`, `2 * filter(ψ)`, `Integral(filter(ψ))`, and so on — hides it from that dispatch and it silently
falls back to the fused path (`Nᵈ` reads per output cell instead of `d × N`, with the filtered operand
recomputed at every stencil point). Both paths return the same values; only the speed differs. See
[Performance notes](@ref filter_performance).

The fix is to materialize the filtered field first: wrap `filter(ψ)` in its own `Field(...)` before
composing it, e.g. write `Field(filter(ψ)) - φ` rather than `filter(ψ) - φ`.

`op` is analyzed as if it were about to be wrapped in a `Field` and computed. Returns `true` when
every multi-direction filter in `op` is positioned to run staged (nothing to fix), and `false` when
at least one will run fused. When `warn = true` (the default), a `false` result also emits a `@warn`
describing the problem. One-direction filters always use the single unrolled kernel (staged and fused
coincide) and are never flagged.

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(8, 8, 8), extent=(1, 1, 1));

julia> c = CenterField(grid); φ = CenterField(grid);

julia> gf = GaussianFilter(; dims=(1, 2), σ=0.1);

julia> check_filter_staging(Field(gf(c)))          # staged: direct Field operand
true

julia> check_filter_staging(Field(gf(c)) - φ)      # staged: filtered field materialized first
true

julia> check_filter_staging(gf(c) - φ; warn=false) # fused: filter nested in a larger operation
false
```
"""
function check_filter_staging(op; warn=true)
    fused = _collect_fused_filters!(Any[], Base.IdSet{Any}(), op, true)
    all_staged = isempty(fused)
    if warn && !all_staged
        n = length(fused)
        @warn string(
            "check_filter_staging found ", n, " multi-direction filter", (n == 1 ? "" : "s"),
            " that will run on the slow fused path. A multi-direction `BoxFilter`/`GaussianFilter` ",
            "is evaluated by its fast staged (separable) kernel only when it is the direct operand ",
            "of a `Field`. Nesting it inside another operation (e.g. `Field(filter(ψ) - φ)`, ",
            "`2 * filter(ψ)`, `Integral(filter(ψ))`) falls back to the fused single-kernel path ",
            "(Nᵈ reads per cell instead of d×N). Materialize the filtered field first — wrap ",
            "`filter(ψ)` in its own `Field(...)` before composing it. See the \"Spatial filters\" ",
            "documentation, section \"Performance notes\", for details.")
    end
    return all_staged
end
#---

end # module
