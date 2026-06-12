"""
    BoxFilterKernel{D} <: Function

Callable singleton that computes a 1D box average along direction `D` (1, 2, or 3).
Has two methods: a terminal one that indexes into an indexable input, and a
recursive one that invokes another kernel function at each stencil point.
"""
struct BoxFilterKernel{D} <: Function end
const BoxFilter = CustomKFO{<:BoxFilterKernel}

# `@unroll_full` (defined in `Filters.jl`) is applied here for the same
# reason as in `GaussianFilterKernel`: without it the per-thread
# accumulator state and per-iteration policy branch can fail to specialize,
# producing a measurable cliff at large widths. For `BoxFilter` there is no
# weights tuple, but unrolling still lets LLVM hoist the boundary-policy
# branch out of the loop body so each iteration becomes branch-free.

#+++ Terminal methods (indexable input).
@inline function (::BoxFilterKernel{1})(i, j, k, grid, ::Val{width}, policy, ψ) where {width}
    Nx = size(grid, 1)
    s = zero(grid); n = 0
    @unroll_full for Δi in -width:width
        val, cnt = x_stencil_fetch(policy, ψ, i + Δi, j, k, Nx)
        s += val; n += cnt
    end
    return s / n
end

@inline function (::BoxFilterKernel{2})(i, j, k, grid, ::Val{width}, policy, ψ) where {width}
    Ny = size(grid, 2)
    s = zero(grid); n = 0
    @unroll_full for Δj in -width:width
        val, cnt = y_stencil_fetch(policy, ψ, i, j + Δj, k, Ny)
        s += val; n += cnt
    end
    return s / n
end

@inline function (::BoxFilterKernel{3})(i, j, k, grid, ::Val{width}, policy, ψ) where {width}
    Nz = size(grid, 3)
    s = zero(grid); n = 0
    @unroll_full for Δk in -width:width
        val, cnt = z_stencil_fetch(policy, ψ, i, j, k + Δk, Nz)
        s += val; n += cnt
    end
    return s / n
end
#---

#+++ Recursive methods (function input — typically another BoxFilterKernel).
@inline function (::BoxFilterKernel{1})(i, j, k, grid, ::Val{width}, policy, f::Function, fargs...) where {width}
    Nx = size(grid, 1)
    s = zero(grid); n = 0
    @unroll_full for Δi in -width:width
        val, cnt = x_stencil_call(policy, f, i + Δi, j, k, Nx, grid, fargs...)
        s += val; n += cnt
    end
    return s / n
end

@inline function (::BoxFilterKernel{2})(i, j, k, grid, ::Val{width}, policy, f::Function, fargs...) where {width}
    Ny = size(grid, 2)
    s = zero(grid); n = 0
    @unroll_full for Δj in -width:width
        val, cnt = y_stencil_call(policy, f, i, j + Δj, k, Ny, grid, fargs...)
        s += val; n += cnt
    end
    return s / n
end

@inline function (::BoxFilterKernel{3})(i, j, k, grid, ::Val{width}, policy, f::Function, fargs...) where {width}
    Nz = size(grid, 3)
    s = zero(grid); n = 0
    @unroll_full for Δk in -width:width
        val, cnt = z_stencil_call(policy, f, i, j, k + Δk, Nz, grid, fargs...)
        s += val; n += cnt
    end
    return s / n
end
#---

"""
    $(SIGNATURES)

Return a `KernelFunctionOperation` that computes a local box-average of `ψ`
over the directions listed in `dims`.

`dims` is a tuple of distinct integers drawn from `(1, 2, 3)` (where `1`, `2`,
`3` correspond to `x`, `y`, `z`).

`N` is the **total number of grid points used by the filter stencil** along
each filtered direction — i.e. how many cells are averaged together to
produce one filtered output value (e.g. `N=3` is a 3-point running mean,
`N=5` is a 5-point running mean). `N` must be an **odd integer ≥ 3** so the
stencil is symmetric around the current cell. (This is the size of the
filter stencil — *not* the size of the grid.)

A multi-directional filter is assembled as a single `KernelFunctionOperation`
whose kernel function is a 1D `BoxFilterKernel{d₁}`, with the next dimension's
`BoxFilterKernel{d₂}` (and so on) threaded into the argument list. The box
average is separable, so when the operation is the operand of a `Field` (the
standard `Field(BoxFilter(...))` / `compute!` path) it is evaluated as `d`
sequential 1D passes through intermediate fields — `d × N` reads per output
cell instead of `Nᵈ`. If the filter is composed into another
`AbstractOperation` (e.g. `2 * BoxFilter(c; dims=(1,2,3))`) the original
fused single-kernel evaluation runs instead.

## Boundary handling

Stencil offsets that leave the interior `1:Nd_grid` of a direction (where
`Nd_grid` is the number of cells along that direction) are handled
per-direction. For `Periodic` directions offsets are always wrapped
periodically, independent of the `boundary` keyword. For `Bounded`
directions the `boundary` keyword picks the policy (default: `:shrink`):

  - `:shrink` — drop out-of-bounds offsets from *both* the sum and the
    count, so the filter is an honest local average whose effective stencil
    shrinks near a wall. **This is the default for `Bounded` directions.**
  - `:edge` — replicate the boundary-cell value (reads `ψ[1]` or `ψ[Nd_grid]`
    for offsets past either end).
  - `(left=a, right=b)` — pad with constant `a` on the low-index side and
    `b` on the high-index side (`a` and `b` are promoted to a common type).

`boundary` may be a single spec applied to every filtered dim, or a tuple
with one entry per dim in `dims` (in the order the user passed them):

    BoxFilter(ψ; dims=(1, 2), N=7, boundary=:edge)
    BoxFilter(ψ; dims=(1, 2), N=7, boundary=(:shrink, :edge))
    BoxFilter(ψ; dims=(1,),   N=7, boundary=(left=0.0, right=1.0))

Because every policy wraps, clamps, or skips indices up front, `halo_size(grid)`
does not constrain `N`: a small halo on a bounded direction is fine. The
output location matches the location of `ψ`, and `ψ` can be any input that
supports the standard Oceananigans `ψ[i, j, k]` indexing contract (e.g. a
`Field` or any `AbstractOperation`).

For `Periodic` directions the stencil must span at most one period: `N ≤
2*Nd_grid + 1`, where `Nd_grid` is the number of cells along that direction.
This is enforced at construction time.

## Examples

Filter a tracer on a 2D (xz) grid that is periodic in `x` and bounded in `z`.
The `boundary=:edge` keyword explicitly overrides the default `:shrink` policy
for the bounded `z`-direction; `x` is `Periodic` so the `boundary` spec is
silently ignored for that direction regardless.

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(8, 8), x=(0, 1), z=(0, 1),
                              topology=(Periodic, Flat, Bounded));

julia> c = CenterField(grid);

julia> BoxFilter(c; dims=(1, 3), N=5, boundary=:edge) isa KernelFunctionOperation
true
```
"""
function BoxFilter(ψ; dims, N, boundary=:shrink)
    validate_N(N)
    grid, loc, sorted_dims, policies = resolve_filter_policies(ψ, dims, boundary)
    width = (N - 1) ÷ 2
    widths = ntuple(_ -> width, length(sorted_dims))
    validate_periodic_widths(grid, sorted_dims, policies, widths)
    return build_filter_kfo((d, _) -> BoxFilterKernel{d}(), grid, loc, sorted_dims, widths, policies, ψ)
end
#---

#+++ Reusable (field-less) box filter
"""
    BoxFilterOperator{D, NN, B}

A reusable, field-less box filter. Stores the `BoxFilter` parameters (`dims`,
`N`, `boundary`) and, when called on a field `ψ`, returns
`BoxFilter(ψ; dims, N, boundary)` — the very same `KernelFunctionOperation` that
the field-first constructor would build. Construct one once with
[`BoxFilter`](@ref)`(; …)` and apply it to many fields.
"""
struct BoxFilterOperator{D, NN, B}
    dims::D
    N::NN
    boundary::B
end

(F::BoxFilterOperator)(ψ) = BoxFilter(ψ; dims=F.dims, N=F.N, boundary=F.boundary)

"""
    BoxFilter(; dims, N, boundary=:shrink)

Build a reusable, field-less box filter (a callable `BoxFilterOperator`) capturing the box-filter
parameters without binding them to a field. The returned object is callable: `F(ψ)` returns
`BoxFilter(ψ; dims, N, boundary)`. Useful for applying the same filter to many
fields or for passing a preconfigured filter to other diagnostics.

See the field-first [`BoxFilter`](@ref)`(ψ; …)` method for the meaning of the
keyword arguments.
"""
BoxFilter(; dims, N, boundary=:shrink) = BoxFilterOperator(dims, N, boundary)

Base.show(io::IO, F::BoxFilterOperator) =
    print(io, "BoxFilter(dims=", F.dims, ", N=", F.N, ", boundary=", repr(F.boundary), ")")
#---

#+++ Staged multi-direction evaluation
# Multi-direction BoxFilters dispatch into the shared
# `_compute_staged_filter!` machinery defined in `Filters.jl`. 1D filters
# (`length(args) == 3`) fall through to the default `compute!` and use the
# unrolled single-direction kernel above.
const _BoxFilter2D = KernelFunctionOperation{LX, LY, LZ, G, T,
                                             <:BoxFilterKernel,
                                             <:Tuple{Val, AbstractBoundaryPolicy,
                                                     BoxFilterKernel, Val, AbstractBoundaryPolicy,
                                                     Any}} where {LX, LY, LZ, G, T}

const _BoxFilter3D = KernelFunctionOperation{LX, LY, LZ, G, T,
                                             <:BoxFilterKernel,
                                             <:Tuple{Val, AbstractBoundaryPolicy,
                                                     BoxFilterKernel, Val, AbstractBoundaryPolicy,
                                                     BoxFilterKernel, Val, AbstractBoundaryPolicy,
                                                     Any}} where {LX, LY, LZ, G, T}

compute!(comp::Field{<:Any, <:Any, <:Any, <:_BoxFilter2D}, time=nothing) = _compute_staged_filter!(comp, time)
compute!(comp::Field{<:Any, <:Any, <:Any, <:_BoxFilter3D}, time=nothing) = _compute_staged_filter!(comp, time)
#---
