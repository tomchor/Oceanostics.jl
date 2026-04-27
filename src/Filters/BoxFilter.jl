#+++ BoxFilter kernel
"""
    BoxFilterKernel{D} <: Function

Callable singleton that computes a 1D box average along direction `D` (1, 2, or 3).
Has two methods: a terminal one that indexes into an indexable input, and a
recursive one that invokes another kernel function at each stencil point.
"""
struct BoxFilterKernel{D} <: Function end

# Terminal methods (indexable input).

@inline function (::BoxFilterKernel{1})(i, j, k, grid, ::Val{width}, policy, ψ) where {width}
    Nx = size(grid, 1)
    s = zero(grid); n = 0
    @inbounds for Δi in -width:width
        val, cnt = x_stencil_fetch(policy, ψ, i + Δi, j, k, Nx)
        s += val; n += cnt
    end
    return s / n
end

@inline function (::BoxFilterKernel{2})(i, j, k, grid, ::Val{width}, policy, ψ) where {width}
    Ny = size(grid, 2)
    s = zero(grid); n = 0
    @inbounds for Δj in -width:width
        val, cnt = y_stencil_fetch(policy, ψ, i, j + Δj, k, Ny)
        s += val; n += cnt
    end
    return s / n
end

@inline function (::BoxFilterKernel{3})(i, j, k, grid, ::Val{width}, policy, ψ) where {width}
    Nz = size(grid, 3)
    s = zero(grid); n = 0
    @inbounds for Δk in -width:width
        val, cnt = z_stencil_fetch(policy, ψ, i, j, k + Δk, Nz)
        s += val; n += cnt
    end
    return s / n
end

# Recursive methods (function input — typically another BoxFilterKernel).

@inline function (::BoxFilterKernel{1})(i, j, k, grid, ::Val{width}, policy, f::Function, fargs...) where {width}
    Nx = size(grid, 1)
    s = zero(grid); n = 0
    @inbounds for Δi in -width:width
        val, cnt = x_stencil_call(policy, f, i + Δi, j, k, Nx, grid, fargs...)
        s += val; n += cnt
    end
    return s / n
end

@inline function (::BoxFilterKernel{2})(i, j, k, grid, ::Val{width}, policy, f::Function, fargs...) where {width}
    Ny = size(grid, 2)
    s = zero(grid); n = 0
    @inbounds for Δj in -width:width
        val, cnt = y_stencil_call(policy, f, i, j + Δj, k, Ny, grid, fargs...)
        s += val; n += cnt
    end
    return s / n
end

@inline function (::BoxFilterKernel{3})(i, j, k, grid, ::Val{width}, policy, f::Function, fargs...) where {width}
    Nz = size(grid, 3)
    s = zero(grid); n = 0
    @inbounds for Δk in -width:width
        val, cnt = z_stencil_call(policy, f, i, j, k + Δk, Nz, grid, fargs...)
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
`boundary` keyword picks the policy (default: `:shrink`):

  - `:shrink` — drop out-of-bounds offsets from *both* the sum and the
    count, so the filter is an honest local average whose effective stencil
    shrinks within `width` cells of a wall. **This is the default for
    `Bounded` directions.**
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

julia> BoxFilter(c; dims=(1, 3), width=2, boundary=:edge) isa KernelFunctionOperation
true
```
"""
function BoxFilter(ψ; dims, width, boundary=:shrink)
    grid, loc, sorted_dims, policies = resolve_filter_policies(ψ, dims, width, boundary)
    return build_filter_kfo(d -> BoxFilterKernel{d}(), grid, loc, sorted_dims, width, policies, ψ)
end
#---
