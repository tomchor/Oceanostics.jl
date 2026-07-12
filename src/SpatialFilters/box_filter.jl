"""
    BoxFilterKernel{D} <: Function

Callable singleton that computes a 1D box average along direction `D` (1, 2, or 3).
Has two methods: a terminal one that indexes into an indexable input, and a
recursive one that invokes another kernel function at each stencil point.
"""
struct BoxFilterKernel{D} <: Function end
const BoxFilter = CustomKFO{<:BoxFilterKernel}

# `@unroll_full` (defined in `SpatialFilters.jl`) is applied here for the same
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

One-step form: apply a box filter to `ψ` directly, returning the `KernelFunctionOperation` that
computes its local box-average. Equivalent to `BoxFilter(; dims, N, boundary_conditions)(ψ)`.

Refer to [`BoxFilter`](@ref)`(; dims, N, boundary_conditions)` for the full description of the
keyword arguments and boundary handling.
"""
function BoxFilter(ψ; dims, N, boundary_conditions=nothing, boundary=nothing)
    no_boundary_keyword(boundary)
    dims = tuplefy_dims(dims)
    validate_N(N)
    validate_boundary_conditions(boundary_conditions)
    ψ = materialize_operand(ψ)
    grid, loc, sorted_dims, policies = resolve_filter_policies(ψ, dims, boundary_conditions)
    width = (N - 1) ÷ 2
    widths = ntuple(_ -> width, length(sorted_dims))
    validate_periodic_widths(grid, sorted_dims, policies, widths)
    return build_filter_kfo((d, _) -> BoxFilterKernel{d}(), grid, loc, sorted_dims, widths, policies, ψ)
end
#---

#+++ Reusable (field-less) box filter
"""
    BoxFilterOperator{D, NN, B}

Returns a reusable box filter. Stores the `BoxFilter` parameters (`dims`, `N`, `boundary_conditions`)
and, when called on a field `ψ`, returns `BoxFilter(ψ; dims, N, boundary_conditions)`. Construct one
once with [`BoxFilter`](@ref)`(; …)` and apply it to many fields.
"""
struct BoxFilterOperator{D, NN, B}
    dims::D
    N::NN
    boundary_conditions::B
end

(F::BoxFilterOperator)(ψ) = BoxFilter(ψ; dims=F.dims, N=F.N, boundary_conditions=F.boundary_conditions)

"""
    $(SIGNATURES)

Build a reusable, field-less box filter that computes a local box-average over the directions listed
in `dims`. The returned object is callable: applying it to a field, `bf(ψ)`, returns a
`KernelFunctionOperation`. Build the filter once and reuse it across many fields, or pass it to other
diagnostics that accept a filter.

`dims` is a tuple of distinct integers drawn from `(1, 2, 3)` (where `1`, `2`, `3` correspond to `x`,
`y`, `z`). A single integer (e.g. `dims=2`) is accepted as shorthand for filtering along that one
direction.

`N` is the **total number of grid points used by the filter stencil** along each filtered direction —
i.e. how many cells are averaged together to produce one filtered output value (e.g. `N=3` is a
3-point running mean, `N=5` is a 5-point running mean). `N` must be an **odd integer ≥ 3** so the
stencil is symmetric around the current cell. (This is the size of the filter stencil — *not* the
size of the grid.)

A multi-directional filter is assembled as a single `KernelFunctionOperation` whose kernel function is
a 1D `BoxFilterKernel{d₁}`, with the next dimension's `BoxFilterKernel{d₂}` (and so on) threaded into
the argument list. The box average is separable, so when the operation is the operand of a `Field`
(the standard `Field(bf(ψ))` / `compute!` path) it is evaluated as `d` sequential 1D passes through
intermediate fields — `d × N` reads per output cell instead of `Nᵈ`. If the filtered field is composed
into another `AbstractOperation` (e.g. `2 * bf(c)`) the original fused single-kernel evaluation runs
instead.

See [Performance notes](@ref filter_performance) in the documentation for what that costs and how to avoid it.

## Boundary handling

Stencil offsets that leave the interior `1:Nd_grid` of a direction are handled per-direction using
Oceananigans boundary conditions, supplied via `boundary_conditions` and kept **separate** from the
filtered field's own boundary conditions:

  - `nothing` (default) — every side inherits the filtered field's own boundary condition for that
    direction.
  - a single `BoundaryCondition` — applied to every filtered side.
  - a `FieldBoundaryConditions` — one condition per side; any side left unset
    (`DefaultBoundaryCondition`) inherits the field's.

Each boundary condition maps to a stencil rule:

  - `ShrinkingBoundaryCondition()` — drop out-of-bounds offsets from *both* the sum and the count, so
    the filter is an honest local average whose effective stencil shrinks near a wall.
  - a zero `GradientBoundaryCondition` / `FluxBoundaryCondition` (incl. `NoFluxBoundaryCondition`) —
    replicate the boundary-cell value (reads `ψ[1]` or `ψ[Nd_grid]` past either end).
  - `ValueBoundaryCondition(v)` — pad with the constant `v`.
  - anything without a discrete analog (a non-zero gradient/flux, an `OpenBoundaryCondition`, …) falls
    back to the shrinking stencil.

For `Periodic` directions offsets are always wrapped, independent of `boundary_conditions`. When an
`AbstractOperation` is filtered it is first materialized into a `Field` (carrying Oceananigans'
default boundary conditions), which then supplies the inherited conditions.

Because every rule wraps, clamps, or skips indices up front, `halo_size(grid)` does not constrain `N`:
a small halo on a bounded direction is fine. The output location matches the location of the filtered
field. For `Periodic` directions the stencil must span at most one period: `N ≤ 2*Nd_grid + 1`.

## Examples

Build a box filter and apply it on a 2D (xz) grid that is periodic in `x` and bounded in `z`. The
`GradientBoundaryCondition(0)` (≡ edge replication) applies to the bounded `z`-direction; `x` is
`Periodic` so it is wrapped regardless.

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(8, 8), x=(0, 1), z=(0, 1),
                              topology=(Periodic, Flat, Bounded));

julia> c = CenterField(grid);

julia> bf = BoxFilter(; dims=(1, 3), N=5, boundary_conditions=GradientBoundaryCondition(0))
BoxFilter(dims=(1, 3), N=5, boundary_conditions=GradientBoundaryCondition: 0)

julia> bf(c) isa KernelFunctionOperation
true
```

A one-step shortcut `BoxFilter(ψ; dims, N, boundary_conditions)` is also accepted, which applies the
filter to `ψ` immediately (equivalent to `BoxFilter(; dims, N, boundary_conditions)(ψ)`).
"""
function BoxFilter(; dims, N, boundary_conditions=nothing, boundary=nothing)
    no_boundary_keyword(boundary)
    dims = tuplefy_dims(dims)
    validate_dims(dims)
    validate_N(N)
    validate_boundary_conditions(boundary_conditions)
    return BoxFilterOperator(dims, N, boundary_conditions)
end

function Base.show(io::IO, F::BoxFilterOperator)
    print(io, "BoxFilter(dims=", F.dims, ", N=", F.N)
    F.boundary_conditions === nothing || print(io, ", boundary_conditions=", bc_summary(F.boundary_conditions))
    print(io, ")")
end
#---

#+++ Staged multi-direction evaluation
# Multi-direction BoxFilters dispatch into the shared
# `_compute_staged_filter!` machinery defined in `SpatialFilters.jl`. 1D filters
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
