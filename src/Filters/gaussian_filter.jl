# A self-contained "fully unroll this loop" macro. This is the same pattern as
# `KernelAbstractions.Extras.@unroll`: attach the LLVM `llvm.loop.unroll.full`
# loopinfo node to the body so the optimizer is required to unroll the loop.
# Done inline here so the `Filters` submodule does not need to add
# `KernelAbstractions` as a direct dependency.
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

"""
    GaussianFilterKernel{D, W} <: Function

Callable struct that computes a 1D Gaussian-weighted average along direction
`D` (1, 2, or 3). Stores precomputed unnormalized weights in `weights::W`.
Like `BoxFilterKernel`, has terminal (indexable input) and recursive (function
input) methods.
"""
struct GaussianFilterKernel{D, W} <: Function
    weights::W
end

const GaussianFilter = CustomKFO{<:GaussianFilterKernel}

GaussianFilterKernel{D}(weights::W) where {D, W} = GaussianFilterKernel{D, W}(weights)

# `@unroll` (LLVM `llvm.loop.unroll.full` hint) is essential here: the weights
# tuple is captured by-value in each thread's register file, and tuple
# indexing by a non-constant `idx` would force spilling the tuple to per-thread
# local memory. Forcing a full unroll keeps every tuple access at a constant
# index, so the weights live in registers (or are folded into the IR as
# constants), avoiding a ~20× cliff at width ≳ 8.

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

# GaussianFilter precomputes its weights in cell-offset units assuming a
# single Δ per direction. This helper rejects non-uniform filtered directions
# up front with a helpful error.
function validate_uniform_spacing(grid, sorted_dims, filter_name)
    for d in sorted_dims
        sp_min, sp_max = extrema(direction_spacings(grid, d))
        sp_min == sp_max || throw(ArgumentError(
            "$filter_name requires uniform grid spacing along filtered directions, but direction $d " *
            "has variable spacing (min=$sp_min, max=$sp_max). Its weights are precomputed in cell-offset " *
            "units assuming a constant Δ along the direction; on a stretched grid the filter's " *
            "physical-space footprint would vary per cell. Filter only directions whose spacing is uniform."))
    end
end

"""
    $(SIGNATURES)

Return a `KernelFunctionOperation` that computes a Gaussian-weighted local average of `ψ` over the
directions listed in `dims`.

`σ` is the standard deviation of the Gaussian kernel in physical units (the same units as the grid
spacing). Internally the filter precomputes its weights once per direction in cell-offset units
using `σ_cells = σ / Δ`, where `Δ` is the (uniform) grid spacing along that direction; each
stencil point at cell offset `Δi` then receives weight `exp(-Δi² / (2 σ_cells²))`. The filter is
normalized: the weighted sum is divided by the sum of the surviving weights, so all boundary
policies behave consistently.

!!! warning "Uniform spacing required"
    Because the per-direction weights are precomputed assuming a single `Δ`, `GaussianFilter` only
    supports **uniform spacing along each filtered direction**. Non-uniform (stretched) directions
    raise an `ArgumentError` at construction time. Other directions on the same grid may be
    non-uniform — only the ones listed in `dims` need to be uniform. Use [`BoxFilter`](@ref) on
    stretched directions, since its weights do not depend on `Δ`.

`N` is the **total number of grid points used by the filter stencil** along each filtered
direction — i.e. how many cells contribute to a single filtered output value. `N` must be an
**odd integer ≥ 3** so the stencil is symmetric around the current cell. (This is the size of the
filter stencil — *not* the size of the grid.) If unspecified, `N` is inferred per-direction from
`σ` and `Δ` so that the stencil extends roughly `2σ` on each side of the current cell. To
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
are competitive on GPUs. If the filter is composed into another `AbstractOperation` (e.g.
`2 * GaussianFilter(c; dims=(1,2,3))`) it falls back to the fused, single-kernel evaluation.

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
    validate_uniform_spacing(grid, sorted_dims, "GaussianFilter")

    sorted_widths = resolve_gaussian_widths(N, σ, grid, dims, sorted_dims)
    validate_periodic_widths(grid, sorted_dims, policies, sorted_widths)
    σT = convert(eltype(grid), σ)
    sorted_weights = ntuple(i -> gaussian_weights(sorted_widths[i], σT / direction_min_spacing(grid, sorted_dims[i])), length(sorted_dims))

    return build_filter_kfo((d, i) -> GaussianFilterKernel{d}(sorted_weights[i]),
                            grid, loc, sorted_dims, sorted_widths, policies, ψ)
end

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
# A `GaussianFilter` over `d ≥ 2` directions is built by `build_filter_kfo` as
# a single `KernelFunctionOperation` whose kernel function is the outermost
# 1D `GaussianFilterKernel`, and whose `arguments` thread the inner 1D
# kernels in. Evaluating that fused KFO costs `N^d` reads per output cell.
#
# Gaussian filters are separable, so the same field can be computed as a
# sequence of `d` 1D passes through intermediate fields, costing `d × N`
# reads per output cell. Here we override `Oceananigans.Fields.compute!` for
# `Field{...,<:GaussianFilter}` so the standard `Field(GaussianFilter(...))`
# path uses the staged evaluation. When the filter is *nested* inside
# another `AbstractOperation` (e.g. `2 * GaussianFilter(...)`) the operand
# stays a single fused KFO and the original inlined code path runs.

# These type aliases match the args-tuple shape produced by `build_filter_kfo`
# for the 2D and 3D cases. 1D filters fall through to the default `compute!`.
const _GaussianFilter2D = KernelFunctionOperation{LX, LY, LZ, G, T,
                                                  <:GaussianFilterKernel,
                                                  <:Tuple{Val, AbstractBoundaryPolicy,
                                                          GaussianFilterKernel, Val, AbstractBoundaryPolicy,
                                                          Any}} where {LX, LY, LZ, G, T}

const _GaussianFilter3D = KernelFunctionOperation{LX, LY, LZ, G, T,
                                                  <:GaussianFilterKernel,
                                                  <:Tuple{Val, AbstractBoundaryPolicy,
                                                          GaussianFilterKernel, Val, AbstractBoundaryPolicy,
                                                          GaussianFilterKernel, Val, AbstractBoundaryPolicy,
                                                          Any}} where {LX, LY, LZ, G, T}

import Oceananigans.Fields: compute!
using Oceananigans: location
using Oceananigans.Fields: Field, offset_index, set_status!
using Oceananigans.AbstractOperations: KernelFunctionOperation, compute_at!, _compute!
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Architectures: architecture
using Oceananigans.Utils: KernelParameters, launch!

@inline _staged_compute!(comp, time) = _compute_staged_gaussian!(comp, time)

compute!(comp::Field{<:Any, <:Any, <:Any, <:_GaussianFilter2D}, time=nothing) = _staged_compute!(comp, time)
compute!(comp::Field{<:Any, <:Any, <:Any, <:_GaussianFilter3D}, time=nothing) = _staged_compute!(comp, time)

# Build the i-th single-dim KFO in the chain at the filter's location.
@inline function _single_dim_kfo(loc, grid, kern, valw, pol, input)
    return KernelFunctionOperation{loc...}(kern, grid, valw, pol, input)
end

# Allocate an intermediate Field and compute the 1D pass into it, *without*
# filling halo regions. The next pass reads only the interior of the
# intermediate (every `*_stencil_fetch` clamps/wraps offsets into `1:N`), so
# halo data is irrelevant. Skipping the halo fill saves several small kernel
# launches per intermediate.
function _stage_into_temp(loc, grid, kern, valw, pol, input)
    kfo = _single_dim_kfo(loc, grid, kern, valw, pol, input)
    temp = Field(kfo, compute=false)
    _launch_compute_into!(temp.data, temp.indices, grid, kfo)
    return temp
end

# Write the result of evaluating `kfo` at every (i,j,k) of the iteration
# space into `data`. Reuses Oceananigans' trivial copy kernel `_compute!`
# (`data[i,j,k] = operand[i,j,k]`).
function _launch_compute_into!(data, indices, grid, kfo)
    arch = architecture(grid)
    params = KernelParameters(size(kfo), map(offset_index, indices))
    launch!(arch, grid, params, _compute!, data, kfo)
    return nothing
end

function _compute_staged_gaussian!(comp, time)
    op    = comp.operand
    grid  = op.grid
    loc   = location(op)
    args  = op.arguments
    kern1 = op.kernel_function

    # Recurse into ψ in case it is itself a computed field that needs
    # refreshing before we read from it.
    ψ = args[end]
    compute_at!(ψ, time)

    if length(args) == 6
        valw1, pol1                                = args[1], args[2]
        kern2, valw2, pol2                         = args[3], args[4], args[5]

        temp1 = _stage_into_temp(loc, grid, kern1, valw1, pol1, ψ)
        final = _single_dim_kfo(loc, grid, kern2, valw2, pol2, temp1)
    else  # length(args) == 9 — 3D filter
        valw1, pol1                                = args[1], args[2]
        kern2, valw2, pol2                         = args[3], args[4], args[5]
        kern3, valw3, pol3                         = args[6], args[7], args[8]

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
