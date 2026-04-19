module Filters
using DocStringExtensions

export BoxFilter

using Oceananigans: location
using Oceananigans.Grids: halo_size, topology, Periodic
using Oceananigans.AbstractOperations: KernelFunctionOperation

using Oceanostics: CustomKFO

"""
    BoxFilterKernel{D} <: Function

Callable singleton that computes a 1D box average along direction `D` (1, 2, or 3).
Has two methods: a terminal one that indexes into an indexable input, and a
recursive one that invokes another kernel function at each stencil point.
"""
struct BoxFilterKernel{D} <: Function end

# Terminal methods (indexable input).

@inline function (::BoxFilterKernel{1})(i, j, k, grid, width, periodic, ψ)
    Nx = size(grid, 1)
    s = zero(grid); n = 0
    @inbounds for di in -width:width
        ii = periodic ? mod1(i + di, Nx) : i + di
        s += ψ[ii, j, k]
        n += 1
    end
    return s / n
end

@inline function (::BoxFilterKernel{2})(i, j, k, grid, width, periodic, ψ)
    Ny = size(grid, 2)
    s = zero(grid); n = 0
    @inbounds for dj in -width:width
        jj = periodic ? mod1(j + dj, Ny) : j + dj
        s += ψ[i, jj, k]
        n += 1
    end
    return s / n
end

@inline function (::BoxFilterKernel{3})(i, j, k, grid, width, periodic, ψ)
    Nz = size(grid, 3)
    s = zero(grid); n = 0
    @inbounds for dk in -width:width
        kk = periodic ? mod1(k + dk, Nz) : k + dk
        s += ψ[i, j, kk]
        n += 1
    end
    return s / n
end

# Recursive methods (function input — typically another BoxFilterKernel).

@inline function (::BoxFilterKernel{1})(i, j, k, grid, width, periodic, f::Function, fargs...)
    Nx = size(grid, 1)
    s = zero(grid); n = 0
    @inbounds for di in -width:width
        ii = periodic ? mod1(i + di, Nx) : i + di
        s += f(ii, j, k, grid, fargs...)
        n += 1
    end
    return s / n
end

@inline function (::BoxFilterKernel{2})(i, j, k, grid, width, periodic, f::Function, fargs...)
    Ny = size(grid, 2)
    s = zero(grid); n = 0
    @inbounds for dj in -width:width
        jj = periodic ? mod1(j + dj, Ny) : j + dj
        s += f(i, jj, k, grid, fargs...)
        n += 1
    end
    return s / n
end

@inline function (::BoxFilterKernel{3})(i, j, k, grid, width, periodic, f::Function, fargs...)
    Nz = size(grid, 3)
    s = zero(grid); n = 0
    @inbounds for dk in -width:width
        kk = periodic ? mod1(k + dk, Nz) : k + dk
        s += f(i, j, kk, grid, fargs...)
        n += 1
    end
    return s / n
end

const BoxFilter = CustomKFO{<:BoxFilterKernel}

"""
    $(SIGNATURES)

Return a `KernelFunctionOperation` that computes a local box-average of `ψ` over
the directions listed in `dims`.

`dims` is a tuple of distinct integers drawn from `(1, 2, 3)` (where `1`, `2`, `3`
correspond to `x`, `y`, `z`). `width` is the half-width of the stencil in cells, so
each selected direction uses a `(2*width + 1)`-point running mean centered on the
current cell.

A multi-directional filter is assembled as a single `KernelFunctionOperation`
whose kernel function is a 1D `BoxFilterKernel{d₁}`, with the next dimension's
`BoxFilterKernel{d₂}` (and so on) threaded into the argument list. The nested
1D kernels inline into a single fused read pass at compile time.

Along any direction whose topology is `Periodic`, stencil indices are wrapped
using `mod1`, so `width` may be chosen freely regardless of `halo_size(grid)`.
Along any non-periodic direction the stencil relies on the grid's halo, and the
constructor enforces `halo_size(grid)[d] ≥ width` for every such selected dim.

The output location matches the location of `ψ`. Any `ψ` supporting the standard
Oceananigans `ψ[i, j, k]` indexing contract is accepted (for example a `Field`
or any `AbstractOperation`).
"""
function BoxFilter(ψ; dims, width)
    validate_dims(dims)
    validate_width(width)

    grid = ψ.grid
    periodic = periodic_dims(grid)
    validate_halo(grid, dims, width, periodic)

    LX, LY, LZ = location(ψ)

    sorted_dims = Tuple(d for d in (1, 2, 3) if d in dims)
    return build_box_filter_kfo(grid, (LX, LY, LZ), sorted_dims, width, periodic, ψ)
end

function build_box_filter_kfo(grid, loc, dims::Tuple{Int}, width, periodic, ψ)
    d = dims[1]
    return KernelFunctionOperation{loc...}(BoxFilterKernel{d}(), grid,
                                           width, d in periodic, ψ)
end

function build_box_filter_kfo(grid, loc, dims::NTuple{2, Int}, width, periodic, ψ)
    d1, d2 = dims
    return KernelFunctionOperation{loc...}(BoxFilterKernel{d1}(), grid,
                                           width, d1 in periodic,
                                           BoxFilterKernel{d2}(), width, d2 in periodic,
                                           ψ)
end

function build_box_filter_kfo(grid, loc, dims::NTuple{3, Int}, width, periodic, ψ)
    d1, d2, d3 = dims
    return KernelFunctionOperation{loc...}(BoxFilterKernel{d1}(), grid,
                                           width, d1 in periodic,
                                           BoxFilterKernel{d2}(), width, d2 in periodic,
                                           BoxFilterKernel{d3}(), width, d3 in periodic,
                                           ψ)
end

validate_dims(dims::Tuple{Vararg{Int}}) =
    (!isempty(dims) && all(d -> d in (1, 2, 3), dims) && allunique(dims)) ||
        throw(ArgumentError("BoxFilter `dims` must be a non-empty tuple of distinct integers drawn from (1, 2, 3); got $dims"))

validate_dims(dims) =
    throw(ArgumentError("BoxFilter `dims` must be a tuple of integers; got $(typeof(dims))"))

validate_width(width::Integer) =
    width >= 1 || throw(ArgumentError("BoxFilter `width` must be a positive integer; got $width"))

validate_width(width) =
    throw(ArgumentError("BoxFilter `width` must be a positive integer; got $(typeof(width))"))

periodic_dims(grid) = Tuple(d for d in (1, 2, 3) if topology(grid, d) === Periodic)

function validate_halo(grid, dims, width, periodic)
    H = halo_size(grid)
    for d in dims
        d in periodic && continue
        H[d] >= width ||
            throw(ArgumentError("BoxFilter requires halo_size(grid)[$d] ≥ width = $width along non-periodic dim $d, but halo is $(H[d])"))
    end
    return nothing
end

end # module
