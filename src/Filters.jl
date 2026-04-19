module Filters
using DocStringExtensions

export BoxFilter

using Oceananigans: location
using Oceananigans.Grids: halo_size, topology, Periodic
using Oceananigans.AbstractOperations: KernelFunctionOperation

using Oceanostics: CustomKFO

@inline function box_filter(i, j, k, grid, ψ, ::Val{dims}, ::Val{periodic}, width) where {dims, periodic}
    w1 = (1 in dims) ? width : 0
    w2 = (2 in dims) ? width : 0
    w3 = (3 in dims) ? width : 0

    Nx, Ny, Nz = size(grid)
    px = 1 in periodic
    py = 2 in periodic
    pz = 3 in periodic

    s = zero(grid)
    n = 0
    @inbounds for di in -w1:w1, dj in -w2:w2, dk in -w3:w3
        ii = px ? mod1(i + di, Nx) : i + di
        jj = py ? mod1(j + dj, Ny) : j + dj
        kk = pz ? mod1(k + dk, Nz) : k + dk
        s += ψ[ii, jj, kk]
        n += 1
    end
    return s / n
end

const BoxFilter = CustomKFO{<:typeof(box_filter)}

"""
    $(SIGNATURES)

Return a `KernelFunctionOperation` that computes a local box-average of `ψ` over
the directions listed in `dims`.

`dims` is a tuple of distinct integers drawn from `(1, 2, 3)` (where `1`, `2`, `3`
correspond to `x`, `y`, `z`). `width` is the half-width of the stencil in cells, so
each selected direction uses a `(2*width + 1)`-point running mean centered on the
current cell. Directions not listed in `dims` are left unfiltered.

Along any direction whose topology is `Periodic`, stencil indices are wrapped
using `mod1`, so `width` may be chosen freely regardless of `halo_size(grid)`.
Along any direction whose topology is not `Periodic` (e.g. `Bounded` or `Flat`),
the stencil relies on the grid's halo, and the constructor enforces
`halo_size(grid)[d] ≥ width` for every such selected direction.

The output location matches the location of `ψ`. Any `ψ` supporting the standard
Oceananigans `ψ[i, j, k]` indexing contract is accepted (for example a `Field` or
any `AbstractOperation`).
"""
function BoxFilter(ψ; dims, width)
    validate_dims(dims)
    validate_width(width)

    grid = ψ.grid
    periodic = periodic_dims(grid)
    validate_halo(grid, dims, width, periodic)

    LX, LY, LZ = location(ψ)
    return KernelFunctionOperation{LX, LY, LZ}(box_filter, grid, ψ,
                                               Val(dims), Val(periodic), width)
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
