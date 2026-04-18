module Filters
using DocStringExtensions

export BoxFilter

using Oceananigans: location
using Oceananigans.Grids: halo_size
using Oceananigans.AbstractOperations: KernelFunctionOperation

using Oceanostics: CustomKFO

@inline function box_filter(i, j, k, grid, ψ, ::Val{dims}, width) where dims
    w1 = (1 in dims) ? width : 0
    w2 = (2 in dims) ? width : 0
    w3 = (3 in dims) ? width : 0

    s = zero(grid)
    n = 0
    @inbounds for di in -w1:w1, dj in -w2:w2, dk in -w3:w3
        s += ψ[i+di, j+dj, k+dk]
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

The output location matches the location of `ψ`. Any `ψ` supporting the standard
Oceananigans `ψ[i, j, k]` indexing contract is accepted (for example a `Field` or
any `AbstractOperation`).

`grid` must have a halo of at least `width` cells along every direction in `dims`.
"""
function BoxFilter(ψ; dims, width)
    validate_dims(dims)
    validate_width(width)

    grid = ψ.grid
    validate_halo(grid, dims, width)

    LX, LY, LZ = location(ψ)
    return KernelFunctionOperation{LX, LY, LZ}(box_filter, grid, ψ, Val(dims), width)
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

function validate_halo(grid, dims, width)
    H = halo_size(grid)
    for d in dims
        H[d] >= width ||
            throw(ArgumentError("BoxFilter requires halo_size(grid)[$d] ≥ width = $width, but halo along dim $d is $(H[d])"))
    end
    return nothing
end

end # module
