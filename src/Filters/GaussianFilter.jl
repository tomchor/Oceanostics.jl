#+++ GaussianFilter kernel
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

GaussianFilterKernel{D}(weights::W) where {D, W} = GaussianFilterKernel{D, W}(weights)

# Terminal methods (indexable input).

@inline function (kern::GaussianFilterKernel{1})(i, j, k, grid, ::Val{width}, policy, ψ) where {width}
    Nx = size(grid, 1)
    s = zero(grid); w_sum = zero(grid)
    @inbounds for idx in 1:(2*width+1)
        Δi = idx - width - 1
        w = kern.weights[idx]
        val, cnt = x_stencil_fetch(policy, ψ, i + Δi, j, k, Nx)
        s += w * val
        w_sum += w * cnt
    end
    return s / w_sum
end

@inline function (kern::GaussianFilterKernel{2})(i, j, k, grid, ::Val{width}, policy, ψ) where {width}
    Ny = size(grid, 2)
    s = zero(grid); w_sum = zero(grid)
    @inbounds for idx in 1:(2*width+1)
        Δj = idx - width - 1
        w = kern.weights[idx]
        val, cnt = y_stencil_fetch(policy, ψ, i, j + Δj, k, Ny)
        s += w * val
        w_sum += w * cnt
    end
    return s / w_sum
end

@inline function (kern::GaussianFilterKernel{3})(i, j, k, grid, ::Val{width}, policy, ψ) where {width}
    Nz = size(grid, 3)
    s = zero(grid); w_sum = zero(grid)
    @inbounds for idx in 1:(2*width+1)
        Δk = idx - width - 1
        w = kern.weights[idx]
        val, cnt = z_stencil_fetch(policy, ψ, i, j, k + Δk, Nz)
        s += w * val
        w_sum += w * cnt
    end
    return s / w_sum
end

# Recursive methods (function input — typically another GaussianFilterKernel).

@inline function (kern::GaussianFilterKernel{1})(i, j, k, grid, ::Val{width}, policy, f::Function, fargs...) where {width}
    Nx = size(grid, 1)
    s = zero(grid); w_sum = zero(grid)
    @inbounds for idx in 1:(2*width+1)
        Δi = idx - width - 1
        w = kern.weights[idx]
        val, cnt = x_stencil_call(policy, f, i + Δi, j, k, Nx, grid, fargs...)
        s += w * val
        w_sum += w * cnt
    end
    return s / w_sum
end

@inline function (kern::GaussianFilterKernel{2})(i, j, k, grid, ::Val{width}, policy, f::Function, fargs...) where {width}
    Ny = size(grid, 2)
    s = zero(grid); w_sum = zero(grid)
    @inbounds for idx in 1:(2*width+1)
        Δj = idx - width - 1
        w = kern.weights[idx]
        val, cnt = y_stencil_call(policy, f, i, j + Δj, k, Ny, grid, fargs...)
        s += w * val
        w_sum += w * cnt
    end
    return s / w_sum
end

@inline function (kern::GaussianFilterKernel{3})(i, j, k, grid, ::Val{width}, policy, f::Function, fargs...) where {width}
    Nz = size(grid, 3)
    s = zero(grid); w_sum = zero(grid)
    @inbounds for idx in 1:(2*width+1)
        Δk = idx - width - 1
        w = kern.weights[idx]
        val, cnt = z_stencil_call(policy, f, i, j, k + Δk, Nz, grid, fargs...)
        s += w * val
        w_sum += w * cnt
    end
    return s / w_sum
end

const GaussianFilter = CustomKFO{<:GaussianFilterKernel}

gaussian_weights(width, σ) = ntuple(idx -> exp(-(idx - width - 1)^2 / (2σ^2)), 2*width + 1)

validate_σ(σ::Real) =
    σ > 0 || throw(ArgumentError("`σ` must be a positive number; got $σ"))

validate_σ(σ) =
    throw(ArgumentError("`σ` must be a positive number; got $(typeof(σ))"))

"""
    $(SIGNATURES)

Return a `KernelFunctionOperation` that computes a Gaussian-weighted local
average of `ψ` over the directions listed in `dims`.

Like `BoxFilter`, the stencil half-width is `width` cells, so each filtered
direction uses a `(2*width + 1)`-point stencil centered on the current cell.
Instead of uniform weights, each point receives a Gaussian weight
`exp(-Δ²/(2σ²))`, where `Δ` is the cell offset and `σ` (in cells) defaults to
`width / 2`. The filter is normalized: the weighted sum is divided by the sum
of the surviving weights, so all boundary policies behave consistently.

See `BoxFilter` for the `dims`, `width`, and `boundary` keyword documentation.

## Examples

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(8, 8), x=(0, 1), z=(0, 1),
                              topology=(Periodic, Flat, Bounded));

julia> c = CenterField(grid);

julia> GaussianFilter(c; dims=(1, 3), width=2, σ=1.0) isa KernelFunctionOperation
true
```
"""
function GaussianFilter(ψ; dims, width, σ=width/2, boundary=:shrink)
    validate_σ(σ)
    grid, loc, sorted_dims, policies = resolve_filter_policies(ψ, dims, width, boundary)
    weights = gaussian_weights(width, σ)
    return build_filter_kfo(d -> GaussianFilterKernel{d}(weights), grid, loc, sorted_dims, width, policies, ψ)
end
#---
