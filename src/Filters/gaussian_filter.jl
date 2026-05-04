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

#+++ Terminal methods (indexable input).
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
#---

#+++ Recursive methods (function input — typically another GaussianFilterKernel).
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

Return a `KernelFunctionOperation` that computes a Gaussian-weighted local
average of `ψ` over the directions listed in `dims`.

`σ` is the standard deviation of the Gaussian kernel in physical units (the
same units as the grid spacing). Internally the filter precomputes its
weights once per direction in cell-offset units using `σ_cells = σ / Δ`,
where `Δ` is the (uniform) grid spacing along that direction; each stencil
point at cell offset `Δi` then receives weight `exp(-Δi² / (2 σ_cells²))`.
The filter is normalized: the weighted sum is divided by the sum of the
surviving weights, so all boundary policies behave consistently.

!!! warning "Uniform spacing required"
    Because the per-direction weights are precomputed assuming a single `Δ`,
    `GaussianFilter` only supports **uniform spacing along each filtered
    direction**. Non-uniform (stretched) directions raise an `ArgumentError`
    at construction time. Other directions on the same grid may be
    non-uniform — only the ones listed in `dims` need to be uniform. Use
    [`BoxFilter`](@ref) on stretched directions, since its weights do not
    depend on `Δ`.

`n_points` is the total number of grid points in the stencil along each
filtered direction; it must be an **odd integer ≥ 3** so the stencil is
symmetric around the current cell. If unspecified, `n_points` is inferred
per-direction from `σ` and `Δ` so that the stencil extends roughly `2σ` on
each side of the current cell. To override, pass either a single odd integer
(applied to every filtered dim) or a tuple with one odd-integer count per
dim in `dims` (in the order the user passed them). For `Periodic`
directions the stencil must span at most one period (`n_points ≤ 2N+1`,
where `N` is the number of cells along that direction); this is enforced at
construction time.

See `BoxFilter` for the `dims` and `boundary` keyword documentation.

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
function GaussianFilter(ψ; dims, σ, n_points=nothing, boundary=:shrink)
    validate_σ(σ)
    grid, loc, sorted_dims, policies = resolve_filter_policies(ψ, dims, boundary)
    validate_uniform_spacing(grid, sorted_dims, "GaussianFilter")

    sorted_widths = resolve_gaussian_widths(n_points, σ, grid, dims, sorted_dims)
    validate_periodic_widths(grid, sorted_dims, policies, sorted_widths)
    σT = convert(eltype(grid), σ)
    sorted_weights = ntuple(i -> gaussian_weights(sorted_widths[i],
                                                  σT / direction_min_spacing(grid, sorted_dims[i])),
                            length(sorted_dims))

    return build_filter_kfo((d, i) -> GaussianFilterKernel{d}(sorted_weights[i]),
                            grid, loc, sorted_dims, sorted_widths, policies, ψ)
end

infer_width(σ, grid, d) = ceil(Int, 2σ / direction_min_spacing(grid, d))

function resolve_gaussian_widths(n_points, σ, grid, dims, sorted_dims)
    if n_points === nothing
        return ntuple(i -> infer_width(σ, grid, sorted_dims[i]), length(sorted_dims))
    elseif n_points isa Tuple
        length(n_points) == length(dims) ||
            throw(ArgumentError("`n_points` tuple must have one entry per dim in `dims`; got length $(length(n_points)) for dims=$dims"))
        foreach(validate_n_points, n_points)
        return ntuple(i -> begin
            user_idx = findfirst(==(sorted_dims[i]), dims)
            (n_points[user_idx] - 1) ÷ 2
        end, length(sorted_dims))
    else
        validate_n_points(n_points)
        return ntuple(_ -> (n_points - 1) ÷ 2, length(sorted_dims))
    end
end
#---
