# Filters

Oceanostics provides spatial filters that operate directly on Oceananigans fields.
All filters are built on top of Oceananigans'
[`KernelFunctionOperation`](https://clima.github.io/OceananigansDocumentation/dev/appendix/library/#Oceananigans.AbstractOperations.KernelFunctionOperation),
so they compose with the rest of the Oceananigans ecosystem (outputs, reductions,
other operations, etc.) and run on both CPU and GPU.

## Box filter

The [`BoxFilter`](@ref) computes a local running-mean (box average) of a field
over one or more grid directions. The stencil size is controlled by the `N`
keyword: each filtered direction uses an `N`-point symmetric average centred on
the current cell — i.e. `N` cells contribute to one filtered output value.
`N` must be an **odd integer ≥ 3**, and refers to the size of the *filter
stencil*, not the grid.

Multi-direction filters are fused into a single kernel at compile time, so
a 3D box filter performs one pass over the data, not three.

### Basic usage

```jldoctest filters
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(32, 32), x=(0, 1), z=(0, 1),
                              topology=(Periodic, Flat, Bounded));

julia> c = CenterField(grid);

julia> set!(c, (x, z) -> sin(2π * x) * z);

julia> c̄ = Field(BoxFilter(c; dims=(1, 3), N=5));

julia> size(c̄)
(32, 1, 32)
```

### Boundary handling

For `Periodic` directions, stencil offsets are always wrapped — the `boundary`
keyword is silently ignored. For `Bounded` directions the `boundary` keyword
selects how out-of-bounds offsets are treated:

| `boundary`            | Behaviour                                                        |
|:----------------------|:-----------------------------------------------------------------|
| `:shrink` *(default)* | Drop out-of-bounds offsets from sum **and** count (honest local average; stencil shrinks near walls). |
| `:edge`               | Replicate the nearest interior cell (`ψ[1]` or `ψ[N]`).          |
| `(left=a, right=b)`   | Pad with constant `a` on the low side and `b` on the high side.  |

A single spec applies to every filtered dimension, or a tuple gives per-dimension control:

```jldoctest filters
julia> c̄_edge = Field(BoxFilter(c; dims=(1, 3), N=3, boundary=:edge));

julia> c̄_mixed = Field(BoxFilter(c; dims=(1, 3), N=3, boundary=(:shrink, (left=0.0, right=0.0))));

julia> size(c̄_edge) == size(c̄_mixed) == (32, 1, 32)
true
```

### API reference

```@docs
BoxFilter
BoxFilterOperator
```

## Gaussian filter

The [`GaussianFilter`](@ref) computes a Gaussian-weighted local average of a
field over one or more grid directions. Along a single filtered direction the
filtered value at cell ``i`` is

```math
\bar{\psi}_i = \frac{\sum_m \Delta_m \, e^{-(x_m - x_i)^2 / (2\sigma^2)} \, \psi_m}
                    {\sum_m \Delta_m \, e^{-(x_m - x_i)^2 / (2\sigma^2)}} ,
```

where ``m`` runs over the cells of the stencil centred on ``i``; ``x_m`` and
``\Delta_m`` are the coordinate and width of cell ``m`` along the filtered
direction, ``x_i`` is the current cell's coordinate, and ``\sigma`` is the
kernel's standard deviation in **physical units** (the same as the grid
spacing). In words: each cell's Gaussian weight
``e^{-(x_m - x_i)^2 / (2\sigma^2)}`` is multiplied by that cell's own width
``\Delta_m``, and the **same** ``\Delta_m``-weighted sum appears in the
normalizing denominator. The normalization returns a constant field unchanged
and keeps every boundary policy consistent, while the ``\Delta_m`` factor is the
quadrature weight (the ``dx'`` of ``\int G_\sigma(x-x')\,\psi(x')\,dx' \big/
\int G_\sigma(x-x')\,dx'``) that makes the discrete sum approximate the
continuous Gaussian convolution. Multi-direction filters apply this to each
filtered direction in turn.

`σ` is the only required parameter beyond `dims` — `N` is inferred
per-direction from `σ` and the minimum cell spacing in that direction so the
stencil extends at least `2σ` on each side of the current cell. To override,
pass `N` explicitly: a single odd integer (≥ 3) applies to every filtered
dim, or a tuple of odd integers sets one count per dim.

`dims` and `boundary` work identically to `BoxFilter`.

For a worked end-to-end example — coarse-graining a turbulent flow, a filter-width sweep, and a
subfilter tracer flux — see the [Spatial filtering example](@ref spatial_filtering_example).

### Variably spaced (stretched) grids

`GaussianFilter` works on stretched grids, but it is around 4 times slower than
on regular grids:

- Along a **uniformly spaced** direction every ``\Delta_m`` is equal, so it
  factors out of the sum and cancels between numerator and denominator. The
  weights then reduce to a plain ``e^{-(x_m - x_i)^2 / (2\sigma^2)}`` that is
  identical for every cell, so they are precomputed once and looked up — the
  fast path used on regular grids.
- Along a **variably spaced** direction ``x_m`` and ``\Delta_m`` differ from cell
  to cell, so the weights cannot be precomputed and are evaluated on the fly.
  Keeping the per-cell ``\Delta_m`` factor is what stops the average from being
  biased toward finely resolved regions; it makes the filter preserve constants
  exactly and linear fields to quadrature accuracy.

Directions are handled independently, so a grid that is uniform in `x` and `y`
but stretched in `z` uses the fast path for the horizontal directions and the
node-distance path for the vertical:

```jldoctest filters
julia> stretched_grid = RectilinearGrid(size=(16, 16),
                                        x=(0, 1),
                                        z=k -> -1 + (k-1)/16 + 0.05sin(2π*(k-1)/16),
                                        topology=(Periodic, Flat, Bounded));

julia> cz = CenterField(stretched_grid); set!(cz, (x, z) -> sin(2π*x) * z);

julia> c̄z = Field(GaussianFilter(cz; dims=(1, 3), σ=0.1));

julia> c̄z isa Field
true
```


```jldoctest filters
julia> c̄_gauss = Field(GaussianFilter(c; dims=(1, 3), σ=0.05));

julia> c̄_gauss isa Field
true
```

Pass `N` to override the default stencil:

```jldoctest filters
julia> c̄_wide = Field(GaussianFilter(c; dims=(1,), σ=0.05, N=11));

julia> c̄_perdim = Field(GaussianFilter(c; dims=(1, 3), σ=0.05, N=(7, 11)));

julia> (c̄_wide isa Field, c̄_perdim isa Field)
(true, true)
```

### API reference

```@docs
GaussianFilter
GaussianFilterOperator
```
