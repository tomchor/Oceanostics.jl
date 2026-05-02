# Filters

Oceanostics provides spatial filters that operate directly on Oceananigans fields.
All filters are built on top of Oceananigans'
[`KernelFunctionOperation`](https://clima.github.io/OceananigansDocumentation/dev/appendix/library/#Oceananigans.AbstractOperations.KernelFunctionOperation),
so they compose with the rest of the Oceananigans ecosystem (outputs, reductions,
other operations, etc.) and run on both CPU and GPU.

## Box filter

The [`BoxFilter`](@ref) computes a local running-mean (box average) of a field
over one or more grid directions. The stencil size is controlled by the
`n_points` keyword: each filtered direction uses an `n_points`-point symmetric
average centred on the current cell. `n_points` must be an **odd integer ≥ 3**.

Multi-direction filters are fused into a single kernel at compile time, so
a 3D box filter performs one pass over the data, not three.

### Basic usage

```@example filters
using Oceananigans
using Oceanostics

grid = RectilinearGrid(size=(32, 32), x=(0, 1), z=(0, 1),
                       topology=(Periodic, Flat, Bounded))

c = CenterField(grid)
set!(c, (x, z) -> sin(2π * x) * z)

c̄ = Field(BoxFilter(c; dims=(1, 3), n_points=5))
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

```@example filters
# Same policy for both dims
c̄_edge = Field(BoxFilter(c; dims=(1, 3), n_points=3, boundary=:edge))

# Per-dim: :shrink in x, constant-pad in z
c̄_mixed = Field(BoxFilter(c; dims=(1, 3), n_points=3, boundary=(:shrink, (left=0.0, right=0.0))))
```

### API reference

```@docs
BoxFilter
```

## Gaussian filter

The [`GaussianFilter`](@ref) computes a Gaussian-weighted local average of a
field over one or more grid directions. Each stencil point at offset `Δ` from
the current cell receives weight `exp(-Δ²/(2σ²))`, where `σ` is the standard
deviation of the kernel in **physical units** (the same units as the grid
spacing). The filter is always normalized: the weighted sum is divided by the
sum of the surviving weights, so all boundary policies behave consistently.

`σ` is the only required parameter beyond `dims` — `n_points` is inferred
per-direction from `σ` and the minimum cell spacing in that direction so the
stencil extends roughly `2σ` on each side of the current cell. To override,
pass `n_points` explicitly: a single odd integer (≥ 3) applies to every
filtered dim, or a tuple of odd integers sets one count per dim.

`dims` and `boundary` work identically to `BoxFilter`.

### Basic usage

```@example filters
# σ in physical units (here, fraction of the unit-extent grid).
c̄_gauss = Field(GaussianFilter(c; dims=(1, 3), σ=0.05))
```

Pass `n_points` to override the default stencil:

```@example filters
# Wider stencil — captures more of the Gaussian's tails.
c̄_wide = Field(GaussianFilter(c; dims=(1,), σ=0.05, n_points=11))

# Per-dim stencil sizes (matching the order of `dims`).
c̄_perdim = Field(GaussianFilter(c; dims=(1, 3), σ=0.05, n_points=(7, 11)))
```

### API reference

```@docs
GaussianFilter
```
