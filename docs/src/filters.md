# Filters

Oceanostics provides spatial filters that operate directly on Oceananigans fields.
All filters are built on top of Oceananigans'
[`KernelFunctionOperation`](https://clima.github.io/OceananigansDocumentation/dev/appendix/library/#Oceananigans.AbstractOperations.KernelFunctionOperation),
so they compose with the rest of the Oceananigans ecosystem (outputs, reductions,
other operations, etc.) and run on both CPU and GPU.

## Box filter

The [`BoxFilter`](@ref) computes a local running-mean (box average) of a field
over one or more grid directions. The stencil width is controlled by the `width`
keyword: each filtered direction uses a `(2*width + 1)`-point symmetric average
centred on the current cell.

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

c̄ = Field(BoxFilter(c; dims=(1, 3), width=2))
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
c̄_edge = Field(BoxFilter(c; dims=(1, 3), width=1, boundary=:edge))

# Per-dim: :shrink in x, constant-pad in z
c̄_mixed = Field(BoxFilter(c; dims=(1, 3), width=1, boundary=(:shrink, (left=0.0, right=0.0))))
```

### API reference

```@docs
BoxFilter
```
