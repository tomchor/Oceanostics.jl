# Background potential energy equation

The `BackgroundPotentialEnergyEquation` module builds the **reference state**: the arrangement of the
buoyancy field with the least potential energy reachable adiabatically. Following
[Winters et al. (1995)](https://doi.org/10.1017/S002211209500125X), rearrange the field with every
parcel sorted by density and the densest fluid at the bottom. The height a parcel ends up at is its
reference height ``z^\star``, and the potential energy of that state is the background (or reference)
potential energy,

```math
E_b = -b z^\star = \frac{g \rho}{\rho_0} z^\star .
```

``E_b`` is the share of the potential energy ``E_p = -bz`` of
[the potential energy equation](potential_energy_equation.md) that the flow *cannot* release. What is
left over is the [available potential energy](available_potential_energy_equation.md), so the two
modules are the two halves of one split and share the reference state built here: `reference_height`,
`reference_buoyancy` and the methods below are exported by both.

Because it moves only with irreversible changes to the buoyancy field, ``\int E_b \, \mathrm{d}V`` is a
standard measure of the diapycnal mixing a simulation actually produces, numerical as well as physical.

## The reference state

height ``z^\star`` is the height a parcel would occupy in a state of minimum potential energy
that can be reached by adiabatic rearrangement flow parcels, and it
is computed by [`reference_height`](@ref). Rearrangement is a nonlocal
operation, so, unlike most other diagnostics in Oceanostics, this one is not a pointwise kernel.
It is rearranged on each `compute!`, so writing it (or anything built on it) out during a simulation
tracks the evolving flow.

Organizing the buoyancy against the reference height it was assigned gives the reference profile
``b^\star(z^\star)``: the stratification the flow would have if all of its available potential energy
were released. The [Lock release](@ref lock_release_example) example follows that profile through a
gravity current, from the step it starts as to the smooth stratification mixing leaves behind, and
builds it with each of the four methods below so their costs and their differences can be compared
directly.

`method` selects one of four strategies to calculate the reference state. All
`method`s produce the same reference state in the continuous limit and the same ``\int E_a \, \mathrm{d}V``.
Mainly what differs is how cells of *equal* buoyancy are placed, and what grid the answer lands on.
Here is a brief summary of the four methods, with more detail in the docstrings of each:

[`ThreeDimensionalSort`](@ref) (the default) ranks the cells and gives each one the height of its
own slot in the sorted state on the model grid. Tied cells take consecutive slots rather than a
shared height, so ``z^\star`` spreads over a grid cell wherever the stratification is horizontally
uniform. Crucially, this method may leave small horizontal buyoancy gradients in the reference state,
which goes against the idea of a reference state being horizontally uniform

[`VerticalSort`](@ref) is similar to [`ThreeDimensionalSort`](@ref) but returns the cells reorganized
into a single column. To achieve this the cells are flattened such that their volume remains the same, but their
horizontal area matches the domain's horizontal area. This has the advantage that the resulting reference
state (correctly) has no horizontal structure. The downside is that the results land on a different grid.
Namely a ``1 \times 1 \times N`` grid whose ``N = N_x N_y N_z`` cells span the domain's full horizontal area.

[`ProfileLookup`](@ref) gives each cell the height of the slot whose buoyancy matches its own, found by
binary search into a sorted profile. Cells are matched by value rather than by identity, so it is the
one method whose profile need not have come from the field being diagnosed. It reproduces on the model
grid what the [`VerticalSort`](@ref) column holds, without calling that method to do it.

```@example ape_profilelookup
using Oceananigans, Oceanostics
using Oceananigans.Fields: compute!, interior

grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1), topology=(Periodic, Periodic, Bounded))
model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b)
set!(model, b = (x, y, z) -> z)

# sort this field, exactly as the other methods do
z✶ = reference_height(model, method=ProfileLookup())

# borrow the column built by VerticalSort; it is recomputed alongside, so it tracks the flow
z✶_column = reference_height(model, method=VerticalSort())
z✶ = reference_height(model, method=ProfileLookup(z✶_column))

# any (b✶, z✶) pair of vectors, here a snapshot of the column held fixed in time
compute!(z✶_column)
b✶      = vec(interior(reference_buoyancy(z✶_column)))
z✶_prof = vec(interior(z✶_column))
z✶ = reference_height(model, method=ProfileLookup(b✶, z✶_prof))

nothing # hide
```

The last two forms do no sorting at all. To hold the reference state fixed while the flow evolves, pass
the last one *arrays*: `Field`s are recomputed on every `compute!`, so a profile given as `Field`s
tracks whatever they are built from instead of staying put. Note also that ``E_a \ge 0`` is guaranteed
only while the profile resolves the buoyancies the field actually has, which a profile sorted from that
same field always does.

[`HeavisideIntegral`](@ref) is Eq. (11) of Winters et al. (1995) verbatim,

```math
z^\star(\boldsymbol{x}) = \frac{1}{A} \int H\!\left(\rho(\boldsymbol{x}') - \rho(\boldsymbol{x})\right) \mathrm{d}V' ,
```

with the Heaviside step function ``H`` taking the value ``1/2`` where the two densities are equal.
That half-weight gives every cell of a given buoyancy the same ``z^\star``, the mid-height of the
layer that buoyancy class fills in the sorted column, which makes ``z^\star`` a function of buoyancy
alone and constant on isopycnals.

Because it builds ``z^\star`` from a volume fraction rather than by stacking cells into a column,
[`HeavisideIntegral`](@ref) is the only method that works on a **stretched grid** (one with non-uniform
cell volumes).

The [Lock release](@ref lock_release_example) example follows the reference profile, and the energy
split that goes with it, through a gravity current that starts as a step and ends well mixed.

## Computational cost per method

Sorting couples every cell in the domain to every other one, so unlike the pointwise diagnostics its
cost is not linear in the number of cells. All four methods pay for the same `sortperm!`, which is
``\mathcal{O}(N \log N)``; what separates them is the work done around it.
[`HeavisideIntegral`](@ref) makes extra passes over the sorted cells to find the tied runs,
[`ProfileLookup`](@ref) adds a binary search per cell, and [`VerticalSort`](@ref) carries the
buoyancy and the original heights into the column.

To measure that, build the same synthetic field — a linear stratification plus noise, so no two cells
are tied and the sort does its full work — on four grids spanning two decades in cell count, and time
one `compute!` of ``z^\star`` on each.

```@example ape_timing
using Oceananigans, Oceanostics, CairoMakie, Random
using Oceananigans.Fields: CenterField, compute!

function noisy_field(N)
    grid = RectilinearGrid(size = (N, N), x = (0, 1), z = (-1, 0), topology = (Periodic, Flat, Bounded))
    b = CenterField(grid)
    Random.seed!(42)
    set!(b, reshape(znodes(grid, Center()), 1, 1, N) .+ 0.1 .* randn(N, 1, N))
    return b
end

Ns      = (32, 64, 128, 256)
cells   = collect(Ns .^ 2)   # a Vector, so Makie can plot it directly
methods = ("ThreeDimensionalSort" => ThreeDimensionalSort(),
           "HeavisideIntegral"    => HeavisideIntegral(),
           "ProfileLookup"        => ProfileLookup(),
           "VerticalSort"         => VerticalSort())

## best of several runs, after a warm-up so compilation stays out of the measurement
function time_sort(N, method; samples = 7)
    z✶ = reference_height(noisy_field(N); method)
    compute!(z✶)
    return minimum(@elapsed(compute!(z✶)) for _ in 1:samples)
end

timings = Dict(name => [1e3 * time_sort(N, method) for N in Ns] for (name, method) in methods)
nothing # hide
```

Plotted against cell count on log axes, alongside an ``N \log N`` reference anchored at the largest
grid:

```@example ape_timing
fig = Figure(size = (620, 420))
ax  = Axis(fig[1, 1]; xlabel = "number of cells", ylabel = "time per sort (ms)",
           xscale = log10, yscale = log10, title = "Cost of building the reference state")

for (name, _) in methods
    scatterlines!(ax, cells, timings[name]; label = name, markersize = 10)
end

## N log N, scaled to meet the cheapest method at the largest grid
reference = cells .* log.(cells)
reference = reference ./ reference[end] .* timings["ThreeDimensionalSort"][end]
lines!(ax, cells, reference; color = :black, linestyle = :dash, label = "N log N")

axislegend(ax; position = :lt, labelsize = 11)
fig
```

The four curves run roughly parallel to the dashed reference, indicating that the sort is what sets the scaling.
If all you need are the volume integrals, the default is the cheapest route to them.

The absolute numbers are machine-dependent, and the docs are built with different optimisation
settings than a typical simulation, so read the *shape* of these curves rather than the values. The
spread between methods in particular is narrower here than on an optimised build.

## Background potential energy

```@docs
Oceanostics.BackgroundPotentialEnergyEquation.BackgroundPotentialEnergy
```

## Reference height and the methods that build it

```@docs
Oceanostics.BackgroundPotentialEnergyEquation.reference_height
Oceanostics.BackgroundPotentialEnergyEquation.reference_buoyancy
Oceanostics.BackgroundPotentialEnergyEquation.AbstractReferenceHeightMethod
Oceanostics.BackgroundPotentialEnergyEquation.ThreeDimensionalSort
Oceanostics.BackgroundPotentialEnergyEquation.HeavisideIntegral
Oceanostics.BackgroundPotentialEnergyEquation.ProfileLookup
Oceanostics.BackgroundPotentialEnergyEquation.VerticalSort
```
