# Available potential energy equation

The `AvailablePotentialEnergyEquation` module splits the specific potential energy
``E_p = -bz`` of [the potential energy equation](potential_energy_equation.md) into the part the
flow can release and the part it cannot.

Not all of ``E_p`` is available to the flow. Following
[Winters et al. (1995)](https://doi.org/10.1017/S002211209500125X), rearrange the buoyancy field
adiabatically into the state of minimum potential energy, with every parcel sorted by density and
the densest fluid at the bottom. The height a parcel ends up at is its reference height
``z^\star``, and the potential energy of the reference state is the background (or reference)
potential energy,

```math
E_b = -b z^\star = \frac{g \rho}{\rho_0} z^\star ,
```

and what is left over is the available potential energy, which Oceanostics computes it in its *local*
form [Holliday & McIntyre (1981)](https://doi.org/10.1017/S0022112081001742):

```math
E_a(b, z) = \int_{z^\star}^{z} \left[b^\star(\tilde z) - b\right] \mathrm{d}\tilde z
          = \frac{g}{\rho_0}\int_{z^\star}^{z} \left[\rho - \rho^\star(\tilde z)\right] \mathrm{d}\tilde z .
```

In this form ``E_a`` is **non-negative everywhere in space**, so it can be mapped as a field. Its
volume integral recovers ``\int E_p - \int E_b`` in the continuum limit, although at finite ``\Delta z`` the
two differ at second order.

## The background potential energy and the reference state

The reference height ``z^\star`` is the height a parcel would occupy in that state and it
is computed by [`reference_height`](@ref), which returns a `Field`. Sorting is a nonlocal
operation, so, unlike most other diagnostics in Oceanostics, this one is not a pointwise kernel.
It is re-sorted on each `compute!`, so writing it (or anything built on it) out during a simulation
tracks the evolving flow.

Organizing the buoyancy against the reference height it was assigned gives the reference profile
``b^\star(z^\star)``: the stratification the flow would have if all of its available potential energy
were released. The [Lock release](@ref lock_release_example) example follows that profile through a
gravity current, from the step it starts as to the smooth stratification mixing leaves behind, and
builds it with each of the four methods below so their costs and their differences can be compared
directly.

`method` selects one of four strategies to calculate the reference state. All `method`s produce the same reference state and agree on every
volume integral, so ``\int E_b \, \mathrm{d}V`` and ``\int E_a \, \mathrm{d}V`` do not depend on the
choice. Mainly what differs is how cells of *equal* buoyancy are placed, and what grid the answer lands on.

That freedom is one ``E_a`` cannot see. A cell's ``z^\star`` always lands inside the run of slots that
its own buoyancy fills, and the reference profile is flat across that run, so sliding ``z^\star`` along
it leaves ``E_a`` unchanged. The four therefore agree on ``E_a`` cell by cell, not merely in the
integral; where they part company is ``z^\star`` itself, and so ``E_b``.

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

```julia
z✶ = reference_height(model, method=ProfileLookup())            # sort this field, as the others do
z✶ = reference_height(model, method=ProfileLookup(z✶_column))   # borrow a column, recomputed with it
z✶ = reference_height(model, method=ProfileLookup(b✶, z✶_prof)) # any profile, e.g. one fixed in time
```

The last two forms do no sorting at all, and the last is the way to hold the reference state fixed
while the flow evolves. Note that ``E_a \ge 0`` is guaranteed only while the profile resolves the
buoyancies the field actually has, which a profile sorted from that same field always does.

[`HeavisideIntegral`](@ref) is Eq. (11) of Winters et al. (1995) verbatim,

```math
z^\star(\boldsymbol{x}) = \frac{1}{A} \int H\!\left(\rho(\boldsymbol{x}') - \rho(\boldsymbol{x})\right) \mathrm{d}V' ,
```

with the Heaviside step function ``H`` taking the value ``1/2`` where the two densities are equal.
That half-weight gives every cell of a given buoyancy the same ``z^\star``, the mid-height of the
layer that buoyancy class fills in the sorted column, which makes ``z^\star`` a function of buoyancy
alone and constant on isopycnals.

The [Lock release](@ref lock_release_example) example follows the reference profile, and the energy
split that goes with it, through a gravity current that starts as a step and ends well mixed.

## What the methods cost

Sorting couples every cell in the domain to every other one, so unlike the pointwise diagnostics its
cost is not linear in the number of cells. All four methods pay for the same `sortperm!`, which is
``\mathcal{O}(N \log N)``; what separates them is the work done around it.
[`HeavisideIntegral`](@ref) makes extra passes over the sorted cells to find the tied runs,
[`ProfileLookup`](@ref) adds a binary search per cell, and [`VerticalSort`](@ref) carries the
buoyancy and the original heights into the column.

To measure that, build the same synthetic field — a linear stratification plus noise, so no two cells
are tied and the sort does its full work — on four grids spanning two decades in cell count, and time
one `compute!` of `z^\star` on each.

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

The four curves run parallel to the dashed reference, so the sort is what sets the scaling: a
fourfold increase in cells costs a little over fourfold in time, the excess being the ``\log N``.
None of the extra work changes that exponent. It moves the curves up rather than tilting them, so the
choice of method is a constant factor and not a scaling penalty. If all you need are the volume
integrals, the default is the cheapest route to them.

The absolute numbers are machine-dependent, and the docs are built with different optimisation
settings than a typical simulation, so read the *shape* of these curves rather than the values. The
spread between methods in particular is narrower here than on an optimised build.

## Background potential energy

```@docs
Oceanostics.AvailablePotentialEnergyEquation.BackgroundPotentialEnergy
```

## Available potential energy

```@docs
Oceanostics.AvailablePotentialEnergyEquation.AvailablePotentialEnergy
```

## Sorted reference height

```@docs
Oceanostics.AvailablePotentialEnergyEquation.reference_height
Oceanostics.AvailablePotentialEnergyEquation.reference_buoyancy
Oceanostics.AvailablePotentialEnergyEquation.AbstractSortingMethod
Oceanostics.AvailablePotentialEnergyEquation.ThreeDimensionalSort
Oceanostics.AvailablePotentialEnergyEquation.HeavisideIntegral
Oceanostics.AvailablePotentialEnergyEquation.ProfileLookup
Oceanostics.AvailablePotentialEnergyEquation.VerticalSort
```
