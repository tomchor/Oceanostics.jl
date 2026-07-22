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

and what is left over is the available potential energy,

```math
E_a = E_p - E_b = -b (z - z^\star) ,
```

the part that an adiabatic rearrangement can release into kinetic energy. Importantly, the split matters
because the two halves respond to different physics: ``E_a`` is exchanged reversibly with kinetic
energy through the buoyancy flux ``wb``, while ``E_b`` can only be changed irreversibly.

``z^\star`` is computed by [`reference_height`](@ref), which returns a `Field` on the model
grid. Sorting is a non-local operation that couples every cell in the domain to every other one, so, unlike most other diagnostics
in Oceanostics, this one is not a pointwise kernel. It is re-sorted on each `compute!`, so writing it
(or anything built on it) out during a simulation tracks the evolving flow.

```julia
z✶ = reference_height(model)                  # share one sort between the two diagnostics
∫E_b = Integral(BackgroundPotentialEnergy(model, z✶))
∫E_a = Integral(AvailablePotentialEnergy(model, z✶))
```

The domain's horizontal cross-sectional area is assumed independent of depth, so an
`ImmersedBoundaryGrid` is rejected.

## The background potential energy and the reference state

The reference state is what characterizes ``E_b``. Usually this is achieved by sorting the buoyancy
field: rank every cell in the domain by buoyancy and stack the
cells from the bottom of the domain up, densest first. Each cell keeps its own volume, so it becomes
a slab of thickness ``\Delta V / A`` spanning the domain's horizontal area ``A``, and the height its
middle lands at is that cell's reference height ``z^\star``. Nothing is added or removed, only
rearranged, which is why the stack fills exactly the depth of the domain and why the rearrangement
counts as adiabatic. Weighting the buoyancy by ``z^\star`` instead of by ``z`` gives ``E_b``, the
potential energy the fluid would still hold once every parcel had been let down to its own level.

Plotting the buoyancy against the reference height it was assigned gives the reference profile
``b^\star(z^\star)``: the stratification the flow would have if all of its available potential energy
were released. The [Lock release](@ref lock_release_example) example follows that profile through a
gravity current, from the step it starts as to the smooth stratification mixing leaves behind, and
builds it with each of the three methods below so their costs and their differences can be compared
directly.

## Choosing how the reference state is built

`method` selects one of three strategies. They describe the same reference state and agree on every
volume integral, so ``\int E_b \, \mathrm{d}V`` and ``\int E_a \, \mathrm{d}V`` do not depend on the
choice. What differs is how cells of *equal* buoyancy are placed, and what grid the answer lands on.
Those are two separate axes: only [`OneDimensionalSort`](@ref) moves the answer off the model grid,
while the other two both stay on it and part ways over ties.

[`ThreeDimensionalSort`](@ref) (the default) ranks the cells and gives each one the height of its
own slot in the sorted column, on the model grid. Tied cells take consecutive slots rather than a
shared height, so ``z^\star`` spreads over a grid cell wherever the stratification is horizontally
uniform. The spread is the volume-weighted mean of what the next method assigns, so it cancels in
the integrals, but it does make a cell-by-cell map noisy in such regions.

[`HeavisideIntegral`](@ref) is eq. (11) of Winters et al. verbatim,

```math
z^\star(\boldsymbol{x}) = \frac{1}{A} \int H\!\left(\rho(\boldsymbol{x}') - \rho(\boldsymbol{x})\right) \mathrm{d}V' ,
```

with the Heaviside step function ``H`` taking the value ``1/2`` where the two densities are equal.
That half-weight gives every cell of a given buoyancy the same ``z^\star``, the mid-height of the
layer that buoyancy class fills in the sorted column, which makes ``z^\star`` a function of buoyancy
alone and constant on isopycnals, exactly as the paper describes it. A horizontally uniform,
statically stable stratification then gives ``z^\star = z`` and ``E_a = 0`` cell by cell rather than
only in the integral, so this is the method to use for local maps. It costs a couple of extra passes
over the sorted cells to find the tied runs.

[`OneDimensionalSort`](@ref) returns the sorted column itself, on a ``1 \times 1 \times N`` grid whose
``N = N_x N_y N_z`` cells span the domain's full horizontal area. The cells are reshaped rather than
re-counted, so each still holds the volume of a model-grid cell and volume integrals over the column
match those over the model grid. This is the form to reach for when you want the reference state as a
profile, to plot ``b^\star(z^\star)`` or differentiate it into a reference stratification. It needs
every cell of the model grid to hold the same volume, since otherwise the column's cell boundaries
would move as the flow evolves.

```julia
z✶ = reference_height(model, method=HeavisideIntegral()) # clean cell-by-cell maps
z✶ = reference_height(model, method=OneDimensionalSort()) # the reference profile itself
```

The [Lock release](@ref lock_release_example) example follows the reference profile, and the energy
split that goes with it, through a gravity current that starts as a step and ends well mixed.

## What the three methods cost

Sorting couples every cell in the domain to every other one, so unlike the pointwise diagnostics its
cost is not linear in the number of cells. All three methods pay for the same `sortperm!`, which is
``\mathcal{O}(N \log N)``; what separates them is the work done around it.
[`HeavisideIntegral`](@ref) makes extra passes over the sorted cells to find the tied runs, and
[`OneDimensionalSort`](@ref) carries the buoyancy and the original heights into the column.

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
           "OneDimensionalSort"   => OneDimensionalSort())

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

The three curves run parallel to the dashed reference, so the sort is what sets the scaling: a
fourfold increase in cells costs a little over fourfold in time, the excess being the ``\log N``.
Neither of the extra passes changes that exponent — they move the curve up, not tilt it — so the
choice of method is a constant factor, not a scaling penalty. If all you need are the volume
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
Oceanostics.AvailablePotentialEnergyEquation.OneDimensionalSort
```
