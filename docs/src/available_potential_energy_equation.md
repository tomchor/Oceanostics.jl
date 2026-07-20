# Available potential energy equation

The `AvailablePotentialEnergyEquation` module splits the specific potential energy
``E_p = -bz`` of [the potential energy equation](potential_energy_equation.md) into the part the
flow can release and the part it cannot.

Not all of ``E_p`` is available to the flow. Following
[Winters et al. (1995)](https://doi.org/10.1017/S002211209500125X), rearrange the buoyancy field
adiabatically into the state of minimum potential energy, with every parcel sorted by density and
the densest fluid at the bottom. The height a parcel ends up at is its reference height
``z^\star``, and the potential energy of the sorted state is the background (or reference)
potential energy,

```math
E_b = -b z^\star = \frac{g \rho}{\rho_0} z^\star ,
```

and what is left over is the available potential energy,

```math
E_a = E_p - E_b = -b (z - z^\star) ,
```

the part that an adiabatic rearrangement can release into kinetic energy. The split matters
because the two halves respond to different physics: ``E_a`` is exchanged reversibly with kinetic
energy through the buoyancy flux ``wb``, while ``E_b`` can only be changed irreversibly. In a
closed domain the continuous equations make ``\int E_b \, \mathrm{d}V`` grow monotonically at the
diapycnal mixing rate, which is what lets it separate stirring from mixing in a simulation.
Numerically it also registers whatever spurious diapycnal transport the advection scheme
introduces, in either direction, which is why it doubles as a standard measure of a scheme's
mixing.

``z^\star`` is computed by [`sorted_reference_height`](@ref), which returns a `Field` on the model
grid. Sorting couples every cell in the domain to every other one, so unlike every other diagnostic
in Oceanostics this one is not a pointwise kernel. It is re-sorted on each `compute!`, so writing it
(or anything built on it) out during a simulation tracks the evolving flow.

```julia
z✶ = sorted_reference_height(model)                  # share one sort between the two diagnostics
∫E_b = Integral(BackgroundPotentialEnergy(model, z✶))
∫E_a = Integral(AvailablePotentialEnergy(model, z✶))
```

The domain's horizontal cross-sectional area is assumed independent of depth, so an
`ImmersedBoundaryGrid` is rejected.

## The background potential energy and the sorted state

Sorting is what turns ``E_p`` into ``E_b``. Rank every cell in the domain by buoyancy and stack the
cells from the bottom of the domain up, densest first. Each cell keeps its own volume, so it becomes
a slab of thickness ``\Delta V / A`` spanning the domain's horizontal area ``A``, and the height its
middle lands at is that cell's reference height ``z^\star``. Nothing is added or removed, only
rearranged, which is why the stack fills exactly the depth of the domain and why the rearrangement
counts as adiabatic. Weighting the buoyancy by ``z^\star`` instead of by ``z`` gives ``E_b``, the
potential energy the fluid would still hold once every parcel had been let down to its own level.

A small synthetic case makes that concrete. Take a two-layer fluid of buoyancy ``0`` and ``1`` on a
2×4 grid, with the middle of the interface overturned so that the field is not already its own sorted
state. The grid is chosen so the arithmetic stays visible: cells hold a volume of 2 against a
horizontal area of 2, so each becomes a slab one unit thick and ``z^\star`` lands on half-integers.

```jldoctest sorted_state
using Oceananigans, Oceanostics

grid = RectilinearGrid(size=(2, 4), x=(0, 2), z=(-8, 0), topology=(Periodic, Flat, Bounded))
model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b)

# rows are the two columns of cells in x; entries run from the bottom of the domain up
b₀ = [0 0 1 1
      0 1 0 1]
set!(model, b = reshape(b₀, 2, 1, 4))

∫dV(op) = sum(Field(Integral(op)))
∫dV(PotentialEnergy(model)), ∫dV(BackgroundPotentialEnergy(model)), ∫dV(AvailablePotentialEnergy(model))

# output

(20.0, 16.0, 4.0)
```

Four units of the twenty are available: that is what the two overturned cells could release by
swapping back. The other sixteen are locked in the sorted state and only irreversible mixing can
touch them.

The three methods below all describe that same sorted state, and each returns ``E_b = 16``. What
differs is where the eight ``z^\star`` values live and how the *tied* cells are placed, and here every
cell is tied with three others, since there are only two distinct buoyancies.

[`ThreeDimensionalSort`](@ref) hands each cell its own slab, so the four dense cells take the four
slots between ``-8`` and ``-4`` and the four light cells take those between ``-4`` and ``0``. Which
tied cell gets which slot is arbitrary, and the spread over each layer is visible here:

```jldoctest sorted_state
z✶ = sorted_reference_height(model, method=ThreeDimensionalSort())
Array(reshape(interior(z✶), 2, 4))

# output

2×4 Matrix{Float64}:
 -7.5  -5.5  -2.5  -1.5
 -6.5  -3.5  -4.5  -0.5
```

[`HeavisideIntegral`](@ref) instead gives every cell of a given buoyancy the mid-height of the layer
that buoyancy fills, which is the ``1/2`` weight eq. (11) puts on equal densities. The dense layer
occupies ``[-8, -4]`` and the light layer ``[-4, 0]``, so ``z^\star`` collapses onto ``-6`` and
``-2``. The result is a function of buoyancy alone, constant on each isopycnal, and reads as a map
rather than as a ranking:

```jldoctest sorted_state
z✶ = sorted_reference_height(model, method=HeavisideIntegral())
Array(reshape(interior(z✶), 2, 4))

# output

2×4 Matrix{Float64}:
 -6.0  -6.0  -2.0  -2.0
 -6.0  -2.0  -6.0  -2.0
```

[`OneDimensionalSort`](@ref) returns the stack itself rather than a map of it: a `1×1×8` column of
unit-thick slabs, carrying the sorted buoyancy profile ``b^\star`` alongside the heights it belongs
to. This is the form to use when you want the reference stratification as a profile:

```jldoctest sorted_state
z✶ = sorted_reference_height(model, method=OneDimensionalSort())
Array(vec(interior(sorted_buoyancy(z✶)))), Array(vec(interior(z✶)))

# output

([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], [-7.5, -6.5, -5.5, -4.5, -3.5, -2.5, -1.5, -0.5])
```

So the three disagree cell by cell but agree on the integral. They have to: the heights
[`ThreeDimensionalSort`](@ref) spreads across a layer average out, over that layer's volume, to the
single height [`HeavisideIntegral`](@ref) assigns it, and every cell in the layer carries the same
buoyancy for that average to act on:

```jldoctest sorted_state
map((ThreeDimensionalSort(), HeavisideIntegral(), OneDimensionalSort())) do method
    ∫dV(BackgroundPotentialEnergy(model; method))
end

# output

(16.0, 16.0, 16.0)
```

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
z✶ = sorted_reference_height(model, method=HeavisideIntegral()) # clean cell-by-cell maps
z✶ = sorted_reference_height(model, method=OneDimensionalSort()) # the reference profile itself
```

The [Kelvin-Helmholtz instability](@ref kelvin_helmholtz_example) example tracks the three energy
reservoirs through a mixing event.

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
Oceanostics.AvailablePotentialEnergyEquation.sorted_reference_height
Oceanostics.AvailablePotentialEnergyEquation.sorted_buoyancy
Oceanostics.AvailablePotentialEnergyEquation.AbstractSortingMethod
Oceanostics.AvailablePotentialEnergyEquation.ThreeDimensionalSort
Oceanostics.AvailablePotentialEnergyEquation.HeavisideIntegral
Oceanostics.AvailablePotentialEnergyEquation.OneDimensionalSort
```
