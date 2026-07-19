# Potential energy equation

The `PotentialEnergyEquation` module provides diagnostics for the specific gravitational
potential energy (per unit mass) and for its split into a background and an available
part. In a Boussinesq fluid, the specific potential energy is defined as

```math
E_p = -bz = \frac{g\rho}{\rho_0} z
```

where ``b = -g\rho/\rho_0`` is buoyancy, ``z`` is the vertical coordinate, ``g`` is gravitational
acceleration, ``\rho`` is density, and ``\rho_0`` is a reference density. The quantity
``E_p`` has units of m² s⁻² (energy per unit mass).
Potential energy is a key quantity in ocean energetics: its conversion to/from
kinetic energy (via the buoyancy production term ``wb``) drives ocean circulation
and mixing.

`PotentialEnergy` is implemented for three buoyancy model types:

- **`BuoyancyTracer`**: uses the buoyancy field ``b`` directly as ``E_p = -bz``.
- **`SeawaterBuoyancy` with `LinearEquationOfState`**: computes buoyancy from a
  linear equation of state applied to temperature and/or salinity tracers.
- **`SeawaterBuoyancy` with `BoussinesqEquationOfState`** (from SeawaterPolynomials.jl):
  computes density from a nonlinear equation of state. An optional `geopotential_height`
  keyword argument allows using a potential density referenced to a fixed depth
  instead of in-situ density.

The diagnostic requires gravity to be aligned with the negative ``z``-direction
(`NegativeZDirection`).

## Example

```jldoctest pe_eq
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=100, z=(-1000, 0), topology=(Flat, Flat, Bounded));

julia> model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b);

julia> Ep = PotentialEnergyEquation.PotentialEnergy(model)
PotentialEnergy KernelFunctionOperation at (Center, Center, Center)
├── grid: 1×1×100 RectilinearGrid{Float64, Flat, Flat, Bounded} on CPU with 0×0×3 halo
├── kernel_function: minus_bz_ccc (generic function with 3 methods)
└── arguments: ("Field",)
└── computes: potential energy per unit volume  Eₚ = -bz
```

## Background and available potential energy

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

### Choosing how the reference state is built

`method` selects one of three strategies. They describe the same reference state and agree on every
volume integral, so ``\int E_b \, \mathrm{d}V`` and ``\int E_a \, \mathrm{d}V`` do not depend on the
choice. What differs is how cells of *equal* buoyancy are placed, and what grid the answer lands on.

[`CellRanking`](@ref) (the default) ranks the cells and gives each one the height of its own slot in
the sorted column, on the model grid. Tied cells take consecutive slots rather than a shared height,
so ``z^\star`` spreads over a grid cell wherever the stratification is horizontally uniform. The
spread is the volume-weighted mean of what the next method assigns, so it cancels in the integrals,
but it does make a cell-by-cell map noisy in such regions.

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

## Potential energy

```@docs
Oceanostics.PotentialEnergyEquation.PotentialEnergy
```

## Background potential energy

```@docs
Oceanostics.PotentialEnergyEquation.BackgroundPotentialEnergy
```

## Available potential energy

```@docs
Oceanostics.PotentialEnergyEquation.AvailablePotentialEnergy
```

## Sorted reference height

```@docs
Oceanostics.PotentialEnergyEquation.sorted_reference_height
Oceanostics.PotentialEnergyEquation.AbstractSortingMethod
Oceanostics.PotentialEnergyEquation.CellRanking
Oceanostics.PotentialEnergyEquation.HeavisideIntegral
Oceanostics.PotentialEnergyEquation.OneDimensionalSort
```
