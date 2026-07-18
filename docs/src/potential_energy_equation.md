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

Two caveats follow from doing the sort on a grid. The domain's horizontal cross-sectional area is
assumed independent of depth, so an `ImmersedBoundaryGrid` is rejected. And cells of exactly equal
buoyancy take consecutive slots in the sorted column instead of a shared height, which spreads
``z^\star`` over a grid cell where the stratification is horizontally uniform; that spread cancels
in the volume integrals these diagnostics are meant for, but it does make a cell-by-cell map of
``E_a`` noisy in such regions.

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
```
