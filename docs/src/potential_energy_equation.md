# Potential energy equation

The `PotentialEnergyEquation` module provides a diagnostic for computing the
specific gravitational potential energy (per unit mass). In a Boussinesq fluid,
the specific potential energy is defined as

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

## Potential energy

```@docs
Oceanostics.PotentialEnergyEquation.PotentialEnergy
```
