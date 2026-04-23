# Potential energy equation

The `PotentialEnergyEquation` module provides a diagnostic for gravitational
potential energy per unit volume:

```math
E_p = -bz = \frac{g\rho}{\rho_0} z
```

where ``b`` is buoyancy, ``z`` is the vertical coordinate, ``g`` is gravitational
acceleration, ``\rho`` is density, and ``\rho_0`` is a reference density.

`PotentialEnergy` supports `BuoyancyTracer`, `SeawaterBuoyancy` with a
`LinearEquationOfState`, and `SeawaterBuoyancy` with a `BoussinesqEquationOfState`
(from SeawaterPolynomials.jl). For the latter, an optional `geopotential_height`
keyword allows computing potential energy with a potential density reference level
instead of in-situ density.

## Potential energy

```@docs
Oceanostics.PotentialEnergyEquation.PotentialEnergy
```
