# Kinetic energy equation

The `KineticEnergyEquation` module provides diagnostics for the terms in the
kinetic energy budget:

```math
\partial_t \tfrac{1}{2} u_i^2 = -u_i \partial_j(u_i u_j) + u_i \partial_j \tau_{ij} - u_i \partial_i p + u_i b_i + u_i F_{u_i}
```

All diagnostics are computed at `(Center, Center, Center)`.

## Kinetic energy

```@docs
Oceanostics.KineticEnergyEquation.KineticEnergy
```

## Tendency

```@docs
Oceanostics.KineticEnergyEquation.KineticEnergyTendency
```

## Advection

```@docs
Oceanostics.KineticEnergyEquation.KineticEnergyAdvection
```

## Stress (diffusive) term

```@docs
Oceanostics.KineticEnergyEquation.KineticEnergyStress
```

## Forcing

```@docs
Oceanostics.KineticEnergyEquation.KineticEnergyForcing
```

## Pressure redistribution

```@docs
Oceanostics.KineticEnergyEquation.KineticEnergyPressureRedistribution
```

## Buoyancy production

```@docs
Oceanostics.KineticEnergyEquation.BuoyancyProduction
```

## Dissipation rate

```@docs
Oceanostics.KineticEnergyEquation.DissipationRate
Oceanostics.KineticEnergyEquation.KineticEnergyIsotropicDissipationRate
```
