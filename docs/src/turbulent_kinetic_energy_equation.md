# Turbulent kinetic energy equation

The `TurbulentKineticEnergyEquation` module provides diagnostics for the turbulent
kinetic energy (TKE) budget. TKE is defined as the kinetic energy of the velocity
perturbations relative to a mean flow:

```math
e = \tfrac{1}{2} u_i' u_i'
```

where ``u_i' = u_i - U_i`` is the velocity perturbation from the mean ``U_i``.
Mean velocities default to zero (in which case TKE equals KE) and can be
specified via the `U`, `V`, `W` keyword arguments.

## Turbulent kinetic energy

```@docs
Oceanostics.TurbulentKineticEnergyEquation.TurbulentKineticEnergy
```

## Isotropic dissipation rate

```@docs
Oceanostics.TurbulentKineticEnergyEquation.TurbulentKineticEnergyIsotropicDissipationRate
```

## Shear production rates

```@docs
Oceanostics.TurbulentKineticEnergyEquation.TurbulentKineticEnergyXShearProductionRate
Oceanostics.TurbulentKineticEnergyEquation.TurbulentKineticEnergyYShearProductionRate
Oceanostics.TurbulentKineticEnergyEquation.TurbulentKineticEnergyZShearProductionRate
Oceanostics.TurbulentKineticEnergyEquation.TurbulentKineticEnergyShearProductionRate
```
