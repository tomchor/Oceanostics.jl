# Turbulent kinetic energy equation

The `TurbulentKineticEnergyEquation` module provides diagnostics for the turbulent
kinetic energy (TKE) budget. While the `KineticEnergyEquation` module deals with the
total (resolved) kinetic energy, this module focuses on the kinetic energy of velocity
perturbations relative to a specified mean flow. TKE is defined as

```math
e = \tfrac{1}{2} u_i' u_i'
```

where ``u_i' = u_i - U_i`` is the velocity perturbation from the mean ``U_i``.
Mean velocities default to zero (in which case TKE reduces to KE) and can be
set via the `U`, `V`, `W` keyword arguments.

A key term in the TKE budget is the shear production rate, which quantifies
the transfer of kinetic energy from the mean flow to the turbulence through
Reynolds stresses acting on the mean shear:

```math
P = -u_i' u_j' \partial_j U_i
```

The module provides both directional components (``P_x``, ``P_y``, ``P_z``) and
the total shear production. It also provides an isotropic dissipation rate
diagnostic that computes ``\varepsilon = 2\nu S'_{ij} S'_{ij}`` using the
perturbation strain rate tensor.

All diagnostics are computed at `(Center, Center, Center)`.

## Example

```jldoctest tke_eq
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(grid; closure=ScalarDiffusivity(ν=1e-4));

julia> tke = TurbulentKineticEnergyEquation.TurbulentKineticEnergy(model)
TurbulentKineticEnergy (KernelFunctionOperation) at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: turbulent_kinetic_energy_ccc (generic function with 1 method)
└── arguments: ("Field", "Field", "Field", "Oceananigans.Fields.ZeroField", "Oceananigans.Fields.ZeroField", "Oceananigans.Fields.ZeroField")
└── computes: turbulent kinetic energy  ½uᵢ′uᵢ′

julia> SP = TurbulentKineticEnergyEquation.ShearProductionRate(model)
TurbulentKineticEnergyShearProductionRate (KernelFunctionOperation) at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: shear_production_rate_ccc (generic function with 1 method)
└── arguments: ("Field", "Field", "Field", "Oceananigans.Fields.ZeroField", "Oceananigans.Fields.ZeroField", "Oceananigans.Fields.ZeroField")
└── computes: total TKE shear production  -uᵢ′uⱼ′ ∂ⱼUᵢ
```

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
