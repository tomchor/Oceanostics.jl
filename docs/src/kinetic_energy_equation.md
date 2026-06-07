# Kinetic energy equation

The `KineticEnergyEquation` module provides diagnostics for every term in the
resolved kinetic energy (KE) budget. The kinetic energy per unit mass is defined as

```math
K = \tfrac{1}{2} u_i u_i
```

and its prognostic equation is obtained by contracting the momentum equation with
the velocity:

```math
\partial_t K = \underbrace{-u_i \partial_j(u_i u_j)}_{\text{advection}}
             + \underbrace{u_i \partial_j \tau_{ij}}_{\text{stress}}
             - \underbrace{u_i \partial_i p}_{\text{pressure}}
             + \underbrace{u_i b_i}_{\text{buoyancy}}
             + \underbrace{u_i F_{u_i}}_{\text{forcing}}
```

where ``\tau_{ij}`` is the viscous/subgrid stress tensor, ``p`` is pressure,
``b_i`` is the buoyancy acceleration component in the ``i``-th direction, and
``F_{u_i}`` is the forcing on the ``i``-th momentum equation.

This decomposition is essential for understanding how kinetic energy is generated
(e.g. by buoyancy production or forcing), redistributed (by advection or pressure work),
and removed (by viscous dissipation). The module also provides two formulations of the
dissipation rate: a general one based on the full stress tensor (``\varepsilon = \partial_j u_i \cdot F_{ij}``),
and an isotropic version (``\varepsilon = 2\nu S_{ij} S_{ij}``) valid when the
turbulence closure uses a single scalar viscosity.

All diagnostics are computed at `(Center, Center, Center)`.

## Example

```jldoctest ke_eq
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b, closure=ScalarDiffusivity(ν=1e-4));

julia> KE = KineticEnergyEquation.KineticEnergy(model)
KineticEnergy (KernelFunctionOperation) at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: kinetic_energy_ccc (generic function with 1 method)
└── arguments: ("Field", "Field", "Field")
└── computes: kinetic energy  ½uᵢuᵢ

julia> ε = KineticEnergyEquation.KineticEnergyIsotropicDissipationRate(model)
KineticEnergyIsotropicDissipationRate (KernelFunctionOperation) at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: isotropic_viscous_dissipation_rate_ccc (generic function with 1 method)
└── arguments: ("Field", "Field", "Field", "NamedTuple")
└── computes: isotropic kinetic energy dissipation rate  ε = 2νSᵢⱼSᵢⱼ

julia> wb = KineticEnergyEquation.BuoyancyProduction(model)
KineticEnergyBuoyancyProduction (KernelFunctionOperation) at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: uᵢbᵢᶜᶜᶜ (generic function with 1 method)
└── arguments: ("NamedTuple", "BuoyancyForce", "NamedTuple")
└── computes: kinetic energy buoyancy production  uᵢbᵢ
```

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
