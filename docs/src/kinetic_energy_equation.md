# Kinetic energy equation

The `KineticEnergyEquation` module provides diagnostics for every term in the
resolved kinetic energy (KE) budget. The kinetic energy per unit mass is defined as

```math
e_k = \tfrac{1}{2} u_i u_i
```

and its prognostic equation is obtained by contracting the momentum equation with
the velocity:

```math
\partial_t e_k = \underbrace{-u_i \partial_j(u_i u_j)}_{\text{advection}}
             - \underbrace{u_i \partial_j \tau_{ij}}_{\text{stress}}
             - \underbrace{u_i \partial_i p}_{\text{pressure}}
             + \underbrace{u_i b_i}_{\text{buoyancy}}
             + \underbrace{u_i F_{u_i}}_{\text{forcing}}
```

where ``\tau_{ij}`` is the viscous/subgrid momentum flux (negative of the stress tensor), ``p`` is pressure,
``b_i`` is the buoyancy acceleration component in the ``i``-th direction, and
``F_{u_i}`` is the forcing on the ``i``-th momentum equation. As throughout Oceanostics, the
lower-case ``e_k`` is the pointwise energy density and the upper-case ``E_k = \int e_k \, \mathrm{d}V``
its volume integral; [the potential energy page](potential_energy_equation.md) states the convention
in full.

This decomposition is essential for understanding how kinetic energy is generated
(e.g. by buoyancy production or forcing), redistributed (by advection or pressure work),
and removed (by viscous dissipation). The module also provides two formulations of the
dissipation rate: a general one based on the full momentum flux
(``\varepsilon_k = -\partial_j u_i \, \tau_{ij}``), and an isotropic version
(``\varepsilon_k = 2\nu S_{ij} S_{ij}``) valid when the turbulence closure uses a single scalar
viscosity. The two agree for a constant viscosity, and both are non-negative for a down-gradient
closure. As in [the momentum equation](momentum_equation.md), ``\tau_{ij}`` is the momentum flux
Oceananigans' kernels carry (i.e. , the negative of the stress tensor; ``-2\nu S_{ij}`` for a constant viscosity).

All diagnostics are computed at `(Center, Center, Center)`.

## Example

```jldoctest ke_eq
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b, closure=ScalarDiffusivity(ν=1e-4));

julia> eₖ = KineticEnergyEquation.KineticEnergy(model)
KineticEnergy KernelFunctionOperation at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: kinetic_energy_ccc (generic function with 1 method)
└── arguments: ("Field", "Field", "Field")
└── computes: kinetic energy  ½uᵢuᵢ

julia> εₖ = KineticEnergyEquation.KineticEnergyIsotropicDissipationRate(model)
KineticEnergyIsotropicDissipationRate KernelFunctionOperation at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: isotropic_viscous_dissipation_rate_ccc (generic function with 1 method)
└── arguments: ("Field", "Field", "Field", "NamedTuple")
└── computes: isotropic kinetic energy dissipation rate  2νSᵢⱼSᵢⱼ

julia> wb = KineticEnergyEquation.PotentialEnergyConversion(model)
PotentialToKineticEnergyConversion KernelFunctionOperation at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: uᵢbᵢᶜᶜᶜ (generic function with 1 method)
└── arguments: ("NamedTuple", "BuoyancyForce", "NamedTuple")
└── computes: potential to kinetic energy conversion  uᵢbᵢ
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

## Potential energy conversion

```@docs
Oceanostics.KineticEnergyEquation.PotentialEnergyConversion
```

## Dissipation rate

```@docs
Oceanostics.KineticEnergyEquation.DissipationRate
Oceanostics.KineticEnergyEquation.KineticEnergyIsotropicDissipationRate
```
