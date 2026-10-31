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
viscosity. On a `NonhydrostaticModel` the two agree for a constant viscosity, and both are non-negative
for a down-gradient closure; on a hydrostatic model only the general one is a term of the budget, as
discussed below. As in [the momentum equation](momentum_equation.md), ``\tau_{ij}`` is the momentum flux
Oceananigans' kernels carry (i.e. , the negative of the stress tensor; ``-2\nu S_{ij}`` for a constant viscosity).

All diagnostics are computed at `(Center, Center, Center)`.

## Hydrostatic models

A `HydrostaticFreeSurfaceModel` steps ``u`` and ``v`` and diagnoses ``w`` from continuity, so on such a
model the kinetic energy is ``e_k = \tfrac{1}{2}(u^2 + v^2)``, and `KineticEnergy`,
`KineticEnergyTendency`, `KineticEnergyAdvection`, `KineticEnergyStress`, `KineticEnergyForcing`,
`KineticEnergyPressureRedistribution` and `KineticEnergyDissipationRate` all sum over ``u`` and ``v``
only. The momentum fluxes inside the stress term and the dissipation rate are still computed with the
diagnosed ``w``, as in the model's own tendencies. Buoyancy does not appear in the hydrostatic momentum
equations. It enters through the hydrostatic pressure anomaly ``p'``, with ``\partial_z p' = b``, so the
budget reads

```math
\partial_t e_k = -u_i (\boldsymbol{u} \cdot \nabla) u_i - u_i \partial_j \tau_{ij} - u_i \partial_i p'
                 + u_i F_{u_i} , \qquad i = 1, 2 ,
```

cell by cell, apart from the work of the barotropic pressure gradient, which an `ImplicitFreeSurface` or
a `SplitExplicitFreeSurface` treats implicitly and `KineticEnergyTendency` therefore leaves out. The
conversion ``wb`` sits inside the pressure work,

```math
-u_i \partial_i p' = -\partial_j (u_j p') + w b , \qquad i = 1, 2 ,
```

which is why `PotentialToKineticEnergyConversion` uses the diagnosed ``w`` on a hydrostatic model. A
`NonhydrostaticModel` that carries a hydrostatic pressure anomaly (Oceananigans' default when there is a
buoyancy model and the vertical direction is bounded) is arranged the same way: its ``w`` equation has no
buoyancy term, and ``wb`` reaches the budget through ``w \partial_z p'`` instead.

The dissipation rate of this budget is the viscous work above less a transport divergence,

```math
-u_i \partial_j \tau_{ij} = -\partial_j (u_i \tau_{ij}) - \varepsilon_k , \qquad
\varepsilon_k = -(\tau_{1j} \partial_j u + \tau_{2j} \partial_j v) ,
```

and `KineticEnergyDissipationRate` computes exactly that, the first two rows of the general form, so
``\int \varepsilon_k \, \mathrm{d}V`` is the viscous sink of ``\tfrac{1}{2}(u^2 + v^2)``; for a constant
isotropic viscosity it equals ``\nu \int (|\nabla u|^2 + |\nabla v|^2) \, \mathrm{d}V``. Pointwise, its
sign depends on the closure's formulation. A horizontal or vertical formulation
(`HorizontalScalarDiffusivity`, `VerticalScalarDiffusivity`, CATKE) builds ``\tau`` from ``u`` and ``v``
alone, and ``\varepsilon_k`` is then non-negative cell by cell: ``\nu_h (\delta^2 + \zeta^2)`` from the
horizontal divergence and vorticity, plus ``\nu_z [(\partial_z u)^2 + (\partial_z v)^2]``. A
three-dimensional `ScalarDiffusivity` puts ``\partial_x w`` into ``\tau_{13}``. In the full budget the
cross term ``\nu \, \partial_z u \, \partial_x w`` pairs with the ``w`` row to complete
``\nu (\partial_z u + \partial_x w)^2``, but the hydrostatic energy has no ``w`` row, so the cross term
stands alone and ``\varepsilon_k`` is indefinite pointwise, by a term of order aspect ratio squared
relative to ``\nu (\partial_z u)^2``, which is the order the hydrostatic approximation neglects
elsewhere; the integral is unaffected. `KineticEnergyIsotropicDissipationRate` is
``2\nu S_{ij} S_{ij}`` of the full strain rate and keeps the ``w`` row on both model types, so on a
hydrostatic model it is not the dissipation term of this budget: it includes the dissipation of
``\tfrac{1}{2} w^2``, which the budget does not carry.

## Vertically implicit closures

`KineticEnergyTendency` and `KineticEnergyStress` wrap Oceananigans' explicit tendency and its viscous
term ``\partial_j \tau_{ij}``, which apply the closure's time discretization. A closure with a
`VerticallyImplicitTimeDiscretization`, which is CATKE's default and an option of `ScalarDiffusivity`
and `VerticalScalarDiffusivity`, takes the ``\partial_z u`` and ``\partial_z v`` parts of its vertical
fluxes out of the explicit tendency and applies them in the implicit step, so both diagnostics leave out
the vertical viscous work the model does. `KineticEnergyDissipationRate` is built from the closure's
explicit fluxes and keeps it. Under such a closure the terms still rebuild `KineticEnergyTendency` cell
by cell, but that tendency is no longer the change of ``e_k`` over a step, the identity
``\int \mathrm{DIFF}\, \mathrm{d}V = \int \varepsilon_k\, \mathrm{d}V`` fails, and an integrated budget
should be written with ``\varepsilon_k`` in place of ``\mathrm{DIFF}``. Hydrostatic runs usually carry
such a closure, so there this is the common case. The potential energy budget splits the same way, with
`PotentialEnergyTendency` and `PotentialEnergyBuoyancyDiffusion` on the explicit side and
`PotentialEnergyDiffusiveVerticalBuoyancyFlux` keeping the implicit flux; see
[the potential energy page](potential_energy_equation.md).

## Immersed boundaries

On an `ImmersedBoundaryGrid` the momentum tendencies subtract two viscous terms, the interior
``\partial_j \tau_{ij}`` and the stress through the immersed faces, which Oceananigans keeps separate.
`KineticEnergyStress` wraps the interior term only, while `KineticEnergyTendency` carries both. The
immersed stress vanishes under the default no-flux immersed boundary conditions, but with an immersed
`ValueBoundaryCondition` or a drag `FluxBoundaryCondition` on the velocities the budget above is short by
the wall-stress work, which no diagnostic here provides; [the momentum equation](momentum_equation.md)
exposes it per component as `ImmersedViscousDissipation`. `KineticEnergyDissipationRate` is built from
the closure's explicit fluxes without the masking that the interior term applies across immersed faces,
so on such a grid it includes shear across faces the model treats as solid and
``\int \mathrm{DIFF}\, \mathrm{d}V = \int \varepsilon_k\, \mathrm{d}V`` does not hold.

## Background fields

Background fields are not supported. On a `NonhydrostaticModel` built with `background_fields`,
`model.velocities` holds the perturbation from the background flow, and every term except
`KineticEnergyTendency` is built from the perturbation alone. `KineticEnergyTendency` comes off the
model's own kernel, which also advects the perturbation with the background flow and the background flow
with the perturbation. With `BackgroundFields(; background_closure_fluxes = true)` its viscous term is
the closure flux of the total velocities, while `KineticEnergyStress` and
`KineticEnergyDissipationRate` take the flux of the perturbation. No diagnostic here supplies the
difference, so on such a model the terms do not add up to `KineticEnergyTendency`.

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
