# Kinetic energy equation

The `KineticEnergyEquation` module provides diagnostics for every term in the resolved kinetic energy
(KE) budget. The kinetic energy per unit mass is

```math
e_k = \tfrac{1}{2} u_i u_i ,
```

where ``i`` runs over the prognostic velocity components of the model: ``i = 1, 2, 3`` on a
`NonhydrostaticModel`, which steps ``u``, ``v`` and ``w``, and ``i = 1, 2`` on a
`HydrostaticFreeSurfaceModel`, which steps ``u`` and ``v`` and diagnoses ``w`` from continuity. Currently,
all diagnostics are computed at `(Center, Center, Center)`.

The budget of ``e_k`` follows from contracting the momentum equations with the velocity. The two model
types differ in where buoyancy and the free surface enter it, so each is briefly derived in turn below, and
[the table](@ref ke_terms_table) that follows lists every quantity with its expression on each
model and the diagnostic that computes it.

## Nonhydrostatic models

A `NonhydrostaticModel` steps the Boussinesq momentum equations

```math
\partial_t u_i + \partial_j (u_j u_i) + (\boldsymbol{f} \times \boldsymbol{u})_i
  = -\partial_i p' - \partial_i p_{nh} + b \, \delta_{i3} - \partial_j \tau_{ij} + F_{u_i} ,
  \qquad \partial_j u_j = 0 , \qquad i = 1, 2, 3 ,
```

where ``\tau_{ij}`` is the viscous or subgrid momentum flux Oceananigans' kernels carry (the negative
of the stress tensor, ``-2\nu S_{ij}`` for a constant viscosity), ``F_{u_i}`` is the forcing, and the
kinematic pressure is split into a hydrostatic anomaly and a nonhydrostatic part, ``p = p' + p_{nh}``,
with ``\partial_z p' = b`` and ``p' = 0`` at the top. The split is Oceananigans' default when there is
a buoyancy model and the vertical direction is bounded; with `hydrostatic_pressure_anomaly = nothing`
the whole pressure is ``p_{nh}`` and ``b`` stays in the ``w`` equation. The nonhydrostatic pressure
comes out of the Poisson problem that enforces ``\partial_j u_j = 0``, and with a `free_surface` its
value at the top is ``g \eta``.

Contracting with ``u_i`` gives, term by term: ``u_i \partial_t u_i = \partial_t e_k``; the advection
``u_i \partial_j (u_j u_i)``, which equals ``\partial_j (u_j e_k)`` for a divergence-free flow; nothing
from the Coriolis acceleration, which is normal to the velocity; ``wb`` from the buoyancy force; the
viscous work ``-u_i \partial_j \tau_{ij} = -\partial_j (u_i \tau_{ij}) - \varepsilon_k``, with
``\varepsilon_k = -\tau_{ij} \partial_j u_i`` the dissipation rate; and the pressure work
``-u_i \partial_i (p' + p_{nh})``. The budget reads

```math
\partial_t e_k = - \underbrace{u_i \partial_j(u_j u_i)}_{\text{advection}}
                 - \underbrace{u_i \partial_i (p' + p_{nh})}_{\text{pressure work}}
                 + \underbrace{wb}_{\text{conversion}}
                 + \underbrace{u_i F_{u_i}}_{\text{forcing}}
                 - \underbrace{u_i \partial_j \tau_{ij}}_{\text{viscous work}}
                 , \qquad i = 1, 2, 3 ,
```

## Hydrostatic models

A `HydrostaticFreeSurfaceModel` steps the horizontal momentum equations

```math
\partial_t u_i + \partial_j (u_j u_i) + (\boldsymbol{f} \times \boldsymbol{u})_i
  = -\partial_i p' - \partial_i (g \eta) - \partial_j \tau_{ij} + F_{u_i}
    , \qquad i = 1, 2 , \quad j = 1, 2, 3 ,
```

with hydrostatic balance, ``\partial_z p' = b`` and ``p' = 0`` at the top, in place of a ``w``
equation. ``w`` comes from continuity, integrated up from the bottom, and the free surface from
``\partial_t \eta = -\nabla_h \cdot \boldsymbol{U}``, with ``\boldsymbol{U}`` the depth-integrated
transport, which by continuity is ``w`` at the surface. The total pressure is ``p' + g \eta``.

Contracting with ``u_i``, ``i = 1, 2``, goes as before, with ``\partial_j u_j = 0`` still holding for
the diagnosed ``w``, and no buoyancy term emerges:

```math
\partial_t e_k = - \underbrace{u_i \partial_j(u_j u_i)}_{\text{advection}}
                 - \underbrace{u_i \partial_i p'}_{\text{pressure work}}
                 - \underbrace{u_i \partial_i (g \eta)}_{\text{barotropic pressure work}}
                 + \underbrace{u_i F_{u_i}}_{\text{forcing}}
                 - \underbrace{u_i \partial_j \tau_{ij}}_{\text{viscous work}}
                 , \qquad i = 1, 2 , \quad j = 1, 2, 3 .
```

Buoyancy reaches ``e_k`` through ``p'`` alone. Using continuity and ``\partial_z p' = b``, the work of
the hydrostatic anomaly is

```math
-u_i \partial_i p' = -\partial_j (u_j p') + wb , \qquad i = 1, 2, \quad j = 1, 2, 3 ,
```

so the conversion ``wb`` is present pointwise, inside the work of ``p'`` together with a transport
term, which is why `PotentialToKineticEnergyConversion` uses the diagnosed ``w`` on a hydrostatic
model. The package writes the budget in the same form as the nonhydrostatic one. Adding and
subtracting the vertical hydrostatic work ``w \, \partial_z p'``, which is ``wb`` by hydrostatic
balance, and gathering the two pressure gradients into one, turns the pressure work into the
three-component contraction with the total pressure ``p' + g \eta`` plus the conversion,

```math
\partial_t e_k = - \underbrace{u_i \partial_j(u_j u_i)}_{\text{advection}}
                 - \underbrace{u_j \partial_j (p' + g \eta)}_{\text{pressure work}}
                 + \underbrace{wb}_{\text{conversion}}
                 + \underbrace{u_i F_{u_i}}_{\text{forcing}}
                 - \underbrace{u_i \partial_j \tau_{ij}}_{\text{viscous work}}
                 , \qquad i = 1, 2 , \quad j = 1, 2, 3 ,
```

where the pressure work now contracts all three components with the diagnosed ``w`` and the total
pressure, and is the same transport term as on the nonhydrostatic model, the divergence of the pressure
flux ``u_j (p' + g \eta)``, while ``wb`` is the same conversion. This is the form
`KineticEnergyPressureRedistribution` computes by default on a hydrostatic model.
An `ImplicitFreeSurface` or a `SplitExplicitFreeSurface` applies ``-g \nabla_h \eta`` in its free-surface
step, outside the tendency kernel, so `KineticEnergyTendency` leaves the barotropic part out and the
change of ``e_k`` over a step is ``u_i G_i - u_h \cdot \nabla_h (g \eta)``; with an
`ExplicitFreeSurface` the gradient is inside ``G_i`` and that part is already counted.

## Vertically implicit closures

`KineticEnergyTendency` and `KineticEnergyStress` wrap Oceananigans' explicit tendency and its viscous
term, which apply the closure's time discretization. A `VerticallyImplicitTimeDiscretization` (CATKE's
default, and an option of `ScalarDiffusivity` and `VerticalScalarDiffusivity`) applies the
``\partial_z u`` and ``\partial_z v`` parts of the vertical fluxes in the implicit step, so both
diagnostics leave that viscous work out, while `KineticEnergyDissipationRate`, built from the closure's
explicit fluxes, keeps it. The terms still rebuild `KineticEnergyTendency` cell by cell, but that
tendency is no longer the change of ``e_k`` over a step and
``\int u_i \partial_j \tau_{ij} \, \mathrm{d}V = \int \varepsilon_k \, \mathrm{d}V`` fails, so an integrated
budget should use ``\varepsilon_k``. Hydrostatic runs usually carry such a closure.
[The potential energy page](potential_energy_equation.md) has the same split between
`PotentialEnergyBuoyancyDiffusion` and `PotentialEnergyDiffusiveVerticalBuoyancyFlux`.

## Immersed boundaries

On an `ImmersedBoundaryGrid` the momentum tendencies subtract the interior ``\partial_j \tau_{ij}`` and
the stress through the immersed faces separately. `KineticEnergyStress` wraps the interior term only
and `KineticEnergyTendency` carries both. The immersed stress vanishes under the default no-flux
immersed conditions, but with an immersed `ValueBoundaryCondition` or a drag `FluxBoundaryCondition`
on the velocities the budget is short by the wall-stress work, which
[the momentum equation](momentum_equation.md) exposes per component as `ImmersedViscousDissipation`.
`KineticEnergyDissipationRate` applies no masking across immersed faces, so on such a grid it includes
shear across faces the model treats as solid and
``\int u_i \partial_j \tau_{ij} \, \mathrm{d}V = \int \varepsilon_k \, \mathrm{d}V`` does not hold.

## Background fields

Background fields are not supported yet. On a `NonhydrostaticModel` built with `background_fields`,
`model.velocities` holds the perturbation from the background flow, and every term except
`KineticEnergyTendency` is built from the perturbation alone. `KineticEnergyTendency` comes off the
model's own kernel, which also advects the perturbation with the background flow and the background flow
with the perturbation. With `BackgroundFields(; background_closure_fluxes = true)` its viscous term is
the closure flux of the total velocities, while `KineticEnergyStress` and
`KineticEnergyDissipationRate` take the flux of the perturbation. No diagnostic here supplies the
difference, so on such a model the terms do not add up to `KineticEnergyTendency`.

## [Terms and diagnostics](@id ke_terms_table)

Every quantity below is pointwise. ``G_i`` is Oceananigans' momentum tendency, and the hydrostatic ``w`` is the one diagnosed from
continuity. The advection, the pressure work and the viscous work enter the budget with a minus sign,
the conversion and the forcing with a plus sign.

| Quantity | Nonhydrostatic | Hydrostatic | Diagnostic |
|:---|:---|:---|:---|
| Kinetic energy | ``e_k = \tfrac{1}{2} u_i u_i``, with ``i = 1, 2, 3`` | same, with ``i = 1, 2`` | [`KineticEnergy`](@ref Oceanostics.KineticEnergyEquation.KineticEnergy) |
| Tendency | ``u_i G_i = \partial_t e_k + u_i \partial_i p_{nh}`` | ``u_i G_i = \partial_t e_k + u_i \partial_i (g \eta)`` with ``i = 1, 2`` | [`KineticEnergyTendency`](@ref Oceanostics.KineticEnergyEquation.KineticEnergyTendency) |
| Advection | ``u_i \partial_j (u_j u_i)`` | same, with ``i = 1, 2`` and the diagnosed ``w`` in ``u_j`` | [`KineticEnergyAdvection`](@ref Oceanostics.KineticEnergyEquation.KineticEnergyAdvection) |
| Pressure work | ``u_j \partial_j (p' + p_{nh})`` | ``u_j \partial_j (p' + g \eta)``, ``j = 1, 2, 3``, with the diagnosed ``w`` | [`KineticEnergyPressureRedistribution`](@ref Oceanostics.KineticEnergyEquation.KineticEnergyPressureRedistribution) |
| Conversion from potential energy | ``wb`` | same, with the diagnosed ``w`` | [`PotentialToKineticEnergyConversion`](@ref Oceanostics.KineticEnergyEquation.PotentialEnergyConversion) |
| Forcing work | ``u_i F_{u_i}``, ``i = 1, 2, 3`` | same, with ``i = 1, 2`` | [`KineticEnergyForcing`](@ref Oceanostics.KineticEnergyEquation.KineticEnergyForcing) |
| Viscous work | ``u_i \partial_j \tau_{ij}`` | same, with ``i = 1, 2`` | [`KineticEnergyStress`](@ref Oceanostics.KineticEnergyEquation.KineticEnergyStress) |
| Dissipation rate | ``\varepsilon_k = -\tau_{ij} \partial_j u_i`` | ``\varepsilon_k = -(\tau_{1j} \partial_j u + \tau_{2j} \partial_j v)`` | [`KineticEnergyDissipationRate`](@ref Oceanostics.KineticEnergyEquation.DissipationRate) |
| Isotropic dissipation rate | ``2 \nu S_{ij} S_{ij}``, equal to ``\varepsilon_k`` for a constant viscosity | ``2 \nu S_{ij} S_{ij}``, which keeps the ``w`` row and so is not the dissipation of this budget | [`KineticEnergyIsotropicDissipationRate`](@ref Oceanostics.KineticEnergyEquation.KineticEnergyIsotropicDissipationRate) |

`using Oceanostics` brings in `KineticEnergyAdvection`, `KineticEnergyForcing`, etc.
At the same time, `using Oceanostics.KineticEnergyEquation` brings in the short aliases:
`Advection`, `Forcing`, etc.


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

## Summary of ``e_k`` equation terms

```@docs
Oceanostics.KineticEnergyEquation.KineticEnergy
Oceanostics.KineticEnergyEquation.KineticEnergyTendency
Oceanostics.KineticEnergyEquation.KineticEnergyAdvection
Oceanostics.KineticEnergyEquation.KineticEnergyStress
Oceanostics.KineticEnergyEquation.KineticEnergyForcing
Oceanostics.KineticEnergyEquation.KineticEnergyPressureRedistribution
Oceanostics.KineticEnergyEquation.PotentialEnergyConversion
Oceanostics.KineticEnergyEquation.DissipationRate
Oceanostics.KineticEnergyEquation.KineticEnergyIsotropicDissipationRate
```
