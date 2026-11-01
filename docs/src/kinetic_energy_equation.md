# Kinetic energy equation

The `KineticEnergyEquation` module provides diagnostics for every term in the resolved kinetic energy
(KE) budget. The kinetic energy per unit mass is

```math
e_k = \tfrac{1}{2} u_i u_i ,
```

where ``i`` runs over the prognostic velocity components of the model: ``i = 1, 2, 3`` on a
`NonhydrostaticModel`, which steps ``u``, ``v`` and ``w``, and ``i = 1, 2`` on a
`HydrostaticFreeSurfaceModel`, which steps ``u`` and ``v`` and diagnoses ``w`` from continuity. Every
diagnostic of the module sums over the same indices, except the pressure term, which keeps the
diagnosed ``w`` on a hydrostatic model for the reason given in its section. As throughout Oceanostics,
the lower-case ``e_k`` is the pointwise energy density and the upper-case
``E_k = \int e_k \, \mathrm{d}V`` its volume integral;
[the potential energy page](potential_energy_equation.md) states the convention in full. All
diagnostics are computed at `(Center, Center, Center)`.

The budget of ``e_k`` follows from contracting the momentum equations with the velocity. The two model
types differ in where buoyancy and the free surface enter it, so each is derived in turn below, and
[the table](@ref ke_terms_table) that follows lists the terms side by side with the function that
computes each.

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
the whole pressure is ``p_{nh}`` and ``b`` stays in the ``w`` equation. With the split, the ``w``
equation carries neither ``b`` nor ``\partial_z p'``, since the two cancel. The nonhydrostatic pressure
comes out of the Poisson problem that enforces ``\partial_j u_j = 0``, and with a `free_surface` its
value at the top is ``g \eta``.

Contracting with ``u_i`` gives, term by term: ``u_i \partial_t u_i = \partial_t e_k``; the advection
``u_i \partial_j (u_j u_i)``, which equals ``\partial_j (u_j e_k)`` for a divergence-free flow; nothing
from the Coriolis acceleration, which is normal to the velocity; ``wb`` from the buoyancy force; the
viscous work ``-u_i \partial_j \tau_{ij} = -\partial_j (u_i \tau_{ij}) - \varepsilon_k``, with
``\varepsilon_k = -\tau_{ij} \partial_j u_i`` the dissipation rate; and the pressure work
``-u_i \partial_i (p' + p_{nh})``. The vertical part of the hydrostatic work is
``-w \, \partial_z p' = -wb``, which cancels the buoyancy term: when the model carries ``p'``, buoyancy
does its work through the horizontal gradient of ``p'``. The budget reads

```math
\partial_t e_k = -\underbrace{u_i \partial_j(u_j u_i)}_{\texttt{ADV}}
             - \underbrace{u_i \partial_j \tau_{ij}}_{\texttt{STRESS}}
             - \underbrace{u_i \partial_i (p' + p_{nh})}_{\texttt{PR}}
             + wb
             + \underbrace{u_i F_{u_i}}_{\texttt{FORC}} , \qquad i = 1, 2, 3 ,
```

with ``wb`` and the ``-wb`` inside ``-\texttt{PR}`` cancelling cell by cell.
`KineticEnergyPressureRedistribution` uses ``p' + p_{nh}`` by default (``p_{nh}`` alone when the model
has no anomaly) and `PotentialToKineticEnergyConversion` is ``wb``. `KineticEnergyTendency` is
``u_i G_i``, Oceananigans' tendency contracted with the velocity, which leaves out the nonhydrostatic
pressure work the pressure correction applies: ``\partial_t e_k = u_i G_i - u_i \partial_i p_{nh}``.
With a free surface, ``\texttt{PR}`` contains the work of the surface pressure ``g \eta``, whose
barotropic part ``u_h \cdot \nabla_h (g \eta)`` is `KineticEnergyBarotropicPressureRedistribution`; the
default rigid lid has none. The dissipation rate ``\varepsilon_k`` is `KineticEnergyDissipationRate`;
`KineticEnergyIsotropicDissipationRate` is ``2 \nu S_{ij} S_{ij}``, equal to it for a constant isotropic
viscosity, and both are non-negative for a down-gradient closure.

## Hydrostatic models

A `HydrostaticFreeSurfaceModel` steps the horizontal momentum equations

```math
\partial_t u_i + \partial_j (u_j u_i) + (\boldsymbol{f} \times \boldsymbol{u})_i
  = -\partial_i p' - \partial_i (g \eta) - \partial_j \tau_{ij} + F_{u_i} , \qquad i = 1, 2 ,
```

with hydrostatic balance, ``\partial_z p' = b`` and ``p' = 0`` at the top, in place of a ``w``
equation. ``w`` comes from continuity, integrated up from the bottom, and the free surface from
``\partial_t \eta = -\nabla_h \cdot \boldsymbol{U}``, with ``\boldsymbol{U}`` the depth-integrated
transport, which by continuity is ``w`` at the surface. The total pressure is ``p' + g \eta``: the
surface pressure ``g \eta`` is uniform in ``z``, the same value the nonhydrostatic model gives its
``p_{nh}`` at the top. Buoyancy does not appear in these equations.

Contracting with ``u_i``, ``i = 1, 2``, goes as before, with ``\partial_j u_j = 0`` still holding for
the diagnosed ``w``, and no buoyancy term arises:

```math
\partial_t e_k = -\underbrace{u_i \partial_j(u_j u_i)}_{\texttt{ADV}}
             - \underbrace{u_i \partial_j \tau_{ij}}_{\texttt{STRESS}}
             - \underbrace{u_i \partial_i p'}_{\texttt{PR}}
             - \underbrace{u_i \partial_i (g \eta)}_{\texttt{BPR}}
             + \underbrace{u_i F_{u_i}}_{\texttt{FORC}} , \qquad i = 1, 2 .
```

Buoyancy reaches ``e_k`` through ``p'`` alone. Using continuity and ``\partial_z p' = b``, the work of
the hydrostatic anomaly is

```math
-u_i \partial_i p' = -\partial_j (u_j p') + wb , \qquad i = 1, 2 ,
```

so the conversion ``wb`` is present pointwise, inside the work of ``p'`` together with a transport
term, which is why `PotentialToKineticEnergyConversion` uses the diagnosed ``w`` on a hydrostatic
model. The package writes the budget in the same form as the nonhydrostatic one. Adding and
subtracting the vertical hydrostatic work ``w \, \partial_z p'``, which is ``wb`` by hydrostatic
balance, turns the two-component pressure work into the three-component one plus the conversion,

```math
\partial_t e_k = -\underbrace{u_i \partial_j(u_j u_i)}_{\texttt{ADV}}
             - \underbrace{u_i \partial_j \tau_{ij}}_{\texttt{STRESS}}
             - \underbrace{u_j \partial_j p'}_{\texttt{PR}}
             - \underbrace{u_i \partial_i (g \eta)}_{\texttt{BPR}}
             + wb
             + \underbrace{u_i F_{u_i}}_{\texttt{FORC}} , \qquad i = 1, 2 , \quad j = 1, 2, 3 ,
```

where ``\texttt{PR}`` now contracts all three components with the diagnosed ``w`` and is the same
transport term as on the nonhydrostatic model, the divergence of the pressure flux ``u_j p'``, while
``wb`` is the same conversion. This is the form `KineticEnergyPressureRedistribution` computes by
default on a hydrostatic model, with ``p = p'``; the two-component work of the previous equation is
``\texttt{PR} - wb``. On Oceananigans' grid the added and subtracted pieces cancel cell by cell at the
interior faces, since ``p'`` is integrated from the same face buoyancy that ``wb`` multiplies by
``w``; at the top face the difference of ``p'`` reads its halo, so with a free surface, where ``w``
does not vanish there, the top cell is short by half the ``wb`` of its upper face, as on a
nonhydrostatic model. The momentum fluxes inside `KineticEnergyStress` and
`KineticEnergyDissipationRate` also use the diagnosed ``w``, as the model's own tendencies do. The
last pressure term is the work of the barotropic pressure gradient,
`KineticEnergyBarotropicPressureRedistribution`. An `ImplicitFreeSurface` or a
`SplitExplicitFreeSurface` applies ``-g \nabla_h \eta`` in its free-surface step, outside the tendency
kernel, so `KineticEnergyTendency` leaves it out and the change of ``e_k`` over a step is
``u_i G_i - \texttt{BPR}``; with an `ExplicitFreeSurface` the gradient is inside ``G_i`` and the term
is already counted.

The dissipation rate of this budget is the first two rows of the general form,
``\varepsilon_k = -(\tau_{1j} \partial_j u + \tau_{2j} \partial_j v)``, which
`KineticEnergyDissipationRate` computes, so ``\int \varepsilon_k \, \mathrm{d}V`` is the viscous sink
of ``\tfrac{1}{2}(u^2 + v^2)``. Pointwise its sign depends on the closure: a horizontal or vertical
formulation (`HorizontalScalarDiffusivity`, `VerticalScalarDiffusivity`, CATKE) builds ``\tau`` from
``u`` and ``v`` alone and gives a non-negative ``\varepsilon_k`` cell by cell, while a
three-dimensional `ScalarDiffusivity` puts ``\partial_x w`` into ``\tau_{13}``, whose cross term
``\nu \, \partial_z u \, \partial_x w`` has no ``w`` row to pair with, so ``\varepsilon_k`` is
indefinite pointwise by a term of order aspect ratio squared relative to ``\nu (\partial_z u)^2``; the
integral is unaffected. `KineticEnergyIsotropicDissipationRate` keeps the ``w`` row on both model
types, so on a hydrostatic model it includes the dissipation of ``\tfrac{1}{2} w^2`` and is not the
dissipation term of this budget.

## [Terms side by side](@id ke_terms_table)

Every row below holds at a point. ``u_h`` is the horizontal velocity, ``\nabla_h`` the horizontal
gradient, ``G_i`` Oceananigans' tendency, and ``e_s = \tfrac{1}{2} g \eta^2`` the potential energy of
the free surface per unit area, described in [the free surface section](@ref free_surface_energy).
``\texttt{PR}``, ``\texttt{BPR}`` and ``\texttt{ADV}`` all enter the budget with a minus sign.

| Quantity | Nonhydrostatic | Hydrostatic | In Oceanostics |
| --- | --- | --- | --- |
| Kinetic energy ``e_k`` | ``\tfrac{1}{2}(u^2 + v^2 + w^2)`` | ``\tfrac{1}{2}(u^2 + v^2)`` | `KineticEnergy`, over the prognostic velocities of each model |
| Pressure in the momentum equations | ``-\nabla p' - \nabla p_{nh}``, with ``p_{nh} = g \eta`` at the surface | ``-\nabla_h p' - g \nabla_h \eta`` | Nothing in this module; per component, `PressureGradient` and `BarotropicPressureGradient` in [the momentum modules](momentum_equation.md) |
| Pressure work as the model steps it | ``-u_h \cdot \nabla_h p' - u_i \partial_i p_{nh}`` | ``-u_h \cdot \nabla_h p' - u_h \cdot \nabla_h (g \eta)`` | ``-\texttt{PR} + wb`` on the nonhydrostatic model and ``-\texttt{PR} - \texttt{BPR} + wb`` on the hydrostatic one: `KineticEnergyPressureRedistribution` contracts all three components on both, with ``p = p' + p_{nh}`` and ``p = p'`` by default, so it holds the ``-wb`` that `PotentialToKineticEnergyConversion` cancels; `KineticEnergyBarotropicPressureRedistribution` is the barotropic part |
| ``wb`` from the buoyancy force of the momentum equations | ``wb``, cancelled by ``-w \, \partial_z p'`` | none | `PotentialToKineticEnergyConversion`, a term of both budgets as the package writes them: it cancels the vertical row of ``\texttt{PR}``, which is ``wb`` by hydrostatic balance |
| Pressure work in flux form | ``-\partial_j [u_j (p' + p_{nh})] + wb`` | ``-\partial_j [u_j (p' + g \eta)] + wb`` | The flux convergence is ``-\texttt{PR}`` on the nonhydrostatic model and ``-\texttt{PR} - \texttt{BPR}`` on the hydrostatic one, since ``u_j \partial_j p = \partial_j (u_j p)``; the residual ``wb`` is `PotentialToKineticEnergyConversion` |
| Normal pressure flux at the surface | ``g \eta \, w = \partial_t e_s`` | ``g \eta \, w = \partial_t e_s`` | `FreeSurfacePotentialEnergyTendency` (``g \eta \, w`` at the surface) and `FreeSurfacePotentialEnergy` (``e_s``), on [the potential energy page](potential_energy_equation.md) |
| Normal pressure flux at the bottom | ``0`` at every point, any bathymetry | ``0`` at every point, any bathymetry | Nothing to compute |
| What one has and the other lacks | the transport ``-\partial_j [u_j (p_{nh} - g \eta)]`` by the nonhydrostatic pressure proper | | `KineticEnergyPressureRedistribution` with `pressure = model.pressures.pNHS`, less ``\texttt{BPR}``, sign flipped |
| Advection | ``-u_i \partial_j (u_j u_i)`` | ``-u_i \partial_j (u_j u_i)``, with the diagnosed ``w`` | ``-\texttt{ADV}``, `KineticEnergyAdvection` |
| Viscous and forcing work | ``-u_i \partial_j \tau_{ij} + u_i F_{u_i}`` | the same, ``i = 1, 2`` | ``-\texttt{STRESS}``, `KineticEnergyStress`; its dissipative part is `KineticEnergyDissipationRate`; the forcing work is `KineticEnergyForcing` |
| Tendency ``\partial_t e_k`` | ``u_i G_i - u_i \partial_i p_{nh}`` | ``u_i G_i - \texttt{BPR}`` with an implicit or split-explicit free surface | `KineticEnergyTendency` is ``u_i G_i``, Oceananigans' tendency contracted with the velocity; it leaves out the work the model applies outside its tendency |

## [Free surface](@id free_surface_energy)

A free surface adds the pressure ``g \eta`` to the budget and a reservoir to exchange with. The work of
its gradient, ``\texttt{BPR}``, is a flux divergence at every point, so its volume integral is a boundary
quantity. With closed or periodic lateral boundaries and ``\boldsymbol{U}`` the depth-integrated
transport,

```math
\int \texttt{BPR} \, \mathrm{d}V = \int \boldsymbol{U} \cdot \nabla_h (g \eta) \, \mathrm{d}A
  = -\int g \eta \, \nabla_h \cdot \boldsymbol{U} \, \mathrm{d}A
  = \int g \eta \, \partial_t \eta \, \mathrm{d}A
  = \frac{\mathrm{d}}{\mathrm{d}t} \int e_s \, \mathrm{d}A ,
```

where ``e_s = \tfrac{1}{2} g \eta^2`` is the potential energy of the free surface per unit area,
`FreeSurfacePotentialEnergy`, and ``g \eta \, \partial_t \eta = g \eta \, w|_{\mathrm{surface}}`` its
tendency, `FreeSurfacePotentialEnergyTendency`, both described on
[the potential energy page](potential_energy_equation.md). Pointwise ``\texttt{BPR}`` is of the order
of the other pressure work; integrated it is small whenever the barotropic flow is narrow compared with
the external deformation radius ``\sqrt{gH}/f``, which is why a budget can look closed without it. With
it, the three reservoirs close together,

```math
\frac{\mathrm{d}}{\mathrm{d}t} \left( \int e_k \, \mathrm{d}V + \int e_p \, \mathrm{d}V + \int e_s \, \mathrm{d}A \right)
  = \int \left( \texttt{FORC} - \varepsilon_k + \Phi \right) \mathrm{d}V - \int e_k \, w \, \mathrm{d}A ,
```

the last integral over the surface: the domain is open there, since ``w`` at the top is
``\partial_t \eta``, so ``\int \texttt{ADV} \, \mathrm{d}V`` is the flux of ``e_k`` through the surface
rather than zero, of order ``\eta / H`` relative to the rest. (``\int \texttt{Advection} \, \mathrm{d}V``
of ``e_p`` is zero all the same, since its surface flux carries a factor ``z = 0``.) On a
`NonhydrostaticModel` built with a `free_surface` the term sits inside ``\texttt{PR}``, which carries
``g \eta`` as the surface value of ``p_{nh}``; the rest of that work integrates to zero.

The model applies ``\nabla_h \eta`` at the new time level, or averaged over its barotropic substeps,
while the diagnostics read ``\eta`` and ``w`` as they are when evaluated, so a budget built from them
closes to ``O(\Delta t)``, the order of a `TimeDerivative`. The differences of ``\eta`` reach into its
halos along the lateral boundaries, which a `NonhydrostaticModel` fills when ``\eta`` is set and never
again (Oceananigans 0.116 and earlier); on that model the term reads a copy of ``\eta`` that is refilled
whenever the operation is computed, so it is evaluated through a `Field` or an output writer rather than
by indexing the operation. A `MutableVerticalDiscretization`, whose cells move with ``\eta``, is outside
what these terms describe.

## Vertically implicit closures

`KineticEnergyTendency` and `KineticEnergyStress` wrap Oceananigans' explicit tendency and its viscous
term, which apply the closure's time discretization. A `VerticallyImplicitTimeDiscretization` (CATKE's
default, and an option of `ScalarDiffusivity` and `VerticalScalarDiffusivity`) applies the
``\partial_z u`` and ``\partial_z v`` parts of the vertical fluxes in the implicit step, so both
diagnostics leave that viscous work out, while `KineticEnergyDissipationRate`, built from the closure's
explicit fluxes, keeps it. The terms still rebuild `KineticEnergyTendency` cell by cell, but that
tendency is no longer the change of ``e_k`` over a step and
``\int \texttt{STRESS} \, \mathrm{d}V = \int \varepsilon_k \, \mathrm{d}V`` fails, so an integrated
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
``\int \texttt{STRESS} \, \mathrm{d}V = \int \varepsilon_k \, \mathrm{d}V`` does not hold.

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

## Barotropic pressure redistribution

```@docs
Oceanostics.KineticEnergyEquation.KineticEnergyBarotropicPressureRedistribution
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
