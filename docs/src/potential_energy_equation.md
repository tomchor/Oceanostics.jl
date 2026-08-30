# Potential energy equation

The `PotentialEnergyEquation` module provides a diagnostic for the specific gravitational
potential energy (per unit mass). In a Boussinesq fluid, the specific potential energy is defined as

```math
e_p = -bz = \frac{g\rho}{\rho_0} z
```

where ``b = -g\rho/\rho_0`` is buoyancy, ``z`` is the vertical coordinate, ``g`` is gravitational
acceleration, ``\rho`` is density, and ``\rho_0`` is a reference density. The quantity
``e_p`` has units of m² s⁻² (energy per unit mass).

!!! note "Lower case for densities, upper case for their integrals"
    Throughout Oceanostics a lower-case ``e`` is an energy density, the pointwise quantity a
    diagnostic returns, and the matching upper-case ``E`` is its volume integral,
    ```math
    E_p = \int e_p \, \mathrm{d}V ,
    ```
    and likewise for the kinetic (``e_k``, ``E_k``), background (``e_b``, ``E_b``) and available
    (``e_a``, ``E_a``) potential energies. A term shared between budgets carries the subscript of the
    budget it belongs to, so ``\varepsilon_k`` is the kinetic energy dissipation rate and
    ``\varepsilon_a`` the available potential energy one. A superscript ``l`` marks a quantity of the
    low-pass **filtered** field and a superscript ``s`` its **subfilter** complement, as in
    ``e_k = e_k^l + e_k^s`` under a filter.

## The potential energy equation

Throughout what follows we assume that `gravity_unit_vector` points towards the `NegativeZDirection`,
as is the default in Oceananigans, but tilted domains where this is not true are possible.
Since ``e_p`` is just ``-z`` times the buoyancy, and ``z`` does not change in time, the equation for
``e_p`` follows directly from [the tracer equation](tracer_equation.md) applied to ``b``. In the
convention Oceananigans uses (and ignoring background fields), that equation reads

```math
\partial_t b = -\partial_j (u_j b) - \partial_j q_j + F_b ,
```

where ``q_j`` is the diffusive flux of buoyancy supplied by the closure (``q_j = -\kappa\,\partial_j b``
for Fickian diffusion with diffusivity ``\kappa``) and ``F_b`` is any applied forcing. Multiplying
through by ``-z``,

```math
\partial_t e_p = -z\,\partial_t b
               = z\,\partial_j(u_j b) + z\,\partial_j q_j - z F_b .
```

Pulling ``z`` inside the derivatives separates them, and since ``\partial_j z`` is non-zero only in the
vertical, the product rule gives

```math
\partial_t e_p = \underbrace{-\partial_j(u_j e_p)}_{\text{advection}}
                 \underbrace{- wb}_{\text{PE to KE conversion}}
                 + \underbrace{\partial_j(z q_j)}_{\text{diffusive transport}}
                 \underbrace{- q_3}_{\text{diffusive buoyancy flux}}
                 \underbrace{- z F_b}_{\text{forcing}}.
```

The two terms written as divergences transport ``e_p`` and vanish when integrated over a periodic or closed domain
(with impermeable, insulating walls).

## Terms and diagnostics

Every term above is implemented, two of them elsewhere in Oceanostics.

| Quantity | Expression | Diagnostic |
|:---|:---|:---|
| Potential energy | ``e_p = -bz`` | [`PotentialEnergy`](@ref Oceanostics.PotentialEnergyEquation.PotentialEnergy) |
| Tendency | ``\partial_t e_p = -z\,\partial_t b`` | [`PotentialEnergyTendency`](@ref Oceanostics.PotentialEnergyEquation.PotentialEnergyTendency) |
| Buoyancy advection | ``z\,\partial_j(u_j b) = -\partial_j(u_j e_p) - wb`` | [`PotentialEnergyBuoyancyAdvection`](@ref Oceanostics.PotentialEnergyEquation.PotentialEnergyBuoyancyAdvection) |
| Advection | ``\partial_j(u_j e_p)`` | [`PotentialEnergyAdvection`](@ref Oceanostics.PotentialEnergyEquation.PotentialEnergyAdvection) |
| PE to KE conversion | ``wb`` | [`PotentialToKineticEnergyConversion`](@ref Oceanostics.KineticEnergyEquation.PotentialEnergyConversion) |
| Buoyancy diffusion | ``z\,\partial_j q_j = \partial_j(z q_j) - q_3`` | [`PotentialEnergyBuoyancyDiffusion`](@ref Oceanostics.PotentialEnergyEquation.PotentialEnergyBuoyancyDiffusion) |
| Diffusive transport | ``\partial_j(z q_j)`` | [`PotentialEnergyDiffusion`](@ref Oceanostics.PotentialEnergyEquation.PotentialEnergyDiffusion) |
| Diffusive vertical buoyancy flux | ``\Phi = -q_3`` | [`DiffusiveVerticalBuoyancyFlux`](@ref Oceanostics.PotentialEnergyEquation.DiffusiveVerticalBuoyancyFlux) |
| Forcing | ``-z F_b`` | [`PotentialEnergyForcing`](@ref Oceanostics.PotentialEnergyEquation.PotentialEnergyForcing) |

Each of the two flux terms appears twice, once in each of the forms the derivation above relates:

```math
\underbrace{z\,\partial_j(u_j b)}_{\texttt{BuoyancyAdvection}}
  = -\underbrace{\partial_j(u_j e_p)}_{\texttt{Advection}} - wb , \qquad
\underbrace{z\,\partial_j q_j}_{\texttt{BuoyancyDiffusion}}
  = \underbrace{\partial_j(z q_j)}_{\texttt{Diffusion}} + \Phi .
```

The `Buoyancy*` pair are the ``-z\,\times`` forms, which is the convention
[the kinetic energy equation](kinetic_energy_equation.md) follows as well (``u_i\partial_j(u_ju_i)``
rather than ``\partial_j(u_j e_k)``). Taken that way they are the model's own buoyancy tendency split
apart, so

```math
\texttt{Tendency} = \texttt{BuoyancyAdvection} + \texttt{BuoyancyDiffusion} + \texttt{Forcing}
```

holds cell by cell rather than to within a truncation error.

Note that the KE to PE conversion term can go by three names since it is shared by different budgets.
It stays defined in [the kinetic energy equation](kinetic_energy_equation.md), where
`PotentialEnergyConversion` names what it does to the kinetic energy. `using Oceanostics` brings in
`PotentialToKineticEnergyConversion`, which names the exchange and says which way it runs. This module
and [the available potential energy equation](available_potential_energy_equation.md) both re-export
that and additionally give it `KineticEnergyConversion`, short enough to read well beside the other
terms of a potential energy budget. That last alias is scoped to those two modules: `using Oceanostics`
does not bring it in, because unprefixed it says nothing about which budget it belongs to.

``e_p`` also splits into two parts that do have their own modules. The
[background potential energy](background_potential_energy_equation.md) ``e_b`` is the portion the flow
could never release, and the [available potential energy](available_potential_energy_equation.md)
``e_a = e_p - e_b`` is the remainder.

## Buoyancy formulations

`PotentialEnergy` is implemented for three buoyancy model types:

- **`BuoyancyTracer`**: uses the buoyancy field ``b`` directly as ``e_p = -bz``.
- **`SeawaterBuoyancy` with `LinearEquationOfState`**: computes buoyancy from a
  linear equation of state applied to temperature and/or salinity tracers.
- **`SeawaterBuoyancy` with `BoussinesqEquationOfState`** (from SeawaterPolynomials.jl):
  computes density from a nonlinear equation of state. An optional `geopotential_height`
  keyword argument allows using a potential density referenced to a fixed depth
  instead of in-situ density.

For now the full budget requires gravity to be aligned with the negative ``z``-direction
(`NegativeZDirection`), and every term checks it at construction. Tilting gravity does not make these
terms approximate, it makes them a different quantity: the height that works against gravity becomes
``z\cos\theta - y\sin\theta``, so ``-bz`` is wrong both by a factor and by a cross-slope term, and
nothing in the output would reveal it.

Two diagnostics are exempt because they do not depend on that alignment.
[`DiffusiveVerticalBuoyancyFlux`](@ref Oceanostics.PotentialEnergyEquation.DiffusiveVerticalBuoyancyFlux)
never touches ``z``; it returns the vertical component of the closure's diffusive flux, which is what it
promises whatever gravity does. `PotentialToKineticEnergyConversion` is the full ``u_i b_i`` contraction
over the gravity-projected buoyancy components, so it is correct under any tilt and only reads as ``wb``
when ``\hat{g} = -\hat{z}``. Neither is the term the budget above needs once gravity tilts, so the split
still does not close there — only the individual quantities remain meaningful.

## Summary of ``e_p`` equation terms

```@docs
Oceanostics.PotentialEnergyEquation.PotentialEnergy
Oceanostics.PotentialEnergyEquation.PotentialEnergyTendency
Oceanostics.PotentialEnergyEquation.PotentialEnergyAdvection
Oceanostics.PotentialEnergyEquation.PotentialEnergyBuoyancyAdvection
Oceanostics.PotentialEnergyEquation.PotentialEnergyDiffusion
Oceanostics.PotentialEnergyEquation.PotentialEnergyBuoyancyDiffusion
Oceanostics.PotentialEnergyEquation.DiffusiveVerticalBuoyancyFlux
Oceanostics.PotentialEnergyEquation.PotentialEnergyForcing
```
