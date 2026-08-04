# Potential energy equation

The `PotentialEnergyEquation` module provides a diagnostic for the specific gravitational
potential energy (per unit mass). In a Boussinesq fluid, the specific potential energy is defined as

```math
e_p = -bz = \frac{g\rho}{\rho_0} z
```

where ``b = -g\rho/\rho_0`` is buoyancy, ``z`` is the vertical coordinate, ``g`` is gravitational
acceleration, ``\rho`` is density, and ``\rho_0`` is a reference density. The quantity
``e_p`` has units of m² s⁻² (energy per unit mass).
Potential energy is a key quantity in ocean energetics: its conversion to/from
kinetic energy (via the term ``u_j b_j`` derived below) drives ocean circulation
and mixing.

!!! note "Lower case for densities, upper case for their integrals"
    Throughout Oceanostics a lower-case ``e`` is an energy density, the pointwise quantity a
    diagnostic returns, and the matching upper-case ``E`` is its volume integral,
    ```math
    E_p = \int e_p \, \mathrm{d}V = \texttt{Integral(PotentialEnergy(model))} ,
    ```
    and likewise for the background and available potential energies built from ``e_p`` below.

## The potential energy equation

Since ``e_p`` is just ``-z`` times the buoyancy, and ``z`` does not change in time, the equation for
``e_p`` follows directly from [the tracer equation](tracer_equation.md) applied to ``b``. In the
convention Oceananigans uses, that equation reads

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

Pulling ``z`` inside the derivatives separates them. Buoyancy enters the momentum equation as an
acceleration ``b_j = \hat{g}_j b``, the component along each direction of the unit vector ``\hat{g}``
that buoyancy acts along, and this module requires that vector to be vertical
(`NegativeZDirection`), so ``\partial_j z = \hat{g}_j`` and the product rule gives

```math
z\,\partial_j(u_j b) = -\partial_j(u_j e_p) - u_j b \hat{g}_j = -\partial_j(u_j e_p) - u_j b_j ,
\qquad
z\,\partial_j q_j    = \partial_j(z q_j) - q_j \hat{g}_j = \partial_j(z q_j) - q_3 ,
```

so that

```math
\partial_t e_p = \underbrace{-\partial_j(u_j e_p)}_{\text{advection}}
                 \underbrace{- u_j b_j}_{\text{PE to KE conversion}}
                 + \underbrace{\partial_j(z q_j)}_{\text{diffusive transport}}
                 \underbrace{- q_3}_{\text{diffusive buoyancy flux}}
                 \underbrace{- z F_b}_{\text{forcing}}.
```

``u_j b_j`` is the general form of the conversion, and it is what the diagnostic computes. It is often
written ``wb``, which is what it reduces to here: with ``\hat{g} = \hat{z}`` the horizontal components
``b_1`` and ``b_2`` vanish and ``b_3 = b``, leaving ``u_j b_j = wb``. The two are the same number for
every model this module accepts, but the diagnostic keeps all three components, so it stays correct in
a kinetic energy budget with a tilted gravity vector, where this page's ``e_p = -bz`` would not.

The two terms written as divergences transport ``e_p`` and vanish when integrated over a periodic or closed domain
(with impermeable, insulating walls).

## Terms and what is implemented

Every term above is implemented, two of them elsewhere in Oceanostics.

| Quantity | Expression | Diagnostic |
|:---|:---|:---|
| Potential energy | ``e_p = -bz`` | [`PotentialEnergy`](@ref Oceanostics.PotentialEnergyEquation.PotentialEnergy) |
| Tendency | ``\partial_t e_p = -z\,\partial_t b`` | [`PotentialEnergyTendency`](@ref Oceanostics.PotentialEnergyEquation.PotentialEnergyTendency) |
| Advection | ``z\,\partial_j(u_j b) = -\partial_j(u_j e_p) - u_j b_j`` | [`PotentialEnergyAdvection`](@ref Oceanostics.PotentialEnergyEquation.PotentialEnergyAdvection) |
| PE to KE conversion | ``u_j b_j`` | [`PotentialToKineticEnergyConversion`](@ref Oceanostics.KineticEnergyEquation.PotentialEnergyConversion) |
| Diffusion | ``z\,\partial_j q_j = \partial_j(z q_j) - q_3`` | [`PotentialEnergyDiffusion`](@ref Oceanostics.PotentialEnergyEquation.PotentialEnergyDiffusion) |
| Diffusive buoyancy flux | ``-q_3`` | [`DiffusiveBuoyancyFlux`](@ref Oceanostics.PotentialEnergyEquation.DiffusiveBuoyancyFlux) |
| Forcing | ``-z F_b`` | [`PotentialEnergyForcing`](@ref Oceanostics.PotentialEnergyEquation.PotentialEnergyForcing) |

`Advection` and `Diffusion` are the ``-z\,\times`` forms rather than the rearranged ones, which is the
convention [the kinetic energy equation](kinetic_energy_equation.md) follows as well
(``u_i\partial_j(u_ju_i)`` rather than ``\partial_j(u_jK)``). Taken that way the four terms are the
model's own buoyancy tendency split apart, so they sum to `PotentialEnergyTendency` cell by cell rather
than to within a truncation error. The rearranged forms are what a *volume-integrated* budget wants,
since the transports drop out and leave

```math
\int \texttt{Advection}\, \mathrm{d}V = -\int u_j b_j\, \mathrm{d}V, \qquad
\int \texttt{Diffusion}\, \mathrm{d}V = \int \Phi\, \mathrm{d}V ,
```

so an integrated budget is usually written with `PotentialToKineticEnergyConversion` and
`DiffusiveBuoyancyFlux` in their place. Those two identities come from the continuum product rule, so
unlike the split above they hold to the truncation error of the discretization.

!!! warning "Background buoyancy fields"
    When ``b`` carries a `BackgroundField` ``B``, the model prognoses the perturbation and its equation
    picks up one more term, ``-\partial_j(u_jB)``, the advection of ``B`` by the perturbation flow.
    Weighted by ``-z`` that is a source of ``e_p`` which does not integrate away, and it has no
    diagnostic here yet. `PotentialEnergyTendency` includes it, since it comes off the model's own
    kernel, but `Advection + Diffusion + Forcing` does not, so the split closes only for a buoyancy with
    no background field.

One caveat on the borrowed term: it computes ``u_j b_j`` as the source of *kinetic* energy, so the
potential energy budget takes it with a minus sign.

It goes by three names, because it is the one term the two budgets share and each side reads it
differently. It stays defined in [the kinetic energy equation](kinetic_energy_equation.md), where
`PotentialEnergyConversion` names what it does to the kinetic energy. `using Oceanostics` brings in
`PotentialToKineticEnergyConversion`, which names the exchange and says which way it runs. This module
and [the available potential energy equation](available_potential_energy_equation.md) both re-export
that and additionally give it `KineticEnergyConversion`, short enough to read well beside the other
terms of a potential energy budget. That last alias is scoped to those two modules: `using Oceanostics`
does not bring it in, because unprefixed it says nothing about which budget it belongs to.

``e_p`` also splits into two parts that do have their own modules. The
[background potential energy](background_potential_energy_equation.md) ``e_b`` is the portion the flow
could never release, and the [available potential energy](available_potential_energy_equation.md)
``e_a = e_p - e_b`` is the remainder. Their budgets are the ones actually closed in practice, and
[the lock release example](@ref lock_release_example) closes ``e_a`` against kinetic energy.

## Diffusive buoyancy flux

The diffusive conversion term reads
``\kappa\,\partial b/\partial z`` off the closure's own diffusive flux, which needs the buoyancy to be a
tracer the closure diffuses, so unlike `PotentialEnergy` it is restricted to `BuoyancyTracer` models. It
is also the second of the two parts
[`AvailablePotentialEnergyDissipationRate`](@ref Oceanostics.AvailablePotentialEnergyEquation.AvailablePotentialEnergyDissipationRate)
is written out of, which is why
[the available potential energy equation](available_potential_energy_equation.md) re-exports it.

Within the module it is `DiffusiveBuoyancyFlux`; `using Oceanostics` brings in the prefixed alias
`PotentialEnergyDiffusiveBuoyancyFlux`, which is the same type and keeps the bare name from colliding
with the tracer-equation fluxes.

```@docs
Oceanostics.PotentialEnergyEquation.DiffusiveBuoyancyFlux
```

## Buoyancy formulations

`PotentialEnergy` is implemented for three buoyancy model types:

- **`BuoyancyTracer`**: uses the buoyancy field ``b`` directly as ``e_p = -bz``.
- **`SeawaterBuoyancy` with `LinearEquationOfState`**: computes buoyancy from a
  linear equation of state applied to temperature and/or salinity tracers.
- **`SeawaterBuoyancy` with `BoussinesqEquationOfState`** (from SeawaterPolynomials.jl):
  computes density from a nonlinear equation of state. An optional `geopotential_height`
  keyword argument allows using a potential density referenced to a fixed depth
  instead of in-situ density.

The diagnostic requires gravity to be aligned with the negative ``z``-direction
(`NegativeZDirection`).

## Example

```jldoctest pe_eq
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=100, z=(-1000, 0), topology=(Flat, Flat, Bounded));

julia> model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b);

julia> ep = PotentialEnergyEquation.PotentialEnergy(model)
PotentialEnergy KernelFunctionOperation at (Center, Center, Center)
├── grid: 1×1×100 RectilinearGrid{Float64, Flat, Flat, Bounded} on CPU with 0×0×3 halo
├── kernel_function: minus_bz_ccc (generic function with 3 methods)
└── arguments: ("Field",)
└── computes: potential energy per unit volume  eₚ = -bz
```

## Potential energy

```@docs
Oceanostics.PotentialEnergyEquation.PotentialEnergy
```

## Terms of the ``e_p`` equation

Inside the module these go by the short names `Tendency`, `Advection`, `Diffusion` and `Forcing`;
`using Oceanostics` brings in the prefixed aliases below, which are the same types. The
[baroclinic adjustment example](@ref baroclinic_adjustment_example) closes an integrated ``e_p`` budget
with them.

```@docs
Oceanostics.PotentialEnergyEquation.PotentialEnergyTendency
Oceanostics.PotentialEnergyEquation.PotentialEnergyAdvection
Oceanostics.PotentialEnergyEquation.PotentialEnergyDiffusion
Oceanostics.PotentialEnergyEquation.PotentialEnergyForcing
```
