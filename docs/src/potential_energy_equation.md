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
kinetic energy (via the buoyancy production term ``wb``) drives ocean circulation
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

Neither of the first two terms is yet in a useful form: each mixes a redistribution, which moves
``e_p`` around without creating any, with a genuine source. Pulling ``z`` inside the derivatives
separates them: with ``\partial_j z = \delta_{j3}``, the product rule gives

```math
z\,\partial_j(u_j b) = -\partial_j(u_j e_p) - wb , \qquad
z\,\partial_j q_j    = \partial_j(z q_j) - q_3 ,
```

so that

```math
\partial_t e_p = \underbrace{-\partial_j(u_j e_p)}_{\text{advection}}
                 \underbrace{- wb}_{\text{buoyancy conversion}}
                 + \underbrace{\partial_j(z q_j)}_{\text{diffusive transport}}
                 \underbrace{- q_3}_{\text{diffusive conversion}}
                 \underbrace{- z F_b}_{\text{forcing}} .
```

The two terms written as divergences move ``e_p`` from place to place and vanish when integrated over
a closed domain with impermeable, insulating walls. What is left is the pair of conversion terms:

```math
\frac{d}{dt}\int e_p\, dV = -\int wb\, dV + \int \kappa \frac{\partial b}{\partial z}\, dV
                            - \int z F_b\, dV ,
```

using ``-q_3 = \kappa\,\partial b/\partial z``. The first is the exchange with kinetic energy, positive
into ``e_p`` when dense fluid is being lifted. The second is the work diffusion does against gravity as
it smooths the stratification, which is the only way diffusion can change the total potential energy of
a closed domain, and it is always positive for a statically stable fluid.

## Terms and what is implemented

Only ``e_p`` itself is implemented in this module. Two of the terms above are available elsewhere in
Oceanostics, and the rest have no diagnostic yet.

| Quantity | Expression | Diagnostic |
|:---|:---|:---|
| Potential energy | ``e_p = -bz`` | [`PotentialEnergy`](@ref Oceanostics.PotentialEnergyEquation.PotentialEnergy) |
| Tendency | ``\partial_t e_p`` | not implemented |
| Advection | ``-\partial_j(u_j e_p)`` | not implemented |
| Buoyancy conversion | ``-wb`` | [`KineticEnergyBuoyancyProduction`](@ref Oceanostics.KineticEnergyEquation.BuoyancyProduction), with the opposite sign |
| Diffusive transport | ``\partial_j(z q_j)`` | not implemented |
| Diffusive conversion | ``-q_3 = \kappa\,\partial b/\partial z`` | [`ReferenceStateDiffusionRate`](@ref Oceanostics.AvailablePotentialEnergyEquation.ReferenceStateDiffusionRate) |
| Forcing | ``-z F_b`` | not implemented |

Two caveats on the borrowed terms. `KineticEnergyBuoyancyProduction` computes ``u_i b_i``, which is the
source of *kinetic* energy, so the potential energy budget takes it with a minus sign; with the
`NegativeZDirection` gravity this module requires, ``u_i b_i = wb``. And `ReferenceStateDiffusionRate`
reads ``\kappa\,\partial b/\partial z`` off the closure's own diffusive flux, which needs the buoyancy
to be a tracer the closure diffuses, so it is restricted to `BuoyancyTracer` models while
`PotentialEnergy` is not.

``e_p`` also splits into two parts that do have their own modules. The
[background potential energy](background_potential_energy_equation.md) ``e_b`` is the portion the flow
could never release, and the [available potential energy](available_potential_energy_equation.md)
``e_a = e_p - e_b`` is the remainder. Their budgets are the ones actually closed in practice, and
[the lock release example](@ref lock_release_example) closes ``e_a`` against kinetic energy.

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
