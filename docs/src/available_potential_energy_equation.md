# Available potential energy equation

The `AvailablePotentialEnergyEquation` module computes the share of the potential energy
``e_p = -bz`` of [the potential energy equation](potential_energy_equation.md) that the flow *can*
release. The other half, the reference state and the background potential energy ``e_b`` it carries,
lives in [the background potential energy equation](background_potential_energy_equation.md). The
reference height ``z^\star`` built there is the input to everything below, and this module re-exports
`reference_height`, `reference_buoyancy` and the reference-height methods so that either module can be
used on its own.

Oceanostics computes the available potential energy in its *local* form,
[Holliday & McIntyre (1981)](https://doi.org/10.1017/S0022112081001742):

```math
e_a(b, z) = \int_{z^\star}^{z} \left[b^\star(\tilde z) - b\right] \mathrm{d}\tilde z
          = \frac{g}{\rho_0}\int_{z^\star}^{z} \left[\rho - \rho^\star(\tilde z)\right] \mathrm{d}\tilde z .
```

In this form ``e_a`` is **non-negative everywhere in space** whenever the reference state is sorted from
the field itself, so it can be mapped as a field. Its volume integral recovers
``\int e_p - \int e_b`` in the continuum limit, although at finite ``\Delta z`` the two differ at second
order.

## Available potential energy

```@docs
Oceanostics.AvailablePotentialEnergyEquation.AvailablePotentialEnergy
```

## Buoyancy displacement potential

Differentiating ``e_a`` with respect to buoyancy gives the displacement potential

```math
\Upsilon = \frac{\partial e_a}{\partial b} = z^\star - z ,
```

the natural conjugate of ``b``: contracting it with a buoyancy gradient gives an APE dissipation rate,
and contracting it with a sub-filter buoyancy flux gives a cross-scale APE flux
([Wenegrat, Chor & Barkan, 2026](https://arxiv.org/abs/2605.15879), who write it for density as
``\Upsilon(\rho, z) = g(z - z^\star)/\rho_0``; the two differ by ``-g/\rho_0``, which cancels in either
contraction).

```@docs
Oceanostics.AvailablePotentialEnergyEquation.BuoyancyDisplacementPotential
```

## Available potential energy dissipation

```math
\varepsilon_A = \kappa \, \partial_i b \, \partial_i \Upsilon
              = \kappa \left[\frac{\partial z^\star}{\partial b}\left|\nabla b\right|^2
                             - \frac{\partial b}{\partial z}\right]
```

is the sink of the local ``e_a`` equation: the diapycnal mixing rate of
[Winters et al. (1995)](https://doi.org/10.1017/S002211209500125X), less the diffusion the reference
state undergoes on its own and which carries no available energy with it. The two cancel exactly for a
statically stable, horizontally uniform stratification, so ``\varepsilon_A`` measures only the APE
actually lost and is **not** the sign-definite ``\kappa|\nabla b|^2`` the name might suggest.

```@docs
Oceanostics.AvailablePotentialEnergyEquation.AvailablePotentialEnergyDissipationRate
```

## Reference state diffusion

The second of those two parts,

```math
\Phi = \kappa \frac{\partial b}{\partial z} ,
```

is the work diffusion does against gravity as it smooths the stratification. It is available separately
because it is what separates ``\varepsilon_A`` from the diapycnal mixing rate: adding it back gives that
rate, and hence the growth rate of the total ``E_b = \int e_b \, \mathrm{d}V``,

```math
\frac{d}{dt}\int e_b\, dV = \int \left(\varepsilon_A + \Phi\right) dV \geq 0 ,
```

so the three of them close the background potential energy budget the way ``\varepsilon_A`` and the
buoyancy production close the available one. Neither ``\varepsilon_A`` nor ``\Phi`` is sign-definite on
its own; their sum is.

```@docs
Oceanostics.AvailablePotentialEnergyEquation.ReferenceStateDiffusionRate
```

The [Lock release](@ref lock_release_example) example closes

```math
\frac{d}{dt}\int e_a\, dV = -\int wb\, dV - \int \varepsilon_A\, dV
```

alongside the matching kinetic energy budget, and shows ``\varepsilon_A`` changing sign as the flow
alternates between stirring and settling.
