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
e_a(b, z) = \int_{z^\star(b)}^{z} \left[b^\star(\tilde z,\, t) - b\right] \,\mathrm{d}\tilde z
          = \frac{g}{\rho_0}\int_{z^\star(\rho)}^{z} \left[\rho - \rho^\star(\tilde z,\, t)\right] \,\mathrm{d}\tilde z .
```

In this form ``e_a`` is **non-negative everywhere in space** whenever the reference state is sorted from
the field itself. Its volume integral recovers ``\int e_p - \int e_b`` in the continuum limit, although
at finite ``\Delta z`` the two differ at second order.

## Deriving the local available potential energy budget

The budget follows from the material derivative of ``e_a(b, z, t)`` along the flow
([Wenegrat, Chor & Barkan, 2026](https://arxiv.org/abs/2605.15879)):


```math
\frac{D e_a}{D t} = \left.\frac{\partial e_a}{\partial b}\right|_{z,t} \frac{D b}{D t}
                  + \left.\frac{\partial e_a}{\partial z}\right|_{b,t} \frac{D z}{D t}
                  + \left.\frac{\partial e_a}{\partial t}\right|_{z,b} \frac{D t}{D t},
```
which we can simplify to

```math
\frac{D e_a}{D t} = \left.\frac{\partial e_a}{\partial b}\right|_{z,t} \frac{D b}{D t}
                  + \left.\frac{\partial e_a}{\partial z}\right|_{b,t} w
                  + R ,
\qquad
R = \int_{z^\star}^{z} \partial_t b^\star(\tilde z, t) \, \mathrm{d}\tilde z .
```

Both partial derivatives come straight from the definition of ``e_a``. In the first, the boundary term
from moving the lower limit ``z^\star(b)`` drops out because ``b^\star(z^\star(b)) = b``, which leaves
the displacement potential ``\Upsilon``. The second is minus the buoyancy anomaly ``b_r`` the parcel
carries relative to the reference profile at its own height:

```math
\left.\frac{\partial e_a}{\partial b}\right|_{z} = z^\star - z = \Upsilon ,
\qquad
\left.\frac{\partial e_a}{\partial z}\right|_{b} = b^\star(z) - b = -b_r .
```

The buoyancy obeys [the tracer equation](tracer_equation.md), ``Db/Dt = -\partial_j q_j`` for an
unforced tracer with the closure's diffusive flux ``q_j``, so ``\Upsilon\,Db/Dt`` splits into a
transport divergence plus the contraction ``q_j\,\partial_j\Upsilon``. Writing the advective part as a
divergence as well (incompressibility), the local APE budget is

```math
\partial_t e_a = \underbrace{-\partial_j(u_j e_a)}_{\text{advection}}
                 \underbrace{-\,w b_r}_{\text{APE to KE conversion}}
                 \underbrace{-\,\partial_j(\Upsilon q_j)}_{\text{diffusive transport}}
                 \underbrace{-\,\varepsilon_a}_{\text{dissipation}}
                 + \underbrace{R}_{\text{reference tendency}} ,
\qquad
\varepsilon_a = -q_j \, \partial_j \Upsilon = \kappa \, \partial_j b \, \partial_j \Upsilon .
```

The two divergences redistribute ``e_a`` and vanish when integrated over a periodic or closed domain,
and so does ``R`` ([Winters et al., 1995](https://doi.org/10.1017/S002211209500125X)), which therefore
shifts APE around in space without creating or destroying any. The exchange with the kinetic energy
carries the anomaly ``b_r`` because in this framework the pressure is measured against the hydrostatic
pressure of the reference profile. Its reference part ``w\,b^\star(z)`` is a divergence in its own
right, so ``\int w b_r \, \mathrm{d}V = \int w b \, \mathrm{d}V`` and the volume-integrated budget
reduces to the one the [Lock release](@ref lock_release_example) example closes below.

Repeating these steps with the filtered buoyancy ``\tilde b`` in place of ``b``, and using the filtered
buoyancy equation, gives the budget for the available potential energy of the filtered field; filtering
the budget above and subtracting that one gives the sub-filter budget. Those two budgets, and the
cross-scale flux ``\Pi_a`` that transfers APE between them, are the subject of
[the filtered](filtered_available_potential_energy_equation.md) and
[the sub-filter](subfilter_available_potential_energy_equation.md) available potential energy equation
pages.

## Available potential energy

```@docs
Oceanostics.AvailablePotentialEnergyEquation.AvailablePotentialEnergy
```

## Buoyancy displacement potential

The factor multiplying ``Db/Dt`` in the budget above, the derivative of ``e_a`` with respect to
buoyancy, is the displacement potential

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
\varepsilon_a = \kappa \, \partial_i b \, \partial_i \Upsilon
              = \kappa \left[\frac{\partial z^\star}{\partial b}\left|\nabla b\right|^2
                             - \frac{\partial b}{\partial z}\right]
```

is the sink of the local ``e_a`` equation: the diapycnal mixing rate of
[Winters et al. (1995)](https://doi.org/10.1017/S002211209500125X), less the diffusion the reference
state undergoes on its own and which carries no available energy with it. The two cancel exactly for a
statically stable, horizontally uniform stratification, so ``\varepsilon_a`` measures only the APE
actually lost and is **not** the sign-definite ``\kappa|\nabla b|^2`` the name might suggest.

```@docs
Oceanostics.AvailablePotentialEnergyEquation.AvailablePotentialEnergyDissipationRate
```

## Diffusive buoyancy flux

The second of those two parts,

```math
\Phi = \kappa \frac{\partial b}{\partial z} ,
```

is the work diffusion does against gravity as it smooths the stratification. It is available separately
because it is what separates ``\varepsilon_a`` from the diapycnal mixing rate: adding it back gives that
rate, and hence the growth rate of the total ``E_b = \int e_b \, \mathrm{d}V``,

```math
\frac{d}{dt}\int e_b\, dV = \int \left(\varepsilon_a + \Phi\right) dV \geq 0 ,
```

so the three of them close the background potential energy budget the way ``\varepsilon_a`` and the
buoyancy production close the available one. Neither ``\varepsilon_a`` nor ``\Phi`` is sign-definite on
its own; their sum is.

``\Phi`` needs no reference state of its own, and it is a term of the ``e_p`` equation before it is
anything to do with ``e_a``, so it is defined in
[the potential energy equation](potential_energy_equation.md) and re-exported
here. See [`PotentialEnergyDiffusiveVerticalBuoyancyFlux`](@ref Oceanostics.PotentialEnergyEquation.DiffusiveVerticalBuoyancyFlux).

The [Lock release](@ref lock_release_example) example closes

```math
\frac{d}{dt}\int e_a\, dV = -\int u_j b_j\, dV - \int \varepsilon_a\, dV
```

alongside the matching kinetic energy budget, and shows ``\varepsilon_a`` changing sign as the flow
alternates between stirring and settling.

The ``u_j b_j`` in that budget is the term the two exchange, the same conversion
[the potential energy equation](potential_energy_equation.md) derives, so this module re-exports
[`PotentialToKineticEnergyConversion`](@ref Oceanostics.KineticEnergyEquation.PotentialEnergyConversion) from
[the kinetic energy equation](kinetic_energy_equation.md), under that name and under the shorter alias
`KineticEnergyConversion`, which is scoped to this module and to
that page. It computes ``u_j b_j``, the source of kinetic energy, so this budget takes it with a minus
sign. With the vertical gravity these modules require it reduces to ``wb``, but the diagnostic keeps
all three components.
